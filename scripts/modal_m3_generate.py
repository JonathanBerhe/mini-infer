"""Real MiniMax-M3 coherence gate + decode-kernel A/B on a multi-GPU Modal node.

Loads the pre-quantized checkpoint (block-FP8 routed experts + bf16 everything
else, staged by `modal_m3_stage_weights.py` into the `minimax-m3-fp8-weights`
Volume) under tensor + expert parallelism via the streaming loader (peak host
RAM ~ one shard per rank), then:

1. **Coherence gate**: greedy-generates from a short factual prompt; the output
   must be coherent text and byte-identical across ranks (TP consistency).
2. **Decode-kernel A/B**: builds a long context (chunked prefill of a repeated
   document), then times decode steps with the MSA block-sparse decode kernel
   off vs on. Tokens must match between arms; the tok/s ratio is the
   ship-or-not signal for default-on (the V4 lesson: only the end-to-end
   number counts).

Run with (after staging completes):
    uv run modal run scripts/modal_m3_generate.py
    M3_GPU=B200:4 uv run modal run scripts/modal_m3_generate.py
"""

import os

import modal

_GPU = os.environ.get("M3_GPU", "H200:4")
_WORLD_SIZE = int(_GPU.split(":")[1]) if ":" in _GPU else 1
_VOLUME_NAME = "minimax-m3-fp8-weights"
_MOUNT = "/weights"
_CKPT_DIR = f"{_MOUNT}/MiniMax-M3"

app = modal.App("mini-infer-m3-generate")
weights_volume = modal.Volume.from_name(_VOLUME_NAME)

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .pip_install("torch==2.6.0", extra_index_url="https://download.pytorch.org/whl/cu124")
    .pip_install(
        "transformers>=5.12,<5.13",
        "safetensors>=0.4",
        "huggingface_hub>=0.20",
    )
    .add_local_python_source("mini_infer")
)

_PROMPT = "The capital of France is"

# Long-context filler for the kernel A/B: a real prose paragraph (not a
# synthetic token pattern) repeated to the target length after tokenization.
_AB_DOC = (
    "Transformer inference engines separate the prefill phase, which processes "
    "the whole prompt in one highly parallel pass, from the decode phase, which "
    "generates one token at a time and is dominated by memory bandwidth rather "
    "than arithmetic. Paged key-value caches divide the attention history into "
    "fixed-size blocks so that memory can be allocated on demand and shared "
    "between requests with common prefixes. Sparse attention mechanisms go one "
    "step further and select only a subset of those blocks for each query, "
    "trading a small amount of selection computation for a large reduction in "
    "the memory traffic that each decoding step must pay. "
)


def _greedy_decode(model, cache, tokenizer, prompt_ids, max_new_tokens, device):  # type: ignore[no-untyped-def]
    """Chunked prefill + greedy decode; returns generated token ids."""
    import torch

    chunk = 1024
    plen = len(prompt_ids)
    done = 0
    with torch.inference_mode():
        while done < plen:
            n = min(chunk, plen - done)
            ids = torch.tensor([prompt_ids[done : done + n]], dtype=torch.long, device=device)
            pos = torch.arange(done, done + n, device=device).unsqueeze(0)
            logits = model(
                input_ids=ids,
                position_ids=pos,
                past_key_values=cache,
                cu_seqlens_q=torch.tensor([0, n], dtype=torch.int32),
            )
            done += n
        nxt = int(logits[0, -1].argmax())
        out = [nxt]
        cache_len = plen
        for _ in range(max_new_tokens - 1):
            logits = model(
                input_ids=torch.tensor([[nxt]], dtype=torch.long, device=device),
                position_ids=torch.tensor([[cache_len]], device=device),
                past_key_values=cache,
                cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
            )
            cache_len += 1
            nxt = int(logits[0, -1].argmax())
            out.append(nxt)
    return out


def _timed_decode(model, cache, start_len, steps, device):  # type: ignore[no-untyped-def]
    """Time `steps` greedy decode steps from an existing cache state."""
    import time

    import torch

    with torch.inference_mode():
        nxt = 1  # deterministic dummy continuation token
        tokens = []
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        cache_len = start_len
        for _ in range(steps):
            logits = model(
                input_ids=torch.tensor([[nxt]], dtype=torch.long, device=device),
                position_ids=torch.tensor([[cache_len]], device=device),
                past_key_values=cache,
                cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
            )
            cache_len += 1
            nxt = int(logits[0, -1].argmax())
            tokens.append(nxt)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    return tokens, steps / dt


def _build_and_load(ckpt_dir: str, device: str):  # type: ignore[no-untyped-def]
    """Construct the sharded model directly on `device` and stream-load weights.

    Constructing under the device context avoids a full-size host-RAM copy of
    the rank's parameters; the streaming loader then replaces them shard by
    shard (block-FP8 experts land on the Fp8Expert buffers untouched).
    """
    import torch
    from transformers import AutoConfig

    from mini_infer.models.minimax_m3 import MiniMaxM3Config, MiniMaxM3ForCausalLM

    hf_cfg = AutoConfig.from_pretrained(ckpt_dir)
    cfg = MiniMaxM3Config.from_hf(hf_cfg)
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device(device):
            model = MiniMaxM3ForCausalLM(cfg)
    finally:
        torch.set_default_dtype(prev_dtype)
    model.eval()
    MiniMaxM3ForCausalLM.load_weights_streaming(
        model, ckpt_dir, device=device, dtype=torch.bfloat16
    )
    return cfg, model


def _make_cache(model, device: str, max_tokens: int):  # type: ignore[no-untyped-def]
    import torch

    from mini_infer.cache.block_pool import BlockPool
    from mini_infer.cache.paged_kv_cache import PagedKVCache

    block_size = 128  # = index_block_size: 1 pool block per MSA scoring block
    pool = BlockPool(
        num_blocks=max_tokens // block_size + 4,
        block_size=block_size,
        num_layers=model.cfg.num_hidden_layers,
        num_kv_heads=model.cfg.num_key_value_heads,
        head_dim=model.cfg.head_dim,
        dtype=torch.bfloat16,
        device=device,
        layer_streams=model.per_layer_streams(),
        attention_backend="torch",
    )
    cache = PagedKVCache(pool)
    cache.add_request_slot()
    return cache


def _run_rank(
    rank: int,
    world_size: int,
    max_new_tokens: int,
    ab_context: int,
    ab_steps: int,
) -> dict:
    import torch
    import torch.distributed as dist
    from transformers import AutoTokenizer

    from mini_infer.distributed.group import destroy_distributed, init_distributed

    init_distributed(
        world_size=world_size,
        rank=rank,
        backend="nccl",
        master_addr="127.0.0.1",
        master_port=29500,
    )
    try:
        device = f"cuda:{rank}"
        torch.cuda.set_device(rank)
        tokenizer = AutoTokenizer.from_pretrained(_CKPT_DIR)
        _cfg, model = _build_and_load(_CKPT_DIR, device)

        # 1. Coherence gate.
        prompt_ids = tokenizer.encode(_PROMPT)
        cache = _make_cache(model, device, len(prompt_ids) + max_new_tokens + 8)
        out_ids = _greedy_decode(model, cache, tokenizer, prompt_ids, max_new_tokens, device)
        text = tokenizer.decode(out_ids)
        del cache

        # 2. Kernel A/B at long context: same prefill state per arm.
        ab_doc_ids = tokenizer.encode(_AB_DOC)
        long_ids = (ab_doc_ids * (ab_context // len(ab_doc_ids) + 1))[:ab_context]
        results = {}
        for arm in ("torch", "kernel"):
            model.set_decode_kernel(arm == "kernel")
            cache = _make_cache(model, device, ab_context + ab_steps + 8)
            _greedy_decode(model, cache, tokenizer, long_ids, 1, device)
            tokens, tps = _timed_decode(model, cache, ab_context, ab_steps, device)
            results[arm] = {"tokens": tokens, "tok_s": tps}
            del cache
        model.set_decode_kernel(False)
        return {
            "rank": rank,
            "text": text,
            "ab": results,
            "mem_gb": torch.cuda.max_memory_allocated() / 1e9,
        }
    finally:
        if dist.is_available() and dist.is_initialized():
            destroy_distributed()


def _child_entry(rank, world_size, max_new_tokens, ab_context, ab_steps, queue):  # type: ignore[no-untyped-def]
    try:
        queue.put(("ok", rank, _run_rank(rank, world_size, max_new_tokens, ab_context, ab_steps)))
    except Exception:
        import traceback

        queue.put(("err", rank, traceback.format_exc()))


@app.function(
    image=image,
    gpu=_GPU,
    volumes={_MOUNT: weights_volume},
    timeout=3600,
    memory=131072,
)
def generate(max_new_tokens: int = 48, ab_context: int = 16384, ab_steps: int = 32) -> str:
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = [
        ctx.Process(
            target=_child_entry,
            args=(rank, _WORLD_SIZE, max_new_tokens, ab_context, ab_steps, queue),
        )
        for rank in range(_WORLD_SIZE)
    ]
    for p in procs:
        p.start()
    results: dict[int, dict] = {}
    errors = []
    for _ in procs:
        status, rank, payload = queue.get()
        if status == "ok":
            results[rank] = payload
        else:
            errors.append(f"rank {rank}:\n{payload}")
    for p in procs:
        p.join()
    if errors:
        return "FAILED\n" + "\n".join(errors)

    lines = []
    r0 = results[0]
    lines.append(f"coherence output (rank 0): {r0['text']!r}")
    same_text = all(results[r]["text"] == r0["text"] for r in results)
    lines.append(f"rank consistency: {'PASS' if same_text else 'FAIL'}")
    ab = r0["ab"]
    same_tokens = ab["torch"]["tokens"] == ab["kernel"]["tokens"]
    speedup = ab["kernel"]["tok_s"] / ab["torch"]["tok_s"]
    lines.append(
        f"kernel A/B @ ctx={16384}: torch {ab['torch']['tok_s']:.2f} tok/s, "
        f"kernel {ab['kernel']['tok_s']:.2f} tok/s, speedup {speedup:.2f}x, "
        f"token identity {'PASS' if same_tokens else 'FAIL'}"
    )
    lines.append(f"peak GPU mem rank0: {r0['mem_gb']:.1f} GB")
    return "\n".join(lines)


@app.local_entrypoint()
def main(max_new_tokens: int = 48, ab_context: int = 16384, ab_steps: int = 32) -> None:
    print(generate.remote(max_new_tokens, ab_context, ab_steps))
