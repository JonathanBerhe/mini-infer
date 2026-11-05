"""Profile one Kimi Linear decode step and the cost of its MLA layers.

Two measurements, on the CPU or on one GPU:

1. `split`: the decode step of a model with the per-layer widths of
   Kimi-Linear-48B-A3B (hidden 2304, 32 heads, `kv_lora_rank` 512, 32 KDA
   heads of 128, experts of 1024) but fewer layers, experts and vocabulary
   entries, so that it fits a laptop. The labelled profile
   (`mini_infer.bench.step_profile`) splits the step into the MLA layers,
   the KDA layers and the MoE. The ATen op count and the ops that read a
   value back to the host (`aten::_local_scalar_dense`, `aten::nonzero`)
   stand in for the kernel launches and the host syncs of a GPU run.
2. `mla`: one MLA layer alone, timed against the history length and against
   the capacity of the state buffer, for one row and for a ragged batch.

With `--mla-path decompressed` or `--mla-path absorbed` the MLA decode runs
that form (`_KimiMlaAttention.absorbed_decode`), so that both forms are
timed in one process on one set of weights.

Run from the repository root:

    uv run python scripts/kimi_linear/profile_kimi_decode.py split
    uv run python scripts/kimi_linear/profile_kimi_decode.py mla
"""

from __future__ import annotations

import argparse
import statistics
import time
from types import SimpleNamespace
from typing import Any

import torch

from mini_infer.bench.step_profile import categorize, profile_steps
from mini_infer.models.kimi_linear import kimi_linear
from mini_infer.models.kimi_linear.kimi_linear import KimiLinearConfig, KimiLinearForCausalLM
from mini_infer.models.kimi_linear.state_cache import KimiMlaLayerState

# Ops that read a device value back to the host. On a GPU each one is a sync.
# `aten::item` is left out: it wraps `aten::_local_scalar_dense`.
_HOST_READ_OPS = frozenset({"aten::_local_scalar_dense", "aten::nonzero"})
# The prefill runs in chunks of this many tokens, so that the score tensor of
# a long history fits the memory of a laptop.
_PREFILL_CHUNK = 1024


def _config(*, layers: int, experts: int, vocab: int) -> KimiLinearConfig:
    """Kimi-Linear-48B-A3B per-layer widths; every fourth layer is MLA, as in the checkpoint."""
    full = [i for i in range(1, layers + 1) if i % 4 == 0] or [layers]
    kda = [i for i in range(1, layers + 1) if i not in full]
    raw = SimpleNamespace(
        vocab_size=vocab,
        hidden_size=2304,
        num_hidden_layers=layers,
        num_attention_heads=32,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        q_lora_rank=None,
        mla_use_nope=True,
        linear_attn_config={
            "kda_layers": kda,
            "full_attn_layers": full,
            "num_heads": 32,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
        },
        intermediate_size=9216,
        moe_intermediate_size=1024,
        num_experts=experts,
        num_experts_per_token=8,
        num_shared_experts=1,
        routed_scaling_factor=2.446,
        moe_renormalize=True,
        moe_router_activation_func="sigmoid",
        num_expert_group=1,
        topk_group=1,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
    )
    return KimiLinearConfig.from_hf(raw)


def _set_mla_path(path: str) -> None:
    if path == "current":
        return
    if not hasattr(kimi_linear._KimiMlaAttention, "absorbed_decode"):
        raise SystemExit("this checkout has one MLA decode form only; use --mla-path current")
    kimi_linear._KimiMlaAttention.absorbed_decode = path == "absorbed"


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _median_ms(
    run: Any, device: torch.device, *, warmup: int, steps: int
) -> tuple[float, float, float]:
    for _ in range(warmup):
        run()
    _sync(device)
    times = []
    for _ in range(steps):
        start = time.perf_counter()
        run()
        _sync(device)
        times.append((time.perf_counter() - start) * 1e3)
    return statistics.median(times), min(times), max(times)


def _op_counts(trace_path: str, steps: int) -> dict[str, dict[str, float]]:
    """Top-level ATen ops and host reads per step, by part of the model."""
    import json

    with open(trace_path) as handle:
        events = json.load(handle)["traceEvents"]
    complete = [e for e in events if e.get("ph") == "X" and "dur" in e]
    counts: dict[str, dict[str, float]] = {}
    for tid in {e["tid"] for e in complete}:
        mine = sorted(
            (
                e
                for e in complete
                if e["tid"] == tid and e.get("cat") in ("cpu_op", "user_annotation")
            ),
            key=lambda e: (e["ts"], -e["dur"]),
        )
        stack: list[dict[str, Any]] = []
        for event in mine:
            while stack and stack[-1]["ts"] + stack[-1]["dur"] <= event["ts"]:
                stack.pop()
            if event.get("cat") == "cpu_op":
                inside_op = any(s.get("cat") == "cpu_op" for s in stack)
                label = next(
                    (s["name"] for s in reversed(stack) if s.get("cat") == "user_annotation"), None
                )
                if label is not None and label != "step":
                    part = categorize(label, "")
                    entry = counts.setdefault(part, {"ops": 0.0, "host_reads": 0.0})
                    if not inside_op:
                        entry["ops"] += 1 / steps
                    if event["name"] in _HOST_READ_OPS:
                        entry["host_reads"] += 1 / steps
            stack.append(event)
    return counts


def run_split(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    _set_mla_path(args.mla_path)
    torch.manual_seed(0)
    cfg = _config(layers=args.layers, experts=args.experts, vocab=args.vocab)
    # Built on the device: 256 experts in fp32 need about 22 GB of host memory.
    with torch.device(device):
        model = KimiLinearForCausalLM(cfg)
    model = model.to(dtype=dtype).eval()
    cache = model.build_state_cache(
        max_seq_len=args.max_seq_len, batch_size=args.batch, device=device, dtype=dtype
    )
    prompt = torch.randint(0, cfg.vocab_size, (args.batch, args.prompt), device=device)
    token = torch.randint(0, cfg.vocab_size, (args.batch, 1), device=device)
    with torch.inference_mode():
        for chunk in prompt.split(_PREFILL_CHUNK, dim=1):
            model.forward_prefill_with_cache(chunk, state_cache=cache)
            cache.advance_start_pos(chunk.shape[1])
        start = cache.start_pos

        def step() -> None:
            # Every step writes the same slot, so the history length stays fixed.
            model.forward_decode_with_cache(token, start_pos=start, state_cache=cache)

        median, low, high = _median_ms(step, device, warmup=4, steps=args.steps)
        trace = f"{args.trace_dir}/kimi_split.json"
        summary = profile_steps(
            step, steps=4, model=model, on_cuda=device.type == "cuda", trace_path=trace
        )
    print(
        f"model: {args.layers} layers ({len(cfg.full_attn_layers)} MLA), {args.experts} experts, "
        f"vocab {args.vocab}, {dtype}, device {device}, batch {args.batch}, "
        f"history {args.prompt}, capacity {args.max_seq_len}, MLA path {args.mla_path}"
    )
    print(f"decode step: median {median:.1f} ms (range {low:.1f}-{high:.1f}, {args.steps} steps)")
    if device.type == "cuda":
        steps = summary["steps"]
        print(
            f"per profiled step: GPU busy {statistics.median(s['gpu_busy_ms'] for s in steps):.1f} "
            f"ms, kernels {statistics.median(s['kernels'] for s in steps):.0f}, "
            f"host syncs {statistics.median(s['syncs'] for s in steps):.0f}"
        )
        for where, count in summary["sync_sites"][:6]:
            print(f"  syncs per step {count:>6.0f}  {where}")
    ops = _op_counts(trace, 4)
    time_key = "gpu_ms" if device.type == "cuda" else "host_ms"
    print(f"\n{'part':<24}{time_key:>10}{'ATen ops':>10}{'host reads':>12}")
    for row in summary["categories"]:
        part = row["key"]
        count = ops.get(part, {"ops": 0.0, "host_reads": 0.0})
        print(f"{part:<24}{row[time_key]:>10.2f}{count['ops']:>10.0f}{count['host_reads']:>12.0f}")
    print("\nper-layer host time (profiled, ms):")
    for row in summary["host_labels"][:12]:
        print(f"  {row['key']:<60}{row['host_ms']:>8.2f}")


def _fill_mla_state(
    attn: kimi_linear._KimiMlaAttention,
    batch: int,
    capacity: int,
    device: torch.device,
    dtype: torch.dtype,
) -> KimiMlaLayerState:
    # The values do not change the time; random rows of the cached width do.
    width = attn.kv_lora_rank + attn.qk_rope_head_dim
    return KimiMlaLayerState(kv=torch.randn(batch, capacity, width, device=device, dtype=dtype))


def run_mla(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    torch.manual_seed(0)
    cfg = _config(layers=4, experts=8, vocab=1024)
    attn = kimi_linear._KimiMlaAttention(cfg, layer_idx=3).to(device=device, dtype=dtype).eval()
    paths = ["current"] if args.mla_path == "current" else ["decompressed", "absorbed"]
    print(f"one MLA layer, Kimi-Linear-48B widths, {dtype}, device {device}")
    print(
        f"{'batch':>5}{'history':>9}{'capacity':>10}"
        + "".join(f"{p + ' ms':>18}" for p in paths)
        + (f"{'max diff':>12}" if len(paths) > 1 else ""),
        flush=True,
    )

    def time_case(batch: int, history: int, capacity: int, ragged: bool) -> list[float]:
        state = _fill_mla_state(attn, batch, capacity, device, dtype)
        x = torch.randn(batch, 1, cfg.hidden_size, device=device, dtype=dtype)
        if ragged:
            # Row b holds history * (b + 1) / batch tokens: one long row, the rest shorter.
            starts = torch.tensor(
                [max(1, history * (b + 1) // batch) for b in range(batch)], device=device
            )
        else:
            starts = torch.full((batch,), history, device=device)
        out = []
        outputs = []
        for path in paths:
            _set_mla_path(path)
            with torch.inference_mode():
                median, _, _ = _median_ms(
                    lambda: attn(x, state, starts), device, warmup=2, steps=args.steps
                )
                outputs.append(attn(x, state, starts).float())
            out.append(median)
        if len(outputs) > 1:
            # The largest difference of the layer outputs of the two forms.
            out.append((outputs[0] - outputs[1]).abs().max().item())
        return out

    cases: list[tuple[int, int, int, bool]] = []
    for history in (256, 1024, 4096, 16384):
        cases.append((1, history, 32768, False))
    for capacity in (2048, 8192, 32768):
        cases.append((1, 1024, capacity, False))
    for history in (1024, 4096):
        cases.append((8, history, 8192, True))
    for batch, history, capacity, ragged in cases:
        values = time_case(batch, history, capacity, ragged)
        label = f"{history}{'*' if ragged else ''}"
        times, diff = values[: len(paths)], values[len(paths) :]
        print(
            f"{batch:>5}{label:>9}{capacity:>10}"
            + "".join(f"{t:>18.2f}" for t in times)
            + "".join(f"{d:>12.2e}" for d in diff),
            flush=True,
        )
    print("* ragged batch: row b holds history * (b + 1) / batch tokens")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("what", choices=["split", "mla"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--mla-path", choices=["current", "decompressed", "absorbed"], default="current"
    )
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--experts", type=int, default=32)
    parser.add_argument("--vocab", type=int, default=32000)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--prompt", type=int, default=1024)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--trace-dir", default="/tmp")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.what == "split":
        run_split(args)
    else:
        run_mla(args)


if __name__ == "__main__":
    main()
