# mini-infer

An open-source LLM inference engine built from scratch in Python and Triton, implementing the techniques that production engines (vLLM, SGLang, TensorRT-LLM) are known for: continuous batching, PagedAttention with a Triton decode kernel, a multi-model registry of owned `nn.Module` implementations, and an OpenAI-compatible FastAPI server with SSE streaming. Nine model families register today: Qwen2, Qwen3, Llama, Mistral, Gemma 3, Gemma 4, Mixtral, DeepSeek-V2, and a from-scratch implementation of every published DeepSeek-V4 primitive (hybrid CSA + HCA attention, hash-routed MoE FFN, Hyper-Connections residuals, YaRN long-context RoPE) integrated end-to-end through a runnable backbone — bit-parity validated against the upstream inference reference where applicable.

## What's working today

* **Continuous batching scheduler** (`ContinuousScheduler`) on a dedicated engine thread with FIFO admission, per-request handles, and backpressure. One forward pass per step over all in-flight decoding requests.
* **Chunked prefill + packed-varlen forward**: long prompts advance one chunk per step alongside in-flight decoders, eliminating head-of-line blocking. Single `model.forward(...)` per step via FlashAttention's varlen API on CUDA, PyTorch reference elsewhere.
* **PagedAttention** with a fixed-size block pool and a batch-aware `PagedKVCache`. Supports per-layer **storage descriptors**: each layer carries a list of named `StreamSpec`s rather than a fixed `(K, V)` pair. Standard MHA / GQA layers use `["k", "v"]` (the streams alias the rectangular layout, no extra memory); MLA layers carry `["kv_latent", "k_rope"]` of differing shape (DeepSeek-V2/V3 — ~7x smaller cache than equivalent MHA). The same primitive handles Gemma 4 31B's heterogeneous `(num_kv_heads, head_dim)` per layer-type and is the foundation DeepSeek-V4's CSA + HCA hybrid attention will extend further.
* **Multi-model framework**: `ModelRegistry` looks up an HF `config.architectures[0]` string and dispatches to an owned `nn.Module` (Llama-style or family-specific). Nine families register today: `Qwen2ForCausalLM`, `Qwen3ForCausalLM`, `LlamaForCausalLM` (covers Llama 2/3/4 + SmolLM2 + TinyLlama + the Llama-shape Nemotron variants), `MistralForCausalLM`, `Gemma3ForCausalLM`, `Gemma4ForConditionalGeneration` (Gemma 4 31B text-only), `MixtralForCausalLM`, `DeepseekV2ForCausalLM` (V2-Lite + V2 + V3 + Kimi-K2 share the same MLA shape), `DeepseekV4ForCausalLM` (hybrid CSA + HCA attention — see the V4 entry below). Adding a new model is one ~80-line file composing the shared block library — RMSNorm + RoPE + GQA + SwiGLU + MoEFFN plus the family-specific extensions (sliding-window attention, dual RoPE, partial RoPE, per-head Q/K/V norm, sandwich norm, top-k MoE with shared experts, k_eq_v, MLA with low-rank Q + compressed KV + interleaved RoPE, V4's hybrid CSA/HCA dispatch + Lightning Indexer + token-level compressor + attention sink + grouped output projection, per-layer attention-backend override). HF safetensors weights load via `model.load_state_dict(...)` for non-TP families, or via `load_state_dict_with_tp` when running multi-GPU (column/row/vocab-parallel layers slice on load). No HF runtime monkey-patching.
* **DeepSeek-V4 from-scratch contribution** — every published V4 primitive implemented and integrated through a runnable backbone:
   * **Attention** (paper §2.3): `TokenLevelCompressor` (overlap + non-overlap), `LightningIndexer` (per-head ReLU dot-product top-k), `AttentionSink` (per-head softmax-denominator logit), `GroupedOutputProjection` (`n_h` heads → `g` groups → `hidden_size`), partial RoPE on the last `rope_head_dim` dims, plus MQA-with-sink for the asymmetric Q-vs-KV core. Both modes (CSA `m=4` overlap+indexer, HCA `m'=128` all-compressed-visible) ship with prefill + incremental decode + cache-aware prefill (aligned and unaligned). Per-request `StateCache` carries SWA circular buffer + compressor in-flight accumulator + indexer sub-cache; slide-on-flush in CSA's overlap mode mirrors the reference exactly. **Bit-parity (cosine sim 1.0000, max abs diff < 1e-9)** vs the [V4-Pro inference reference](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main/inference) for prefill, decode, and the prefill→decode transition.
   * **Hash-routed MoE FFN** (paper §2.2): `HashRoutedGate` supports `routing_mode="hash"` (per-token-id `tid2eid` lookup for the first `num_hash_routed_layers` of V4) and `routing_mode="score_topk"` (the rest); three score functions (softmax / sigmoid / softplus_sqrt) with the right renorm rule per V4 paper. `HashRoutedMoEFFN` composes the gate with `MixtralExpert`-shaped MLPs + the V2/V3-style "collapse N shared experts into one MLP" pattern. Decoder layer threads `input_ids` through prefill and decode for the hash branch.
   * **Hyper-Connections residuals** (paper §2.5): `hc_split_sinkhorn` transcribed line-by-line from the reference's tilelang kernel into pure PyTorch. `HyperConnections` block wraps `(hc_pre, hc_post)` with the learnable `(fn, base, scale)` parameters; the hidden state carries `hc_mult` copies through every decoder layer, mediated by Sinkhorn-normalized doubly-stochastic mixing. `HCHeadReduction` collapses `(B, T, hc_mult, dim) → (B, T, dim)` before the LM head.
   * **YaRN long-context RoPE** (paper §3.1): wave-frequency-correction kicks in past `yarn_original_seq_len`. Wired into `RotaryEmbedding` so V2 / V3 / V4 share the same primitive.
   * **Backbone** (`DeepseekV4ForCausalLM`): per-layer dispatch over three attention modes driven by `compress_ratios` — `0` → pure SWA (no compressor, no indexer; V4-Flash uses 2 such layers at the head of its 43-layer stack), `4` → CSA with `LightningIndexer` + top-k, `128` → HCA. End-to-end prefill + decode runs with all primitives composed (SWA + CSA + HCA + Sink + Grouped Output + StateCache + hash-routed MoE + Hyper-Connections); 100+ V4-related tests cover the integration matrix.
   * **V4-Flash loader** (`DeepseekV4ForCausalLM.load_weights`): handles the published V4-Flash storage format end-to-end. Block-FP8 e4m3fn dequant for non-MoE weights (128x128 block scales), NVFP4 dequant for MoE experts (block-32 e8m0fnu scales; int8/uint8 packed storage — the way safetensors actually ships them). V4-Flash uses the reference's compact key convention (`layers.X.attn.wq_a`, `attn.attn_sink`, `ffn.experts.Y.w1`, ...); a rename map remaps the entire checkpoint onto our module hierarchy. `from_hf` parses the real `config.json` directly (transformers 5.x doesn't yet ship a native V4 config and drops unknown fields), wiring MoE (256 routed experts, 6 active per token, 1 shared, 3 hash-routed layers, `sqrtsoftplus` scoring) and Hyper-Connections (`hc_mult=4`) from published values. Expert-parallel global→local index remap shards 256 experts across ranks. Meta-device construction + CPU-resident state_dict + per-slice GPU streaming keep peak memory bounded — every rank holds its ~80 GB share, never the full 158 GB. Dry-run against the real V4-Flash safetensors index: all 34,223 model parameters match exactly (zero missing, zero unexpected after dequant pairing consumes 33,389 sibling `.scale` companions); dequant numerically verified against a real V4-Flash shard.
   * **Tensor parallelism**: Megatron-style column/row-parallel linears, vocab-parallel embedding, and expert-parallel MoE all live in `src/mini_infer/distributed/`. Every attention type (GQA, MLA, HCA, CSA, SWA), FFN type (SwiGLU, GeGLU, MoE, HashRoutedMoEFFN), and `embed_tokens` / `lm_head` are TP-aware. At `world_size=1` every module is bit-equivalent to its plain-PyTorch counterpart, so single-device behaviour is unchanged. Multi-process CPU parity tests (gloo backend) validate every TP-aware primitive at `world_size=2`. Validated on real H100 hardware: a Qwen2.5-7B two-rank TP run produced finite per-rank sliced outputs with the expected `hidden_size / world_size` head sharding.
   * **Demo**: `scripts/demo_deepseek_v4_hybrid.py` runs a 4-layer CSA/HCA stack end-to-end (prefill + 12 decode steps) on synthetic weights.
* **Prefix caching**: chained-hash, block-granular, refcounted LRU. Repeat or shared-prefix prompts skip prefill on the cached prefix; opt-in via `prefix_cache=True`. Verified token-for-token against the no-cache path.
* **Weight-only INT8 quantization (W8A16)**: symmetric per-output-channel scales applied at load time; opt-in via `quant="int8"`. Drops model-weight HBM by ~30% on Qwen2.5-0.5B with cosine-sim > 0.99 on logits and first-token greedy parity preserved. Forward dispatches to a fused Triton W8A16 GEMM kernel on CUDA — keeps weights in INT8 in HBM and dequants tile-by-tile in registers, skipping the bf16-weight HBM round-trip the naive path pays.
* **Speculative decoding** (vanilla two-model, greedy V1): small draft model proposes K tokens, large target verifies them in one forward, accept-reject emits target's argmax sequence. `PagedKVCache.truncate_to` rolls back on rejections. 1.14x decode throughput on Qwen2.5-7B target + 0.5B draft on A10 at bf16; the regime is constrained by the modest target/draft size ratio (the same implementation scales to the published 1.5–2x range at 70B+ on Hopper).
* **Prefill / decode disaggregation (PD)**: `PrefillWorker` and `DecodeWorker` each own a `ModelRunner` and a phase; a `KVHandoff` (per-layer per-stream packed KV + first sampled token + sampling params) is the contract between them. `kv_transfer.send_handoff` / `recv_handoff` ship the handoff over `torch.distributed` (gloo on CPU, NCCL on CUDA) with a 7×int64 header + KV streams in `pool.stream_names`-sorted order. `prefill_batch` / `decode_batch` methods give intra-worker batching (one forward per phase instead of N). Greedy parity vs `ContinuousScheduler` is the correctness contract: PD's output is token-for-token identical to the non-disaggregated path. Validated in-process and multi-process (CPU + gloo for the wire protocol). The 2× H100 Modal smoke script ships with the slice; the live `modal run` remains as the final hardware gate. See [ADR-016](docs/decisions/ADR-016-pd-disaggregation.md).
* **TurboQuant KV cache** (V1 + V3 full + fused dequant kernel): rotation + per-block 4-bit (V1, `kv_quant="turbo4"`) or rotation + polar transform + Lloyd-Max codebook + QJL residual sign + asymmetric K(3-bit + QJL)/V(4-bit) (V3, `kv_quant="turbo3"`). V3 saves ~74% KV memory (vs V1's 62%) and produces less-degenerate output at 7B; V1 still preserves full argmax parity at 0.5B. A Triton kernel collapses turbo3's per-block dequant into a single launch per layer, lifting decode throughput from 0.04–0.18x of bf16 (Python loop) to 0.31–0.41x on Qwen2.5-0.5B + A10. Per-storage-class throughput vs the same-class Python loop is 6.2–6.9x.
* **FlashInfer attention backend** with three KV storage modes selectable via `attention_backend="flashinfer"`:
   * **bf16** (default with the FlashInfer backend): same paged storage as flash-attn, routed through FlashInfer's tensor-core wrapper. Token-for-token parity with the flash-attn path.
   * **FP8** (`kv_quant="fp8"`): FP8 e4m3fn paged storage with FlashInfer fusing dequant into the attention kernel. **50% KV memory savings vs bf16** on Hopper (H100/H200); logit cosine sim 0.999985, 0.93x throughput on Qwen2.5-0.5B at 2k context. The production-grade 8-bit KV path on Hopper.
   * **NVFP4** (`kv_quant="nvfp4"`, Blackwell only): FP4-packed paged storage + per-16-element FP8 block scales, dequant fused into the kernel via `kv_cache_sf`. **71.9% KV memory savings vs bf16** on B200; 0.91x throughput on Qwen2.5-7B. Token-level accuracy is degraded under greedy decode (per-(layer,side) global scale leaves bulk K/V values quantized toward FP4-zero in the presence of outliers); production NVFP4 deployments need outlier-aware preprocessing we haven't built. Integration infrastructure is correct (kernel path validated within FlashInfer's own 1e-1 tolerance); treat this as working FP4 plumbing pending a calibration layer.
* **Triton decode kernel** (single-request and batched variants) with online-softmax accumulation, validated against a PyTorch reference within cosine similarity > 0.99.
* **OpenAI-compatible HTTP API** (`/v1/completions`) with both non-streaming responses and SSE streaming.
* **Sampler** with greedy, temperature, top-k, top-p; pure-logic unit tests.
* **CI** running `ruff`, `mypy --strict`, and the full unit + golden test suite (CPU-only path) on every push.

Throughput on Qwen2.5-0.5B-Instruct, NVIDIA A10, bf16:

| Concurrency | Tokens/sec | Multiplier vs C=1 |
|---:|---:|---:|
| 1 | 45.7 | 1.00x |
| 2 | 79.5 | 1.74x |
| 4 | 127.7 | 2.79x |
| 8 | 187.5 | 4.10x |

Full report: [docs/benchmarks/2026-04-27-continuous-batching.md](docs/benchmarks/2026-04-27-continuous-batching.md).

Prefix caching on a 15.9k-token shared system prompt + 8 unique short user questions, A10, bf16:

| Concurrency | cache OFF | cache ON | Throughput speedup |
|---:|---:|---:|---:|
| 1 | 2.57 tok/s | 34.1 tok/s | **13.27x** |
| 4 | 2.87 tok/s | 74.3 tok/s | **25.88x** |
| 8 | 2.91 tok/s | 94.4 tok/s | **32.45x** |

Warm-cache TTFT: ~74ms vs ~11.7s (cold), a 158x reduction once the system prompt has been served once. Full report: [docs/benchmarks/2026-04-28-prefix-caching.md](docs/benchmarks/2026-04-28-prefix-caching.md).

Weight-only INT8 (W8A16) on Qwen2.5-0.5B, A10: model-weight HBM drops from 1142 MiB (fp16) to 794 MiB (**−30.5%**) with cosine similarity > 0.99 on logits and first-token greedy parity preserved. Throughput is ~neutral (no FLOP savings; a fused dequant kernel is the path to compute speedup). Full report: [docs/benchmarks/2026-04-28-int8-weight-quant.md](docs/benchmarks/2026-04-28-int8-weight-quant.md).

Fused W8A16 Triton kernel on Qwen2.5-7B, A10, bf16: same kernel that loses 0.74–0.99x to cuBLAS at 0.5B wins **2.74x at decode (C=1)** and 1.47–1.65x at higher concurrency on 7B. The regime flip is exactly where the math predicts: at 7B the bf16-weight HBM round-trip in the naive int8 path becomes the dominant cost per forward, and that's what the fused kernel skips. Full report: [docs/benchmarks/2026-04-29-fused-int8-kernel.md](docs/benchmarks/2026-04-29-fused-int8-kernel.md).

Speculative decoding on Qwen2.5-7B target + Qwen2.5-0.5B draft, bf16, K=4: aggregate **1.14x on A10**, **1.00x on H100** vs target-alone greedy on a 3-prompt workload. Mean acceptance 2.3–3.4 / K=4 (58–85%). Below the published 1.5–2x range; the H100 result is the more interesting finding — faster baseline decode on Hopper *narrows* the win at this target size rather than widening it (1.29x on the longest prompt, 0.83x on the shorter ones). Full report: [docs/benchmarks/2026-04-29-speculative-decoding.md](docs/benchmarks/2026-04-29-speculative-decoding.md).

FlashInfer FP8 KV on Qwen2.5-0.5B + H100 (2k-token prefill, single request): KV pool 192 MiB → 96 MiB (**−50.0%**), logit cos sim 0.999985 vs bf16, 0.93x throughput. Full report: [docs/benchmarks/2026-05-02-flashinfer-fp8-kv.md](docs/benchmarks/2026-05-02-flashinfer-fp8-kv.md).

FlashInfer NVFP4 KV on Qwen2.5-7B + B200 (2k-token prefill, single request): KV pool 1792 MiB → 504 MiB (**−71.9%**), 0.91x throughput vs bf16. Token-level accuracy degraded under greedy decode (FP4 precision tradeoff; production deployments need outlier-aware calibration). Full report: [docs/benchmarks/2026-05-02-flashinfer-nvfp4-kv.md](docs/benchmarks/2026-05-02-flashinfer-nvfp4-kv.md).

## Quickstart

```bash
# Install (Python 3.11+; uv as the package manager)
uv sync

# Run the unit tests (CPU only, no model download)
uv run pytest tests/unit/ -v

# Run the API server (downloads Qwen2.5-0.5B-Instruct on first start)
uv run python -m mini_infer.api.server

# In another shell, hit the API
curl -N http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen","prompt":"The capital of France is","max_tokens":8,"stream":true}'

# Demo: DeepSeek-V4 hybrid CSA/HCA backbone (synthetic weights, no download)
uv run python scripts/demo_deepseek_v4_hybrid.py
```

The default model is `Qwen/Qwen2.5-0.5B-Instruct`; override with `MINI_INFER_MODEL=...`.

Device selection is automatic: CUDA if available, else MPS on Apple Silicon, else CPU. The Triton paged kernel runs on CUDA; other devices fall back to a numerically-equivalent PyTorch reference.

## How to use mini-infer

### Programmatic API

The engine entry point is `mini_infer.engine.model_runner.ModelRunner.from_pretrained`. It accepts knobs for all the optional features:

```python
from mini_infer.engine.model_runner import ModelRunner
from mini_infer.engine.sampler import SamplingParams
from mini_infer.scheduler import ContinuousScheduler, Request

runner = ModelRunner.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    # Default: bf16 KV / flash-attn varlen on CUDA, PyTorch reference elsewhere.
    quant=None,                  # "int8" enables W8A16 weight-only quantization
    kv_quant=None,               # "turbo4" | "turbo3" | "fp8" | "nvfp4"
    attention_backend="flash_attn",  # "flashinfer" required for fp8/nvfp4
    block_size=16,
    num_blocks=512,
)

scheduler = ContinuousScheduler(runner)
scheduler.start()
try:
    result = scheduler.run(Request(
        prompt="The capital of France is",
        sampling_params=SamplingParams(temperature=0.0),  # greedy
        max_tokens=32,
    ))
    print(result.text)
finally:
    scheduler.stop()
```

### Supported model families

Nine HF architecture keys register today; pass any HF model id whose config matches and `ModelRunner.from_pretrained(...)` will route through the right owned class. (V4 is registered for the architecture string; `load_weights` walks the HF state_dict with FP8 dequant + MoE renames and dispatches through TP, with NVFP4 expert dequant pending — see the V4 row.)

| HF architecture | Examples | Family-specific primitives |
|---|---|---|
| `Qwen2ForCausalLM` | Qwen2.5-0.5B / 7B / 32B-Instruct | Biased Q/K/V projections |
| `Qwen3ForCausalLM` | Qwen3-0.6B / 1.7B / 4B / 8B / 14B / 32B | Per-head Q/K norm, tied embeddings |
| `LlamaForCausalLM` | Llama 2 / 3 / 4, SmolLM2, TinyLlama, Llama-Nemotron variants | Standard Llama-shape baseline |
| `MistralForCausalLM` | Mistral-7B-Instruct-v0.1/0.2/0.3, Mistral Small / Large | Same as Llama (registered separately for the HF arch key) |
| `Gemma3ForCausalLM` | Gemma 3 1B / 4B (text-only) | Sliding-window + global alternating attention, dual RoPE, sandwich norms, GemmaRMSNorm (`(1+w)*x`), GeGLU (`gelu_pytorch_tanh`), embed scaling, Q/K norm |
| `Gemma4ForConditionalGeneration` | Gemma 4 31B-it (text-only; vision/audio towers filtered at load) | Heterogeneous-KV per layer-type (sliding `head_dim=256, kv=16` / full `head_dim=512, kv=4`), `attention_k_eq_v` (full layers reuse `k_proj` output as V), unscaled `v_norm`, per-layer `layer_scalar`, dual RoPE with different `head_dim` per type (full layers use proportional rotation, `partial_rotary_factor=0.25`), final logit softcap, model-side attention-backend override (forces materialized SDPA because head_dim=512 exceeds flash-attn / FlashInfer limits — same conclusion vLLM and SGLang reach with their Triton unified kernel) |
| `MixtralForCausalLM` | Mixtral-8x7B / 8x22B-Instruct | Top-k sparse MoE FFN (8 experts, top-2) |
| `DeepseekV2ForCausalLM` | DeepSeek-V2-Lite / V2-Lite-Chat (16B); same class extends to V2 (236B), V3 (671B), Kimi-K2 (1T) via `q_lora_rank` toggle | **Multi-head Latent Attention** (compressed `kv_latent` of dim `kv_lora_rank=512` + decoupled `k_rope` of dim `qk_rope_head_dim=64`, both shared across heads — ~7x smaller cache than equivalent MHA), low-rank Q (`q_a_proj → q_a_layernorm → q_b_proj` for V2/V3, direct `q_proj` for V2-Lite), interleaved RoPE (DeepSeek convention, pairs `(x[2i], x[2i+1])` rotate together), asymmetric Q/K vs V head_dim (192 vs 128), heterogeneous FFN per layer (`SwiGLU` for `first_k_dense_replace` layers, `MoEFFN` after with shared experts + `routed_scaling_factor`), model-side `"torch"` backend override |
| `DeepseekV4ForCausalLM` | DeepSeek-V4-Pro (862 GB) / V4-Flash (158 GB) *(load path proven against real V4-Flash safetensors; end-to-end forward on 2× B200 is the remaining gate)* | **Every published V4 primitive integrated end-to-end.** Attention: three-mode per-layer dispatch via `compress_ratios` — ratio=0 → pure SWA, ratio=4 → CSA with `LightningIndexer` + top-k, ratio=128 → HCA — plus `TokenLevelCompressor` (overlap + non-overlap), `AttentionSink`, `GroupedOutputProjection`, partial RoPE, per-request `StateCache` with cache-aware prefill (aligned + unaligned) + decode. **MoE FFN** via `HashRoutedMoEFFN` with per-layer hash / score-topk routing, three score functions (`softmax`, `sigmoid`, `sqrtsoftplus`), shared-expert collapse. **Hyper-Connections** via `HyperConnections` (line-by-line transcription of the reference's `hc_split_sinkhorn` kernel) + `HCHeadReduction` for the LM-head collapse. **YaRN long-context RoPE** via the same `RotaryEmbedding` block V2/V3 use. Bit-parity validated against the [V4-Pro inference reference](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main/inference) for HCA/CSA prefill and decode (cosine sim 1.0000, max abs diff < 1e-9), the MoE gate + FFN (4 routing × score_func cells), and `Block.hc_pre` / `Block.hc_post` (`hc_mult ∈ {2, 3, 4}`). `load_weights` handles the published V4-Flash format end-to-end: block-FP8 (e4m3fn weight + e8m0fnu 128x128 scales) for non-MoE projections, NVFP4 (int8/uint8 packed weight + e8m0fnu block-32 scales) for MoE experts, V4-reference compact key remap, expert-parallel global→local index remap, meta-device + CPU state_dict + per-slice GPU streaming. Dry-run vs the real V4-Flash safetensors index: 34,223 model params match the 34,223 renamed checkpoint keys exactly (zero missing / zero unexpected after dequant pairing); dequant numerically verified against a real V4-Flash shard. See [ADR-014](docs/decisions/ADR-014-deepseek-v4-hybrid-attention.md). |

Validated path: each loadable family has at least one CPU/MPS smoke test that loads an ungated checkpoint and produces "Paris" for `"The capital of France is"`. Mixtral 8x7B (47B params) is too large for M1 fp16 — the MoE block was bit-validated against HF's `MixtralSparseMoeBlock` on synthetic input instead. Gemma 4 31B (62 GB at bf16) and DeepSeek-V2-Lite-Chat (~31 GB at bf16) are similarly out of M1 memory budget; both validated end-to-end on B200 Modal runs. The `MLAAttention` block has bit-parity (cosine sim > 0.999) against HF's `DeepseekV2Attention` on synthetic configs. **Tensor parallelism** is validated on real H100 hardware: a Qwen2.5-7B two-rank TP run produces finite per-rank sliced outputs with the expected `hidden_size / world_size` head sharding. **DeepSeek-V4** runs on synthetic configs end-to-end; V4 weights are public (V4-Flash 158 GB, V4-Pro 862 GB) and exceed single-GPU HBM, so loading them uses mini-infer's Megatron-style tensor parallelism (column/row-parallel linears, expert-parallel MoE, vocab-parallel embedding). The V4-Flash loader now covers the full published storage format (block-FP8 + NVFP4 dequant, reference-compact key rename, expert-parallel index remap, meta-device + per-slice GPU streaming); a dry-run against the real V4-Flash safetensors index matches all 34,223 model parameters exactly, and dequant is numerically verified against a real shard. Every published V4 primitive is bit-parity validated against the upstream inference reference, and the full backbone runs end-to-end via `scripts/demo_deepseek_v4_hybrid.py`.

Adding a model family is a one-file change in `src/mini_infer/models/<family>.py`: declare a config + a class that composes the shared blocks, decorate with `@register_model`, and add the import to `_register_builtin_models()` in `src/mini_infer/models/__init__.py`. Examples to copy, in order of complexity:

* Pure Llama-shape (no biases, no quirks): [src/mini_infer/models/mistral.py](src/mini_infer/models/mistral.py) — 30 lines.
* Llama-shape + Q/K norm + tied embeddings: [src/mini_infer/models/qwen3.py](src/mini_infer/models/qwen3.py) — ~140 lines.
* Sandwich norm + dual RoPE + GemmaRMSNorm: [src/mini_infer/models/gemma3.py](src/mini_infer/models/gemma3.py) — ~170 lines.
* Heterogeneous-KV + k_eq_v + v_norm + dual-RoPE-different-head_dim + softcap + multimodal weight prefix-strip + model-side backend override: [src/mini_infer/models/gemma4.py](src/mini_infer/models/gemma4.py) — ~280 lines.
* MLA attention + per-stream KV cache + heterogeneous FFN (dense + MoE with shared experts) + interleaved RoPE: [src/mini_infer/models/deepseek_v2.py](src/mini_infer/models/deepseek_v2.py) — ~280 lines.
* Hybrid per-layer attention (CSA + HCA) + per-request `StateCache` + Lightning Indexer + token-level compressor (with overlap mode + slide-on-flush in decode) + attention sink + grouped output projection: [src/mini_infer/models/deepseek_v4.py](src/mini_infer/models/deepseek_v4.py) + [src/mini_infer/models/blocks/deepseek_v4_decoder_layer.py](src/mini_infer/models/blocks/deepseek_v4_decoder_layer.py) + [src/mini_infer/models/blocks/v4/](src/mini_infer/models/blocks/v4/) — ~1.6k lines across the family.

### Picking a KV-cache mode

| `kv_quant` | `attention_backend` | KV memory vs bf16 | Best on | Notes |
|---|---|---|---|---|
| `None` (default) | `"flash_attn"` | 100% (baseline) | All CUDA | Token-for-token parity; the reference path. |
| `None` | `"flashinfer"` | 100% | All CUDA | bf16 KV via FlashInfer; same parity, different kernel. Useful as a stepping stone before enabling FP8/NVFP4. |
| `"turbo4"` | `"flash_attn"` | 38% (−62%) | A10/A100 | Argmax parity at 0.5B; 7B drifts. Production-grade kernel not built. |
| `"turbo3"` | `"flash_attn"` | 26% (−74%) | A10/A100 | Coherent-but-different output at 0.5B; less degenerate at 7B. Fused Triton kernel keeps decode at 0.31–0.41x of bf16. |
| `"fp8"` | `"flashinfer"` | 50% (−50%) | **H100/H200** | Logit cos sim 0.999985 vs bf16; 0.93x throughput. The production answer for 8-bit KV on Hopper. |
| `"nvfp4"` | `"flashinfer"` | 28.1% (−71.9%) | **B200** | 4-bit KV on Blackwell. 0.91x throughput. Token-level accuracy degraded under greedy decode without outlier-aware calibration; ship infrastructure only, not production-quality outputs (see [bench doc](docs/benchmarks/2026-05-02-flashinfer-nvfp4-kv.md)). |

Quick examples for each non-default mode:

```python
# Weight-only INT8 (any CUDA GPU)
runner = ModelRunner.from_pretrained("Qwen/Qwen2.5-7B-Instruct", quant="int8")

# TurboQuant V3 (KV compression to ~26% of bf16, A10-grade)
runner = ModelRunner.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", kv_quant="turbo3")

# FP8 KV via FlashInfer (Hopper)
runner = ModelRunner.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    kv_quant="fp8",
    attention_backend="flashinfer",
)

# NVFP4 KV via FlashInfer (Blackwell only)
runner = ModelRunner.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    kv_quant="nvfp4",
    attention_backend="flashinfer",
)
```

### Speculative decoding

Use `SpeculativeRunner` when you have a target + draft pair (same vocab family, e.g. Qwen2.5-7B target + Qwen2.5-0.5B draft):

```python
from mini_infer.engine.model_runner import ModelRunner
from mini_infer.engine.speculative import SpeculativeRunner

target = ModelRunner.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
draft = ModelRunner.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
runner = SpeculativeRunner(target=target, draft=draft, K=4)
token_ids, stats = runner.run_greedy(prompt="Explain recursion.", max_tokens=128)
print(target.tokenizer.decode(token_ids))
print(f"acceptance: {stats.mean_acceptance_per_iter:.2f}/{runner.K}")
```

### HTTP server

The `/v1/completions` endpoint mirrors the OpenAI Completions API (subset). The server entry point is `mini_infer.api.server`; today only `MINI_INFER_MODEL` is wired as an env var. For non-default KV quantization or attention backends, build the runner programmatically and pass it to `make_app(...)` in your own startup script — the server module is small enough to copy-and-modify.

```bash
# Default: bf16 KV / flash-attn, model from MINI_INFER_MODEL or Qwen2.5-0.5B-Instruct
uv run python -m mini_infer.api.server

# Override the model only
MINI_INFER_MODEL="Qwen/Qwen2.5-7B-Instruct" uv run python -m mini_infer.api.server
```

### Modal benchmarks

Each major feature ships with a Modal bench script that reproduces the published numbers. Examples:

```bash
# Continuous-batching throughput on A10
modal run scripts/modal_packed_bench.py --config throughput

# Prefix caching speedup on a long shared prompt
modal run scripts/modal_packed_bench.py --config prefix

# FP8 KV on H100
MINI_INFER_BENCH_GPU=H100 modal run scripts/modal_packed_bench.py \
    --config flashinfer_fp8 --model "Qwen/Qwen2.5-0.5B-Instruct"

# NVFP4 KV on B200 (HF token required for 7B+ model downloads)
HF_TOKEN=$(hf auth token) MINI_INFER_BENCH_GPU=B200 \
    modal run scripts/modal_packed_bench.py \
    --config flashinfer_nvfp4 --model "Qwen/Qwen2.5-7B-Instruct" --num-blocks 2048
```

See `scripts/modal_packed_bench.py --help` for the full list of configs.

## Architecture

```
HTTP request
   │
   ▼
FastAPI (api/server.py) ──► ContinuousScheduler.submit(Request)
                                    │
                              engine thread
                                    │
                ┌───────────────────┼─────────────────────┐
                ▼                   ▼                     ▼
          admit waiting       prefill (B=1)         batched decode
                                    │                     │
                                    └────► PagedKVCache ◄─┘
                                                │
                                                ▼
                                         BlockPool (shared)
                                                │
                                                ▼
                                Triton kernel  /  PyTorch reference
                                  (CUDA)           (CPU/MPS)
```

* **`engine/`** loads the model via `mini_infer.models.load_model(...)`, owns prefill and decode.
* **`models/`** is the multi-model registry: per-family `nn.Module` implementations + a shared block library (RMSNorm, RoPE, GQA, SwiGLU, MoE FFN, GemmaDecoderLayer, ...). HF safetensors weights load directly into our hierarchies with no runtime patching.
* **`scheduler/`** owns the engine thread, admission control, the running batch, and the shared `PagedKVCache`.
* **`cache/`** holds the block pool, the batch-aware cache, and the paged attention kernel + reference. The block pool supports per-layer named KV streams via `StreamSpec` (Gemma 4's heterogeneous KV, MLA's `kv_latent` + `k_rope`). V4 attention adds a separate `StateCache` for the per-request SWA circular buffer + compressor in-flight accumulator + Lightning Indexer sub-cache.
* **`api/`** is FastAPI with OpenAI-compatible endpoints and SSE streaming.

## Design decisions

Each non-trivial choice has an Architecture Decision Record under [docs/decisions/](docs/decisions/):

* [ADR-001](docs/decisions/ADR-001-uv-as-package-manager.md): `uv` as package manager
* [ADR-002](docs/decisions/ADR-002-starter-model.md): Qwen2.5-0.5B-Instruct as the starter model
* [ADR-003](docs/decisions/ADR-003-paged-kv-cache.md): block-based KV cache layout
* [ADR-004](docs/decisions/ADR-004-paged-attention-kernel.md): Triton paged attention kernel
* [ADR-005](docs/decisions/ADR-005-continuous-batching-integration.md): batch-aware cache and the continuous-batching forward pass
* [ADR-006](docs/decisions/ADR-006-chunked-prefill.md): chunked prefill (two-forward design)
* [ADR-007](docs/decisions/ADR-007-packed-forward-integration.md): packed-varlen forward, one `model.forward` per step
* [ADR-008](docs/decisions/ADR-008-paged-fa-varlen.md): paged FlashAttention varlen, kept as a tunable
* [ADR-009](docs/decisions/ADR-009-prefix-caching.md): prefix caching (chained-hash, block-granular, refcounted LRU)
* [ADR-010](docs/decisions/ADR-010-int8-weight-quant.md): weight-only INT8 quantization (W8A16)
* [ADR-011](docs/decisions/ADR-011-speculative-decoding.md): speculative decoding (vanilla two-model, greedy, single-request)
* [ADR-012](docs/decisions/ADR-012-fused-int8-kernel.md): fused W8A16 Triton kernel for `Int8Linear`
* [ADR-013](docs/decisions/ADR-013-turboquant-kv.md): TurboQuant KV cache (V1 + V3 + V2a fused dequant; V2b deselected); FlashInfer FP8 + NVFP4 KV
* [ADR-014](docs/decisions/ADR-014-deepseek-v4-hybrid-attention.md): DeepSeek-V4 hybrid attention contribution (CSA + HCA primitives, per-request StateCache, registered hybrid backbone)
* [ADR-015](docs/decisions/ADR-015-tensor-parallelism.md): tensor parallelism (Megatron-style column/row/vocab linears, expert-parallel MoE, per-rank weight loader)
* [ADR-016](docs/decisions/ADR-016-pd-disaggregation.md): prefill/decode disaggregation (PD) (`PrefillWorker` + `DecodeWorker` + `KVHandoff` + multi-process KV transport)

## Tests

Three layers, each with a clear contract:

* **Unit** (`tests/unit/`): CPU-only, fast, run on every CI push. Covers the block pool, paged cache, paged attention reference, scheduler, sampler, tokenizer, API schemas.
* **Golden** (`tests/golden/`): CPU/fp32. Token-for-token comparison against Hugging Face's reference output for a fixed set of prompts. Catches numerical drift before it ships.
* **Modal smokes + benchmarks** (`scripts/`): real CUDA validation on Modal A10. Runs on demand only.

Markers: `requires_model` (skipped in CI; needs an HF model download) and `requires_cuda` (skipped on non-CUDA hardware).

```bash
uv run pytest tests/unit/ -q                       # CI-equivalent
uv run pytest tests/golden/ -q                     # token-for-token vs HF
uv run pytest tests/unit/ -m "not requires_model"  # skip model-dependent tests
```

## What's NOT in here

By design:

* No multimodal support (text-only). Gemma 3's vision/audio towers, Gemma 4's image/audio inputs, and Phi-3/4-Vision are explicitly skipped — the text decoder is what we implement.
* No fine-tuning or training pipeline (inference only).
* No reimplementation of FlashAttention or other vendor kernels for attention math itself. We use `flash-attn` 2.8+ for varlen attention on CUDA, FlashInfer for FP8/NVFP4 paged paths, and a PyTorch SDPA reference on CPU/MPS. We do own custom Triton kernels for INT8 GEMM (`Int8Linear`) and TurboQuant KV dequant.

Future model support (planned, see [docs/plans/multi-model-support.md](docs/plans/multi-model-support.md)):

* **DeepSeek-V3 (671B) + Kimi-K2 (1T)** — same MLA architecture as V2; `from_hf` and the registry already accept them once the larger checkpoints are exercised on a Modal run with sufficient HBM.
* **Tensor parallelism** — Megatron-style column/row-parallel linears, vocab-parallel embedding, and expert-parallel MoE all live in `src/mini_infer/distributed/`. Every attention type and FFN primitive in the codebase is TP-aware (no-op at `world_size=1`). Required for >100B models that don't fit a single GPU; the remaining work is real-weight validation on Modal.
* **DeepSeek-V4-Flash end-to-end forward on 2× B200** — every published V4 architecture primitive is implemented (see [ADR-014](docs/decisions/ADR-014-deepseek-v4-hybrid-attention.md)) and the V4-Flash loader handles the real published storage format end-to-end (block-FP8 + NVFP4 dequant, reference-compact key rename, expert-parallel sharding, meta-device + per-slice GPU streaming; load-path dry-run matches the real safetensors index exactly). The remaining gate is a full forward pass through TP on 2× B200. Load is proven on real hardware; three dtype + meta-init fixes (compressor projection dtype, Hyper-Connections FP32 preservation, rotary `inv_freq` re-materialization) are statically clean. One end-to-end validation run remains.
* **State-space hybrids** (Mamba, Nemotron-H, Jamba) — different cache abstraction entirely; deliberately out of scope.

## License

(Pending.)
