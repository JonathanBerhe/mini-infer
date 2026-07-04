# MLX custom Metal kernels as an Apple Silicon inference path for mini-infer

Status: research report, decision-oriented
Date: 2026-06-16
Scope: evaluate `mx.fast.metal_kernel` as a candidate first-class Apple Silicon (M1-M4) inference target, and decide adopt / don't-adopt / adopt-narrowly with a phased "how" if warranted.

## TL;DR recommendation

**Adopt narrowly.** Keep CUDA as the only benchmarked production target. Do NOT promote Apple Silicon to a co-equal performance backend, and do NOT rewrite mini-infer's PyTorch model code onto MLX arrays. The interop crux is the blocker: MLX is its own array framework, not a PyTorch/MPS backend, and the standard zero-copy bridge (DLPack on MPS) was absent in PyTorch until roughly mid-2025 and was only ever characterized by PyTorch maintainers as a "simple change" that was proposed and then deprioritized, not confirmed shipped. What IS warranted is a small, self-contained MLX correctness-and-feasibility track: stand up an MLX-native reference for the two kernels where Apple has a credible story (TurboQuant KV-cache dequant and quantized attention), use it to prove paper-faithful bit-behaviour on Metal, and treat any speed result as a secondary "does Apple Silicon hold up" data point rather than a headline benchmark. This fits mini-infer's stated niche (first-from-scratch + readable + bit-parity correct) without violating the "not a multi-hardware backend project" non-goal.

The reason this is "narrowly" and not "yes": every published custom-kernel-vs-built-in number we found shows the custom Metal quant-attention kernel *losing to MLX's own FP16 fast SDPA* on realistic grouped-query-attention decode shapes (0.52x-0.78x of FP16 speed). The custom kernel's win is memory capacity (fit longer context in unified memory), not latency. mini-infer already gets the latency story from CUDA; the Apple value is correctness coverage plus a long-context-on-a-laptop demo, not throughput.

---

## 1. Capability mechanics: what `mx.fast.metal_kernel` is

MLX (Apple's `ml-explore/mlx` array framework) exposes custom Metal kernels through `mx.fast.metal_kernel()`, available in both the Python and C++ APIs. The feature was authored by Alex Barron (PR #1325), merged 2024-08-22, shipped in MLX v0.17.0 on 2024-08-23, and announced the same day by MLX lead Awni Hannun. It is an official part of MLX, not a third-party add-on. The API surface is stable through the current docs (MLX 0.30 / 0.31.2, mid-2026) and was re-presented by Apple at WWDC 2025 session 315.

The defining ergonomic: **you write only the kernel body**, as a Metal-shading-language string passed in the `source` argument. MLX auto-generates the full `[[kernel]]` function signature from the declared `input_names` and `output_names` plus the shapes/dtypes of the arrays the kernel is called with. Metal thread attributes such as `[[thread_position_in_grid]]` are detected in the body and added to the signature automatically. The MLX reviewer who merged it called body-only authoring "an awesome decision that solves a bunch of problems" because the developer never hand-writes the prototype or buffer-binding boilerplate. Awni Hannun's framing: it "lets you write GPU kernels in Python which get JIT compiled into fast MLX ops." Caveat on that framing: you write the *compute logic* in the Metal shading language inside a Python string; this is not a pure-Python GPU DSL like Triton.

Construction vs invocation are split:

```python
kernel = mx.fast.metal_kernel(
    name="myexp",
    input_names=["inp"],
    output_names=["out"],
    source=source,          # Metal BODY only; signature auto-generated
    header="",              # optional: #includes / helper fns prepended verbatim
    ensure_row_contiguous=True,
    atomic_outputs=False,
)
out = kernel(
    inputs=[x],
    grid=(x.size, 1, 1),        # grid + threadgroup chosen at CALL time,
    threadgroup=(256, 1, 1),    # so one kernel object handles varying sizes
    output_shapes=[x.shape],
    output_dtypes=[x.dtype],
    template=[("T", mx.float32)],  # optional C++-template specialization
)[0]
```

`metal_kernel()` returns a Python-callable object. Specialization is via a `template` list of `(name, value)` tuples that map to C++ template parameters in the generated signature (the canonical example uses a dtype, e.g. `("T", mx.float32)`, but template values can also be int/bool compile-time constants such as a `bits` parameter for 4/8-bit packing).

JIT and caching: the Metal source is JIT-compiled at runtime and integrated as a first-class MLX op (it returns `mx.array` and participates in MLX's transforms such as autograd/vjp). The documented sharp edge is per-object overhead: "Every time you make a kernel, a new Metal library is created and possibly JIT compiled." The official guidance is to **build the kernel object once and call it many times** to amortize compilation. (OS-level Metal pipeline caching can persist across runs, which softens worst-case recompilation, but the MLX-side library creation is the cost the docs warn about.)

What you can express: arbitrary elementwise / reduction / gather-scatter / fused-attention-style compute over MLX arrays, with full control of grid and threadgroup geometry and threadgroup memory, plus template specialization. What you cannot do: avoid writing Metal (the body is Metal, not Python), hand-author the signature (it is generated for you), or reach below MLX's allocator into raw Metal command-buffer scheduling. And critically you cannot beat MLX's *built-in* fast ops on their own turf (see section 2).

How this differs from MLX's built-in fast ops: `mx.fast` already ships highly-tuned, highly-configurable implementations of the core transformer primitives, specifically `mx.fast.scaled_dot_product_attention`, `mx.fast.rms_norm` (and `layer_norm`), and `mx.fast.rope`. Apple's explicit guidance (WWDC 2025) is that a custom Metal kernel is warranted only "for cases where your function ... is not already in `mx.fast`." For mini-infer this is the decisive filter: RMSNorm, RoPE, and standard SDPA are already covered by tuned MLX ops, so there is no reason to port those; the candidates are the things `mx.fast` does NOT cover (quantized KV dequant, paper-specific primitives).

**Primary sources:** MLX custom-kernels dev guide (https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html); API reference `mlx.core.fast.metal_kernel` (https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.metal_kernel.html); MLX PR #1325 (https://github.com/ml-explore/mlx/pull/1325); WWDC 2025 session 315 (https://developer.apple.com/videos/play/wwdc2025/315/); Hannun announcement (https://x.com/awnihannun/status/1827087059431125004).

---

## 2. Performance: what the numbers actually say

Be skeptical here. The custom-kernel performance story on Apple Silicon is real but narrow, and the most-cited wins are memory-capacity wins, not latency wins.

The strongest concrete data comes from two open-source quantized-attention/KV projects that use `mx.fast.metal_kernel` exactly the way mini-infer would.

**`Thump604/mlx-qsdpa`** (fused quantized attention decode kernel, M2 Ultra, B=1, head_dim D=256, 4-bit group_size 32, median of 100 iters after warmup) gives the cleanest apples-to-apples three-way comparison: custom fused kernel vs composed `quantized_matmul` vs built-in FP16 SDPA.

On a grouped-query-attention (GQA) decode shape (H_q=32, H_kv=2), at 128K context: FP16 SDPA = 1478 us, composed `quantized_matmul` = 4161 us, fused custom kernel = 2431 us. So the custom kernel is 1.71x faster than the composed quant path but 0.61x the speed of FP16 SDPA (i.e. slower than the built-in fast op). Across all GQA context lengths tested (1K-128K) the fused kernel ran 0.52x-0.78x of FP16 speed: consistently slower than FP16 SDPA. The author's own framing: at shorter contexts "the primary value is not speed but memory reduction" (KV cache 24GB to 6GB, making 1M-context viable on 128GB hardware). On a non-GQA / multi-head shape the same kernel *did* beat FP16 (64K = 1.17x, 128K = 1.28x), which is exactly the point: **the win is conditional on attention shape.** Caveat: the benchmark uses head_dim 256, not the 128 that standard Llama/Qwen GQA models use, so generalization to production head dims is unverified.

**`sharpner/turboquant-mlx`** (TurboQuant KV-cache, the same technique mini-infer implements, Apple M4 Max 64GB, Llama 3.2 3B, T=8192) quantifies the speed-vs-memory tradeoff of KV quant without a fully fused custom attention kernel. Compression: up to 5.5x (969 MB fp16 KV down to 177 MB at 2.5-bit mixed). Throughput cost: fp16 = 148 tok/s; the "V2" 3-bit path that rides `mx.quantized_matmul` (a built-in Metal kernel) = 45 tok/s (~30% of fp16); the "V3" software-dequant path (centroid lookup + `mx.matmul`, the paper-correct Lloyd-Max codebook) = 27 tok/s (~18% of fp16). The repo states plainly that V3 runs ~5-6x slower than V2 "without custom Metal kernels," and that on A100 "with custom CUDA kernels, the paper avoids this penalty." This is the same launch-overhead problem mini-infer's own `turbo_kernel.py` docstring describes on CUDA (hundreds-to-thousands of kernel launches per decode step crushing throughput to 0.04-0.18x of bf16) and solves with a single fused Triton launch. A separate project (`arozanov/turboquant-mlx`) reports 4.6x compression at ~98% of FP16 speed using fully-fused zero-decompression Metal kernels, which validates the causal story: the penalty is launch overhead, and a fused custom Metal kernel is the fix.

Unified-memory implication: on Apple Silicon CPU and GPU share one physical memory pool, so the headline benefit is fitting much larger KV caches (and therefore much longer contexts) on a single laptop/desktop than the equivalent discrete-GPU VRAM would allow. This is a capacity story that plays to mini-infer's "long context on an M1" demo value, not a throughput story.

Net read: a custom Metal kernel reliably beats *composed MLX quant ops* (1.5x-1.71x), but on realistic GQA decode it loses to MLX's *built-in FP16 SDPA*. So "when does a custom kernel beat composed MLX ops?" = when the op is not in `mx.fast` and you would otherwise stitch it from primitives (quant dequant, packing). "When does it beat the built-in fast op?" = essentially never on its native path; the value there is memory, longer context, or non-GQA shapes.

**Primary sources:** https://github.com/Thump604/mlx-qsdpa; https://github.com/sharpner/turboquant-mlx.

---

## 3. Comparison and interop: the crux for a PyTorch engine

This is the section that drives the recommendation, because mini-infer's model code runs on PyTorch (CUDA + CPU/MPS), and **MLX is a separate array framework, not a PyTorch backend.** Using an MLX kernel from mini-infer means handing a tensor out of PyTorch into MLX and back on every call.

### (a) vs PyTorch MPS / torch custom Metal ops
PyTorch's MPS backend runs the existing mini-infer model code on Apple GPUs today (that is the current "MPS reference path"). It uses Apple's MPS/MPSGraph under the hood and requires no second framework. The cost: it is a reference/correctness path, not a tuned one, and PyTorch does not give you the body-only custom-Metal-kernel ergonomics MLX does (writing a torch custom Metal op means C++/Objective-C extension boilerplate, not a Python string). So the ergonomic advantage is genuinely MLX's. The catch is everything below.

### (b) vs raw Metal / MPSGraph / Metal Performance Shaders
Raw Metal or MPSGraph gives maximum control and zero framework-handoff cost but maximum boilerplate (Objective-C/Swift host code, manual pipeline state, command buffers). MLX's whole pitch is collapsing that into a Python string plus JIT. For a "readable, paper-next-to-code" project, MLX's body-only kernel is far more legible than raw Metal. But again this only matters if you are already in MLX-array land.

### (c) The interop cost (the actual blocker)
The standard zero-copy tensor-exchange protocol is DLPack. MLX added DLPack device support (`__dlpack_device__` on `mx.array`) in late May 2024 (issue #1159 opened 2024-05-27, closed 2024-05-31 via merged PR #1165, maintainer-confirmed). So MLX can *export* to DLPack. Two problems make a clean zero-copy PyTorch<->MLX handoff on Metal unreliable:

1. **MLX has no per-array device.** `mx.array.__dlpack_device__` is resolved *globally at conversion time*: it reports the Metal device if `metal::is_available()`, else CPU (verified in MLX source `python/src/array.cpp`, which accepts the array argument but never reads it). So the DLPack device type reflects runtime Metal availability, not where the array lives.

2. **PyTorch had no MPS DLPack support until ~mid-2025.** On PyTorch 2.7.0 (released 2025-04-23), calling `__dlpack_device__()` on an MPS tensor raised `ValueError: Unknown device type mps for Dlpack` (pytorch/pytorch issue #153789, filed 2025-05-18). PyTorch core maintainers (albanD, malfet) agreed to route MPS to the standard DLPack `kDLMetal` device (code 8) and called it "a pretty simple change," milestoned it for 2.8.0, then *removed the milestone* (2025-07-09); we found no clearly merged kDLMetal-for-MPS PR. So as of the evidence window, DLPack-level zero-copy MPS<->MLX was the intended path but not confirmed shipped.

The practical consequence: today the dependable handoff between a PyTorch/MPS engine and an MLX kernel is **not guaranteed zero-copy.** It is either (i) gated on a PyTorch version new enough to have MPS DLPack working end-to-end against MLX (uncertain, version-sensitive), or (ii) a copy through host memory / NumPy, which on unified-memory hardware is cheaper than on a discrete GPU but is still a per-call materialization that erodes the very memory-bandwidth and launch-overhead advantages you wrote the kernel to capture. Note: claims that MLX docs explicitly flag MLX-to-PyTorch as "experimental, route through NumPy" did NOT survive verification (split vote), so treat the *exact* recommended workaround as unconfirmed; what IS confirmed is that the zero-copy MPS path was missing/uncertain in the relevant window.

**Conclusion for mini-infer:** the interop tax means you do not want a hybrid "PyTorch model, MLX kernel mid-forward" architecture on the hot path. If you adopt MLX kernels seriously, the clean design is an **MLX-native island** (an alternate model path written in MLX arrays end-to-end for the layers you care about), not a per-op bridge from the existing PyTorch runner. That is a much larger commitment, which is why the recommendation is "narrowly."

**Primary sources:** https://github.com/ml-explore/mlx/issues/1159; MLX `python/src/array.cpp`; https://github.com/pytorch/pytorch/issues/153789; DLPack header `dmlc/dlpack` (kDLMetal=8).

---

## 4. Kernel-port feasibility for mini-infer specifically

Grounding in the actual repo: every performance kernel mini-infer has is CUDA-only with an explicit non-CUDA fallback, and `src/mini_infer/device.py` centralizes the device check (per the project's own "no scattered `device.type == cuda`" rule). The CUDA kernels are: fused INT8 W8A16 GEMM (`quant/int8_kernel.py`, Triton), TurboQuant V3 KV dequant (`cache/turbo_kernel.py`, Triton), `hc_split_sinkhorn` (`models/blocks/hc_sinkhorn_kernel.py`, Triton), and FlashInfer paged attention (`cache/flashinfer_backend.py`). LightningIndexer and AttentionSink live as model blocks under `models/blocks/v4/`. On Apple there is no FlashInfer and no Triton-on-Metal, so each would need an MLX path or must stay on the composed-op fallback.

Ranked by feasibility / worth:

| Primitive | mini-infer today | Port to Metal via MLX? | Verdict |
|---|---|---|---|
| **TurboQuant KV-cache dequant** | `turbo_kernel.py` fused Triton, CUDA-only | Yes. `sharpner/turboquant-mlx` already implements quantize / sign-pack / QJL-scoring as `mx.fast.metal_kernel`; `mlx-qsdpa` fuses 4/8-bit KV dequant inline in one Metal dispatch. | **Best first candidate.** Highest-value, proven portable, exercises the full API surface, and it is the one place a custom Metal kernel clearly beats the composed-op fallback. |
| **Quantized attention (KV dequant fused into SDPA)** | composed via FlashInfer (CUDA) | Yes, demonstrated by `mlx-qsdpa` (online softmax + inline dequant in a single kernel). | **Second candidate**, naturally bundled with TurboQuant. Caveat: loses to FP16 SDPA on GQA decode; value is memory/long-context, not latency. |
| **Attention (general)** | torch SDPA / FlashInfer | Use `mx.fast.scaled_dot_product_attention` (built-in). | **Do not port.** Already in `mx.fast`, and custom kernels lose to it. |
| **RMSNorm / RoPE** | torch / model blocks | Use `mx.fast.rms_norm` / `mx.fast.rope`. | **Do not port.** Already tuned in `mx.fast`. |
| **INT8 W8A16 dequant GEMM** | `int8_kernel.py` Triton, CUDA-only | Possible, but MLX's built-in `mx.quantized_matmul` already covers the common quantized-GEMM case (affine quant) with a tuned Metal kernel; a bespoke W8A16 kernel would have to beat it, and the `turboquant-mlx` data shows the built-in quant matmul path is the *fast* one. | **Low priority.** Prefer composing on `mx.quantized_matmul`; only write custom if mini-infer's exact W8A16 contract diverges from MLX's affine scheme. |
| **`hc_split_sinkhorn`** (DeepSeek-V4) | `hc_sinkhorn_kernel.py` Triton, CUDA-only | Mechanically yes (small per-row iterated matrix, identical fusion pattern: keep `(hc,hc)` in registers, no HBM round-trip). No existing MLX implementation found. | **Feasible but niche.** Worth it only if you want V4 to run end-to-end on Metal at speed; for correctness the pure-PyTorch/MPS fallback already exists. |
| **LightningIndexer** | `models/blocks/v4/lightning_indexer.py` | Mechanically yes; no published MLX port found. | **Defer.** Port only after the V4 model path is proven on MLX. |
| **AttentionSink** | `models/blocks/v4/sink.py` | Yes; it is a softmax-renormalization tweak, expressible as a custom kernel or composed ops. | **Defer / compose.** Low compute; composed MLX ops likely fine. |

The pattern: port exactly the things `mx.fast` does NOT give you and where a fused kernel beats composing primitives, which is the quantized-KV family. Everything covered by `mx.fast` (SDPA, RMSNorm, RoPE) should ride the built-ins. The DeepSeek-V4 paper-specific primitives are portable but only worth the effort if Apple Silicon becomes a real V4 target, which is a bigger decision than this report recommends taking now.

**Effort estimate (rough):** an MLX-native TurboQuant KV + quantized-attention island, validated for bit-behaviour against the existing CUDA/CPU reference, is on the order of 1-2 focused weeks given that two reference implementations already exist to learn from. A full DeepSeek-V4-on-MLX path (Sinkhorn + LightningIndexer + Sink + MLA, all in MLX arrays) is materially larger (multiple weeks) and is not recommended at this stage.

**Repo sources:** `src/mini_infer/device.py`; `src/mini_infer/quant/int8_kernel.py`; `src/mini_infer/cache/turbo_kernel.py`; `src/mini_infer/models/blocks/hc_sinkhorn_kernel.py`; `src/mini_infer/cache/flashinfer_backend.py`; `src/mini_infer/models/blocks/v4/lightning_indexer.py`; `src/mini_infer/models/blocks/v4/sink.py`. **External:** `sharpner/turboquant-mlx`; `Thump604/mlx-qsdpa`.

---

## 5. Maturity and limitations

- **API stability:** good. Shipped Aug 2024 (v0.17.0), unchanged in shape through 0.31.2 (mid-2026), re-blessed at WWDC 2025. Low churn risk.
- **Body-only ergonomics:** the standout strength; legible, Python-driven, JIT-compiled.
- **Low-precision support on Metal:** MLX has first-class affine quantization (`mx.quantize` / `mx.quantized_matmul`) at 4/8-bit (group sizes 32/64/128), and the quantized-attention projects exercise 4-bit and 8-bit KV. fp8/fp4/int4 as *distinct hardware-accelerated formats* are not established on Metal the way FlashInfer FP8/NVFP4 are on CUDA; treat sub-4-bit and FP8/FP4 on Apple as "compose it yourself / not hardware-accelerated" rather than turnkey. (This report did not find primary confirmation of native FP8/FP4 Metal kernel paths, so flag as an open gap, see open questions.)
- **Debugging / profiling:** kernels are standard Metal, so Xcode Metal capture and Instruments apply in principle, but this is a thinner story than CUDA's Nsight Systems / Nsight Compute / `torch.profiler` stack that mini-infer's CLAUDE.md already standardizes on. Expect a tooling step-down vs CUDA.
- **Known sharp edges (verified):**
  - Build-once-call-many is mandatory; constructing a kernel per call pays JIT/library-creation overhead each time.
  - Throughput degradation when calling the same kernel a very large number of times (MLX issue #1828, >10000 calls).
  - **Silicon-specific correctness divergence:** MLX issue #2205 reports `mx.fast.metal_kernel` returning *incorrect results on M1 Max but correct on M3 Max*. This directly matters because mini-infer's daily dev machine is an M1: a custom kernel that passes on a newer chip could silently fail bit-parity on the M1 you develop on. Any MLX kernel must be golden-tested on the actual target silicon, not assumed portable across the M-series.
  - Docs gaps noted historically (MLX issue #1547).
- **Interop maturity:** the weakest link, per section 3. DLPack export from MLX is mature; zero-copy import into PyTorch-on-MPS is version-sensitive and was not confirmed shipped in the evidence window.

---

## Phased "how" (only if narrow adoption proceeds)

Framed to respect the non-goals: this is a correctness-reference + capacity-demo track, not a new production backend, and it lives as an MLX-native island, not a per-op bridge into the PyTorch hot path.

- **Phase A (spike, ~days):** stand up `mx.fast.metal_kernel` locally on the M1. Port the TurboQuant *dequant* primitive to a single fused Metal kernel (mirror `turbo_kernel.py`'s "one launch per layer" design). Validate output bit-behaviour against the existing CPU reference for the *same* quantized inputs. **Gate: it must pass on the M1 specifically** (issue #2205 risk).
- **Phase B (KV + attention island, ~1-2 weeks):** assemble an MLX-native quantized-KV + quantized-attention path (learn from `mlx-qsdpa` for the fused inline-dequant attention, `sharpner/turboquant-mlx` for the rotation/codebook/QJL). Keep it behind an `attention_backend="mlx"` selector analogous to the existing `flashinfer` selector, isolated from the PyTorch runner (no mid-forward tensor bridging). Add golden tests at temperature=0 against HuggingFace, run on Apple Silicon.
- **Phase C (measure, secondary):** benchmark *memory capacity* first (longest context that fits on the dev M1 / a larger M-series), report throughput as a secondary "Apple Silicon holds up" figure, and be explicit that GQA decode latency trails FP16 SDPA. Do not present Apple numbers as competitive with the CUDA path; that is not the claim.
- **Phase D (defer):** DeepSeek-V4 primitives (`hc_split_sinkhorn`, LightningIndexer, AttentionSink) on MLX, only if a full V4-on-Apple target is later greenlit. Reassess after Phase C; likely not worth it under the current budget and non-goals.

Explicitly out of scope: rewriting the PyTorch model code onto MLX wholesale; competing with the CUDA throughput path; FP8/FP4 on Metal until native support is confirmed.

---

## Caveats and time-sensitivity

- **Time-sensitive interop:** the PyTorch-MPS-DLPack situation was actively moving in mid-2025 and was not confirmed shipped in the evidence window. Re-check current PyTorch (>=2.9) before committing to any DLPack-based handoff; if it now works end-to-end against MLX on Metal, the interop tax in section 3 shrinks.
- **Benchmark provenance:** the strongest performance numbers come from small (72-119 star) single-author proof-of-concept repos (`turboquant-mlx`, `mlx-qsdpa`), not vendor benchmarks. They are transparent about methodology and notably candid about *unfavourable* results (custom kernel slower than FP16), which raises trust, but they are not independently reproduced at scale. The M2 Ultra qsdpa numbers use head_dim 256, not the 128 of standard GQA models.
- **Refuted-but-relevant:** the specific claim that MLX docs flag MLX-to-PyTorch as experimental and tell you to route through NumPy did not survive verification (split vote). Do not cite that as the official workaround; cite only the confirmed gap (MPS DLPack absent/uncertain in the window).
- **FP8/FP4 on Metal:** not positively confirmed in primary sources here; treated as a gap, not a capability.
