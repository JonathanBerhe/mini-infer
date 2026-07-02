# MiniMax-M3 (MSA) text-only port: implementable spec

> Phase-0 research synthesis (2026-06-26). The build reference for Phases 1-5.
> Authoritative source: HF `transformers` `models/minimax_m3_vl/` (upstreamed,
> NOT trust_remote_code). Where the arXiv paper (2606.13392v1) or the
> `MiniMax-AI/MSA` kernel repo disagree with the HF code, the HF code wins
> (flagged `[PAPER-CONTRADICTION]` / `[KERNEL-CONTRADICTION]`).

Config name mapping (deployment `config.json` -> transformers `MiniMaxM3VLTextConfig`):
`sparse_block_size`->`index_block_size`=128, `sparse_topk_blocks`->`index_topk_blocks`=16,
`sparse_num_index_heads`->`index_n_heads`=4, `sparse_index_dim`->`index_head_dim`=128,
`sparse_local_block`->`index_local_blocks`=1, `sparse_score_type=max`->`amax` pool.
Layer routing auto-derived in `__post_init__` from the `[0,0,0,1,...]` freqs:
`layer_types` -> layers 0-2 `full_attention`, 3-59 `minimax_m3_sparse`;
`mlp_layer_types` -> layers 0-2 `dense`, 3-59 `sparse` (MoE).

## Corrections (verified against HF `minimax_m3_vl` source during Phase 2)

Two Phase-0 synthesis claims were wrong; the HF source is authoritative:

1. **Block selection is GLOBAL, not per-GQA-group.** `build_block_mask`'s input
   comes from `block_scores = scores.view(...).amax(-1).amax(1)` which max-pools
   over the block tokens AND the index heads, giving one selected-block set per
   `(batch, query)`, shared across ALL main attention heads. The additive mask is
   `[B, 1, S, k_len]` (broadcast over heads). Ignore the earlier "4 independent
   per-group selections / repeat_interleave 16" language in §1b/§1c/§1e-item-9.
2. **RoPE is FULL over head_dim, not partial** (resolved by the Phase-3
   model-level parity harness, superseding the earlier partial-RoPE reading).
   The HF config carries a `rotary_dim` field but it is NOT wired into the rope:
   under `rope_type="default"` HF builds `inv_freq` of length `head_dim/2` (all
   non-zero) and `apply_rotary_pos_emb` rotates the whole 128-dim head. Use
   `RotaryEmbedding(head_dim=128, base=5e6)` (width-128 table) + the standard
   `apply_rotary_pos_emb`, for the main branch AND the indexer. Resolves
   open-question #1. The block mask also REPLACES the causal mask (folds causal
   in), it is not added on top of a separate causal mask.

## 1. MSA attention

Same module class for dense and sparse layers (`MiniMaxM3VLAttention`); sparse
layers additionally build an indexer. GQA 64q/4kv (G=16), head_dim 128,
`scaling=128**-0.5`, `attention_output_gate=false` (output is plain `o_proj`).

### 1a. Main branch (per layer)
```
q = q_proj(x).view(B,S,64,128); k = k_proj(x).view(B,S,4,128); v = v_proj(x).view(B,S,4,128)
q = q_norm(q); k = k_norm(k)            # per-head Gemma RMSNorm(128), pre-RoPE, pre-transpose; V NOT normed
q,k = transpose(1,2)
q,k = apply_rotary_pos_emb(q,k,cos,sin) # full RoPE over all 128 dims (rotary_dim is not wired)
k,v = kv_cache.update(k,v)              # full KV, no pruning
# sparse layers: block_indices = indexer(...); attention_mask += build_block_mask(block_indices)
k,v = repeat_kv(k,v,16)                 # 4 -> 64
scores = (q @ kᵀ) * 128**-0.5           # bf16 matmul
scores += attention_mask                # additive block-sparse + causal
p = softmax(scores, dim=-1, dtype=fp32).to(bf16)
o = p @ v ; out = o_proj(o)
```

### 1b. Index branch (`MiniMaxM3VLIndexer`)
`q_proj` 6144->512 (4 heads x128), `k_proj` 6144->128 (single shared key head),
`q_norm`/`k_norm` = Gemma RMSNorm(128). No value/output proj (pure selection).
4 index heads = one per GQA group.
```
idx_q = q_norm(q_proj(x).view(B,S,4,128)).transpose(1,2)   # norm BEFORE transpose/RoPE
idx_k = k_norm(k_proj(x).view(B,S,1,128)).transpose(1,2)
idx_q,idx_k = apply_rotary_pos_emb(idx_q,idx_k,cos,sin)     # full RoPE, same tables as main branch
idx_k = cache.update_index(idx_k)                          # full idx_k history cached separately
scores = idx_q.float() @ idx_k.float().transpose(-1,-2)    # FP32, RAW dot, NO /sqrt(d)
scores = scores.masked_fill(arange(k_len) > position_ids[...,None], -inf)  # causal, TOKEN granularity, BEFORE pool
scores = pad(scores, last block to 128, value=-inf)
block_scores = scores.view(B,4,S,num_blocks,128).amax(-1)  # MAX-pool
block_scores.scatter_(-1, (q_block - arange(local_blocks)).clamp(min=0), +inf)  # force local block (=current)
topk = min(16, num_blocks)
topk_scores, topk_indices = block_scores.topk(topk, -1)
block_indices = topk_indices.masked_fill(topk_scores == -inf, -1)   # [B,4,S,<=16]
```

### 1c. Selection -> dense additive mask (`build_block_mask`, the parity target)
`[KERNEL-CONTRADICTION]` The HF eager/sdpa reference does NOT gather selected K/V.
It expands the block selection into a dense `[B,64,S,Sk]` additive float mask and
runs ordinary dense attention (the `block_indices` kwarg is ignored in the attn
fn). Reimplement this dense-mask path for bit-parity:
- `-1` -> throwaway column; scatter `0.0` at selected blocks else `finfo.min`; drop throwaway.
- selected block keeps all 128 key slots (`repeat_interleave(128)`), then causal token mask re-applied.
- block selection broadcast 4 index heads -> 64 query heads (`repeat_interleave(16, dim=1)`): all 16 query heads in a group share their group's selection; 4 independent selections.

### 1d. Prefill vs decode
Full main KV + full `idx_k` cached (separate stream). Decode re-scores ALL cached
blocks every step (no incremental block-score state); top-16 recomputed fresh.
Savings are in the main branch (<=16 blocks attended), not the index branch
(O(context)/step). Partial last block: right-pad with -inf, token causal before pool.

### 1e. Bit-parity-critical details (do not deviate)
1. Index score has NO `/sqrt(d_idx)` scale `[PAPER-CONTRADICTION]` (raw fp32 dot).
2. Indexer applies full RoPE AND per-head qk-norm BEFORE RoPE `[PAPER-CONTRADICTION]`.
3. RMSNorm = `out*(1.0+weight)`, fp32, cast back AFTER multiply (Gemma order).
4. Index scores fp32; main softmax fp32 then cast bf16; matmuls bf16 at scale `128**-0.5`.
5. Causal mask at TOKEN granularity BEFORE block max-pool (`k_pos > q_pos`, strict).
6. Block pool = `amax`; right-pad partial last block with -inf.
7. Force local block via `scatter_(+inf)` on `q_block - arange(local_blocks)` clamp>=0; `index_local_blocks=1` = current block; NO init block (no HF hook).
8. `topk = min(16, num_blocks)`; -inf-scoring selected slots -> -1. Short ctx (<16 blocks) -> attend everything visible (verify).
9. Selection shared across the 16 query heads per group; 4 independent selections.
10. Selected block: all 128 slots attendable, then re-masked causally.
11. Masked scores use `torch.finfo(dtype).min`, not `-inf`.
12. Blocks slot-anchored (`block b` = key slots `[128b, 128b+128)`). Only RIGHT-padding is bit-equivalent; left-padding shifts boundaries (HF skips `test_left_padding_compatibility`).
13. Parity target is the dense-mask reference, not a gather.

## 2. Model assembly

### 2a. Layer stack (60 layers, pre-norm)
```
h = h + self_attn(input_layernorm(h))          # Gemma RMSNorm(6144)
h = h + mlp(post_attention_layernorm(h))        # dense MLP (L0-2) OR MoE (L3-59)
# final model.norm before lm_head; no pre-MoE/post-MLP norm
```
| Layers | Attention | FFN |
|---|---|---|
| 0-2 | dense full-attn (`indexer=None`) | dense MLP, intermediate 12288 |
| 3-59 | MSA (indexer present) | MoE: 128 experts @3072 + 1 shared @3072 |

### 2b. swigluoai activation (alpha=1.702, limit=7.0)
Used by dense MLP and every expert (routed + shared). `gate, up = chunk(gate_up, 2, -1)`
(gate first, up second; contiguous blocks, NOT interleaved):
```
swigluoai(gate, up) = (clamp(up, -7, 7) + 1) * clamp(gate, max=7) * sigmoid(1.702 * clamp(gate, max=7))
```
gate clamp is `max` only; up clamp symmetric; `+1` on the up branch; `1.702` only in the sigmoid arg.

### 2c. MoE gate (`MiniMaxM3VLTopKRouter` / `...SparseMoeBlock`)
128 experts, top-4, shared 1, `routed_scaling_factor`=2.0, sigmoid scoring, routing bias.
```
router_logits = linear(x, gate.weight)                     # no bias
w = sigmoid(router_logits.float())                          # FP32
top_k_index = topk(w + e_score_correction_bias, 4).indices  # SELECTION uses biased scores
top_k_w = w.gather(1, top_k_index)                          # WEIGHTS from UNBIASED sigmoid
top_k_w /= top_k_w.sum(-1, keepdim=True)                     # renormalize (unconditional)
out = 2.0 * Σ_{e in top4} top_k_w_e * swigluoai_e(x)  +  swigluoai_shared(x)
```
`gate.weight` and `e_score_correction_bias` are FP32 (bias is a buffer). DeepSeek-V3
aux-loss-free bias correction with `n_group=topk_group=1`.

### 2d. Gemma RMSNorm (`use_gemma_norm=true`, eps 1e-6)
`out = (x.float() * rsqrt(mean(x.float()^2,-1,keepdim)+eps)) * (1.0 + weight.float())`, cast to x.dtype.
All seven sites: `input_layernorm`, `post_attention_layernorm`, `model.norm` (dim 6144);
`q_norm`, `k_norm`, `index_q_norm`, `index_k_norm` (dim 128, per-head, applied on `[B,S,H,128]` BEFORE transpose/RoPE).

### 2e. RoPE / embeddings / LM head
- Full RoPE: `dim=head_dim=128`; `inv_freq=1/(5e6**(arange(0,128,2)/128))`; cos/sin width 128.
  The config's `rotary_dim` field is inert under `rope_type="default"` (verified: HF inv_freq
  spans head_dim/2 non-zero freqs; full-width rotation matches bit-for-bit, first-64 partial diverges).
  `rotate_half(x)=cat([-x[...,d/2:], x[...,:d/2]])` = NeoX/non-interleaved (NOT DeepSeek interleaved). cos/sin fp32 -> bf16.
- Embeddings: `nn.Embedding(200064, 6144)`, no embed scaling (unlike Gemma).
- Final `model.norm(6144)` -> `lm_head Linear(6144, 200064, bias=False)`, UNTIED (load `lm_head.weight` separately). No logit softcap.
- MTP: `num_mtp_modules=7` weights not in checkpoint (`_keys_to_ignore_on_load_unexpected=[r"(^|\.)mtp\..*"]`); skip.

### 2f. On-disk weight names (verified from `model.safetensors.index.json`)
All text weights under `language_model.` prefix; drop `vision_tower.*`,
`multi_modal_projector.*`, `patch_merge_mlp.*`, `mtp.*`. All `*_proj.weight` are
`[out, in]` (used via `F.linear`, no transpose on load). bf16 except router
`gate.weight` + `e_score_correction_bias` (fp32).
```
language_model.model.embed_tokens.weight                  [200064,6144]
language_model.model.norm.weight                          [6144]
language_model.lm_head.weight                             [200064,6144]  (untied)
# per layer L 0..59 (attention)
...layers.L.input_layernorm.weight                        [6144]
...layers.L.post_attention_layernorm.weight               [6144]
...layers.L.self_attn.{q_proj[8192,6144], k_proj[512,6144], v_proj[512,6144], o_proj[6144,8192]}
...layers.L.self_attn.{q_norm[128], k_norm[128]}
# sparse-attn layers L 3..59 (indexer)
...layers.L.self_attn.{index_q_proj[512,6144], index_k_proj[128,6144], index_q_norm[128], index_k_norm[128]}
# dense MLP layers L 0,1,2
...layers.L.mlp.{gate_proj[12288,6144], up_proj[12288,6144], down_proj[6144,12288]}
# MoE layers L 3..59
...layers.L.block_sparse_moe.gate.weight                  [128,6144]  FP32
...layers.L.block_sparse_moe.e_score_correction_bias      [128]       FP32
...layers.L.block_sparse_moe.experts.E.{w1[3072,6144]=GATE, w3[3072,6144]=UP, w2[6144,3072]=DOWN}  (E 0..127)
...layers.L.block_sparse_moe.shared_experts.{gate_proj[3072,6144], up_proj[3072,6144], down_proj[6144,3072]}
```
Expert mapping (load-bearing): w1=gate, w3=up, w2=down. If fusing gate+up: concat `[w1;w3]` (contiguous, not interleaved) and `chunk(2)`.

## 3. MSA kernel + PyTorch-oracle parity target
The efficient `MiniMax-AI/MSA` kernel is NOT upstream (default eager path is
dense-mask + SDPA). Parity target for the from-scratch port = the HF dense-mask
path (1c), not the kernel. Kernel study informs an optional later Triton port
(Phase 5); it does not gate correctness.
- Reference kernel pipeline (for the future Triton port): indexer/scoring -> exp-free top-k (histogram/radix, NOT the paper's register heap `[PAPER-CONTRADICTION]`) -> CSR build (prefill) -> KV-outer main attention -> split-K LSE combine (inline online-softmax, NOT the paper's two-phase O_buf `[KERNEL-CONTRADICTION]`). head_dim=128 only; bf16/fp8; topK in {4,8,16,32}; decode requires Hq/Hkv=16 (M3's G).
- PyTorch oracle = the dense-then-mask formulation (numerically identical to gather, and it IS the HF path). Validate index/router selections as EXACT int sets; output cosine > 0.999 (bf16) / allclose fp32; greedy token must match at temp 0.

## 4. mini-infer reuse map
| Component | Verdict | Existing | Notes |
|---|---|---|---|
| GQA q/k/v/o + qk-norm hooks | reuse | `models/blocks/gqa.py:37` (norm hooks :82-86,:118-122) | 64q/4kv/128 are config |
| Paged KV cache (full KV) | reuse | `cache/paged_kv_cache.py:14`, `cache/block_pool.py:83` | PagedKVCache, NOT StateCache |
| Packed varlen attention | extend | `cache/packed_attention.py:190` `_torch` | inject block-sparse mask; mirror `cache/mla_attention.py:86-94` `dsa_topk` |
| Full RoPE (128, theta 5e6) | reuse | `models/blocks/rope.py` `RotaryEmbedding` + `apply_rotary_pos_emb` | verified by parity: NeoX pairing, `rotary_dim` inert |
| Per-head qk-norm (q AND k) | reuse | `models/blocks/transformer_block.py:42-43`; `models/qwen3.py:96` | must be Gemma `(1+w)` norm |
| Gemma RMSNorm `(1+w)` | reuse | `models/blocks/gemma_rmsnorm.py:12` | for ALL norms incl. 4 per-head qk/index norms |
| SwiGLU FFN shape | extend | `models/blocks/swiglu.py:23` | shape reusable, activation differs |
| swigluoai activation | BUILD NEW | none | clamped GLU; dense FFN + all experts |
| Sigmoid MoE gate+bias+shared+scaling | reuse | `models/blocks/glm_moe_gate.py:37` `GlmNoAuxTcGate` + `:107` `GlmMoeFFN` | GLM is the base, NOT Mixtral. Only change: experts use swigluoai. n_group=topk_group=1, 128 exp, top4, shared1, scale2.0 |
| Block-level top-k indexer | BUILD NEW (heaviest) | template `v4/lightning_indexer.py:61`; wiring `glm_dsa_indexer.py:160`; mask `mla_attention.py:86-94` | see hardest point |
| Model registry + `...ForCausalLM` | reuse | `models/__init__.py:63`,:136; `base.py:40` | write `MiniMaxM3ForCausalLM` |
| StateCache routing | confirm OFF | `models/__init__.py:99`; `base.py:60` | leave `USES_STATE_CACHE=False` (like GLM-MoE-DSA) |
| ModelRunner | reuse | `engine/model_runner.py:66`,:107 | set `required_attention_backend()="torch"` (additive mask needs materialized SDPA; like GLM-DSA/V2) |
| ContinuousScheduler / prefix cache / TP | reuse | `scheduler/continuous_scheduler.py`; `cache/prefix_cache.py`; `distributed/*` | full KV => unchanged; indexer recomputed per-query so prefix sharing is safe |

### Hardest integration point
Block-level top-k selection wired to PagedKVCache, expressed as a per-query
additive mask. (1) Two "block" concepts collide: KV-cache page size vs MSA
`index_block_size=128` (scoring block) — score against materialized packed K
(`paged_kv_cache.py:256`/`:597`), reduce token scores -> block scores -> per-token
mask; do not conflate. (2) Indexer is new but `GlmDsaIndexer` is the near-template
(scores tokens over a paged stream) plus a max-pool-into-blocks step + block-level
causality. (3) Result is an additive block mask (reuse `mla_attention.py:86-94`),
scatter at block granularity, pinned to the `torch` backend (FA fast paths can't
take an arbitrary per-query mask).

## 5. Parity strategy + pins
- Reference = upstreamed transformers `minimax_m3_vl` (the HF repo `auto_map` only maps `AutoConfig`; the model resolves from installed transformers).
- Pin HF model repo `MiniMaxAI/MiniMax-M3` @ SHA `bfd6c97f0296da547f10ecb20102c5d51a5c462e` (main HEAD 2026-06-23), `revision=` on every `from_pretrained`.
- Pin `transformers==5.12.1` as a TEST-ONLY dep (`transformers>=5.12,<5.13`); do not perturb the engine runtime pin. (Docs anchor v5.12.0; re-verify the `minimax_m3_vl` file diff 5.12.0 vs 5.12.1 and pin whichever you validate against.)
- Pure-PyTorch eager reference EXISTS (kernel not mandatory). Use `attn_implementation="eager"` + fp32 on CPU. The whole text forward runs on CPU.
- Tiny-random text-only CPU instantiation works (`MiniMaxM3VLForCausalLM` text path takes only input_ids/attention/positions/past). Build a small config (3 dense + 2 MoE/MSA layers, tiny `index_block_size` so a short seq spans >topk+local blocks), export its `state_dict`, load identical weights into the from-scratch model (isolates math from loading), compare per-layer top-down.
- Per-layer parity test (first divergence = root cause): embedding -> input norm -> q/k/v + qk-norm -> full RoPE -> indexer block-scores + indices (EXACT int) + bias -> attn out -> swigluoai MLP -> MoE (router weights/indices EXACT int + experts + shared + combined) -> residual + final norm + logits (argmax). Gate `allclose(rtol=1e-4, atol=1e-5)` fp32; exact-int for router/indexer selections.
- Real-weights golden (428B, multi-GPU, temp 0 vs `MiniMaxM3SparseForConditionalGeneration`) is the deferred GPU gate; the tiny-random per-layer harness is the CI gate.

## 6. Open questions (resolve during implementation)
1. RESOLVED (Phase 3): RoPE is full-width over head_dim with NeoX half-rotation; `rope.py` `RotaryEmbedding` + `apply_rotary_pos_emb` match HF bit-for-bit (model parity harness green).
2. transformers 5.12.0 vs 5.12.1 `minimax_m3_vl` file diff — pin whichever you validate against.
3. Verbatim-code drift — diff the spec's code blocks against a local `pip install transformers==5.12.1` checkout before freezing fixtures.
4. Indexer per-head norm — confirm `MiniMaxM3VLIndexer.__init__` uses `MiniMaxM3VLRMSNorm` (Gemma 1+w), not LayerNorm; cover index_q_norm/index_k_norm in the harness.
5. Index-K cache stream — confirm whether to add a third PagedKVCache stream (`["k","v","idx_k"]`) or recompute idx_k from cached hidden (HF caches idx_k separately, favoring a dedicated stream).
