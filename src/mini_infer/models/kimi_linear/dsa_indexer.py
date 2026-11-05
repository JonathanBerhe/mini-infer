"""DeepSeek Sparse Attention indexer of Kimi K3.1 Pro (its "Gated DSA" layers).

Kimi K3.1 Pro keeps the text stack of Kimi K3 and turns its 24 Gated MLA
layers into Gated DSA layers. Before the attention, a small indexer picks
for each query the `index_topk` (2048) earlier tokens that the attention can
read; the attention masks every other token out of its softmax. The MLA
output gate (`g_proj`) stays.

The checkpoint ships no modeling code. The math is the DeepSeek-V3.2
indexer with the K3.1 settings that SGLang's K3.1 support pins
(sgl-project/sglang#48415, `KimiLinearConfig._init_dsa`):

    q_t,h     = wq_b(q_lora_t)                   # q_lora: the MLA's q_a_layernorm output
    k_j       = LayerNorm_fp32(wk(x_j))          # one key per token, eps = rms_norm_eps
    w_t,h     = weights_proj(x_t) * n_heads^-0.5 # rounded to the model dtype
    score_t,j = sum_h w_t,h * ReLU(q_t,h . k_j * head_dim^-0.5)
    keep_t    = top index_topk of score_t,j over the keys j <= t

- No RoPE (`index_rope_dim` 0): the whole family is NoPE.
- No Hadamard rotation: SGLang turns it off for K3.1. The rotation is
  orthogonal, so it does not change a q.k product in exact arithmetic.
- `index_topk_freq` 1: every DSA layer runs its own indexer, and the
  checkpoint has indexer weights on all 24 of them. vLLM's IndexShare rule
  skips a layer when `max(layer - offset + 1, 0) % freq != 0`, which is
  never true for a frequency of 1, so `index_skip_topk_offset` changes
  nothing.

The serving engines quantize the indexer to FP8 in their kernels. This
module scores in fp32, so its selection can differ from theirs only where
two scores tie at the top-k boundary.

A query with at most `index_topk` earlier tokens keeps all of them. So while
a request holds at most `index_topk` tokens, the layer equals the dense K3
MLA layer, and the attention does not run the scoring at all.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional


class KimiDsaIndexer(nn.Module):
    """Raw-token top-k selector of one Kimi K3.1 DSA layer.

    The parameter names match the checkpoint:
    `self_attn.indexer.{wq_b,wk,weights_proj}.weight` and
    `self_attn.indexer.k_norm.{weight,bias}`.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        q_lora_rank: int,
        num_heads: int,
        head_dim: int,
        index_topk: int,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.index_topk = index_topk
        self.wq_b = nn.Linear(q_lora_rank, num_heads * head_dim, bias=False)
        self.wk = nn.Linear(hidden_size, head_dim, bias=False)
        self.k_norm = nn.LayerNorm(head_dim, eps=norm_eps)
        self.weights_proj = nn.Linear(hidden_size, num_heads, bias=False)
        self.softmax_scale = head_dim**-0.5

    def keys(self, x: torch.Tensor) -> torch.Tensor:
        """The indexer keys of new tokens: `(B, T, hidden)` -> `(B, T, head_dim)`.

        The LayerNorm runs in fp32 (SGLang's `index_k_norm_fp32`), and the
        keys come back in `x.dtype`, the dtype the state cache stores.
        """
        k = functional.layer_norm(
            self.wk(x).float(),
            (self.head_dim,),
            self.k_norm.weight.float(),
            self.k_norm.bias.float(),
            self.k_norm.eps,
        )
        return k.to(x.dtype)

    def select(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        keys: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """The keys that each query keeps.

        Args:
            x: `(B, T, hidden)` new tokens.
            q_lora: `(B, T, q_lora_rank)`, the MLA's normalized query latent.
            keys: `(B, K, head_dim)` indexer keys of every token so far, this
                step's tokens included.
            valid: `(B, T, K)` causal mask: key `j` exists for query `t`.

        Returns:
            `(B, T, K)` bool: valid and among the `index_topk` best scores of
            its query. The caller runs this only when `K > index_topk`.
        """
        batch, seq_len, _ = x.shape
        q = self.wq_b(q_lora).view(batch, seq_len, self.num_heads, self.head_dim).float()
        # The head gate is rounded to the model dtype, then scored in fp32
        # (SGLang's `index_bf16_head_gate` for K3.1).
        weights = (self.weights_proj(x) * self.num_heads**-0.5).float()
        per_head = functional.relu(
            torch.einsum("bqhd,bkd->bqhk", q, keys.float()) * self.softmax_scale
        )
        score = torch.einsum("bqhk,bqh->bqk", per_head, weights)
        score = score.masked_fill(~valid, float("-inf"))
        top = score.topk(self.index_topk, dim=-1).indices
        keep = torch.zeros_like(valid).scatter_(-1, top, True)
        return keep & valid
