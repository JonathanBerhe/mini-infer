"""Qwen3 DSpark drafter: the semi-autoregressive block proposer.

One backbone forward produces `gamma` positions' base logits in a single,
bidirectional pass (KV-injected with the target's own hidden states as
context); `VanillaMarkovHead.sample_block_tokens` then samples the block
sequentially, correcting each position's logits with a bias conditioned on
the previously SAMPLED token so the block's final distribution is exactly
causal (see `markov_head.py`). Mechanics, citations, and the alternatives
considered live in `docs/decisions/ADR-027-dspark-drafter-port.md`.

`Qwen3DSparkConfig.from_hf` reads the released drafter's `config.json`
(`deepseek-ai/dspark_qwen3_4b_block7` and friends) and `load_weights` takes
its safetensors state dict; every tensor name is an identity rename onto
this module hierarchy, so no key remapping table is needed. The real
checkpoint only runs under GPU (16 GB unified memory can't comfortably hold
it alongside the Qwen3-4B target), so the CPU tests here exercise
micro-configs with random weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from mini_infer.engine.dspark.attention import DSparkAttention
from mini_infer.engine.dspark.confidence_head import ConfidenceHead
from mini_infer.engine.dspark.draft_cache import DSparkDraftCache
from mini_infer.engine.dspark.markov_head import VanillaMarkovHead
from mini_infer.engine.dspark.sampling import sample_tokens
from mini_infer.models.blocks.rmsnorm import RMSNorm
from mini_infer.models.blocks.rope import RotaryEmbedding
from mini_infer.models.blocks.swiglu import SwiGLU


@dataclass
class Qwen3DSparkConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    # Which TARGET decoder layers' post-block hidden states get injected as
    # context, in order. `deepspec` also allows `-1` for the target's
    # embedding output; not used by the released Qwen3 drafters, not ported.
    target_layer_ids: list[int]
    mask_token_id: int
    block_size: int
    # 0 disables the Markov head (`deepspec`'s `build_markov_head` returns
    # None); the released checkpoint uses 256.
    markov_rank: int
    enable_confidence_head: bool
    confidence_head_with_markov: bool

    @classmethod
    def from_hf(cls, hf_config: dict[str, Any]) -> Qwen3DSparkConfig:
        """Build from a released drafter's `config.json` (parsed dict).

        `rope_theta` moved into a `rope_parameters` sub-dict in recent
        transformers configs; the released drafters carry the new form and
        Qwen3 targets the old flat one, so read both (same fallback order as
        `Qwen3Config.from_hf`).
        """
        rope_params = hf_config.get("rope_parameters")
        if rope_params is not None and "rope_theta" in rope_params:
            rope_theta = float(rope_params["rope_theta"])
        else:
            rope_theta = float(hf_config.get("rope_theta", 10000.0))
        head_dim = hf_config.get("head_dim") or (
            hf_config["hidden_size"] // hf_config["num_attention_heads"]
        )
        markov_head_type = str(hf_config.get("markov_head_type", "vanilla")).lower()
        if int(hf_config["markov_rank"]) > 0 and markov_head_type != "vanilla":
            # `deepspec` also ships "gated" and "rnn" heads. No released Qwen3
            # drafter uses them, so porting them would be untested code.
            raise NotImplementedError(
                f"markov_head_type={markov_head_type!r} is not ported; "
                "only the released checkpoints' 'vanilla' head is supported"
            )
        target_layer_ids = [int(i) for i in hf_config["target_layer_ids"]]
        if any(i < 0 for i in target_layer_ids):
            # `deepspec` maps -1 to the target's embedding output. No released
            # Qwen3 drafter uses it, and it would need a separate tap point.
            raise NotImplementedError(
                f"target_layer_ids={target_layer_ids} contains -1 (target embedding "
                "output), which is not ported; only decoder-layer taps are supported"
            )
        return cls(
            vocab_size=int(hf_config["vocab_size"]),
            hidden_size=int(hf_config["hidden_size"]),
            intermediate_size=int(hf_config["intermediate_size"]),
            num_hidden_layers=int(hf_config["num_hidden_layers"]),
            num_attention_heads=int(hf_config["num_attention_heads"]),
            num_key_value_heads=int(hf_config["num_key_value_heads"]),
            head_dim=int(head_dim),
            rms_norm_eps=float(hf_config["rms_norm_eps"]),
            rope_theta=rope_theta,
            target_layer_ids=target_layer_ids,
            mask_token_id=int(hf_config["mask_token_id"]),
            block_size=int(hf_config["block_size"]),
            markov_rank=int(hf_config["markov_rank"]),
            enable_confidence_head=bool(hf_config["enable_confidence_head"]),
            confidence_head_with_markov=bool(hf_config.get("confidence_head_with_markov", False)),
        )


class Qwen3DSparkDecoderLayer(nn.Module):
    def __init__(self, cfg: Qwen3DSparkConfig, layer_idx: int) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.self_attn = DSparkAttention(
            hidden_size=cfg.hidden_size,
            num_attention_heads=cfg.num_attention_heads,
            num_key_value_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            rms_norm_eps=cfg.rms_norm_eps,
            layer_idx=layer_idx,
        )
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.mlp = SwiGLU(cfg.hidden_size, cfg.intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_values: DSparkDraftCache | None,
    ) -> torch.Tensor:
        # target_hidden_states is NOT normalized here: it was already
        # projected+normalized once in forward_backbone (fc + hidden_norm)
        # before the layer loop, and every layer reads that same tensor
        # through its OWN k_proj/v_proj. Only the draft block's own
        # hidden_states go through this layer's input_layernorm.
        residual = hidden_states
        x = self.input_layernorm(hidden_states)
        x = self.self_attn(x, target_hidden_states, position_embeddings, past_key_values)
        hidden_states = residual + x

        residual = hidden_states
        x = self.post_attention_layernorm(hidden_states)
        x = self.mlp(x)
        out: torch.Tensor = residual + x
        return out


class Qwen3DSparkDrafter(nn.Module):
    def __init__(self, cfg: Qwen3DSparkConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3DSparkDecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers)]
        )
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(cfg.head_dim, base=cfg.rope_theta)
        # Projects the concatenation of len(target_layer_ids) target-layer
        # hidden states (each cfg.hidden_size wide) down to one cfg.hidden_size
        # vector the draft layers' k_proj/v_proj can consume.
        self.fc = nn.Linear(
            len(cfg.target_layer_ids) * cfg.hidden_size, cfg.hidden_size, bias=False
        )
        self.hidden_norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        # Own copy, not tied to the target's lm_head: the released checkpoint
        # ships full independent embed_tokens/lm_head tensors
        # (tie_word_embeddings: false; see ADR-027).
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        self.markov_head: VanillaMarkovHead | None = None
        if cfg.markov_rank > 0:
            self.markov_head = VanillaMarkovHead(
                vocab_size=cfg.vocab_size, markov_rank=cfg.markov_rank
            )

        self.confidence_head: ConfidenceHead | None = None
        if cfg.enable_confidence_head:
            input_dim = cfg.hidden_size
            if cfg.confidence_head_with_markov:
                if self.markov_head is None:
                    raise ValueError("confidence_head_with_markov=True requires markov_rank > 0")
                input_dim += cfg.markov_rank
            self.confidence_head = ConfidenceHead(input_dim)

    def forward_backbone(
        self,
        *,
        noise_embedding: torch.Tensor,
        target_hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: DSparkDraftCache | None = None,
    ) -> torch.Tensor:
        """One bidirectional pass over the draft block, KV-injected with target context.

        `position_ids` spans BOTH the context and the draft block in one
        contiguous, monotonically-increasing range (context positions first,
        draft positions last): see `attention.py`'s `apply_dspark_rotary_pos_emb`
        for why that single span has to cover both.
        """
        hidden_states = noise_embedding
        target_hidden_states = self.hidden_norm(self.fc(target_hidden_states))
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, target_hidden_states, position_embeddings, past_key_values
            )
        out: torch.Tensor = self.norm(hidden_states)
        return out

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.lm_head(hidden_states)
        return out

    def predict_confidence_step(
        self, hidden_states: torch.Tensor, prev_token_ids: torch.Tensor | None = None
    ) -> torch.Tensor | None:
        """Per-position conditional survival logit (sigmoid applied by the caller, not here)."""
        if self.confidence_head is None:
            return None
        if self.cfg.confidence_head_with_markov:
            assert self.markov_head is not None
            assert prev_token_ids is not None
            prev_embeddings = self.markov_head.get_prev_embeddings(prev_token_ids).to(
                dtype=hidden_states.dtype
            )
            features = torch.cat([hidden_states, prev_embeddings], dim=-1)
            conf_with_markov: torch.Tensor = self.confidence_head(features).float()
            return conf_with_markov
        conf: torch.Tensor = self.confidence_head(hidden_states).float()
        return conf

    def sample_draft_tokens(
        self,
        base_logits: torch.Tensor,
        *,
        first_prev_token_ids: torch.Tensor,
        temperature: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sequentially samples the block, applying the Markov correction per position.

        Returns `(sampled_tokens, corrected_logits)`. Without a Markov head
        (`markov_rank == 0`), each position samples independently from its
        own base logits (no bias, no sequential dependency).
        """
        batch_size, proposal_len = base_logits.shape[:2]
        if proposal_len == 0:
            empty = torch.empty(batch_size, 0, dtype=torch.long, device=base_logits.device)
            return empty, base_logits
        if self.markov_head is None:
            return sample_tokens(base_logits, temperature), base_logits
        return self.markov_head.sample_block_tokens(
            base_logits, first_prev_token_ids=first_prev_token_ids, temperature=temperature
        )

    @staticmethod
    def load_weights(model: Qwen3DSparkDrafter, hf_state_dict: dict[str, torch.Tensor]) -> None:
        """Load a released drafter checkpoint. Every key is an identity rename.

        The checkpoint ships its own full `embed_tokens` / `lm_head` rather than
        tying them to the target's (`tie_word_embeddings: false`; they were
        value-copied from the target and frozen at training time, see ADR-027),
        so this is a plain `load_state_dict` with no aliasing and nothing
        expected to be missing.
        """
        missing, unexpected = model.load_state_dict(hf_state_dict, strict=False)
        if missing or unexpected:
            raise ValueError(
                f"weight load mismatch for Qwen3DSparkDrafter: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
