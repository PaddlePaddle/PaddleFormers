# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Greedy inference for Fleet models using the native KV cache path.

PaddleFleet already wires KV cache through the entire stack:

* ``GPTEmbedding.forward`` sets ``_kv_layer_counter: 0`` in the output dict.
* ``TransformerLayer.forward`` reads / increments the counter and passes
  ``past_key_values``, ``layer_idx``, ``use_cache`` through to the attention
  layers.
* ``DotProductAttention.forward`` calls ``past_key_values.update(k, v,
  layer_idx)`` and switches causal masking based on query length.

This module provides a :class:`DynamicKVCache` that satisfies the
``.update(k, v, layer_idx) -> (k, v)`` protocol and a :class:`GreedyGenerator`
that drives the prefill / decode loop.

Usage::

    from paddleformers.fleet.generation import GreedyGenerator

    gen = GreedyGenerator(model)
    out_ids = gen.generate(input_ids, max_new_tokens=128,
                           eos_token_id=tok.eos_token_id)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import paddle

if TYPE_CHECKING:
    from paddleformers.fleet.models.gpt.gpt_model import GPTModel

logger = logging.getLogger(__name__)


def _apply_repetition_penalty(
    logits: paddle.Tensor, input_ids: paddle.Tensor, penalty: float
) -> paddle.Tensor:
    """Apply repetition penalty to logits.

    Tokens with positive logits are divided by penalty,
    tokens with negative logits are multiplied by penalty.
    """
    if penalty == 1.0:
        return logits

    batch_size, seq_len = input_ids.shape
    vocab_size = logits.shape[-1]

    # Create mask for tokens that appeared in input_ids using scatter
    # This is more efficient than the loop version
    token_mask = paddle.zeros([batch_size, vocab_size], dtype="float32")

    # Flatten input_ids and create batch indices
    flat_input_ids = input_ids.reshape([-1])  # [batch_size * seq_len]
    batch_indices = paddle.arange(batch_size, dtype="int64").unsqueeze(-1)
    batch_indices = batch_indices.expand([batch_size, seq_len]).reshape(
        [-1]
    )  # [batch_size * seq_len]

    # Create indices for scatter
    scatter_indices = paddle.stack(
        [batch_indices, flat_input_ids], axis=-1
    )  # [batch_size * seq_len, 2]

    # Scatter 1.0 to mark appeared tokens
    token_mask = paddle.scatter_nd(
        scatter_indices,
        paddle.ones([batch_size * seq_len], dtype="float32"),
        [batch_size, vocab_size],
    )

    # Apply penalty: divide positive, multiply negative
    mask = token_mask > 0
    logits = paddle.where(
        mask,
        paddle.where(logits > 0, logits / penalty, logits * penalty),
        logits,
    )
    return logits


# ---------------------------------------------------------------------------
# KV cache
# ---------------------------------------------------------------------------


class DynamicKVCache:
    """HF-style dynamic KV cache: per-layer tensors grow by concat.

    Implements the ``.update(k_new, v_new, layer_idx) -> (k, v)`` protocol
    expected by :class:`DotProductAttention`.
    """

    def __init__(self, num_layers: int):
        self.k: list[paddle.Tensor | None] = [None] * num_layers
        self.v: list[paddle.Tensor | None] = [None] * num_layers

    def get_seq_len(self, layer_idx: int = 0) -> int:
        if self.k[layer_idx] is not None:
            return self.k[layer_idx].shape[1]
        # Fallback: find first non-empty layer
        for k in self.k:
            if k is not None:
                return k.shape[1]
        return 0

    def update(
        self, k_new: paddle.Tensor, v_new: paddle.Tensor, layer_idx: int
    ):
        if self.k[layer_idx] is None:
            self.k[layer_idx] = k_new
            self.v[layer_idx] = v_new
        else:
            self.k[layer_idx] = paddle.concat(
                [self.k[layer_idx], k_new], axis=1
            )
            self.v[layer_idx] = paddle.concat(
                [self.v[layer_idx], v_new], axis=1
            )
        return self.k[layer_idx], self.v[layer_idx]

    def reset(self) -> None:
        for i in range(len(self.k)):
            self.k[i] = None
            self.v[i] = None


# ---------------------------------------------------------------------------
# Greedy generator
# ---------------------------------------------------------------------------


class GreedyGenerator:
    """Greedy decode on top of a FleetGPTModel using the native KV cache path.

    No monkey-patching is needed — the model's own forward pass already
    supports KV cache via the ``past_key_values`` / ``use_cache`` mechanism.

    Usage::

        model = Qwen3MoeForCausalLM.from_pretrained(model_dir, config=config)
        gen = GreedyGenerator(model)
        out = gen.generate(input_ids, max_new_tokens=128,
                           eos_token_id=tok.eos_token_id)
    """

    def __init__(self, fleet_model: GPTModel):
        cfg = fleet_model.config

        if getattr(cfg, "sequence_parallel", False):
            raise ValueError(
                "sequence_parallel must be False for inference with KV cache. "
                "Set config.sequence_parallel = False before building the model."
            )
        if getattr(cfg, "apply_rope_fusion", False):
            logger.warning(
                "apply_rope_fusion=True may cause issues with KV cache "
                "inference. If outputs are incorrect, set "
                "config.apply_rope_fusion = False."
            )
        if getattr(cfg, "recompute_granularity", None) == "full":
            logger.warning(
                "recompute_granularity='full' drops KV cache kwargs. "
                "Make sure model.eval() is called before generate()."
            )

        self.model = fleet_model
        num_layers = cfg.num_hidden_layers
        # Account for empty layers in head/tail that offset layer_number
        num_empty_layers_add_in_head = getattr(
            cfg, "num_empty_layers_add_in_head", 0
        )
        num_empty_layers_add_in_tail = getattr(
            cfg, "num_empty_layers_add_in_tail", 0
        )
        total_layers = (
            num_layers
            + num_empty_layers_add_in_head
            + num_empty_layers_add_in_tail
        )
        self.cache = DynamicKVCache(num_layers=total_layers)

    @paddle.no_grad()
    def generate(
        self,
        input_ids: paddle.Tensor,
        max_new_tokens: int,
        eos_token_id: int | None = None,
        repetition_penalty: float = 1.0,
    ) -> paddle.Tensor:
        """Run greedy auto-regressive decoding.

        Args:
            input_ids: Token IDs with shape ``[B, L]``.
            max_new_tokens: Maximum number of new tokens to generate.
            eos_token_id: Stop generation when this token is produced (all
                batches must be done).
            repetition_penalty: Penalty for repeated tokens (1.0 = no penalty,
                >1.0 = discourage repetition). Default: 1.0.

        Returns:
            Tensor of shape ``[B, L + num_generated]`` containing the
            prompt plus generated tokens.
        """
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be [B, L]")
        self.cache.reset()
        self.model.eval()

        bsz, prompt_len = input_ids.shape
        generated = input_ids.clone()

        with paddle.amp.auto_cast(True, level="O2", dtype="bfloat16"):
            # ---- Prefill ----
            position_ids = (
                paddle.arange(prompt_len, dtype="int64")
                .unsqueeze(0)
                .expand([bsz, prompt_len])
            )
            logits = self.model(
                {
                    "input_ids": input_ids,
                    "position_ids": position_ids,
                    "past_key_values": self.cache,
                    "use_cache": True,
                }
            )
            # Apply repetition penalty to prefill output
            logits = _apply_repetition_penalty(
                logits, generated, repetition_penalty
            )
            next_tok = logits[:, -1].argmax(axis=-1, keepdim=True)
            generated = paddle.concat([generated, next_tok], axis=1)

            # ---- Decode ----
            done = paddle.zeros([bsz, 1], dtype="bool")
            for step in range(max_new_tokens - 1):
                cur_len = self.cache.get_seq_len()
                position_ids = paddle.full([bsz, 1], cur_len, dtype="int64")
                logits = self.model(
                    {
                        "input_ids": next_tok,
                        "position_ids": position_ids,
                        "past_key_values": self.cache,
                        "use_cache": True,
                    }
                )
                # Apply repetition penalty
                logits = _apply_repetition_penalty(
                    logits, generated, repetition_penalty
                )
                next_tok = logits[:, -1].argmax(axis=-1, keepdim=True)
                generated = paddle.concat([generated, next_tok], axis=1)
                if eos_token_id is not None:
                    done = done | (next_tok == eos_token_id)
                    if done.all().item():
                        break

        return generated
