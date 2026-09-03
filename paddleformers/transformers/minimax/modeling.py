# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
""" Paddle MiniMax (Text-01) model."""

from __future__ import annotations

from typing import Callable

import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.distributed.fleet.recompute.recompute import recompute

from ...nn.attention.interface import ALL_ATTENTION_FUNCTIONS
from ...nn.criterion.interface import CriterionLayer
from ...nn.embedding import Embedding as GeneralEmbedding
from ...nn.linear import Linear as GeneralLinear
from ...nn.lm_head import LMHead as GeneralLMHead
from ...nn.norm import Norm as GeneralNorm
from ...utils.log import logger
from ..cache_utils import Cache, DynamicCache
from ..masking_utils import create_causal_mask_and_row_indices
from ..model_outputs import MoECausalLMOutputWithPast, MoEModelOutputWithPast
from ..model_utils import PretrainedModel, register_base_model
from ..modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from .configuration import MiniMaxConfig


def rotate_half(x: paddle.Tensor) -> paddle.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.cat((-x2, x1), axis=-1)


def apply_rotary_pos_emb(
    q: paddle.Tensor,
    k: paddle.Tensor,
    cos: paddle.Tensor,
    sin: paddle.Tensor,
    position_ids: paddle.Tensor | None = None,
    unsqueeze_dim: int = 1,
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q: query tensor with shape [..., head_dim]
        k: key tensor with shape [..., head_dim]
        cos: cosine values
        sin: sine values
        unsqueeze_dim: dimension to unsqueeze cos/sin for broadcasting
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.astype(q.dtype), k_embed.astype(k.dtype)


class MiniMaxCache(DynamicCache):
    """Cache for MiniMax that supports both standard KV-cache and linear-attention (KV-statistic) cache."""

    def __init__(self, config: MiniMaxConfig):
        super().__init__(config=config)
        self.linear_cache: list[paddle.Tensor] = []

    def set_linear_cache(self, layer_idx: int, linear_cache: paddle.Tensor):
        for _ in range(len(self.linear_cache), layer_idx + 1):
            self.linear_cache.append(None)
        self.linear_cache[layer_idx] = linear_cache

    def get_linear_cache(self, layer_idx: int):
        if layer_idx < len(self.linear_cache):
            return self.linear_cache[layer_idx]
        return None

    def __len__(self):
        return max(super().__len__(), len(self.linear_cache))

    def crop(self, max_length: int):
        raise RuntimeError("MiniMaxCache does not support `crop` method")


class MiniMaxLightningAttention(nn.Layer):
    """Linear attention ("lightning attention") for MiniMax.

    Operates on intra-block (within the current block) and inter-block (using cached KV statistics)
    components. Statistics are computed by a gated linear unit (GLU) on a fused QKV projection.
    """

    def __init__(self, config: MiniMaxConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        self.num_attention_heads = config.num_attention_heads
        self.num_hidden_layers = config.num_hidden_layers
        self.block_size = config.block_size

        hidden_size = config.hidden_size
        qkv_out = self.num_attention_heads * self.head_dim * 3
        attn_out = self.num_attention_heads * self.head_dim

        self.qkv_proj = GeneralLinear.create(
            hidden_size,
            qkv_out,
            has_bias=False,
            config=config,
            tp_plan="colwise",
        )
        self.out_proj = GeneralLinear.create(
            attn_out,
            hidden_size,
            has_bias=False,
            config=config,
            tp_plan="rowwise",
        )
        self.output_gate = GeneralLinear.create(
            hidden_size,
            attn_out,
            has_bias=False,
            config=config,
            tp_plan="colwise",
        )
        self.norm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=self.head_dim * self.num_attention_heads,
            norm_eps=config.rms_norm_eps,
            input_is_parallel=config.sequence_parallel,
        )

        self.act_fn = F.silu

        slope_rate = self.get_slope_rate()
        query_decay, key_decay, diagonal_decay = self.decay_factors(slope_rate)

        self.register_buffer("slope_rate", slope_rate, persistable=False)
        self.register_buffer("query_decay", query_decay, persistable=False)
        self.register_buffer("key_decay", key_decay, persistable=False)
        self.register_buffer("diagonal_decay", diagonal_decay, persistable=False)

    def get_slope_rate(self) -> paddle.Tensor:
        base = 1.0 / (2.0 ** (8.0 / self.num_attention_heads))
        dtype = paddle.get_default_dtype()
        exponent = paddle.arange(self.num_attention_heads).astype(dtype) + 1
        factor = 1.0 - self.layer_idx / (self.num_hidden_layers - 1 + 1e-5) + 1e-5

        rate = paddle.pow(paddle.to_tensor(base, dtype=dtype), exponent)
        rate = rate * factor
        rate = rate.unsqueeze(-1).unsqueeze(-1)
        return rate

    def decay_factors(self, slope_rate: paddle.Tensor) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        block_size_range = paddle.arange(self.block_size).astype(slope_rate.dtype) + 1

        query_decay = paddle.exp(-slope_rate * block_size_range.unsqueeze(-1))
        key_decay = paddle.exp(-slope_rate * (self.block_size - block_size_range.unsqueeze(-1)))

        diff = block_size_range.unsqueeze(-1) - block_size_range.unsqueeze(0)
        diff = diff.unsqueeze(0).unsqueeze(0)
        decay = slope_rate * diff
        neg_inf = paddle.full_like(decay, fill_value=float("-inf"))
        decay = paddle.where(decay >= 0, -decay, neg_inf)
        diagonal_decay = paddle.exp(decay)

        return query_decay, key_decay, diagonal_decay

    def _contiguous_path_attention(
        self,
        query_states: paddle.Tensor,
        key_states: paddle.Tensor,
        value_states: paddle.Tensor,
        slope_rate: paddle.Tensor,
    ) -> tuple[paddle.Tensor, paddle.Tensor]:
        """Run the original block algorithm on one independent causal path."""
        _, _, seq_len, head_dim = query_states.shape
        attn_weights_inter = paddle.zeros(
            [query_states.shape[0], self.num_attention_heads, head_dim, head_dim],
            dtype=query_states.dtype,
        )
        attn_output = []
        for start_idx in range(0, seq_len, self.block_size):
            end_idx = min(start_idx + self.block_size, seq_len)
            cur_bs = end_idx - start_idx

            cur_q = query_states[:, :, start_idx:end_idx]
            cur_k = key_states[:, :, start_idx:end_idx]
            cur_v = value_states[:, :, start_idx:end_idx]

            cur_qd = self.query_decay[:, :cur_bs].astype(cur_q.dtype)
            cur_kd = self.key_decay[:, -cur_bs:].astype(cur_k.dtype)
            cur_dd = self.diagonal_decay[:, :, :cur_bs, :cur_bs].astype(cur_q.dtype)
            block_decay = paddle.exp(-slope_rate * cur_bs).astype(attn_weights_inter.dtype)

            attn_intra = paddle.matmul(cur_q, cur_k.transpose([0, 1, 3, 2]))
            attn_output_intra = paddle.matmul(attn_intra * cur_dd, cur_v)
            attn_output_inter = paddle.matmul(cur_q * cur_qd, attn_weights_inter)
            attn_output.append(attn_output_inter + attn_output_intra)

            next_stat = paddle.matmul((cur_k * cur_kd).transpose([0, 1, 3, 2]), cur_v)
            attn_weights_inter = attn_weights_inter * block_decay + next_stat

        return paddle.cat(attn_output, axis=-2), attn_weights_inter

    def _positioned_path_attention(
        self,
        query_states: paddle.Tensor,
        key_states: paddle.Tensor,
        value_states: paddle.Tensor,
        position_ids: paddle.Tensor,
        slope_rate: paddle.Tensor,
    ) -> tuple[paddle.Tensor, paddle.Tensor]:
        """Run block lightning attention on a causal path whose positions may repeat."""
        _, _, seq_len, head_dim = query_states.shape
        attn_weights_inter = paddle.zeros(
            [query_states.shape[0], self.num_attention_heads, head_dim, head_dim],
            dtype=query_states.dtype,
        )
        attn_output = []
        slope_rate_4d = slope_rate.unsqueeze(0)

        for start_idx in range(0, seq_len, self.block_size):
            end_idx = min(start_idx + self.block_size, seq_len)
            cur_q = query_states[:, :, start_idx:end_idx]
            cur_k = key_states[:, :, start_idx:end_idx]
            cur_v = value_states[:, :, start_idx:end_idx]

            cur_positions = position_ids[start_idx:end_idx].astype(slope_rate.dtype)
            previous_position = (
                position_ids[start_idx - 1].astype(slope_rate.dtype) if start_idx > 0 else cur_positions[0] - 1
            )
            last_position = cur_positions[-1]

            query_distance = cur_positions - previous_position
            key_distance = last_position - cur_positions
            pair_distance = cur_positions.unsqueeze(-1) - cur_positions.unsqueeze(0)
            physical_causal_mask = paddle.tril(
                paddle.ones([end_idx - start_idx, end_idx - start_idx], dtype=paddle.bool)
            )

            query_decay = paddle.exp(-slope_rate * query_distance.reshape([1, -1, 1])).astype(cur_q.dtype)
            key_decay = paddle.exp(-slope_rate * key_distance.reshape([1, -1, 1])).astype(cur_k.dtype)
            diagonal_decay = paddle.where(
                physical_causal_mask.reshape([1, 1, end_idx - start_idx, end_idx - start_idx])
                & (pair_distance.reshape([1, 1, end_idx - start_idx, end_idx - start_idx]) >= 0),
                paddle.exp(-slope_rate_4d * pair_distance.reshape([1, 1, end_idx - start_idx, end_idx - start_idx])),
                paddle.zeros(
                    [1, self.num_attention_heads, end_idx - start_idx, end_idx - start_idx],
                    dtype=slope_rate.dtype,
                ),
            ).astype(cur_q.dtype)
            block_decay = paddle.exp(-slope_rate * (last_position - previous_position)).astype(
                attn_weights_inter.dtype
            )

            attn_intra = paddle.matmul(cur_q, cur_k.transpose([0, 1, 3, 2]))
            attn_output_intra = paddle.matmul(attn_intra * diagonal_decay, cur_v)
            attn_output_inter = paddle.matmul(cur_q * query_decay, attn_weights_inter)
            attn_output.append(attn_output_inter + attn_output_intra)

            next_stat = paddle.matmul((cur_k * key_decay).transpose([0, 1, 3, 2]), cur_v)
            attn_weights_inter = attn_weights_inter * block_decay + next_stat

        return paddle.cat(attn_output, axis=-2), attn_weights_inter

    @staticmethod
    def _end_indices_from_4d_mask(attention_mask: paddle.Tensor) -> list[list[int]]:
        """Convert a causal dense mask into each key's exclusive query-end index."""
        if attention_mask.shape[1] != 1 or attention_mask.shape[-2] != attention_mask.shape[-1]:
            raise ValueError("MiniMax linear_attention expects a 4D causal mask with shape [batch, 1, seq, seq].")

        dense_masks = attention_mask[:, 0].astype("bool").numpy()
        all_end_indices = []
        for dense_mask in dense_masks:
            seq_len = dense_mask.shape[0]
            end_indices = []
            for key_idx in range(seq_len):
                visible_queries = [query_idx for query_idx in range(seq_len) if dense_mask[query_idx, key_idx]]
                if not visible_queries:
                    end_indices.append(key_idx)
                    continue
                expected_queries = list(range(key_idx, visible_queries[-1] + 1))
                if visible_queries != expected_queries:
                    raise ValueError(
                        "MiniMax linear_attention only supports causal masks where each key is visible "
                        "to one contiguous range of query rows."
                    )
                end_indices.append(visible_queries[-1] + 1)
            all_end_indices.append(end_indices)
        return all_end_indices

    def _structured_attention(
        self,
        query_states: paddle.Tensor,
        key_states: paddle.Tensor,
        value_states: paddle.Tensor,
        position_ids: paddle.Tensor,
        end_indices_by_batch: list[list[int]],
        slope_rate: paddle.Tensor,
    ) -> tuple[paddle.Tensor, paddle.Tensor]:
        """Evaluate packed or branched causal paths without leaking recurrent state."""
        batch_outputs = []
        batch_final_states = []
        seq_len = query_states.shape[2]
        position_ids_by_batch = position_ids.astype("int64").numpy().tolist()

        for batch_idx, (end_indices, batch_positions) in enumerate(zip(end_indices_by_batch, position_ids_by_batch)):
            if len(end_indices) != seq_len or len(batch_positions) != seq_len:
                raise ValueError(
                    "MiniMax linear_attention mask and position_ids must match the input sequence length."
                )

            valid_length = next(
                (key_idx for key_idx, end_idx in enumerate(end_indices) if end_idx <= key_idx),
                seq_len,
            )
            if any(end_idx <= key_idx for key_idx, end_idx in enumerate(end_indices[:valid_length])):
                raise ValueError(
                    "MiniMax linear_attention structured masks must contain one contiguous valid-token prefix."
                )
            if any(
                end_idx > key_idx or end_idx > seq_len
                for key_idx, end_idx in enumerate(end_indices[valid_length:], start=valid_length)
            ):
                raise ValueError(
                    "MiniMax linear_attention only supports an invisible, contiguous right-padding suffix."
                )

            valid_end_indices = end_indices[:valid_length]
            if any(end_idx > valid_length for end_idx in valid_end_indices):
                raise ValueError("MiniMax linear_attention valid tokens cannot attend into the right-padding suffix.")

            if valid_length == 0:
                batch_outputs.append(paddle.zeros_like(query_states[batch_idx : batch_idx + 1]))
                batch_final_states.append(
                    paddle.zeros(
                        [1, self.num_attention_heads, query_states.shape[-1], query_states.shape[-1]],
                        dtype=query_states.dtype,
                    )
                )
                continue

            boundaries = sorted(set(valid_end_indices))
            if not boundaries or boundaries[-1] != valid_length:
                raise ValueError("MiniMax linear_attention structured mask must cover every valid query row.")

            interval_outputs = []
            final_state = None
            interval_start = 0
            for interval_end in boundaries:
                if interval_end <= interval_start:
                    continue

                active_prefix = [
                    key_idx for key_idx in range(interval_start) if valid_end_indices[key_idx] > interval_start
                ]
                current_interval = list(range(interval_start, interval_end))
                path_indices = active_prefix + current_interval
                path_positions = [batch_positions[index] for index in path_indices]
                if any(right < left for left, right in zip(path_positions, path_positions[1:])):
                    raise ValueError(
                        "MiniMax linear_attention cannot represent this structured mask as non-decreasing "
                        "logical-position paths."
                    )

                index_tensor = paddle.to_tensor(path_indices, dtype="int64")
                path_q = paddle.index_select(query_states[batch_idx : batch_idx + 1], index_tensor, axis=2)
                path_k = paddle.index_select(key_states[batch_idx : batch_idx + 1], index_tensor, axis=2)
                path_v = paddle.index_select(value_states[batch_idx : batch_idx + 1], index_tensor, axis=2)
                path_position_ids = paddle.to_tensor(path_positions, dtype="int64")

                has_unit_position_steps = all(
                    right == left + 1 for left, right in zip(path_positions, path_positions[1:])
                )
                if has_unit_position_steps:
                    path_output, final_state = self._contiguous_path_attention(path_q, path_k, path_v, slope_rate)
                else:
                    path_output, final_state = self._positioned_path_attention(
                        path_q,
                        path_k,
                        path_v,
                        path_position_ids,
                        slope_rate,
                    )

                interval_outputs.append(path_output[:, :, -len(current_interval) :])
                interval_start = interval_end

            if interval_start != valid_length:
                raise ValueError("MiniMax linear_attention structured mask did not cover the valid-token prefix.")

            batch_output = paddle.cat(interval_outputs, axis=2)
            if valid_length < seq_len:
                batch_output = paddle.cat(
                    [
                        batch_output,
                        paddle.zeros_like(query_states[batch_idx : batch_idx + 1, :, valid_length:]),
                    ],
                    axis=2,
                )
            batch_outputs.append(batch_output)
            batch_final_states.append(final_state)

        return paddle.cat(batch_outputs, axis=0), paddle.cat(batch_final_states, axis=0)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        position_embeddings: tuple[paddle.Tensor, paddle.Tensor] | None = None,
        attention_mask: paddle.Tensor | None = None,
        position_ids: paddle.Tensor | None = None,
        attn_mask_startend_row_indices: paddle.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[paddle.Tensor, paddle.Tensor | None]:
        batch_size, seq_len, _ = hidden_states.shape
        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        slope_rate = self.slope_rate.astype(hidden_states.dtype)

        qkv_states = self.act_fn(self.qkv_proj(hidden_states))
        qkv_states = qkv_states.reshape([batch_size, seq_len, self.num_attention_heads, 3 * self.head_dim])
        query_states, key_states, value_states = paddle.split(qkv_states, num_or_sections=3, axis=-1)

        query_states = query_states.transpose([0, 2, 1, 3])
        key_states = key_states.transpose([0, 2, 1, 3])
        value_states = value_states.transpose([0, 2, 1, 3])

        attn_weights_inter = None
        if past_key_values is not None and isinstance(past_key_values, MiniMaxCache):
            attn_weights_inter = past_key_values.get_linear_cache(self.layer_idx)

        end_indices_by_batch = None
        if attn_mask_startend_row_indices is not None:
            row_indices = attn_mask_startend_row_indices
            if row_indices.ndim == 3:
                row_indices = row_indices.unsqueeze(-1)
            if (
                row_indices.ndim != 4
                or row_indices.shape[0] != batch_size
                or row_indices.shape[1] != 1
                or row_indices.shape[2] != seq_len
                or row_indices.shape[3] != 1
            ):
                raise ValueError(
                    "MiniMax linear_attention currently supports causal row indices with shape " "[batch, 1, seq, 1]."
                )
            end_indices_by_batch = row_indices[:, 0, :, 0].astype("int64").numpy().tolist()
        elif attention_mask is not None and attention_mask.ndim == 4:
            end_indices_by_batch = self._end_indices_from_4d_mask(attention_mask)
            attention_mask = None

        if end_indices_by_batch is not None:
            if attn_weights_inter is not None:
                raise ValueError("MiniMax cached linear_attention does not support structured training masks.")
            if position_ids is None or position_ids.ndim != 2:
                raise ValueError("MiniMax structured linear_attention requires 2D position_ids.")
            attn_output, attn_weights_inter = self._structured_attention(
                query_states,
                key_states,
                value_states,
                position_ids,
                end_indices_by_batch,
                slope_rate,
            )
        elif attn_weights_inter is None:
            attn_weights_inter = paddle.zeros(
                [batch_size, self.num_attention_heads, self.head_dim, self.head_dim],
                dtype=hidden_states.dtype,
            )

            if attention_mask is not None:
                bool_mask = attention_mask.astype("bool")
                expanded = bool_mask.unsqueeze(1).unsqueeze(-1)
                value_states = paddle.where(expanded, value_states, paddle.zeros_like(value_states))

            attn_output = []
            for i in range(num_blocks):
                start_idx = i * self.block_size
                end_idx = min(start_idx + self.block_size, seq_len)
                cur_bs = end_idx - start_idx

                cur_q = query_states[:, :, start_idx:end_idx]
                cur_k = key_states[:, :, start_idx:end_idx]
                cur_v = value_states[:, :, start_idx:end_idx]

                cur_qd = self.query_decay[:, :cur_bs].astype(cur_q.dtype)
                cur_kd = self.key_decay[:, -cur_bs:].astype(cur_k.dtype)
                cur_dd = self.diagonal_decay[:, :, :cur_bs, :cur_bs].astype(cur_q.dtype)
                block_decay = paddle.exp(-slope_rate * cur_bs).astype(attn_weights_inter.dtype)

                attn_intra = paddle.matmul(cur_q, cur_k.transpose([0, 1, 3, 2]))
                attn_output_intra = paddle.matmul(attn_intra * cur_dd, cur_v)

                attn_output_inter = paddle.matmul(cur_q * cur_qd, attn_weights_inter)

                cur_out = attn_output_inter + attn_output_intra
                attn_output.append(cur_out)

                next_stat = paddle.matmul((cur_k * cur_kd).transpose([0, 1, 3, 2]), cur_v)
                attn_weights_inter = attn_weights_inter * block_decay + next_stat
        else:
            ratio = paddle.exp(-slope_rate).astype(attn_weights_inter.dtype)
            attn_output = []
            for i in range(seq_len):
                cur_q = query_states[:, :, i : i + 1]
                cur_k = key_states[:, :, i : i + 1]
                cur_v = value_states[:, :, i : i + 1]

                cur_stat = paddle.matmul(cur_k.transpose([0, 1, 3, 2]), cur_v)
                attn_weights_inter = ratio * attn_weights_inter + cur_stat
                cur_out = paddle.matmul(cur_q, attn_weights_inter)
                attn_output.append(cur_out)

        if isinstance(attn_output, list):
            attn_output = paddle.cat(attn_output, axis=-2)

        attn_output = attn_output.transpose([0, 2, 1, 3])
        attn_output = attn_output.reshape([batch_size, seq_len, self.num_attention_heads * self.head_dim])
        attn_output = self.norm(attn_output)
        attn_output = F.sigmoid(self.output_gate(hidden_states)).astype(attn_output.dtype) * attn_output
        attn_output = attn_output.astype(hidden_states.dtype)
        attn_output = self.out_proj(attn_output)

        if past_key_values is not None and isinstance(past_key_values, MiniMaxCache):
            past_key_values.set_linear_cache(self.layer_idx, attn_weights_inter)

        return attn_output, attn_weights_inter


class MiniMaxRotaryEmbedding(nn.Layer):
    inv_freq: paddle.Tensor  # for `register_buffer` typing

    def __init__(self, config: MiniMaxConfig):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        if hasattr(config, "rope_parameters") and isinstance(config.rope_parameters, dict):
            self.rope_type = config.rope_parameters.get("rope_type", "default")
        else:
            self.rope_type = "default"

        rope_init_fn = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config)

        self.register_buffer("inv_freq", inv_freq, persistable=False)
        self.original_inv_freq = inv_freq

    @staticmethod
    def compute_default_rope_parameters(
        config: MiniMaxConfig | None = None,
        seq_len: int | None = None,
    ) -> tuple["paddle.Tensor", float]:
        """Compute default RoPE inverse frequencies."""
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        attention_factor = 1.0
        inv_freq = 1.0 / (base ** (paddle.arange(0, dim, 2, dtype=paddle.int64).astype(dtype=paddle.float32) / dim))
        return inv_freq, attention_factor

    @dynamic_rope_update
    def forward(self, x: paddle.Tensor, position_ids: paddle.Tensor) -> tuple[paddle.Tensor, paddle.Tensor]:
        with paddle.amp.auto_cast(enable=False):
            inv_freq_expanded = self.inv_freq[None, :, None].float().expand([position_ids.shape[0], -1, 1]).to(x.dtype)
            position_ids_expanded = position_ids[:, None, :].float()

            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose([0, 2, 1])
            emb = paddle.concat((freqs, freqs), axis=-1)

            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.astype(x.dtype), sin.astype(x.dtype)


class MiniMaxAttention(nn.Layer):
    """Multi-headed attention (full attention) from MiniMax.

    Includes Grouped Query Attention (GQA) when num_key_value_heads != num_attention_heads.
    """

    def __init__(self, config: MiniMaxConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout

        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads

        self.q_proj = GeneralLinear.create(
            config.hidden_size,
            self.num_attention_heads * self.head_dim,
            has_bias=False,
            config=config,
            tp_plan="colwise",
        )
        self.k_proj = GeneralLinear.create(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            has_bias=False,
            config=config,
            tp_plan="colwise",
        )
        self.v_proj = GeneralLinear.create(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            has_bias=False,
            config=config,
            tp_plan="colwise",
        )
        self.o_proj = GeneralLinear.create(
            self.num_attention_heads * self.head_dim,
            config.hidden_size,
            has_bias=False,
            config=config,
            tp_plan="rowwise",
        )

    def forward(
        self,
        hidden_states: paddle.Tensor,
        position_embeddings: tuple[paddle.Tensor, paddle.Tensor] | None = None,
        attention_mask: paddle.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool = False,
        attn_mask_startend_row_indices: paddle.Tensor | None = None,
        **kwargs,
    ) -> tuple[paddle.Tensor, paddle.Tensor | None]:
        bsz, q_len, _ = hidden_states.shape
        hidden_shape = [bsz, q_len, -1, self.head_dim]

        query_states = self.q_proj(hidden_states).reshape(hidden_shape).transpose([0, 2, 1, 3])
        key_states = self.k_proj(hidden_states).reshape(hidden_shape).transpose([0, 2, 1, 3])
        value_states = self.v_proj(hidden_states).reshape(hidden_shape).transpose([0, 2, 1, 3])

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS[getattr(self.config, "_attn_implementation", "eager")]

        attn_output, attn_weights = attention_interface(
            self,
            query=query_states,
            key=key_states,
            value=value_states,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
        )

        attn_output = attn_output.reshape([bsz, q_len, -1]).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class MiniMaxBlockSparseTop2MLP(nn.Layer):
    def __init__(self, config: MiniMaxConfig):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.intermediate_size
        self.w1 = GeneralLinear.create(
            self.hidden_dim,
            self.intermediate_dim,
            has_bias=False,
            config=config,
            tp_plan="colwise",
        )
        self.w2 = GeneralLinear.create(
            self.intermediate_dim,
            self.hidden_dim,
            has_bias=False,
            config=config,
            tp_plan="rowwise",
        )
        self.w3 = GeneralLinear.create(
            self.hidden_dim,
            self.intermediate_dim,
            has_bias=False,
            config=config,
            tp_plan="colwise",
        )
        self.act_fn = F.silu

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        return self.w2(hidden_states)


class MiniMaxSparseMoeBlock(nn.Layer):
    """Sparse MoE block (router + experts) for MiniMax."""

    def __init__(self, config: MiniMaxConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.jitter_noise = config.router_jitter_noise
        self.gate = GeneralLinear.create(
            config.hidden_size,
            self.num_experts,
            has_bias=False,
            config=config,
            tp_plan="colwise",
            gather_output=True,
        )
        self.experts = nn.LayerList([MiniMaxBlockSparseTop2MLP(config) for _ in range(config.num_local_experts)])

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if self.training and self.jitter_noise > 0:
            hidden_states = hidden_states * paddle.uniform(
                hidden_states.shape,
                dtype=hidden_states.dtype,
                min=1.0 - self.jitter_noise,
                max=1.0 + self.jitter_noise,
            )
        hidden_states_flat = hidden_states.reshape([-1, hidden_states.shape[-1]])
        final_hidden_states = paddle.zeros_like(hidden_states_flat)
        router_logits = F.softmax(self.gate(hidden_states_flat).astype("float32"), axis=-1)
        routing_weights, selected_experts = paddle.topk(router_logits, self.top_k, axis=-1)
        routing_weights = routing_weights / routing_weights.sum(axis=-1, keepdim=True)
        routing_weights = routing_weights.astype(hidden_states.dtype)

        with paddle.no_grad():
            expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts)
            expert_mask = expert_mask.transpose([2, 1, 0])
            expert_indices = [
                int(expert_idx[0].item())
                for expert_idx in paddle.greater(
                    expert_mask.sum(axis=(-1, -2)),
                    paddle.to_tensor(0, dtype="int64"),
                ).nonzero()
            ]

        for expert_idx in expert_indices:
            top_k_pos, token_idx = paddle.where(expert_mask[expert_idx])
            current_state = hidden_states_flat[token_idx]
            current_hidden_states = self.experts[expert_idx](current_state)
            current_hidden_states = current_hidden_states * routing_weights[token_idx, top_k_pos, None]
            final_hidden_states = final_hidden_states.index_add_(
                axis=0,
                index=token_idx,
                value=current_hidden_states.astype(final_hidden_states.dtype),
            )

        return final_hidden_states.reshape([batch_size, sequence_length, hidden_dim])


class MiniMaxDecoderLayer(nn.Layer):
    """A single decoder layer. Selects between full and linear attention by `config.layer_types[layer_idx]`."""

    def __init__(self, config: MiniMaxConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.mlp_alpha_factor = config.mlp_alpha_factor
        self.mlp_beta_factor = config.mlp_beta_factor

        self.input_layernorm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=config.hidden_size,
            norm_eps=config.rms_norm_eps,
            input_is_parallel=config.sequence_parallel,
        )
        self.post_attention_layernorm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=config.hidden_size,
            norm_eps=config.rms_norm_eps,
            input_is_parallel=config.sequence_parallel,
        )

        if self.layer_type == "linear_attention":
            self.self_attn = MiniMaxLightningAttention(config, layer_idx)
            self.attn_alpha_factor = config.linear_attn_alpha_factor
            self.attn_beta_factor = config.linear_attn_beta_factor
        elif self.layer_type == "full_attention":
            self.self_attn = MiniMaxAttention(config, layer_idx)
            self.attn_alpha_factor = config.full_attn_alpha_factor
            self.attn_beta_factor = config.full_attn_beta_factor
        else:
            raise ValueError(
                f"Unknown layer_type '{self.layer_type}'. Expected 'full_attention' or 'linear_attention'."
            )

        self.block_sparse_moe = MiniMaxSparseMoeBlock(config)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: paddle.Tensor | None = None,
        position_ids: paddle.Tensor | None = None,
        position_embeddings: tuple[paddle.Tensor, paddle.Tensor] | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        attn_mask_startend_row_indices: paddle.Tensor | None = None,
        **kwargs,
    ) -> paddle.Tensor:
        hidden_states = self.input_layernorm(hidden_states)
        residual = hidden_states

        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            **kwargs,
        )
        hidden_states = attn_outputs[0]
        hidden_states = residual * self.attn_alpha_factor + hidden_states * self.attn_beta_factor

        hidden_states = self.post_attention_layernorm(hidden_states)
        residual = hidden_states
        hidden_states = self.block_sparse_moe(hidden_states)
        hidden_states = residual * self.mlp_alpha_factor + hidden_states * self.mlp_beta_factor

        return hidden_states


class MiniMaxPretrainedModel(PretrainedModel):
    config_class = MiniMaxConfig
    config: MiniMaxConfig
    base_model_prefix = "model"

    transpose_weight_keys = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate",
        "w1",
        "w2",
        "w3",
        "output_gate",
        "qkv_proj",
        "out_proj",
    ]

    @classmethod
    def _gen_aoa_config(cls, config: MiniMaxConfig):
        """AOA config: mapping from HF safetensors key to PaddleFormers model key."""
        model_prefix = "model." if cls != cls.base_model_class else ""

        aoa_statements = [
            f"model.embed_tokens.weight -> {model_prefix}embed_tokens.weight",
            f"model.norm.weight -> {model_prefix}norm.weight",
        ]

        # Layer-wise mappings
        for layer_idx in range(config.num_hidden_layers):
            layer_type = config.layer_types[layer_idx]
            prefix = f"model.layers.{layer_idx}"
            dst_prefix = f"{model_prefix}layers.{layer_idx}"

            # Layer norms (always present).
            aoa_statements.extend(
                [
                    f"{prefix}.input_layernorm.weight -> {dst_prefix}.input_layernorm.weight",
                    f"{prefix}.post_attention_layernorm.weight -> {dst_prefix}.post_attention_layernorm.weight",
                ]
            )

            # Attention weights (with transpose for Linear)
            if layer_type == "full_attention":
                aoa_statements.extend(
                    [
                        f"{prefix}.self_attn.q_proj.weight^T -> {dst_prefix}.self_attn.q_proj.weight",
                        f"{prefix}.self_attn.k_proj.weight^T -> {dst_prefix}.self_attn.k_proj.weight",
                        f"{prefix}.self_attn.v_proj.weight^T -> {dst_prefix}.self_attn.v_proj.weight",
                        f"{prefix}.self_attn.o_proj.weight^T -> {dst_prefix}.self_attn.o_proj.weight",
                    ]
                )
            elif layer_type == "linear_attention":
                aoa_statements.extend(
                    [
                        f"{prefix}.self_attn.qkv_proj.weight^T -> {dst_prefix}.self_attn.qkv_proj.weight",
                        f"{prefix}.self_attn.output_gate.weight^T -> {dst_prefix}.self_attn.output_gate.weight",
                        f"{prefix}.self_attn.out_proj.weight^T -> {dst_prefix}.self_attn.out_proj.weight",
                        f"{prefix}.self_attn.norm.weight -> {dst_prefix}.self_attn.norm.weight",
                    ]
                )

            aoa_statements.append(
                f"{prefix}.block_sparse_moe.gate.weight^T -> {dst_prefix}.block_sparse_moe.gate.weight"
            )
            for expert_idx in range(config.num_local_experts):
                expert_prefix = f"{prefix}.block_sparse_moe.experts.{expert_idx}"
                dst_expert_prefix = f"{dst_prefix}.block_sparse_moe.experts.{expert_idx}"
                aoa_statements.extend(
                    [
                        f"{expert_prefix}.w1.weight^T -> {dst_expert_prefix}.w1.weight",
                        f"{expert_prefix}.w2.weight^T -> {dst_expert_prefix}.w2.weight",
                        f"{expert_prefix}.w3.weight^T -> {dst_expert_prefix}.w3.weight",
                    ]
                )

        # lm_head
        if cls != cls.base_model_class:
            if config.tie_word_embeddings:
                aoa_statements.append("model.embed_tokens.weight -> lm_head.weight")
            else:
                # GeneralLMHead stores weights in [vocab_size, hidden_size],
                # which is already the layout used by Hugging Face/vLLM.
                aoa_statements.append("lm_head.weight -> lm_head.weight")

        return {"aoa_statements": aoa_statements}

    @classmethod
    def _gen_inv_aoa_config(cls, config: MiniMaxConfig):
        """AOA config: mapping from PaddleFormers model key back to HF safetensors key."""
        model_prefix = "model." if cls != cls.base_model_class else ""

        aoa_statements = [
            f"{model_prefix}embed_tokens.weight -> model.embed_tokens.weight",
            f"{model_prefix}norm.weight -> model.norm.weight",
        ]

        for layer_idx in range(config.num_hidden_layers):
            layer_type = config.layer_types[layer_idx]
            prefix = f"model.layers.{layer_idx}"
            dst_prefix = f"{model_prefix}layers.{layer_idx}"

            aoa_statements.extend(
                [
                    f"{dst_prefix}.input_layernorm.weight -> {prefix}.input_layernorm.weight",
                    f"{dst_prefix}.post_attention_layernorm.weight -> {prefix}.post_attention_layernorm.weight",
                ]
            )

            if layer_type == "full_attention":
                aoa_statements.extend(
                    [
                        f"{dst_prefix}.self_attn.q_proj.weight^T -> {prefix}.self_attn.q_proj.weight",
                        f"{dst_prefix}.self_attn.k_proj.weight^T -> {prefix}.self_attn.k_proj.weight",
                        f"{dst_prefix}.self_attn.v_proj.weight^T -> {prefix}.self_attn.v_proj.weight",
                        f"{dst_prefix}.self_attn.o_proj.weight^T -> {prefix}.self_attn.o_proj.weight",
                    ]
                )
            elif layer_type == "linear_attention":
                aoa_statements.extend(
                    [
                        f"{dst_prefix}.self_attn.qkv_proj.weight^T -> {prefix}.self_attn.qkv_proj.weight",
                        f"{dst_prefix}.self_attn.output_gate.weight^T -> {prefix}.self_attn.output_gate.weight",
                        f"{dst_prefix}.self_attn.out_proj.weight^T -> {prefix}.self_attn.out_proj.weight",
                        f"{dst_prefix}.self_attn.norm.weight -> {prefix}.self_attn.norm.weight",
                    ]
                )

            aoa_statements.append(
                f"{dst_prefix}.block_sparse_moe.gate.weight^T -> {prefix}.block_sparse_moe.gate.weight"
            )
            for expert_idx in range(config.num_local_experts):
                expert_prefix = f"{prefix}.block_sparse_moe.experts.{expert_idx}"
                dst_expert_prefix = f"{dst_prefix}.block_sparse_moe.experts.{expert_idx}"
                aoa_statements.extend(
                    [
                        f"{dst_expert_prefix}.w1.weight^T -> {expert_prefix}.w1.weight",
                        f"{dst_expert_prefix}.w2.weight^T -> {expert_prefix}.w2.weight",
                        f"{dst_expert_prefix}.w3.weight^T -> {expert_prefix}.w3.weight",
                    ]
                )

        if not config.tie_word_embeddings and cls != cls.base_model_class:
            aoa_statements.append("lm_head.weight -> lm_head.weight")

        return {"aoa_statements": aoa_statements}


@register_base_model
class MiniMaxModel(MiniMaxPretrainedModel):
    """The bare MiniMax (Text-01) decoder model."""

    def __init__(self, config: MiniMaxConfig):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        self.embed_tokens = GeneralEmbedding.create(
            config=config,
            num_embeddings=self.vocab_size,
            embedding_dim=self.hidden_size,
            padding_idx=self.padding_idx,
        )
        self.layers = nn.LayerList(
            [MiniMaxDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=config.hidden_size,
            norm_eps=config.rms_norm_eps,
            input_is_parallel=config.sequence_parallel,
        )
        self.rotary_emb = MiniMaxRotaryEmbedding(config=config)

    @paddle.jit.not_to_static
    def recompute_training_full(
        self,
        layer_module: nn.Layer,
        hidden_states: paddle.Tensor,
        attention_mask: paddle.Tensor | None,
        position_ids: paddle.Tensor | None,
        position_embeddings: tuple[paddle.Tensor, paddle.Tensor] | None,
        past_key_values: Cache | None,
        use_cache: bool,
        attn_mask_startend_row_indices: paddle.Tensor | None,
    ) -> paddle.Tensor:
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        return recompute(
            create_custom_forward(layer_module),
            hidden_states,
            attention_mask,
            position_ids,
            position_embeddings,
            past_key_values,
            use_cache,
            attn_mask_startend_row_indices,
            use_reentrant=self.config.recompute_use_reentrant,
        )

    def forward(
        self,
        input_ids: paddle.Tensor | None = None,
        attention_mask: paddle.Tensor | None = None,
        position_ids: paddle.Tensor | None = None,
        past_key_values: MiniMaxCache | None = None,
        inputs_embeds: paddle.Tensor | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        attn_mask_startend_row_indices: paddle.Tensor | None = None,
        **kwargs,
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) and (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if (input_ids is not None) and (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds (not both)")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids).astype(self.embed_tokens.weight.dtype)

        bsz, seq_length, _ = inputs_embeds.shape

        if use_cache and past_key_values is None:
            past_key_values = MiniMaxCache(config=self.config)
        elif use_cache and not isinstance(past_key_values, MiniMaxCache):
            raise ValueError(
                f"MiniMax uses cache of its own and is not compatible with `past_key_values` of type {type(past_key_values)}."
            )

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = paddle.arange(
                past_seen_tokens, seq_length + past_seen_tokens, dtype=paddle.int64
            ).unsqueeze(0)
            position_ids = position_ids.expand([bsz, -1])

        mask_kwargs = {
            "config": self.config,
            "inputs_embeds": inputs_embeds,
            "batch_size": bsz,
            "seq_length": seq_length,
            "cache_length": past_key_values.get_seq_length() if past_key_values is not None else 0,
            "attention_mask": attention_mask,
            "attn_mask_startend_row_indices": attn_mask_startend_row_indices,
            "prepare_decoder_attention_mask": self._prepare_decoder_attention_mask,
        }
        causal_mask, attn_mask_startend_row_indices = create_causal_mask_and_row_indices(**mask_kwargs)
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds
        all_hidden_states = [] if output_hidden_states else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            if self.config.layer_types[idx] == "full_attention":
                input_attention_mask = causal_mask
                input_mask_startend = attn_mask_startend_row_indices
            else:
                input_attention_mask = attention_mask
                input_mask_startend = attn_mask_startend_row_indices

            has_gradient = not hidden_states.stop_gradient
            if (
                self.config.recompute_granularity == "full"
                and self.config.recompute_method == "uniform"
                and self.config.recompute_num_layers == 1
                and has_gradient
            ):
                hidden_states = self.recompute_training_full(
                    layer_module=decoder_layer,
                    hidden_states=hidden_states,
                    attention_mask=input_attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    attn_mask_startend_row_indices=input_mask_startend,
                )
            else:
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=input_attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    attn_mask_startend_row_indices=input_mask_startend,
                    **kwargs,
                )

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        if not return_dict:
            outputs = (hidden_states,)
            if output_hidden_states:
                outputs = outputs + (tuple(all_hidden_states) if all_hidden_states else None,)
            if use_cache:
                outputs = outputs + (past_key_values,)
            return outputs

        return MoEModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=tuple(all_hidden_states) if all_hidden_states else None,
        )


class MiniMaxForCausalLM(MiniMaxPretrainedModel):
    """MiniMax (Text-01) model with a language modeling head."""

    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config: MiniMaxConfig):
        super().__init__(config)
        self.config = config
        self.model = MiniMaxModel(config)
        self.lm_head = GeneralLMHead(config)
        self.criterion = CriterionLayer(config)
        self.tie_weights()

    def forward(
        self,
        input_ids: paddle.Tensor,
        position_ids: paddle.Tensor | None = None,
        attention_mask: paddle.Tensor | None = None,
        attn_mask_startend_row_indices: paddle.Tensor | None = None,
        inputs_embeds: paddle.Tensor | None = None,
        labels: paddle.Tensor | None = None,
        loss_mask: paddle.Tensor | None = None,
        use_cache: bool = False,
        past_key_values: Cache | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        return_dict: bool = False,
        **kwargs,
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        if output_router_logits:
            raise NotImplementedError("MiniMax does not support output_router_logits or router auxiliary loss yet.")

        if attention_mask is not None and attention_mask.dtype != paddle.bool:
            attention_mask = paddle.cast(attention_mask, paddle.bool)

        if attn_mask_startend_row_indices is not None and attention_mask is not None:
            logger.warning(
                "You have provided both attn_mask_startend_row_indices and attention_mask. "
                "The attn_mask_startend_row_indices will be used."
            )
            attention_mask = None

        outputs = self.model(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            past_key_values=past_key_values,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss, _ = self.criterion(logits, labels, loss_mask)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return MoECausalLMOutputWithPast(
            loss=loss,
            aux_loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )


__all__ = [
    "MiniMaxConfig",
    "MiniMaxPretrainedModel",
    "MiniMaxModel",
    "MiniMaxForCausalLM",
]
