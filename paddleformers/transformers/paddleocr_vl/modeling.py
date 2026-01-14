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


"""Paddle PaddleOCR-VL model."""

from contextvars import ContextVar
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import paddle
from paddle import nn
from paddle.distributed.fleet.utils import recompute
from paddle.distributed.fleet.utils.sequence_parallel_utils import (
    ScatterOp,
    mark_as_sequence_parallel_parameter,
)
from paddle.incubate.nn.functional import fused_rotary_position_embedding as fused_rope

from ...generation import GenerationMixin
from ...nn.activation import ACT2FN
from ...nn.attention.interface import ALL_ATTENTION_FUNCTIONS
from ...nn.criterion.interface import CriterionLayer
from ...nn.embedding import Embedding as GeneralEmbedding
from ...nn.linear import Linear as GeneralLinear
from ...nn.lm_head import LMHead as GeneralLMHead
from ...nn.mlp import MLP as Ernie4_5MLP
from ...nn.norm import Norm as GeneralNorm
from ...utils.log import logger
from ..cache_utils import Cache, DynamicCache
from ..masking_utils import create_causal_mask_and_row_indices
from ..model_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPooling,
    ModelOutput,
)
from ..model_utils import PretrainedModel, register_base_model
from ..modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from ..tensor_parallel_utils import model_parallel_dropout
from .configuration import PaddleOCRVisionConfig, PaddleOCRVLConfig


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.concat((-x2, x1), axis=-1)


def _ensure_cos_sin_dim(cos, sin, dim_needed):
    last = cos.shape[-1]
    if last == dim_needed:
        return cos, sin
    elif last * 2 == dim_needed:
        cos = paddle.concat([cos, cos], axis=-1)
        sin = paddle.concat([sin, sin], axis=-1)
        return cos, sin
    else:
        raise ValueError(f"Unexpected cos/sin last-dim: {last}, expected {dim_needed} or {dim_needed//2}")


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors (https://qwenlm.github.io/blog/qwen2-vl/)."""
    mrope_section = mrope_section * 2
    cos = paddle.concat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, axis=-1))], axis=-1).unsqueeze(
        unsqueeze_dim
    )
    sin = paddle.concat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, axis=-1))], axis=-1).unsqueeze(
        unsqueeze_dim
    )

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb_vision(q, k, cos, sin):
    """Applies Rotary Position Embedding to the query and key tensors."""
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    with paddle.amp.auto_cast(False):
        q, k = q.astype(dtype="float32"), k.astype(dtype="float32")
        cos, sin = cos.unsqueeze(-2).astype(dtype="float32"), sin.unsqueeze(-2).astype(dtype="float32")
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed.astype(orig_q_dtype), k_embed.astype(orig_k_dtype)


def apply_fused_rope(query_states, key_states, rope_theta):
    # b h l d -> b l h d
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    _, _, num_heads, _ = query_states.shape
    _, kv_seq_len, num_key_value_heads, _ = key_states.shape
    if num_heads != num_key_value_heads:
        query_states, _, _ = fused_rope(query_states, None, None, rotary_emb_base=rope_theta)
        key_states, _, _ = fused_rope(key_states, None, None, rotary_emb_base=rope_theta)
    else:
        query_states, key_states, _ = fused_rope(
            query_states,
            key_states,
            None,
            rotary_emb_base=rope_theta,
        )
    return query_states.transpose(1, 2), key_states.transpose(1, 2)


class PaddleOCRAttention(nn.Layer):
    def __init__(self, config: PaddleOCRVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        assert self.head_dim * self.num_heads == self.embed_dim
        self.scale = self.head_dim**-0.5
        self.dropout = getattr(config, "attention_dropout", 0.0)
        self.is_causal = False

        self.q_proj = GeneralLinear.create(
            self.embed_dim,
            self.embed_dim,
            config=config,
            tp_plan="colwise",
        )
        self.k_proj = GeneralLinear.create(
            self.embed_dim,
            self.embed_dim,
            config=config,
            tp_plan="colwise",
        )
        self.v_proj = GeneralLinear.create(
            self.embed_dim,
            self.embed_dim,
            config=config,
            tp_plan="colwise",
        )
        self.out_proj = GeneralLinear.create(
            self.embed_dim,
            self.embed_dim,
            config=config,
            tp_plan="rowwise",
        )

    def forward(
        self,
        hidden_states: paddle.Tensor,  # [B, L, D]
        attention_mask: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = False,
        cu_seqlens: Optional[List[paddle.Tensor]] = None,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        rope_emb: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,  # (cos, sin)
    ):

        B, L, D = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        # [B, L, H, Dh]
        q = q.reshape([B, L, self.num_heads, self.head_dim])
        k = k.reshape([B, L, self.num_heads, self.head_dim])
        v = v.reshape([B, L, self.num_heads, self.head_dim])
        if rope_emb is not None:
            cos, sin = rope_emb
            q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        q = q.transpose(2, 1)
        k = k.transpose(2, 1)
        v = v.transpose(2, 1)

        attn_output, attn_weights = attention_interface(
            self,
            query=q,
            key=k,
            value=v,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            dropout=self.config.get("attention_dropout_prob", 0.0) if self.training else 0.0,
            scaling=self.scale,
            is_causal=self.is_causal,
        )

        attn_output = attn_output.reshape([B, L, D])

        attn_output = self.out_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights


class PaddleOCRVisionEmbeddings(nn.Layer):
    def __init__(self, config: PaddleOCRVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size  # 1152
        self.image_size = config.image_size  # 384
        self.patch_size = config.patch_size  # 14

        # Note：Paddle should use "VALID" or 0
        self.patch_embedding = nn.Conv2D(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="VALID",
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2  # 729
        self.num_positions = self.num_patches
        self.cache_position_embedding = dict()
        self.cache_position_count = dict()
        self.position_embedding = GeneralEmbedding.create(
            config=config, num_embeddings=self.num_positions, embedding_dim=self.embed_dim
        )

        self.register_buffer(
            "position_ids",
            paddle.arange(self.num_positions).unsqueeze(0),
            persistable=False,
        )

    def interpolate_pos_encoding(self, embeddings, height: int, width: int, is_after_patchify: bool = False):

        num_positions = self.position_embedding.weight.shape[0]

        patch_pos_embed = self.position_embedding.weight.unsqueeze(0)

        dim = embeddings.shape[-1]

        if is_after_patchify:
            new_height = height
            new_width = width
        else:
            new_height = height // self.patch_size
            new_width = width // self.patch_size

        sqrt_num_positions = paddle.to_tensor(num_positions**0.5, dtype=paddle.int64)
        patch_pos_embed = patch_pos_embed.reshape((1, sqrt_num_positions, sqrt_num_positions, dim))
        patch_pos_embed = patch_pos_embed.transpose((0, 3, 1, 2))

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,
        )

        patch_pos_embed = patch_pos_embed.transpose((0, 2, 3, 1)).reshape((1, -1, dim))
        return patch_pos_embed

    @staticmethod
    def flatten_list(image_grid_thw):
        tmp_image_grid_thw = list()
        for image_grid in image_grid_thw:
            if isinstance(image_grid, list):
                tmp_image_grid_thw.extend(image_grid)
            else:
                tmp_image_grid_thw.append(image_grid)
        return tmp_image_grid_thw

    def fetch_position_embedding_lfu_cache(self, embeddings, h, w, max_cache=20):
        grid = (h, w)
        if grid in self.cache_position_embedding:
            self.cache_position_count[grid] += 1
            return self.cache_position_embedding[grid]

        if len(self.cache_position_embedding) >= max_cache:
            min_hit_grid = min(self.cache_position_count, key=self.cache_position_count.get)
            self.cache_position_count.pop(min_hit_grid)
            self.cache_position_embedding.pop(min_hit_grid)

        position_embedding = self.interpolate_pos_encoding(embeddings, h, w, True)
        self.cache_position_count[grid] = 1
        self.cache_position_embedding[grid] = position_embedding
        return position_embedding

    def forward(
        self,
        pixel_values: paddle.Tensor,  # [B, L, C, H, W]
        position_ids: Optional[paddle.Tensor] = None,  # [B or 1, S]
        image_grid_thw: Optional[List[Union[Tuple[int, int, int], List[Tuple[int, int, int]]]]] = None,
        interpolate_pos_encoding: bool = False,
    ) -> paddle.Tensor:
        if pixel_values.dim() == 5:
            assert position_ids is not None

            batch_size, squence_len, channel, height, width = pixel_values.shape
            target_dtype = self.patch_embedding.weight.dtype
            pixel_values = pixel_values.reshape(batch_size * squence_len, channel, height, width)
            patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
            embeddings = patch_embeds.flatten(-2).squeeze(-1)
            embeddings = embeddings.reshape(batch_size, squence_len, -1)

            flatten_image_grid_thw = self.flatten_list(image_grid_thw)
            assert sum([np.prod(x) for x in flatten_image_grid_thw]) == embeddings.shape[1], (
                flatten_image_grid_thw,
                embeddings.shape,
            )

            start = 0
            embeddings = embeddings.squeeze(0)
            tmp_embeddings = list()
            for image_grid in image_grid_thw:
                t, h, w = image_grid
                end = start + t * h * w
                image_embeddings = embeddings[int(start) : int(end), :]
                position_embedding = (
                    self.interpolate_pos_encoding(image_embeddings, h, w, True).squeeze(0).tile((t, 1))
                )
                image_embeddings = image_embeddings + position_embedding
                tmp_embeddings.append(image_embeddings)
                start = end
            embeddings = paddle.concat(tmp_embeddings, axis=0).unsqueeze(0)
            return embeddings
        else:
            raise NotImplementedError(str(pixel_values.shape))


class PaddleOCRMLP(nn.Layer):
    def __init__(self, config: PaddleOCRVisionConfig):
        super().__init__()
        self.config = config
        self.act_fn = ACT2FN[config.hidden_act]
        self.fc1 = GeneralLinear.create(
            config.hidden_size,
            config.intermediate_size,
            config=config,
            tp_plan="colwise",
        )
        self.fc2 = GeneralLinear.create(
            config.intermediate_size,
            config.hidden_size,
            config=config,
            tp_plan="rowwise",
        )

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class PaddleOCREncoderLayer(nn.Layer):
    def __init__(self, config: PaddleOCRVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.layer_norm1 = GeneralNorm.create(
            config=config,
            norm_type="layer_norm",
            hidden_size=self.embed_dim,
            has_bias=False,
            norm_eps=config.layer_norm_eps,
            input_is_parallel=False,
        )
        self.self_attn = PaddleOCRAttention(config)
        self.layer_norm2 = GeneralNorm.create(
            config=config,
            norm_type="layer_norm",
            hidden_size=self.embed_dim,
            has_bias=False,
            norm_eps=config.layer_norm_eps,
            input_is_parallel=config.sequence_parallel,
        )
        self.mlp = PaddleOCRMLP(config)

        if config.sequence_parallel:
            if not hasattr(config, "disable_ffn_model_parallel"):
                self.layer_norm1.enable_sequence_parallel()

    def forward(
        self,
        hidden_states,
        attention_mask,
        output_attentions=False,
        cu_seqlens=None,
        attn_mask_startend_row_indices=None,
        rope_emb=None,
    ):

        residual = hidden_states

        ln1_out = self.layer_norm1(hidden_states)

        x, attn_w = self.self_attn(
            hidden_states=ln1_out,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            cu_seqlens=cu_seqlens,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            rope_emb=rope_emb,
        )

        hs_post_attn = residual + x

        residual = hs_post_attn
        ln2_out = self.layer_norm2(residual)

        mlp_out = self.mlp(ln2_out)

        hidden_states_out = residual + mlp_out

        outputs = (hidden_states_out,)
        if output_attentions:
            outputs += (attn_w,)
        return outputs


class PaddleOCRVisionRotaryEmbedding(nn.Layer):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.rope_init()

    def rope_init(self):
        arange = paddle.arange(0, self.dim, 2, dtype="float32")
        inv_freq = 1.0 / (self.theta ** (arange / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistable=False)

    def forward(self, seqlen: int) -> paddle.Tensor:
        seq = paddle.arange(seqlen, dtype=self.inv_freq.dtype)
        freqs = paddle.outer(seq, self.inv_freq)
        return freqs


class PaddleOCREncoder(nn.Layer):
    def __init__(self, config: PaddleOCRVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads
        self.layers = nn.LayerList([PaddleOCREncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.rotary_pos_emb = PaddleOCRVisionRotaryEmbedding(head_dim // 2)

    @staticmethod
    def flatten_list(image_grid_thw):
        tmp_image_grid_thw = list()
        for image_grid in image_grid_thw:
            if isinstance(image_grid, list):
                tmp_image_grid_thw.extend(image_grid)
            else:
                tmp_image_grid_thw.append(image_grid)
        return tmp_image_grid_thw

    def build_window_index(self, image_grid, window_size):
        """
        返回：
          window_indices: int64 [sum(t*h*w_valid)]
          cu_seqlens_within_windows: int32 [num_windows_total*t]
        """

        window_indices = list()
        pad_values = -100
        start_window_index = 0
        cu_seqlens_within_windows = list()

        for t, h, w in map(int, image_grid):
            window_index = paddle.arange(t * h * w).reshape((t, h, w))
            pad_h = (-h) % window_size
            pad_w = (-w) % window_size
            assert pad_h >= 0 and pad_w >= 0, (pad_h, pad_w)
            window_index = nn.functional.pad(window_index, (0, pad_w, 0, pad_h), value=pad_values)
            h, w = h // window_size, w // window_size
            window_index = window_index.reshape([t, h, window_size, w, window_size])
            window_index = window_index.transpose([0, 1, 3, 2, 4])
            window_index = window_index.reshape([t, h * w, window_size * window_size])

            window_seqlens = (window_index != pad_values).long().sum(-1).reshape(-1)
            window_index = window_index.reshape(-1)
            window_index = window_index[window_index != pad_values]
            window_indices.append(window_index + start_window_index)
            cu_seqlens_within_windows.append(window_seqlens.cumsum(0) + start_window_index)
            start_window_index += t * h * w
        window_indices = paddle.concat(window_indices, axis=0)
        cu_seqlens_within_windows = paddle.concat(cu_seqlens_within_windows, axis=0)
        cu_seqlens_within_windows = nn.functional.pad(cu_seqlens_within_windows, (1, 0), value=0).astype("int32")
        return window_indices, cu_seqlens_within_windows

    @paddle.jit.not_to_static
    def recompute_training(
        self,
        layer_module,
        hidden_states,
        attention_mask,
        output_attentions=False,
        cu_seqlens=None,
        attn_mask_startend_row_indices=None,
        rope_emb=None,
    ):
        """Perform gradient checkpointing for memory-efficient training.

        Args:
            layer_module (nn.Layer): Transformer layer to recompute
            hidden_states (paddle.Tensor): Input hidden states
            attention_mask (paddle.Tensor): Attention mask
            output_attentions (bool): Whether to output attention weights
            cu_seqlens (List[paddle.Tensor]):
            attn_mask_startend_row_indices (paddle.Tensor): Variable length indices
            rope_emb (paddle.Tensor): RoPE Position embeddings

        Returns:
            paddle.Tensor: Output hidden states after recomputation
        """

        hidden_states = recompute(
            layer_module,
            hidden_states,
            attention_mask,
            output_attentions=output_attentions,
            cu_seqlens=cu_seqlens,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            rope_emb=rope_emb,
        )
        return hidden_states

    def forward(
        self,
        inputs_embeds: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cu_seqlens: Optional[paddle.Tensor] = None,
        image_grid_thw: Optional[List[Union[Tuple[int, int, int], List[Tuple[int, int, int]]]]] = None,
        height_position_ids: Optional[paddle.Tensor] = None,
        width_position_ids: Optional[paddle.Tensor] = None,
        use_rope: Optional[bool] = False,
        window_size: Optional[int] = -1,
        vision_or_text: str = "vision",
    ):

        vision_or_text = "vision"
        assert vision_or_text in ["vision", "text"]
        use_window_attn = window_size > 0 and vision_or_text == "vision"
        use_rope = (use_rope is True) and (vision_or_text == "vision")
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        hidden_states = inputs_embeds
        attention_mask = attention_mask.to(inputs_embeds.dtype) if attention_mask is not None else None

        if use_rope is True:
            flatten_image_grid_thw = self.flatten_list(image_grid_thw)
            assert sum([np.prod(x) for x in flatten_image_grid_thw]) == hidden_states.shape[1], (
                flatten_image_grid_thw,
                hidden_states.shape,
            )

            if width_position_ids is None or height_position_ids is None:
                split_hids = list()
                split_wids = list()
                for t, h, w in flatten_image_grid_thw:
                    t, h, w = map(int, (t, h, w))
                    image_pids = paddle.arange(t * h * w) % (h * w)
                    sample_hids = image_pids // w
                    sample_wids = image_pids % w
                    split_hids.append(sample_hids)
                    split_wids.append(sample_wids)
                width_position_ids = paddle.concat(split_wids, axis=0)
                height_position_ids = paddle.concat(split_hids, axis=0)

            window_indices, cu_seqlens_within_windows = None, None

            if use_window_attn:
                window_indices, cu_seqlens_within_windows = self.build_window_index(
                    flatten_image_grid_thw, window_size
                )
                reversed_window_indices = window_indices.argsort()
                height_position_ids = height_position_ids[window_indices]
                width_position_ids = width_position_ids[window_indices]

            pids = paddle.stack([height_position_ids, width_position_ids], axis=-1).astype(paddle.int64)
            max_grid_size = pids.max() + 1
            rope_emb_max_grid = self.rotary_pos_emb(max_grid_size)

            rope_emb = rope_emb_max_grid[pids].flatten(1)

            rope_emb = rope_emb.tile((1, 2))
            rope_emb = (rope_emb.cos(), rope_emb.sin())

        else:
            rope_emb = None

            window_indices, cu_seqlens_within_windows = None, None

            if use_window_attn:
                flatten_image_grid_thw = self.flatten_list(image_grid_thw)
                assert (
                    sum([np.prod(x.astype("float32").cpu().numpy()) for x in flatten_image_grid_thw])
                    == hidden_states.shape[1]
                ), (flatten_image_grid_thw, hidden_states.shape)

                window_indices, cu_seqlens_within_windows = self.build_window_index(
                    flatten_image_grid_thw, window_size
                )
                reversed_window_indices = window_indices.argsort()

        if use_window_attn:
            assert cu_seqlens_within_windows is not None
            attn_cu_seqlens = cu_seqlens_within_windows
            hidden_states = hidden_states[:, window_indices, :]
        else:
            attn_cu_seqlens = cu_seqlens

        if cu_seqlens is not None and attention_mask is None:
            cu_seqlens_rm_first = cu_seqlens[1:]
            cu_seqlens_rm_last = cu_seqlens[:-1]
            repeats = cu_seqlens_rm_first - cu_seqlens_rm_last

            startend_row_indices_lts = paddle.repeat_interleave(cu_seqlens_rm_first, repeats).reshape([1, 1, -1, 1])
            startend_row_indices_ute = paddle.repeat_interleave(cu_seqlens_rm_last, repeats).reshape([1, 1, -1, 1])
            startend_row_indices = paddle.concat([startend_row_indices_lts, startend_row_indices_ute], axis=-1)

        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (
                    (hidden_states[:, reversed_window_indices, :],) if use_window_attn else (hidden_states,)
                )
            has_gradient = not hidden_states.stop_gradient
            if (
                self.config.recompute_granularity == "full"
                and self.config.recompute_method == "uniform"
                and self.config.recompute_num_layers == 1
                and has_gradient
            ):
                layer_outputs = self.recompute_training(
                    encoder_layer,
                    hidden_states,
                    attention_mask,
                    output_attentions=output_attentions,
                    cu_seqlens=attn_cu_seqlens,
                    attn_mask_startend_row_indices=startend_row_indices,
                    rope_emb=rope_emb,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=output_attentions,
                    cu_seqlens=attn_cu_seqlens,
                    attn_mask_startend_row_indices=startend_row_indices,
                    rope_emb=rope_emb,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if use_window_attn:
            hidden_states = hidden_states[:, reversed_window_indices, :]
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )


class PaddleOCRVisionTransformer(nn.Layer):
    def __init__(self, config: PaddleOCRVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = PaddleOCRVisionEmbeddings(config)
        self.encoder = PaddleOCREncoder(config)
        self.post_layernorm = GeneralNorm.create(
            config=config,
            norm_type="layer_norm",
            hidden_size=embed_dim,
            has_bias=False,
            norm_eps=config.layer_norm_eps,
            input_is_parallel=config.sequence_parallel,
        )

    def forward(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = False,
        attention_mask=None,
        sample_indices=None,
        image_indices=None,
        position_ids=None,
        height_position_ids=None,
        width_position_ids=None,
        cu_seqlens=None,
        padding_mask=None,
        vision_return_embed_list: Optional[bool] = False,
        image_grid_thw: Optional[List[Union[Tuple[int, int, int], List[Tuple[int, int, int]]]]] = None,
        return_pooler_output: Optional[bool] = True,
        use_rope: Optional[bool] = False,
        window_size: Optional[bool] = -1,
    ) -> BaseModelOutputWithPooling:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        hidden_states = self.embeddings(
            pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
            position_ids=position_ids,
            image_grid_thw=image_grid_thw,
        )

        encoder_outputs: BaseModelOutput = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            attention_mask=attention_mask,
            cu_seqlens=cu_seqlens,
            image_grid_thw=image_grid_thw,
            use_rope=use_rope,
            height_position_ids=height_position_ids,
            width_position_ids=width_position_ids,
            window_size=window_size,
            vision_or_text="vision",
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = self.post_layernorm(last_hidden_state)

        if return_pooler_output is True:
            if sample_indices is not None:
                assert self.use_head is True
                dim = last_hidden_state.shape[-1]
                sample_hidden_state_list = list()

                hidden_state = last_hidden_state.squeeze(0)
                sample_index = sample_indices
                unique_sample_index = paddle.unique(sample_index).sort().values.unbind(0)
                unique_sample_index = list(unique_sample_index)
                if len(unique_sample_index) > 0 and unique_sample_index[0] == -1:
                    unique_sample_index = unique_sample_index[1:]
                for sample_idx in unique_sample_index:
                    token_indices = (sample_index == sample_idx).nonzero().flatten()
                    sample_hidden_state = hidden_state[token_indices]
                    sample_hidden_state_list.append(sample_hidden_state)

                if not vision_return_embed_list:
                    max_length = max([_state.shape[0] for _state in sample_hidden_state_list])
                    tmp_sample_hidden_state_list = list()
                    padding_mask = list()
                    for idx, _state in enumerate(sample_hidden_state_list):
                        padding_length = max_length - _state.shape[0]
                        mask = _state.new_zeros(size=(max_length,), dtype=paddle.int64)
                        mask[-padding_length:] = 1
                        padding_mask.append(mask)
                        padding = _state.new_zeros(size=(padding_length, dim))
                        new_state = paddle.concat([_state, padding], axis=0)
                        tmp_sample_hidden_state_list.append(new_state)
                    sample_hidden_state = paddle.stack(tmp_sample_hidden_state_list, axis=0)
                    padding_mask = paddle.stack(padding_mask, axis=0).astype("float32").to(last_hidden_state.dtype)
                    pooler_output = self.head(sample_hidden_state, key_padding_mask=padding_mask)
                else:
                    pooler_output = list()
                    for state in sample_hidden_state_list:
                        sample_pooler_output = self.head(state.unsqueeze(0))
                        pooler_output.append(sample_pooler_output)
                    pooler_output = paddle.concat(pooler_output, axis=0)
                    sample_hidden_state = sample_hidden_state_list

                return BaseModelOutputWithPooling(
                    last_hidden_state=sample_hidden_state,
                    pooler_output=pooler_output,
                    hidden_states=encoder_outputs.hidden_states,
                    attentions=encoder_outputs.attentions,
                )
            else:
                pooler_output = self.head(last_hidden_state) if self.use_head else None

            return BaseModelOutputWithPooling(
                last_hidden_state=last_hidden_state,
                pooler_output=pooler_output,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions,
            )

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=None,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class PaddleOCRVisionPreTrainedModel(PretrainedModel):
    """Base class for PaddleOCR pretrained models."""

    config_class = PaddleOCRVisionConfig
    base_model_prefix = "paddleocr"

    _no_split_modules = [
        "PaddleOCREncoderLayer",
        "PaddleOCRVisionEmbeddings",
    ]

    transpose_weight_keys = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]


@register_base_model
class PaddleOCRVisionModel(PaddleOCRVisionPreTrainedModel):
    config_class = PaddleOCRVisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: PaddleOCRVisionConfig):
        super().__init__(config)

        self.vision_model = PaddleOCRVisionTransformer(config)

    def get_input_embeddings(self) -> nn.Layer:
        return self.vision_model.embeddings.patch_embedding

    def forward(
        self,
        pixel_values,
        sample_indices=None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
        position_ids=None,
        vision_return_embed_list: Optional[bool] = False,
        image_grid_thw: Optional[List[Union[Tuple[int, int, int], List[Tuple[int, int, int]]]]] = None,
        cu_seqlens=None,
        return_pooler_output: Optional[bool] = True,
        use_rope: Optional[bool] = False,
        window_size: Optional[bool] = -1,
    ) -> BaseModelOutputWithPooling:
        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            position_ids=position_ids,
            vision_return_embed_list=vision_return_embed_list,
            image_grid_thw=image_grid_thw,
            sample_indices=sample_indices,
            cu_seqlens=cu_seqlens,
            return_pooler_output=return_pooler_output,
            use_rope=use_rope,
            window_size=window_size,
        )


class Projector(nn.Layer):
    def __init__(self, text_config: PaddleOCRVLConfig, vision_config: PaddleOCRVisionConfig):
        super().__init__()
        self.text_config = text_config
        self.vision_config = vision_config
        self.merge_kernel_size = (2, 2)

        self.hidden_size = self.vision_config.hidden_size * self.merge_kernel_size[0] * self.merge_kernel_size[1]

        self.pre_norm = GeneralNorm.create(
            config=vision_config,
            norm_type="layer_norm",
            hidden_size=self.vision_config.hidden_size,
            has_bias=False,
            norm_eps=1e-05,
            input_is_parallel=vision_config.sequence_parallel,
        )
        self.linear_1 = GeneralLinear.create(
            self.hidden_size,
            self.hidden_size,
            has_bias=True,
            config=text_config,
        )
        self.act = ACT2FN["gelu"]
        self.linear_2 = GeneralLinear.create(
            self.hidden_size,
            self.text_config.hidden_size,
            has_bias=True,
            config=text_config,
        )

    def forward(self, image_features, image_grid_thw):

        image_features_chunks = image_features.split(image_grid_thw.prod(axis=1).tolist(), axis=1)
        m1, m2 = self.merge_kernel_size

        processed_features = list()
        for image_feature, image_grid in zip(image_features_chunks, image_grid_thw):
            image_feature = image_feature.squeeze(0)
            image_feature = self.pre_norm(image_feature)  # shape: (T*H*W, D)
            t, h, w = image_grid
            d = image_feature.shape[-1]
            h_block = h // m1
            w_block = w // m2

            image_feature = image_feature.reshape([t, h_block, m1, w_block, m2, d])
            image_feature = image_feature.transpose([0, 1, 3, 2, 4, 5])
            image_feature = image_feature.reshape([t * h_block * w_block, m1 * m2 * d])

            hidden_states = self.linear_1(image_feature)
            hidden_states = self.act(hidden_states)
            hidden_states = self.linear_2(hidden_states)
            processed_features.append(hidden_states)

        return paddle.concat(processed_features, axis=0)


class PaddleOCRRotaryEmbedding(nn.Layer):
    def __init__(self, config: PaddleOCRVLConfig):
        super().__init__()
        self.config = config
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        rope_parameters = self.config.rope_parameters
        self.rope_type = rope_parameters.get("rope_type", rope_parameters.get("type", "default"))
        rope_init_fn = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config)

        self.register_buffer("inv_freq", inv_freq, persistable=False)
        self.original_inv_freq = inv_freq

    @staticmethod
    def compute_default_rope_parameters(
        config: Optional[PaddleOCRVLConfig] = None,
        seq_len: Optional[int] = None,
    ) -> tuple["paddle.Tensor", float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`PreTrainedConfig`]):
                The model configuration.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
        Returns:
            Tuple of (`paddle.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (base ** (paddle.arange(0, dim, 2, dtype=paddle.int64).astype(dtype=paddle.float32) / dim))
        return inv_freq, attention_factor

    @dynamic_rope_update
    @paddle.no_grad()
    def forward(self, x, position_ids):
        # Core RoPE block. In contrast to other models, PaddleOCR-VL has different position ids for the grids
        # So we expand the inv_freq to shape (3, ...)
        with paddle.amp.auto_cast(enable=False):
            inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand([3, position_ids.shape[1], -1, 1])

            position_ids_expanded = position_ids[:, :, None, :].float()

            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)

            emb = paddle.concat((freqs, freqs), axis=-1)

            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Ernie4_5Attention(nn.Layer):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, layer_idx=0):
        """Initialize the attention layer.

        Args:
            config (PaddleOCRVLConfig): Model configuration.
            layer_idx (int, optional): Index in transformer stack. Defaults to 0.
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.rope_scaling = config.rope_scaling
        self.is_causal = True

        if config.tensor_model_parallel_size > 1:
            assert (
                self.num_heads % config.tensor_model_parallel_size == 0
            ), f"num_heads: {self.num_heads}, tensor_model_parallel_size: {config.tensor_model_parallel_size}"
            self.num_heads = self.num_heads // config.tensor_model_parallel_size

            assert (
                self.num_key_value_heads % config.tensor_model_parallel_size == 0
            ), f"num_heads: {self.num_key_value_heads}, tensor_model_parallel_size: {config.tensor_model_parallel_size}"
            self.num_key_value_heads = self.num_key_value_heads // config.tensor_model_parallel_size

        logger.warning_once(f"use GQA - num_heads: {self.num_heads}- num_key_value_heads: {self.num_key_value_heads}")
        assert (
            self.num_heads % self.num_key_value_heads == 0
        ), f"num_heads: {self.num_heads}, num_key_value_heads: {self.num_key_value_heads}"
        kv_hidden_size = self.head_dim * config.num_key_value_heads
        q_hidden_size = self.head_dim * config.num_attention_heads

        self.q_proj = GeneralLinear.create(
            self.hidden_size,
            q_hidden_size,
            has_bias=config.use_bias,
            config=config,
            tp_plan="colwise",
        )
        self.k_proj = GeneralLinear.create(
            self.hidden_size,
            kv_hidden_size,
            has_bias=config.use_bias,
            config=config,
            tp_plan="colwise",
        )
        self.v_proj = GeneralLinear.create(
            self.hidden_size,
            kv_hidden_size,
            has_bias=config.use_bias,
            config=config,
            tp_plan="colwise",
        )

        self.o_proj = GeneralLinear.create(
            q_hidden_size,
            self.hidden_size,
            has_bias=config.use_bias,
            config=config,
            tp_plan="rowwise",
        )

        self.config = config
        self.scaling = self.head_dim**-0.5

    def forward(
        self,
        hidden_states,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        position_embeddings: Optional[Tuple[paddle.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        """Compute attention outputs.

        Args:
            hidden_states (paddle.Tensor): Input tensor [bsz, seq_len, hidden_size]
            past_key_values (Optional[Tuple[Cache]]): Cached key/value states
            attention_mask (Optional[paddle.Tensor]): Attention mask tensor
            attn_mask_startend_row_indices (Optional[paddle.Tensor]): Variable length attention indices
            position_ids (Optional[paddle.Tensor]): Position indices for RoPE
            output_attentions (bool): Return attention weights if True
            use_cache (bool): Cache key/value states if True

        Returns:
            Tuple containing:
                - attention_output: [bsz, seq_len, hidden_size]
                - attention_weights: Optional attention probabilities
                - updated_key_value_cache: Optional updated cache
        """
        if self.config.sequence_parallel:
            max_sequence_length = self.config.max_sequence_length
            bsz = hidden_states.shape[0] * self.config.tensor_model_parallel_size // max_sequence_length
            q_len = max_sequence_length
        else:
            bsz, q_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states).reshape([bsz, q_len, -1, self.head_dim]).transpose(2, 1)
        key_states = self.k_proj(hidden_states).reshape([bsz, q_len, -1, self.head_dim]).transpose(2, 1)
        value_states = self.v_proj(hidden_states).reshape([bsz, q_len, -1, self.head_dim]).transpose(2, 1)
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        if self.config.apply_rope_fusion:
            query_states, key_states = apply_fused_rope(query_states, key_states, self.config.rope_theta)
        else:
            cos, sin = position_embeddings
            query_states, key_states = apply_multimodal_rotary_pos_emb(
                query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
            )

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attn_output, attn_weights = attention_interface(
            self,
            query=query_states,
            key=key_states,
            value=value_states,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            dropout=self.config.get("attention_dropout_prob", 0.0) if self.training else 0.0,
            scaling=self.scaling,
            is_causal=self.is_causal,
        )

        if self.config.sequence_parallel:
            attn_output = attn_output.reshape([-1, attn_output.shape[-1]])
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_values


class Ernie4_5DecoderLayer(nn.Layer):
    """A single transformer decoder layer in ERNIE model.

    Contains self-attention and feed-forward components,
    support, residual connections, and layer normalization.
    """

    def __init__(self, config, layer_idx):
        """Initialize the decoder layer.

        Args:
            config (PaddleOCRVLConfig): Model configuration.
            layer_idx (int): Index of this layer in the transformer stack
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.config = config
        self.self_attn = Ernie4_5Attention(config, layer_idx)
        self.mlp = Ernie4_5MLP(config)
        self.input_layernorm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=config.hidden_size,
            has_bias=config.use_bias,
            norm_eps=self.config.rms_norm_eps,
            input_is_parallel=False,
        )
        self.post_attention_layernorm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=config.hidden_size,
            has_bias=config.use_bias,
            norm_eps=self.config.rms_norm_eps,
            input_is_parallel=config.sequence_parallel,
        )

        self.hidden_dropout = nn.Dropout(p=config.hidden_dropout_prob, mode="upscale_in_train")

        if config.sequence_parallel:
            if not hasattr(config, "disable_ffn_model_parallel"):
                self.input_layernorm.enable_sequence_parallel()
                if config.use_bias:
                    mark_as_sequence_parallel_parameter(self.self_attn.o_proj.bias)
                    mark_as_sequence_parallel_parameter(self.mlp.down_proj.bias)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        position_embeddings: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = False,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
    ) -> Tuple[paddle.Tensor, Optional[Tuple[paddle.Tensor, paddle.Tensor]]]:
        """Forward pass through the decoder layer.

        Args:
            hidden_states (paddle.Tensor): Input tensor [batch_size, seq_len, hidden_size]
            attention_mask (Optional[paddle.Tensor]): Attention mask tensor
            attn_mask_startend_row_indices (Optional[paddle.Tensor]): Indices for variable length attention
            position_ids (Optional[paddle.Tensor]): Position indices for rotary embeddings
            position_embeddings (Optional[paddle.Tensor]): Position embeddings tensor
            output_attentions (Optional[bool]): Whether to return attention weights
            past_key_values (Optional[Cache]]): Cached key/value states
            use_cache (Optional[bool]): Whether to cache key/value states

        Returns:
            Union: Various output combinations depending on arguments:
                - Base case: Hidden states tensor
                - With attention: Tuple of (hidden_states, attention_weights)
                - With cache: Tuple of (hidden_states, cached_key_value)
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        with model_parallel_dropout(self.config):
            hidden_states = self.hidden_dropout(hidden_states) + residual

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        with model_parallel_dropout(self.config):
            hidden_states = self.hidden_dropout(hidden_states) + residual

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        # remove empty tuple for pipeline parallel
        if type(outputs) is tuple and len(outputs) == 1:
            outputs = outputs[0]
        return outputs


class Ernie4_5PretrainedModel(PretrainedModel):
    """Base class for ERNIE pretrained models."""

    config_class = PaddleOCRVLConfig
    base_model_prefix = "model"
    transpose_weight_keys = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "out_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "linear_1",
        "linear_2",
        "fc1",
        "fc2",
    ]

    @classmethod
    def _gen_aoa_config(cls, config: PaddleOCRVLConfig):

        aoa_config = {
            "aoa_statements": [],
        }

        # language model
        llm_prefix = "model."
        aoa_config["aoa_statements"] += [
            f"model.embed_tokens.weight -> {llm_prefix}embed_tokens.weight",
            f"model.layers.$LAYER_ID.self_attn.o_proj.weight^T -> {llm_prefix}layers.$LAYER_ID.self_attn.o_proj.weight",
            f"model.layers.$LAYER_ID.mlp.down_proj.weight^T -> {llm_prefix}layers.$LAYER_ID.mlp.down_proj.weight",
            f"model.layers.$LAYER_ID.input_layernorm.weight -> {llm_prefix}layers.$LAYER_ID.input_layernorm.weight",
            f"model.layers.$LAYER_ID.post_attention_layernorm.weight -> {llm_prefix}layers.$LAYER_ID.post_attention_layernorm.weight",
            f"model.norm.weight -> {llm_prefix}norm.weight",
        ]

        aoa_config["aoa_statements"] += [
            f"model.layers.$LAYER_ID.self_attn.{x}_proj.weight^T -> {llm_prefix}layers.$LAYER_ID.self_attn.{x}_proj.weight"
            for x in ("q", "k", "v")
        ]

        aoa_config["aoa_statements"] += [
            f"model.layers.$LAYER_ID.mlp.{x}_proj.weight^T -> {llm_prefix}layers.$LAYER_ID.mlp.{x}_proj.weight"
            for x in ("gate", "up")
        ]

        # visual model
        visual_prefix = "visual.vision_model."
        aoa_config["aoa_statements"] += [
            f"visual.vision_model.embeddings.patch_embedding.weight -> {visual_prefix}embeddings.patch_embedding.weight",
            f"visual.vision_model.embeddings.patch_embedding.bias -> {visual_prefix}embeddings.patch_embedding.bias",
            f"visual.vision_model.embeddings.position_embedding.weight -> {visual_prefix}embeddings.position_embedding.weight",
            f"visual.vision_model.encoder.layers.$LAYER_ID.self_attn.out_proj.weight^T -> {visual_prefix}encoder.layers.$LAYER_ID.self_attn.out_proj.weight",
            f"visual.vision_model.encoder.layers.$LAYER_ID.self_attn.out_proj.bias -> {visual_prefix}encoder.layers.$LAYER_ID.self_attn.out_proj.bias",
            f"visual.vision_model.encoder.layers.$LAYER_ID.layer_norm1.weight -> {visual_prefix}encoder.layers.$LAYER_ID.layer_norm1.weight",
            f"visual.vision_model.encoder.layers.$LAYER_ID.layer_norm1.bias -> {visual_prefix}encoder.layers.$LAYER_ID.layer_norm1.bias",
            f"visual.vision_model.encoder.layers.$LAYER_ID.layer_norm2.weight -> {visual_prefix}encoder.layers.$LAYER_ID.layer_norm2.weight",
            f"visual.vision_model.encoder.layers.$LAYER_ID.layer_norm2.bias -> {visual_prefix}encoder.layers.$LAYER_ID.layer_norm2.bias",
            f"visual.vision_model.post_layernorm.weight -> {visual_prefix}post_layernorm.weight",
            f"visual.vision_model.post_layernorm.bias -> {visual_prefix}post_layernorm.bias",
        ]

        aoa_config["aoa_statements"] += [
            f"visual.vision_model.encoder.layers.$LAYER_ID.self_attn.{x}_proj.weight^T -> {visual_prefix}encoder.layers.$LAYER_ID.self_attn.{x}_proj.weight"
            for x in ("q", "k", "v")
        ]
        aoa_config["aoa_statements"] += [
            f"visual.vision_model.encoder.layers.$LAYER_ID.self_attn.{x}_proj.bias -> {visual_prefix}encoder.layers.$LAYER_ID.self_attn.{x}_proj.bias"
            for x in ("q", "k", "v")
        ]

        aoa_config["aoa_statements"] += [
            f"visual.vision_model.encoder.layers.$LAYER_ID.mlp.{x}.weight^T -> {visual_prefix}encoder.layers.$LAYER_ID.mlp.{x}.weight"
            for x in ("fc1", "fc2")
        ]
        aoa_config["aoa_statements"] += [
            f"visual.vision_model.encoder.layers.$LAYER_ID.mlp.{x}.bias -> {visual_prefix}encoder.layers.$LAYER_ID.mlp.{x}.bias"
            for x in ("fc1", "fc2")
        ]

        # projector
        projector_prefix = "mlp_AR."
        aoa_config["aoa_statements"] += [
            f"mlp_AR.pre_norm.weight -> {projector_prefix}pre_norm.weight",
            f"mlp_AR.pre_norm.bias -> {projector_prefix}pre_norm.bias",
        ]
        aoa_config["aoa_statements"] += [
            f"mlp_AR.{x}.weight^T -> {projector_prefix}{x}.weight" for x in ("linear_1", "linear_2")
        ]
        aoa_config["aoa_statements"] += [
            f"mlp_AR.{x}.bias -> {projector_prefix}{x}.bias" for x in ("linear_1", "linear_2")
        ]

        # lm_head
        aoa_config["aoa_statements"] += [
            f"{'model.embed_tokens.weight^T' if config.tie_word_embeddings else 'lm_head.weight'} -> lm_head.weight",
        ]

        return aoa_config

    @classmethod
    def _gen_inv_aoa_config(cls, config: PaddleOCRVLConfig):

        aoa_config = {
            "aoa_statements": [],
        }

        # language model
        llm_prefix = "model."
        aoa_config["aoa_statements"] += [
            f"{llm_prefix}embed_tokens.weight -> model.embed_tokens.weight",
            f"{llm_prefix}layers.$LAYER_ID.self_attn.o_proj.weight^T -> model.layers.$LAYER_ID.self_attn.o_proj.weight",
            f"{llm_prefix}layers.$LAYER_ID.mlp.down_proj.weight^T -> model.layers.$LAYER_ID.mlp.down_proj.weight",
            f"{llm_prefix}layers.$LAYER_ID.input_layernorm.weight -> model.layers.$LAYER_ID.input_layernorm.weight",
            f"{llm_prefix}layers.$LAYER_ID.post_attention_layernorm.weight -> model.layers.$LAYER_ID.post_attention_layernorm.weight",
            f"{llm_prefix}norm.weight -> model.norm.weight",
        ]

        aoa_config["aoa_statements"] += [
            f"{llm_prefix}layers.$LAYER_ID.self_attn.{x}_proj.weight^T -> model.layers.$LAYER_ID.self_attn.{x}_proj.weight"
            for x in ("q", "k", "v")
        ]

        aoa_config["aoa_statements"] += [
            f"{llm_prefix}layers.$LAYER_ID.mlp.{x}_proj.weight^T -> model.layers.$LAYER_ID.mlp.{x}_proj.weight"
            for x in ("gate", "up")
        ]

        # visual model
        visual_prefix = "visual.vision_model."
        aoa_config["aoa_statements"] += [
            f"{visual_prefix}embeddings.patch_embedding.weight -> visual.vision_model.embeddings.patch_embedding.weight",
            f"{visual_prefix}embeddings.patch_embedding.bias -> visual.vision_model.embeddings.patch_embedding.bias",
            f"{visual_prefix}embeddings.position_embedding.weight -> visual.vision_model.embeddings.position_embedding.weight",
            f"{visual_prefix}encoder.layers.$LAYER_ID.self_attn.out_proj.weight^T -> visual.vision_model.encoder.layers.$LAYER_ID.self_attn.out_proj.weight",
            f"{visual_prefix}encoder.layers.$LAYER_ID.self_attn.out_proj.bias -> visual.vision_model.encoder.layers.$LAYER_ID.self_attn.out_proj.bias",
            f"{visual_prefix}encoder.layers.$LAYER_ID.layer_norm1.weight -> visual.vision_model.encoder.layers.$LAYER_ID.layer_norm1.weight",
            f"{visual_prefix}encoder.layers.$LAYER_ID.layer_norm1.bias -> visual.vision_model.encoder.layers.$LAYER_ID.layer_norm1.bias",
            f"{visual_prefix}encoder.layers.$LAYER_ID.layer_norm2.weight -> visual.vision_model.encoder.layers.$LAYER_ID.layer_norm2.weight",
            f"{visual_prefix}encoder.layers.$LAYER_ID.layer_norm2.bias -> visual.vision_model.encoder.layers.$LAYER_ID.layer_norm2.bias",
            f"{visual_prefix}post_layernorm.weight -> visual.vision_model.post_layernorm.weight",
            f"{visual_prefix}post_layernorm.bias -> visual.vision_model.post_layernorm.bias",
        ]

        aoa_config["aoa_statements"] += [
            f"{visual_prefix}encoder.layers.$LAYER_ID.self_attn.{x}_proj.weight^T -> visual.vision_model.encoder.layers.$LAYER_ID.self_attn.{x}_proj.weight"
            for x in ("q", "k", "v")
        ]
        aoa_config["aoa_statements"] += [
            f"{visual_prefix}encoder.layers.$LAYER_ID.self_attn.{x}_proj.bias -> visual.vision_model.encoder.layers.$LAYER_ID.self_attn.{x}_proj.bias"
            for x in ("q", "k", "v")
        ]

        aoa_config["aoa_statements"] += [
            f"{visual_prefix}encoder.layers.$LAYER_ID.mlp.{x}.weight^T -> visual.vision_model.encoder.layers.$LAYER_ID.mlp.{x}.weight"
            for x in ("fc1", "fc2")
        ]
        aoa_config["aoa_statements"] += [
            f"{visual_prefix}encoder.layers.$LAYER_ID.mlp.{x}.bias -> visual.vision_model.encoder.layers.$LAYER_ID.mlp.{x}.bias"
            for x in ("fc1", "fc2")
        ]

        # projector
        projector_prefix = "mlp_AR."
        aoa_config["aoa_statements"] += [
            f"{projector_prefix}pre_norm.weight -> mlp_AR.pre_norm.weight",
            f"{projector_prefix}pre_norm.bias -> mlp_AR.pre_norm.bias",
        ]
        aoa_config["aoa_statements"] += [
            f"{projector_prefix}{x}.weight^T -> mlp_AR.{x}.weight" for x in ("linear_1", "linear_2")
        ]
        aoa_config["aoa_statements"] += [
            f"{projector_prefix}{x}.bias -> mlp_AR.{x}.bias" for x in ("linear_1", "linear_2")
        ]

        # lm_head
        aoa_config["aoa_statements"] += [
            f"lm_head.weight -> {'_' if config.tie_word_embeddings else 'lm_head.weight'}",
        ]

        return aoa_config


@register_base_model
class Ernie4_5Model(Ernie4_5PretrainedModel):
    """The core ERNIE transformer model"""

    def __init__(self, config: PaddleOCRVLConfig):
        """Initialize the ERNIE model architecture.

        Args:
            config (PaddleOCRVLConfig): Model configuration.
        """
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.config = config
        self.embed_tokens = GeneralEmbedding.create(
            config=config, num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
        )
        self.layers = nn.LayerList([Ernie4_5DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=config.hidden_size,
            has_bias=config.use_bias,
            norm_eps=self.config.rms_norm_eps,
            input_is_parallel=config.sequence_parallel,
        )

        self.rotary_emb = PaddleOCRRotaryEmbedding(config=config)

    @paddle.jit.not_to_static
    def recompute_training(
        self,
        layer_module,
        hidden_states,
        attention_mask,
        attn_mask_startend_row_indices,
        position_ids,
        position_embeddings,
        output_attentions,
        past_key_values,
        use_cache,
    ):
        """Perform gradient checkpointing for memory-efficient training.

        Args:
            layer_module (nn.Layer): Transformer layer to recompute
            hidden_states (paddle.Tensor): Input hidden states
            attention_mask (paddle.Tensor): Attention mask
            attn_mask_startend_row_indices (paddle.Tensor): Variable length indices
            position_ids (paddle.Tensor): Position indices
            position_embeddings (paddle.Tensor): Position embeddings
            output_attentions (bool): Whether to output attention weights
            past_key_values (Optional[Cache]): Cached key/value states
            use_cache (bool): Whether to cache key/value states

        Returns:
            paddle.Tensor: Output hidden states after recomputation
        """

        hidden_states = recompute(
            layer_module,
            hidden_states,
            attention_mask,
            attn_mask_startend_row_indices,
            position_ids,
            position_embeddings,
            output_attentions,
            past_key_values,
            use_cache,
        )
        return hidden_states

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        attn_mask_startend_row_indices=None,
        inputs_embeds=None,
        use_cache=None,
        past_key_values=None,
        output_attentions=False,
        output_hidden_states=None,
        return_dict=False,
    ):
        """Forward pass through the ERNIE model.

        Args:
            input_ids (Optional[paddle.Tensor]): Input token IDs
            position_ids (Optional[paddle.Tensor]): Position indices
            attention_mask (Optional[paddle.Tensor]): Attention mask
            attn_mask_startend_row_indices (Optional[paddle.Tensor]): Variable length attention indices
            inputs_embeds (Optional[paddle.Tensor]): Precomputed embeddings
            use_cache (Optional[bool]): Whether to cache key/value states
            past_key_values (Optional[Cache]]): Cached key/value states
            output_attentions (Optional[bool]): Whether to output attention weights
            output_hidden_states (Optional[bool]): Whether to output all hidden states
            return_dict (Optional[bool]): Whether to return dict or tuple

        Returns:
            Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
                Various outputs depending on configuration, including:
                - last_hidden_state: Final layer hidden states
                - past_key_values: Cached key/value states if use_cache=True
                - hidden_states: All hidden states if output_hidden_states=True
                - attentions: Attention weights if output_attentions=True
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            bsz, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            bsz, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        kv_seq_len = past_key_values.get_seq_length() if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self.config.sequence_parallel:
            inputs_embeds = inputs_embeds.reshape([-1, inputs_embeds.shape[-1]])
            inputs_embeds = ScatterOp.apply(inputs_embeds)

        hidden_states = inputs_embeds

        mask_kwargs = {
            "config": self.config,
            "inputs_embeds": inputs_embeds,
            "batch_size": bsz,
            "seq_length": seq_length,
            "cache_length": kv_seq_len,
            "attention_mask": attention_mask,
            "attn_mask_startend_row_indices": attn_mask_startend_row_indices,
            "prepare_decoder_attention_mask": self._prepare_decoder_attention_mask,
        }

        causal_attention_mask, attn_mask_startend_row_indices = create_causal_mask_and_row_indices(**mask_kwargs)

        if position_ids is None:
            position_ids = paddle.arange(kv_seq_len, seq_length).unsqueeze(0).tile((bsz, 1))

        if not self.config.apply_rope_fusion:
            position_embeddings = self.rotary_emb(hidden_states, position_ids)  # cos and sin
        else:
            position_embeddings = None

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for idx, (decoder_layer) in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            has_gradient = not hidden_states.stop_gradient
            if (
                self.config.recompute_granularity == "full"
                and self.config.recompute_method == "uniform"
                and self.config.recompute_num_layers == 1
                and has_gradient
            ):
                layer_outputs = self.recompute_training(
                    decoder_layer,
                    hidden_states,
                    causal_attention_mask,
                    attn_mask_startend_row_indices,
                    position_ids,
                    position_embeddings,
                    output_attentions,
                    past_key_values,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    causal_attention_mask,
                    attn_mask_startend_row_indices,
                    position_ids,
                    position_embeddings,
                    output_attentions,
                    past_key_values,
                    use_cache,
                )

            if isinstance(layer_outputs, (tuple, list)):
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    past_key_values,
                    all_hidden_states,
                    all_self_attns,
                ]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=None,
        )


@dataclass
class PaddleOCRVLCausalLMOutputWithPast(ModelOutput):
    loss: Optional[paddle.Tensor] = None
    logits: paddle.Tensor = None
    past_key_values: Optional[List[paddle.Tensor]] = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None
    attentions: Optional[Tuple[paddle.Tensor]] = None
    rope_deltas: Optional[paddle.Tensor] = None


class PaddleOCRVLModel(Ernie4_5PretrainedModel):
    config_class = PaddleOCRVLConfig

    def __init__(self, config: PaddleOCRVLConfig):
        super().__init__(config)

        raise NotImplementedError("PaddleOCRVLModel is not implemented yet")


class PaddleOCRVLForConditionalGeneration(Ernie4_5PretrainedModel, GenerationMixin):
    config_class = PaddleOCRVLConfig
    base_model_prefix = "model"
    _no_split_modules = ["Ernie4_5DecoderLayer", "PaddleOCREncoderLayer"]
    _tied_weights_keys = ["lm_head.weight"]
    _keys_to_ignore_on_load_unexpected = ["packing_position_embedding", "vision_model.head"]

    def __init__(self, config: PaddleOCRVLConfig):
        super().__init__(config)

        self.mlp_AR = Projector(config, config.vision_config)
        self.visual = PaddleOCRVisionModel(config.vision_config)
        self.model = Ernie4_5Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = GeneralLMHead(config)
        self.criterion = CriterionLayer(config)
        self.rope_deltas_var = ContextVar("rope_deltas", default=None)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def freeze_vision(self):
        for p in self.visual.vision_model.parameters():
            p.stop_gradient = True

    def get_rope_index(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        image_grid_thw: Optional[paddle.Tensor] = None,
        video_grid_thw: Optional[paddle.Tensor] = None,
        second_per_grid_ts: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embedding for text part.
            Examples:
                Temporal (Time): 3 patches, representing different segments of the video in time.
                Height: 2 patches, dividing each frame vertically.
                Width: 2 patches, dividing each frame horizontally.
                We also have some important parameters:
                fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
                tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
                temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
                interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [101, 102, 103, 104, 105]
                text height position_ids: [101, 102, 103, 104, 105]
                text width position_ids: [101, 102, 103, 104, 105]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`paddle.Tensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`paddle.Tensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`paddle.Tensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            second_per_grid_ts (`paddle.Tensor` of shape `(num_videos)`, *optional*):
                The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
            attention_mask (`paddle.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`paddle.Tensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`paddle.Tensor` of shape `(batch_size)`)
        """
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = paddle.ones_like(total_input_ids)
            position_ids = paddle.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
            )
            image_index, video_index = 0, 0
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = paddle.nonzero(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        second_per_grid_t = 0
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image

                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        if second_per_grid_ts is not None:
                            second_per_grid_t = second_per_grid_ts[video_index]
                        else:
                            second_per_grid_t = 1.0
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(paddle.arange(text_len).reshape((1, -1)).expand((3, -1)) + st_idx)

                    if paddle.is_tensor(second_per_grid_t):
                        second_per_grid_t = second_per_grid_t.detach().item()
                    range_tensor = paddle.arange(llm_grid_t).reshape((-1, 1))
                    expanded_range = range_tensor.expand((-1, llm_grid_h * llm_grid_w))

                    time_tensor = expanded_range * second_per_grid_t * self.config.vision_config.tokens_per_second

                    time_tensor_long = time_tensor.astype("int64")
                    t_index = time_tensor_long.flatten()

                    h_index = (
                        paddle.arange(llm_grid_h).reshape((1, -1, 1)).expand((llm_grid_t, -1, llm_grid_w)).flatten()
                    )
                    w_index = (
                        paddle.arange(llm_grid_w).reshape((1, 1, -1)).expand((llm_grid_t, llm_grid_h, -1)).flatten()
                    )
                    llm_pos_ids_list.append(paddle.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(paddle.arange(text_len).reshape((1, -1)).expand((3, -1)) + st_idx)

                llm_positions = paddle.concat(llm_pos_ids_list, axis=1).reshape((3, -1))
                position_ids[..., i, attention_mask[i] == 1] = llm_positions
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = paddle.to_tensor(mrope_position_deltas).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand((3, -1, -1))
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    paddle.arange(input_ids.shape[1]).reshape((1, 1, -1)).expand((3, input_ids.shape[0], -1))
                )
                mrope_position_deltas = paddle.zeros(
                    [input_ids.shape[0], 1],
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

    def prepare_attention_mask_for_generation(self, input_ids, pad_token_id, eos_token_id):
        """Avoid using attention_mask with flash_attn on generation."""
        if self.config._attn_implementation == "sdpa":
            return None
        return super().prepare_attention_mask_for_generation(input_ids, pad_token_id, eos_token_id)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model
        batch_size, seq_length = input_ids.shape
        if past_key_values is None:
            cache_position = paddle.arange(input_ids.shape[1])
        else:
            cache_position = paddle.to_tensor([seq_length - 1])

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            use_cache=use_cache,
            **kwargs,
        )

        model_inputs["position_ids"] = None

        if cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

        return model_inputs

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[paddle.Tensor] = None,
        pixel_values_videos: Optional[paddle.Tensor] = None,
        image_grid_thw: Optional[paddle.Tensor] = None,
        video_grid_thw: Optional[paddle.Tensor] = None,
        rope_deltas: Optional[paddle.Tensor] = None,
        second_per_grid_ts: Optional[paddle.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, PaddleOCRVLCausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        curr_rope_deltas = self.rope_deltas_var.get()

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:

                pixel_values = pixel_values.astype(inputs_embeds.dtype)
                pixel_values = pixel_values.unsqueeze(0)
                siglip_position_ids = list()
                image_grid_hws = list()
                sample_indices = list()
                cu_seqlens = [0]

                for idx, thw in enumerate(image_grid_thw):
                    thw_tuple = tuple(thw.detach().cpu().numpy().tolist())
                    numel = np.prod(thw_tuple)
                    image_grid_hws.append(thw_tuple)
                    image_position_ids = paddle.arange(numel) % np.prod(thw_tuple[1:])
                    siglip_position_ids.append(image_position_ids)
                    sample_indices.append(paddle.full((numel,), idx, dtype=paddle.int64))
                    cu_seqlens.append(cu_seqlens[-1] + numel)

                siglip_position_ids = paddle.concat(siglip_position_ids, axis=0)
                cu_seqlens = paddle.to_tensor(cu_seqlens, dtype=paddle.int32)
                sample_indices = paddle.concat(sample_indices, axis=0)

                vision_outputs = self.visual(
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_hws,
                    position_ids=siglip_position_ids,
                    vision_return_embed_list=True,
                    interpolate_pos_encoding=True,
                    sample_indices=sample_indices,
                    cu_seqlens=cu_seqlens,
                    return_pooler_output=False,
                    use_rope=True,
                    window_size=-1,
                )
                image_embeds = vision_outputs.last_hidden_state

                image_embeds = self.mlp_AR(image_embeds, image_grid_thw)

                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded

                image_embeds = image_embeds.astype(inputs_embeds.dtype)

                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if attention_mask is not None and attention_mask.dtype != paddle.bool:
            attention_mask = paddle.cast(attention_mask, paddle.bool)

        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if curr_rope_deltas is None or past_key_values is None or past_key_values.get_seq_length() == 0:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
                self.rope_deltas_var.set(rope_deltas)
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (past_key_values.get_seq_length() + curr_rope_deltas)
                    if past_key_values is not None and past_key_values.get_seq_length() > 0
                    else 0
                )
                position_ids = paddle.arange(seq_length)
                position_ids = position_ids.reshape((1, -1)).expand((batch_size, -1))
                if (
                    past_key_values is not None and past_key_values.get_seq_length() > 0
                ):  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], axis=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand((3, -1, -1))

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss_mask = labels != -100
            loss, _ = self.criterion(logits, labels, loss_mask)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return PaddleOCRVLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=curr_rope_deltas,
        )

    def _get_image_nums_and_video_nums(
        self,
        input_ids: Optional[paddle.Tensor],
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """
        Get the number of images and videos for each sample to calculate the separation length of the sample tensor.
        These parameters are not passed through the processor to avoid unpredictable impacts from interface modifications.

        Args:
            input_ids (`paddle.Tensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.

        Returns:
            image_nums (`paddle.Tensor` of shape `(batch_size, num_images_sample)`)
            video_nums (`paddle.Tensor` of shape `(batch_size, num_videos_sample)`)
        """
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id

        vision_start_mask = input_ids == vision_start_token_id
        vision_first_mask = paddle.roll(vision_start_mask, shifts=1, axis=1)
        image_mask = input_ids == image_token_id
        video_mask = input_ids == video_token_id
        image_nums = paddle.sum(vision_first_mask & image_mask, axis=1)
        video_nums = paddle.sum(vision_first_mask & video_mask, axis=1)

        return image_nums, video_nums


__all__ = ["PaddleOCRVLForConditionalGeneration"]
