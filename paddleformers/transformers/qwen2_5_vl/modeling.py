# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""Paddle Qwen2_5_VL model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import paddle
import paddle.nn.functional as F
from paddle import Tensor, nn
from paddle.distributed.fleet.utils import recompute
from paddle.distributed.fleet.utils.sequence_parallel_utils import ScatterOp

from ...nn.attention.interface import ALL_ATTENTION_FUNCTIONS
from ...nn.criterion.interface import CriterionLayer
from ...nn.embedding import Embedding as GeneralEmbedding
from ...nn.linear import Linear as GeneralLinear
from ...nn.lm_head import LMHead as GeneralLMHead
from ...nn.mlp import MLP
from ...nn.norm import Norm as GeneralNorm
from ..cache_utils import Cache, DynamicCache
from ..masking_utils import (
    create_causal_mask_and_row_indices,
    create_sliding_window_causal_mask_and_row_indices,
)
from ..model_outputs import BaseModelOutputWithPast, ModelOutput
from ..model_utils import PretrainedModel, register_base_model
from ..modeling_rope_utils import ROPE_INIT_FUNCTIONS
from ..utils import logger
from .configuration import (
    Qwen2_5_VLConfig,
    Qwen2_5_VLTextConfig,
    Qwen2_5_VLVisionConfig,
)


class Qwen2_5_VLMLP(MLP):
    pass


class Qwen2_5_VisionPatchEmbed(nn.Layer):
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False)

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


class Qwen2_5_VisionRotaryEmbedding(nn.Layer):
    inv_freq: paddle.Tensor

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (paddle.arange(0, dim, 2, dtype=paddle.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistable=False)

    def forward(self, seqlen: int) -> paddle.Tensor:
        seq = paddle.arange(seqlen, dtype=self.inv_freq.dtype)
        freqs = paddle.outer(seq, self.inv_freq)
        return freqs


class Qwen2_5_VLPatchMerger(nn.Layer):
    def __init__(self, config: Qwen2_5_VLConfig, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = GeneralNorm.create(config, norm_type="rms_norm", hidden_size=context_dim, norm_eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.mlp(self.ln_q(x).view([-1, self.hidden_size]))
        return x


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.cat([-x2, x1], axis=-1)


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


class Qwen2_5_VLVisionAttention(nn.Layer):
    def __init__(self, config: Qwen2_5_VLVisionConfig) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1  # needed for eager attention
        self.qkv = GeneralLinear.create(
            self.dim,
            self.dim * 3,
            has_bias=True,
            linear_type="default",
        )
        self.proj = GeneralLinear.create(
            self.dim,
            self.dim,
            linear_type="default",
        )
        self.scaling = self.head_dim**-0.5
        self.config = config
        self.attention_dropout = 0.0
        self.is_causal = False

    def forward(
        self,
        hidden_states: paddle.Tensor,
        cu_seqlens: paddle.Tensor,
        rotary_pos_emb: Optional[paddle.Tensor] = None,
        position_embeddings: Optional[tuple[paddle.Tensor, paddle.Tensor]] = None,
        **kwargs,
    ) -> paddle.Tensor:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        splits = [
            paddle.split(tensor, lengths.tolist(), axis=2) for tensor in (query_states, key_states, value_states)
        ]
        attn_outputs = [
            attention_interface(
                self,
                q,
                k,
                v,
                attention_mask=None,
                attn_mask_startend_row_indices=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                is_causal=False,
                **kwargs,
            )[0]
            for q, k, v in zip(*splits)
        ]
        attn_output = paddle.cat(attn_outputs, axis=-2)

        attn_output = attn_output.reshape([seq_length, -1]).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen2_5_VLVisionBlock(nn.Layer):
    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        self.norm1 = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=config.hidden_size,
            norm_eps=1e-6,
        )
        self.norm2 = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=config.hidden_size,
            norm_eps=1e-6,
        )
        self.attn = Qwen2_5_VLVisionAttention(config=config)
        self.mlp = Qwen2_5_VLMLP(config, has_bias=True)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        cu_seqlens: paddle.Tensor,
        rotary_pos_emb: Optional[paddle.Tensor] = None,
        position_embeddings: Optional[tuple[paddle.Tensor, paddle.Tensor]] = None,
        **kwargs,
    ) -> paddle.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen2_5_VLPretrainedModel(PretrainedModel):
    config_class = Qwen2_5_VLConfig
    base_model_prefix = "model"
    input_modalities = ["image", "video", "text"]
    _no_split_modules = ["Qwen2_5_VLDecoderLayer", "Qwen2_5_VLVisionBlock"]
    _keys_to_ignore_on_load_unexpected = [r"self_attn.rotary_emb.inv_freq"]
    transpose_weight_keys = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "qkv",
        "gate_proj",
        "up_proj",
        "down_proj",
        "proj",
        "merger.mlp\.\d+",
    ]

    @classmethod
    def _gen_aoa_config(cls, config: Qwen2_5_VLConfig):
        mapping = cls._checkpoint_conversion_mapping
        llm_target = next((v for v in mapping.values() if "language_model" in v), "language_model")
        visual_target = next((v for v in mapping.values() if "visual" in v), "visual")
        llm_prefix = f"{llm_target}." if not llm_target.endswith(".") else llm_target
        visual_prefix = f"{visual_target}." if not visual_target.endswith(".") else visual_target

        # language model
        aoa_config = {
            "aoa_statements": [
                f"model.embed_tokens.weight -> {llm_prefix}embed_tokens.weight",
                f"model.norm.weight -> {llm_prefix}norm.weight",
                f"model.layers.$LAYER_ID.input_layernorm.weight -> {llm_prefix}layers.$LAYER_ID.input_layernorm.weight",
                f"model.layers.$LAYER_ID.post_attention_layernorm.weight -> {llm_prefix}layers.$LAYER_ID.post_attention_layernorm.weight",
                f"model.layers.$LAYER_ID.self_attn.o_proj.weight^T -> {llm_prefix}layers.$LAYER_ID.self_attn.o_proj.weight",
                f"model.layers.$LAYER_ID.mlp.down_proj.weight^T -> {llm_prefix}layers.$LAYER_ID.mlp.down_proj.weight",
            ]
        }

        # vision model
        aoa_config["aoa_statements"] += (
            [
                f"visual.blocks.$LAYER_ID.attn.{x}.weight^T -> {visual_prefix}blocks.$LAYER_ID.attn.{x}.weight"
                for x in ("qkv", "proj")
            ]
            + [
                f"visual.blocks.$LAYER_ID.attn.{x}.bias -> {visual_prefix}blocks.$LAYER_ID.attn.{x}.bias"
                for x in ("qkv", "proj")
            ]
            + [
                f"visual.blocks.$LAYER_ID.mlp.{x}_proj.weight^T -> {visual_prefix}blocks.$LAYER_ID.mlp.{x}_proj.weight"
                for x in ("up", "gate", "down")
            ]
            + [
                f"visual.blocks.$LAYER_ID.mlp.{x}_proj.bias -> {visual_prefix}blocks.$LAYER_ID.mlp.{x}_proj.bias"
                for x in ("up", "gate", "down")
            ]
        )
        aoa_config["aoa_statements"] += [
            f"visual.patch_embed.proj.weight -> {visual_prefix}patch_embed.proj.weight",
            f"visual.merger.ln_q.weight -> {visual_prefix}merger.ln_q.weight",
            f"visual.blocks.$LAYER_ID.norm1.weight -> {visual_prefix}blocks.$LAYER_ID.norm1.weight",
            f"visual.blocks.$LAYER_ID.norm2.weight -> {visual_prefix}blocks.$LAYER_ID.norm2.weight",
        ]
        aoa_config["aoa_statements"] += [
            f"visual.merger.mlp.{x}.weight^T -> {visual_prefix}merger.mlp.{x}.weight" for x in ("0", "2")
        ] + [f"visual.merger.mlp.{x}.bias -> {visual_prefix}merger.mlp.{x}.bias" for x in ("0", "2")]

        # attention qkv
        if not config.text_config.fuse_attention_qkv:
            aoa_config["aoa_statements"] += [
                f"model.layers.$LAYER_ID.self_attn.{x}_proj.weight^T -> {llm_prefix}layers.$LAYER_ID.self_attn.{x}_proj.weight"
                for x in ("q", "k", "v")
            ]
            aoa_config["aoa_statements"] += [
                f"model.layers.$LAYER_ID.self_attn.{x}_proj.bias -> {llm_prefix}layers.$LAYER_ID.self_attn.{x}_proj.bias"
                for x in ("q", "k", "v")
            ]
        else:
            aoa_config["aoa_statements"] += [
                f"model.layers.$LAYER_ID.self_attn.q_proj.weight^T, model.layers.$LAYER_ID.self_attn.k_proj.weight^T, model.layers.$LAYER_ID.self_attn.v_proj.weight^T -> {llm_prefix}layers.$LAYER_ID.self_attn.qkv_proj.weight, fused_qkv, num_heads={config.text_config.num_attention_heads}, num_key_value_groups={config.text_config.num_key_value_heads}",
                f"model.layers.$LAYER_ID.self_attn.q_proj.bias, model.layers.$LAYER_ID.self_attn.k_proj.bias, model.layers.$LAYER_ID.self_attn.v_proj.bias -> {llm_prefix}layers.$LAYER_ID.self_attn.qkv_proj.bias, fused_qkv, num_heads={config.text_config.num_attention_heads}, num_key_value_groups={config.text_config.num_key_value_heads}, axis=0",
            ]

        # FFN
        if not config.text_config.fuse_attention_ffn:
            aoa_config["aoa_statements"] += [
                f"model.layers.$LAYER_ID.mlp.{p}_proj.weight^T -> {llm_prefix}layers.$LAYER_ID.mlp.{p}_proj.weight"
                for p in ("gate", "up")
            ]
        else:
            aoa_config["aoa_statements"] += [
                f"model.layers.$LAYER_ID.mlp.gate_proj.weight^T, model.layers.$LAYER_ID.mlp.up_proj.weight^T -> {llm_prefix}layers.$LAYER_ID.mlp.up_gate_proj.weight, fused_ffn",
            ]

        # Qwen2_5_VLModel without lm_head
        if cls.base_model_prefix:
            aoa_config["aoa_statements"] += [
                f"{'model.embed_tokens.weight' if config.tie_word_embeddings else 'lm_head.weight'} -> lm_head.weight",
            ]

        return aoa_config

    @classmethod
    def _gen_inv_aoa_config(cls, config: Qwen2_5_VLConfig):
        mapping = cls._checkpoint_conversion_mapping
        llm_target = next((v for v in mapping.values() if "language_model" in v), "language_model")
        visual_target = next((v for v in mapping.values() if "visual" in v), "visual")
        llm_prefix = f"{llm_target}." if not llm_target.endswith(".") else llm_target
        visual_prefix = f"{visual_target}." if not visual_target.endswith(".") else visual_target

        # language model
        aoa_config = {
            "aoa_statements": [
                f"{llm_prefix}embed_tokens.weight -> model.embed_tokens.weight",
                f"{llm_prefix}norm.weight -> model.norm.weight",
                f"{llm_prefix}layers.$LAYER_ID.input_layernorm.weight -> model.layers.$LAYER_ID.input_layernorm.weight",
                f"{llm_prefix}layers.$LAYER_ID.post_attention_layernorm.weight -> model.layers.$LAYER_ID.post_attention_layernorm.weight",
                f"{llm_prefix}layers.$LAYER_ID.self_attn.o_proj.weight^T -> model.layers.$LAYER_ID.self_attn.o_proj.weight",
                f"{llm_prefix}layers.$LAYER_ID.mlp.down_proj.weight^T -> model.layers.$LAYER_ID.mlp.down_proj.weight",
            ]
        }

        # visual model
        aoa_config["aoa_statements"] += (
            [
                f"{visual_prefix}blocks.$LAYER_ID.attn.{x}.weight^T -> visual.blocks.$LAYER_ID.attn.{x}.weight"
                for x in ("qkv", "proj")
            ]
            + [
                f"{visual_prefix}blocks.$LAYER_ID.attn.{x}.bias -> visual.blocks.$LAYER_ID.attn.{x}.bias"
                for x in ("qkv", "proj")
            ]
            + [
                f"{visual_prefix}blocks.$LAYER_ID.mlp.{x}_proj.weight^T -> visual.blocks.$LAYER_ID.mlp.{x}_proj.weight"
                for x in ("up", "gate", "down")
            ]
            + [
                f"{visual_prefix}blocks.$LAYER_ID.mlp.{x}_proj.bias -> visual.blocks.$LAYER_ID.mlp.{x}_proj.bias"
                for x in ("up", "gate", "down")
            ]
        )
        aoa_config["aoa_statements"] += [
            f"{visual_prefix}patch_embed.proj.weight -> visual.patch_embed.proj.weight",
            f"{visual_prefix}merger.ln_q.weight -> visual.merger.ln_q.weight",
            f"{visual_prefix}blocks.$LAYER_ID.norm1.weight -> visual.blocks.$LAYER_ID.norm1.weight",
            f"{visual_prefix}blocks.$LAYER_ID.norm2.weight -> visual.blocks.$LAYER_ID.norm2.weight",
        ]
        aoa_config["aoa_statements"] += [
            f"{visual_prefix}merger.mlp.{x}.weight^T -> visual.merger.mlp.{x}.weight" for x in ("0", "2")
        ] + [f"{visual_prefix}merger.mlp.{x}.bias -> visual.merger.mlp.{x}.bias" for x in ("0", "2")]

        # attention qkv
        if not config.text_config.fuse_attention_qkv:
            aoa_config["aoa_statements"] += [
                f"{llm_prefix}layers.$LAYER_ID.self_attn.{x}_proj.weight^T -> model.layers.$LAYER_ID.self_attn.{x}_proj.weight"
                for x in ("q", "k", "v")
            ]
            aoa_config["aoa_statements"] += [
                f"{llm_prefix}layers.$LAYER_ID.self_attn.{x}_proj.bias -> model.layers.$LAYER_ID.self_attn.{x}_proj.bias"
                for x in ("q", "k", "v")
            ]
        else:
            aoa_config["aoa_statements"] += [
                f"{llm_prefix}layers.$LAYER_ID.self_attn.qkv_proj.weight  -> model.layers.$LAYER_ID.self_attn.q_proj.weight, model.layers.$LAYER_ID.self_attn.k_proj.weight, model.layers.$LAYER_ID.self_attn.v_proj.weight, fused_qkv, num_heads={config.text_config.num_attention_heads}, num_key_value_groups = {config.text_config.num_key_value_heads}",
                f"{llm_prefix}layers.$LAYER_ID.self_attn.qkv_proj.bias -> model.layers.$LAYER_ID.self_attn.q_proj.bias, model.layers.$LAYER_ID.self_attn.k_proj.bias, model.layers.$LAYER_ID.self_attn.v_proj.bias, fused_qkv, num_heads={config.text_config.num_attention_heads}, num_key_value_groups = {config.text_config.num_key_value_heads}, axis=0",
            ]
            aoa_config["aoa_statements"] += [
                f"model.layers.{layer_id}.self_attn.{x}_proj.weight^T -> model.layers.{layer_id}.self_attn.{x}_proj.weight"
                for layer_id in range(config.text_config.num_hidden_layers)
                for x in ("q", "k", "v")
            ]

        # FFN
        if not config.text_config.fuse_attention_ffn:
            aoa_config["aoa_statements"] += [
                f"{llm_prefix}layers.$LAYER_ID.mlp.{p}_proj.weight^T -> model.layers.$LAYER_ID.mlp.{p}_proj.weight"
                for p in ("gate", "up")
            ]
        else:
            aoa_config["aoa_statements"] += [
                f"{llm_prefix}layers.$LAYER_ID.mlp.up_gate_proj.weight -> model.layers.$LAYER_ID.mlp.gate_proj.weight, model.layers.$LAYER_ID.mlp.up_proj.weight, fused_ffn"
            ]
            aoa_config["aoa_statements"] += [
                f"model.layers.{layer_id}.mlp.{x}_proj.weight^T -> model.layers.{layer_id}.mlp.{x}_proj.weight"
                for layer_id in range(config.text_config.num_hidden_layers)
                for x in ("gate", "up")
            ]

        # Qwen2_5_VLModel without lm_head
        if cls.base_model_prefix:
            aoa_config["aoa_statements"] += [
                f"lm_head.weight -> {'_' if config.tie_word_embeddings else 'lm_head.weight'}",
            ]

        return aoa_config


class Qwen2_5_VisionTransformerPretrainedModel(Qwen2_5_VLPretrainedModel):
    config_class = Qwen2_5_VLVisionConfig
    _no_split_modules = ["Qwen2_5_VLVisionBlock"]

    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.fullatt_block_indexes = config.fullatt_block_indexes
        self.window_size = config.window_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = Qwen2_5_VisionPatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
        )

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.LayerList([Qwen2_5_VLVisionBlock(config) for _ in range(config.depth)])
        self.merger = Qwen2_5_VLPatchMerger(
            config=config,
            dim=config.out_hidden_size,
            context_dim=config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
        )
        self.gradient_checkpointing = False

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = paddle.arange(h).unsqueeze(1).expand([-1, w])
            hpos_ids = hpos_ids.reshape(
                [
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                ]
            )
            hpos_ids = hpos_ids.transpose(perm=[0, 2, 1, 3])
            hpos_ids = hpos_ids.flatten()

            wpos_ids = paddle.arange(w).unsqueeze(0).expand([h, -1])
            wpos_ids = wpos_ids.reshape(
                [
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                ]
            )
            wpos_ids = wpos_ids.transpose([0, 2, 1, 3])
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(paddle.stack(x=[hpos_ids, wpos_ids], axis=-1).tile(repeat_times=[t, 1]))
        pos_ids = paddle.cat(x=pos_ids, axis=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(start_axis=1)
        return rotary_pos_emb

    def get_window_index(self, grid_thw):
        window_index: list = []
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = self.window_size // self.spatial_merge_size // self.patch_size

        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (
                grid_h // self.spatial_merge_size,
                grid_w // self.spatial_merge_size,
            )
            index = paddle.arange(end=grid_t * llm_grid_h * llm_grid_w).reshape([grid_t, llm_grid_h, llm_grid_w])
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                [
                    grid_t,
                    num_windows_h,
                    vit_merger_window_size,
                    num_windows_w,
                    vit_merger_window_size,
                ]
            )
            index_padded = index_padded.transpose(perm=[0, 1, 3, 2, 4]).reshape(
                [
                    grid_t,
                    num_windows_h * num_windows_w,
                    vit_merger_window_size,
                    vit_merger_window_size,
                ]
            )
            seqlens = (index_padded != -100).sum(axis=[2, 3]).reshape([-1])
            index_padded = index_padded.reshape([-1])
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = seqlens.cumsum(axis=0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = paddle.cat(x=window_index, axis=0)

        return window_index, cu_window_seqlens

    @paddle.jit.not_to_static
    def recompute_training_full(
        self,
        layer_module: nn.Layer,
        hidden_states: paddle.Tensor,
        cu_seqlens: paddle.Tensor,
        rotary_pos_emb: Optional[paddle.Tensor] = None,
        position_embeddings: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
    ):
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        hidden_states = recompute(
            create_custom_forward(layer_module),
            hidden_states,
            cu_seqlens,
            rotary_pos_emb,
            position_embeddings,
        )
        return hidden_states

    def forward(self, hidden_states: paddle.Tensor, grid_thw: paddle.Tensor) -> paddle.Tensor:
        """
        Args:
            hidden_states (`paddle.Tensor` of shape `(batch_size, seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`paddle.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `paddle.Tensor`: hidden_states.
        """
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = paddle.to_tensor(data=cu_window_seqlens, dtype="int32", place=hidden_states.place)
        cu_window_seqlens = paddle.unique_consecutive(x=cu_window_seqlens)

        seq_len, _ = tuple(hidden_states.shape)
        hidden_states = hidden_states.reshape([seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1])
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape([seq_len, -1])
        rotary_pos_emb = rotary_pos_emb.reshape([seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1])
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape([seq_len, -1])
        emb = paddle.cat((rotary_pos_emb, rotary_pos_emb), axis=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = paddle.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            axis=0, dtype="int32"
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens

            has_gradient = not hidden_states.stop_gradient
            if (
                self.config.recompute_granularity == "full"
                and self.config.recompute_method == "uniform"
                and self.config.recompute_num_layers == 1
                and has_gradient
            ):
                hidden_states = self.recompute_training_full(
                    blk,
                    hidden_states,
                    cu_seqlens=cu_seqlens_now,
                    position_embeddings=position_embeddings,
                )
            else:
                hidden_states = blk(
                    hidden_states,
                    cu_seqlens=cu_seqlens_now,
                    position_embeddings=position_embeddings,
                )

        hidden_states = self.merger(hidden_states)
        reverse_indices = paddle.argsort(x=window_index)
        hidden_states = hidden_states[reverse_indices, :]

        return hidden_states


@dataclass
class Qwen2_5_VLModelOutputWithPast(ModelOutput):
    """
    Args:
        past_key_values (`Cache)`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(paddle.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `paddle.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
        attentions (`tuple(paddle.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        rope_deltas (`paddle.Tensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
    """

    last_hidden_state: Optional[paddle.Tensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None
    attentions: Optional[Tuple[paddle.Tensor]] = None
    rope_deltas: Optional[paddle.Tensor] = None


class Qwen2_5_VLRotaryEmbedding(nn.Layer):
    inv_freq: paddle.Tensor

    def __init__(self, config: Qwen2_5_VLTextConfig):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config

        rope_parameters = config.rope_parameters
        self.rope_type = rope_parameters.get("rope_type", rope_parameters.get("type", "default"))
        rope_init_fn = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config)

        self.register_buffer("inv_freq", inv_freq, persistable=False)
        self.original_inv_freq = inv_freq

    @staticmethod
    def compute_default_rope_parameters(
        config: Optional[Qwen2_5_VLTextConfig] = None,
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

    def forward(self, x, position_ids):
        with paddle.amp.auto_cast(enable=False):
            inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand([3, position_ids.shape[1], -1, 1])

            position_ids_expanded = position_ids[:, :, None, :].float()

            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)

            emb = paddle.concat((freqs, freqs), axis=-1)

            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen2MLP(MLP):
    pass


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors (https://qwenlm.github.io/blog/qwen2-vl/).

    Explanation:
        Multimodal 3D rotary position embedding is an extension to 1D rotary position embedding. The input embedding
        sequence contains vision (images / videos) embedding and text embedding or just contains text embedding. For
        vision embedding part, we apply rotary position embedding on temporal, height and width dimension separately.
        Here we split the channel dimension to 3 chunks for the temporal, height and width rotary position embedding.
        For text embedding part, we just apply 1D rotary position embedding. The three rotary position index (temporal,
        height and width) of text embedding is always the same, so the text embedding rotary position embedding has no
        difference with modern LLMs.

    Args:
        q (`paddle.Tensor`): The query tensor.
        k (`paddle.Tensor`): The key tensor.
        cos (`paddle.Tensor`): The cosine part of the rotary embedding.
        sin (`paddle.Tensor`): The sine part of the rotary embedding.
        position_ids (`paddle.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        mrope_section(`List(int)`):
            Multimodal rope section is for channel dimension of temporal, height and width in rope calculation.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(paddle.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    mrope_section = mrope_section * 2
    cos = paddle.cat(x=[m[i % 3] for i, m in enumerate(cos.split(mrope_section, axis=-1))], axis=-1).unsqueeze(
        axis=unsqueeze_dim
    )
    sin = paddle.cat(x=[m[i % 3] for i, m in enumerate(sin.split(mrope_section, axis=-1))], axis=-1).unsqueeze(
        axis=unsqueeze_dim
    )

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen2_5_VLAttention(nn.Layer):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: Qwen2_5_VLTextConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        self.scaling = self.head_dim**-0.5

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.sequence_parallel = config.sequence_parallel
        self.fuse_attention_qkv = config.fuse_attention_qkv
        self.gqa_or_mqa = config.num_attention_heads != config.num_key_value_heads

        if config.tensor_model_parallel_size > 1:
            assert (
                self.num_heads % config.tensor_model_parallel_size == 0
            ), f"num_heads: {self.num_heads}, tensor_model_parallel_size: {config.tensor_model_parallel_size}"
            self.num_heads = self.num_heads // config.tensor_model_parallel_size

            assert (
                self.num_key_value_heads % config.tensor_model_parallel_size == 0
            ), f"num_key_value_heads: {self.num_key_value_heads}, tensor_model_parallel_size: {config.tensor_model_parallel_size}"
            self.num_key_value_heads = self.num_key_value_heads // config.tensor_model_parallel_size

        kv_hidden_size = self.config.num_key_value_heads * self.head_dim
        q_hidden_size = self.config.num_attention_heads * self.head_dim

        if not self.fuse_attention_qkv:
            self.q_proj = GeneralLinear.create(
                config.hidden_size,
                q_hidden_size,
                has_bias=True,
                config=config,
                tp_plan="colwise",
            )
            self.k_proj = GeneralLinear.create(
                config.hidden_size,
                kv_hidden_size,
                has_bias=True,
                config=config,
                tp_plan="colwise",
            )
            self.v_proj = GeneralLinear.create(
                config.hidden_size,
                kv_hidden_size,
                has_bias=True,
                config=config,
                tp_plan="colwise",
            )
        else:
            self.qkv_proj = GeneralLinear.create(
                config.hidden_size,
                q_hidden_size + 2 * kv_hidden_size,
                has_bias=True,
                config=config,
                tp_plan="colwise",
            )
        self.o_proj = GeneralLinear.create(
            q_hidden_size,
            config.hidden_size,
            has_bias=False,
            config=config,
            tp_plan="rowwise",
        )
        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,  # default true
        position_embeddings: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        **kwargs,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        if not self.fuse_attention_qkv:
            if self.sequence_parallel:
                max_sequence_length = self.config.max_sequence_length
                bsz = hidden_states.shape[0] * self.config.tensor_model_parallel_size // max_sequence_length
                q_len = max_sequence_length
            else:
                bsz, q_len, _ = hidden_states.shape

            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            query_states = query_states.reshape(bsz, q_len, -1, self.head_dim)
            key_states = key_states.reshape(bsz, q_len, -1, self.head_dim)
            value_states = value_states.reshape(bsz, q_len, -1, self.head_dim)

        else:
            mix_layer = self.qkv_proj(hidden_states)
            if self.sequence_parallel:
                max_sequence_length = self.config.max_sequence_length
                bsz = hidden_states.shape[0] * self.config.tensor_model_parallel_size // max_sequence_length
                q_len = max_sequence_length
                target_shape = [
                    bsz,
                    q_len,
                    self.num_key_value_heads,
                    (self.num_key_value_groups + 2) * self.head_dim,
                ]
            else:
                target_shape = [0, 0, self.num_key_value_heads, (self.num_key_value_groups + 2) * self.head_dim]
            # mix_layer = mix_layer.reshape(target_shape)
            mix_layer = paddle.reshape_(mix_layer, target_shape)
            query_states, key_states, value_states = paddle.split(
                mix_layer,
                num_or_sections=[self.num_key_value_groups * self.head_dim, self.head_dim, self.head_dim],
                axis=-1,
            )
            if self.gqa_or_mqa:
                # query_states = query_states.reshape([0, 0, self.num_heads, self.head_dim])
                query_states = paddle.reshape_(query_states, [0, 0, self.num_heads, self.head_dim])

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.config.rope_parameters["mrope_section"]
        )

        # [bs, num_head, seq_len, head_dim]
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        if self.config.sequence_parallel:
            attn_output = attn_output.reshape([-1, attn_output.shape[-1]])
        attn_output = self.o_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights


class Qwen2_5_VLDecoderLayer(nn.Layer):
    def __init__(self, config: Qwen2_5_VLTextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        if config.use_sliding_window and config._attn_implementation != "flashmask":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )
        self.self_attn = Qwen2_5_VLAttention(config, layer_idx)

        self.mlp = Qwen2MLP(config, fuse_up_gate=config.fuse_attention_ffn)
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
        self.attention_type = config.layer_types[layer_idx]

        if config.sequence_parallel:
            self.post_attention_layernorm.enable_sequence_parallel()
            if not hasattr(config, "disable_ffn_model_parallel"):
                self.input_layernorm.enable_sequence_parallel()

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        position_embeddings: Optional[tuple[paddle.Tensor, paddle.Tensor]] = None,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        **kwargs,
    ):
        """
        Args:
            hidden_states (`paddle.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`paddle.Tensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            past_key_values (`Cache`, *optional*): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            position_embeddings (`Tuple[paddle.Tensor, paddle.Tensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class Qwen2_5_VLTextModel(Qwen2_5_VLPretrainedModel):
    config: Qwen2_5_VLTextConfig
    input_modalities = "text"

    def __init__(self, config: Qwen2_5_VLTextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = GeneralEmbedding.create(
            config=config,
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            padding_idx=self.padding_idx,
        )
        self.layers = nn.LayerList(
            [Qwen2_5_VLDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=config.hidden_size,
            norm_eps=self.config.rms_norm_eps,
            input_is_parallel=config.sequence_parallel,
        )
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types
        self.rotary_emb = Qwen2_5_VLRotaryEmbedding(config=config)

        self.gradient_checkpointing = False
        self.has_sliding_layers = getattr(
            self.config, "sliding_window", None
        ) is not None and "sliding_attention" in getattr(self.config, "layer_types", [])

    @paddle.jit.not_to_static
    def recompute_training_full(
        self,
        layer_module: nn.Layer,
        hidden_states: Tensor,
        attention_mask: Tensor,
        position_embeddings: Optional[Tuple[paddle.Tensor, paddle.Tensor]],
        position_ids: Optional[paddle.Tensor],
        past_key_values: Optional[Cache],
        output_attentions: bool,
        use_cache: bool,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
    ):
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        hidden_states = recompute(
            create_custom_forward(layer_module),
            hidden_states,
            attention_mask,
            position_ids,
            past_key_values,
            output_attentions,
            use_cache,
            position_embeddings,
            attn_mask_startend_row_indices,
        )

        return hidden_states

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[paddle.Tensor] = None,
        attn_mask_startend_row_indices=None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        cache_length = past_key_values.get_seq_length() if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids).astype(self.embed_tokens.weight.dtype)

        if self.config.sequence_parallel:
            # [bs, seq_len, num_head * head_dim] -> [bs * seq_len, num_head * head_dim]
            bs, seq_len, hidden_size = inputs_embeds.shape
            inputs_embeds = paddle.reshape_(inputs_embeds, [bs * seq_len, hidden_size])
            # [seq_len * bs / n, num_head * head_dim] (n is mp parallelism)
            inputs_embeds = ScatterOp.apply(inputs_embeds)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = paddle.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1])

        if position_ids is None:
            # the hard coded `3` is for temporal, height and width.
            position_ids = cache_position.reshape([1, 1, -1]).expand([3, inputs_embeds.shape[0], -1])
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0])

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            # If inputs are not packed (usual 3D positions), do not prepare mask from position_ids
            text_position_ids = None

        # Prepare mask arguments
        mask_kwargs = {
            "config": self.config,
            "inputs_embeds": inputs_embeds,
            "batch_size": batch_size,
            "seq_length": seq_length,
            "cache_length": cache_length,
            "attention_mask": attention_mask,
            "attn_mask_startend_row_indices": attn_mask_startend_row_indices,
            "prepare_decoder_attention_mask": self._prepare_decoder_attention_mask,
        }
        # Create the causal mask and row indices
        full_mask, full_indices = create_causal_mask_and_row_indices(**mask_kwargs)

        causal_mask_mapping = {"full_attention": full_mask}
        attn_mask_startend_row_indices_mapping = {"full_attention": full_indices}

        # if model has sliding layer
        if self.has_sliding_layers:
            (
                causal_mask_mapping["sliding_attention"],
                attn_mask_startend_row_indices_mapping["sliding_attention"],
            ) = create_sliding_window_causal_mask_and_row_indices(**mask_kwargs)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

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
                layer_outputs = self.recompute_training_full(
                    decoder_layer,
                    hidden_states,
                    attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                    position_embeddings=position_embeddings,
                    position_ids=text_position_ids,
                    past_key_values=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    attn_mask_startend_row_indices=attn_mask_startend_row_indices_mapping[
                        decoder_layer.attention_type
                    ],
                    **kwargs,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                    position_embeddings=position_embeddings,
                    position_ids=text_position_ids,
                    past_key_values=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    attn_mask_startend_row_indices=attn_mask_startend_row_indices_mapping[
                        decoder_layer.attention_type
                    ],
                    **kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, past_key_values, all_hidden_states, all_self_attns] if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


@register_base_model
class Qwen2_5_VLModel(Qwen2_5_VLPretrainedModel):
    base_model_prefix = ""
    _checkpoint_conversion_mapping = {"^model": "language_model"}
    config: Qwen2_5_VLConfig
    _no_split_modules = ["Qwen2_5_VLDecoderLayer", "Qwen2_5_VLVisionBlock"]

    def __init__(self, config):
        super().__init__(config)
        self.visual = Qwen2_5_VisionTransformerPretrainedModel._from_config(config.vision_config)
        self.language_model = Qwen2_5_VLTextModel._from_config(config.text_config)
        self.rope_deltas = None  # cache rope_deltas here

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.language_model = decoder

    def get_decoder(self):
        return self.language_model

    def get_rope_index(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        image_grid_thw: Optional[paddle.Tensor] = None,
        video_grid_thw: Optional[paddle.Tensor] = None,
        second_per_grid_ts: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
    ) -> tuple[paddle.Tensor, paddle.Tensor]:
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
            if attention_mask is not None:
                attention_mask = attention_mask == 1
            position_ids = paddle.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
            )
            image_index, video_index = 0, 0
            for i, input_ids in enumerate(total_input_ids):
                if attention_mask is not None:
                    input_ids = input_ids[attention_mask[i]]
                image_nums, video_nums = 0, 0
                vision_start_indices = paddle.argwhere(input_ids == vision_start_token_id).squeeze(1)
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
                    llm_pos_ids_list.append(paddle.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    range_tensor = paddle.arange(llm_grid_t).view(-1, 1)
                    expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                    time_tensor = expanded_range * second_per_grid_t * self.config.vision_config.tokens_per_second

                    time_tensor_long = time_tensor.astype(dtype="int64")
                    t_index = time_tensor_long.flatten()

                    h_index = paddle.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = paddle.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(paddle.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(paddle.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = paddle.cat(llm_pos_ids_list, axis=1).reshape([3, -1])
                if attention_mask is not None:
                    position_ids[..., i, attention_mask[i]] = llm_positions
                else:
                    position_ids[..., i, :] = llm_positions
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = paddle.to_tensor(mrope_position_deltas).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.astype(dtype="int64").cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = paddle.arange(input_ids.shape[1]).view(1, 1, -1).expand(3, input_ids.shape[0], -1)
                mrope_position_deltas = paddle.zeros(
                    [input_ids.shape[0], 1],
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

    def get_video_features(self, pixel_values_videos: paddle.Tensor, video_grid_thw: Optional[paddle.Tensor] = None):
        """
        Encodes videos into continuous embeddings that can be forwarded to the language model.

        Args:
            pixel_values_videos (`paddle.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input videos.
            video_grid_thw (`paddle.Tensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
        """
        pixel_values_videos = pixel_values_videos.astype(self.visual.patch_embed.proj.weight.dtype)
        video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
        split_sizes = (video_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        video_embeds = paddle.split(video_embeds, split_sizes)
        return video_embeds

    def get_image_features(self, pixel_values: paddle.Tensor, image_grid_thw: Optional[paddle.Tensor] = None):
        """
        Encodes images into continuous embeddings that can be forwarded to the language model.

        Args:
            pixel_values (`paddle.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input images.
            image_grid_thw (`paddle.Tensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
        """
        pixel_values = pixel_values.astype(self.visual.patch_embed.proj.weight.dtype)
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        image_embeds = paddle.split(image_embeds, split_sizes)
        return image_embeds

    def get_placeholder_mask(
        self,
        input_ids: paddle.Tensor,
        inputs_embeds: paddle.Tensor,
        image_features: Optional[paddle.Tensor] = None,
        video_features: Optional[paddle.Tensor] = None,
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                paddle.to_tensor(self.config.image_token_id, dtype="int64")
            )
            special_image_mask = special_image_mask.all(-1)
            special_video_mask = inputs_embeds == self.get_input_embeddings()(
                paddle.to_tensor(self.config.video_token_id, dtype="int64")
            )
            special_video_mask = special_video_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
            special_video_mask = input_ids == self.config.video_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds)
        if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}"
            )

        n_video_tokens = special_video_mask.sum()
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds)
        if video_features is not None and inputs_embeds[special_video_mask].numel() != video_features.numel():
            raise ValueError(
                f"Videos features and video tokens do not match: tokens: {n_video_tokens}, features {video_features.shape[0]}"
            )

        return special_image_mask, special_video_mask

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[paddle.Tensor] = None,
        pixel_values_videos: Optional[paddle.Tensor] = None,
        image_grid_thw: Optional[paddle.Tensor] = None,
        video_grid_thw: Optional[paddle.Tensor] = None,
        rope_deltas: Optional[paddle.Tensor] = None,
        cache_position: Optional[paddle.Tensor] = None,
        second_per_grid_ts: Optional[paddle.Tensor] = None,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        **kwargs,
    ) -> Union[tuple, Qwen2_5_VLModelOutputWithPast]:
        r"""
        image_grid_thw (`paddle.Tensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`paddle.Tensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        rope_deltas (`paddle.Tensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
        second_per_grid_ts (`paddle.Tensor` of shape `(num_videos)`, *optional*):
            The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = paddle.cat(image_embeds, dim=0).astype(inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = paddle.cat(video_embeds, axis=0).astype(inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if position_ids is None:
            if self.rope_deltas is None or cache_position is None or cache_position[0] == 0:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    attention_mask=attention_mask,
                )
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                position_ids = paddle.arange(seq_length)
                position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
                if cache_position is not None:
                    delta = cache_position[0] + self.rope_deltas
                else:
                    delta = paddle.zeros((batch_size, seq_length))
                delta = delta.repeat_interleave(batch_size // delta.shape[0], axis=1)
                position_ids = position_ids + delta

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            **kwargs,
        )

        output = Qwen2_5_VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )
        return output if return_dict else output.to_tuple()


@dataclass
class Qwen2_5_VLCausalLMOutputWithPast(ModelOutput):
    r"""
    loss (`paddle.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`paddle.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    past_key_values (`Cache)`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    rope_deltas (`paddle.Tensor` of shape `(batch_size, )`, *optional*):
        The rope index difference between sequence length and multimodal rope.
    """

    loss: Optional[paddle.Tensor] = None
    logits: Optional[paddle.Tensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[paddle.Tensor]] = None
    attentions: Optional[tuple[paddle.Tensor]] = None
    rope_deltas: Optional[paddle.Tensor] = None


class Qwen2_5_VLForConditionalGeneration(Qwen2_5_VLPretrainedModel):
    _checkpoint_conversion_mapping = {
        "^visual": "model.visual",
        r"^model(?!\.(language_model|visual))": "model.language_model",
    }
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
    config_class = Qwen2_5_VLConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2_5_VLModel(config)
        self.lm_head = GeneralLMHead(config.text_config)
        self.criterion = CriterionLayer(config.text_config)
        self.tie_weights()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

    def get_video_features(self, pixel_values_videos: paddle.Tensor, video_grid_thw: Optional[paddle.Tensor] = None):
        return self.model.get_video_features(pixel_values_videos, video_grid_thw)

    def get_image_features(self, pixel_values: paddle.Tensor, image_grid_thw: Optional[paddle.Tensor] = None):
        return self.model.get_image_features(pixel_values, image_grid_thw)

    @property
    def language_model(self):
        return self.model.language_model

    @property
    def visual(self):
        return self.model.visual

    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        pixel_values: Optional[paddle.Tensor] = None,
        pixel_values_videos: Optional[paddle.Tensor] = None,
        image_grid_thw: Optional[paddle.Tensor] = None,
        video_grid_thw: Optional[paddle.Tensor] = None,
        rope_deltas: Optional[paddle.Tensor] = None,
        cache_position: Optional[paddle.Tensor] = None,
        second_per_grid_ts: Optional[paddle.Tensor] = None,
        logits_to_keep: Union[int, paddle.Tensor] = 0,
        return_dict: Optional[bool] = True,
        **kwargs,
    ) -> Union[tuple, Qwen2_5_VLCausalLMOutputWithPast]:
        r"""
        labels (`paddle.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        image_grid_thw (`paddle.Tensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`paddle.Tensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        rope_deltas (`paddle.Tensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
        second_per_grid_ts (`paddle.Tensor` of shape `(num_videos)`, *optional*):
            The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.

        Example:

        ```python
        >>> from paddleformers.transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        >>> model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example1.jpg",
                    },
                    {"type": "text", "text": "Describe the image."},
                ],
            }
        ]

        >>> inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pd"
        )

        >>> # Generate
        >>> generated_ids = model.generate(**inputs, max_new_tokens=1024)
        >>> output_text = processor.batch_decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        >>> print(output_text)
        ```
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            **kwargs,
        )

        hidden_states = outputs[0]

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[..., slice_indices, :])

        loss = None
        if labels is not None:
            loss, _ = self.criterion(logits, labels)

        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )

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

        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        # NOTE: Due to differences in cache_position, it must be passed as an argument.
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

        # Qwen2-5-VL position_ids are prepared with rope_deltas
        if position_ids is None:
            # Calculate RoPE index once per generation in the pre-fill stage only.
            # When compiling, we can't check tensor values thus we check only input length
            # It is safe to assume that `length!=1` means we're in pre-fill because compiled
            # models currently cannot do assisted decoding
            if cache_position[0] == 0 or self.model.rope_deltas is None:
                vision_positions, rope_deltas = self.model.get_rope_index(
                    model_inputs.get("input_ids", None),
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    attention_mask=attention_mask,
                )
                self.model.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            elif "position_ids" in model_inputs:
                batch_size, seq_length = model_inputs["position_ids"].shape
                position_ids = paddle.arange(seq_length)
                position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
                delta = cache_position[0] + self.model.rope_deltas
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                vision_positions = position_ids + delta.expand_as(position_ids)
            # Concatenate "text + vision" positions into [4, bs, seq-len]
            text_positions = model_inputs["position_ids"][None, ...]
            model_inputs["position_ids"] = paddle.cat([text_positions, vision_positions], dim=0)

        if cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

        return model_inputs

    def _get_image_nums_and_video_nums(
        self,
        input_ids: Optional[paddle.Tensor],
        inputs_embeds: Optional[paddle.Tensor] = None,
    ) -> tuple[paddle.Tensor, paddle.Tensor]:
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

        if inputs_embeds is not None:
            vision_start_mask = (
                inputs_embeds == self.get_input_embeddings()(paddle.to_tensor(vision_start_token_id, dtype="int64"))
            )[..., 0]
            image_mask = (
                inputs_embeds == self.get_input_embeddings()(paddle.to_tensor(image_token_id, dtype="int64"))
            )[..., 0]
            video_mask = (
                inputs_embeds == self.get_input_embeddings()(paddle.to_tensor(video_token_id, dtype="int64"))
            )[..., 0]
        else:
            vision_start_mask = input_ids == vision_start_token_id
            image_mask = input_ids == image_token_id
            video_mask = input_ids == video_token_id

        vision_first_mask = paddle.roll(vision_start_mask, shifts=1, dims=1)
        image_nums = paddle.sum(vision_first_mask & image_mask, dim=1)
        video_nums = paddle.sum(vision_first_mask & video_mask, dim=1)

        return image_nums, video_nums

    def _expand_inputs_for_generation(
        self,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[paddle.int64] = None,
        **model_kwargs,
    ) -> tuple[paddle.int64, dict[str, Any]]:
        # Overwritten -- Support for expanding tensors without a batch size dimension
        # e.g., pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw, second_per_grid_t
        # pixel_values.shape[0] is sum(seqlen_images for samples)
        # image_grid_thw.shape[0] is sum(num_images for samples)

        if expand_size == 1:
            return input_ids, model_kwargs

        visual_keys = ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw", "second_per_grid_ts"]

        def _expand_dict_for_generation_visual(dict_to_expand):
            image_grid_thw = model_kwargs.get("image_grid_thw", None)
            video_grid_thw = model_kwargs.get("video_grid_thw", None)
            image_nums, video_nums = self._get_image_nums_and_video_nums(
                input_ids, inputs_embeds=model_kwargs.get("inputs_embeds", None)
            )

            def _repeat_interleave_samples(x, lengths, repeat_times):
                samples = paddle.split(x, lengths)
                repeat_args = [repeat_times] + [1] * (x.dim() - 1)
                result = paddle.cat([sample.repeat(*repeat_args) for sample in samples], dim=0)
                return result

            for key in dict_to_expand:
                if key == "pixel_values":
                    # split images into samples
                    samples = paddle.split(image_grid_thw, list(image_nums))
                    # compute the sequence length of images for each sample
                    lengths = [paddle.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "image_grid_thw":
                    # get the num of images for each sample
                    lengths = list(image_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "pixel_values_videos":
                    samples = paddle.split(video_grid_thw, list(video_nums))
                    lengths = [paddle.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "video_grid_thw":
                    lengths = list(video_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "second_per_grid_ts":
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=list(video_nums), repeat_times=expand_size
                    )
            return dict_to_expand

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if (
                    key != "cache_position"
                    and dict_to_expand[key] is not None
                    and isinstance(dict_to_expand[key], paddle.Tensor)
                    and key not in visual_keys
                ):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        model_kwargs = _expand_dict_for_generation_visual(model_kwargs)

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs


__all__ = ["Qwen2_5_VLForConditionalGeneration", "Qwen2_5_VLModel", "Qwen2_5_VLPretrainedModel", "Qwen2_5_VLTextModel"]
