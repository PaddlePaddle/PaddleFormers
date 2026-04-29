# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2026 The LLaVA-OneVision-1.5 Authors and The HuggingFace Inc. team. All rights reserved.
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
"""Paddle LLaVA-OneVision-1.5 model components."""

from __future__ import annotations

from typing import Optional, Tuple

import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.distributed.fleet.utils import recompute

from ...nn.activation import ACT2FN
from ...nn.attention.interface import ALL_ATTENTION_FUNCTIONS
from ...nn.linear import Linear as GeneralLinear
from ..model_utils import PretrainedModel
from .configuration import Llavaonevision1_5Config, LLaVAOneVision1_5TextConfig, RiceConfig


def rotate_half(x: paddle.Tensor) -> paddle.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.cat([-x2, x1], axis=-1)


def apply_rotary_pos_emb_vision(
    q: paddle.Tensor, k: paddle.Tensor, cos: paddle.Tensor, sin: paddle.Tensor
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    with paddle.amp.auto_cast(False):
        q = q.astype("float32")
        k = k.astype("float32")
        cos = cos.unsqueeze(-2).astype("float32")
        sin = sin.unsqueeze(-2).astype("float32")
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.astype(orig_q_dtype), k_embed.astype(orig_k_dtype)


class RiceRotaryEmbedding(nn.Layer):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (paddle.arange(0, dim, 2, dtype="float32") / dim))
        self.register_buffer("inv_freq", inv_freq, persistable=False)

    def forward(self, seqlen: int) -> paddle.Tensor:
        seq = paddle.arange(seqlen, dtype=self.inv_freq.dtype)
        freqs = paddle.outer(seq, self.inv_freq)
        return freqs


class RicePatchEmbed(nn.Layer):
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 1,
        in_channels: int = 3,
        embed_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.proj = nn.Conv2D(
            in_channels,
            embed_dim,
            kernel_size=[patch_size, patch_size],
            stride=[patch_size, patch_size],
            bias_attr=False,
        )

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.reshape([-1, self.in_channels, self.patch_size, self.patch_size])
        hidden_states = self.proj(hidden_states.astype(target_dtype)).reshape([-1, self.embed_dim])
        return hidden_states


class RicePatchMerger(nn.Layer):
    def __init__(
        self,
        dim: int,
        context_dim: int,
        spatial_merge_size: int = 2,
        layer_norm_eps: float = 1e-05,
    ) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = nn.LayerNorm(context_dim, epsilon=layer_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        return self.mlp(self.ln_q(x).reshape([-1, self.hidden_size]))


class RiceMlp(nn.Layer):
    def __init__(self, dim: int, hidden_dim: int, hidden_act: str) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = ACT2FN[hidden_act]
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class RiceAttention(nn.Layer):
    def __init__(self, config: RiceConfig) -> None:
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.num_key_value_groups = 1
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = 0.0
        self.is_causal = False
        self.qkv = GeneralLinear.create(
            config.hidden_size,
            config.hidden_size * 3,
            has_bias=True,
            linear_type="default",
        )
        self.proj = GeneralLinear.create(
            config.hidden_size,
            config.hidden_size,
            linear_type="default",
        )

    def forward(
        self,
        hidden_states: paddle.Tensor,
        cu_seqlens: paddle.Tensor,
        position_embeddings: Tuple[paddle.Tensor, paddle.Tensor],
        **kwargs,
    ) -> paddle.Tensor:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape([seq_length, 3, self.num_heads, -1]).transpose([1, 0, 2, 3]).unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        query_states = query_states.transpose([1, 0, 2]).unsqueeze(0)
        key_states = key_states.transpose([1, 0, 2]).unsqueeze(0)
        value_states = value_states.transpose([1, 0, 2]).unsqueeze(0)

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
        attn_output = attn_output.reshape([seq_length, -1])
        return self.proj(attn_output)


class RiceBlock(nn.Layer):
    def __init__(self, config: RiceConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.attn = RiceAttention(config)
        self.mlp = RiceMlp(config.hidden_size, int(config.intermediate_size), config.hidden_act)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        cu_seqlens: paddle.Tensor,
        position_embeddings: Tuple[paddle.Tensor, paddle.Tensor],
        **kwargs,
    ) -> paddle.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class LLaVAOneVision1_5PretrainedModel(PretrainedModel):
    config_class = Llavaonevision1_5Config
    base_model_prefix = "model"
    input_modalities = ["image", "video", "text"]
    _no_split_modules = ["LLaVAOneVision1_5DecoderLayer", "RiceBlock"]
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
        "fc1",
        "fc2",
        "proj",
        "merger.mlp\.\d+",
    ]

    @classmethod
    def _gen_aoa_config(cls, config: Llavaonevision1_5Config):
        mapping = getattr(cls, "_checkpoint_conversion_mapping", {})
        visual_target = next((v for v in mapping.values() if "visual" in v), "visual")
        visual_prefix = f"{visual_target}." if not visual_target.endswith(".") else visual_target

        aoa_statements = [
            f"visual.patch_embed.proj.weight -> {visual_prefix}patch_embed.proj.weight",
            f"visual.class_embedding -> {visual_prefix}class_embedding",
            f"visual.class_pos_emb -> {visual_prefix}class_pos_emb",
            f"visual.pre_layernorm.weight -> {visual_prefix}pre_layernorm.weight",
            f"visual.pre_layernorm.bias -> {visual_prefix}pre_layernorm.bias",
            f"visual.merger.ln_q.weight -> {visual_prefix}merger.ln_q.weight",
            f"visual.merger.ln_q.bias -> {visual_prefix}merger.ln_q.bias",
            f"visual.blocks.$LAYER_ID.norm1.weight -> {visual_prefix}blocks.$LAYER_ID.norm1.weight",
            f"visual.blocks.$LAYER_ID.norm1.bias -> {visual_prefix}blocks.$LAYER_ID.norm1.bias",
            f"visual.blocks.$LAYER_ID.norm2.weight -> {visual_prefix}blocks.$LAYER_ID.norm2.weight",
            f"visual.blocks.$LAYER_ID.norm2.bias -> {visual_prefix}blocks.$LAYER_ID.norm2.bias",
            f"visual.blocks.$LAYER_ID.attn.qkv.weight^T -> {visual_prefix}blocks.$LAYER_ID.attn.qkv.weight",
            f"visual.blocks.$LAYER_ID.attn.qkv.bias -> {visual_prefix}blocks.$LAYER_ID.attn.qkv.bias",
            f"visual.blocks.$LAYER_ID.attn.proj.weight^T -> {visual_prefix}blocks.$LAYER_ID.attn.proj.weight",
            f"visual.blocks.$LAYER_ID.attn.proj.bias -> {visual_prefix}blocks.$LAYER_ID.attn.proj.bias",
            f"visual.blocks.$LAYER_ID.mlp.fc1.weight^T -> {visual_prefix}blocks.$LAYER_ID.mlp.fc1.weight",
            f"visual.blocks.$LAYER_ID.mlp.fc1.bias -> {visual_prefix}blocks.$LAYER_ID.mlp.fc1.bias",
            f"visual.blocks.$LAYER_ID.mlp.fc2.weight^T -> {visual_prefix}blocks.$LAYER_ID.mlp.fc2.weight",
            f"visual.blocks.$LAYER_ID.mlp.fc2.bias -> {visual_prefix}blocks.$LAYER_ID.mlp.fc2.bias",
            f"visual.merger.mlp.0.weight^T -> {visual_prefix}merger.mlp.0.weight",
            f"visual.merger.mlp.0.bias -> {visual_prefix}merger.mlp.0.bias",
            f"visual.merger.mlp.2.weight^T -> {visual_prefix}merger.mlp.2.weight",
            f"visual.merger.mlp.2.bias -> {visual_prefix}merger.mlp.2.bias",
        ]
        return {"aoa_statements": aoa_statements}


class RiceTransformerPretrainedModel(LLaVAOneVision1_5PretrainedModel):
    config_class = RiceConfig
    _no_split_modules = ["RiceBlock"]

    def __init__(self, config: RiceConfig) -> None:
        super().__init__(config)
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.patch_embed = RicePatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
        )

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = RiceRotaryEmbedding(head_dim // 2)
        scale = config.hidden_size**-0.5
        self.class_embedding = self.create_parameter(
            shape=[config.hidden_size],
            default_initializer=nn.initializer.Normal(mean=0.0, std=scale),
        )
        self.class_pos_emb = self.create_parameter(
            shape=[1, head_dim // 2],
            default_initializer=nn.initializer.Normal(mean=0.0, std=1.0),
        )
        self.window_size = None
        self.pre_layernorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.blocks = nn.LayerList([RiceBlock(config) for _ in range(config.depth)])
        self.merger = RicePatchMerger(
            dim=config.text_hidden_size,
            context_dim=config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
            layer_norm_eps=config.layer_norm_eps,
        )
        self.gradient_checkpointing = False

    def get_dtype(self):
        return self.blocks[0].mlp.fc2.weight.dtype

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
            hpos_ids = hpos_ids.transpose([0, 2, 1, 3]).flatten()

            wpos_ids = paddle.arange(w).unsqueeze(0).expand([h, -1])
            wpos_ids = wpos_ids.reshape(
                [
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                ]
            )
            wpos_ids = wpos_ids.transpose([0, 2, 1, 3]).flatten()
            pos_ids.append(paddle.stack([hpos_ids, wpos_ids], axis=-1).tile([t, 1]))
        pos_ids = paddle.cat(pos_ids, axis=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        return rotary_pos_emb_full[pos_ids].flatten(start_axis=1)

    @paddle.jit.not_to_static
    def recompute_training_full(
        self,
        layer_module: nn.Layer,
        hidden_states: paddle.Tensor,
        cu_seqlens: paddle.Tensor,
        position_embeddings: Tuple[paddle.Tensor, paddle.Tensor],
    ):
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        return recompute(create_custom_forward(layer_module), hidden_states, cu_seqlens, position_embeddings)

    def forward(self, hidden_states: paddle.Tensor, grid_thw: paddle.Tensor, is_verifying: bool = False) -> paddle.Tensor:
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        img_feats = hidden_states.shape[0]

        cu_seqlens = paddle.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            axis=0, dtype="int32"
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        cu = cu_seqlens.astype("int64")

        hidden_segments = []
        rotary_segments = []
        new_cu = [0]
        for i in range(1, cu.shape[0]):
            seg_start = int(cu[i - 1].item())
            seg_end = int(cu[i].item())
            segment = hidden_states[seg_start:seg_end]
            rotary_segment = rotary_pos_emb[seg_start:seg_end]
            hidden_segments.append(paddle.cat([self.class_embedding.astype(segment.dtype).unsqueeze(0), segment], axis=0))
            rotary_segments.append(paddle.cat([self.class_pos_emb.astype(rotary_segment.dtype), rotary_segment], axis=0))
            new_cu.append(new_cu[-1] + seg_end - seg_start + 1)

        hidden_states = paddle.cat(hidden_segments, axis=0)
        rotary_pos_emb = paddle.cat(rotary_segments, axis=0)
        cu_seqlens = paddle.to_tensor(new_cu, dtype="int32", place=hidden_states.place)

        hidden_states = self.pre_layernorm(hidden_states)
        emb = paddle.cat((rotary_pos_emb, rotary_pos_emb), axis=-1)
        position_embeddings = (emb.cos(), emb.sin())

        for blk in self.blocks:
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
                    cu_seqlens=cu_seqlens,
                    position_embeddings=position_embeddings,
                )
            else:
                hidden_states = blk(
                    hidden_states,
                    cu_seqlens=cu_seqlens,
                    position_embeddings=position_embeddings,
                )

        if is_verifying:
            segments = []
            for i in range(1, len(new_cu)):
                segments.append(hidden_states[new_cu[i - 1] + 1 : new_cu[i]])
            return paddle.cat(segments, axis=0)

        output_segments = []
        for i in range(1, len(new_cu)):
            output_segments.append(hidden_states[new_cu[i - 1] + 1 : new_cu[i]])
        hidden_states = paddle.cat(output_segments, axis=0)
        if hidden_states.shape[0] != img_feats:
            raise ValueError(f"Unexpected Rice vision sequence length: {hidden_states.shape[0]} != {img_feats}")
        return self.merger(hidden_states)


__all__ = [
    "LLaVAOneVision1_5PretrainedModel",
    "RiceTransformerPretrainedModel",
]
