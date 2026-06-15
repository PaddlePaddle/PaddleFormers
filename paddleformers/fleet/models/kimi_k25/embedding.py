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

from collections.abc import Sequence
from dataclasses import dataclass

import paddle
from paddle import nn
from paddle.distributed.fleet.meta_parallel import LayerSpec, build_spec_layer
from paddle.nn import functional as F

from ...transformer import TransformerConfig
from ...transformer.layer import FleetLayer


@dataclass
class VisionEmbeddingSpec:
    rope_embedding: LayerSpec = None


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    From:
    https://github.com/OpenGVLab/InternVideo/blob/421f6d2361fc8f61a3394244571f2601a4e99e29/InternVideo2/multi_modality/models/backbones/internvideo2/pos_embed.py#L86
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = paddle.arange(embed_dim // 2, dtype=paddle.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = paddle.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = paddle.sin(out)  # (M, D/2)
    emb_cos = paddle.cos(out)  # (M, D/2)

    emb = paddle.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, t_size, cls_token=False):
    """
    t_size: int of the temporal size
    return:
    pos_embed: [t_size, embed_dim] or [1+t_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_t = paddle.arange(t_size, dtype=paddle.float32)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid_t)
    if cls_token:
        pos_embed = paddle.concatenate([paddle.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_rope_shape_decorate(func):
    _get_rope_shape_first_call_flag = set()

    def wrapper(org, interpolation_mode, shape):
        key = (org.requires_grad, paddle.is_grad_enabled(), interpolation_mode)
        if key not in _get_rope_shape_first_call_flag:
            _get_rope_shape_first_call_flag.add(key)
            _ = func(org, interpolation_mode, shape=(64, 64))
        return func(org, interpolation_mode, shape)

    return wrapper


@get_rope_shape_decorate
def get_rope_shape(org, interpolation_mode, shape):
    return (
        F.interpolate(
            org.permute((2, 0, 1)).unsqueeze(0),
            size=shape,
            mode=interpolation_mode,
        )
        .squeeze(0)
        .permute((1, 2, 0))
        .flatten(end_dim=1)
    )


class Learnable2DInterpPosEmbDivided_fixed(nn.Layer):
    def __init__(
        self,
        height: int,
        width: int,
        num_frames: int,
        dim: int,
        interpolation_mode: str = "bicubic",
    ) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.dim = dim
        self.interpolation_mode = interpolation_mode
        self.weight = nn.Parameter(paddle.empty(height, width, dim))
        self.register_buffer(
            "time_weight",
            get_1d_sincos_pos_embed(self.dim, self.num_frames).float().unsqueeze(1),
            persistent=False,
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight)

    def forward(self, x: paddle.Tensor, grid_thws: paddle.Tensor) -> paddle.Tensor:
        pos_embs = []
        for t, h, w in grid_thws.tolist():
            assert t <= self.num_frames, f"t:{t} > self.num_frames:{self.num_frames}"
            if (h, w) == self.weight.shape[:-1]:
                pos_emb_2d = self.weight.flatten(end_dim=1)
            else:
                pos_emb_2d = get_rope_shape(
                    self.weight,
                    interpolation_mode=self.interpolation_mode,
                    shape=(h, w),
                )

            if t == 1:
                pos_emb_3d = pos_emb_2d
            else:
                pos_emb_3d = pos_emb_2d.unsqueeze(0).repeat(t, 1, 1) + self.time_weight[0:t]

            pos_embs.append(pos_emb_3d.reshape(-1, pos_emb_3d.shape[-1]))

        out = x + paddle.cat(pos_embs)
        return out


class MoonVision3dPatchEmbed(FleetLayer):
    def __init__(
        self,
        config: TransformerConfig,
        sublayers_spec: VisionEmbeddingSpec,
    ):
        super().__init__(config)
        # Use getattr to get config values with fallback to defaults
        self.out_dim = getattr(config, "hidden_size", None)
        self.in_dim = 3
        self.patch_size = getattr(config, "patch_size", None)
        self.pos_emb_height = getattr(config, "init_pos_emb_height", 14)
        self.pos_emb_width = getattr(config, "init_pos_emb_width", 14)
        self.pos_emb_time = getattr(config, "init_pos_emb_time", 4)
        self.pos_emb_type = getattr(config, "pos_emb_type", "divided_fixed")

        # If out_dim or patch_size are not provided, they are mandatory
        if self.out_dim is None:
            raise ValueError("hidden_size is required in config")
        if self.patch_size is None:
            raise ValueError("patch_size is required in config")

        assert isinstance(self.patch_size, int | Sequence), f"Invalid patch_size type: {type(self.patch_size)}"
        if isinstance(self.patch_size, int):
            self.patch_size = (self.patch_size, self.patch_size)
        assert len(self.patch_size) == 2, f"Expected patch_size to be a tuple of 2, got {self.patch_size}"

        self.proj = nn.Conv2d(
            self.in_dim,
            self.out_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            dtype=config.params_dtype,
        )

        if config.pos_emb_type == "divided_fixed":
            self.pos_emb = Learnable2DInterpPosEmbDivided_fixed(
                height=self.pos_emb_height,
                width=self.pos_emb_width,
                num_frames=self.pos_emb_time,
                dim=self.out_dim,
            )
        else:
            raise NotImplementedError(f"Not support pos_emb_type: {config.pos_emb_type}")

        assert sublayers_spec.rope_embedding is not None, "rotary_pos_emb must be specified"
        self.rotary_pos_emb = build_spec_layer(
            sublayers_spec.rope_embedding,
        )

    def forward(self, dict_args: dict) -> paddle.Tensor:
        """
        Args:
            x (L, Channels): ipaddleut tensor
            grid_hws (N, 3): temporal, height and width
        Returns:
            (L, Cout) tensor
        """
        pixel_values = dict_args["pixel_values"]
        grid_thws = dict_args["grid_thws"]

        hidden_states = self.proj(pixel_values).view(pixel_values.size(0), -1)
        # apply positional embedding
        hidden_states = self.pos_emb(hidden_states, grid_thws)

        rope_freqs_cis = self.rotary_pos_emb.get_freqs_cis(grid_thws)

        preproc_output = {
            "grid_thws": grid_thws,
            "hidden_states": hidden_states,
            "attention_mask": dict_args.get("attention_mask", None),
            "attn_mask_startend_row_indices": dict_args.get("attn_mask_startend_row_indices", None),
            "rope_freqs_cis": rope_freqs_cis,
        }
        return preproc_output
