# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2026 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
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

import math
from collections.abc import Sequence
from copy import deepcopy

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from transformers.utils import is_flash_attn_2_available

from paddleformers.transformers.deepseek_v3 import DeepseekV3ForCausalLM

from ...nn.attention.interface import ALL_ATTENTION_FUNCTIONS
from ..activations import PaddleGELUTanh
from ..cache_utils import Cache
from ..configuration_utils import PretrainedConfig
from ..model_utils import PretrainedModel
from .configuration import KimiK25Config

# Flash attention imports
if is_flash_attn_2_available():
    from flash_attn import flash_attn_varlen_func
else:
    flash_attn_varlen_func = None

from ..model_outputs import ModelOutput


class LlavaCausalLMOutputWithPast(ModelOutput):
    r"""
    loss (`paddle.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`paddle.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    image_hidden_states (`paddle.Tensor`, *optional*):
        A `paddle.Tensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
        image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    """

    loss: paddle.Tensor | None = None
    logits: paddle.Tensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple[paddle.Tensor] | None = None
    attentions: tuple[paddle.Tensor] | None = None
    image_hidden_states: paddle.Tensor | None = None


# def multihead_attention(
#     q: paddle.Tensor,
#     k: paddle.Tensor,
#     v: paddle.Tensor,
#     q_cu_seqlens: paddle.Tensor | None = None,
#     k_cu_seqlens: paddle.Tensor | None = None,
#     max_seqlen_q: int | None = None,
#     max_seqlen_k: int | None = None,
#     deterministic: bool = False,
# ):
#     """Multi-head attention using flash attention 2.
#     Args:
#         q, k, v: tensor of shape (batch_size, seqlen, num_heads, head_dim),
#             or (tot_seqlens, num_heads, head_dim) if packing.
#         q_cu_seqlens (paddle.Tensor): cumulative sequence lengths of q.
#             The first element should be 0 and the last element should be q.shape[0].
#         k_cu_seqlens (paddle.Tensor): cumulative sequence lengths of k.
#             The first element should be 0 and the last element should be k.shape[0].
#     Returns:
#         output: shape (batch_size, seqlen, dim) or (tot_seqlens, dim) if packing,
#             where dim = num_heads * head_dim
#     """
#     attn_out = flash_attn_varlen_func(
#         q,
#         k,
#         v,
#         q_cu_seqlens,
#         k_cu_seqlens,
#         max_seqlen_q,
#         max_seqlen_k,
#         causal=False,
#         deterministic=deterministic,
#     )
#     if isinstance(attn_out, tuple):
#         attn_out = attn_out[0]

#     attn_out = attn_out.flatten(start_dim=-2)

#     return attn_out


# def eager_attention(
#     q: paddle.Tensor,
#     k: paddle.Tensor,
#     v: paddle.Tensor,
#     q_cu_seqlens: Optional[paddle.Tensor] = None,
#     k_cu_seqlens: Optional[paddle.Tensor] = None,
#     **kwargs,
# ) -> paddle.Tensor:
#     seq_length = q.shape[0]
#     attention_mask = paddle.zeros([1, seq_length, seq_length],
#                                  device=q.device,
#                                  dtype=paddle.bool)
#     for i in range(1, len(q_cu_seqlens)):
#         attention_mask[
#             ...,
#             q_cu_seqlens[i - 1]:q_cu_seqlens[i],
#             q_cu_seqlens[i - 1]:q_cu_seqlens[i],
#         ] = True
#     q = q.transpose(0, 1)
#     k = k.transpose(0, 1)
#     v = v.transpose(0, 1)

#     attn_weight = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
#     attn_weight += attention_mask
#     attn_weight = paddle.softmax(attn_weight, dim=-1,
#                                 dtype=paddle.float32).to(q.dtype)

#     attn_output = attn_weight @ v
#     attn_output = attn_output.transpose(0, 1)
#     attn_output = attn_output.reshape(seq_length, -1)
#     return attn_output


# VL_VISION_ATTENTION_FUNCTIONS = {
#     "flash_attention_2": multihead_attention,
#     "eager": eager_attention,
# }


def _apply_rope_input_validation(x, freqs_cis):
    assert x.ndim == freqs_cis.ndim + 1, (x.shape, freqs_cis.shape)
    assert x.shape[:-2] == freqs_cis.shape[:-1], (x.shape, freqs_cis.shape)
    assert x.shape[-1] == 2 * freqs_cis.shape[-1], (x.shape, freqs_cis.shape)
    assert freqs_cis.dtype == paddle.complex64, freqs_cis.dtype


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


def apply_rope(xq: paddle.Tensor, xk: paddle.Tensor, freqs_cis: paddle.Tensor) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    Args: (The leading dimensions of all inputs should be the same)
        xq: query, tensor of shape (..., num_heads, head_dim)
        xk: key, tensor of shape (..., num_heads, head_dim)
        freqs_cis: tensor of shape (..., head_dim/2), dtype=paddle.complex64. It contains the precomputed cis(freqs) for each position in the 2D grid.
    Returns:
        xq_out, xk_out: tensors of shape (..., num_heads, head_dim)
    """
    _apply_rope_input_validation(xq, freqs_cis)
    _apply_rope_input_validation(xk, freqs_cis)

    freqs_cis = freqs_cis.unsqueeze(-2)  # ..., 1, head_dim/2
    # ..., num_heads, head_dim/2
    xq_ = paddle.view_as_complex(xq.float().view(*xq.shape[:-1], -1, 2))
    xk_ = paddle.view_as_complex(xk.float().view(*xq.shape[:-1], -1, 2))
    xq_out = paddle.view_as_real(xq_ * freqs_cis).flatten(-2)  # ..., num_heads, head_dim
    xk_out = paddle.view_as_real(xk_ * freqs_cis).flatten(-2)  # ..., num_heads, head_dim
    return xq_out.type_as(xq), xk_out.type_as(xk)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    From:
    https://github.com/OpenGVLab/InternVideo/blob/421f6d2361fc8f61a3394244571f2601a4e99e29/InternVideo2/multi_modality/models/backbones/internvideo2/pos_embed.py#L86
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, t_size, cls_token=False):
    """
    t_size: int of the temporal size
    return:
    pos_embed: [t_size, embed_dim] or [1+t_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_t = np.arange(t_size, dtype=np.float32)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid_t)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


class Learnable2DInterpPosEmbDivided_fixed(nn.Layer):
    def __init__(
        self, height: int, width: int, num_frames: int, dim: int, interpolation_mode: str = "bicubic"
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
            paddle.from_numpy(get_1d_sincos_pos_embed(self.dim, self.num_frames)).float().unsqueeze(1),
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


class MoonVision3dPatchEmbed(nn.Layer):
    def __init__(
        self,
        out_dim: int,
        in_dim: int = 3,
        patch_size: int | tuple[int, int] = (14, 14),
        pos_emb_height: int = 14,
        pos_emb_width: int = 14,
        pos_emb_time: int = 4,
        pos_emb_type: str = "divided_fixed",
    ):
        super().__init__()
        assert isinstance(patch_size, int | Sequence), f"Invalid patch_size type: {type(patch_size)}"
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        assert len(patch_size) == 2, f"Expected patch_size to be a tuple of 2, got {patch_size}"
        self.patch_size = patch_size

        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=patch_size, stride=patch_size)

        if pos_emb_type == "divided_fixed":
            self.pos_emb = Learnable2DInterpPosEmbDivided_fixed(
                height=pos_emb_height, width=pos_emb_width, num_frames=pos_emb_time, dim=out_dim
            )
        else:
            raise NotImplementedError(f"Not support pos_emb_type: {pos_emb_type}")

    def forward(self, x: paddle.Tensor, grid_thws: paddle.Tensor) -> paddle.Tensor:
        """
        Args:
            x (L, Channels): input tensor
            grid_hws (N, 3): temporal, height and width
        Returns:
            (L, Cout) tensor
        """
        x = self.proj(x).view(x.size(0), -1)
        # apply positional embedding
        x = self.pos_emb(x, grid_thws)
        return x


class Rope2DPosEmbRepeated(nn.Layer):
    """2D rotary position embedding with multi-resolution support.
    This class is intended to be used in the following way:
    1. Before training, create an instance of Rope2DPosEmb. This instance will hold the precomputed cis.
    2. Before each forward pass, call `get_freqs_cis_by_*` to get the `freqs_cis` tensor for this iteration.
    3. During the forward pass, pass the `freqs_cis` tensor to each attention layer, and call `apply` just before each attention operation.
        The rope is shared across all attention layers and all heads.
    Refs:
    - RoFormer: https://arxiv.org/abs/2104.09864
    - VisionLLaMA: https://arxiv.org/abs/2403.00522
    - https://github.com/Meituan-AutoML/VisionLLaMA/blob/main/dit/models.py
    Args:
        dim (int): usually the multi-head attention dimension, should be divisible by 4 (TODO: relax this constraint if needed)
        max_height (int): the maximum height of the 2D grid
        max_width (int): the maximum width of the 2D grid
        theta_base (float): the base of the theta
        device (str): the device to store the precomputed cis
    """

    def __init__(self, dim: int, max_height: int, max_width: int, theta_base=10000):
        super().__init__()
        self.dim = dim
        assert self.dim % 4 == 0, "dim must be divisible by 4"
        self.max_height = max_height
        self.max_width = max_width
        self.theta_base = theta_base

    def extra_repr(self):
        return (
            f"dim={self.dim}, max_height={self.max_height}, max_width={self.max_width}, theta_base={self.theta_base}"
        )

    def _precompute_freqs_cis(self, device: paddle.device) -> paddle.Tensor:
        """Calculate the cis(freqs) for each position in the 2D grid.
        Return: complex tensor of shape (max_height, max_width, dim//2) and value:
            height axis: ret[h, w, 2*i] = cis(h * theta_base**(-4*i/dim))
            weight axis: ret[h, w, 2*i+1] = cis(w * theta_base**(-4*i/dim))   with (i in [0, dim//4))
            note: `cis` is a mathematical notation defined by cis x = cos x + i sin x,
        """
        N = self.max_height * self.max_width
        flat_pos = paddle.arange(0, N).float().to(device)
        x_pos = flat_pos % self.max_width
        y_pos = flat_pos // self.max_width
        dim_range = paddle.arange(0, self.dim, 4)[: (self.dim // 4)].float().to(device)  # C/4
        freqs = 1.0 / (self.theta_base ** (dim_range / self.dim))
        x_freqs = paddle.outer(x_pos, freqs).float()  # N, C/4
        y_freqs = paddle.outer(y_pos, freqs).float()  # N, C/4
        x_cis = paddle.polar(paddle.ones_like(x_freqs), x_freqs)  # N, C/4
        y_cis = paddle.polar(paddle.ones_like(y_freqs), y_freqs)  # N, C/4
        # N, C/4, 2
        freqs_cis = paddle.cat([x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1)
        # max_height, max_width, C/2
        freqs_cis = freqs_cis.reshape(self.max_height, self.max_width, -1)
        return freqs_cis

    def get_freqs_cis(self, grid_thws: paddle.Tensor, device: paddle.device) -> paddle.Tensor:
        """
        Args:
            grid_thws (paddle.Tensor): grid time, height and width
        Returns:
            freqs_cis: tensor of shape (sum(t * height * width), dim//2)
        """
        if not hasattr(self, "freqs_cis"):
            self.register_buffer("freqs_cis", self._precompute_freqs_cis(device), persistent=False)

        shapes = grid_thws.tolist()
        assert all(1 <= h <= self.max_height and 1 <= w <= self.max_width for t, h, w in shapes), (
            shapes,
            self.max_height,
            self.max_width,
        )
        freqs_cis = paddle.cat(
            [self.freqs_cis[:h, :w].reshape(-1, self.dim // 2).repeat(t, 1) for t, h, w in shapes],
            dim=0,
        )
        return freqs_cis


class MLP2(nn.Layer):
    """
    Args:
        dims: [in_dim, hidden_dim, out_dim]
        bias: whether to use bias in linear layer.
    """

    def __init__(self, dims: list[int], activation, bias=True):
        super().__init__()
        assert len(dims) == 3
        self.fc0 = nn.Linear(dims[0], dims[1], bias_attr=bias)
        self.fc1 = nn.Linear(dims[1], dims[2], bias_attr=bias)
        self.activation = activation
        for m in [self.fc0, self.fc1]:
            nn.init.normal_(m.weight, mean=0.0, std=math.sqrt(2 / m.weight.shape[0]))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.fc0(x)
        x = self.activation(x)
        return self.fc1(x)


class MoonViTEncoderLayer(nn.Layer):
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        *,
        attn_implementation: str = "flash_attention_2",
        activation=F.gelu,
        attn_bias: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.hidden_size_per_attention_head = self.hidden_dim // self.num_heads
        self.attn_implementation = attn_implementation

        self.norm0 = nn.LayerNorm(hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP2([hidden_dim, mlp_dim, hidden_dim], activation)
        self.wqkv = nn.Linear(hidden_dim, hidden_dim * 3, bias_attr=attn_bias)
        self.wo = nn.Linear(hidden_dim, hidden_dim, bias_attr=attn_bias)

    def attention_qkvpacked(
        self,
        x: paddle.Tensor,
        cu_seqlens: paddle.Tensor,
        max_seqlen: paddle.Tensor,
        rope_freqs_cis: paddle.Tensor | None = None,
    ):
        """
        Args:
            x (paddle.Tensor): (batch_size, seqlen, hidden_dim)
            cu_seqlens (paddle.Tensor):
        """
        xqkv = self.wqkv(x)

        qkv_shape = xqkv.size()[:-1] + (
            3,
            self.num_heads,
            self.hidden_size_per_attention_head,
        )
        # xqkv: (batch_size, seqlen, 3, nheads, headdim)
        xqkv = xqkv.view(*qkv_shape)
        xq, xk, xv = paddle.unbind(xqkv, dim=-3)

        xq, xk = apply_rope(xq, xk, rope_freqs_cis)

        # FlashAttentionVarlen cu_seqlens to FlashMask mask
        cu_seqlens_rm_first = cu_seqlens[1:]
        cu_seqlens_rm_last = cu_seqlens[:-1]
        repeats = cu_seqlens_rm_first - cu_seqlens_rm_last

        startend_row_indices_lts = paddle.repeat_interleave(cu_seqlens_rm_first, repeats).reshape([1, 1, -1, 1])
        startend_row_indices_ute = paddle.repeat_interleave(cu_seqlens_rm_last, repeats).reshape([1, 1, -1, 1])
        startend_row_indices = paddle.concat([startend_row_indices_lts, startend_row_indices_ute], axis=-1)

        attn_func = ALL_ATTENTION_FUNCTIONS[self.attn_implementation]
        attn_out = attn_func(
            self,
            xq,
            xk,
            xv,
            attn_mask_startend_row_indices=startend_row_indices,
        )

        attn_out = self.wo(attn_out)
        return attn_out

    def forward(
        self,
        hidden_states: paddle.Tensor,
        cu_seqlens: paddle.Tensor,
        max_seqlen: int,
        rope_freqs_cis: paddle.Tensor | None = None,
    ):
        residual = hidden_states
        hidden_states = self.norm0(hidden_states)

        hidden_states = self.attention_qkvpacked(hidden_states, cu_seqlens, max_seqlen, rope_freqs_cis)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class MoonViT3dEncoder(nn.Layer):
    def __init__(
        self, hidden_dim: int, num_layers: int, block_cfg: dict, video_attn_type: str = "spatial_temporal"
    ) -> None:
        super().__init__()

        assert (
            video_attn_type == "spatial_temporal"
        ), f'video_attn_type must be "spatial_temporal", got {video_attn_type}'
        self.video_attn_type = video_attn_type
        self.rope_2d = Rope2DPosEmbRepeated(block_cfg["hidden_dim"] // block_cfg["num_heads"], 512, 512)
        self.blocks = nn.LayerList(
            [
                MoonViTEncoderLayer(
                    **block_cfg,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layernorm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        grid_thws: paddle.Tensor,
    ) -> paddle.Tensor:
        rope_freqs_cis = self.rope_2d.get_freqs_cis(grid_thws=grid_thws, device=hidden_states.device)

        lengths = paddle.cat(
            (
                paddle.zeros(1, dtype=grid_thws.dtype, device=grid_thws.device),
                grid_thws[:, 0] * grid_thws[:, 1] * grid_thws[:, 2],
            )
        )

        max_seqlen = lengths.max()
        cu_seqlens = lengths.to(hidden_states.device).cumsum(dim=0, dtype=paddle.int32)
        for block in self.blocks:
            hidden_states = block(hidden_states, cu_seqlens, max_seqlen, rope_freqs_cis=rope_freqs_cis)

        hidden_states = self.final_layernorm(hidden_states)
        return hidden_states


def tpool_patch_merger(
    x: paddle.Tensor,
    grid_thws: paddle.Tensor,
    merge_kernel_size: tuple[int, int] = (2, 2),
) -> list[paddle.Tensor]:
    d_model = x.size(-1)

    outputs = []
    pre_sum = 0
    for t, h, w in grid_thws.tolist():
        # Get the current sequence
        seq = x[pre_sum : pre_sum + t * h * w]
        # Reshape along self.merge_kernel_size and concat to the last dimension
        kernel_height, kernel_width = merge_kernel_size
        new_height, new_width = h // kernel_height, w // kernel_width
        reshaped_seq = seq.view(t, new_height, kernel_height, new_width, kernel_width, d_model)
        reshaped_seq = reshaped_seq.permute(0, 1, 3, 2, 4, 5).contiguous().mean(dim=0)  # temporal pooling
        padded_seq = reshaped_seq.view(new_height * new_width, kernel_height * kernel_width, -1)
        outputs.append(padded_seq)
        pre_sum += t * h * w

    return outputs


class MoonViT3dPretrainedModel(PretrainedModel):
    config_class = None
    model_type = "moonvit3d"
    _no_split_modules = ["PackingTransformer"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        config = deepcopy(config)
        self.merge_kernel_size = config.merge_kernel_size
        self.patch_size = config.patch_size
        self.merge_type = config.merge_type

        self.patch_embed = MoonVision3dPatchEmbed(
            out_dim=config.hidden_size,
            patch_size=config.patch_size,
            pos_emb_height=config.init_pos_emb_height,
            pos_emb_width=config.init_pos_emb_width,
            pos_emb_time=config.init_pos_emb_time,
            pos_emb_type=config.pos_emb_type,
        )

        self.encoder = MoonViT3dEncoder(
            hidden_dim=config.hidden_size,
            num_layers=config.num_hidden_layers,
            block_cfg={
                "num_heads": config.num_attention_heads,
                "hidden_dim": config.hidden_size,
                "mlp_dim": config.intermediate_size,
                "activation": PaddleGELUTanh(),
                "attn_bias": True,
                "attn_implementation": config._attn_implementation,
            },
            video_attn_type=config.video_attn_type,
        )

    def forward(self, pixel_values: paddle.Tensor, grid_thws: paddle.Tensor) -> paddle.Tensor:
        """
        Args:
            pixel_values (paddle.Tensor): The input pixel values.
            grid_thws (paddle.Tensor): Temporal, height and width.
        Returns:
            paddle.Tensor: The output tokens.
        """
        # grid_thws = grid_thws.to('cpu')
        assert grid_thws.ndim == 2, f"grid_thws should be 2D, got {grid_thws.ndim}"
        assert grid_thws.size(1) == 3, f"No support for thw: {grid_thws}"
        hidden_states = self.patch_embed(pixel_values, grid_thws)
        hidden_states = self.encoder(hidden_states, grid_thws)
        if self.merge_type == "sd2_tpool":  # spatial downsampling 2x with temporal pooling all
            hidden_states = tpool_patch_merger(hidden_states, grid_thws, merge_kernel_size=self.merge_kernel_size)
        else:
            raise NotImplementedError(f"Not support {self.merge_type}")

        return hidden_states


# ============================================================================
# MM Projector Helper Classes (from mm_projector/modeling_mm_projectors.py)
# ============================================================================


class IdentityMap(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x


class MLP(nn.Layer):
    def __init__(self, config):
        super().__init__()
        # TODO, use faster LayerNorm
        self.pre_norm = nn.LayerNorm(config.mm_hidden_size)
        self.proj = nn.Sequential(
            nn.Linear(config.mm_hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

    def forward(self, x, *args, **kwargs):
        assert isinstance(x, list | tuple), f"x is not a list or tuple: {type(x)}"
        lengths = [item.shape[0] for item in x]
        x = paddle.cat(x, dim=0)
        x = self.pre_norm(x)
        x = self.proj(x)
        x = paddle.split(x, lengths, dim=0)

        return x


class PatchMergerMLP(nn.Layer):
    def __init__(self, config):
        super().__init__()
        eps = config.projector_ln_eps
        self.hidden_size = config.mm_hidden_size * (config.merge_kernel_size[0] * config.merge_kernel_size[1])
        self.pre_norm = nn.LayerNorm(config.mm_hidden_size, eps=eps)
        self.proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, config.hidden_size),
        )

    def forward(self, x, *args, **kwargs):
        if isinstance(x, list) or isinstance(x, tuple):
            x = [self.proj(self.pre_norm(item).view(item.shape[0], -1)) for item in x]
        else:
            # B, N, N_k, C = x.shape
            B = x.shape[0]
            x = self.proj(self.pre_norm(x).view(B, -1, self.hidden_size))
        return x


class KimiK25PretrainedModel(PretrainedModel):
    config_class = KimiK25Config
    base_model_prefix = "model"
    _no_split_modules = [
        "MoonViT3dPretrainedModel",
        "MoonViTEncoderLayer",
        "DeepseekDecoderLayer",
        "PatchMergerMLP",
    ]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = False

    def _init_weights(self, module):
        # important: this ported version of Llava isn't meant for training from scratch - only
        # inference and fine-tuning - so the proper init weights code has been removed - the original codebase
        # https://github.com/haotian-liu/LLaVA/tree/main/llava should serve for that purpose
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class VisionTowerConfig(PretrainedConfig):
    model_type = "moonvit3d"

    def __init__(self, config: KimiK25Config, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = config.patch_size
        self.init_pos_emb_height = config.init_pos_emb_height
        self.init_pos_emb_width = config.init_pos_emb_width
        self.init_pos_emb_time = config.init_pos_emb_time
        self.pos_emb_type = config.pos_emb_type
        self.num_attention_heads = config.vt_num_attention_heads
        self.num_hidden_layers = config.vt_num_hidden_layers
        self.hidden_size = config.vt_hidden_size
        self.intermediate_size = config.vt_intermediate_size
        self.merge_kernel_size = config.merge_kernel_size
        self.video_attn_type = config.video_attn_type
        self.merge_type = config.merge_type
        self._attn_implementation = config._attn_implementation


class ProjectorConfig:
    def __init__(self, config: KimiK25Config):
        self.mm_projector_type = config.mm_projector_type
        self.mm_hidden_size = config.mm_hidden_size
        self.hidden_size = config.text_hidden_size
        self.merge_kernel_size = config.merge_kernel_size
        self.projector_hidden_act = config.projector_hidden_act
        self.projector_ln_eps = config.projector_ln_eps


# ref https://github.com/huggingface/transformers/blob/78b2929c0554b79e0489b451ce4ece14d265ead2/src/transformers/models/llava/modeling_llava.py#L240
class KimiK25ForConditionalGeneration(KimiK25PretrainedModel):
    def __init__(self, config: KimiK25Config):
        super().__init__(config)

        vt_config = VisionTowerConfig(config.vision_config)
        self.vision_tower = MoonViT3dPretrainedModel(vt_config)

        proj_config = ProjectorConfig(config.vision_config)
        if proj_config.mm_projector_type == "identity":
            self.mm_projector = IdentityMap()
        elif proj_config.mm_projector_type == "mlp":
            self.mm_projector = MLP(proj_config)
        elif proj_config.mm_projector_type == "patchmerger":
            self.mm_projector = PatchMergerMLP(proj_config)
        else:
            raise ValueError(f"Unsupported mm_projector_type: {proj_config.mm_projector_type}")

        self.language_model = DeepseekV3ForCausalLM(config.text_config)
        self.post_init()

        if hasattr(self.language_model, "dtype"):
            target_dtype = self.language_model.dtype
            self.vision_tower = self.vision_tower.to(dtype=target_dtype)
            self.mm_projector = self.mm_projector.to(dtype=target_dtype)

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def tie_weights(self):
        return self.language_model.tie_weights()

    def resize_token_embeddings(self, new_num_tokens: int | None = None, pad_to_multiple_of=None) -> nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds

    def _merge_input_ids_with_image_features(
        self,
        image_features: list[paddle.Tensor],
        inputs_embeds: paddle.Tensor,
        input_ids: paddle.Tensor,
        attention_mask: paddle.Tensor,
        labels: paddle.Tensor | None = None,
    ):
        """
        Args:
            image_features (:obj:`paddle.Tensor` of shape :obj:`(num_image_tokens, embed_dim)`):
                The image features to merge with the input embeddings.
            inputs_embeds (:obj:`paddle.Tensor` of shape :obj:`(batch_size, sequence_length, embed_dim)`):
                The input embeddings.
            input_ids (:obj:`paddle.Tensor` of shape :obj:`(batch_size, sequence_length)`):
                The input ids.
            attention_mask (:obj:`paddle.Tensor` of shape :obj:`(batch_size, sequence_length)`):
                The attention mask.
            labels (:obj:`paddle.Tensor` of shape :obj:`(batch_size, sequence_length)`, *optional*):
                The labels.
        """
        _, embed_dim = image_features[0].shape
        feature_lengths = [x.shape[0] for x in image_features]
        image_features = paddle.cat(image_features, dim=0)

        image_token_index: int = self.config.media_placeholder_token_id
        pad_token_id: int = self.config.pad_token_id
        ignore_index: int = self.config.ignore_index

        batch_size, sequence_length = input_ids.shape
        left_padding = not paddle.sum(input_ids[:, -1] == paddle.tensor(pad_token_id))

        # 1. Create a mask to know where special image tokens are
        _token_occupation_table = paddle.ones_like(input_ids.flatten())
        _token_occupation_table[input_ids.flatten() == image_token_index] = paddle.tensor(
            feature_lengths, dtype=paddle.long, device=input_ids.device
        )
        _token_occupation_table = _token_occupation_table.reshape(input_ids.shape)

        max_embed_dim = _token_occupation_table.sum(-1).max().item()
        assert (
            max_embed_dim >= sequence_length
        ), f"The maximum embedding dimension ({max_embed_dim}) is less than the sequence length ({sequence_length})"
        batch_indices, non_image_indices = paddle.where(input_ids != image_token_index)

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged image-text sequence.
        new_token_positions = paddle.cumsum(_token_occupation_table, -1) - 1
        nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_image_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = paddle.zeros(
            batch_size,
            max_embed_dim,
            embed_dim,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        final_attention_mask = paddle.zeros(
            batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device
        )
        if labels is not None:
            final_labels = paddle.full(
                (batch_size, max_embed_dim),
                ignore_index,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
        # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        batch_indices, non_image_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_image_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )
        attention_mask = attention_mask.to(target_device)

        # 4. Fill the embeddings based on the mask.
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]
        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_image_indices]

        # 5. Fill the embeddings corresponding to the images. Anything that is not `text_positions` needs filling (#29835)
        image_to_overwrite = paddle.full(
            (batch_size, max_embed_dim), True, dtype=paddle.bool, device=inputs_embeds.device
        )
        image_to_overwrite[batch_indices, text_to_overwrite] = False
        image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None].to(target_device)

        if image_to_overwrite.sum() != image_features.shape[:-1].numel():
            raise ValueError(
                f"The input provided to the model are wrong. The number of image tokens is {image_to_overwrite.sum()} while"
                f" the number of image features given to the model is {image_features.shape[:-1].numel()}. "
                "This prevents correct indexing and breaks batch generation."
            )

        final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim).to(target_device)
        final_attention_mask |= image_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

        # 6. Mask out the embedding at padding positions, as we later use the past_key_value value to determine the non-attended tokens.
        batch_indices, pad_indices = paddle.where(input_ids == pad_token_id)
        indices_to_mask = new_token_positions[batch_indices, pad_indices]

        final_embedding[batch_indices, indices_to_mask] = 0

        if labels is None:
            final_labels = None

        return final_embedding, final_attention_mask, final_labels, position_ids

    def _extract_image_features(self, pixel_values: paddle.Tensor, grid_thws: paddle.Tensor) -> list[paddle.Tensor]:
        """
        Args:
            pixel_values (:obj:`paddle.FloatTensor` of shape :obj:`(batch_size, num_channels, height, width)`):
                The pixel values of the images processed by image processor.
            grid_thws (:obj:`paddle.Tensor` of shape :obj:`(batch_size, 3)`):
                The grid, height, width of the images.
        Returns:
            selected_image_feature (:obj:`paddle.FloatTensor` of shape :obj:`(num_image_tokens, embed_dim)`):
                The selected image features to use as input to the projector head.
        """

        target_dtype = self.vision_tower.patch_embed.proj.weight.dtype
        pixel_values = pixel_values.to(target_dtype)

        image_features = self.vision_tower(pixel_values, grid_thws)
        return image_features

    def forward(
        self,
        input_ids: paddle.LongTensor | None = None,
        pixel_values: paddle.FloatTensor | list[paddle.FloatTensor] | None = None,
        grid_thws: paddle.Tensor | None = None,
        attention_mask: paddle.Tensor | None = None,
        position_ids: paddle.LongTensor | None = None,
        past_key_values: list[paddle.FloatTensor] | None = None,
        inputs_embeds: paddle.FloatTensor | None = None,
        labels: paddle.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple | LlavaCausalLMOutputWithPast:
        r"""
        Args:
            labels (`paddle.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        ```"""
        assert self.vision_tower is not None, "vision_tower is not loaded"
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            # 1. Extra the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # 2. Merge text and images
            if pixel_values is not None and len(pixel_values) > 0 and input_ids.shape[1] != 1:
                image_features = self._extract_image_features(pixel_values, grid_thws)
                if self.mm_projector:
                    image_features = self.mm_projector(image_features)

                inputs_embeds = inputs_embeds.to(image_features[0].dtype)  # num_tokens, embed_dim
                inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_image_features(
                    image_features,
                    inputs_embeds,
                    input_ids,
                    attention_mask,
                    labels,
                )

            # In case input_ids.shape[1] == 1 & pixel_values==None & past_key_values != None, we are in the case of
            # generation with cache
            elif past_key_values is not None and pixel_values is not None and input_ids.shape[1] == 1:
                # Retrieve the first layer to inspect the logits and mask out the hidden states
                # that are set to 0
                first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

                # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
                batch_index, non_attended_tokens = paddle.where(first_layer_past_key_value.float().sum(-2) == 0)

                # Get the target length
                target_length = input_ids.shape[1]
                past_length = first_layer_past_key_value.shape[-1]

                extended_attention_mask = paddle.ones(
                    (attention_mask.shape[0], past_length),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

                # Filter out only the tokens that can be un-attended, this can happen
                # if one uses Llava + Fused modules where the cache on the
                # first iteration is already big enough, or if one passes custom cache
                valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
                new_batch_index = batch_index[valid_indices]
                new_non_attended_tokens = non_attended_tokens[valid_indices]

                # Zero-out the places where we don't need to attend
                extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

                attention_mask = paddle.cat((extended_attention_mask, attention_mask[:, -target_length:]), dim=1)
                position_ids = paddle.sum(attention_mask, dim=1).unsqueeze(-1) - 1

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = outputs[0]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1).to(shift_logits.device),
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return LlavaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        grid_thws=None,
        attention_mask=None,
        **kwargs,
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = getattr(past_key_values, "seen_tokens", cache_length)
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.
            elif self.config.media_placeholder_token_id in input_ids:
                input_ids = input_ids[:, input_ids.shape[1] - 1 :]
            # If the cache has seen more tokens than it can hold, then the cache has a size limit. Let's discard the
            # older attention values, as their corresponding values are not part of the input.
            if cache_length < past_length and attention_mask is not None:
                attention_mask = attention_mask[:, -(cache_length + input_ids.shape[1]) :]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "grid_thws": grid_thws,
            }
        )
        return model_inputs

    def _reorder_cache(self, *args, **kwargs):
        return self.language_model._reorder_cache(*args, **kwargs)


__all__ = [
    "KimiK25ForConditionalGeneration",
]
