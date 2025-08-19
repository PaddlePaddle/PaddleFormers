# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import paddle
import paddle.distributed.fleet.meta_parallel as mpu
import paddle.nn.functional as F
from paddle import Tensor, nn
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker
from paddle.distributed.fleet.utils import recompute

from ...transformers import linear_utils
from ...utils.tools import get_env_device
from ..activations import ACT2FN
from ..configuration_utils import PretrainedConfig
from ..linear_utils import Linear
from ..model_outputs import BaseModelOutputWithPast, ModelOutput
from ..model_utils import PretrainedModel
from ..qwen2_vl.flash_attn_utils import has_flash_attn_func
from ..utils import logger
from .configuration import Qwen2_5_VLConfig, Qwen2_5_VLVisionConfig

flash_attn_func, flash_attn_varlen_func = has_flash_attn_func()


def get_triangle_upper_mask(x, mask=None):
    if mask is not None:
        return mask
    # [bsz, n_head, q_len, kv_seq_len]
    shape = x.shape
    #  [bsz, 1, q_len, kv_seq_len]
    shape[1] = 1
    mask = paddle.full(shape, paddle.finfo(x.dtype).min, dtype=x.dtype)
    mask = paddle.triu(mask, diagonal=1)
    mask.stop_gradient = True
    return mask


def parallel_matmul(x: Tensor, y: Tensor, transpose_y=True, tensor_parallel_output=True):
    is_fleet_init = True
    tensor_parallel_degree = 1
    try:
        hcg = fleet.get_hybrid_communicate_group()
        model_parallel_group = hcg.get_model_parallel_group()
        tensor_parallel_degree = hcg.get_model_parallel_world_size()
    except:
        is_fleet_init = False

    if paddle.in_dynamic_mode():
        y_is_distributed = y.is_distributed
    else:
        y_is_distributed = tensor_parallel_degree > 1

    if is_fleet_init and tensor_parallel_degree > 1 and y_is_distributed:

        # if not running under distributed.launch, it will raise AttributeError: 'Fleet' object has no attribute '_hcg'
        input_parallel = paddle.distributed.collective._c_identity(x, group=model_parallel_group)
        logits = paddle.matmul(input_parallel, y, transpose_y=transpose_y)

        if tensor_parallel_output:
            return logits
        return paddle.distributed.collective._c_concat(logits, group=model_parallel_group)

    else:
        logits = paddle.matmul(x, y, transpose_y=transpose_y)
        return logits


def _compute_default_rope_parameters(
    config: Optional[PretrainedConfig] = None,
    device: Optional["paddle.device"] = None,
    seq_len: Optional[int] = None,
    **rope_kwargs,
) -> Tuple["paddle.Tensor", float]:
    """
    Computes the inverse frequencies according to the original RoPE implementation
    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration.
        device (`paddle.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.
        rope_kwargs (`Dict`, *optional*):
            BC compatibility with the previous RoPE class instantiation, will be removed in v4.45.
    Returns:
        Tuple of (`paddle.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
    """
    if config is not None and len(rope_kwargs) > 0:
        raise ValueError(
            "Unexpected arguments: `**rope_kwargs` and `config` are mutually exclusive in "
            f"`_compute_default_rope_parameters`, got `rope_kwargs`={rope_kwargs} and `config`={config}"
        )
    if len(rope_kwargs) > 0:
        base = rope_kwargs["base"]
        dim = rope_kwargs["dim"]
    elif config is not None:
        base = config.rope_theta
        partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        dim = int(head_dim * partial_rotary_factor)

    attention_factor = 1.0  # Unused in this type of RoPE

    # Compute the inverse frequencies
    inv_freq = 1.0 / (base ** (paddle.arange(0, dim, 2, dtype="int64").astype("float32") / dim))
    return inv_freq, attention_factor


ROPE_INIT_FUNCTIONS = {
    "default": _compute_default_rope_parameters,
}


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(axis=-1, dtype="int32")
    indices = paddle.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()  # [2, 1, 1323]
    cu_seqlens = F.pad(paddle.cumsum(seqlens_in_batch, axis=0), (1, 0), data_format="NCL")
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def is_casual_mask(attention_mask):
    """
    Upper triangular of attention_mask equals to attention_mask is casual
    """
    return (paddle.triu(attention_mask) == attention_mask).all().item()


def _make_causal_mask(input_ids_shape, past_key_values_length):
    """
    Make causal mask used for self-attention
    """
    batch_size, target_length = input_ids_shape  # target_length: seq_len

    mask = paddle.tril(paddle.ones((target_length, target_length), dtype="bool"))

    if past_key_values_length > 0:
        # [tgt_len, tgt_len + past_len]
        mask = paddle.concat([paddle.ones([target_length, past_key_values_length], dtype="bool"), mask], axis=-1)

    # [bs, 1, tgt_len, tgt_len + past_len]
    return mask[None, None, :, :].expand([batch_size, 1, target_length, target_length + past_key_values_length])


def _expand_2d_mask(mask, dtype, tgt_length):
    """
    Expands attention_mask from `[batch_size, src_length]` to `[batch_size, 1, tgt_length, src_length]`.
    """
    batch_size, src_length = mask.shape[0], mask.shape[-1]
    tgt_length = tgt_length if tgt_length is not None else src_length

    mask = mask[:, None, None, :].astype("bool")
    mask.stop_gradient = True
    expanded_mask = mask.expand([batch_size, 1, tgt_length, src_length])

    return expanded_mask


@dataclass
class Qwen2_5_VLCausalLMOutputWithPast(ModelOutput):
    """
    Base class for Qwen2_5_VL causal language model (or autoregressive) outputs.

    Args:
        loss (`paddle.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`paddle.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(paddle.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(paddle.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(paddle.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `paddle.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(paddle.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `paddle.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        rope_deltas (`paddle.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
    """

    loss: Optional[paddle.Tensor] = None
    logits: paddle.float32 = None
    past_key_values: Optional[List[paddle.Tensor]] = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None
    attentions: Optional[Tuple[paddle.Tensor]] = None
    rope_deltas: Optional[paddle.Tensor] = None


class Qwen2_5_VLRotaryEmbedding(nn.Layer):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config: Optional[Qwen2_5_VLConfig] = None,
    ):
        super().__init__()
        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {}
        if config is None:
            logger.warning_once(
                "`Qwen2_5_VLRotaryEmbedding` can now be fully parameterized by passing the model config through the "
                "`config` argument. All other arguments will be removed in v4.46"
            )
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            # BC: "rope_type" was originally "type"
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        self.inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, **self.rope_kwargs)
        self.original_inv_freq = self.inv_freq
        self._set_cos_sin_cache(seq_len=max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        # [seq_len]
        t = paddle.arange(seq_len, dtype="float32")
        # [seq_len, dim/2]
        freqs = paddle.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        # [seq_len, dim]
        emb = paddle.concat([freqs, freqs], axis=-1)
        # [1, seqlen, 1, dim]
        self.cos_cached = emb.cos()
        self.sin_cached = emb.sin()

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = paddle.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.inv_freq = inv_freq
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.inv_freq = self.original_inv_freq
            self.max_seq_len_cached = self.original_max_seq_len

    @paddle.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block. In contrast to other models, Qwen2_VL has different position ids for thw grids
        # So we expand the inv_freq to shape (3, ...)
        inv_freq_expanded = (
            self.inv_freq[None, None, :, None].astype("float32").expand([3, position_ids.shape[1], -1, 1])
        )
        position_ids_expanded = position_ids[:, :, None, :].astype("float32")  # shape (3, bs, 1, positions)
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = paddle.get_device()
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with paddle.amp.auto_cast(enable=False):
            # Compute frequencies by matrix multiplication and transpose
            # inv_freq_expanded shape: [3, bs, dim/2, 1]
            # position_ids_expanded shape: [3, bs, 1, positions]
            # Result shape after matmul: [3, bs, dim/2, positions]
            # After transpose: [3, bs, positions, dim/2]
            freqs = paddle.matmul(inv_freq_expanded, position_ids_expanded)
            freqs = freqs.transpose([0, 1, 3, 2])
            emb = paddle.concat((freqs, freqs), axis=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.astype(x.dtype), sin.astype(x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.concat([-x2, x1], axis=-1)  # shape is the same as x


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

    # cos = cos[position_ids]
    # sin = sin[position_ids]
    mrope_section = mrope_section * 2
    cos = paddle.concat(x=[m[i % 3] for i, m in enumerate(cos.split(mrope_section, axis=-1))], axis=-1).unsqueeze(
        axis=unsqueeze_dim
    )
    sin = paddle.concat(x=[m[i % 3] for i, m in enumerate(sin.split(mrope_section, axis=-1))], axis=-1).unsqueeze(
        axis=unsqueeze_dim
    )

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb_vision(tensor: paddle.Tensor, freqs: paddle.Tensor) -> paddle.Tensor:
    orig_dtype = tensor.dtype

    with paddle.amp.auto_cast(False):
        tensor = tensor.astype(dtype="float32")
        cos = freqs.cos()
        sin = freqs.sin()
        cos = cos.unsqueeze(1).tile(repeat_times=[1, 1, 2]).unsqueeze(0).astype(dtype="float32")
        sin = sin.unsqueeze(1).tile(repeat_times=[1, 1, 2]).unsqueeze(0).astype(dtype="float32")
        output = tensor * cos + rotate_half(tensor) * sin
    output = paddle.cast(output, orig_dtype)
    return output


class Qwen2_5_VisionRotaryEmbedding(nn.Layer):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.inv_freq = 1.0 / theta ** (paddle.arange(start=0, end=dim, step=2, dtype="float32") / dim)

    def forward(self, seqlen: int) -> paddle.Tensor:
        seq = paddle.arange(seqlen).cast(self.inv_freq.dtype)
        freqs = paddle.outer(x=seq, y=self.inv_freq)
        return freqs


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
        self.proj = nn.Conv3D(in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias_attr=False)

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:

        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.reshape(
            [-1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size]
        )

        # NOTE（changwenbin）: AttributeError: 'Variable' object has no attribute 'to'.
        # hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).reshape([-1, self.embed_dim])
        hidden_states = self.proj(paddle.cast(hidden_states, dtype=target_dtype)).reshape([-1, self.embed_dim])
        return hidden_states


class Qwen2_5_VLPatchMerger(paddle.nn.Layer):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = Qwen2RMSNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.mlp(self.ln_q(x).reshape([-1, self.hidden_size]))
        return x


class Qwen2_5_VLMLP(paddle.nn.Layer):
    def __init__(self, config, bias: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = paddle.nn.Linear(
            in_features=self.hidden_size, out_features=self.intermediate_size, bias_attr=bias
        )
        self.up_proj = paddle.nn.Linear(
            in_features=self.hidden_size, out_features=self.intermediate_size, bias_attr=bias
        )
        self.down_proj = paddle.nn.Linear(
            in_features=self.intermediate_size, out_features=self.hidden_size, bias_attr=bias
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class Qwen2_5_VLVisionAttention(nn.Layer):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias_attr=True)
        self.proj = nn.Linear(dim, dim)
        self.head_dim = dim // num_heads  # must added

    def forward(
        self, hidden_states: paddle.Tensor, cu_seqlens: paddle.Tensor, rotary_pos_emb: paddle.Tensor = None
    ) -> paddle.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = (
            self.qkv(hidden_states).reshape([seq_length, 3, self.num_heads, -1]).transpose([1, 0, 2, 3]).unbind(0)
        )
        q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

        attention_mask = paddle.zeros([1, seq_length, seq_length], dtype="bool")
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = True

        zero = paddle.zeros(attention_mask.shape, dtype=hidden_states.dtype)
        neg_inf = paddle.full_like(attention_mask, paddle.finfo(hidden_states.dtype).min, dtype=hidden_states.dtype)
        attention_mask = paddle.where(attention_mask, zero, neg_inf)

        q = q.transpose([1, 0, 2])
        k = k.transpose([1, 0, 2])
        v = v.transpose([1, 0, 2])
        attn_weights = paddle.matmul(q, k.transpose([0, 2, 1])) / math.sqrt(self.head_dim)
        attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, axis=-1)
        attn_output = paddle.matmul(attn_weights, v)
        attn_output = attn_output.transpose([1, 0, 2])
        attn_output = attn_output.reshape([seq_length, -1])
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen2_5_VLVisionFlashAttention2(nn.Layer):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias_attr=True)
        self.proj = nn.Linear(dim, dim)
        self.head_dim = dim // num_heads  # must added

    def forward(
        self, hidden_states: paddle.Tensor, cu_seqlens: paddle.Tensor, rotary_pos_emb: paddle.Tensor = None
    ) -> paddle.Tensor:
        seq_length = tuple(hidden_states.shape)[0]
        qkv = self.qkv(hidden_states).reshape([seq_length, 3, self.num_heads, -1]).transpose(perm=[1, 0, 2, 3])
        q, k, v = qkv.unbind(axis=0)
        q = apply_rotary_pos_emb_flashatt(q.unsqueeze(axis=0), rotary_pos_emb).squeeze(axis=0)
        k = apply_rotary_pos_emb_flashatt(k.unsqueeze(axis=0), rotary_pos_emb).squeeze(axis=0)
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        softmax_scale = self.head_dim**-0.5  # TODO: 需要手动加上
        data_type = q.dtype
        attn_output = (
            flash_attn_varlen_func(  # flash_attn_unpadded
                q.astype("bfloat16"),  # 不支持float32
                k.astype("bfloat16"),
                v.astype("bfloat16"),
                cu_seqlens,
                cu_seqlens,
                max_seqlen,
                max_seqlen,
                scale=softmax_scale,  # TODO: 需要手动加上
            )[0]
            .squeeze(0)
            .reshape([seq_length, -1])
        )
        attn_output = attn_output.astype(data_type)
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen2_5_VLVisionSdpaAttention(nn.Layer):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias_attr=True)
        self.proj = nn.Linear(dim, dim)

        is_bfloat16_supported = paddle.amp.is_bfloat16_supported()
        if is_bfloat16_supported:
            self.compute_dtype = "bfloat16"
        else:
            self.compute_dtype = "float16"

    def forward(
        self, hidden_states: paddle.Tensor, cu_seqlens: paddle.Tensor, rotary_pos_emb: paddle.Tensor = None
    ) -> paddle.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = (
            self.qkv(hidden_states).reshape([seq_length, 3, self.num_heads, -1]).transpose([1, 0, 2, 3]).unbind(0)
        )
        q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb)
        k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb)
        attention_mask = paddle.zeros([1, 1, seq_length, seq_length], dtype="bool")
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = True

        zero = paddle.zeros(attention_mask.shape, dtype=hidden_states.dtype)
        neg_inf = paddle.full_like(attention_mask, paddle.finfo(hidden_states.dtype).min, dtype=hidden_states.dtype)
        attention_mask = paddle.where(attention_mask, zero, neg_inf)
        v = v.unsqueeze(0)

        attn_output = paddle.nn.functional.scaled_dot_product_attention(
            query=q.astype(self.compute_dtype),
            key=k.astype(self.compute_dtype),
            value=v.astype(self.compute_dtype),
            attn_mask=attention_mask.astype(self.compute_dtype),
            dropout_p=0.0,
        )

        attn_output = attn_output.transpose([1, 0, 2])
        attn_output = attn_output.reshape([seq_length, -1])
        attn_output = self.proj(attn_output)

        return attn_output


QWEN2_5_VL_VISION_ATTENTION_CLASSES = {
    "eager": Qwen2_5_VLVisionAttention,
    "flash_attention_2": Qwen2_5_VLVisionFlashAttention2,
    "sdpa": Qwen2_5_VLVisionSdpaAttention,
}


class Qwen2_5_VLVisionBlock(paddle.nn.Layer):
    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        self.norm1 = Qwen2RMSNorm(config.hidden_size, eps=1e-6)
        self.norm2 = Qwen2RMSNorm(config.hidden_size, eps=1e-6)

        if flash_attn_func is None and attn_implementation == "flash_attention_2":
            attn_implementation = "eager"
            logger.warning_once(f"Warning: Flash Attention2 is not available for vision module, fallback to normal attention.")

        self.attn = QWEN2_5_VL_VISION_ATTENTION_CLASSES[attn_implementation](
            config.hidden_size, num_heads=config.num_heads
        )

        self.mlp = Qwen2_5_VLMLP(config, bias=True)

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb) -> paddle.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


def apply_rotary_emb(tensor, cos, sin):
    """
    Apply rotary position embedding to the input tensor.
    Args:
        tensor (paddle.Tensor): The input tensor of shape [batch_size, seq_len, num_heads, head_dim]
        cos (paddle.Tensor): The cosine part of the rotary embedding [seq_len, head_dim/2]
        sin (paddle.Tensor): The sine part of the rotary embedding [seq_len, head_dim/2]
    Returns:
        paddle.Tensor: The tensor after applying rotary embedding
    """
    # Split the tensor into two halves along the last dimension
    dim = tensor.shape[-1]
    half_dim = dim // 2
    tensor1 = tensor[..., :half_dim]
    tensor2 = tensor[..., half_dim:]

    # Reshape cos/sin for broadcasting
    # From [seq_len, head_dim/2] to [1, seq_len, 1, head_dim/2]
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)

    # Apply rotary embedding
    # tensor1/tensor2 shape: [batch_size, seq_len, num_heads, head_dim/2]
    # cos/sin shape: [1, seq_len, 1, head_dim/2]
    rotated = paddle.concat([tensor1 * cos - tensor2 * sin, tensor1 * sin + tensor2 * cos], axis=-1)

    return rotated


def apply_rotary_pos_emb_flashatt(tensor: paddle.Tensor, freqs: paddle.Tensor) -> paddle.Tensor:
    tensor_ = tensor.astype(dtype="float32")
    cos = freqs.cos()
    sin = freqs.sin()
    output = apply_rotary_emb(tensor_, cos, sin).astype(dtype=tensor.dtype)
    return output


class Qwen2RMSNorm(nn.Layer):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = paddle.create_parameter(
            shape=[hidden_size],
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        if paddle.in_dynamic_mode():
            with paddle.amp.auto_cast(False):
                variance = hidden_states.astype("float32").pow(2).mean(-1, keepdim=True)
                hidden_states = paddle.rsqrt(variance + self.variance_epsilon) * hidden_states
        else:
            variance = hidden_states.astype("float32").pow(2).mean(-1, keepdim=True)
            hidden_states = paddle.rsqrt(variance + self.variance_epsilon) * hidden_states

        if self.weight.dtype in [paddle.float16, paddle.bfloat16]:
            hidden_states = paddle.cast(hidden_states, self.weight.dtype)
        return hidden_states * self.weight


class Qwen2MLP(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.fuse_attention_ffn = config.fuse_attention_ffn
        self.tensor_parallel_degree = config.tensor_parallel_degree

        # else:
        ColumnParallelLinear = linear_utils.ColumnParallelLinear
        RowParallelLinear = linear_utils.RowParallelLinear

        if config.tensor_parallel_degree > 1:

            self.gate_proj = ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                gather_output=False,
                has_bias=False,
            )
            self.up_proj = ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                gather_output=False,
                has_bias=False,
            )
            self.down_proj = RowParallelLinear(
                self.intermediate_size,
                self.hidden_size,
                input_is_parallel=True,
                has_bias=False,
            )
        else:
            if get_env_device() == "xpu":
                self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias_attr=False)  # w1
                self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias_attr=False)  # w3
                self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias_attr=False)  # w2
            else:
                self.gate_proj = Linear(self.hidden_size, self.intermediate_size, bias_attr=False)  # w1
                self.up_proj = Linear(self.hidden_size, self.intermediate_size, bias_attr=False)  # w3
                self.down_proj = Linear(self.intermediate_size, self.hidden_size, bias_attr=False)  # w2

        self.act_fn = ACT2FN[config.hidden_act]
        self.fuse_swiglu = False

    def forward(self, x):
        x, y = self.gate_proj(x), self.up_proj(x)
        if self.fuse_swiglu:
            x = self.act_fn(x, y)
        else:
            x = self.act_fn(x) * y

        return self.down_proj(x)


def repeat_kv(hidden_states: paddle.Tensor, n_rep: int) -> paddle.Tensor:
    """
    This is the equivalent of paddle.repeat_interleave(x, axis=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand([batch, num_key_value_heads, n_rep, slen, head_dim])
    return hidden_states.reshape([batch, num_key_value_heads * n_rep, slen, head_dim])


class Qwen2_5_VLAttention(paddle.nn.Layer):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: Qwen2_5_VLConfig, layer_idx: Optional[int] = None):
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
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        self.rope_scaling = config.rope_scaling
        # self.sequence_parallel = config.sequence_parallel

        if config.tensor_parallel_degree > 1:
            assert (
                self.num_heads % config.tensor_parallel_degree == 0
            ), f"num_heads: {self.num_heads}, tensor_parallel_degree: {config.tensor_parallel_degree}"
            self.num_heads = self.num_heads // config.tensor_parallel_degree

            assert (
                self.num_key_value_heads % config.tensor_parallel_degree == 0
            ), f"num_key_value_heads: {self.num_key_value_heads}, tensor_parallel_degree: {config.tensor_parallel_degree}"
            self.num_key_value_heads = self.num_key_value_heads // config.tensor_parallel_degree

        ColumnParallelLinear = linear_utils.ColumnParallelLinear
        RowParallelLinear = linear_utils.RowParallelLinear

        if config.tensor_parallel_degree > 1:
            self.q_proj = ColumnParallelLinear(self.hidden_size, self.hidden_size, has_bias=True, gather_output=False)
            self.k_proj = ColumnParallelLinear(self.hidden_size, self.config.num_key_value_heads * self.head_dim, has_bias=True, gather_output=False)  # fmt:skip
            self.v_proj = ColumnParallelLinear(self.hidden_size, self.config.num_key_value_heads * self.head_dim, has_bias=True, gather_output=False)  # fmt:skip
            self.o_proj = RowParallelLinear(self.hidden_size, self.hidden_size, has_bias=False, input_is_parallel=True)
        else:
            self.q_proj = Linear(self.hidden_size, self.hidden_size, bias_attr=True)
            self.k_proj = Linear(self.hidden_size, self.config.num_key_value_heads * self.head_dim, bias_attr=True)
            self.v_proj = Linear(self.hidden_size, self.config.num_key_value_heads * self.head_dim, bias_attr=True)
            self.o_proj = Linear(self.hidden_size, self.hidden_size, bias_attr=False)

        self.rotary_emb = Qwen2_5_VLRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,  # Cache
        output_attentions: bool = False,
        use_cache: bool = False,  # default true
        cache_position: Optional[paddle.Tensor] = None,
        position_embeddings: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,  # necessary, but kept here for BC
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        bsz, q_len, _ = hidden_states.shape

        try:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        except:
            hidden_states = hidden_states.astype(self.config.dtype)
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        target_query_shape = [0, 0, self.num_heads, self.head_dim]
        target_key_value_shape = [0, 0, self.num_key_value_heads, self.head_dim]
        query_states = query_states.reshape(shape=target_query_shape)
        key_states = key_states.reshape(shape=target_key_value_shape)
        value_states = value_states.reshape(shape=target_key_value_shape)

        new_perm = [0, 2, 1, 3]
        query_states = query_states.transpose(new_perm)
        key_states = key_states.transpose(new_perm)
        value_states = value_states.transpose(new_perm)

        kv_seq_len = key_states.shape[-2]  # q_len ######## [bs, num_head, seq_len, head_dim]      # qwen2是 [-3]
        if past_key_value is not None:
            kv_seq_len += cache_position[0] + 1
            # kv_seq_len += past_key_value[0].shape[-2] # qwen2是 [-3]

        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        # [bs, num_head, seq_len, head_dim]
        if past_key_value is not None:
            # cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            # key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            key_states = paddle.concat([past_key_value[0], key_states], axis=2)  # qwen2是 axis=1, qwen2_vl是 axis=2
            value_states = paddle.concat([past_key_value[1], value_states], axis=2)  # qwen2是 axis=1
        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        query_states = query_states.astype("float32")
        key_states = key_states.astype("float32")
        value_states = value_states.astype("float32")

        attn_weights = paddle.matmul(query_states, key_states.transpose([0, 1, 3, 2])) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, axis=-1)
        attn_weights = nn.functional.dropout(x=attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = paddle.matmul(attn_weights.cast(self.config.dtype), value_states.cast(self.config.dtype))

        if attn_output.shape != [bsz, self.num_heads, q_len, self.head_dim]:
            raise ValueError(
                f"`attn_output` should be of size {(bsz, q_len, self.num_heads, self.head_dim)}, but is"
                f" {attn_output.shape}"
            )

        attn_output = attn_output.transpose([0, 2, 1, 3])
        attn_output = attn_output.reshape([bsz, q_len, -1])
        attn_output = self.o_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value


class Qwen2_5_VLFlashAttention2(Qwen2_5_VLAttention):
    """
    Qwen2_5_VL flash attention module, following Qwen2_5_VL attention module. This module inherits from `Qwen2_5_VLAttention`
    as the weights of the module stays untouched. The only required change would be on the forward pass
    where it needs to correctly call the public API of flash attention and deal with padding tokens
    in case the input contains any of them. Additionally, for sliding window attention, we apply SWA only to the bottom
    config.max_window_layers layers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,  # Cache
        output_attentions: bool = False,
        use_cache: bool = False,  # default true
        cache_position: Optional[paddle.Tensor] = None,
        position_embeddings: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,  # necessary, but kept here for BC
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        bsz, q_len, _ = tuple(hidden_states.shape)
        try:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        except:
            hidden_states = hidden_states.astype("bfloat16")
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        target_query_shape = [0, 0, self.num_heads, self.head_dim]
        target_key_value_shape = [0, 0, self.num_key_value_heads, self.head_dim]
        query_states = query_states.reshape(shape=target_query_shape)
        key_states = key_states.reshape(shape=target_key_value_shape)
        value_states = value_states.reshape(shape=target_key_value_shape)

        new_perm = [0, 2, 1, 3]
        # [1, 3599, 1536] [bsz, q_len, self.num_heads * self.head_dim]
        query_states = query_states.transpose(new_perm)
        key_states = key_states.transpose(new_perm)
        value_states = value_states.transpose(new_perm)

        kv_seq_len = key_states.shape[-2]  # q_len ######## [bs, num_head, seq_len, head_dim]      # qwen2是 [-3]
        if past_key_value is not None:
            kv_seq_len += cache_position[0] + 1

        # Because the input can be padded, the absolute sequence length depends on the max position id.
        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        if past_key_value is not None:
            key_states = paddle.concat([past_key_value[0], key_states], axis=2)  # qwen2是 axis=1, qwen2_vl是 axis=2
            value_states = paddle.concat([past_key_value[1], value_states], axis=2)  # qwen2是 axis=1
        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=self.attention_dropout if self.training else 0.0,
            is_causal=self.is_causal,
        )

        attn_output = attn_output.reshape([bsz, q_len, -1])
        attn_output = self.o_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length: int,
        is_causal: bool,
        dropout: float = 0.0,
        softmax_scale: Optional[float] = None,
        use_top_left_mask: bool = False,
        softcap: Optional[float] = None,
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`paddle.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`paddle.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`paddle.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`paddle.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`float`):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
            use_top_left_mask (`bool`, defaults to `False`):
                flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference.
            softcap (`float`, *optional*):
                Softcap for the attention logits, used e.g. in gemma2.
            deterministic (`bool`, *optional*):
                Determines if the deterministic option introduced in flash_attn>=2.4.1 is enabled.
        """
        if not use_top_left_mask:
            causal = is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1.
            causal = is_causal and query_length != 1

        # Assuming 4D tensors, key_states.shape[1] is the key/value sequence length (source length).
        flash_kwargs = {}

        if softcap is not None:
            flash_kwargs["softcap"] = softcap

        datatype = query_states.dtype
        # query_states diff!
        if attention_mask is not None:
            # Reashape to the expected shape for Flash Attention
            # [1, 3599, 12, 128]
            query_states = query_states.transpose(perm=[0, 2, 1, 3])
            key_states = key_states.transpose(perm=[0, 2, 1, 3])
            value_states = value_states.transpose(perm=[0, 2, 1, 3])

            assert query_states.shape[0] == key_states.shape[0] == value_states.shape[0] == 1
            query_states = query_states.squeeze(0)
            key_states = key_states.squeeze(0)
            value_states = value_states.squeeze(0)
            cu_seqlens = attention_mask
            head_dim = query_states.shape[-1]
            softmax_scale = head_dim**-0.5
            with paddle.no_grad():
                max_seqlen = max(
                    [cu_seqlens[idx + 1] - cu_seqlens[idx] for idx in range(cu_seqlens.shape[0] - 1)]
                ).item()
            attn_output = flash_attn_varlen_func(
                query_states.cast("bfloat16"),
                key_states.cast("bfloat16"),
                value_states.cast("bfloat16"),
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                dropout=dropout,
                scale=softmax_scale,
                causal=causal,
            )[0]
            attn_output = attn_output.unsqueeze(0)
            query_states = query_states.unsqueeze(0)
            key_states = key_states.unsqueeze(0)
            value_states = value_states.unsqueeze(0)
            attn_output = attn_output.cast(datatype)
        elif query_states.shape[2] == key_states.shape[2]:
            # Reashape to the expected shape for Flash Attention
            # [1, 3599, 12, 128]
            query_states = query_states.transpose(perm=[0, 2, 1, 3])
            key_states = key_states.transpose(perm=[0, 2, 1, 3])
            value_states = value_states.transpose(perm=[0, 2, 1, 3])

            attn_output = flash_attn_func(
                query_states.cast("bfloat16"),
                key_states.cast("bfloat16"),
                value_states.cast("bfloat16"),
                dropout,
                softmax_scale=softmax_scale,
                causal=causal,
                **flash_kwargs,
            )[0]
            attn_output = attn_output.cast(datatype)
        else:
            attn_weights = paddle.matmul(query_states, key_states.transpose([0, 1, 3, 2])) / math.sqrt(self.head_dim)
            attn_weights = nn.functional.softmax(attn_weights, axis=-1)
            attn_weights = nn.functional.dropout(x=attn_weights, p=self.attention_dropout, training=self.training)
            attn_output = paddle.matmul(attn_weights.cast(self.config.dtype), value_states.cast(self.config.dtype))
            attn_output = attn_output.transpose([0, 2, 1, 3])

        return attn_output


class Qwen2_5_VLSdpaAttention(Qwen2_5_VLAttention):
    """
    Qwen2 attention module using paddle.nn.functional.scaled_dot_product_attention. This module inherits from
    `Qwen2Attention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[paddle.Tensor] = None,
        position_embeddings: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        if output_attentions:
            logger.warning_once(
                'Qwen2_5_VLModel is using Qwen2_5_VLSdpaAttention, but `paddle.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
        bsz, q_len, _ = hidden_states.shape

        try:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        except:
            hidden_states = hidden_states.astype(self.config.dtype)
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        target_query_shape = [0, 0, self.num_heads, self.head_dim]
        target_key_value_shape = [0, 0, self.num_key_value_heads, self.head_dim]
        query_states = query_states.reshape(shape=target_query_shape)
        key_states = key_states.reshape(shape=target_key_value_shape)
        value_states = value_states.reshape(shape=target_key_value_shape)

        new_perm = [0, 2, 1, 3]
        query_states = query_states.transpose(new_perm)
        key_states = key_states.transpose(new_perm)
        value_states = value_states.transpose(new_perm)

        kv_seq_len = key_states.shape[-2]  # q_len ######## [bs, num_head, seq_len, head_dim]      # qwen2是 [-3]
        if past_key_value is not None:
            kv_seq_len += cache_position[0] + 1
            # kv_seq_len += past_key_value[0].shape[-2] # qwen2是 [-3]

        # Because the input can be padded, the absolute sequence length depends on the max position id.
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        if past_key_value is not None:
            # cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            # key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            key_states = paddle.concat([past_key_value[0], key_states], axis=2)  # qwen2是 axis=1, qwen2_vl是 axis=2
            value_states = paddle.concat([past_key_value[1], value_states], axis=2)  # qwen2是 axis=1
        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Reashape to the expected shape for Flash Attention
        # [1, 3599, 12, 128]
        query_states = query_states.transpose(perm=[0, 2, 1, 3])
        key_states = key_states.transpose(perm=[0, 2, 1, 3])
        value_states = value_states.transpose(perm=[0, 2, 1, 3])

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        attention_mask = None
        causal_mask = attention_mask
        # Convert attention mask slicing
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-3]]

        # Ensure contiguous tensors for PaddlePaddle
        if query_states.place.is_gpu_place() and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # Determine if the operation is causal
        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = paddle.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.reshape([bsz, q_len, -1])

        # Apply the output projection
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


QWEN2_5_VL_ATTENTION_CLASSES = {
    "eager": Qwen2_5_VLAttention,
    "flash_attention_2": Qwen2_5_VLFlashAttention2,
    "sdpa": Qwen2_5_VLSdpaAttention,
}


class Qwen2_5_VLDecoderLayer(nn.Layer):
    def __init__(self, config: Qwen2_5_VLConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        # use_sliding_window false
        if config.use_sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )
        
        if flash_attn_func is None and config._attn_implementation == "flash_attention_2":
            attn_implementation = "eager"
            logger.warning_once(f"Warning: Flash Attention2 is not available for decoder layer, fallback to normal attention.")
        else:
            attn_implementation = config._attn_implementation

        self.self_attn = QWEN2_5_VL_ATTENTION_CLASSES[attn_implementation](config, layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[paddle.Tensor] = None,
        position_embeddings: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ):
        """
        Args:
            hidden_states (`paddle.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`paddle.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(paddle.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`paddle.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`Tuple[paddle.FloatTensor, paddle.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
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

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class Qwen2_5_VLPreTrainedModel(PretrainedModel):
    config_class = Qwen2_5_VLConfig
    base_model_prefix = "model"
    _no_split_modules = ["Qwen2_5_VLDecoderLayer", "Qwen2_5_VLVisionBlock"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, layer):
        std = 0.2
        if isinstance(layer, (nn.Linear, nn.Conv3D)):
            nn.initializer.Normal(mean=0.0, std=std)(layer.weight)
            if layer.bias is not None:
                nn.initializer.Constant(0.0)(layer.bias)
        elif isinstance(layer, nn.Embedding):
            nn.initializer.Normal(mean=0.0, std=std)(layer.weight)
            if layer._padding_idx is not None:
                with paddle.no_grad():
                    layer.weight[layer._padding_idx] = 0.0


class Qwen2_5_VisionTransformerPretrainedModel(Qwen2_5_VLPreTrainedModel):
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
        self.blocks = nn.LayerList(
            sublayers=[Qwen2_5_VLVisionBlock(config, config._attn_implementation) for _ in range(config.depth)]
        )
        self.merger = Qwen2_5_VLPatchMerger(
            dim=config.out_hidden_size, context_dim=config.hidden_size, spatial_merge_size=config.spatial_merge_size
        )
        self.enable_recompute = False

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
        pos_ids = paddle.concat(x=pos_ids, axis=0)
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
            llm_grid_h, llm_grid_w = (grid_h // self.spatial_merge_size, grid_w // self.spatial_merge_size)
            index = paddle.arange(end=grid_t * llm_grid_h * llm_grid_w).reshape([grid_t, llm_grid_h, llm_grid_w])
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = paddle.nn.functional.pad(
                x=index, pad=(0, pad_w, 0, pad_h), mode="constant", value=-100, pad_from_left_axis=False
            )
            index_padded = index_padded.reshape(
                [grid_t, num_windows_h, vit_merger_window_size, num_windows_w, vit_merger_window_size]
            )
            index_padded = index_padded.transpose(perm=[0, 1, 3, 2, 4]).reshape(
                [grid_t, num_windows_h * num_windows_w, vit_merger_window_size, vit_merger_window_size]
            )
            seqlens = (index_padded != -100).sum(axis=[2, 3]).reshape([-1])
            index_padded = index_padded.reshape([-1])
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = seqlens.cumsum(axis=0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = paddle.concat(x=window_index, axis=0)
        return window_index, cu_window_seqlens

    @paddle.jit.not_to_static
    def recompute_training_full(
        self,
        layer_module: nn.Layer,
        hidden_states: paddle.Tensor,
        cu_seqlens_now: paddle.Tensor,
        rotary_pos_emb: paddle.Tensor,
    ):
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        hidden_states = recompute(
            create_custom_forward(layer_module),
            hidden_states,
            cu_seqlens_now,
            rotary_pos_emb,
            # use_reentrant=self.config.recompute_use_reentrant,
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

        cu_seqlens = paddle.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            axis=0, dtype="int32"
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens
            if self.enable_recompute and self.training:
                hidden_states = self.recompute_training_full(blk, hidden_states, cu_seqlens_now, rotary_pos_emb)
            else:
                hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens_now, rotary_pos_emb=rotary_pos_emb)

        hidden_states = self.merger(hidden_states)
        reverse_indices = paddle.argsort(x=window_index)
        hidden_states = hidden_states[reverse_indices, :]

        return hidden_states


class Qwen2_5_VLModel(Qwen2_5_VLPreTrainedModel):
    def __init__(self, config: Qwen2_5_VLConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.config = config
        # Recompute defaults to False and is controlled by Trainer

        if config.tensor_parallel_degree > 1 and config.vocab_size % config.tensor_parallel_degree == 0:
            self.embed_tokens = mpu.VocabParallelEmbedding(
                self.vocab_size,
                self.hidden_size,
                weight_attr=paddle.ParamAttr(initializer=nn.initializer.XavierNormal()),
            )
        else:
            self.embed_tokens = nn.Embedding(
                self.vocab_size,
                self.hidden_size,
            )

        self.layers = nn.LayerList(
            [Qwen2_5_VLDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2_5_VLRotaryEmbedding(
            self.hidden_size // config.num_attention_heads,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

        self.enable_recompute = False

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @staticmethod
    def _prepare_decoder_attention_mask(attention_mask, input_shape, past_key_values_length, dtype):
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            if len(attention_mask.shape) == 2:
                expanded_attn_mask = _expand_2d_mask(attention_mask, dtype, tgt_length=input_shape[-1])
                # For decoding phase in generation, seq_length = 1, we don't need to add causal mask
                if input_shape[-1] > 1:
                    combined_attention_mask = _make_causal_mask(
                        input_shape,
                        past_key_values_length=past_key_values_length,
                    )
                    expanded_attn_mask = expanded_attn_mask & combined_attention_mask
            # [bsz, seq_len, seq_len] -> [bsz, 1, seq_len, seq_len]
            elif len(attention_mask.shape) == 3:
                expanded_attn_mask = attention_mask.unsqueeze(1).astype("bool")
            # if attention_mask is already 4-D, do nothing
            else:
                expanded_attn_mask = attention_mask
        else:
            expanded_attn_mask = _make_causal_mask(
                input_shape,
                past_key_values_length=past_key_values_length,
            )
        # Convert bool attention_mask to float attention mask, which will be added to attention_scores later
        expanded_attn_mask = paddle.where(expanded_attn_mask.cast("bool"), 0.0, paddle.finfo(dtype).min).astype(dtype)
        return expanded_attn_mask

    @paddle.jit.not_to_static
    def recompute_training_full(
        self,
        layer_module: nn.Layer,
        hidden_states: paddle.Tensor,
        position_ids: Optional[paddle.Tensor],
        attention_mask: paddle.Tensor,
        output_attentions: bool,
        past_key_value: paddle.Tensor,
        use_cache: bool,
        cache_position: Optional[paddle.Tensor] = None,
        position_embeddings=None,
    ):
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        hidden_states = recompute(
            create_custom_forward(layer_module),
            hidden_states,
            position_ids,
            attention_mask,
            output_attentions,
            past_key_value,
            use_cache,
            cache_position,
            position_embeddings,
            use_reentrant=self.config.recompute_use_reentrant,
        )

        return hidden_states

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[paddle.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.layers))
        # NOTE: to make cache can be clear in-time
        past_key_values = list(past_key_values)

        seq_length_with_past = seq_length
        cache_length = 0
        if past_key_values[0] is not None:
            cache_length = past_key_values[0][0].shape[2]  # shape[1] in qwen2
            seq_length_with_past += cache_length

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # embed positions
        if attention_mask is None:
            # [bs, seq_len]
            attention_mask = paddle.ones((batch_size, seq_length_with_past), dtype=paddle.bool)

        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and len(attention_mask.shape) <= 1:
                causal_mask = attention_mask
            else:
                causal_mask = None
        else:
            causal_mask = self._prepare_decoder_attention_mask(
                attention_mask, (batch_size, seq_length), cache_length, inputs_embeds.dtype
            )  # [bs, 1, seq_len, seq_len]

        if cache_position is None:
            past_seen_tokens = past_key_values[0][0].shape[2] if past_key_values[0] is not None else 0
            cache_position = paddle.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1])

        if position_ids is None:
            # the hard coded `3` is for temporal, height and width.
            position_ids = cache_position.reshape([1, 1, -1]).expand([3, inputs_embeds.shape[0], -1])

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = ()

        for idx, (decoder_layer) in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.enable_recompute and self.training:
                layer_outputs = self.recompute_training_full(
                    decoder_layer,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,  # False
                    use_cache=use_cache,  # True
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            # NOTE: clear outdate cache after it has been used for memory saving
            past_key_value = past_key_values[idx] = None

            hidden_states = layer_outputs[0]

            next_decoder_cache = next_decoder_cache + (layer_outputs[-1],) if use_cache else None

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class Qwen2LMHead(nn.Layer):
    def __init__(self, config, embedding_weights=None, transpose_y=False):
        super(Qwen2LMHead, self).__init__()
        self.config = config
        if config.tensor_parallel_degree > 1 and config.vocab_size % config.tensor_parallel_degree == 0:
            vocab_size = config.vocab_size // config.tensor_parallel_degree
        else:
            vocab_size = config.vocab_size

        self.transpose_y = transpose_y
        if transpose_y:
            # only for weight from embedding_weights
            if embedding_weights is not None:
                self.weight = embedding_weights
            else:
                self.weight = self.create_parameter(
                    shape=[vocab_size, config.hidden_size],
                    dtype=paddle.get_default_dtype(),
                )
        else:

            if vocab_size != config.vocab_size:
                with get_rng_state_tracker().rng_state():
                    self.weight = self.create_parameter(
                        shape=[config.hidden_size, vocab_size],
                        dtype=paddle.get_default_dtype(),
                    )
            else:
                self.weight = self.create_parameter(
                    shape=[config.hidden_size, vocab_size],
                    dtype=paddle.get_default_dtype(),
                )

        # Must set distributed attr for Tensor Parallel !
        self.weight.is_distributed = True if (vocab_size != config.vocab_size) else False
        if self.weight.is_distributed:
            # for tie_word_embeddings
            self.weight.split_axis = 0 if self.transpose_y else 1

    def forward(self, hidden_states, tensor_parallel_output=None):
        if tensor_parallel_output is None:
            tensor_parallel_output = self.config.tensor_parallel_output

        # 确保数据类型一致
        if self.weight.dtype != hidden_states.dtype:
            hidden_states = paddle.cast(hidden_states, self.weight.dtype)

        logits = parallel_matmul(
            hidden_states, self.weight, transpose_y=self.transpose_y, tensor_parallel_output=tensor_parallel_output
        )
        return logits


class Qwen2_5_VLForConditionalGeneration(Qwen2_5_VLPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]
    config_class = Qwen2_5_VLConfig
    _no_split_modules = ["Qwen2VLDecoderLayer", "Qwen2_5_VLVisionBlock"]

    def __init__(self, config, attn_implementation="flash_attention_2"):
        super().__init__(config)
        config._attn_implementation = attn_implementation
        config.vision_config._attn_implementation = attn_implementation

        self.visual = Qwen2_5_VisionTransformerPretrainedModel._from_config(config.vision_config)
        self.model = Qwen2_5_VLModel(config)
        self.vocab_size = config.vocab_size
        if config.tie_word_embeddings:
            self.lm_head = Qwen2LMHead(config, embedding_weights=self.model.embed_tokens.weight, transpose_y=True)
            self.tie_weights()
        else:
            self.lm_head = Qwen2LMHead(config)
        self.padding_side = "left"  # set it to left by default, user can use setter to change padding_sides

        self.enable_recompute = False

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

    @classmethod
    def _get_tensor_parallel_mappings(cls, config: Qwen2_5_VLConfig, is_split=True):

        logger.info("Qwen2_5_VL inference model _get_tensor_parallel_mappings")

        from ..conversion_utils import split_or_merge_func

        fn = split_or_merge_func(
            is_split=is_split,
            tensor_parallel_degree=config.tensor_parallel_degree,
            tensor_parallel_rank=config.tensor_parallel_rank,
            num_attention_heads=config.num_attention_heads,
        )

        def get_tensor_parallel_split_mappings(num_layers):
            final_actions = {}

            base_actions = {
                "lm_head.weight": partial(fn, is_column=True),
                # Row Linear
                "embed_tokens.weight": partial(fn, is_column=False),
                "layers.0.self_attn.o_proj.weight": partial(fn, is_column=False),
                "layers.0.mlp.down_proj.weight": partial(fn, is_column=False),
            }

            # Column Linear
            # if config.fuse_attention_qkv:
            #     base_actions["layers.0.self_attn.qkv_proj.weight"] = partial(fn, is_column=True)
            # else:
            base_actions["layers.0.self_attn.q_proj.weight"] = partial(fn, is_column=True)
            base_actions["layers.0.self_attn.q_proj.bias"] = partial(fn, is_column=True)
            # if we have enough num_key_value_heads to split, then split it.
            if config.num_key_value_heads % config.tensor_parallel_degree == 0:
                base_actions["layers.0.self_attn.k_proj.weight"] = partial(fn, is_column=True)
                base_actions["layers.0.self_attn.v_proj.weight"] = partial(fn, is_column=True)
                base_actions["layers.0.self_attn.k_proj.bias"] = partial(fn, is_column=True)
                base_actions["layers.0.self_attn.v_proj.bias"] = partial(fn, is_column=True)

            if config.fuse_attention_ffn:
                base_actions["layers.0.mlp.gate_up_fused_proj.weight"] = partial(
                    fn, is_column=True, is_naive_2fuse=True
                )
            else:
                base_actions["layers.0.mlp.gate_proj.weight"] = partial(fn, is_column=True)
                base_actions["layers.0.mlp.up_proj.weight"] = partial(fn, is_column=True)

            for key, action in base_actions.items():
                if "layers.0." in key:
                    for i in range(num_layers):
                        final_actions[key.replace("layers.0.", f"layers.{i}.")] = action
                final_actions[key] = action

            return final_actions

        mappings = get_tensor_parallel_split_mappings(config.num_hidden_layers)

        return mappings

    @staticmethod
    def get_rope_index(
        spatial_merge_size,
        image_token_id,
        video_token_id,
        vision_start_token_id,
        tokens_per_second,
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
            input_ids (`paddle.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`paddle.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`paddle.LongTensor` of shape `(num_videos, 3)`, *optional*):
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
        # spatial_merge_size = self.config.vision_config.spatial_merge_size
        # image_token_id = self.config.image_token_id
        # video_token_id = self.config.video_token_id
        # vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if image_grid_thw is not None or video_grid_thw is not None:
            total_input_ids = input_ids
            position_ids = paddle.ones([3, input_ids.shape[0], input_ids.shape[1]], dtype=input_ids.dtype)
            image_index, video_index = 0, 0
            for i, input_ids in enumerate(total_input_ids):
                # TODO: CUDA error in some paddle version
                if attention_mask is not None:
                    input_ids = paddle.to_tensor(input_ids.cpu()[attention_mask[i].cpu() == 1])
                image_nums, video_nums = 0, 0
                vision_start_indices = paddle.nonzero(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum() if vision_tokens.numel() > 0 else 0
                video_nums = (vision_tokens == video_token_id).sum() if vision_tokens.numel() > 0 else 0
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
                    llm_pos_ids_list.append(paddle.arange(text_len).reshape([1, -1]).expand([3, -1]) + st_idx)
                    range_tensor = paddle.arange(end=llm_grid_t).reshape([-1, 1])
                    expanded_range = range_tensor.expand(shape=[-1, llm_grid_h * llm_grid_w])
                    time_tensor = expanded_range * second_per_grid_t * tokens_per_second
                    time_tensor_long = time_tensor.astype(dtype="int64")
                    t_index = time_tensor_long.flatten()
                    h_index = (
                        paddle.arange(end=llm_grid_h)
                        .reshape([1, -1, 1])
                        .expand(shape=[llm_grid_t, -1, llm_grid_w])
                        .flatten()
                    )
                    w_index = (
                        paddle.arange(end=llm_grid_w)
                        .reshape([1, 1, -1])
                        .expand(shape=[llm_grid_t, llm_grid_h, -1])
                        .flatten()
                    )
                    llm_pos_ids_list.append(paddle.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(paddle.arange(text_len).reshape([1, -1]).expand([3, -1]) + st_idx)
                llm_positions = paddle.concat(llm_pos_ids_list, axis=1).reshape([3, -1])
                position_ids[..., i, attention_mask[i] == 1] = llm_positions

                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = paddle.to_tensor(mrope_position_deltas).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = paddle.cast(attention_mask, dtype="int64").cumsum(-1) - 1
                position_ids.masked_fill_(mask=attention_mask == 0, value=1)
                position_ids = position_ids.unsqueeze(0).expand([3, -1, -1])
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    paddle.arange(input_ids.shape[1]).reshape([1, 1, -1]).expand(shape=[3, input_ids.shape[0], -1])
                )
                mrope_position_deltas = paddle.zeros([input_ids.shape[0], 1], dtype=input_ids.dtype)
            return position_ids, mrope_position_deltas

    def update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        # num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        model_kwargs = super().update_model_kwargs_for_generation(
            outputs=outputs,
            model_kwargs=model_kwargs,
            is_encoder_decoder=is_encoder_decoder,
            # num_new_tokens=num_new_tokens,
        )

        # return logits + 28 layers k and v, TODO:
        if getattr(outputs, "rope_deltas", None) is not None:
            model_kwargs["rope_deltas"] = outputs.rope_deltas

        return model_kwargs

    def forward(
        self,
        input_ids: paddle.Tensor = None,  # [1, 400] sum 49356255
        attention_mask: Optional[paddle.Tensor] = None,  # [1, 400] sum 396
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,  # [1, 400] sum 354841
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[paddle.Tensor] = None,  # [1, 1224, 1176] sum 2658700.50000000
        pixel_values_videos: Optional[paddle.Tensor] = None,
        image_grid_thw: Optional[paddle.Tensor] = None,  # [[1 , 36, 34]]
        video_grid_thw: Optional[paddle.Tensor] = None,
        rope_deltas: Optional[paddle.Tensor] = None,
        second_per_grid_ts: Optional[paddle.Tensor] = None,
    ):
        """
        Args:
            labels (`paddle.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        >>> model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states  # fmt:skip
        # Note：始终为True
        return_dict = True  # return_dict if return_dict is not None else self.config.use_return_dict
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                # 确保 pixel_values 和 inputs_embeds 使用相同的数据类型
                pixel_values = paddle.cast(pixel_values, inputs_embeds.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                # 确保 image_embeds 和 inputs_embeds 使用相同的数据类型
                image_embeds = paddle.cast(image_embeds, inputs_embeds.dtype)
                image_mask = input_ids == self.config.image_token_id
                if self.training:
                    inputs_embeds = inputs_embeds.clone()
                inputs_embeds[image_mask] = image_embeds
            if pixel_values_videos is not None:
                # 确保 pixel_values_videos 和 inputs_embeds 使用相同的数据类型
                pixel_values_videos = paddle.cast(pixel_values_videos, inputs_embeds.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                # 确保 video_embeds 和 inputs_embeds 使用相同的数据类型
                video_embeds = paddle.cast(video_embeds, inputs_embeds.dtype)
                video_mask = input_ids == self.config.video_token_id
                inputs_embeds[video_mask] = video_embeds
            if attention_mask is not None:
                attention_mask = attention_mask

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        tensor_parallel_output = self.config.tensor_parallel_output and self.config.tensor_parallel_degree > 1

        logits = self.lm_head(hidden_states, tensor_parallel_output=tensor_parallel_output)
        # logits = paddle.cast(logits, "float32")

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :]  # [1, 395, 151936]
            shift_labels = labels[..., 1:]  # [1, 395]
            # Flatten the tokens
            shift_logits = shift_logits.reshape([-1, self.config.vocab_size])
            shift_labels = shift_labels.reshape([-1])
            loss_fct = nn.CrossEntropyLoss(reduction="sum")
            loss = loss_fct(shift_logits, shift_labels)
            label_sum = paddle.sum(shift_labels != -100).cast("float32")
            loss = loss / label_sum

        if not return_dict:
            # output = (logits,) + outputs[1:]
            # Note: (changwenbin) fix "can only concatenate tuple (not "list") to tuple".
            output = (logits,) + tuple(outputs[1:])
            return (loss,) + output if loss is not None else output
            # return logits + 28 layers k and v

        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=rope_deltas,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,  # [1, 3602] # [[151644,   8948,    198,  ..., 151644,  77091,    198]]
        past_key_values=None,  # DynamicCache
        attention_mask=None,  # [1, 3602] 1
        inputs_embeds=None,  # None
        cache_position=None,  # [   0,    1,    2,  ..., 3599, 3600, 3601]
        position_ids=None,  # None
        use_cache=True,
        pixel_values=None,  # [14308, 1176]
        pixel_values_videos=None,
        image_grid_thw=None,  # [1, 3]  # [[  1,  98, 146]]
        video_grid_thw=None,
        second_per_grid_ts=None,
        **kwargs,
    ):
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        batch_size, seq_length = input_ids.shape
        if past_key_values is None:
            cache_position = paddle.arange(input_ids.shape[1])
        else:
            cache_position = paddle.to_tensor([seq_length - 1])

        if past_key_values is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        rope_deltas = kwargs.get("rope_deltas", None)

        if attention_mask is not None and position_ids is None:
            if cache_position is None or (cache_position is not None and cache_position[0] == 0):
                position_ids, rope_deltas = self.get_rope_index(
                    self.config.vision_config.spatial_merge_size,
                    self.config.image_token_id,
                    self.config.video_token_id,
                    self.config.vision_start_token_id,
                    self.config.vision_config.tokens_per_second,
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
            else:
                batch_size, seq_length = input_ids.shape
                delta = (
                    cache_position[0] + rope_deltas if cache_position is not None and rope_deltas is not None else 0
                )
                position_ids = paddle.arange(seq_length)
                position_ids = position_ids.reshape([1, -1]).expand([batch_size, -1])
                position_ids = position_ids + delta
                position_ids = position_ids.unsqueeze(axis=0).expand([3, -1, -1])

        if cache_position[0] != 0:
            pixel_values = None
            pixel_values_videos = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,  # [3, 1, 3602]
                "past_key_values": past_key_values,  # DynamicCache()
                "use_cache": use_cache,  # 1
                "attention_mask": attention_mask,  # [1, 3602]
                "pixel_values": pixel_values,  # [14308, 1176]
                "pixel_values_videos": pixel_values_videos,
                "image_grid_thw": image_grid_thw,  # [[  1,  98, 146]]
                "video_grid_thw": video_grid_thw,
                "rope_deltas": rope_deltas,  # [[-3504]]
                "second_per_grid_ts": second_per_grid_ts,
            }
        )
        return model_inputs


__all__ = ["Qwen2_5_VLForConditionalGeneration", "Qwen2_5_VLModel", "Qwen2_5_VLPreTrainedModel"]
