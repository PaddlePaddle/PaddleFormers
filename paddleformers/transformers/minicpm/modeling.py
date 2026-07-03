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

import logging

import paddle
from paddle import nn
from paddle.distributed.fleet.utils import recompute
from paddle.distributed.fleet.utils.sequence_parallel_utils import (
    mark_as_sequence_parallel_parameter,
)

import paddleformers

from ...nn.attention.interface import ALL_ATTENTION_FUNCTIONS
from ...nn.criterion.interface import CriterionLayer
from ...nn.linear import Linear as GeneralLinear
from ...nn.lm_head import LMHead as GeneralLMHead
from ...nn.mlp import MLP as MiniCPMMLP
from ...nn.norm import Norm as GeneralNorm
from ...nn.pp_model import GeneralModelForCausalLMPipe
from ...trainer.utils.doc import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)
from ..cache_utils import Cache, DynamicCache, DynamicLayer
from ..masking_utils import create_causal_mask_and_row_indices
from ..model_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from ..model_utils import PretrainedModel, register_base_model

"""Paddle MiniCPM model."""
import math
import re
import warnings
from typing import Dict, List, Optional, Tuple, Union

from .configuration import MiniCPMConfig

try:
    pass
    from infllm_v2 import infllmv2_attn_stage1, max_pooling_1d_varlen
except:
    pass
from functools import lru_cache, partial

PD_RETURN_INTRODUCTION = r"""
    Returns:
        [`{full_output_type}`] or `tuple(paddle.Tensor)`: A [`{full_output_type}`] or a tuple of
        `paddle.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
        elements depending on the configuration ([`{config_class}`]) and inputs.

"""


def _get_indent(t):
    search = re.search(r"^(\s*)\S", t)
    return "" if search is None else search.groups()[0]


def _convert_output_args_doc(output_args_doc):
    indent = _get_indent(output_args_doc)
    blocks = []
    current_block = ""
    for line in output_args_doc.split("\n"):
        if _get_indent(line) == indent:
            if len(current_block) > 0:
                blocks.append(current_block[:-1])
            current_block = f"{line}\n"
        else:
            current_block += f"{line[2:]}\n"
    blocks.append(current_block[:-1])

    for i in range(len(blocks)):
        blocks[i] = re.sub(r"^(\s+)(\S+)(\s+)", r"\1- **\2**\3", blocks[i])
        blocks[i] = re.sub(r":\s*\n\s*(\S)", r" -- \1", blocks[i])

    return "\n".join(blocks)


def _prepare_output_docstrings(output_type, config_class, min_indent=None):
    output_docstring = output_type.__doc__
    params_docstring = None
    if output_docstring is not None:
        lines = output_docstring.split("\n")
        i = 0
        while i < len(lines) and re.search(r"^\s*(Args|Parameters):\s*$", lines[i]) is None:
            i += 1
        if i < len(lines):
            params_docstring = "\n".join(lines[(i + 1) :])
            params_docstring = _convert_output_args_doc(params_docstring)
        else:
            raise ValueError(
                f"No `Args` or `Parameters` section is found in the docstring of `{output_type.__name__}`."
            )

    full_output_type = f"{output_type.__module__}.{output_type.__name__}"
    result = PD_RETURN_INTRODUCTION.format(full_output_type=full_output_type, config_class=config_class)
    if params_docstring is not None:
        result += params_docstring

    if min_indent is not None:
        lines = result.split("\n")
        i = 0
        while i < len(lines) and len(lines[i]) == 0:
            i += 1
        if i < len(lines):
            indent = len(_get_indent(lines[i]))
            if indent < min_indent:
                to_add = " " * (min_indent - indent)
                result = "\n".join(f"{to_add}{line}" if len(line) > 0 else line for line in lines)

    return result


def replace_return_docstrings(output_type=None, config_class=None):
    def docstring_decorator(fn):
        func_doc = fn.__doc__
        lines = func_doc.split("\n")
        i = 0
        while i < len(lines) and re.search(r"^\s*Returns?:\s*$", lines[i]) is None:
            i += 1
        if i < len(lines):
            indent = len(_get_indent(lines[i]))
            lines[i] = _prepare_output_docstrings(output_type, config_class, min_indent=indent)
            func_doc = "\n".join(lines)
        else:
            raise ValueError(
                f"The function {fn} should have an empty 'Return:' or 'Returns:' in its docstring as placeholder, "
                f"current docstring is:\n{func_doc}"
            )
        fn.__doc__ = func_doc
        return fn

    return docstring_decorator


_MINICPM_CONFIG_DEFAULTS = {
    "fuse_attention_qkv": False,
    "fuse_attention_ffn": False,
}


def _ensure_minicpm_config_defaults(config):
    for key, value in _MINICPM_CONFIG_DEFAULTS.items():
        if not hasattr(config, key):
            setattr(config, key, value)
    return config


def _tensor_max(tensor, *args, **kwargs):
    if "other" in kwargs:
        kwargs["y"] = kwargs.pop("other")
        ret = paddle.maximum(tensor, *args, **kwargs)
    elif len(args) == 1 and isinstance(args[0], paddle.Tensor):
        ret = paddle.maximum(tensor, *args, **kwargs)
    else:
        if "dim" in kwargs:
            kwargs["axis"] = kwargs.pop("dim")

        if "axis" in kwargs or len(args) >= 1:
            ret = paddle.max(tensor, *args, **kwargs), paddle.argmax(tensor, *args, **kwargs)
        else:
            ret = paddle.max(tensor, *args, **kwargs)

    return ret


def _split_tensor(tensor, split_size_or_sections, dim=0):
    if isinstance(split_size_or_sections, int):
        dim_size = tensor.shape[dim]
        sections = [split_size_or_sections] * (dim_size // split_size_or_sections)
        if dim_size % split_size_or_sections:
            sections.append(dim_size % split_size_or_sections)
        split_size_or_sections = sections
    return paddle.split(tensor, split_size_or_sections, axis=dim)


def _convert_head_mask_to_5d(head_mask, num_hidden_layers):
    if head_mask.dim() == 1:
        head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
    elif head_mask.dim() == 2:
        head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
    assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
    head_mask = head_mask.to(dtype=paddle.get_default_dtype())  # switch to float if need + fp16 compatibility
    return head_mask


def _get_head_mask(
    self,
    head_mask: Optional[paddle.Tensor],
    num_hidden_layers: int,
    is_attention_chunked: bool = False,
):
    if head_mask is not None:
        head_mask = _convert_head_mask_to_5d(head_mask, num_hidden_layers)
        if is_attention_chunked is True:
            head_mask = head_mask.unsqueeze(-1)
    else:
        head_mask = [None] * num_hidden_layers
    return head_mask


setattr(paddleformers.transformers.model_utils.PretrainedModel, "get_head_mask", _get_head_mask)

setattr(paddleformers.transformers.model_utils.PretrainedModel, "device", None)


def _post_init(self):
    if hasattr(self, "init_weights"):
        self.init_weights()
    elif hasattr(self, "_init_weights"):
        self._init_weights()


setattr(paddleformers.transformers.model_utils.PretrainedModel, "post_init", _post_init)


def compressed_attention(
    q: paddle.Tensor,
    k: paddle.Tensor,
    k2: paddle.Tensor,
    kernel_size: int,
    kernel_stride: int,
    block_size: int,
    topk: int,
    cu_seqlens_q: paddle.Tensor,
    cu_seqlens_k: paddle.Tensor,
    cu_seqlens_k2: paddle.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sm_scale: float = None,
    init_blocks: int = 1,
    local_blocks: int = 2,
    cache_lens=None,
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    with paddle.no_grad():
        batch_size = cu_seqlens_q.shape[0] - 1
        is_prefilling = cache_lens is None or (cache_lens == 0).all().item()
        if is_prefilling:
            cache_lens = paddle.zeros(batch_size, dtype=paddle.int32, device=q.device)
            q_idx = paddle.cat(
                [
                    (
                        (
                            paddle.arange(cu_seqlens_q[i + 1] - cu_seqlens_q[i], device=q.device)
                            + max_seqlen_q
                            - (cu_seqlens_q[i + 1] - cu_seqlens_q[i])
                        )
                        // block_size
                    )
                    for i in range(batch_size)
                ],
                dim=0,
            )
        else:
            q_idx = cache_lens // block_size
        score = infllmv2_attn_stage1(
            q.contiguous(),
            k.contiguous(),
            k2.contiguous(),
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            cu_seqlens_v=cu_seqlens_k2,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal=is_prefilling,
        )
        score = score[:, : q_idx.shape[0], :]
        block_score = max_pooling_1d_varlen(
            score.contiguous(),
            cu_seqlens_q,
            cu_seqlens_k,
            cache_lens,
            max_seqlen_q,
            max_seqlen_k,
            local_blocks=local_blocks,
            init_blocks=init_blocks,
            block_size=block_size,
            stride=kernel_stride,
        )
        topk = min(topk, block_score.shape[-1])
        topk_idx = block_score.topk(topk, dim=-1).indices.sort(-1).values
        topk_idx[topk_idx > q_idx[None, :, None]] = -1
        topk_idx = topk_idx.to(paddle.int32)
    return topk_idx


@lru_cache(maxsize=16)
def calc_chunks_with_stride(cu_seqlen, chunk_size, kernel_stride):
    """
    Compute the chunks that require Sparse attention, with stride support.

    Args:
        cu_seqlen (paddle.Tensor): Cumulative sequence lengths for each sample.
        chunk_size (int): Chunk size used for Sparse attention.
        kernel_stride (int): Stride size when sliding over the sequence.

    Returns:
        filtered_indices (paddle.Tensor): Indices used to directly index into the key/value tensors.
        cu_seqlens_compressed (paddle.Tensor): Cumulative sequence lengths after compression.
    """
    batch_sizes = cu_seqlen[1:] - cu_seqlen[:-1]
    max_seq_len = paddle.compat.max(batch_sizes)
    max_num_chunks_per_seq = (max_seq_len - chunk_size) // kernel_stride + 1
    chunk_start_offsets = paddle.arange(
        0,
        max_num_chunks_per_seq * kernel_stride,
        kernel_stride,
        device=cu_seqlen.device,
    )
    seq_starts = cu_seqlen[:-1]
    chunk_start_in_seq = seq_starts[:, None] + chunk_start_offsets[None, :]
    chunk_end_in_seq = chunk_start_in_seq + chunk_size
    valid_chunk_mask = chunk_end_in_seq <= seq_starts[:, None] + batch_sizes[:, None]
    valid_chunk_starts = chunk_start_in_seq[valid_chunk_mask]
    del chunk_start_in_seq
    chunk_indices = paddle.arange(0, chunk_size, device=cu_seqlen.device)[None, :]
    filtered_indices = valid_chunk_starts[:, None] + chunk_indices
    filtered_indices = filtered_indices.view(-1)
    num_filtered_chunks_per_batch = valid_chunk_mask.sum(dim=1)
    cu_seqlens_compressed = paddle.zeros(len(cu_seqlen), dtype=paddle.int32, device=cu_seqlen.device)
    cu_seqlens_compressed[1:] = num_filtered_chunks_per_batch.cumsum(dim=0)
    del (
        num_filtered_chunks_per_batch,
        chunk_start_offsets,
        seq_starts,
        chunk_end_in_seq,
        valid_chunk_mask,
        chunk_indices,
    )
    return filtered_indices, cu_seqlens_compressed


class CompressK(nn.Layer):
    def __init__(self, head_num_k, head_dim, kernel_size, kernel_stride=16):
        """
        Module for compressing key (K) representations.

        Args:
            head_num_k (int): Number of key attention heads.
            head_dim (int): Dimension of each attention head.
            kernel_size (int): Size of each chunk used for compression.
            kernel_stride (int, optional): Stride used when dividing input into chunks. Default is 16.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.head_num_k = head_num_k
        self.head_dim = head_dim
        self.kernel_stride = kernel_stride

    def forward(self, k: paddle.Tensor, cu_seqlens):
        """
        Forward pass for compressing the key (K) tensor.

        Args:
            k (paddle.Tensor): Input key tensor of shape (total_seq_len, num_heads, head_dim).
            cu_seqlens (paddle.Tensor): Cumulative sequence lengths for each sample in the batch.

        Returns:
            compressed_k (paddle.Tensor): Compressed key tensor.
            cu_seqlens_compressed (paddle.Tensor): Updated cumulative sequence lengths after compression.
        """
        filtered_k_indices, cu_seqlens_compressed = calc_chunks_with_stride(
            cu_seqlens, self.kernel_size, self.kernel_stride
        )
        filtered_k = k.index_select(0, filtered_k_indices.view(-1))
        filtered_k = filtered_k.view(
            filtered_k.shape[0] // self.kernel_size,
            self.kernel_size,
            self.head_num_k,
            self.head_dim,
        )
        compressed_k = filtered_k.mean(dim=1)
        return compressed_k, cu_seqlens_compressed


class InfLLMv2CacheLayer(DynamicLayer):
    def __init__(self):
        super().__init__()
        self.no_rope_keys = paddle.tensor([], dtype=paddle.float32)
        self.compress_k_cache = []
        self.no_compress_k_cache = []
        self.cached_compressed_cu_seqlens = paddle.tensor([], dtype=paddle.int32)
        self.compress_k_cache_varlen = paddle.tensor([], dtype=paddle.float32)
        self.compress_k2_cache = []
        self.cached_compressed_cu_seqlens2 = paddle.tensor([], dtype=paddle.int32)
        self.compress_k2_cache_varlen = paddle.tensor([], dtype=paddle.float32)
        self.no_compress_k2_cache = []

    def update_no_rope_key(self, key_states):
        if self.no_rope_keys.size == 0:
            self.no_rope_keys = key_states
        else:
            self.no_rope_keys = paddle.cat([self.no_rope_keys, key_states], dim=1)
        return self.no_rope_keys

    def update_compress_k(self, key_states, cu_seqlens=None):
        if len(self.compress_k_cache) == 0:
            if cu_seqlens is not None:
                self.cached_compressed_cu_seqlens = cu_seqlens.clone()
            self.compress_k_cache_varlen = key_states
            split_sizes = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
            self.compress_k_cache = list(paddle.compat.split(key_states, split_sizes))
        else:
            for index, k in enumerate(key_states):
                if k is not None:
                    self.compress_k_cache[index] = paddle.cat([self.compress_k_cache[index], k], dim=0)
            new_seq_lens = paddle.tensor(
                [tensor.shape[0] for tensor in self.compress_k_cache],
                dtype=paddle.int32,
            )
            new_cumsum = paddle.cumsum(new_seq_lens, dim=0, dtype=paddle.int32)
            self.compress_k_cache_varlen = paddle.cat(self.compress_k_cache, dim=0)
            self.cached_compressed_cu_seqlens = paddle.cat([paddle.tensor([0], dtype=paddle.int32), new_cumsum]).to(
                self.compress_k_cache_varlen.device
            )
        return self.compress_k_cache_varlen, self.cached_compressed_cu_seqlens

    def update_no_compress_k(self, key_states, kernel_size=32, kernel_stride=16):
        k_chunk_list = []
        for index, k in enumerate(key_states):
            if len(self.no_compress_k_cache) <= index:
                self.no_compress_k_cache.append(k)
            else:
                self.no_compress_k_cache[index] = paddle.cat([self.no_compress_k_cache[index], k], dim=0)
                current_len = self.no_compress_k_cache[index].shape[0]
                if current_len >= kernel_size:
                    k_chunk_list.append(self.no_compress_k_cache[index][:kernel_size])
                    self.no_compress_k_cache[index] = self.no_compress_k_cache[index][kernel_stride:]
                else:
                    k_chunk_list.append(None)
        return k_chunk_list

    def update_compress_k2(self, key_states, cu_seqlens=None):
        if len(self.compress_k2_cache) == 0:
            if cu_seqlens is not None:
                self.cached_compressed_cu_seqlens2 = cu_seqlens.clone()
            self.compress_k2_cache_varlen = key_states
            split_sizes = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
            self.compress_k2_cache = list(paddle.compat.split(key_states, split_sizes))
        else:
            for index, k in enumerate(key_states):
                if k is not None:
                    self.compress_k2_cache[index] = paddle.cat([self.compress_k2_cache[index], k], dim=0)
            new_seq_lens = paddle.tensor(
                [tensor.shape[0] for tensor in self.compress_k2_cache],
                dtype=paddle.int32,
            )
            new_cumsum = paddle.cumsum(new_seq_lens, dim=0, dtype=paddle.int32)
            self.compress_k2_cache_varlen = paddle.cat(self.compress_k2_cache, dim=0)
            self.cached_compressed_cu_seqlens2 = paddle.cat([paddle.tensor([0], dtype=paddle.int32), new_cumsum]).to(
                self.compress_k2_cache_varlen.device
            )
        return (self.compress_k2_cache_varlen, self.cached_compressed_cu_seqlens2)

    def update_no_compress_k2(self, key_states, kernel_size=128, kernel_stride=64):
        k_chunk_list = []
        for index, k in enumerate(key_states):
            if len(self.no_compress_k2_cache) <= index:
                self.no_compress_k2_cache.append(k)
            else:
                self.no_compress_k2_cache[index] = paddle.cat([self.no_compress_k2_cache[index], k], dim=0)
                current_len = self.no_compress_k2_cache[index].shape[0]
                if current_len >= kernel_size:
                    k_chunk_list.append(self.no_compress_k2_cache[index][:kernel_size])
                    self.no_compress_k2_cache[index] = self.no_compress_k2_cache[index][kernel_stride:]
                else:
                    k_chunk_list.append(None)
        return k_chunk_list


class InfLLMv2Cache(DynamicCache):
    def __init__(self, config, num_hidden_layers: Optional[int] = None) -> None:
        super().__init__(config=config)
        self.layers = [InfLLMv2CacheLayer() for _ in range(num_hidden_layers)] if num_hidden_layers else []
        self._seen_tokens = 0

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        return self.layers[layer_idx].update(key_states, value_states, cache_kwargs)

    def update_no_rope_key(self, key_states, layer_idx, cache_kwargs=None):
        return self.layers[layer_idx].update_no_rope_key(key_states)

    def update_compress_k(self, key_states, layer_idx, cu_seqlens=None, cache_kwargs=None):
        return self.layers[layer_idx].update_compress_k(key_states, cu_seqlens)

    def update_no_compress_k(self, key_states, layer_idx, kernel_size=32, kernel_stride=16, cache_kwargs=None):
        return self.layers[layer_idx].update_no_compress_k(key_states, kernel_size, kernel_stride)

    def update_compress_k2(self, key_states, layer_idx, cu_seqlens=None, cache_kwargs=None):
        return self.layers[layer_idx].update_compress_k2(key_states, cu_seqlens)

    def update_no_compress_k2(
        self,
        key_states,
        layer_idx,
        kernel_size=128,
        kernel_stride=64,
        cache_kwargs=None,
    ):
        return self.layers[layer_idx].update_no_compress_k2(key_states, kernel_size, kernel_stride)

    def crop(self, max_length):
        for layer in self.layers:
            layer.crop(max_length)

    def batch_repeat_interleave(self, repeats):
        for layer in self.layers:
            layer.batch_repeat_interleave(repeats)

    def batch_select_indices(self, indices):
        for layer in self.layers:
            layer.batch_select_indices(indices)


logger = logging.getLogger(name=__name__)
_CONFIG_FOR_DOC = "MiniCPMConfig"


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=paddle.int32)
    indices = paddle.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = _tensor_max(seqlens_in_batch).item()
    cu_seqlens = nn.functional.pad(paddle.cumsum(seqlens_in_batch, dim=0, dtype=paddle.paddle.int32), (1, 0))
    return indices, cu_seqlens, max_seqlen_in_batch


def rms_layernorm(hidden: paddle.Tensor, weight: paddle.Tensor, eps: float):
    old_dtype = hidden.dtype
    variance = hidden.to(paddle.float32).pow(2).mean(dim=-1, keepdim=True)
    hidden = (hidden * paddle.rsqrt(variance + eps)).to(old_dtype)
    return hidden * weight


class MiniCPMRMSNorm(nn.Layer):
    def __init__(self, hidden_size, eps=1e-06):
        """
        MiniCPMRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = paddle.nn.Parameter(paddle.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return rms_layernorm(hidden_states, self.weight, self.variance_epsilon)


class MiniCPMRotaryEmbedding(nn.Layer):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / self.base ** (
            paddle.arange(0, int(self.dim), 2, dtype=paddle.int64).astype(dtype=paddle.float32) / self.dim
        )
        self.register_buffer("inv_freq", inv_freq, persistable=False)
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=inv_freq.place,
            dtype=paddle.float32,
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = paddle.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = paddle.outer(t, self.inv_freq)
        emb = paddle.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistable=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistable=False)

    def forward(self, x, position_ids):
        seq_len = int(position_ids.max()) + 1
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class MiniCPMLongRoPE(MiniCPMRotaryEmbedding):
    """MiniCPMRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        short_factor=None,
        long_factor=None,
        original_max_position_embeddings=None,
    ):
        self.short_factor = short_factor
        self.long_factor = long_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        scale = max_position_embeddings / self.original_max_position_embeddings
        self.scaling_factor = math.sqrt(1 + math.log(scale) / math.log(self.original_max_position_embeddings))
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = paddle.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        if seq_len > self.original_max_position_embeddings:
            ext_factors = paddle.tensor(self.long_factor, dtype=paddle.float32, device=device)
        else:
            ext_factors = paddle.tensor(self.short_factor, dtype=paddle.float32, device=device)
        freqs = paddle.mul(
            paddle.outer(t, 1.0 / ext_factors).to(device=device),
            self.inv_freq.to(device=device).to(dtype),
        )
        emb = paddle.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype) * self.scaling_factor, persistable=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype) * self.scaling_factor, persistable=False)


class MiniCPMLinearScalingRotaryEmbedding(MiniCPMRotaryEmbedding):
    """MiniCPMRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = paddle.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor
        freqs = paddle.outer(t, self.inv_freq)
        emb = paddle.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


class MiniCPMDynamicNTKScalingRotaryEmbedding(MiniCPMRotaryEmbedding):
    """MiniCPMRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        if seq_len > self.max_position_embeddings:
            base = self.base * (
                self.scaling_factor * seq_len / self.max_position_embeddings - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / base ** (paddle.arange(0, self.dim, 2).float().to(device) / self.dim)
            self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = paddle.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = paddle.outer(t, self.inv_freq)
        emb = paddle.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`paddle.Tensor`): The query tensor.
        k (`paddle.Tensor`): The key tensor.
        cos (`paddle.Tensor`): The cosine part of the rotary embedding.
        sin (`paddle.Tensor`): The sine part of the rotary embedding.
        position_ids (`paddle.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The dimension along which to unsqueeze cos[position_ids] and sin[position_ids] so that they can be properly
            broadcasted to the dimensions of q and k. For example, if q and k have shape
            [batch_size, heads, seq_len, head_dim], setting unsqueeze_dim=1 makes cos[position_ids] and
            sin[position_ids] broadcastable to q and k. If q and k have shape
            [batch_size, seq_len, heads, head_dim], set unsqueeze_dim=2.

    Returns:
        `tuple(paddle.Tensor)` comprising the query and key tensors rotated using Rotary Position Embedding.
    """
    orig_dtype = k.dtype
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)

    q_fp32 = q.to(dtype=paddle.float32)
    k_fp32 = k.to(dtype=paddle.float32)
    q_embed = q_fp32 * cos + rotate_half(q_fp32) * sin
    k_embed = k_fp32 * cos + rotate_half(k_fp32) * sin
    return q_embed.to(dtype=orig_dtype), k_embed.to(dtype=orig_dtype)


def _unpad_one_tensor(hidden_states, attention_mask):
    indices, cu_seqlens, max_seqlen_in_batch = _get_unpad_data(attention_mask)
    batch_size, seq_len = hidden_states.shape[:2]
    remaining_dims = hidden_states.shape[2:]
    reshaped_states = hidden_states.reshape(batch_size * seq_len, *remaining_dims)
    unpadded_states = paddle.gather(reshaped_states, indices)
    return unpadded_states, indices, cu_seqlens, max_seqlen_in_batch


def repeat_kv(hidden_states: paddle.Tensor, n_rep: int) -> paddle.Tensor:
    """
    This is the equivalent of repeat_interleave on the head dimension. The hidden states go from
    (batch, num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim).
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class MiniCPMAttention(nn.Layer):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: MiniCPMConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` when creating this class."
            )
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.fuse_attention_qkv = getattr(config, "fuse_attention_qkv", False)
        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`: {self.num_heads})."
            )
        if self.fuse_attention_qkv:
            qkv_hidden_size = (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim
            self.qkv_proj = GeneralLinear.create(
                self.hidden_size,
                qkv_hidden_size,
                has_bias=config.use_bias,
                config=config,
                tp_plan="colwise",
            )
        else:
            self.q_proj = GeneralLinear.create(
                self.hidden_size,
                self.num_heads * self.head_dim,
                has_bias=config.use_bias,
                config=config,
                tp_plan="colwise",
            )
            self.k_proj = GeneralLinear.create(
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                has_bias=config.use_bias,
                config=config,
                tp_plan="colwise",
            )
            self.v_proj = GeneralLinear.create(
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                has_bias=config.use_bias,
                config=config,
                tp_plan="colwise",
            )
        self.o_proj = GeneralLinear.create(
            self.hidden_size,
            self.num_heads * self.head_dim,
            has_bias=config.use_bias,
            config=config,
            tp_plan="rowwise",
        )

        self._init_rope()
        self.scaling = self.head_dim**-0.5
        self.attn_implementation = config._attn_implementation

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = MiniCPMRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["rope_type"]
            scaling_factor = self.config.rope_scaling.get("factor", None)
            if scaling_type == "linear":
                self.rotary_emb = MiniCPMLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = MiniCPMDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "longrope":
                self.rotary_emb = MiniCPMLongRoPE(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    short_factor=self.config.rope_scaling["short_factor"],
                    long_factor=self.config.rope_scaling["long_factor"],
                    base=self.rope_theta,
                    original_max_position_embeddings=self.config.rope_scaling["original_max_position_embeddings"],
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: paddle.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        if self.config.sequence_parallel:
            max_sequence_length = self.config.max_sequence_length
            bsz = hidden_states.shape[0] * self.config.tensor_model_parallel_size // max_sequence_length
            q_len = max_sequence_length
        else:
            bsz, q_len, _ = hidden_states.shape

        if self.fuse_attention_qkv:
            mix_layer = self.qkv_proj(hidden_states)
            mix_layer = mix_layer.reshape(
                [
                    bsz,
                    q_len,
                    -1,
                    (self.num_key_value_groups + 2) * self.head_dim,
                ]
            )
            query_states, key_states, value_states = paddle.split(
                mix_layer,
                num_or_sections=[
                    self.num_key_value_groups * self.head_dim,
                    self.head_dim,
                    self.head_dim,
                ],
                axis=-1,
            )
            query_states = query_states.reshape([bsz, q_len, -1, self.head_dim])
        elif self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = _split_tensor(
                self.q_proj.weight, (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=1
            )
            key_slices = _split_tensor(self.k_proj.weight, key_value_slicing, dim=1)
            value_slices = _split_tensor(self.v_proj.weight, key_value_slicing, dim=1)
            query_states = [
                nn.functional.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            query_states = paddle.cat(query_states, dim=-1)
            key_states = [
                nn.functional.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            key_states = paddle.cat(key_states, dim=-1)
            value_states = [
                nn.functional.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            value_states = paddle.cat(value_states, dim=-1)
            query_states = query_states.reshape([bsz, q_len, -1, self.head_dim])
            key_states = key_states.reshape([bsz, q_len, -1, self.head_dim])
            value_states = value_states.reshape([bsz, q_len, -1, self.head_dim])
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
            query_states = query_states.reshape([bsz, q_len, -1, self.head_dim])
            key_states = key_states.reshape([bsz, q_len, -1, self.head_dim])
            value_states = value_states.reshape([bsz, q_len, -1, self.head_dim])

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = self.rotary_emb(value_states.to(paddle.float32), position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface = ALL_ATTENTION_FUNCTIONS[self.attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query=query_states,
            key=key_states,
            value=value_states,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            dropout=self.config.get("attention_dropout_prob", 0.0) if self.training else 0.0,
            scaling=self.scaling,
        )
        if self.config.sequence_parallel:
            attn_output = attn_output.reshape([-1, attn_output.shape[-1]])
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value


class MiniCPMSdpaAttention(MiniCPMAttention):
    """
    MiniCPM attention module using scaled dot product attention. This module inherits from `MiniCPMAttention` as the
    weights of the module stay untouched. The only changes are on the forward pass to adapt to the SDPA API.
    """

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        if output_attentions:
            logger.warning_once(
                'MiniCPMModel is using MiniCPMSdpaAttention. Falling back to the manual attention implementation, but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
        bsz, q_len, _ = hidden_states.shape
        if self.fuse_attention_qkv:
            mix_layer = self.qkv_proj(hidden_states)
            mix_layer = mix_layer.reshape(
                [
                    bsz,
                    q_len,
                    -1,
                    (self.num_key_value_groups + 2) * self.head_dim,
                ]
            )
            query_states, key_states, value_states = paddle.split(
                mix_layer,
                num_or_sections=[
                    self.num_key_value_groups * self.head_dim,
                    self.head_dim,
                    self.head_dim,
                ],
                axis=-1,
            )
            query_states = query_states.reshape([bsz, q_len, -1, self.head_dim])
        else:
            query_states = self.q_proj(hidden_states).reshape([bsz, q_len, -1, self.head_dim])
            key_states = self.k_proj(hidden_states).reshape([bsz, q_len, -1, self.head_dim])
            value_states = self.v_proj(hidden_states).reshape([bsz, q_len, -1, self.head_dim])
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        kv_seq_len = _tensor_max(position_ids).item() + 1
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {bsz, 1, q_len, kv_seq_len}, but is {attention_mask.size()}"
                )
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()
        attn_output = nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output, None, past_key_value


MINICPM_ATTENTION_CLASSES = {
    "eager": MiniCPMAttention,
    "sdpa": MiniCPMSdpaAttention,
}


class MiniCPMDecoderLayer(nn.Layer):
    def __init__(self, config: MiniCPMConfig, layer_idx: int):
        super().__init__()
        _ensure_minicpm_config_defaults(config)
        self.hidden_size = config.hidden_size
        self.self_attn = MiniCPMAttention(config, layer_idx)
        self.mlp = MiniCPMMLP(config, fuse_up_gate=getattr(config, "fuse_attention_ffn", False))
        self.config = config
        self.input_layernorm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=config.hidden_size,
            has_bias=config.use_bias,
            norm_eps=self.config.rms_norm_eps,
            input_is_parallel=config.sequence_parallel,
        )
        self.post_attention_layernorm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=config.hidden_size,
            has_bias=config.use_bias,
            norm_eps=self.config.rms_norm_eps,
            input_is_parallel=config.sequence_parallel,
        )
        self.scale_depth = config.scale_depth
        self.num_hidden_layers = config.num_hidden_layers
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
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[paddle.Tensor, Optional[Tuple[paddle.Tensor, paddle.Tensor]]]:
        """
        Args:
            hidden_states (`paddle.Tensor`): Input to the layer of shape `(batch, seq_len, embed_dim)`.
            attention_mask (`paddle.Tensor`, *optional*):
                Attention mask of size `(batch_size, sequence_length)` if flash attention is used or
                `(batch_size, 1, query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(paddle.Tensor)`, *optional*): Cached past key and value projection states.
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states * (self.scale_depth / math.sqrt(self.num_hidden_layers))
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states * (self.scale_depth / math.sqrt(self.num_hidden_layers))
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        if type(outputs) is tuple and len(outputs) == 1:
            outputs = outputs[0]
        return outputs


MINICPM_START_DOCSTRING = """
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a Paddle [`paddle.nn.Layer`] subclass. Use it as a regular Paddle Layer and refer to the Paddle
    documentation for all matters related to general usage and behavior.

    Parameters:
        config ([`MiniCPMConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare MiniCPM Model outputting raw hidden-states without any specific head on top.",
    MINICPM_START_DOCSTRING,
)
class MiniCPMPreTrainedModel(PretrainedModel):
    config_class = MiniCPMConfig
    base_model_prefix = "model"
    transpose_weight_keys = [
        "q_proj",
        "k_proj",
        "v_proj",
        "qkv_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "up_gate_proj",
        "down_proj",
    ]
    supports_gradient_checkpointing = True
    _no_split_modules = ["MiniCPMDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def __init__(self, config):
        _ensure_minicpm_config_defaults(config)
        super().__init__(config)

    def _init_weights(self, module):
        if isinstance(self.config, dict):
            std = self.config.get("initializer_range", 0.02)
        else:
            std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, paddle.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @classmethod
    def _get_tensor_parallel_mappings(cls, config, is_split=True):
        """Generate tensor parallel mappings for model conversion."""
        from ..conversion_utils import split_or_merge_func

        _ensure_minicpm_config_defaults(config)
        fn = split_or_merge_func(
            is_split=is_split,
            tensor_model_parallel_size=config.tensor_model_parallel_size,
            tensor_parallel_rank=config.tensor_parallel_rank,
            num_attention_heads=config.num_attention_heads,
        )

        if getattr(config, "fuse_attention_qkv", False):
            attention_colwise_keys = ["self_attn.qkv_proj.weight"]
        else:
            attention_colwise_keys = [
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
            ]

        if getattr(config, "fuse_attention_ffn", False):
            ffn_colwise_keys = ["mlp.up_gate_proj.weight"]
        else:
            ffn_colwise_keys = [
                "mlp.up_proj.weight",
                "mlp.gate_proj.weight",
            ]

        LAYER_COLWISE = attention_colwise_keys + ffn_colwise_keys

        LAYER_ROWWISE = ["self_attn.o_proj.weight", "mlp.down_proj.weight"]

        BIAS_KEYS = [key.replace(".weight", ".bias") for key in LAYER_COLWISE] + [
            "self_attn.o_proj.bias",
            "mlp.down_proj.bias",
            "lm_head.bias",
        ]

        def make_base_actions():
            actions = {
                "lm_head.weight": partial(fn, is_column=False),
                "embed_tokens.weight": partial(fn, is_column=False),
            }
            for layer_idx in range(config.num_hidden_layers):
                actions.update(
                    {
                        f"{cls.base_model_prefix}.layers.{layer_idx}.{k}": partial(fn, is_column=True)
                        for k in LAYER_COLWISE
                    }
                )
                actions.update(
                    {
                        f"{cls.base_model_prefix}.layers.{layer_idx}.{k}": partial(fn, is_column=False)
                        for k in LAYER_ROWWISE
                    }
                )
                if config.use_bias:
                    actions.update(
                        {f"{cls.base_model_prefix}.layers.0.{b}": partial(fn, is_column=True) for b in BIAS_KEYS}
                    )

            return actions

        mappings = make_base_actions()
        return mappings

    @classmethod
    def _gen_aoa_config(cls, config: MiniCPMConfig):
        _ensure_minicpm_config_defaults(config)
        model_prefix = "" if cls == cls.base_model_class else "model."
        fuse_attention_qkv = getattr(config, "fuse_attention_qkv", False)
        fuse_attention_ffn = getattr(config, "fuse_attention_ffn", False)
        aoa_config = {
            "aoa_statements": [
                f"model.layers.$LAYER_ID.self_attn.o_proj.weight^T -> {model_prefix}layers.$LAYER_ID.self_attn.o_proj.weight",
                f"model.layers.$LAYER_ID.mlp.down_proj.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.down_proj.weight",
                f"model.embed_tokens.weight -> {model_prefix}embed_tokens.weight",
                f"model.layers.$LAYER_ID.input_layernorm.weight -> {model_prefix}layers.$LAYER_ID.input_layernorm.weight",
                f"model.layers.$LAYER_ID.post_attention_layernorm.weight -> {model_prefix}layers.$LAYER_ID.post_attention_layernorm.weight",
                f"model.norm.weight -> {model_prefix}norm.weight",
            ]
        }

        # attention qkv
        if not fuse_attention_qkv:
            aoa_config["aoa_statements"] += [
                f"model.layers.$LAYER_ID.self_attn.{x}_proj.weight^T -> {model_prefix}layers.$LAYER_ID.self_attn.{x}_proj.weight"
                for x in ("q", "k", "v")
            ]
        else:
            aoa_config["aoa_statements"] += [
                f"model.layers.$LAYER_ID.self_attn.q_proj.weight^T, model.layers.$LAYER_ID.self_attn.k_proj.weight^T, model.layers.$LAYER_ID.self_attn.v_proj.weight^T -> {model_prefix}layers.$LAYER_ID.self_attn.qkv_proj.weight, fused_qkv, num_heads={config.num_attention_heads}, num_key_value_groups={config.num_key_value_heads}",
            ]
            if config.use_bias:
                aoa_config["aoa_statements"] += [
                    f"model.layers.$LAYER_ID.self_attn.q_proj.bias, model.layers.$LAYER_ID.self_attn.k_proj.bias, model.layers.$LAYER_ID.self_attn.v_proj.bias -> {model_prefix}layers.$LAYER_ID.self_attn.qkv_proj.bias, fused_qkv, num_heads={config.num_attention_heads}, num_key_value_groups={config.num_key_value_heads}, axis=0",
                ]

        # FFN
        if not fuse_attention_ffn:
            aoa_config["aoa_statements"] += [
                f"model.layers.$LAYER_ID.mlp.{p}_proj.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.{p}_proj.weight"
                for p in ("gate", "up")
            ]
        else:
            aoa_config["aoa_statements"] += [
                f"model.layers.$LAYER_ID.mlp.gate_proj.weight^T, model.layers.$LAYER_ID.mlp.up_proj.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.up_gate_proj.weight, fused_ffn",
            ]

        # lm_head
        if config.tie_word_embeddings:
            aoa_config["aoa_statements"] += ["model.embed_tokens.weight -> lm_head.weight"]
        elif cls != cls.base_model_class:
            aoa_config["aoa_statements"] += ["lm_head.weight -> lm_head.weight"]

        return aoa_config

    @classmethod
    def _gen_inv_aoa_config(cls, config: MiniCPMConfig):
        _ensure_minicpm_config_defaults(config)
        model_prefix = "" if cls == cls.base_model_class else "model."
        fuse_attention_qkv = getattr(config, "fuse_attention_qkv", False)
        fuse_attention_ffn = getattr(config, "fuse_attention_ffn", False)
        aoa_statements = [
            f"{model_prefix}layers.$LAYER_ID.self_attn.o_proj.weight^T -> model.layers.$LAYER_ID.self_attn.o_proj.weight",
            f"{model_prefix}layers.$LAYER_ID.mlp.down_proj.weight^T -> model.layers.$LAYER_ID.mlp.down_proj.weight",
            f"{model_prefix}embed_tokens.weight -> model.embed_tokens.weight",
            f"{model_prefix}layers.$LAYER_ID.input_layernorm.weight -> model.layers.$LAYER_ID.input_layernorm.weight",
            f"{model_prefix}layers.$LAYER_ID.post_attention_layernorm.weight -> model.layers.$LAYER_ID.post_attention_layernorm.weight",
            f"{model_prefix}norm.weight -> model.norm.weight",
        ]

        if not fuse_attention_qkv:
            aoa_statements += [
                f"{model_prefix}layers.$LAYER_ID.self_attn.{x}_proj.weight^T -> model.layers.$LAYER_ID.self_attn.{x}_proj.weight"
                for x in ("q", "k", "v")
            ]
        else:
            aoa_statements += [
                f"{model_prefix}layers.$LAYER_ID.self_attn.qkv_proj.weight -> model.layers.$LAYER_ID.self_attn.q_proj.weight, model.layers.$LAYER_ID.self_attn.k_proj.weight, model.layers.$LAYER_ID.self_attn.v_proj.weight , fused_qkv, num_heads={config.num_attention_heads}, num_key_value_groups = {config.num_key_value_heads}",
            ]
            for layer_id in range(config.num_hidden_layers):
                for x in ("q", "k", "v"):
                    aoa_statements += [
                        f"model.layers.{layer_id}.self_attn.{x}_proj.weight^T -> model.layers.{layer_id}.self_attn.{x}_proj.weight"
                    ]
            if config.use_bias:
                aoa_statements += [
                    f"{model_prefix}layers.$LAYER_ID.self_attn.qkv_proj.bias -> model.layers.$LAYER_ID.self_attn.q_proj.bias, model.layers.$LAYER_ID.self_attn.k_proj.bias, model.layers.$LAYER_ID.self_attn.v_proj.bias, fused_qkv, num_heads={config.num_attention_heads}, num_key_value_groups={config.num_key_value_heads}, axis=0",
                ]

        if not fuse_attention_ffn:
            aoa_statements += [
                f"{model_prefix}layers.$LAYER_ID.mlp.{y}_proj.weight^T -> model.layers.$LAYER_ID.mlp.{y}_proj.weight"
                for y in ("gate", "up")
            ]
        else:
            aoa_statements += [
                f"{model_prefix}layers.$LAYER_ID.mlp.up_gate_proj.weight -> model.layers.$LAYER_ID.mlp.gate_proj.weight, model.layers.$LAYER_ID.mlp.up_proj.weight, fused_ffn",
            ]
            for layer_id in range(config.num_hidden_layers):
                aoa_statements += [
                    f"model.layers.{layer_id}.mlp.gate_proj.weight^T -> model.layers.{layer_id}.mlp.gate_proj.weight",
                    f"model.layers.{layer_id}.mlp.up_proj.weight^T -> model.layers.{layer_id}.mlp.up_proj.weight",
                ]

        if config.tie_word_embeddings:
            aoa_statements += ["lm_head.weight -> _"]
        elif cls != cls.base_model_class:
            aoa_statements += ["lm_head.weight -> lm_head.weight"]

        aoa_config = {"aoa_statements": aoa_statements}
        return aoa_config


MINICPM_INPUTS_DOCSTRING = """
    Args:
        input_ids (`paddle.Tensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`paddle.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`paddle.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Only the [`~cache_utils.Cache`] format is supported. If `use_cache=True` and no `past_key_values` are
            passed, a new dynamic cache will be created.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`paddle.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare MiniCPM Model outputting raw hidden-states without any specific head on top.",
    MINICPM_START_DOCSTRING,
)
@register_base_model
class MiniCPMModel(MiniCPMPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MiniCPMDecoderLayer`]

    Args:
        config: MiniCPMConfig
    """

    def __init__(self, config: MiniCPMConfig):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.LayerList(
            [MiniCPMDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._use_sdpa = config._attn_implementation == "sdpa"
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.norm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=config.hidden_size,
            has_bias=config.use_bias,
            norm_eps=self.config.rms_norm_eps,
            input_is_parallel=config.sequence_parallel,
        )
        self.gradient_checkpointing = False
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.rotary_emb = MiniCPMRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    @paddle.jit.not_to_static
    def recompute_training(
        self,
        layer_module,
        hidden_states,
        attention_mask,
        attn_mask_startend_row_indices,
        position_ids,
        position_embeddings,
        past_key_values,
        output_attentions,
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
            past_key_values (Optional[Cache]): Cached key/value states
            output_attentions (bool): Whether to output attention weights
            use_cache (bool): Whether to cache key/value states

        Returns:
            paddle.Tensor: Output hidden states after recomputation
        """

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        hidden_states = recompute(
            create_custom_forward(layer_module),
            hidden_states,
            attention_mask,
            attn_mask_startend_row_indices,
            position_ids,
            position_embeddings,
            past_key_values,
            output_attentions,
            use_cache,
        )
        return hidden_states

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(MINICPM_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids,
        attention_mask=None,
        attn_mask_startend_row_indices=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=None,
        return_dict=False,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
        past_key_values_length = 0
        if use_cache:
            if past_key_values is None:
                past_key_values = DynamicCache(config=self.config)
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache()
                raise ValueError(
                    "You must use the new past_key_values format, such as the Cache class, instead of the old tuple format."
                )
            past_key_values_length = (
                past_key_values.get_seq_length() if isinstance(past_key_values, InfLLMv2Cache) else 0
            )
            if self.config.sparse_config is not None and paddle.cuda.is_available() and past_key_values_length == 0:
                past_key_values = InfLLMv2Cache(config=self.config, num_hidden_layers=self.config.num_hidden_layers)
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.config.scale_emb

        mask_kwargs = {
            "config": self.config,
            "inputs_embeds": inputs_embeds,
            "batch_size": batch_size,
            "seq_length": seq_length,
            "cache_length": past_key_values_length,
            "attention_mask": attention_mask,
            "attn_mask_startend_row_indices": attn_mask_startend_row_indices,
            "prepare_decoder_attention_mask": self._prepare_decoder_attention_mask,
        }
        causal_attention_mask, attn_mask_startend_row_indices = create_causal_mask_and_row_indices(**mask_kwargs)

        hidden_states = inputs_embeds

        if position_ids is None:
            position_ids = (
                paddle.arange(past_key_values_length, seq_length + past_key_values_length, dtype=paddle.int64)
                .unsqueeze(0)
                .tile((batch_size, 1))
            )

        position_embeddings = self.rotary_emb(hidden_states.to(paddle.float32), position_ids)  # cos and sin

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        for idx, decoder_layer in enumerate(self.layers):
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
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    causal_attention_mask,
                    attn_mask_startend_row_indices,
                    position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            if isinstance(layer_outputs, (tuple, list)):
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class MiniCPMForCausalLM(MiniCPMPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = MiniCPMModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = GeneralLMHead(config)
        self.criterion = CriterionLayer(config)
        self.tie_weights()

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

    @add_start_docstrings_to_model_forward(MINICPM_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        attn_mask_startend_row_indices=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        loss_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
        logits_to_keep: Union[int, paddle.Tensor] = 0,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Args:
            labels (`paddle.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in
                `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100`
                are ignored (masked), and the loss is only computed for the tokens with labels in
                `[0, ..., config.vocab_size]`.

        Returns:
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if kwargs.get("attn_mask_start_row_indices", None) is not None and attn_mask_startend_row_indices is None:
            attn_mask_startend_row_indices = kwargs.pop("attn_mask_start_row_indices")

        if attention_mask is not None and attention_mask.dtype != paddle.bool:
            attention_mask = paddle.cast(attention_mask, paddle.bool)

        if attn_mask_startend_row_indices is not None and attention_mask is not None:
            logger.warning(
                "You have provided both attn_mask_startend_row_indices and attention_mask. "
                "The attn_mask_startend_row_indices will be used."
            )
            attention_mask = None

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        if labels is None:
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            hidden_states = hidden_states[:, slice_indices, :].contiguous()
        if self.config.pretraining_tp > 1:
            lm_head_slices = _split_tensor(self.lm_head.weight, self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [
                nn.functional.linear(hidden_states, lm_head_slices[i].transpose([1, 0]))
                for i in range(self.config.pretraining_tp)
            ]
            logits = paddle.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states / (self.config.hidden_size / self.config.dim_model_base))
        if not isinstance(logits, (tuple, list)):
            logits = logits.float()
        loss = None
        if labels is not None:
            loss, _ = self.criterion(logits, labels, loss_mask)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return CausalLMOutputWithPast(
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
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                if self.config.sparse_config is not None and paddle.cuda.is_available() and cache_length == 0:
                    past_key_values = InfLLMv2Cache(
                        config=self.config,
                        num_hidden_layers=self.config.num_hidden_layers,
                    )
                past_length = cache_length
                max_cache_length = None
            else:
                raise ValueError(
                    "You must use the new past_key_values format, such as the Cache class, instead of the old tuple format."
                )
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]
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
            }
        )
        for key, value in kwargs.items():
            if key not in model_inputs:
                model_inputs[key] = value
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past

    @paddle.no_grad()
    def chat(
        self,
        tokenizer,
        query: str,
        history: List[Dict] = None,
        role: str = "user",
        max_length: int = 4096,
        num_beams=1,
        do_sample=True,
        top_p=0.8,
        temperature=0.3,
        logits_processor=None,
        **kwargs,
    ):
        if history is None:
            history = []
        gen_kwargs = {
            "max_length": max_length,
            "num_beams": num_beams,
            "do_sample": do_sample,
            "top_p": top_p,
            "temperature": temperature,
            "logits_processor": logits_processor,
            **kwargs,
        }
        history.append({"role": role, "content": query})
        history_str = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=False)
        inputs = tokenizer(history_str, return_tensors="pd")
        outputs = self.generate(**inputs, **gen_kwargs)
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]
        input_length = inputs["input_ids"].shape[-1]
        outputs = outputs.tolist()[0][input_length:-1]
        response = tokenizer.decode(outputs)
        pattern = re.compile(r".*?(?=<\|im_end\|>|<\|im_start\|>(?:user|assistant|system)\n|<AI>|<用户>)", re.DOTALL)
        matches = pattern.findall(response)
        if len(matches) > 0:
            response = matches[0]
        history.append({"role": "assistant", "content": response})
        return response, history


@add_start_docstrings(
    """
    The MiniCPM Model transformer with a sequence classification head on top (linear layer).

    [`MiniCPMForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    MINICPM_START_DOCSTRING,
)
class MiniCPMForSequenceClassification(MiniCPMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = MiniCPMModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias_attr=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(MINICPM_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: paddle.LongTensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.LongTensor] = None,
        past_key_values: Optional[List[paddle.FloatTensor]] = None,
        inputs_embeds: Optional[paddle.FloatTensor] = None,
        labels: Optional[paddle.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        """
        labels (`paddle.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in
            `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed
            (Mean-Square loss), and if `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)
        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]
        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        elif input_ids is not None:
            sequence_lengths = (paddle.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1).to(logits.device)
        else:
            sequence_lengths = -1
        pooled_logits = logits[paddle.arange(batch_size, device=logits.device), sequence_lengths]
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == paddle.long or labels.dtype == paddle.int32):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
            if self.config.problem_type == "regression":
                loss_fct = paddle.nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = paddle.nn.CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = paddle.nn.BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return (loss,) + output if loss is not None else output
        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class MiniCPMForCausalLMPipe(GeneralModelForCausalLMPipe):
    config_class = MiniCPMConfig
    _decoder_layer_cls = MiniCPMDecoderLayer
    _get_tensor_parallel_mappings = MiniCPMModel._get_tensor_parallel_mappings
    _init_weights = MiniCPMModel._init_weights
    _keep_in_fp32_modules = MiniCPMModel._keep_in_fp32_modules
    _tied_weights_keys = ["lm_head.weight"]
    transpose_weight_keys = MiniCPMModel.transpose_weight_keys
    _gen_aoa_config = MiniCPMForCausalLM._gen_aoa_config
    _gen_inv_aoa_config = MiniCPMForCausalLM._gen_inv_aoa_config


__all__ = ["MiniCPMModel", "MiniCPMForCausalLM", "MiniCPMForCausalLMPipe"]
