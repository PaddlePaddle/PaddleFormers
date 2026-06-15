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

from __future__ import annotations

import logging
import math
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from paddleformers.fleet.transformer import TransformerConfig

import paddle
from paddle import Tensor

from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
    RotaryEmbedding,
)

logger = logging.getLogger(__name__)


class YarnRotaryEmbedding(RotaryEmbedding):
    """Yarn Rotary Embedding for language model.

    Args:
        head_dim (int): Projection weights dimension in multi-head attention. Obtained from
            transformer config.
        rotary_percent (float): Percent of rotary dimension to use for rotary position embeddings.
        rotary_interleaved (bool, optional): If True, interleaved rotary position embeddings.
            Defaults to False.
        seq_len_interpolation_factor (float, optional): scale of linearly interpolating RoPE for
            longer sequences. The value must be a float larger than 1.0. Defaults to None
        rotary_base (float, optional): Base period for rotary position embeddings. Defaults to
            10000.
        scaling_factor (float, optional): Scaling factor for Yarn RoPE. Defaults to 1.0.
        original_max_position_embeddings (int, optional): Original maximum position embeddings
            length. Defaults to 4096.
        beta_fast (float, optional): Fast beta value for Yarn RoPE. Defaults to 32.
        beta_slow (float, optional): Slow beta value for Yarn RoPE. Defaults to 1.
        mscale (float, optional): Mscale value for Yarn RoPE. Defaults to 1.
        mscale_all_dim (float, optional): Mscale all dim value for Yarn RoPE. Defaults to 0.
        correction_range_round_to_int (bool): Whether to round dim range bounds to integer.
            Defaults to True
    """

    def __init__(
        self,
        head_dim: int,
        rotary_percent: float = 1.0,
        rotary_interleaved: bool = False,
        seq_len_interpolation_factor: float | None = None,
        rotary_base: float = 10000.0,
        scaling_factor: float = 1.0,
        original_max_position_embeddings: int = 4096,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        mscale: float = 1.0,
        mscale_all_dim: float = 0.0,
        correction_range_round_to_int: bool = True,
    ):
        self.dim = head_dim
        self.rotary_base = rotary_base
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim
        self.correction_range_round_to_int = correction_range_round_to_int

        super().__init__(
            head_dim=head_dim,
            rotary_percent=rotary_percent,
            rotary_interleaved=rotary_interleaved,
            seq_len_interpolation_factor=seq_len_interpolation_factor,
            rotary_base=rotary_base,
        )

        self.inv_freq_extra = 1.0 / (
            self.rotary_base ** (paddle.arange(0, self.dim, 2).astype(paddle.float32) / self.dim)
        )
        self.inv_freq_inter = 1.0 / (
            self.scaling_factor * self.rotary_base ** (paddle.arange(0, self.dim, 2).astype(paddle.float32) / self.dim)
        )
        self._set_cos_sin_cache(
            self.original_max_position_embeddings,
            offset=0,
            dtype=paddle.get_default_dtype(),
        )

    def forward(
        self,
        max_seq_len: int,
        offset: int = 0,
        packed_seq: bool = False,
        position_ids: Tensor | None = None,
    ) -> Tensor:
        """Forward pass of Yarn Rotary Embedding.

        Args:
            max_seq_len (int): Maximum size of sequence
            offset (int, optional): RoPE offset. Defaults to 0.
            packed_seq (bool, optional): Whether to use packed sequence. Defaults to False.

        Returns:
            Tensor: Embeddings after applying Yarn RoPE.
        """
        low, high = _yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            self.dim,
            self.rotary_base,
            self.original_max_position_embeddings,
            self.correction_range_round_to_int,
        )
        inv_freq_mask = 1.0 - _yarn_linear_ramp_mask(low, high, self.dim // 2).to(dtype=paddle.float32)
        inv_freq = self.inv_freq_inter * (1 - inv_freq_mask) + self.inv_freq_extra * inv_freq_mask

        if position_ids is not None:
            # Handle different position_ids shapes:
            # - 1D [S]: use directly (also covers fastdeploy decode mode)
            # - 2D [B, S]: use first batch (assume all batches have same positions)
            # - 3D: not supported by RotaryEmbedding, use MultimodalRotaryEmbedding instead
            if position_ids.ndim == 1:
                seq = position_ids.astype(self.inv_freq.dtype)
            elif position_ids.ndim == 2:
                # Take first batch, assuming all batches have same position_ids
                seq = position_ids[0].astype(self.inv_freq.dtype)
            else:
                # For 3D position_ids (M-RoPE), this function should not be called
                # Fall back to max_seq_len to avoid cryptic errors
                seq = paddle.arange(max_seq_len).astype(self.inv_freq.dtype) + offset
        else:
            seq = (
                paddle.arange(
                    max_seq_len,
                ).astype(self.inv_freq_extra.dtype)
                + offset
            )

        freqs = paddle.outer(seq, inv_freq)

        _mscale = _yarn_get_concentration_factor(self.scaling_factor, self.mscale, self.mscale_all_dim)

        if not self.rotary_interleaved:
            emb = paddle.cat((freqs, freqs), axis=-1)
        else:
            emb = paddle.stack((freqs.view(-1, 1), freqs.view(-1, 1)), axis=-1).view(freqs.shape[0], -1)
        # emb [1, seq_len, 1, dim]
        emb = emb[None, :, None, :]
        return emb, _mscale

    def _set_cos_sin_cache(self, seq_len, offset, dtype, packed_seq=False):
        self.max_seq_len_cached = seq_len
        self.offset_cached = offset
        self.dtype_cached = dtype
        self.packed_seq_cached = packed_seq

        emb, _mscale = self.forward(seq_len, offset, packed_seq)
        self.register_buffer(
            "cos_cached",
            (emb.cos() * _mscale).to(dtype).contiguous(),
            persistable=False,
        )
        self.register_buffer(
            "sin_cached",
            (emb.sin() * _mscale).to(dtype).contiguous(),
            persistable=False,
        )

    def get_cached_cos_sin(
        self,
        seq_len,
        offset=0,
        dtype=paddle.get_default_dtype(),
        packed_seq=False,
    ):
        """Get cached cos and sin values."""
        if (
            seq_len > self.max_seq_len_cached
            or offset != self.offset_cached
            or dtype != self.dtype_cached
            or packed_seq != self.packed_seq_cached
        ):
            self._set_cos_sin_cache(seq_len, offset, dtype, packed_seq)
        return (self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...])


# Inverse dim formula to find dim based on number of rotations
def _yarn_find_correction_dim(
    num_rotations: float,
    dim: int,
    rotary_base: float = 10000,
    max_position_embeddings: int = 2048,
) -> float:
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(rotary_base))


# Find dim range bounds based on rotations
def _yarn_find_correction_range(
    low_rot: float,
    high_rot: float,
    dim: int,
    rotary_base: float = 10000,
    max_position_embeddings: int = 2048,
    round_to_int: bool = True,
) -> tuple[int, int]:
    low = _yarn_find_correction_dim(low_rot, dim, rotary_base, max_position_embeddings)
    high = _yarn_find_correction_dim(high_rot, dim, rotary_base, max_position_embeddings)
    if round_to_int:
        low = math.floor(low)
        high = math.ceil(high)
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def _yarn_linear_ramp_mask(min: float, max: float, dim: int) -> Tensor:
    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (paddle.arange(dim).astype(paddle.float32) - min) / (max - min)
    ramp_func = paddle.clamp(linear_func, 0, 1)
    return ramp_func


def _yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


@lru_cache(maxsize=8)
def _yarn_get_concentration_factor(scaling_factor: float, mscale: float, mscale_all_dim: float) -> float:
    """
    Get the concentration factor (factor multiplied to the sine and cosine components of the
    embedding). This factor is also known as attention factor, and sometimes homonymously known as
    "mscale"
    """
    return float(_yarn_get_mscale(scaling_factor, mscale) / _yarn_get_mscale(scaling_factor, mscale_all_dim))


def _yarn_get_concentration_factor_from_config(
    config: TransformerConfig,
) -> float:
    fields = [
        "yarn_rotary_scaling_factor",
        "yarn_mscale",
        "yarn_mscale_all_dim",
    ]
    if all(hasattr(config, f) for f in fields):
        return _yarn_get_concentration_factor(
            config.yarn_rotary_scaling_factor,
            config.yarn_mscale,
            config.yarn_mscale_all_dim,
        )
    return 1.0
