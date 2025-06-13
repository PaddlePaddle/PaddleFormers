# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle import nn

__all__ = [
    "RotaryEmbedding",
    "LinearScalingRotaryEmbedding",
    "NTKScalingRotaryEmbedding",
    "DynamicNTKScalingRotaryEmbedding",
    "YaRNScalingRotaryEmbedding",
]


class RotaryEmbedding(nn.Layer):
    def __init__(self, **init_args):
        super().__init__()
        self.dim = init_args["dim"]
        self.max_position_embeddings = init_args["max_position_embeddings"]
        self.base = init_args["base"]
        self.position_encoding_2d = init_args["position_encoding_2d"] if "position_encoding_2d" in init_args else False
        if self.position_encoding_2d:
            # [dim / 4]# 2D--Embedding
            self.dim = self.dim / 2
            inv_freq = 1.0 / (
                self.base ** (paddle.cast(paddle.arange(0, self.dim, 2), dtype=paddle.float32) / self.dim)
            )
        else:
            # [dim / 2]
            inv_freq = 1.0 / (
                self.base ** (paddle.cast(paddle.arange(0, self.dim, 2), dtype=paddle.float32) / self.dim)
            )
        self.register_buffer("inv_freq", inv_freq)
        self._set_cos_sin_cache(seq_len=self.max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        # [seq_len]
        t = paddle.arange(seq_len, dtype=paddle.float32)
        # [seq_len, dim/2]
        with paddle.amp.auto_cast(enable=False):
            freqs = paddle.outer(t.astype(self.inv_freq.dtype), self.inv_freq)
        # [seq_len, dim]
        emb = paddle.concat([freqs, freqs], axis=-1)
        self.cos_cached = emb.cos()[:, :]
        self.sin_cached = emb.sin()[:, :]

    def forward(self, seq_len=None, ntk_alpha=None):

        return self.cos_cached[:, :], self.sin_cached[:, :]


class LinearScalingRotaryEmbedding(RotaryEmbedding):
    def __init__(self, **init_args):
        self.scaling_factor = init_args["scaling_factor"]
        super().__init__(**init_args)

    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        # [seq_len]
        t = paddle.arange(seq_len, dtype=paddle.float32)
        t = t / self.scaling_factor
        # [seq_len, dim/2]
        with paddle.amp.auto_cast(enable=False):
            freqs = paddle.outer(t.astype(self.inv_freq.dtype), self.inv_freq)
        # [seq_len, dim]
        emb = paddle.concat([freqs, freqs], axis=-1)
        self.cos_cached = emb.cos()[:, :]
        self.sin_cached = emb.sin()[:, :]


class NTKScalingRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with NTK scaling. https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/"""

    def __init__(self, **init_args):
        init_args["base"] = init_args["base"] * init_args["scaling_factor"] ** (
            init_args["dim"] / (init_args["dim"] - 2)
        )
        super().__init__(**init_args)


class DynamicNTKScalingRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with Dynamic NTK scaling. https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/"""

    def __init__(self, **init_args):
        self.scaling_factor = init_args["scaling_factor"]
        self._seq_len_cached = 0
        super().__init__(**init_args)

    def _scale_cos_sin(self, seq_len, ntk_alpha=None):
        # [seq_len]
        t = paddle.arange(seq_len, dtype=paddle.float32)
        if ntk_alpha is None:
            ntk_alpha = (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
        base = self.base * ntk_alpha ** (self.dim / (self.dim - 2))

        # [seq_len, dim/2]
        inv_freq = 1.0 / (base ** (paddle.cast(paddle.arange(0, self.dim, 2), dtype=paddle.float32) / self.dim))
        with paddle.amp.auto_cast(enable=False):
            freqs = paddle.outer(t.astype(inv_freq.dtype), inv_freq)
        # [seq_len, dim]
        emb = paddle.concat([freqs, freqs], axis=-1)
        self.cos_cached = emb.cos()[:, :]
        self.sin_cached = emb.sin()[:, :]

    def forward(self, seq_len=None, ntk_alpha=None):

        if seq_len > self.max_position_embeddings:
            self._scale_cos_sin(seq_len=seq_len, ntk_alpha=ntk_alpha)

        return self.cos_cached[:, :], self.sin_cached[:, :]


class YaRNScalingRotaryEmbedding(nn.Layer):
    """RotaryEmbedding extended with YaRN scaling."""

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        scaling_factor=1,
        original_max_position_embeddings=2048,
        extrapolation_factor=1,
        attn_factor=1,
        beta_fast=32,
        beta_slow=1,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor  # scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow

        self.yarn()

        self._set_cos_sin_cache(seq_len=self.max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        # [seq_len]
        t = paddle.arange(seq_len, dtype=paddle.float32)
        # [seq_len, dim/2]
        with paddle.amp.auto_cast(enable=False):
            freqs = paddle.outer(t.astype(self.inv_freq.dtype), self.inv_freq)
        # [seq_len, dim]
        emb = paddle.concat([freqs, freqs], axis=-1)
        self.cos_cached = emb.cos()[:, :] * self.mscale
        self.sin_cached = emb.sin()[:, :] * self.mscale

    def _scale_cos_sin(self, seq_len):
        self.max_seq_len_cached = seq_len

        t = paddle.arange(self.max_seq_len_cached, dtype=self.inv_freq.dtype)
        freqs = paddle.einsum("i,j->ij", t, self.inv_freq)
        emb = paddle.concat((freqs, freqs), axis=-1)

        self.cos_cached = emb.cos()[:, :] * self.mscale
        self.sin_cached = emb.sin()[:, :] * self.mscale

    def forward(self, seq_len=None, ntk_alpha=None):
        if seq_len > self.max_seq_len_cached:
            self._scale_cos_sin(seq_len=seq_len)

        return self.cos_cached[:, :], self.sin_cached[:, :]

    def yarn(self):
        inv_freq = 1.0 / (self.base ** (paddle.cast(paddle.arange(0, self.dim, 2), dtype=paddle.float32) / self.dim))

        low, high = self._yarn_find_correction_range(
            self.beta_fast, self.beta_slow, self.dim, self.base, self.original_max_position_embeddings
        )
        inv_freq_mask = (
            1 - paddle.cast(self._yarn_linear_ramp_mask(low, high, self.dim // 2), dtype=paddle.float32)
        ) * self.extrapolation_factor

        inv_freq = inv_freq / ((1 - inv_freq_mask) * self.scaling_factor + inv_freq_mask)
        self.register_buffer("inv_freq", inv_freq)
        self.mscale = self._yarn_get_mscale(self.scaling_factor) * self.attn_factor

    @classmethod
    def _yarn_find_correction_dim(cls, num_rotations, dim, base=10000, max_position_embeddings=2048):
        return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

    @classmethod
    def _yarn_find_correction_range(cls, low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
        low = math.floor(cls._yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
        high = math.ceil(cls._yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
        return max(low, 0), min(high, dim - 1)  # Clamp values just in case

    @classmethod
    def _yarn_linear_ramp_mask(cls, low, high, dim):
        if low == high:
            high += 0.001  # Prevent singularity

        linear_func = (paddle.arange(dim, dtype=paddle.float32) - low) / (high - low)
        ramp_func = paddle.clip(linear_func, 0, 1)
        return ramp_func

    @classmethod
    def _yarn_get_mscale(cls, scaling_factor=1):
        if scaling_factor <= 1:
            return 1.0
        return 0.1 * math.log(scaling_factor) + 1.0
