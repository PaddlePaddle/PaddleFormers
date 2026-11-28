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

"""Source-compatible temporal and golden spatial rotary positions."""
import paddle


def precompute_freqs_cis(dim, end, theta=10000.0):
    # The source computes these non-persistent frequencies on CPU, then moves
    # them to the model. GPU pow would introduce growing phase error at long positions.
    freqs = 1.0 / (theta ** (paddle.arange(0, dim, 2, dtype="float32", device="cpu")[: dim // 2] / dim))
    angles = paddle.outer(paddle.arange(end, dtype="float32", device="cpu"), freqs)
    return paddle.complex(paddle.cos(angles), paddle.sin(angles)).to(paddle.get_device())


def _rotate(x, cos, sin):
    even = x[..., 0::2].astype("float32")
    odd = x[..., 1::2].astype("float32")
    return paddle.stack([even * cos - odd * sin, even * sin + odd * cos], axis=-1).flatten(-2).astype(x.dtype)


def apply_rotary_emb(xq, xk, freqs_cis):
    cos, sin = freqs_cis.real().unsqueeze(2), freqs_cis.imag().unsqueeze(2)
    return _rotate(xq, cos, sin), _rotate(xk, cos, sin)


def apply_golden_freqs_cis_to_visual_pos(freqs_hFP, pos_BSP):
    valid = paddle.all(~paddle.isnan(pos_BSP), axis=-1)
    pos = pos_BSP[valid].astype("float32")
    theta = paddle.einsum("tp,hfp->thf", pos, freqs_hFP.astype("float32"))
    return paddle.complex(paddle.cos(theta), paddle.sin(theta))


def apply_golden_rotary_emb(x, freqs, positions):
    valid = paddle.all(~paddle.isnan(positions), axis=-1)
    output = x.clone()
    output[valid] = _rotate(x[valid], freqs.real(), freqs.imag())
    return output


def apply_3d_rotary_emb(xq, xk, freqs_cis, freqs_cis_2d, pos_hw):
    qt, qs = paddle.chunk(xq, 2, axis=-1)
    kt, ks = paddle.chunk(xk, 2, axis=-1)
    qt, kt = apply_rotary_emb(qt, kt, freqs_cis)
    if freqs_cis_2d is not None and pos_hw is not None:
        qs = apply_golden_rotary_emb(qs, freqs_cis_2d, pos_hw)
        ks = apply_golden_rotary_emb(ks, freqs_cis_2d, pos_hw)
    return paddle.concat([qt, qs], axis=-1), paddle.concat([kt, ks], axis=-1)
