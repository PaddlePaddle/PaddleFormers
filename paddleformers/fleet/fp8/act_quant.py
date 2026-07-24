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

"""Block-wise FP8 activation quantization for DeepSeek-V4."""

import paddle

FP8_MAX = 448.0
FP8_MIN = -448.0


def act_quant(
    x: paddle.Tensor,
    block_size: int = 128,
    scale_fmt: str | None = None,
    inplace: bool = False,
) -> paddle.Tensor:
    """Block-wise FP8 quantization using pure paddle ops.

    Each block of `block_size` elements along the last dim is independently scaled.
    """
    N = x.shape[-1]
    assert N % block_size == 0, "N must be divisible by block_size"
    orig_shape = x.shape

    # reshape to (..., num_blocks, block_size)
    z = (
        x.contiguous()
        .reshape([-1, N // block_size, block_size])
        .astype("float32")
    )

    # per-block absmax -> scale
    amax = z.abs().max(axis=-1, keepdim=True).clip(min=1e-4)
    if scale_fmt is not None:
        # round scale to power-of-2 (MXFP style)
        scale = 2.0 ** paddle.ceil(paddle.log2(amax / FP8_MAX))
    else:
        scale = amax / FP8_MAX

    # quantize: clamp to fp8 range
    y_q = (z / scale).clip(min=FP8_MIN, max=FP8_MAX)

    if inplace:
        # Match tilelang kernel: Cast(bf16, Cast(fp32, Cast(bf16, clamp(x/s))) * s)
        # In inplace mode, out_dtype=in_dtype (bf16), NOT fp8
        y_bf16 = y_q.astype(
            x.dtype
        )  # cast to bf16 (simulates quantization rounding)
        y_dq = y_bf16.astype("float32") * scale
        x_out = y_dq.reshape(orig_shape).astype(x.dtype)
        paddle.assign(x_out, x)
        return x

    # Return quantized values (as float8) and scales
    y_out = paddle.cast(y_q, "float8_e4m3fn").reshape(orig_shape)
    s_out = scale.squeeze(-1).reshape([*list(orig_shape[:-1]), N // block_size])
    return y_out, s_out
