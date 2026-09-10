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

"""Layout- and partition-independent L2 norm of FP32 gradients.

Squares are computed in FP32 on the gradient device. Their mantissas are
accumulated into base-65536 integer bins, which may be SUM-reduced across
owners before rounding the total once to FP32 and applying the native sqrt.
No gradient values or floating-point norm computation leave the device.

The 20 limbs cover FP32 squares and the supported 2**40 global element count.
Each uncarried limb is bounded by 2**40 * 65535 < 2**56, so arbitrary parameter
and rank reduction orders cannot overflow int64. The last three bins count
infinities, NaNs, and elements. This deliberately costs more than a fused norm
and is intended only for explicitly enabled accuracy compatibility.
"""

from __future__ import annotations

import paddle


class ReproducibleL2Norm:
    """Accumulate on a single device; reduce bins before calling ``finish``."""

    def __init__(self, place: paddle.base.libpaddle.Place | None = None) -> None:
        self.place = place

    def cast(self, value: paddle.Tensor, dtype: str) -> paddle.Tensor:
        return value.astype(dtype)

    def zeros(self, count: int = 23) -> paddle.Tensor:
        return self.tensor([0] * count, "int64")

    def tensor(self, value: list[int | float] | int | float, dtype: str = "float32") -> paddle.Tensor:
        return paddle.to_tensor(value, dtype=dtype, place=self.place)

    def view(self, value: paddle.Tensor, dtype: str) -> paddle.Tensor:
        return value.view(getattr(paddle, dtype))

    def add(self, bins: paddle.Tensor, indices: paddle.Tensor, values: paddle.Tensor) -> paddle.Tensor:
        return paddle.scatter_nd_add(bins, indices.reshape([-1, 1]), values)

    def accumulate(self, bins: paddle.Tensor, gradient: paddle.Tensor, chunk_size: int = 1048576) -> paddle.Tensor:
        if gradient.dtype != paddle.float32:
            raise TypeError("Reproducible clipping requires FP32 gradients")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        flat = gradient.reshape([-1])
        if flat.shape[0] > 2**40:
            raise OverflowError("Reproducible norm supports at most 2**40 global elements")
        bins = self.add(bins, self.tensor([22], "int64"), self.tensor([flat.shape[0]], "int64"))
        for start in range(0, flat.shape[0], chunk_size):
            value = flat[start : start + chunk_size]
            bits = self.cast(self.view(value * value, "int32"), "int64")
            exponent = (bits >> 23) & self.tensor(255, "int64")
            fraction = bits & self.tensor(0x7FFFFF, "int64")
            finite = exponent != 255
            mantissa = fraction | paddle.where(
                exponent > 0, paddle.full_like(exponent, 0x800000), paddle.zeros_like(exponent)
            )
            shift = paddle.maximum(exponent - 1, paddle.zeros_like(exponent))
            limb = shift // 16
            shifted = (mantissa << (shift % 16)) * self.cast(finite, "int64")
            for offset in range(3):
                bins = self.add(bins, limb + offset, (shifted >> (16 * offset)) & self.tensor(65535, "int64"))
            flags = paddle.stack(
                [
                    self.cast((exponent == 255) & (fraction == 0), "int64").sum(),
                    self.cast((exponent == 255) & (fraction != 0), "int64").sum(),
                ]
            )
            bins = self.add(bins, self.tensor([20, 21], "int64"), flags)
        return bins

    def finish(self, bins: paddle.Tensor) -> tuple[paddle.Tensor, paddle.Tensor]:
        if int(bins[22].item()) > 2**40:
            raise OverflowError("Reproducible norm supports at most 2**40 global elements")
        carry = self.zeros(1)[0]
        digits = []
        for i in range(20):
            value = bins[i] + carry
            digits.append(value & self.tensor(65535, "int64"))
            carry = value >> 16
        digits = paddle.stack(digits)
        index = self.tensor(list(range(20)), "int64")
        top = paddle.where(digits != 0, index, paddle.full_like(index, -1)).max()

        def get(i):
            return paddle.where(index == i, digits, paddle.zeros_like(digits)).sum()

        word = get(top)
        leading = self.zeros(1)[0]
        for width in [8, 4, 2, 1]:
            take = word >= (1 << width)
            word = paddle.where(take, word >> width, word)
            leading = leading + self.cast(take, "int64") * width
        highest = top * 16 + leading
        cut = paddle.maximum(highest - 23, self.zeros(1)[0])
        limb, shift = cut // 16, cut % 16
        significand = (
            (get(limb) >> shift) | (get(limb + 1) << (16 - shift)) | (get(limb + 2) << (32 - shift))
        ) & self.tensor(0xFFFFFF, "int64")
        round_position = paddle.maximum(cut - 1, self.zeros(1)[0])
        round_limb, round_shift = round_position // 16, round_position % 16
        round_word = get(round_limb)
        round_bit = ((round_word >> round_shift) & self.tensor(1, "int64")) * self.cast(cut > 0, "int64")
        sticky = paddle.where(index < round_limb, digits, paddle.zeros_like(digits)).sum() != 0
        sticky = sticky | ((round_word & ((self.tensor(1, "int64") << round_shift) - 1)) != 0)
        significand = significand + round_bit * self.cast(
            sticky | ((significand & self.tensor(1, "int64")) != 0), "int64"
        )
        exponent = highest - 22 + self.cast(significand == 0x1000000, "int64")
        raw = (exponent << 23) | (significand & self.tensor(0x7FFFFF, "int64"))
        raw = paddle.where(exponent >= 255, paddle.full_like(raw, 0x7F800000), raw)
        raw = paddle.where(highest < 23, get(0) | (get(1) << 16), raw)
        raw = paddle.where(top < 0, paddle.zeros_like(raw), raw)
        raw = paddle.where(bins[20] != 0, paddle.full_like(raw, 0x7F800000), raw)
        raw = paddle.where(bins[21] != 0, paddle.full_like(raw, 0x7FC00000), raw)
        square_sum = self.view(self.cast(raw.reshape([1]), "int32"), "float32")
        return paddle.sqrt(square_sum), square_sum


class ReproducibleClipGradByGlobalNorm(paddle.nn.ClipGradByGlobalNorm):
    """FP32 gradient clipping whose norm is invariant to tensor partitioning."""

    reproducible_norm = True

    @paddle.no_grad()
    def _dygraph_clip(self, params_grads):
        params_grads = list(params_grads)
        accumulator = ReproducibleL2Norm()
        bins = accumulator.zeros()
        for parameter, gradient in params_grads:
            if gradient is not None and getattr(parameter, "need_clip", True):
                bins = accumulator.accumulate(bins, gradient)
        norm, _ = accumulator.finish(bins)
        coefficient = paddle.minimum(self.clip_norm / (norm + 1e-6), paddle.ones_like(norm))
        for parameter, gradient in params_grads:
            if gradient is not None and getattr(parameter, "need_clip", True):
                gradient.multiply_(coefficient)
                parameter._reset_grad_inplace_version(True)
        return params_grads
