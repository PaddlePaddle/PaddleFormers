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

import paddle
from paddle import Tensor, nn

from paddleformers.fleet.transformer.enums import AttnMaskType
from paddleformers.fleet.transformer.utils import (
    get_default_causal_mask,
    get_sliding_window_causal_mask,
)


class SoftmaxOne(nn.Layer):
    r"""
    Softmax-off-by-one function as introduced in
    https://www.evanmiller.org/attention-is-off-by-one.html
    Supports fixed or learnable offset
    """

    def __init__(
        self, dim: int | None = None, denominator_offset: Tensor | float = 1.0
    ) -> None:
        super().__init__()
        self.dim = dim
        self.denominator_offset = denominator_offset

    def forward(self, x: Tensor) -> Tensor:
        """forward pass"""
        # sink: [np] --> [1, np, 1, 1] --> [b, np, sq, 1]
        sink = self.denominator_offset.reshape(1, -1, 1, 1).expand(
            x.size(0), -1, x.size(2), -1
        )
        # qk: [b, np, sq, sk] --> [b, np, sq, sk+1]
        qk = paddle.concat([x, sink], axis=-1)
        # do softmax, and remove sink token at the end
        ret = paddle.softmax(qk, axis=-1)[..., :-1]
        return ret


class FusedScaleMaskSoftmax(nn.Layer):
    """
    fused operation: scaling + mask + softmax

    Args:
        input_in_fp16: flag to indicate if input in fp16 data format.
        input_in_bf16: flag to indicate if input in bf16 data format.
        attn_mask_type: attention mask type (pad or causal)
        scaled_masked_softmax_fusion: flag to indicate user want to use softmax fusion
        mask_func: mask function to be applied.
        softmax_in_fp32: if true, softmax in performed at fp32 precision.
        scale: scaling factor used in input tensor scaling.
    """

    def __init__(
        self,
        input_in_fp16,
        input_in_bf16,
        attn_mask_type,
        scaled_masked_softmax_fusion,
        mask_func,
        softmax_in_fp32,
        scale,
        sliding_window=None,
    ):
        super().__init__()
        self.input_in_fp16 = input_in_fp16
        self.input_in_bf16 = input_in_bf16
        assert not (self.input_in_fp16 and self.input_in_bf16), (
            "both fp16 and bf16 flags cannot be active at the same time."
        )
        self.input_in_float16 = self.input_in_fp16 or self.input_in_bf16
        self.attn_mask_type = attn_mask_type
        self.scaled_masked_softmax_fusion = scaled_masked_softmax_fusion
        self.mask_func = mask_func
        self.softmax_in_fp32 = softmax_in_fp32
        self.scale = scale
        self.sliding_window = sliding_window
        assert self.scale is None or softmax_in_fp32, (
            "softmax should be in fp32 when scaled"
        )

    def forward(
        self,
        input: Tensor,
        mask: Tensor | None,
        softmax_offset: Tensor | None = None,
    ):
        """Forward pass of softmax with masked input.

        In case attn_mask_type is causal the mask is generated and None can be passed.
        A user-defined mask is only needed when attn_mask_type is not causal.
        """
        # [b, np, sq, sk]
        assert input.dim() == 4

        if self.input_in_float16 and self.softmax_in_fp32:
            input = input.float()

        if self.scale is not None:
            input = input * self.scale

        # Generate causal mask if not given
        sq, sk = input.shape[2], input.shape[3]
        if self.sliding_window is not None:
            mask = get_sliding_window_causal_mask(sq, sk, self.sliding_window)
        elif (
            self.attn_mask_type == AttnMaskType.causal
            and mask is None
            and sq > 1
        ):
            # If sq == 1 then either KV cache is used or one-element context is passed
            # so keeping mask=None in this case; subsequent code should handle it
            assert sq == sk, "causal mask is only for self attention"
            mask = get_default_causal_mask(sq)

        mask_output = self.mask_func(input, mask) if mask is not None else input
        if softmax_offset is None:
            softmax_fn = paddle.nn.Softmax(axis=-1)
        else:
            softmax_fn = SoftmaxOne(-1, softmax_offset.to(input.place))

        probs = softmax_fn(mask_output)
        if self.input_in_float16 and self.softmax_in_fp32:
            if self.input_in_fp16:
                probs = probs.half()
            else:
                probs = probs.bfloat16()

        return probs
