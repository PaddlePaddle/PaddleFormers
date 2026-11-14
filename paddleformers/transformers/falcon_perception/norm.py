# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.
"""RMS normalization with the source model's opmath epsilon semantics."""
import paddle
from paddle import nn
from paddle.nn import functional as F


def rms_norm(x, normalized_shape, weight=None, eps=None):
    # PyTorch 2.11 uses the float32 compute epsilon for half/bfloat16 inputs.
    if eps is None:
        eps = 2.220446049250313e-16 if x.dtype == paddle.float64 else 1.1920928955078125e-7
    return F.rms_norm(x, normalized_shape, weight=weight, eps=eps)


class RMSNorm(nn.Layer):
    def __init__(self, dim, eps=None):
        super().__init__()
        self.eps = eps
        self.weight = self.create_parameter([dim], default_initializer=nn.initializer.Constant(1.0))

    def forward(self, x):
        return rms_norm(x, [x.shape[-1]], weight=self.weight, eps=self.eps)
