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

# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import numbers

import paddle
from paddle import Tensor
from paddle.nn import init
from paddle.nn.parameter import Parameter

from paddleformers.fleet.transformer import TransformerConfig

HAVE_PERSIST_LAYER_NORM = False

try:
    from paddle.incubate.nn.functional.fused_layer_norm import fused_layer_norm

    HAVE_FUSED_LAYER_NORM = True
except ImportError:
    HAVE_FUSED_LAYER_NORM = False


class FusedLayerNorm(paddle.nn.Layer):
    """Layer Norm, fused into a single CUDA kernel.

    Args:
      hidden_size (int): Transformer hidden dimension.

      eps (float): Epsilon added to denominator, for numerical stability.

      persist_layer_norm (bool): Use persistent fused layer norm kernel.
      This kernel supports only a set of hidden sizes. Please
      check persist_ln_hidden_sizes if your hidden size is supported.

      zero_centered_gamma (bool): Adjust LayerNorm weights such that they are
      centered around zero. This improves numerical stability.

      config (TransformerConfig): Transformer config. Include to match custom
      layer norm interfaces.

      normalization (str): Normalization type, used for Transformer Engine.
      Must equal 'LayerNorm' here.
    """

    def __init__(
        self,
        config: TransformerConfig,
        hidden_size: int,
        eps: float = 1e-5,
        persist_layer_norm: bool = True,
        zero_centered_gamma: bool = False,
        normalization: str = "LayerNorm",  # included to match TE interface
    ):
        super().__init__()

        self.config = config

        self.zero_centered_gamma = self.config.layernorm_zero_centered_gamma
        assert self.config.normalization == "LayerNorm", (
            f"({self.config.normalization}) is not supported in FusedLayerNorm"
        )

        # List of hiddens sizes supported in the persistent layer norm kernel
        # If the hidden size is not supported, fall back to the non-persistent
        # kernel.
        persist_ln_hidden_sizes = [
            1024,
            1536,
            2048,
            2304,
            3072,
            3840,
            4096,
            5120,
            6144,
            8192,
            10240,
            12288,
            12800,
            15360,
            16384,
            18432,
            20480,
            24576,
            25600,
            30720,
            32768,
            40960,
            49152,
            65536,
        ]
        persist_layer_norm = self.config.persist_layer_norm
        if (
            hidden_size not in persist_ln_hidden_sizes
            or not HAVE_PERSIST_LAYER_NORM
        ):
            persist_layer_norm = False

        if not persist_layer_norm and not HAVE_FUSED_LAYER_NORM:
            # TODO: Add paddle only layer norm
            raise ValueError("Apex must be installed to use FusedLayerNorm.")

        if isinstance(hidden_size, numbers.Integral):
            hidden_size = (hidden_size,)
        self.hidden_size = hidden_size
        self.eps = eps
        # Parameters need to be initialized with paddle.empty rather than paddle.Tensor for correct device placement .
        self.weight = Parameter(paddle.empty(*hidden_size))
        self.bias = Parameter(paddle.empty(*hidden_size))
        self.reset_parameters()
        self.persist_layer_norm = persist_layer_norm
        self.sequence_parallel = self.config.sequence_parallel

        # set sequence parallelism flag on weight and bias parameters
        self.weight.sequence_parallel = self.sequence_parallel
        self.bias.sequence_parallel = self.sequence_parallel

    def reset_parameters(self):
        if self.zero_centered_gamma:
            init.zeros_(self.weight)
            init.zeros_(self.bias)
        else:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        weight = self.weight + 1 if self.zero_centered_gamma else self.weight

        input_shape = list(input.shape)
        input_ndim = len(input_shape)
        if isinstance(self.hidden_size, numbers.Integral):
            self.hidden_size = [self.hidden_size]
        elif isinstance(self.hidden_size, tuple):
            self.hidden_size = list(self.hidden_size)
        elif not isinstance(self.hidden_size, list):
            raise ValueError(
                "`self.hidden_size` should be int, list of ints or tuple of ints."
            )

        normalized_ndim = len(self.hidden_size)
        begin_norm_axis = input_ndim - normalized_ndim
        if input_ndim < normalized_ndim or (
            not paddle.utils.is_same_shape(
                input_shape[begin_norm_axis:], self.hidden_size
            )
        ):
            str_normalized_shape = str(self.hidden_size)
            raise ValueError(
                "Given normalized_shape is "
                + str_normalized_shape
                + ", expected input with shape [*, "
                + str_normalized_shape[1:]
                + ", but got input shape "
                + str(input_shape)
            )
        output = fused_layer_norm(
            input,
            weight.to(dtype="float32"),
            self.bias.to(dtype="float32"),
            self.eps,
            begin_norm_axis=begin_norm_axis,
        )

        return output[0]
