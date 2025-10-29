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

import paddle
import paddle.nn as nn

from ..generation.configuration_utils import PretrainedConfig
from ..utils.log import logger
from .general import GeneralInterface

try:
    from paddle.distributed.fleet.utils.sequence_parallel_utils import (
        mark_as_sequence_parallel_parameter,
    )
except ImportError:
    logger.warning_once("Fail to import mark_as_sequence_parallel_parameter!")

    def mark_as_sequence_parallel_parameter(parameter):
        return parameter


try:
    from paddle.incubate.nn.functional import fused_rms_norm_ext
except ImportError:
    logger.warning_once("Fail to import fused_rms_norm_ext!")
    fused_rms_norm_ext = None

__all__ = ["Norm"]


class LayerNorm(nn.LayerNorm):
    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size=None,
        norm_eps=None,
        has_bias=None,
        input_is_parallel=False,
        **kwargs
    ):
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.norm_eps = config.get("norm_eps", 1e-5) if norm_eps is None else norm_eps
        super().__init__(self.hidden_size, epsilon=self.norm_eps)
        self.config = config

        if input_is_parallel:
            self.enable_sequence_parallel()

    def enable_sequence_parallel(self):
        mark_as_sequence_parallel_parameter(self.weight)
        if self.bias is not None:
            mark_as_sequence_parallel_parameter(self.bias)


class RMSNorm(nn.Layer):
    def __init__(self, config: PretrainedConfig, hidden_size=None, norm_eps=None, input_is_parallel=False, **kwargs):
        super().__init__()
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.variance_epsilon = config.rms_norm_eps if norm_eps is None else norm_eps
        self.weight = paddle.create_parameter(
            shape=[self.hidden_size],
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.config = config

        if input_is_parallel:
            self.enable_sequence_parallel()

    def forward(self, hidden_states):
        if self.config.get("fuse_rms_norm", False):
            return fused_rms_norm_ext(hidden_states, self.weight, self.variance_epsilon)[0].astype(self.weight.dtype)

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

    def enable_sequence_parallel(self):
        mark_as_sequence_parallel_parameter(self.weight)


class Norm(GeneralInterface):
    _global_mapping = {"layer_norm": LayerNorm, "rms_norm": RMSNorm}

    @classmethod
    def create(
        self, config, hidden_size=None, has_bias=None, norm_eps=None, norm_type=None, input_is_parallel=False, **kwargs
    ):
        if norm_type is None:
            norm_type = "rms_norm"
        if has_bias is None:
            has_bias = config.get("use_bias", False)
        norm_cls = self._global_mapping[norm_type]
        return norm_cls(
            config, hidden_size, has_bias=has_bias, norm_eps=norm_eps, input_is_parallel=input_is_parallel, **kwargs
        )
