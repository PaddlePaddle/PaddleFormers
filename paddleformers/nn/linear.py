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

import paddle.nn as nn
from paddle.incubate.nn import FusedLinear

from ..transformers.configuration_utils import PretrainedConfig
from ..transformers.linear_utils import (
    ColumnParallelLinear,
    ColumnSequenceParallelLinear,
    RowParallelLinear,
    RowSequenceParallelLinear,
)
from .general import GeneralInterface

__all__ = ["Linear"]


class Linear(GeneralInterface):
    _global_mapping = {
        "default": nn.Linear,
        "fuse_linear": FusedLinear,
        "colwise_parallel": ColumnParallelLinear,
        "rowwise_parallel": RowParallelLinear,
        "sequence_colwise_parallel": ColumnSequenceParallelLinear,
        "sequence_rowwise_parallel": RowSequenceParallelLinear,
    }

    @classmethod
    def create(self, in_features, out_features, weight_attr=None, has_bias=None, **kwargs):
        linear_type = kwargs.pop("linear_type", "default")
        linear_cls = self._global_mapping[linear_type]
        kwargs = self.process_kwargs(linear_type, has_bias, **kwargs)
        return linear_cls(in_features=in_features, out_features=out_features, weight_attr=weight_attr, **kwargs)

    @classmethod
    def process_kwargs(self, linear_type, has_bias, **kwargs):
        # validate kwargs

        assert (
            kwargs.get("bias_attr", None) is None or has_bias is None
        ), "bias_attr and has_bias can not be simultaneously specified"

        # add default kwargs
        if linear_type in ("default", "fuse_linear"):
            kwargs["bias_attr"] = has_bias
        else:
            kwargs.pop("bias_attr", None)
            kwargs["has_bias"] = has_bias
        return kwargs

    @classmethod
    def get_linear_type(self, config: PretrainedConfig, is_column_parallel=True):
        if config.tensor_parallel_degree <= 1:
            if config.get("fuse_linear", False):
                return "fuse_linear"
            else:
                return "default"
        linear_type = "colwise_parallel" if is_column_parallel else "rowwise_parallel"

        if config.sequence_parallel:
            linear_type = "sequence_" + linear_type
        return linear_type

    @classmethod
    def get_linear_kwargs(self, linear_type, gather_output=False, input_is_parallel=True, fuse_matmul_bias=False):
        ALL_LINEAR_KWARGS = {
            "default": {"linear_type": linear_type},
            "fuse_linear": {"linear_type": linear_type},
            "colwise_parallel": {
                "linear_type": linear_type,
                "gather_output": gather_output,
                "fuse_matmul_bias": fuse_matmul_bias,
            },
            "rowwise_parallel": {
                "linear_type": linear_type,
                "input_is_parallel": input_is_parallel,
                "fuse_matmul_bias": fuse_matmul_bias,
            },
            "sequence_colwise_parallel": {
                "linear_type": linear_type,
                "gather_output": gather_output,
                "fuse_matmul_bias": fuse_matmul_bias,
            },
            "sequence_rowwise_parallel": {
                "linear_type": linear_type,
                "input_is_parallel": input_is_parallel,
                "fuse_matmul_bias": fuse_matmul_bias,
            },
        }

        return ALL_LINEAR_KWARGS[linear_type]
