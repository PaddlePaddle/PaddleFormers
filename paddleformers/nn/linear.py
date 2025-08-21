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
        "colwise": ColumnParallelLinear,
        "rowwise": RowParallelLinear,
        "sequence_colwise": ColumnSequenceParallelLinear,
        "sequence_rowwise": RowSequenceParallelLinear,
    }

    @classmethod
    def create(
        self,
        in_features,
        out_features,
        weight_attr=None,
        has_bias: bool = None,
        linear_type: str = None,
        tp_plan: str = "colwise",
        config: PretrainedConfig = None,
        gather_output: bool = False,
        input_is_parallel: bool = True,
        fuse_matmul_bias: bool = False,
    ):
        if linear_type is None and config is None:
            raise ValueError("linear_type or config must be specified")

        if linear_type is None and config is not None:
            linear_type = self.get_linear_type(config, tp_plan)

        linear_cls = self._global_mapping[linear_type]
        kwargs = self.get_linear_kwargs(linear_type, has_bias, gather_output, input_is_parallel, fuse_matmul_bias)
        return linear_cls(in_features=in_features, out_features=out_features, weight_attr=weight_attr, **kwargs)

    @classmethod
    def get_linear_type(self, config: PretrainedConfig, tp_plan=None):
        if config.tensor_parallel_degree <= 1:
            if config.get("fuse_linear", False):
                return "fuse_linear"
            else:
                return "default"
        linear_type = tp_plan

        if config.sequence_parallel:
            linear_type = "sequence_" + linear_type
        return linear_type

    @classmethod
    def get_linear_kwargs(
        self, linear_type, has_bias=False, gather_output=False, input_is_parallel=True, fuse_matmul_bias=False
    ):
        ALL_LINEAR_KWARGS = {
            "default": {"bias_attr": has_bias},
            "fuse_linear": {"bias_attr": has_bias},
            "colwise": {
                "has_bias": has_bias,
                "gather_output": gather_output,
                "fuse_matmul_bias": fuse_matmul_bias,
            },
            "rowwise": {
                "has_bias": has_bias,
                "input_is_parallel": input_is_parallel,
                "fuse_matmul_bias": fuse_matmul_bias,
            },
            "sequence_colwise": {
                "has_bias": has_bias,
                "gather_output": gather_output,
                "fuse_matmul_bias": fuse_matmul_bias,
            },
            "sequence_rowwise": {
                "has_bias": has_bias,
                "input_is_parallel": input_is_parallel,
                "fuse_matmul_bias": fuse_matmul_bias,
            },
        }

        return ALL_LINEAR_KWARGS[linear_type]
