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
from paddle.incubate.nn.functional import swiglu as fused_swiglu

from ..generation.configuration_utils import PretrainedConfig
from .activation import ACT2FN
from .linear import Linear

__all__ = ["MLP"]


class MLP(nn.Layer):
    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size=None,
        intermediate_size=None,
        has_bias=None,
        fuse_up_gate=False,
        gate_proj_name="gate_proj",
        up_proj_name="up_proj",
        gate_up_proj_name="up_gate_proj",
        down_proj_name="down_proj",
        **kwargs
    ):
        super().__init__()
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.tensor_parallel = config.tensor_parallel_degree > 1
        self.has_bias = has_bias if has_bias else config.get("mlp_bias", False)
        self.fuse_swiglu = config.get("fuse_swiglu", False)
        self.act_type = config.get("hidden_act", "silu")
        self.act_fn = ACT2FN[self.act_type]
        self.fuse_up_gate = fuse_up_gate

        if self.fuse_up_gate:
            setattr(
                self,
                gate_up_proj_name,
                Linear.create(
                    self.hidden_size,
                    self.intermediate_size * 2,
                    has_bias=self.has_bias,
                    config=config,
                    fuse_matmul_bias=config.fuse_linear,
                    tp_plan="colwise",
                ),
            )
            self.up_gate_proj = getattr(self, gate_up_proj_name)
        else:
            # set attr for gate_proj
            setattr(
                self,
                gate_proj_name,
                Linear.create(
                    self.hidden_size,
                    self.intermediate_size,
                    has_bias=self.has_bias,
                    config=config,
                    fuse_matmul_bias=config.fuse_linear,
                    tp_plan="colwise",
                ),
            )
            self.gate_proj = getattr(self, gate_proj_name)

            # set attr for up_proj
            setattr(
                self,
                up_proj_name,
                Linear.create(
                    self.hidden_size,
                    self.intermediate_size,
                    has_bias=self.has_bias,
                    config=config,
                    fuse_matmul_bias=config.fuse_linear,
                    tp_plan="colwise",
                ),
            )
            self.up_proj = getattr(self, up_proj_name)

        # set attr for down_proj
        setattr(
            self,
            down_proj_name,
            Linear.create(
                self.intermediate_size,
                self.hidden_size,
                has_bias=self.has_bias,
                config=config,
                fuse_matmul_bias=config.fuse_linear,
                tp_plan="rowwise",
            ),
        )
        self.down_proj = getattr(self, down_proj_name)

    def forward(self, x):
        if self.fuse_up_gate:
            if self.fuse_swiglu:
                x = self.up_gate_proj(x)
                x = fused_swiglu(x)
            else:
                gate, x = self.up_gate_proj(x).chunk(2, axis=-1)
                x = self.act_fn(gate) * x
        else:
            gate = self.gate_proj(x)
            up = self.up_proj(x)
            if self.fuse_swiglu:
                x = paddle.concat([gate, up], axis=-1)
                x = fused_swiglu(x)
            else:
                x = self.act_fn(gate) * up
        return self.down_proj(x)
