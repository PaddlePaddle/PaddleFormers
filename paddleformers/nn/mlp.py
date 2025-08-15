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
        skip_recompute_ops=None,
        gate_proj_name="gate_proj",
        up_proj_name="up_proj",
        gate_up_proj_name="up_gate_proj",
        down_proj_name="down_proj",
        **kwargs
    ):
        super().__init__()
        if skip_recompute_ops is None:
            skip_recompute_ops = {}

        self.skip_recompute_ops = skip_recompute_ops
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.tensor_parallel = config.tensor_parallel_degree > 1
        self.gate_up_linear_type = Linear.get_linear_type(config)
        self.down_linear_type = Linear.get_linear_type(config, is_column_parallel=False)
        self.gate_up_kwargs = Linear.get_linear_kwargs(self.gate_up_linear_type)
        self.down_kwargs = Linear.get_linear_kwargs(self.down_linear_type)
        self.has_bias = config.get("mlp_bias", False)
        self.fuse_swiglu = config.get("fuse_swiglu", False)
        self.act_type = "fused_swiglu" if self.fuse_swiglu else config.get("hidden_act", "silu")
        self.act_fn = ACT2FN[self.act_type]
        self.fuse_attention_ffn = getattr(config, "fuse_attention_ffn", False)

        if self.fuse_attention_ffn:
            setattr(
                self,
                gate_up_proj_name,
                Linear.create(
                    self.hidden_size,
                    self.intermediate_size * 2,
                    has_bias=self.has_bias,
                    **self.gate_up_kwargs,
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
                    **self.gate_up_kwargs,
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
                    **self.gate_up_kwargs,
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
                **self.down_kwargs,
            ),
        )
        self.down_proj = getattr(self, down_proj_name)

    def forward(self, x):
        if self.fuse_attention_ffn:
            if self.fuse_swiglu:
                x = self.up_gate_proj(x)
                x = self.act_fn(x)
            else:
                gate, x = self.up_gate_proj(x).chunk(2, axis=-1)
                x = self.act_fn(gate) * x
        else:
            gate = self.gate_proj(x)
            up = self.up_proj(x)
            x = self.act_fn(gate) * up
        return self.down_proj(x)
