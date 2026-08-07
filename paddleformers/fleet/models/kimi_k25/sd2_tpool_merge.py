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
from collections import OrderedDict
from dataclasses import dataclass

from paddle import nn
from paddle.distributed.fleet.meta_parallel import (
    LayerSpec,
    build_spec_layer,
)
from paddle.nn import functional as F

from ...tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from ...transformer.identity_op import IdentityOp
from ...transformer.mlp import MLP, MLPSublayersSpec


class KimiK25VisionSd2TpoolMerger(nn.Layer):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.merge_kernel_size = config.merge_kernel_size

    def forward(self, dict_args: dict):
        hidden_states = dict_args["hidden_states"]
        grid_thws = dict_args["grid_thws"]
        d_model = hidden_states.size(-1)
        hidden_states = hidden_states.view(hidden_states.shape[1:])
        outputs = []
        pre_sum = 0
        for t, h, w in grid_thws.tolist():
            # Get the current sequence
            seq = hidden_states[pre_sum : pre_sum + t * h * w]
            # Reshape along self.merge_kernel_size and concat to the last dimension
            kernel_height, kernel_width = self.merge_kernel_size
            new_height, new_width = h // kernel_height, w // kernel_width
            reshaped_seq = seq.view(
                t, new_height, kernel_height, new_width, kernel_width, d_model
            )
            reshaped_seq = (
                reshaped_seq.permute(0, 1, 3, 2, 4, 5).contiguous().mean(dim=0)
            )  # temporal pooling
            padded_seq = reshaped_seq.view(
                new_height * new_width, kernel_height * kernel_width, -1
            )
            outputs.append(padded_seq)
            pre_sum += t * h * w

        rst = OrderedDict()
        rst = {"hidden_states": outputs}
        rst = {**dict_args, **rst}
        return rst


@dataclass
class KimiK25VisionPatchMergerSpec:
    norm: LayerSpec = IdentityOp


class KimiK25VisionPathMerger(nn.Layer):
    def __init__(
        self,
        config,
        sublayers_spec: KimiK25VisionPatchMergerSpec,
    ):
        super().__init__()
        eps = config.projector_ln_eps
        self.hidden_size = config.mm_hidden_size * (
            config.merge_kernel_size[0] * config.merge_kernel_size[1]
        )
        self.pre_norm = build_spec_layer(
            sublayers_spec.norm,
            config=config,
            hidden_size=config.mm_hidden_size,
            eps=eps,
        )

        self.proj = build_spec_layer(
            LayerSpec(
                layer=MLP,
                sublayers_spec=MLPSublayersSpec(
                    up_gate_proj=ColumnParallelLinear,
                    down_proj=RowParallelLinear,
                    hidden_act=F.gelu,
                ),
                extra_kwargs={
                    "config": config,
                    "input_size": self.hidden_size,
                    "intermediate_size": self.hidden_size,
                    "hidden_size": config.text_hidden_size,
                },
            )
        )

    def forward(self, dict_args: dict):
        x = dict_args["hidden_states"]
        if isinstance(x, (list, tuple)):
            # fleet mlp return two tensor, out, bias
            x = [
                self.proj(self.pre_norm(item).view(item.shape[0], -1))[0]
                for item in x
            ]
        else:
            # B, N, N_k, C = x.shape
            B = x.shape[0]
            x = self.proj(self.pre_norm(x).view(B, -1, self.hidden_size))

        rst = OrderedDict()
        rst = {"hidden_states": x}
        rst = {**dict_args, **rst}
        return rst
