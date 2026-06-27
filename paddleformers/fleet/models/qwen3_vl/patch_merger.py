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
from dataclasses import dataclass

from paddle import nn
from paddle.distributed.fleet.meta_parallel import LayerSpec, build_spec_layer
from paddle.nn import functional as F

from ...tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from ...transformer.identity_op import IdentityOp
from ...transformer.mlp import MLP, MLPSublayersSpec


@dataclass
class Qwen3VLVisionPatchMergerSpec:
    norm: LayerSpec = IdentityOp


class Qwen3VLVisionPathMerger(nn.Module):
    def __init__(
        self,
        config,
        sublayers_spec: Qwen3VLVisionPatchMergerSpec,
        dim: int | None = None,
        context_dim: int | None = None,
        use_postshuffle_norm: bool = False,
    ):
        super().__init__()
        context_dim = (
            context_dim if context_dim is not None else config.hidden_size
        )
        dim = dim if dim is not None else config.out_hidden_size

        self.hidden_size = context_dim * (config.spatial_merge_size**2)
        norm_dim = self.hidden_size if use_postshuffle_norm else context_dim
        self.norm = build_spec_layer(
            sublayers_spec.norm, config=config, hidden_size=norm_dim
        )
        self.use_postshuffle_norm = use_postshuffle_norm
        self.mlp = build_spec_layer(
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
                    "hidden_size": dim,
                },
            )
        )

    def forward(self, x):
        if isinstance(x, dict):
            x = x["hidden_states"].squeeze(0)
        if self.use_postshuffle_norm:
            x = self.norm(x.reshape([-1, self.hidden_size]))
            x = x.reshape([-1, self.hidden_size])
        else:
            x = self.norm(x)
            x = x.reshape([-1, self.hidden_size])

        x, output_bias = self.mlp(x)
        if output_bias is not None:
            x += output_bias
        return x, None
