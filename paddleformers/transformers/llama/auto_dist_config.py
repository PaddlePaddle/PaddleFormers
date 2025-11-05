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


import paddle.distributed as dist
from paddle.distributed.auto_parallel.intermediate.tensor_parallel import (
    PrepareLayerInput,
)


def layer_input_parallel_row_hook(process_mesh):
    def hook(layer, inputs, output=None):
        res_inputs = []
        for input in inputs:
            if not input.is_dist():
                x = dist.shard_tensor(input, process_mesh, [dist.Shard(0), dist.Replicate()])
                res_inputs.append(dist.reshard(x, process_mesh, [dist.Shard(0), dist.Replicate()]))
            else:
                res_inputs.append(dist.reshard(input, process_mesh, [dist.Shard(0), dist.Replicate()]))
        return tuple(res_inputs)

    return hook


def layer_input_parallel_row_and_col_hook(process_mesh):
    def hook(layer, inputs, output=None):
        res_inputs = []
        for input in inputs:
            if not input.is_dist():
                x = dist.shard_tensor(input, process_mesh, [dist.Shard(0), dist.Shard(1)])
                res_inputs.append(dist.reshard(x, process_mesh, [dist.Shard(0), dist.Shard(1)]))
            else:
                res_inputs.append(dist.reshard(input, process_mesh, [dist.Shard(0), dist.Shard(1)]))
        return tuple(res_inputs)

    return hook


def layer_input_replicate_hook(process_mesh):
    def hook(layer, inputs, output=None):
        res_inputs = []
        for input in inputs:
            if not input.is_dist():
                x = dist.shard_tensor(input, process_mesh, [dist.Replicate(), dist.Replicate()])
                res_inputs.append(dist.reshard(x, process_mesh, [dist.Replicate(), dist.Replicate()]))
            else:
                res_inputs.append(dist.reshard(input, process_mesh, [dist.Replicate(), dist.Replicate()]))
        return tuple(res_inputs)

    return hook


def auto_dist_config(self, prefix=""):
    if prefix != "":
        assert prefix.endswith(".")
    config = {
        "sp_config": {
            "parallelize_plan": {
                f"{prefix}llama.embed_tokens": [
                    dist.ColWiseParallel(),
                    dist.SequenceParallelBegin(),
                ],
                f"{prefix}llama.reshard_row": PrepareLayerInput(layer_input_parallel_row_hook),
                f"{prefix}llama.reshard_row_and_col": PrepareLayerInput(layer_input_parallel_row_and_col_hook),
                f"{prefix}llama.global_layer.reshard_replicate": PrepareLayerInput(layer_input_replicate_hook),
                f"{prefix}llama.layers.*.self_attn.qkv_proj": dist.ColWiseParallel(),
                f"{prefix}llama.layers.*.self_attn.q_proj": dist.ColWiseParallel(),
                f"{prefix}llama.layers.*.self_attn.k_proj": dist.ColWiseParallel(),
                f"{prefix}llama.layers.*.self_attn.v_proj": dist.ColWiseParallel(),
                f"{prefix}llama.layers.*.self_attn.o_proj": dist.RowWiseParallel(),
                f"{prefix}llama.layers.*.self_attn": dist.SequenceParallelDisable(),
                f"{prefix}llama.layers.*.mlp.gate_proj": dist.ColWiseParallel(),
                f"{prefix}llama.layers.*.mlp.up_proj": dist.ColWiseParallel(),
                f"{prefix}llama.layers.*.mlp.gate_up_fused_proj": dist.ColWiseParallel(),
                f"{prefix}llama.layers.*.mlp.down_proj": dist.RowWiseParallel(),
                f"{prefix}llama.layers.*.mlp": dist.SequenceParallelDisable(need_transpose=False),
                f"{prefix}lm_head.weight": dist.ColWiseParallel(),
                f"{prefix}lm_head": dist.SequenceParallelEnd(),
            }
        },
        "mp_config": {
            "parallelize_plan": {
                f"{prefix}llama.embed_tokens": dist.ColWiseParallel(gather_output=True),
                f"{prefix}llama.reshard_row": PrepareLayerInput(layer_input_parallel_row_hook),
                f"{prefix}llama.reshard_row_and_col": PrepareLayerInput(layer_input_parallel_row_and_col_hook),
                f"{prefix}llama.global_layer.reshard_replicate": PrepareLayerInput(layer_input_replicate_hook),
                f"{prefix}llama.layers.*.self_attn.qkv_proj": dist.ColWiseParallel(),
                f"{prefix}llama.layers.*.self_attn.q_proj": dist.ColWiseParallel(),
                f"{prefix}llama.layers.*.self_attn.k_proj": dist.ColWiseParallel(),
                f"{prefix}llama.layers.*.self_attn.v_proj": dist.ColWiseParallel(),
                f"{prefix}llama.layers.*.self_attn.o_proj": dist.RowWiseParallel(),
                f"{prefix}llama.layers.*.mlp.gate_proj": dist.ColWiseParallel(),
                f"{prefix}llama.layers.*.mlp.up_proj": dist.ColWiseParallel(),
                f"{prefix}llama.layers.*.mlp.gate_up_fused_proj": dist.ColWiseParallel(),
                f"{prefix}llama.layers.*.mlp.down_proj": dist.RowWiseParallel(),
                f"{prefix}lm_head.weight": dist.ColWiseParallel(),
            }
        },
        "pp_config": {"split_spec": f"{prefix}llama.layers", "global_spec": f"{prefix}llama.global_layer"},
    }

    return config
