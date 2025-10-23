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
import paddle.distributed as dist
from paddle.distributed.auto_parallel.intermediate.tensor_parallel import (
    PrepareLayerInput,
    PrepareLayerOutput,
)


def layer_input_parallel_row_hook(process_mesh):
    def hook(layer, inputs, output=None):
        res_inputs = []
        for input in inputs:
            if not input.is_dist():
                x = dist.shard_tensor(input, process_mesh, [dist.Shard(0), dist.Replicate(), dist.Replicate()])
                res_inputs.append(dist.reshard(x, process_mesh, [dist.Shard(0), dist.Replicate(), dist.Replicate()]))
            else:
                res_inputs.append(
                    dist.reshard(input, process_mesh, [dist.Shard(0), dist.Replicate(), dist.Replicate()])
                )
        return tuple(res_inputs)

    return hook


def layer_input_parallel_row_and_col_hook(process_mesh):
    def hook(layer, inputs, output=None):
        res_inputs = []
        for input in inputs:
            if not input.is_dist():
                x = dist.shard_tensor(input, process_mesh, [dist.Shard(0), dist.Replicate(), dist.Shard(1)])
                res_inputs.append(dist.reshard(x, process_mesh, [dist.Shard(0), dist.Replicate(), dist.Shard(1)]))
            else:
                res_inputs.append(dist.reshard(input, process_mesh, [dist.Shard(0), dist.Replicate(), dist.Shard(1)]))
        return tuple(res_inputs)

    return hook


def layer_input_replicate_hook(process_mesh):
    def hook(layer, inputs, output=None):
        res_inputs = []
        for input in inputs:
            if not input.is_dist():
                x = dist.shard_tensor(input, process_mesh, [dist.Replicate(), dist.Replicate(), dist.Replicate()])
                res_inputs.append(
                    dist.reshard(x, process_mesh, [dist.Replicate(), dist.Replicate(), dist.Replicate()])
                )
            else:
                res_inputs.append(dist.reshard(input, process_mesh, [dist.Replicate(), dist.Replicate()]))
        return tuple(res_inputs)

    return hook


def layer_input_rope_hook(process_mesh):
    def hook(layer, inputs, output=None):
        res_inputs = []
        batch_size = None
        seq_length = None
        process_mesh = None
        placements = None
        for index in range(len(inputs)):
            if index == 0:
                batch_size, seq_length, _, _ = inputs[index]._local_shape
                process_mesh = inputs[index].process_mesh
                placements = inputs[index].placements
            # process position_ids
            if index == len(inputs) - 1:
                mesh = dist.auto_parallel.get_mesh()
                assert "sep" in mesh.dim_names, f"mesh.dim_names:{mesh.dim_names} must contain sep"
                group = mesh._get_group("sep")
                chunk_size = seq_length // 2
                chunk_num = group.nranks * 2
                rank = group.rank
                first_chunk_ids = paddle.arange(rank * chunk_size, (rank + 1) * chunk_size, dtype="int64")
                second_chunk_ids = paddle.arange(
                    (chunk_num - rank - 1) * chunk_size, (chunk_num - rank) * chunk_size, dtype="int64"
                )
                position_ids = paddle.concat([first_chunk_ids, second_chunk_ids]).expand((batch_size, seq_length))
                mp_axis = process_mesh.dim_names.index("mp")
                placements[mp_axis] = dist.Replicate()  # mp placament shard(2) -> replicate
                position_ids = dist.auto_parallel.api.dtensor_from_local(position_ids, process_mesh, placements)
                res_inputs.append(position_ids)
            else:
                res_inputs.append(inputs[index])
        return tuple(res_inputs)

    return hook


def layer_output_rope_hook(process_mesh):
    def hook(layer, inputs, outputs):
        res_outputs = []
        for output in outputs:
            process_mesh = output.process_mesh
            placements = output.placements
            cp_index = process_mesh.dim_names.index("sep")  # get the axis for the split
            cp_degree = process_mesh.shape[cp_index]
            assert cp_degree > 1, f"cp_degree:{cp_degree} must > 1"
            placements[cp_index] = dist.Shard(1)  # seq_dim:1
            output = dist.reshard(output, process_mesh, placements)
            res_outputs.append(output)
        return tuple(res_outputs)

    return hook


def get_dist_config(model, prefix=""):
    """Generate distributed configuration for Llama model"""
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
        "cp_config": {
            "parallelize_plan": {
                f"{prefix}llama.layers.*.self_attn.sdpa": dist.ContextParallel(
                    backend="p2p" if model.config.context_parallel_degree > 1 else "all2all"
                ),
            }
        },
    }

    if model.config.context_parallel_degree > 1:
        config["cp_config"]["parallelize_plan"].update(
            {
                f"{prefix}llama.layers.*.self_attn.rope_func": [
                    PrepareLayerInput(layer_input_rope_hook),
                    PrepareLayerOutput(layer_output_rope_hook),
                ]
            }
        )
    elif model.config.sep_parallel_degree > 1:
        # fuse_rope is not support dtensor spmd yet,thus need to extraly reshard sequence dim
        config["cp_config"]["parallelize_plan"].update(
            {
                f"{prefix}llama.layers.*.self_attn.rope_func": PrepareLayerOutput(layer_output_rope_hook),
            }
        )

    return config
