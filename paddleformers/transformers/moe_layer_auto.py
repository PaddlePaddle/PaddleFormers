# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) Microsoft Corporation.
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# Copyright (C) 2024 THL A29 Limited, a Tencent company.  All rights reserved.
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
from __future__ import annotations

import copy

import paddle
import paddle.distributed as dist
import paddle.nn.functional as F

try:
    from paddle.distributed.auto_parallel.local_layer import LocalLayer
except:

    class LocalLayer(object):
        """
        A dummy class for LocalLayer, used when the actual class
        cannot be imported.
        """

        pass


from paddle import nn

from .auto_utils import einsum, get_mesh
from .moe_gate_auto import PretrainedMoEGate


def dispatching(x, dispatch_mask, scatter_index, num_experts, capacity):
    """
    Rearranges the input tensor `x` based on gate results, truncates it according to the specified capacity, and performs padding.

    Args:
        x (Tensor)[Seq, Dim]: The input tensor.
        dispatch_mask (List[Tensor[Seq, 1], Tensor[Seq, 1]]): A list of dispatch masks.
        scatter_index (Union[List[Tensor[Seq,], Tensor[Seq]], Tensor[Seq, 2]]): A list or tensor representing scatter indices.
        num_experts (int): The number of experts.
        capacity (int): The capacity size.

    Returns:
        Tensor [Expert*Capacity, Dim]: The output tensor after dispatching.
    """
    output = None
    orig_dtype = x.dtype
    if isinstance(scatter_index, paddle.Tensor):
        scatter_index = scatter_index.unbind(1)
    for i_scatter_index, i_dispatch_mask in zip(scatter_index, dispatch_mask):
        init_output = paddle.zeros([num_experts * capacity, x.shape[-1]], dtype="float32")
        updates = x * i_dispatch_mask.cast(x.dtype)
        if output is None:
            output = paddle.scatter(
                init_output,
                i_scatter_index,
                updates,
                overwrite=False,
            )
        else:
            output = output + paddle.scatter(
                init_output,
                i_scatter_index,
                updates,
                overwrite=False,
            )
        if output.dtype != orig_dtype:
            output = output.cast(orig_dtype)
    return output


def combining(x, combine_weights, scatter_index):
    """
    Performs combination and aggregation operations on the input matrix.

    Args:
        x: Tensor[num_experts * capacity, dim] - The input matrix to be processed, where the last dimension represents the number of features.
        combine_weights: Union[List[Tensor[seq, 1], Tensor[seq, 1]], Tensor[seq, 2, 1]] - A list or tensor containing combination weights for each feature.
        scatter_index: Union[List[Tensor[seq], Tensor[seq]], Tensor[seq, 2]] - A tuple of indices indicating which elements are to be aggregated, where the first element is the row index and the second element is the column index.

    Returns:
        Tensor: The output matrix after combination and aggregation, with a shape of [n, dim * num_features], where n is the number of samples in the input matrix.
    """

    dim = x.shape[-1]
    if isinstance(scatter_index, (list, tuple)):
        scatter_index = paddle.concat([i.unsqueeze([-1]) for i in scatter_index], -1)
    scatter_index = scatter_index.reshape([-1])
    num_k = len(combine_weights) if isinstance(combine_weights, (list, tuple)) else combine_weights.shape[-1]
    x = paddle.gather(x, scatter_index).reshape([-1, num_k, dim])  # [seq,2,dim]
    if isinstance(combine_weights, (list, tuple)):
        combine_weights = paddle.concat(combine_weights, -1).unsqueeze([1])
    return paddle.matmul(combine_weights, x).squeeze(1)  # [seq,1,2] @ [seq,2,dim] -> [seq,1,dim]


class LocalGatePart1(LocalLayer):
    def __init__(self, config, gate: PretrainedMoEGate, ipp=None):
        mesh = get_mesh(ipp)
        out_dist_attrs = [
            (mesh, [dist.Shard(0)]),  # reshaped_input [b*s, h]
            (mesh, [dist.Shard(0)]),  # scores [b*s, e]
            (mesh, [dist.Partial(dist.ReduceType.kRedMax)]),  # expert_counts [e]
            (mesh, [dist.Partial(dist.ReduceType.kRedAvg)]),  # l_aux, scalar
            (mesh, [dist.Partial(dist.ReduceType.kRedAvg)]),  # l_zloss, scalar
        ]
        grad_dist_attrs = [
            None,
            (mesh, [dist.Partial(dist.ReduceType.kRedAvg)]),  # gate_weights.grad
            (mesh, [dist.Partial(dist.ReduceType.kRedAvg)]),  # e_score_correction_bias.grad
        ]
        super().__init__(out_dist_attrs, grad_dist_attrs)
        self.config = config
        self.gate = gate

    def forward(self, hidden_state, gate_weight, e_score_correction_bias, used_token=None):
        # Implement Algorithm 2 from GShard paper.
        batch_size, seq_len, d_model = hidden_state.shape
        reshaped_input = hidden_state.reshape([-1, d_model])

        # compute gating score
        logits = F.linear(hidden_state, gate_weight, None)
        with paddle.amp.auto_cast(False):
            scores = self.gate.gate_score_func(logits=logits)
            scores = scores.cast(paddle.get_default_dtype())

        exp_counts, l_aux, l_zloss = self.gate.topkgating_part1(scores, e_score_correction_bias)

        reshaped_scores = scores.reshape([-1, scores.shape[-1]])
        return reshaped_input, reshaped_scores, exp_counts, l_aux, l_zloss


class LocalGateAndDispatch(LocalLayer):
    def __init__(self, gate: PretrainedMoEGate, ipp=None):
        mesh = get_mesh(ipp)
        out_dist_attrs = [
            (mesh, [dist.Shard(1)]),  # dispatched_input [e,c,h]
            (mesh, [dist.Shard(0)]),  # combine_weights [s,e,c]
        ]
        grad_dist_attrs = [
            None,
            None,
        ]
        super().__init__(out_dist_attrs, grad_dist_attrs)
        self.gate = gate

    def forward(self, reshaped_input, scores):
        combine_weights, dispatch_mask = self.gate.topkgating_part2(scores)
        dispatched_input = einsum("sec,sm->ecm", paddle.cast(dispatch_mask, reshaped_input.dtype), reshaped_input)
        return dispatched_input, combine_weights


class LocalCombine(LocalLayer):
    def __init__(self, ipp=None):
        self.mesh = get_mesh(ipp)
        out_dist_attrs = [(self.mesh, [dist.Shard(0)])]
        grad_dist_attrs = [None, None]
        super().__init__(out_dist_attrs, grad_dist_attrs)

    def forward(self, combine_weights, expert_output, dtype="float32", out_shape=None):
        combined_output = einsum("sec,ecm->sm", combine_weights.cast(dtype), expert_output)
        if out_shape is not None:
            if dist.get_rank() in self.mesh.process_ids:
                out_shape = dist.auto_parallel.moe_utils._cal_local_shape(
                    out_shape, self.out_dist_attrs[0][0], self.out_dist_attrs[0][1]
                )
            combined_output = combined_output.reshape(out_shape)
        return combined_output


class MoELayer(nn.Layer):
    def __init__(
        self,
        config,
        moe_num_experts: int,
        expert_class: nn.Layer,
        expert_kwargs: dict,
        gate: PretrainedMoEGate,
        capacity: int = 1.0,
        moe_group: str = "data",
        all_to_all_dropout=0.0,
        ipp: int = None,
    ):
        super().__init__()

        self.config = config

        self.moe_num_experts = moe_num_experts
        self.capacity = capacity
        self.ipp = ipp

        self.all_to_all_dropout = all_to_all_dropout
        self.enable_recompute = False

        self.experts = nn.LayerList([])
        for i in range(self.moe_num_experts):
            self.experts.append(expert_class(**expert_kwargs))

        self.expert_parallel_degree, self.moe_num_experts_per_device = self._parse_moe_expert_parallel(
            self.moe_num_experts, config
        )
        self._redistribute_experts(self.experts, config.moe_group)

        self.moe_group = None
        self.gate = gate
        self.gate.group = self.moe_group
        self.is_dummy_moe = True
        self._post_init()

        self.local_gate_part1 = LocalGatePart1(config, gate, ipp)
        self.local_gate_and_dispatch = LocalGateAndDispatch(gate, ipp)
        self.local_combine = LocalCombine(ipp)

    def _redistribute_experts(self, experts, moe_group: str):
        if moe_group != "None":
            index = 0 if moe_group == "dp" else 1
            self.moe_mesh_dim = index
            ep_sub_meshes = dist.auto_parallel.api.split_mesh(get_mesh(self.ipp), index)
            for i, expert in enumerate(experts):
                ep_group_id = i // self.moe_num_experts_per_device
                experts[i].redistribute_expert(ep_sub_meshes[ep_group_id], [dist.Replicate(), dist.Replicate()])

    def _parse_moe_expert_parallel(self, moe_num_experts, config):
        assert config.moe_group in ["dp", "mp", "None"], f"moe_group={config.moe_group} not in ['dp', 'mp', 'None']"
        if config.moe_group == "None":
            expert_parallel_degree = 1
        else:
            expert_parallel_degree = dist.fleet.auto.get_mesh().get_dim_size(config.moe_group)
        assert (
            moe_num_experts >= expert_parallel_degree
        ), f"expert moe_num_experts={moe_num_experts} >= moe_world_size={expert_parallel_degree}"

        assert (
            moe_num_experts % expert_parallel_degree == 0
        ), f"expert moe_num_experts={moe_num_experts} % moe_world_size={expert_parallel_degree} == 0"
        moe_num_experts_per_device = moe_num_experts // expert_parallel_degree

        return expert_parallel_degree, moe_num_experts_per_device

    def _post_init(self):
        for p in self.gate.parameters():
            p.is_gate = True

        for k in self.experts:
            if k is not None:
                for p in k.parameters():
                    p.expert = not self.is_dummy_moe
                    p.no_sync = not self.is_dummy_moe
                    # logger.info(f"expert param={p.name}, no-sync={p.no_sync}")

    def expert_forward(self, dispatched_input):
        sub_mesh_tensors = dist.auto_parallel.api.moe_sub_mesh_tensors(
            dispatched_input, get_mesh(self.ipp), self.moe_mesh_dim, dispatched_input.placements
        )
        chunks = paddle.utils.flatten([t.unbind(1) for t in sub_mesh_tensors])

        # try to simplify the code below
        ep_group_outputs = []
        expert_outputs = []
        for i, (chunk, expert) in enumerate(zip(chunks, self.experts)):
            chunk = chunk.contiguous()
            expert_outputs += [expert(chunk)]
            if (i + 1) % self.moe_num_experts_per_device == 0:
                ep_group_outputs += [paddle.stack(expert_outputs, axis=1)]
                expert_outputs = []

        expert_output = dist.auto_parallel.api.moe_global_mesh_tensor(
            ep_group_outputs, get_mesh(self.ipp), dispatched_input.placements, self.moe_mesh_dim
        )
        return expert_output

    def forward(
        self,
        hidden_state: paddle.Tensor,
        used_token: paddle.Tensor = None,
    ):
        """_summary_

        Args:
            input (_type_): _description_
            used_token

        Returns:
            _type_: _description_
        """
        # Implement Algorithm 2 from GShard paper.
        batch_size, seq_len, d_model = hidden_state.shape

        reshaped_input, gate_scores, exp_counts, l_aux, l_zloss = self.local_gate_part1(
            hidden_state, self.gate.weight, self.gate.e_score_correction_bias, used_token=used_token
        )
        if self.gate.drop_tokens is False:
            capacity = paddle.max(exp_counts)
            capacity = dist.reshard(capacity, get_mesh(), [dist.Replicate()])
            self.gate.capacity = int(capacity)
        dispatched_input, combine_weights = self.local_gate_and_dispatch(reshaped_input, gate_scores)
        ori_dispatched_placements = copy.deepcopy(dispatched_input.placements)

        ep_placements = copy.deepcopy(dispatched_input.placements)
        ep_placements[self.moe_mesh_dim] = dist.Shard(0)
        dispatched_input = dist.reshard(dispatched_input, get_mesh(self.ipp), ep_placements)

        # Re-shape after all-to-all: ecm -> gecm
        dispatched_input = dispatched_input.reshape(
            [self.expert_parallel_degree, self.moe_num_experts_per_device, -1, d_model]
        )
        expert_output = self.expert_forward(dispatched_input)
        # Re-shape before drop_tokens: gecm -> ecm
        expert_output = expert_output.reshape(
            [self.expert_parallel_degree * self.moe_num_experts_per_device, -1, d_model]
        )
        expert_output = dist.reshard(expert_output, get_mesh(self.ipp), ori_dispatched_placements)

        combined_output = self.local_combine(
            combine_weights, expert_output, dtype=hidden_state[0].dtype, out_shape=hidden_state.shape
        )

        return combined_output, l_aux, l_zloss
