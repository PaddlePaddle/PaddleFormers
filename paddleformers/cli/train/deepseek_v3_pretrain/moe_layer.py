# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import paddle
import paddle.distributed as dist
from paddle import nn

from paddleformers.transformers import _AllToAll
from paddleformers.transformers.fp8_utils import (
    FP8GroupGemmMlpFunctionNode,
    extract_first_if_tuple,
)
from paddleformers.transformers.fused_a2a import (
    CombineNode,
    DispatchNode,
    get_buffer,
    get_hidden_bytes,
)
from paddleformers.transformers.moe_utils import offload, reload
from paddleformers.utils.log import logger

from .moe_gate import PretrainedMoEGate
from .moe_utils import (
    UnZipNode,
    ZipNode,
    merge_subbatch_cast,
    tokens_zip_unique_add_with_subbatch,
)
from .token_dispatcher import MoEFlexTokenDispatcherFast as MoEFlexTokenDispatcher
from .token_dispatcher import PreDispatchNode

try:
    import paddle.distributed.communication.deep_ep as deep_ep
except ImportError:
    deep_ep = None

try:
    import paddlefleet as paddlefleet
except ImportError:
    paddlefleet = None


def record_stream_for_multi_input(x):
    if isinstance(x, (tuple, list)):
        for i in range(len(x)):
            x[i]._record_stream()
    else:
        x._record_stream()


def stop_gradient_for_multi_input(x):
    if isinstance(x, (tuple, list)):
        x[0].stop_gradient = False
    else:
        x.stop_gradient = False


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
        using_post_norm_recompute=False,
    ):
        super().__init__()

        self.config = config

        self.moe_num_experts = moe_num_experts
        self.capacity = capacity

        try:
            dist.fleet.get_hybrid_communicate_group()
            is_fleet_init = True
        except AttributeError:
            is_fleet_init = False

        if is_fleet_init and dist.get_world_size() > 1:
            if moe_group == "data":
                self.moe_group = dist.fleet.get_hybrid_communicate_group().get_data_parallel_group()
            elif moe_group == "expert":
                self.moe_group = dist.fleet.get_hybrid_communicate_group().expert_parallel_group
            self.moe_rank = dist.get_rank(self.moe_group)
            self.moe_rank = 0 if self.moe_rank < 0 else self.moe_rank
            self.expert_model_parallel_size = dist.get_world_size(self.moe_group)
            self.expert_model_parallel_size = (
                1 if self.expert_model_parallel_size < 0 else self.expert_model_parallel_size
            )
            self.moe_num_experts_per_device = self._parse_moe_expert_parallel(
                self.moe_num_experts, self.expert_model_parallel_size
            )
            self.is_dummy_moe = False if self.expert_model_parallel_size > 1 else True
        else:
            # when moe_group is dummy, we don't need to use all_to_all
            self.moe_group = None
            self.moe_rank = 0
            self.expert_model_parallel_size = 1
            self.moe_num_experts_per_device = self.moe_num_experts
            self.is_dummy_moe = True

        self.all_to_all_dropout = all_to_all_dropout
        self.enable_recompute = False

        self.experts = nn.LayerList([])
        for i in range(self.moe_num_experts):
            if i // self.moe_num_experts_per_device == self.moe_rank:
                self.experts.append(expert_class(**expert_kwargs))
            else:
                self.experts.append(None)

        self.gate = gate
        self.gate.group = self.moe_group
        # for flex token moe layer
        self.router = gate
        self.ep_size = dist.get_world_size(self.moe_group)
        self.moe_router_topk = gate.top_k
        self.num_local_experts = moe_num_experts // self.ep_size
        if self.moe_group is not None:
            self.token_dispatcher = MoEFlexTokenDispatcher(
                self.num_local_experts, self.moe_router_topk, self.moe_num_experts, self.moe_group
            )
        self.token_drop_steps = config.token_drop_steps if hasattr(config, "token_drop_steps") else None
        self.using_flex_token = False

        self.using_post_norm_recompute = using_post_norm_recompute
        self._post_init()

    def update_flex_token(self):
        from .modeling import get_global_step

        if (
            (not hasattr(self.config, "using_flex_token"))
            or (not self.config.using_flex_token)
            or (get_global_step() < self.token_drop_steps)
        ):
            self.using_flex_token = False
            self.router.using_flex_token = False
        else:
            if not self.using_flex_token:
                logger.info("Changing to flex token moe mode")
            self.using_flex_token = True
            self.router.using_flex_token = True

    def _parse_moe_expert_parallel(self, moe_num_experts, expert_model_parallel_size):
        assert (
            moe_num_experts >= expert_model_parallel_size
        ), f"expert moe_num_experts={moe_num_experts} >= moe_world_size={expert_model_parallel_size}"
        assert (
            moe_num_experts % expert_model_parallel_size == 0
        ), f"expert moe_num_experts={moe_num_experts} % moe_world_size={expert_model_parallel_size} == 0"
        moe_num_experts_per_device = moe_num_experts // expert_model_parallel_size
        return moe_num_experts_per_device

    def _post_init(self):
        for p in self.gate.parameters():
            p.is_gate = True

        for k in self.experts:
            if k is not None:
                for p in k.parameters():
                    p.expert = not self.is_dummy_moe
                    p.no_sync = not self.is_dummy_moe
                    # logger.info(f"expert param={p.name}, no-sync={p.no_sync}")

    def forward(
        self,
        hidden_states: paddle.Tensor,
        probs=None,
        routing_map=None,
        capacity=None,
        topk_weight=None,
        topk_ids=None,
        token_priority=None,
        l_aux=None,
        l_zloss=None,
    ):
        self.update_flex_token()

        if self.using_flex_token:
            return self.forward_flex_token(hidden_states, probs, routing_map, l_aux, l_zloss)
        else:
            return self.forward_drop_token(
                hidden_states, capacity, topk_weight, topk_ids, token_priority, l_aux, l_zloss
            )

    def forward_drop_token(
        self,
        hidden_state: paddle.Tensor,
        capacity=None,
        topk_weight=None,
        topk_ids=None,
        token_priority=None,
        l_aux=None,
        l_zloss=None,
    ):
        """MoE Layer forward function
            1. Gate Forward.
            2. Dispatch export.
            3. Experts Forward.

        Args:
            hidden_state: MoE Layer input

        Returns:
            final_out: MoE Layer main output.
            l_aux: MoE auxiliary loss.  l_zloss: MoE z loss."""
        batch_size, seq_len, d_model = hidden_state.shape

        reshaped_input = hidden_state.reshape([-1, d_model])

        # self.l_aux       :
        # topk_weight  : se
        # topk_ids    : sk
        # token_priority    : se
        # self.exp_counts  :
        if self.using_post_norm_recompute:
            assert (
                capacity is not None
                and topk_weight is not None
                and topk_ids is not None
                and token_priority is not None
                and l_aux is not None
                and l_zloss is not None
            )
        else:
            capacity, topk_weight, topk_ids, token_priority, l_aux, l_zloss = self.gate(hidden_state)

        """MoE expert dispatch from: https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/modeling_deepseek.py"""
        cnts = paddle.zeros([topk_ids.shape[0], len(self.experts)], dtype=topk_ids.dtype)
        cnts = cnts.put_along_axis(topk_ids, 1, axis=1)

        tokens_per_expert = cnts.sum(axis=0)
        idxs = topk_ids.reshape([topk_ids.shape[0] * topk_ids.shape[1]]).argsort()
        sorted_tokens = reshaped_input[idxs // topk_ids.shape[1]]
        tokens_per_expert = tokens_per_expert.detach()
        sorted_tokens_shape = sorted_tokens.shape

        if self.expert_model_parallel_size > 1:
            tokens_per_ep_rank = tokens_per_expert.reshape([self.expert_model_parallel_size, -1]).sum(axis=1)
            tokens_per_expert_group = _AllToAll.apply(
                [tokens_per_expert.shape[0]], tokens_per_expert, group=self.moe_group
            )
            output_splits = (
                tokens_per_expert_group.reshape([self.expert_model_parallel_size, -1]).sum(axis=1).cpu().tolist()
            )
            input_split_sizes = tokens_per_ep_rank.cpu().tolist()
            gathered_tokens = _AllToAll.apply(
                [tokens_per_expert_group.sum(axis=0).cpu().item(), sorted_tokens.shape[1]],
                sorted_tokens,
                out_split_sizes=output_splits,
                in_split_sizes=input_split_sizes,
                group=self.moe_group,
            )

            tokens_per_expert_post_gather = tokens_per_expert_group.reshape(
                [self.expert_model_parallel_size, self.moe_num_experts_per_device]
            ).sum(axis=0)
            gatherd_idxs = np.zeros(shape=(gathered_tokens.shape[0],), dtype=np.int32)
            s = 0
            for i, k in enumerate(tokens_per_expert_group.cpu().numpy()):
                gatherd_idxs[s : s + k] = i % self.moe_num_experts_per_device
                s += k
            gatherd_idxs = gatherd_idxs.argsort()
            sorted_tokens = gathered_tokens[gatherd_idxs]
            tokens_per_expert = tokens_per_expert_post_gather

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i + self.moe_rank * self.moe_num_experts_per_device]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx
        outs = paddle.cat(outputs, axis=0) if len(outputs) > 0 else paddle.to_tensor(0, dtype=sorted_tokens.dtype)
        if self.expert_model_parallel_size > 1:
            new_x = paddle.empty_like(outs)
            new_x[gatherd_idxs] = outs
            gathered_tokens = _AllToAll.apply(
                sorted_tokens_shape,
                new_x,
                out_split_sizes=input_split_sizes,
                in_split_sizes=output_splits,
                group=self.moe_group,
            )
            outs = gathered_tokens

        new_x = paddle.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.reshape(topk_ids.shape + [-1])
            .astype(topk_weight.dtype)
            .multiply_(topk_weight.unsqueeze(-1))
            .multiply_(token_priority.unsqueeze(-1))
            .sum(axis=1)
            .astype(new_x.dtype)
            .reshape([batch_size, seq_len, -1])
        )

        return final_out, l_aux, l_zloss

    def expert_forward(self, dispatched_input):
        outputs = []
        chunks = paddle.split(dispatched_input, num_or_sections=self.get_tokens_per_expert(), axis=0)
        for i, chunk in enumerate(chunks):
            chunk = chunk.contiguous()
            # assert chunk.shape[0] != 0, "Cannot dispatch empty input"
            expert = self.experts[i + self.moe_rank * self.moe_num_experts_per_device]
            outputs += [expert(chunk)]

        return paddle.concat(outputs, axis=0)

    def forward_flex_token(self, hidden_states: paddle.Tensor, probs=None, routing_map=None, l_aux=None, l_zloss=None):
        _, _, d_model = hidden_states.shape
        # reshaped_input = hidden_states.reshape([-1, d_model])
        if self.using_post_norm_recompute:
            assert probs is not None and routing_map is not None and l_aux is not None and l_zloss is not None
        else:
            probs, routing_map, l_aux, l_zloss = self.router(hidden_states)
        if hasattr(self.config, "dsv3_use_fp8_gemm") and self.config.dsv3_use_fp8_gemm:
            if hasattr(self.config, "dsv3_use_fp8_dispatch") and self.config.dsv3_use_fp8_dispatch:
                output = FusionMoe.apply(
                    hidden_states,
                    probs,
                    routing_map,
                    self,
                    recompute_fwd_gate_up=self.config.recompute_fwd_gate_up,
                    is_split_group_gemm=self.config.is_split_group_gemm,
                )
            else:
                hidden_states, token_indices, token_probs = self.token_dispatcher.pre_dispatch(
                    hidden_states, probs, routing_map
                )
                output = FusionMoe.apply(
                    hidden_states,
                    token_indices,
                    token_probs,
                    self,
                    recompute_fwd_gate_up=self.config.recompute_fwd_gate_up,
                    is_split_group_gemm=self.config.is_split_group_gemm,
                )
        else:
            (
                dispatched_input,
                token_permuted_indices,
                prob_permuted_indices,
                dispatched_probs,
            ) = self.token_dispatcher.token_permutation(hidden_states, probs, routing_map)

            expert_output = self.expert_forward(dispatched_input)
            output, _ = self.token_dispatcher.token_unpermutation(
                expert_output, token_permuted_indices, prob_permuted_indices, dispatched_probs, None
            )
        return output, l_aux, l_zloss

    def get_tokens_per_expert(self):
        return self.token_dispatcher._comm_manager.tokens_per_expert_list

    def set_tokens_per_expert(self, tokens_per_expert_list):
        self.token_dispatcher._comm_manager.tokens_per_expert_list = tokens_per_expert_list

    def pre_dispatch_compute(self, hidden_states):
        _, _, d_model = hidden_states.shape
        probs, routing_map, l_aux, l_zloss = self.router(hidden_states)
        hidden_states, token_indices, token_probs = self.token_dispatcher.pre_dispatch(
            hidden_states, probs, routing_map
        )
        return l_aux, l_zloss, hidden_states, token_indices, token_probs

    def post_dispatch_compute(self, hidden_states, dispatched_indices, dispatched_probs):
        (global_input_tokens, token_permuted_indices, prob_permuted_indices) = self.token_dispatcher.post_dispatch(
            hidden_states, dispatched_indices
        )
        return (global_input_tokens, token_permuted_indices, prob_permuted_indices)

    def pre_combine_compute(self, hidden_states, token_permuted_indices, prob_permuted_indices, dispatched_probs):
        hidden_states = self.token_dispatcher.pre_combine(
            hidden_states, token_permuted_indices, prob_permuted_indices, dispatched_probs
        )
        return hidden_states

    def post_combine_compute(self, hidden_states):
        hidden_states = self.token_dispatcher.post_combine(hidden_states)
        return hidden_states


class MoEFlexTokenLayer(nn.Layer):
    def __init__(self, config, moe_num_experts, expert_class, expert_kwargs, gate, moe_group):

        super().__init__()
        self.config = config
        self.moe_group = moe_group
        self.ep_size = dist.get_world_size(self.moe_group)
        self.moe_router_topk = gate.top_k
        self.moe_num_experts = moe_num_experts
        self.num_local_experts = moe_num_experts // self.ep_size
        self.token_dispatcher = MoEFlexTokenDispatcher(
            self.num_local_experts, self.moe_router_topk, self.moe_num_experts, moe_group
        )

        self.experts = nn.LayerList([expert_class(**expert_kwargs)] * self.num_local_experts)
        self.router = gate

    def expert_forward(self, dispatched_input, tokens_per_expert):
        outputs = []
        tokens_per_expert = tokens_per_expert.tolist()
        # print(f"all tokens: {sum(tokens_per_expert)}, detail: {tokens_per_expert}")
        chunks = paddle.split(dispatched_input, num_or_sections=tokens_per_expert, axis=0)
        for chunk, expert in zip(chunks, self.experts):
            chunk = chunk.contiguous()
            # assert chunk.shape[0] != 0, "Cannot dispatch empty input"
            outputs += [expert(chunk)]

        return paddle.cat(outputs, axis=0)

    def forward(self, hidden_states: paddle.Tensor):
        _, _, d_model = hidden_states.shape
        # reshaped_input = hidden_states.reshape([-1, d_model])
        probs, routing_map, l_aux, l_zloss = self.router(hidden_states)
        (dispatched_input, tokens_per_expert) = self.token_dispatcher.token_permutation(
            hidden_states, probs, routing_map
        )
        expert_output = self.expert_forward(dispatched_input, tokens_per_expert)
        output, _ = self.token_dispatcher.token_unpermutation(expert_output, None)
        return output, l_aux, l_zloss

    def forward_flex_token(self, hidden_states: paddle.Tensor, probs=None, routing_map=None, l_aux=None, l_zloss=None):
        _, _, d_model = hidden_states.shape
        # reshaped_input = hidden_states.reshape([-1, d_model])
        if self.using_post_norm_recompute:
            assert probs is not None and routing_map is not None and l_aux is not None and l_zloss is not None
        else:
            probs, routing_map, l_aux, l_zloss = self.router(hidden_states)
        if hasattr(self.config, "dsv3_use_fp8_gemm") and self.config.dsv3_use_fp8_gemm:
            if hasattr(self.config, "dsv3_use_fp8_dispatch") and self.config.dsv3_use_fp8_dispatch:
                output = FusionMoe.apply(
                    hidden_states,
                    probs,
                    routing_map,
                    self,
                    recompute_fwd_gate_up=self.config.recompute_fwd_gate_up,
                    is_split_group_gemm=self.config.is_split_group_gemm,
                )
            else:
                hidden_states, token_indices, token_probs = self.token_dispatcher.pre_dispatch(
                    hidden_states, probs, routing_map
                )
                output = FusionMoe.apply(
                    hidden_states,
                    token_indices,
                    token_probs,
                    self,
                    recompute_fwd_gate_up=self.config.recompute_fwd_gate_up,
                    is_split_group_gemm=self.config.is_split_group_gemm,
                )
        else:
            (
                dispatched_input,
                token_permuted_indices,
                prob_permuted_indices,
                dispatched_probs,
            ) = self.token_dispatcher.token_permutation(hidden_states, probs, routing_map)

            expert_output = self.expert_forward(dispatched_input)
            output, _ = self.token_dispatcher.token_unpermutation(
                expert_output, token_permuted_indices, prob_permuted_indices, dispatched_probs, None
            )
        return output, l_aux, l_zloss

    def get_tokens_per_expert(self):
        return self.token_dispatcher._comm_manager.tokens_per_expert_list

    def set_tokens_per_expert(self, tokens_per_expert_list):
        self.token_dispatcher._comm_manager.tokens_per_expert_list = tokens_per_expert_list

    def pre_dispatch_compute(self, hidden_states):
        _, _, d_model = hidden_states.shape
        probs, routing_map, l_aux, l_zloss = self.router(hidden_states)
        hidden_states, token_indices, token_probs = self.token_dispatcher.pre_dispatch(
            hidden_states, probs, routing_map
        )
        return l_aux, l_zloss, hidden_states, token_indices, token_probs

    def post_dispatch_compute(self, hidden_states, dispatched_indices, dispatched_probs):
        (global_input_tokens, token_permuted_indices, prob_permuted_indices) = self.token_dispatcher.post_dispatch(
            hidden_states, dispatched_indices
        )
        return (global_input_tokens, token_permuted_indices, prob_permuted_indices)

    def pre_combine_compute(self, hidden_states, token_permuted_indices, prob_permuted_indices, dispatched_probs):
        hidden_states = self.token_dispatcher.pre_combine(
            hidden_states, token_permuted_indices, prob_permuted_indices, dispatched_probs
        )
        return hidden_states

    def post_combine_compute(self, hidden_states):
        hidden_states = self.token_dispatcher.post_combine(hidden_states)
        return hidden_states


class Fp8DispatchQuantNode:
    def __init__(self, token_dispatcher, dsv3_use_fp8_dispatch, name="fp8_dispatch_quant_node"):
        self.token_dispatcher = token_dispatcher
        self.pre_dispatch_node = PreDispatchNode(token_dispatcher)
        self.name = name
        self.dsv3_use_fp8_dispatch = dsv3_use_fp8_dispatch

    @paddle.no_grad()
    def forward(self, hidden_states, probs, routing_map):
        # reshape
        self.token_dispatcher.hidden_shape = hidden_states.shape
        hs_2d = hidden_states.view([-1, self.token_dispatcher.hidden_shape[-1]])

        if self.dsv3_use_fp8_dispatch:
            # quant
            hs_fp8, hs_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
                hs_2d, output_scale_transpose=False, quant_method="1x128", input_transpose=False
            )

            # pre_dispatch
            token_indices, token_probs = self.pre_dispatch_node.forward(routing_map, probs)

            self.hidden_states_shape = hidden_states.shape
            hs_fp8.stop_gradient = False
            token_probs.stop_gradient = False
            return (hs_fp8, hs_scale), token_indices, token_probs
        else:
            # pre_dispatch
            token_indices, token_probs = self.pre_dispatch_node.forward(routing_map, probs)

            self.hidden_states_shape = hidden_states.shape
            hs_2d.stop_gradient = False
            token_probs.stop_gradient = False
            return hs_2d, token_indices, token_probs

    @paddle.no_grad()
    def backward(self, hs_grad, token_probs_grad):
        # predispatch grad
        probs_grad = self.pre_dispatch_node.backward(token_probs_grad)
        token_probs_grad._record_stream()

        # reshape_grad
        hs_grad = hs_grad.view(self.hidden_states_shape)
        hs_grad._record_stream()

        return hs_grad, probs_grad, None


class Fp8DispatchNode:
    def __init__(self, token_dispatcher, name="fp8_dispatch_node"):
        self.token_dispatcher = token_dispatcher
        self.dispatch_act_node = DispatchNode(token_dispatcher)
        self.name = name

    @paddle.no_grad()
    def forward(
        self,
        hs_2d,
        token_indices,
        token_probs,
        previous_event=None,
        async_finish=False,
        allocate_on_comm_stream=False,
    ):
        # dispatch
        hs_2d_dispatched, dispatched_probs, states = self.dispatch_act_node.forward(
            hs_2d,
            token_indices,
            token_probs,
            self.token_dispatcher._comm_manager.num_experts,
            self.token_dispatcher._comm_manager.group,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        self.token_dispatcher._comm_manager.handle = states["handle"]
        self.token_dispatcher._comm_manager.tokens_per_expert = states["tokens_per_expert"]
        dispatched_indices = states["dispatched_indices"]

        stop_gradient_for_multi_input(hs_2d_dispatched)
        dispatched_probs.stop_gradient = False
        return hs_2d_dispatched, dispatched_indices, dispatched_probs

    @paddle.no_grad()
    def backward(
        self,
        hs_dispatched_grad,
        dispatched_probs_grad,
        previous_event=None,
        async_finish=False,
        allocate_on_comm_stream=False,
    ):
        # dispatch grad
        hs_grad, _, token_probs_grad = self.dispatch_act_node.backward(
            hs_dispatched_grad,
            dispatched_probs_grad,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        return hs_grad, token_probs_grad


class Fp8CombineNode:
    def __init__(self, token_dispatcher, name="fp8_combine_node"):
        self.token_dispatcher = token_dispatcher
        self.combine_node = CombineNode(token_dispatcher)
        self.name = name

    @paddle.no_grad()
    def forward(self, hidden_states_out, previous_event=None, async_finish=False, allocate_on_comm_stream=False):
        # combine
        output_combine = self.combine_node.forward(
            hidden_states_out,
            self.token_dispatcher._comm_manager.group,
            self.token_dispatcher._comm_manager.handle,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        output_combine.stop_gradient = False
        self.token_dispatcher._comm_manager.handle = None
        return output_combine

    @paddle.no_grad()
    def backward(self, output_combine_grad, previous_event=None, async_finish=False, allocate_on_comm_stream=False):
        # combine grad -> fp8
        hidden_states_out_grad = self.combine_node.backward(
            output_combine_grad,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        return hidden_states_out_grad


class Fp8CombineQuantNode:
    def __init__(self, token_dispatcher, dsv3_use_fp8_dispatch, moe_group=None, name="fp8_combine_quant_node"):
        self.token_dispatcher = token_dispatcher
        self.name = name
        self.moe_group = moe_group
        self.dsv3_use_fp8_dispatch = dsv3_use_fp8_dispatch

    @paddle.no_grad()
    def forward(self, output_combine):
        # post combine
        output = output_combine.reshape(self.token_dispatcher.hidden_shape)
        output_combine._record_stream()
        self.output_combine_shape = output_combine.shape
        output.stop_gradient = False
        return output

    @paddle.no_grad()
    def backward(self, output_grad, event_to_wait=None):
        # post combine grad
        if self.dsv3_use_fp8_dispatch:
            if event_to_wait is not None:
                assert self.moe_group is not None
                event_to_wait.comm_stream_wait(self.moe_group.id)
                buffer = get_buffer(self.token_dispatcher._comm_manager.group, get_hidden_bytes(output_grad))
                custom_stream = paddle.device.Stream(stream_base=buffer.runtime.get_comm_stream())
            else:
                custom_stream = paddle.device.current_stream()
            with paddle.device.stream_guard(custom_stream):
                output_combine_grad = paddle.reshape(output_grad, [-1, output_grad.shape[-1]])
                # output_combine_grad quant to fp8
                output_combine_grad_fp8, output_combine_grad_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
                    output_combine_grad, output_scale_transpose=False, quant_method="1x128", input_transpose=False
                )
                output_grad._record_stream()
                quant_event = None
                if event_to_wait is not None:
                    quant_event = deep_ep.get_event_from_custom_stream(custom_stream.stream_base)
            return (output_combine_grad_fp8, output_combine_grad_scale), quant_event
        else:
            output_combine_grad = paddle.reshape(output_grad, [-1, output_grad.shape[-1]])
            return output_combine_grad, None


class FusionMlpNode:
    """
    The FusedMoeLayer class includes operations for unzipping, expert computation, and zipping.
    """

    def __init__(
        self,
        custom_map,
        max_topk,
        recompute_fwd_gate_up=False,
        is_split_group_gemm=True,
        dsv3_use_fp8_dispatch=True,
        mlp_fwd_subbatch_rows=0,
        mlp_bwd_subbatch_rows=0,
        output_subbatch_rows=0,
    ):
        self.token_dispatcher = custom_map.token_dispatcher
        self.experts = custom_map.experts
        self.unzip_node = UnZipNode()
        self.zip_node = ZipNode()
        self.experts_group_gemm_node = FP8GroupGemmMlpFunctionNode(
            custom_map,
            recompute_fwd_gate_up=recompute_fwd_gate_up,
            is_split_group_gemm=is_split_group_gemm,
        )
        self.dsv3_use_fp8_dispatch = dsv3_use_fp8_dispatch

        self.seq_length = custom_map.config.seq_length
        self.num_experts_per_tok = custom_map.config.num_experts_per_tok
        self.adaptive_remained_O1_recompute_ratio = custom_map.config.adaptive_remained_O1_recompute_ratio

        self.recompute_fwd_gate_up = recompute_fwd_gate_up
        self.dispatched_indices = None
        self.dispatched_probs = None
        self.tokens_per_expert = None
        self.padding_token_per_experts = None
        self.router_topk = max_topk
        self.mlp_fwd_subbatch_rows = mlp_fwd_subbatch_rows
        self.mlp_bwd_subbatch_rows = mlp_bwd_subbatch_rows
        self.output_subbatch_rows = output_subbatch_rows

    def set_recompute_fwd_gate_up(self, recompute_fwd_gate_up):
        self.experts_group_gemm_node.recompute_fwd_gate_up = recompute_fwd_gate_up

    def reset_statue(self):
        """
        Reset the state of the FusionMlpNode object.

        Args:
            None.

        Returns:
            None.

        """
        self.dispatched_indices = None
        self.dispatched_probs = None
        self.tokens_per_expert = None
        self.padding_token_per_experts = None
        self.router_topk = None

        del self.unzip_node
        del self.zip_node
        self.unzip_node = None
        self.zip_node = None

        self.experts_group_gemm_node.reset_statue()
        self.experts_group_gemm_node = None

    def prepare_env_subbatch(self, unzipped_tokens=None, unzipped_tokens_scale=None, is_fwd=True):
        if is_fwd:
            assert unzipped_tokens is not None and unzipped_tokens_scale is not None
            self.experts_group_gemm_node.input_fp8 = unzipped_tokens
            self.experts_group_gemm_node.input_scale = unzipped_tokens_scale
            self.m_indices = self.experts_group_gemm_node.gen_m_indices(self.padding_token_per_experts)
            self.experts_group_gemm_node.fwd_subbatch = True
        else:
            self.m_indices = (
                self.experts_group_gemm_node.gen_m_indices(self.padding_token_per_experts)
                if not hasattr(self, "m_indices")
                else self.m_indices
            )
            self.experts_group_gemm_node.bwd_subbatch = True
            reload(self.experts_group_gemm_node.input_fp8)
            reload(self.experts_group_gemm_node.input_scale)

    def gemm_forward_subbatch(
        self,
        unzipped_tokens,
        unzipped_tokens_scale,
        unzipped_probs,
        map_unzipped_indices_to_zipped,
        output,
        total_zipped_tokens,
        padding_token_per_experts,
        start_idx=None,
        end_idx=None,
        output_subbatch_rows=None,
    ):
        if start_idx is None or end_idx is None:
            start_idx = 0
            end_idx = unzipped_tokens.shape[0]
        start_idx = max(0, start_idx)
        end_idx = min(unzipped_tokens.shape[0], end_idx)

        expert_out = self.experts_group_gemm_node.forward(
            (unzipped_tokens[start_idx:end_idx], unzipped_tokens_scale[start_idx:end_idx]),
            unzipped_probs[start_idx:end_idx],
            padding_token_per_experts,
            m_indices=self.m_indices[start_idx:end_idx],
        )

        output = tokens_zip_unique_add_with_subbatch(
            output,
            expert_out,
            map_unzipped_indices_to_zipped[start_idx:end_idx],
            total_zipped_tokens,
            subbatch_rows=output_subbatch_rows,
        )
        return output

    def gemm_backward_subbatch(
        self,
        unzipped_grad,
        map_unzipped_indices_to_zipped,
        total_zipped_tokens,
        output,
        padding_token_per_experts,
        start_idx=None,
        end_idx=None,
        output_subbatch_rows=None,
        reset_status=False,
    ):
        def split_list_prefix(l, start, end):
            prefix_sum = [0] * (len(l) + 1)
            for i in range(len(l)):
                prefix_sum[i + 1] = prefix_sum[i] + l[i]

            result = []
            for i in range(len(l)):
                segment_start = prefix_sum[i]
                segment_end = prefix_sum[i + 1]
                overlap_start = max(start, segment_start)
                overlap_end = min(end, segment_end)
                selected = max(0, overlap_end - overlap_start)
                result.append(selected)
            return result

        if start_idx is None or end_idx is None:
            start_idx = 0
            end_idx = extract_first_if_tuple(unzipped_grad).shape[0]

        start_idx = max(0, start_idx)
        end_idx = min(extract_first_if_tuple(unzipped_grad).shape[0], end_idx)

        # m_indices = self.experts_group_gemm_node.gen_m_indices(self.tokens_per_expert)
        unzipped_inp_grad = (
            (unzipped_grad[0][start_idx:end_idx].contiguous(), unzipped_grad[1][start_idx:end_idx].contiguous())
            if isinstance(unzipped_grad, tuple)
            else unzipped_grad[start_idx:end_idx].contiguous()
        )
        unzipped_grad, unzipped_probs_grad = self.experts_group_gemm_node.backward(
            unzipped_inp_grad,
            self.unzipped_probs[start_idx:end_idx].contiguous(),
            input_fp8_slice=self.experts_group_gemm_node.input_fp8[start_idx:end_idx].contiguous(),
            input_scale_slice=self.experts_group_gemm_node.input_scale[start_idx:end_idx].contiguous(),
            tokens_per_expert=split_list_prefix(padding_token_per_experts, start_idx, end_idx),
            m_indices=self.m_indices[start_idx:end_idx].contiguous(),
            reset_status=reset_status,
        )

        output = tokens_zip_unique_add_with_subbatch(
            output,
            unzipped_grad,
            map_unzipped_indices_to_zipped[start_idx:end_idx],
            zipped_rows=total_zipped_tokens,
            subbatch_rows=output_subbatch_rows,
        )

        return output, unzipped_probs_grad

    @paddle.no_grad()
    def forward(self, hs_2d_dispatched, dispatched_indices, dispatched_probs):
        """
        Perform forward computation on input data.

        Args:
            hs_fp8_dispatched (Tensor): Input data dispatched to experts.
            dispatched_indices (Tensor): Expert indices assigned to input data.
            dispatched_probs (Tensor): Probabilities of input data being dispatched to experts.

        Returns:
            Tensor: Output data after forward computation.
        """
        self.tokens_per_expert = self.token_dispatcher._comm_manager.tokens_per_expert
        self.dispatched_probs = dispatched_probs
        num_experts = len(self.tokens_per_expert)
        padding_token_per_experts = [(x + 127) // 128 * 128 for x in self.tokens_per_expert]
        self.padding_token_per_experts = padding_token_per_experts
        # 1 unzip
        self.dispatched_indices = dispatched_indices.to(paddle.int32)

        total_zipped_tokens = extract_first_if_tuple(hs_2d_dispatched).shape[0]
        (unzipped_tokens, zipped_expertwise_rowmap, unzipped_probs, unzipped_tokens_scale,) = self.unzip_node.forward(
            hs_2d_dispatched,
            self.dispatched_indices,
            dispatched_probs,
            topk=self.router_topk,
            num_experts=num_experts,
            tokens_per_expert=self.tokens_per_expert,
        )
        record_stream_for_multi_input(hs_2d_dispatched)
        dispatched_indices._record_stream()
        dispatched_probs._record_stream()

        self.unzipped_probs = unzipped_probs.unsqueeze(-1)

        if self.dsv3_use_fp8_dispatch:
            total_unzipped_tokens = extract_first_if_tuple(unzipped_tokens).shape[0]
            # If adaptive O1 recompute is enabled, determine whether to enable recompute O1 based on the degree of imbalance
            if self.recompute_fwd_gate_up == -1:
                if (
                    total_unzipped_tokens
                    > self.seq_length * self.num_experts_per_tok * self.adaptive_remained_O1_recompute_ratio
                ):
                    # logger.debug(f"recompute_fwd_gate_up changed to True, Because the receives {unzipped_tokens.shape[0]} Tensors greater then {self.seq_length*self.num_experts_per_tok*self.adaptive_remained_O1_recompute_ratio}.")
                    self.set_recompute_fwd_gate_up(True)
                else:
                    # logger.debug(f"recompute_fwd_gate_up changed to False, Because the receives {unzipped_tokens.shape[0]} Tensors less then {self.seq_length*self.num_experts_per_tok*self.adaptive_remained_O1_recompute_ratio}.")
                    self.set_recompute_fwd_gate_up(False)

            # if use_mlp_subbatch is enabled, then split the unzipped_tokens into subbatches
            if self.mlp_fwd_subbatch_rows != 0 and total_unzipped_tokens > self.mlp_fwd_subbatch_rows * 2:
                assert (
                    self.experts_group_gemm_node.recompute_fwd_gate_up
                ), "recompute_fwd_gate_up must be true when use_mlp_subbatch = True"
                map_unzipped_indices_to_zipped = paddlefleet.extensions.ops.tokens_unzip_slice(
                    extract_first_if_tuple(hs_2d_dispatched),
                    zipped_expertwise_rowmap,
                    num_experts,
                    total_unzipped_tokens,
                    0,
                    total_unzipped_tokens + 1,
                )
                if isinstance(hs_2d_dispatched, tuple):
                    hs_2d_dispatched[0]._clear_to_zero_allocation()
                    hs_2d_dispatched[1]._clear_to_zero_allocation()
                else:
                    hs_2d_dispatched._clear_to_zero_allocation()

                subbatch_rows = min((total_unzipped_tokens // num_experts) // 128 * 128, self.mlp_fwd_subbatch_rows)
                nparts = (total_unzipped_tokens + subbatch_rows - 1) // subbatch_rows
                output = paddle.empty([0, extract_first_if_tuple(hs_2d_dispatched).shape[-1]], dtype=paddle.float32)
                self.prepare_env_subbatch(unzipped_tokens, unzipped_tokens_scale, True)
                logger.info(
                    f"Enable subbatch_forward!! total_zipped_tokens:{total_zipped_tokens}, total_unzipped_tokens:{total_unzipped_tokens}, nparts:{nparts}, subbatch_rows:{subbatch_rows}, output_sub_rows:{self.output_subbatch_rows}"
                )
                for i in range(nparts):
                    start_idx = i * subbatch_rows
                    end_idx = min(start_idx + subbatch_rows, total_unzipped_tokens)
                    output = self.gemm_forward_subbatch(
                        unzipped_tokens,
                        unzipped_tokens_scale,
                        unzipped_probs,
                        map_unzipped_indices_to_zipped,
                        output,
                        total_zipped_tokens,
                        padding_token_per_experts,
                        start_idx=start_idx,
                        end_idx=end_idx,
                        output_subbatch_rows=self.output_subbatch_rows,
                    )

                output = merge_subbatch_cast(output, paddle.bfloat16)
                output.stop_gradient = False
                offload(self.experts_group_gemm_node.input_fp8)
                offload(self.experts_group_gemm_node.input_scale)
                return output

            # 2 experts
            expert_out = self.experts_group_gemm_node.forward(
                (unzipped_tokens, unzipped_tokens_scale), unzipped_probs, padding_token_per_experts
            )
        else:
            # If adaptive O1 recompute is enabled, determine whether to enable recompute O1 based on the degree of imbalance
            if self.recompute_fwd_gate_up == -1:
                if (
                    unzipped_tokens.shape[0]
                    > self.seq_length * self.num_experts_per_tok * self.adaptive_remained_O1_recompute_ratio
                ):
                    self.set_recompute_fwd_gate_up(True)
                else:
                    self.set_recompute_fwd_gate_up(False)

            # 2 experts
            expert_out = self.experts_group_gemm_node.forward(
                unzipped_tokens, unzipped_probs, padding_token_per_experts
            )

        # 3 zip
        if isinstance(hs_2d_dispatched, tuple):
            hs_2d_dispatched[0]._clear_to_zero_allocation()
            hs_2d_dispatched[1]._clear_to_zero_allocation()
        else:
            hs_2d_dispatched._clear_to_zero_allocation()
        expert_out_tmp = expert_out.reshape([-1, expert_out.shape[-1]])

        expert_out_zipped = self.zip_node.forward(
            expert_out_tmp,
            zipped_expertwise_rowmap,
            self.dispatched_indices,
            unzipped_probs,
            total_zipped_tokens=total_zipped_tokens,
            num_experts=num_experts,
        )

        expert_out_zipped.stop_gradient = False
        return expert_out_zipped

    @paddle.no_grad()
    def backward(self, hidden_states_out_grad):
        """
        Backward propagation function.

        Args:
            hidden_states_out_grad_fp8 (Tensor): Gradient of hidden states.

        Returns:
            Tuple[Tensor, Tensor]: Contains two elements:
                - hs_fp8_dispatched_grad (Tensor): Gradient of unzipped hidden states.
                - dispatched_probs_grad (Tensor): Gradient of dispatch probabilities.
        """
        # zip_grad
        unzipped_grad = self.zip_node.backward(
            hidden_states_out_grad,
            self.dispatched_indices,
            self.dispatched_probs,
            top_k=self.router_topk,
            num_experts=len(self.tokens_per_expert),
            tokens_per_expert=self.tokens_per_expert,
        )
        record_stream_for_multi_input(hidden_states_out_grad)

        total_zipped_tokens = extract_first_if_tuple(hidden_states_out_grad).shape[0]
        total_unzipped_tokens = extract_first_if_tuple(unzipped_grad).shape[0]
        hidden_states_size = extract_first_if_tuple(hidden_states_out_grad).shape[-1]
        num_experts = len(self.tokens_per_expert)
        padding_token_per_experts = [(x + 127) // 128 * 128 for x in self.tokens_per_expert]

        if self.mlp_bwd_subbatch_rows != 0 and total_unzipped_tokens > self.mlp_bwd_subbatch_rows * 2:
            map_unzipped_indices_to_zipped = paddlefleet.extensions.ops.tokens_unzip_slice(
                extract_first_if_tuple(hidden_states_out_grad),
                self.unzip_node.zipped_expertwise_rowmap,
                num_experts,
                total_unzipped_tokens,
                0,
                total_unzipped_tokens + 1,
            )
            if isinstance(hidden_states_out_grad, tuple):
                hidden_states_out_grad[0]._clear_to_zero_allocation()
                hidden_states_out_grad[1]._clear_to_zero_allocation()
            else:
                hidden_states_out_grad._clear_to_zero_allocation()

            subbatch_rows = min((total_unzipped_tokens // num_experts) // 128 * 128, self.mlp_bwd_subbatch_rows)
            nparts = (total_unzipped_tokens + subbatch_rows - 1) // subbatch_rows
            output = paddle.empty([0, hidden_states_size], dtype=paddle.float32)
            probs_grad_list = []
            self.prepare_env_subbatch(is_fwd=False)
            logger.info(
                f"Enable subbatch_backward!! total_zipped_tokens:{total_zipped_tokens}, total_unzipped_tokens:{total_unzipped_tokens}, nparts:{nparts}, subbatch_rows:{subbatch_rows}, output_sub_rows:{self.output_subbatch_rows}"
            )
            for i in range(nparts):
                reset_status = True if i == nparts - 1 else False  # release saved status in the last part.
                start_idx = i * subbatch_rows
                end_idx = min(start_idx + subbatch_rows, total_unzipped_tokens)
                output, probs_grad = self.gemm_backward_subbatch(
                    unzipped_grad,
                    map_unzipped_indices_to_zipped,
                    total_zipped_tokens,
                    output,
                    padding_token_per_experts,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    output_subbatch_rows=self.output_subbatch_rows,
                    reset_status=reset_status,
                )
                probs_grad_list.append(probs_grad)
            if isinstance(unzipped_grad, tuple):
                unzipped_grad[0]._clear_to_zero_allocation()
                unzipped_grad[1]._clear_to_zero_allocation()
            else:
                unzipped_grad._clear_to_zero_allocation()
            hs_dispatched_grad = merge_subbatch_cast(output, paddle.bfloat16)
            dispatched_probs_grad = paddlefleet.extensions.ops.tokens_zip_prob_seq_subbatch(
                probs_grad_list, self.unzip_node.zipped_expertwise_rowmap, self.dispatched_indices, subbatch_rows
            )
            self.reset_statue()
            return hs_dispatched_grad, dispatched_probs_grad

        if isinstance(hidden_states_out_grad, tuple):
            hidden_states_out_grad[0]._clear_to_zero_allocation()
            hidden_states_out_grad[1]._clear_to_zero_allocation()
        else:
            hidden_states_out_grad._clear_to_zero_allocation()

        # expert_grad
        expert_out, probs_grad = self.experts_group_gemm_node.backward(
            unzipped_grad, self.unzipped_probs, padding_token_per_experts
        )

        hs_dispatched_grad, dispatched_probs_grad = self.unzip_node.backward(
            expert_out,
            total_zipped_tokens,
            probs_grad,
            self.dispatched_indices,
            num_experts=num_experts,
        )

        self.reset_statue()
        return hs_dispatched_grad, dispatched_probs_grad


class FusionMoeNode:
    def __init__(
        self,
        custom_map,
        recompute_fwd_gate_up=False,
        is_split_group_gemm=True,
        dsv3_use_fp8_dispatch=True,
        mlp_fwd_subbatch_rows=0,
        mlp_bwd_subbatch_rows=0,
        output_subbatch_rows=0,
        name="fusion_moe_node",
    ):
        self.token_dispatcher = custom_map.token_dispatcher
        self.moe_router_topk = custom_map.moe_router_topk
        self.dsv3_use_fp8_dispatch = dsv3_use_fp8_dispatch
        self.dispatch_quant_node = Fp8DispatchQuantNode(self.token_dispatcher, dsv3_use_fp8_dispatch)
        self.dispatch_node = Fp8DispatchNode(self.token_dispatcher)
        self.mlp_node = FusionMlpNode(
            custom_map,
            self.moe_router_topk,
            recompute_fwd_gate_up=recompute_fwd_gate_up,
            is_split_group_gemm=is_split_group_gemm,
            dsv3_use_fp8_dispatch=dsv3_use_fp8_dispatch,
            mlp_fwd_subbatch_rows=mlp_fwd_subbatch_rows,
            mlp_bwd_subbatch_rows=mlp_bwd_subbatch_rows,
            output_subbatch_rows=output_subbatch_rows,
        )
        self.combine_node = Fp8CombineNode(self.token_dispatcher)
        self.combine_quant_node = Fp8CombineQuantNode(
            self.token_dispatcher, dsv3_use_fp8_dispatch, custom_map.moe_group
        )
        self.name = name

    @paddle.no_grad()
    def forward(self, hidden_states, probs, routing_map):
        if self.dsv3_use_fp8_dispatch:
            (hs_fp8, hs_scale), token_indices, token_probs = self.dispatch_quant_node.forward(
                hidden_states, probs, routing_map
            )
            (
                (hs_fp8_dispatched, hs_scale_dispatched),
                dispatched_indices,
                dispatched_probs,
            ) = self.dispatch_node.forward((hs_fp8, hs_scale), token_indices, token_probs)
            hidden_states_out = self.mlp_node.forward(
                (hs_fp8_dispatched, hs_scale_dispatched), dispatched_indices, dispatched_probs
            )
            output_combine = self.combine_node.forward(hidden_states_out)
            output = self.combine_quant_node.forward(output_combine)
            output.stop_gradient = False
            return output
        else:
            hs_2d_dispatched, dispatched_indices, dispatched_probs = self.dispatch_node.forward(
                hidden_states, probs, routing_map
            )
            hidden_states_out = self.mlp_node.forward(hs_2d_dispatched, dispatched_indices, dispatched_probs)
            output_combine = self.combine_node.forward(hidden_states_out)
            output = self.combine_quant_node.forward(output_combine)
            output.stop_gradient = False
            return output

    @paddle.no_grad()
    def backward(self, output_grad):
        output_combine_grad, _ = self.combine_quant_node.backward(output_grad)
        hidden_states_out_grad = self.combine_node.backward(output_combine_grad)

        hs_dispatched_grad, dispatched_probs_grad = self.mlp_node.backward(hidden_states_out_grad)

        if self.dsv3_use_fp8_dispatch:
            hs_fp8_grad, token_probs_grad = self.dispatch_node.backward(hs_dispatched_grad, dispatched_probs_grad)
            hs_grad, probs_grad, routing_map_grad = self.dispatch_quant_node.backward(hs_fp8_grad, token_probs_grad)
            return hs_grad, probs_grad, routing_map_grad
        else:
            hs_bf16_grad, token_probs_grad = self.dispatch_node.backward(hs_dispatched_grad, dispatched_probs_grad)
            return hs_bf16_grad, None, token_probs_grad


class FusionMoe(paddle.autograd.PyLayer):
    @staticmethod
    def forward(
        ctx,
        hidden_states,
        probs,
        routing_map,
        custom_map,
        recompute_fwd_gate_up=False,
        is_split_group_gemm=True,
        dsv3_use_fp8_dispatch=True,
    ):
        ctx.node = FusionMoeNode(
            custom_map,
            recompute_fwd_gate_up=recompute_fwd_gate_up,
            is_split_group_gemm=is_split_group_gemm,
            dsv3_use_fp8_dispatch=dsv3_use_fp8_dispatch,
        )
        return ctx.node.forward(hidden_states, probs, routing_map)

    @staticmethod
    def backward(ctx, output_grad):
        return ctx.node.backward(output_grad)
