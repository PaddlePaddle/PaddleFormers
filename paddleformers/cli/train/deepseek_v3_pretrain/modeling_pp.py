# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import math
from typing import OrderedDict, Tuple, Union

import paddle
import paddle.distributed.fleet as fleet
import paddle.nn as nn
from paddle.distributed.fleet.meta_parallel import (
    LayerDesc,
    LocalSharedLayerDesc,
    PipelineLayer,
    ScheduleChunk,
    ScheduleNode,
    SharedLayerDesc,
)
from paddle.distributed.fleet.meta_parallel.zero_bubble_utils import WeightGradStore

try:
    from paddle.distributed.fleet.meta_parallel.zero_bubble_utils import EventStore
except ImportError:
    EventStore = None

from paddle.distributed.fleet.recompute.recompute import recompute
from paddle.distributed.fleet.utils.sequence_parallel_utils import ScatterOp

from paddleformers.transformers.model_utils import PipelinePretrainedModel
from paddleformers.utils.log import logger

from .configuration import DeepseekV2FastConfig
from .modeling import DeepseekV2DecoderLayer, DeepseekV2LMHead
from .modeling import DeepseekV2ModelFast as DeepseekV2Model
from .modeling import DeepseekV2MoE, DeepseekV2MTPLayer
from .modeling import DeepseekV2PretrainedModelFast as DeepseekV2PretrainedModel
from .modeling import (
    DeepseekV2PretrainingCriterionFast,
    DeepseekV2RMSNorm,
    TemporaryVarContext,
    set_global_step,
)
from .moe_utils import get_env_device

try:
    import paddle.distributed.communication.deep_ep as deep_ep
except ImportError:
    deep_ep = None

from paddleformers.transformers.fp8_utils import (
    FP8LinearFunction,
    FP8LinearFunctionBase,
)
from paddleformers.transformers.fused_a2a import (
    fused_combine_backward_func,
    fused_combine_forward_func,
    fused_dispatch_backward_func,
    fused_dispatch_forward_func,
)

from .moe_layer import FusionMoeNode

__all__ = [
    "DeepseekV2ForCausalLMPipe",
]

import queue

global_inputs_embeds_mtp_queue = queue.Queue()


def parse_args(args):
    if isinstance(args, (tuple, list)):
        if len(args) == 4:
            hidden_states, attention_mask, attn_mask_startend_row_indices, position_ids = args

        elif len(args) == 3:
            hidden_states, attention_mask, attn_mask_startend_row_indices = args
            position_ids = None
        elif len(args) == 2:
            hidden_states, attention_mask = args
            attn_mask_startend_row_indices, position_ids = None, None
        else:  # len(args) == 1:
            hidden_states = args[0]
            attention_mask, attn_mask_startend_row_indices, position_ids = None, None, None
    else:
        hidden_states = args
        attention_mask, attn_mask_startend_row_indices, position_ids = None, None, None

    if position_ids is not None:
        position_ids.stop_gradient = True

    if attention_mask is not None:
        attention_mask.stop_gradient = True

    if attn_mask_startend_row_indices is not None:
        attn_mask_startend_row_indices.stop_gradient = True

    return hidden_states, attention_mask, attn_mask_startend_row_indices, position_ids


def return_args(hidden_states, attention_mask=None, attn_mask_startend_row_indices=None, position_ids=None):
    ret = (hidden_states,)

    if attention_mask is not None:
        ret += (attention_mask.clone(),)
    if attn_mask_startend_row_indices is not None:
        ret += (attn_mask_startend_row_indices.clone(),)
    if position_ids is not None:
        ret += (position_ids.clone(),)
    if len(ret) == 1:
        ret = ret[0]

    return ret


def get_attr(layer, name):
    if getattr(layer, name, None) is not None:
        return getattr(layer, name, None)
    else:
        return get_attr(layer._layer, name)


def calc_stream_wait(group_id):
    comm_event = deep_ep.get_event_from_comm_stream(group_id)
    comm_event.calc_stream_wait(group_id)


class TensorMeta:
    """Recording the meta info of forward inputs, to avoid 0-size problems"""

    def __init__(self, tensor):
        self.shape = tensor.shape
        self.dtype = tensor.dtype


class PostProcessNode(ScheduleNode):
    def __init__(
        self,
        send_mtp_embed,
        training,
        alpha,
        config,
        shared_experts=None,
        using_post_norm_recompute=False,
        output_mtp_embed_first=False,
        name="PostProcessNode",
    ):
        self.send_mtp_embed = send_mtp_embed
        self.shared_experts = shared_experts
        self.traning = training
        self.config = config
        self.alpha = alpha
        self.using_post_norm_recompute = using_post_norm_recompute
        self.output_mtp_embed_first = output_mtp_embed_first
        self.name = name

        if self.using_post_norm_recompute:
            assert self.shared_experts is not None
            assert self.shared_experts.norm_weight is not None and self.shared_experts.norm_eps is not None

    def forward_without_residual(self, inputs):

        if isinstance(inputs, list):
            inputs = tuple(inputs)
        if self.using_post_norm_recompute:
            if self.send_mtp_embed:
                (inputs_embeds_mtp, hidden_states, residual, l_aux, final_hidden_states, norm_out) = inputs
            else:
                (hidden_states, residual, l_aux, final_hidden_states, norm_out) = inputs
        else:
            if self.send_mtp_embed:
                (inputs_embeds_mtp, hidden_states, residual, l_aux, final_hidden_states) = inputs
            else:
                (hidden_states, residual, l_aux, final_hidden_states) = inputs

        with paddle.no_grad():
            if self.shared_experts is not None:
                if self.using_post_norm_recompute:
                    _, _, _, shared_expert_output = FP8LinearFunctionBase.fp8_mlp_fwd(
                        norm_out, self.shared_experts.w1, self.shared_experts.w2
                    )
                    norm_out = None
                    del norm_out
                else:
                    _, _, _, shared_expert_output = FP8LinearFunctionBase.fp8_mlp_fwd(
                        hidden_states, self.shared_experts.w1, self.shared_experts.w2
                    )
                residual = residual + shared_expert_output

        self.x = hidden_states
        self.l_aux = l_aux

        hidden_states = residual
        hidden_states.stop_gradient = False

        if self.send_mtp_embed:
            assert not self.output_mtp_embed_first, "forward_without_residual doesn't support output_mtp_embed_first"
            hidden_states = paddle.cat([hidden_states, inputs_embeds_mtp], axis=-1)
            self.mtp_embed_shape = (
                inputs_embeds_mtp.shape
            )  # Save the shape of mtp_embed, used for backward propagation

        return return_args(hidden_states)

    def forward(self, inputs):

        if isinstance(inputs, list):
            inputs = tuple(inputs)

        if self.using_post_norm_recompute:
            if self.send_mtp_embed:
                (inputs_embeds_mtp, hidden_states, residual, l_aux, final_hidden_states, norm_out) = inputs
            else:
                (hidden_states, residual, l_aux, final_hidden_states, norm_out) = inputs
        else:
            if self.send_mtp_embed:
                (inputs_embeds_mtp, hidden_states, residual, l_aux, final_hidden_states) = inputs
            else:
                (hidden_states, residual, l_aux, final_hidden_states) = inputs

        with paddle.no_grad():
            if self.shared_experts is not None:
                if self.using_post_norm_recompute:
                    _, _, _, shared_expert_output = FP8LinearFunctionBase.fp8_mlp_fwd(
                        norm_out, self.shared_experts.w1, self.shared_experts.w2
                    )
                    norm_out = None
                    del norm_out
                else:
                    _, _, _, shared_expert_output = FP8LinearFunctionBase.fp8_mlp_fwd(
                        hidden_states, self.shared_experts.w1, self.shared_experts.w2
                    )
                final_hidden_states = final_hidden_states + shared_expert_output

        self.x = hidden_states
        self.l_aux = l_aux
        hidden_states = residual + final_hidden_states

        if self.send_mtp_embed:
            if self.output_mtp_embed_first:
                hidden_states = paddle.cat([inputs_embeds_mtp, hidden_states], axis=-1)
            else:
                hidden_states = paddle.cat([hidden_states, inputs_embeds_mtp], axis=-1)
            self.mtp_embed_shape = (
                inputs_embeds_mtp.shape
            )  # Save the shape of mtp_embed shape, used for backward propagation

        return return_args(hidden_states)

    @paddle.no_grad()
    def backward(self, output_grad):
        (do3,) = output_grad

        if self.send_mtp_embed:
            # Split gradient: first part of do3 corresponds to hidden_states, second part corresponds to inputs_embeds_mtp
            hidden_size = do3.shape[-1] - self.mtp_embed_shape[-1]
            if self.output_mtp_embed_first:
                hidden_states_grad = do3[..., hidden_size:]
                inputs_embeds_mtp_grad = do3[..., :hidden_size]
            else:
                hidden_states_grad = do3[..., :hidden_size]
                inputs_embeds_mtp_grad = do3[..., hidden_size:]
        else:
            hidden_states_grad = do3
            inputs_embeds_mtp_grad = None

        if self.using_post_norm_recompute:
            dx, norm_out, invar = FP8LinearFunctionBase.fp8_mlp_bwd_norm_rc(
                hidden_states_grad,
                self.x,
                self.shared_experts.norm_weight,
                self.shared_experts.norm_eps,
                self.shared_experts.w1,
                self.shared_experts.w2,
            )
        else:
            dx = FP8LinearFunctionBase.fp8_mlp_bwd(
                hidden_states_grad, self.x, self.shared_experts.w1, self.shared_experts.w2, True
            )

        self.x = None

        residual_grad = hidden_states_grad
        l_aux_grad = paddle.ones(1, dtype=self.l_aux.dtype) * self.alpha
        final_hidden_states_grad = hidden_states_grad

        if self.using_post_norm_recompute:
            if self.send_mtp_embed:
                return (
                    inputs_embeds_mtp_grad,
                    dx,
                    residual_grad,
                    l_aux_grad,
                    final_hidden_states_grad,
                    norm_out,
                    invar,
                )
            else:
                return (dx, residual_grad, l_aux_grad, final_hidden_states_grad, norm_out, invar)
        else:
            if self.send_mtp_embed:
                return (inputs_embeds_mtp_grad, dx, residual_grad, l_aux_grad, final_hidden_states_grad)
            else:
                return (dx, residual_grad, l_aux_grad, final_hidden_states_grad)


class DecoderLayerNode(ScheduleNode):
    def __init__(
        self,
        attn_node,
        dispatch_node,
        mlp_node,
        combine_node,
        post_process_node,
        mlp_layer,
        name="DecoderLayerNode",
    ):
        super().__init__(fwd_func=None, name=name)
        assert (dispatch_node is None and combine_node is None) or (
            dispatch_node is not None and combine_node is not None
        )
        self.attn_node = attn_node
        self.dispatch_node = dispatch_node
        self.mlp_node = mlp_node
        self.combine_node = combine_node
        self.post_process_node = post_process_node

        self.mlp_layer = mlp_layer
        self.moe_group = mlp_layer.moe_group
        self.moe_num_experts = mlp_layer.moe_num_experts

        self.states = None
        self.hidden_states_meta = None
        self.dispatched_probs_meta = None
        self.combine_output_meta = None

    def dispatch_forward(self, inputs, previous_event=None, allocate_on_comm_stream=False):
        paddle.base.core.nvprof_nvtx_push("raw_dispatch_forward")
        if isinstance(inputs, list):
            inputs = tuple(inputs)
        (
            inputs_embeds_mtp,
            hidden_states,
            residual,
            l_aux,
            intermediate_hidden_states,
            token_indices,
            token_probs,
        ) = inputs

        with paddle.no_grad():
            intermediate_hidden_states, dispatched_probs, states, _ = fused_dispatch_forward_func(
                intermediate_hidden_states,
                token_indices,
                token_probs,
                self.moe_num_experts,
                self.moe_group,
                previous_event=previous_event,
                async_finish=True,
                allocate_on_comm_stream=allocate_on_comm_stream,
            )
        dispatched_indices = states["dispatched_indices"]
        self.mlp_layer.set_tokens_per_expert(states["tokens_per_expert"])
        dispatched_indices.stop_gradient = True
        intermediate_hidden_states.stop_gradient = False
        dispatched_probs.stop_gradient = False
        self.states = states
        self.hidden_states_meta = TensorMeta(intermediate_hidden_states)
        self.dispatched_probs_meta = TensorMeta(dispatched_probs)

        inputs = (
            inputs_embeds_mtp,
            hidden_states,
            residual,
            l_aux,
            intermediate_hidden_states,
            dispatched_indices,
            dispatched_probs,
        )
        paddle.base.core.nvprof_nvtx_pop()
        return inputs

    def combine_forward(self, inputs, previous_event=None):
        paddle.base.core.nvprof_nvtx_push("raw_combine_forward")
        if isinstance(inputs, list):
            inputs = tuple(inputs)
        (inputs_embeds_mtp, hidden_states, residual, l_aux, expert_output) = inputs

        with paddle.no_grad():
            combine_output = fused_combine_forward_func(
                expert_output, self.moe_group, self.states, previous_event=previous_event, async_finish=True
            )
        combine_output.stop_gradient = False
        self.combine_output_meta = TensorMeta(combine_output)
        inputs = (inputs_embeds_mtp, hidden_states, residual, l_aux, combine_output)
        paddle.base.core.nvprof_nvtx_pop()
        return inputs

    def dispatch_backward(self, output_grad):
        paddle.base.core.nvprof_nvtx_push("raw_dispatch_backward")
        (
            inputs_embeds_mtp_grad,
            hidden_states_grad,
            residual_grad,
            l_aux_grad,
            intermediate_hidden_states_grad,
            dispatched_indices_grad,
            dispatched_probs_grad,
        ) = output_grad

        if intermediate_hidden_states_grad is None:
            intermediate_hidden_states_grad = paddle.zeros(
                self.hidden_states_meta.shape, self.hidden_states_meta.dtype
            )
        if dispatched_probs_grad is None:
            dispatched_probs_grad = paddle.zeros(self.dispatched_probs_meta.shape, self.dispatched_probs_meta.dtype)
        with paddle.no_grad():
            intermediate_hidden_states_grad, token_indices_grad, token_probs_grad = fused_dispatch_backward_func(
                intermediate_hidden_states_grad,
                dispatched_probs_grad,
                self.moe_group,
                self.states["handle"],
                async_finish=True,
            )

        output_grad = (
            inputs_embeds_mtp_grad,
            hidden_states_grad,
            residual_grad,
            l_aux_grad,
            intermediate_hidden_states_grad,
            token_indices_grad,
            token_probs_grad,
        )
        paddle.base.core.nvprof_nvtx_pop()
        return output_grad

    def combine_backward(self, output_grad):
        paddle.base.core.nvprof_nvtx_push("raw_combine_backward")
        (
            inputs_embeds_mtp_grad,
            hidden_states_grad,
            residual_grad,
            l_aux_grad,
            combine_output_grad,
        ) = output_grad

        if combine_output_grad is None:
            combine_output_grad = paddle.zeros(self.combine_output_meta.shape, self.combine_output_meta.dtype)
        with paddle.no_grad():
            expert_output_grad = fused_combine_backward_func(
                combine_output_grad, self.moe_group, self.states["handle"], async_finish=True
            )

        output_grad = (
            inputs_embeds_mtp_grad,
            hidden_states_grad,
            residual_grad,
            l_aux_grad,
            expert_output_grad,
        )
        paddle.base.core.nvprof_nvtx_pop()
        return output_grad

    def forward(self, inputs):
        inputs = self.attn_node.forward(inputs)

        if self.dispatch_node is None:
            inputs = self.dispatch_forward(inputs)
            calc_stream_wait(self.moe_group.id)
        else:
            inputs = self.dispatch_node.forward(inputs)

        inputs = self.mlp_node.forward(inputs)

        if self.combine_node is None:
            inputs = self.combine_forward(inputs)
            calc_stream_wait(self.moe_group.id)
        else:
            inputs = self.combine_node.forward(inputs)

        inputs = self.post_process_node.forward(inputs)
        return inputs

    def backward(self, output_grad=None, scaler=None):
        assert (output_grad is not None) and (scaler is None)

        output_grad = self.post_process_node.backward(output_grad)

        if self.combine_node is None:
            output_grad = self.combine_backward(output_grad)
            calc_stream_wait(self.moe_group.id)
        else:
            output_grad = self.combine_node.backward(output_grad)

        output_grad = self.mlp_node.backward(output_grad)

        if self.dispatch_node is None:
            output_grad = self.dispatch_backward(output_grad)
            calc_stream_wait(self.moe_group.id)
        else:
            output_grad = self.dispatch_node.backward(output_grad)

        output_grad = self.attn_node.backward(output_grad)
        return output_grad


class OverlapedScheduleChunk:
    def __init__(self, forward_nodes, backward_nodes, use_fuion=True):
        assert len(forward_nodes) == len(backward_nodes)
        self.nodes = []
        for f, b in zip(forward_nodes, backward_nodes):
            schedule_node_class = OverlapedScheduleNode
            if use_fuion:
                schedule_node_class = OverlapedFUsionScheduleNode
                if isinstance(f, DenseDecoderLayerNode) or isinstance(b, DenseDecoderLayerNode):
                    schedule_node_class = OverlapedDenseFusionScheduleNode
            self.nodes.append(schedule_node_class(f, b, f"OverlapedNode_{len(self.nodes)}"))

    def forward_backward(self, inputs, output_grad, combine_bw_event_to_wait=None, pp_stream=None):
        event_to_wait = combine_bw_event_to_wait
        for i, n in enumerate(self.nodes):
            pp_stream_t = pp_stream
            if i + 1 != len(self.nodes):
                pp_stream_t = None

            inputs, output_grad, event_to_wait = n.forward_backward(
                inputs, output_grad, combine_bw_event_to_wait=event_to_wait, pp_stream=pp_stream_t
            )
        return inputs, output_grad, None


class DecoderBackwardScheduleChunk:
    def __init__(self, nodes):
        self.nodes = nodes

    def backward(self, output_grad, combine_bw_event_to_wait=None, pp_stream=None):
        event_to_wait = combine_bw_event_to_wait
        for i, n in enumerate(self.nodes):
            pp_stream_t = pp_stream if i + 1 == len(self.nodes) else None
            output_grad, event_to_wait = n.backward_for_fusion(
                output_grad, combine_bw_event_to_wait=event_to_wait, pp_stream=pp_stream_t
            )
        return output_grad


class OverlapedScheduleNode:
    def __init__(self, forward_node, backward_node, name=""):
        assert isinstance(forward_node, DecoderLayerNode) and isinstance(backward_node, DecoderLayerNode)
        self.forward_node = forward_node
        self.backward_node = backward_node
        self.name = name

    def forward_backward(self, inputs, output_grad, event_to_wait=None):
        paddle.base.core.nvprof_nvtx_push("forward_backward")
        output_grad = self.backward_node.post_process_node.backward(output_grad)

        output_grad = self.backward_node.combine_backward(output_grad)
        inputs = self.forward_node.attn_node.forward(inputs)

        calc_stream_wait(self.backward_node.moe_group.id)
        attn_compute_event = deep_ep.get_event_from_calc_stream(self.forward_node.moe_group.id)
        output_grad = self.backward_node.mlp_node.backward(output_grad)
        inputs = self.forward_node.dispatch_forward(
            inputs, previous_event=attn_compute_event, allocate_on_comm_stream=True
        )

        calc_stream_wait(self.forward_node.moe_group.id)
        output_grad = self.backward_node.dispatch_backward(output_grad)
        inputs = self.forward_node.mlp_node.forward(inputs)

        calc_stream_wait(self.backward_node.moe_group.id)
        inputs = self.forward_node.combine_forward(inputs)
        output_grad = self.backward_node.attn_node.backward(output_grad)

        calc_stream_wait(self.forward_node.moe_group.id)
        inputs = self.forward_node.post_process_node.forward(inputs)
        paddle.base.core.nvprof_nvtx_pop()
        return inputs, output_grad


class FusionFp8DecoderLayerNode(ScheduleNode):
    def __init__(
        self,
        attn_and_gate_node,
        fp8_fusion_moe_node,
        post_process_node,
        mlp_layer,
        send_mtp_embed,
        using_post_norm_recompute=False,
        stepped_recompute_fwd_gate_up=False,
        dsv3_use_fp8_dispatch=True,
        name="",
    ):
        self.attn_and_gate_node = attn_and_gate_node
        self.fp8_fusion_moe_node = fp8_fusion_moe_node
        self.post_process_node = post_process_node
        self.send_mtp_embed = send_mtp_embed

        self.using_post_norm_recompute = using_post_norm_recompute
        self.stepped_recompute_fwd_gate_up = stepped_recompute_fwd_gate_up
        self.name = name

        self.moe_group = mlp_layer.moe_group
        self.dsv3_use_fp8_dispatch = dsv3_use_fp8_dispatch

    def attn_forward(self, inputs):
        inputs = self.attn_and_gate_node.forward(inputs)

        if self.send_mtp_embed:
            if self.using_post_norm_recompute:
                inputs_embeds_mtp, hidden_states, residual, probs, routing_map, l_aux, norm_out = inputs
            else:
                inputs_embeds_mtp, hidden_states, residual, probs, routing_map, l_aux = inputs
        else:
            if self.using_post_norm_recompute:
                hidden_states, residual, probs, routing_map, l_aux, norm_out = inputs
            else:
                hidden_states, residual, probs, routing_map, l_aux = inputs

        if self.using_post_norm_recompute:
            hs_2d, token_indices, token_probs = self.fp8_fusion_moe_node.dispatch_quant_node.forward(
                norm_out, probs, routing_map
            )
            # common return values
            ret = (hidden_states, residual, l_aux, hs_2d, token_indices, token_probs, norm_out)

            # append mtp embed if needed
            ret = (inputs_embeds_mtp, *ret) if self.send_mtp_embed else ret
            return ret
        else:
            hs_2d, token_indices, token_probs = self.fp8_fusion_moe_node.dispatch_quant_node.forward(
                hidden_states, probs, routing_map
            )

            # common return values
            ret = (hidden_states, residual, l_aux, hs_2d, token_indices, token_probs)

            # append mtp embed if needed
            ret = (inputs_embeds_mtp, *ret) if self.send_mtp_embed else ret
            return ret

    def dispatch_forward(self, inputs, previous_event=None, async_finish=False, allocate_on_comm_stream=False):
        if self.using_post_norm_recompute:
            if self.send_mtp_embed:
                inputs_embeds_mtp, hidden_states, residual, l_aux, hs_2d, token_indices, token_probs, norm_out = inputs
            else:
                hidden_states, residual, l_aux, hs_2d, token_indices, token_probs, norm_out = inputs
        else:
            if self.send_mtp_embed:
                inputs_embeds_mtp, hidden_states, residual, l_aux, hs_2d, token_indices, token_probs = inputs
            else:
                hidden_states, residual, l_aux, hs_2d, token_indices, token_probs = inputs

        (hs_dispatched, dispatched_indices, dispatched_probs,) = self.fp8_fusion_moe_node.dispatch_node.forward(
            hs_2d,
            token_indices,
            token_probs,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )

        ret = (hidden_states, residual, l_aux, hs_dispatched, dispatched_indices, dispatched_probs)

        # append mtp embed if needed
        ret = (inputs_embeds_mtp, *ret) if self.send_mtp_embed else ret
        ret = (*ret, norm_out) if self.using_post_norm_recompute else ret
        return ret

    def mlp_forward(self, inputs):
        if self.using_post_norm_recompute:
            if self.send_mtp_embed:
                (
                    inputs_embeds_mtp,
                    hidden_states,
                    residual,
                    l_aux,
                    hs_dispatched,
                    dispatched_indices,
                    dispatched_probs,
                    norm_out,
                ) = inputs
            else:
                hidden_states, residual, l_aux, hs_dispatched, dispatched_indices, dispatched_probs, norm_out = inputs
        else:
            if self.send_mtp_embed:
                (
                    inputs_embeds_mtp,
                    hidden_states,
                    residual,
                    l_aux,
                    hs_dispatched,
                    dispatched_indices,
                    dispatched_probs,
                ) = inputs
            else:
                hidden_states, residual, l_aux, hs_dispatched, dispatched_indices, dispatched_probs = inputs

        hidden_states_out = self.fp8_fusion_moe_node.mlp_node.forward(
            hs_dispatched, dispatched_indices, dispatched_probs
        )
        ret = (hidden_states, residual, l_aux, hidden_states_out)

        # append mtp embed if needed
        ret = (inputs_embeds_mtp, *ret) if self.send_mtp_embed else ret
        ret = (*ret, norm_out) if self.using_post_norm_recompute else ret
        return ret

    def combine_forward(self, inputs, async_finish=False, previous_event=None, allocate_on_comm_stream=False):
        if self.using_post_norm_recompute:
            if self.send_mtp_embed:
                (inputs_embeds_mtp, hidden_states, residual, l_aux, hidden_states_out, norm_out) = inputs
            else:
                (hidden_states, residual, l_aux, hidden_states_out, norm_out) = inputs
        else:
            if self.send_mtp_embed:
                (inputs_embeds_mtp, hidden_states, residual, l_aux, hidden_states_out) = inputs
            else:
                (hidden_states, residual, l_aux, hidden_states_out) = inputs

        output_combine = self.fp8_fusion_moe_node.combine_node.forward(
            hidden_states_out,
            async_finish=async_finish,
            previous_event=previous_event,
            allocate_on_comm_stream=allocate_on_comm_stream and previous_event is not None,
        )

        ret = (hidden_states, residual, l_aux, output_combine)

        # append mtp embed if needed
        ret = (inputs_embeds_mtp, *ret) if self.send_mtp_embed else ret
        ret = (*ret, norm_out) if self.using_post_norm_recompute else ret
        return ret

    def post_process_forward(self, inputs, with_residual=True):
        if self.using_post_norm_recompute:
            if self.send_mtp_embed:
                (inputs_embeds_mtp, hidden_states, residual, l_aux, output_combine, norm_out) = inputs
            else:
                (hidden_states, residual, l_aux, output_combine, norm_out) = inputs
        else:
            if self.send_mtp_embed:
                (inputs_embeds_mtp, hidden_states, residual, l_aux, output_combine) = inputs
            else:
                (hidden_states, residual, l_aux, output_combine) = inputs
        final_hidden_states = self.fp8_fusion_moe_node.combine_quant_node.forward(output_combine)

        inputs = (hidden_states, residual, l_aux, final_hidden_states)
        inputs = (inputs_embeds_mtp, *inputs) if self.send_mtp_embed else inputs
        inputs = (*inputs, norm_out) if self.using_post_norm_recompute else inputs

        if with_residual:
            inputs = self.post_process_node.forward(inputs)
        else:
            inputs = self.post_process_node.forward_without_residual(inputs)
        return inputs

    def post_process_backward(self, output_grad, event_to_wait=None):
        if event_to_wait is not None:
            event_to_wait.calc_stream_wait(self.moe_group.id)

        grad = self.post_process_node.backward(output_grad)

        if self.using_post_norm_recompute:
            if self.send_mtp_embed:
                (
                    inputs_embeds_mtp_grad,
                    hidden_states_grad,
                    residual_grad,
                    l_aux_grad,
                    final_hidden_states_grad,
                    norm_out,
                    invar,
                ) = grad
            else:
                hidden_states_grad, residual_grad, l_aux_grad, final_hidden_states_grad, norm_out, invar = grad
        else:
            if self.send_mtp_embed:
                inputs_embeds_mtp_grad, hidden_states_grad, residual_grad, l_aux_grad, final_hidden_states_grad = grad
            else:
                hidden_states_grad, residual_grad, l_aux_grad, final_hidden_states_grad = grad

        output_combine_grad, quant_event = self.fp8_fusion_moe_node.combine_quant_node.backward(
            final_hidden_states_grad, event_to_wait
        )

        ret = (hidden_states_grad, residual_grad, l_aux_grad, output_combine_grad, quant_event)
        ret = (inputs_embeds_mtp_grad, *ret) if self.send_mtp_embed else ret
        ret = (*ret, norm_out, invar) if self.using_post_norm_recompute else ret
        return ret

    def combine_backward(self, output_grad, previous_event=None, async_finish=False, allocate_on_comm_stream=False):
        if self.using_post_norm_recompute:
            if self.send_mtp_embed:
                (
                    inputs_embeds_mtp_grad,
                    hidden_states_grad,
                    residual_grad,
                    l_aux_grad,
                    output_combine_grad,
                    quant_event,
                    norm_out,
                    invar,
                ) = output_grad
            else:
                (
                    hidden_states_grad,
                    residual_grad,
                    l_aux_grad,
                    output_combine_grad,
                    quant_event,
                    norm_out,
                    invar,
                ) = output_grad
        else:
            if self.send_mtp_embed:
                (
                    inputs_embeds_mtp_grad,
                    hidden_states_grad,
                    residual_grad,
                    l_aux_grad,
                    output_combine_grad,
                    quant_event,
                ) = output_grad
            else:
                (
                    hidden_states_grad,
                    residual_grad,
                    l_aux_grad,
                    output_combine_grad,
                    quant_event,
                ) = output_grad

        if self.dsv3_use_fp8_dispatch and quant_event is not None:
            combine_backward_wait_event = quant_event
        else:
            combine_backward_wait_event = previous_event
        hidden_states_out_grad = self.fp8_fusion_moe_node.combine_node.backward(
            output_combine_grad,
            async_finish=async_finish,
            previous_event=combine_backward_wait_event,
            allocate_on_comm_stream=allocate_on_comm_stream and quant_event is not None,
        )

        ret = (hidden_states_grad, residual_grad, l_aux_grad, hidden_states_out_grad)
        ret = (inputs_embeds_mtp_grad, *ret) if self.send_mtp_embed else ret
        ret = (*ret, norm_out, invar) if self.using_post_norm_recompute else ret
        return ret

    def mlp_backward(self, output_grad):
        if self.using_post_norm_recompute:
            if self.send_mtp_embed:
                (
                    inputs_embeds_mtp_grad,
                    hidden_states_grad,
                    residual_grad,
                    l_aux_grad,
                    hidden_states_out_grad,
                    norm_out,
                    invar,
                ) = output_grad
            else:
                hidden_states_grad, residual_grad, l_aux_grad, hidden_states_out_grad, norm_out, invar = output_grad
        else:
            if self.send_mtp_embed:
                (
                    inputs_embeds_mtp_grad,
                    hidden_states_grad,
                    residual_grad,
                    l_aux_grad,
                    hidden_states_out_grad,
                ) = output_grad
            else:
                hidden_states_grad, residual_grad, l_aux_grad, hidden_states_out_grad = output_grad
        hs_dispatched_grad, dispatched_probs_grad = self.fp8_fusion_moe_node.mlp_node.backward(hidden_states_out_grad)

        ret = (hidden_states_grad, residual_grad, l_aux_grad, hs_dispatched_grad, dispatched_probs_grad)
        ret = (inputs_embeds_mtp_grad, *ret) if self.send_mtp_embed else ret
        ret = (*ret, norm_out, invar) if self.using_post_norm_recompute else ret
        return ret

    def dispatch_backward(self, output_grad, async_finish=False, previous_event=None, allocate_on_comm_stream=False):
        if self.using_post_norm_recompute:
            if self.send_mtp_embed:
                (
                    inputs_embeds_mtp_grad,
                    hidden_states_grad,
                    residual_grad,
                    l_aux_grad,
                    hs_dispatched_grad,
                    dispatched_probs_grad,
                    norm_out,
                    invar,
                ) = output_grad
            else:
                (
                    hidden_states_grad,
                    residual_grad,
                    l_aux_grad,
                    hs_dispatched_grad,
                    dispatched_probs_grad,
                    norm_out,
                    invar,
                ) = output_grad
        else:
            if self.send_mtp_embed:
                (
                    inputs_embeds_mtp_grad,
                    hidden_states_grad,
                    residual_grad,
                    l_aux_grad,
                    hs_dispatched_grad,
                    dispatched_probs_grad,
                ) = output_grad
            else:
                hidden_states_grad, residual_grad, l_aux_grad, hs_dispatched_grad, dispatched_probs_grad = output_grad

        hs_grad, token_probs_grad = self.fp8_fusion_moe_node.dispatch_node.backward(
            hs_dispatched_grad,
            dispatched_probs_grad,
            async_finish=async_finish,
            previous_event=previous_event,
            allocate_on_comm_stream=allocate_on_comm_stream and previous_event is not None,
        )

        ret = (hidden_states_grad, residual_grad, l_aux_grad, hs_grad, token_probs_grad)
        ret = (inputs_embeds_mtp_grad, *ret) if self.send_mtp_embed else ret
        ret = (*ret, norm_out, invar) if self.using_post_norm_recompute else ret
        return ret

    def attn_backward(self, output_grad):
        if self.using_post_norm_recompute:
            if self.send_mtp_embed:
                (
                    inputs_embeds_mtp_grad,
                    hidden_states_grad,
                    residual_grad,
                    l_aux_grad,
                    hs_grad,
                    token_probs_grad,
                    norm_out,
                    invar,
                ) = output_grad
                inputs_embeds_mtp_grad_shape = hidden_states_grad.shape
                inputs_embeds_mtp_grad_shape[-1] = -1
                inputs_embeds_mtp_grad = inputs_embeds_mtp_grad.view(inputs_embeds_mtp_grad_shape)
            else:
                hidden_states_grad, residual_grad, l_aux_grad, hs_grad, token_probs_grad, norm_out, invar = output_grad
        else:
            if self.send_mtp_embed:
                (
                    inputs_embeds_mtp_grad,
                    hidden_states_grad,
                    residual_grad,
                    l_aux_grad,
                    hs_grad,
                    token_probs_grad,
                ) = output_grad
                inputs_embeds_mtp_grad_shape = hidden_states_grad.shape
                inputs_embeds_mtp_grad_shape[-1] = -1
                inputs_embeds_mtp_grad = inputs_embeds_mtp_grad.view(inputs_embeds_mtp_grad_shape)
            else:
                hidden_states_grad, residual_grad, l_aux_grad, hs_grad, token_probs_grad = output_grad

        hidden_states_grad_, probs_grad, routing_map_grad = self.fp8_fusion_moe_node.dispatch_quant_node.backward(
            hs_grad, token_probs_grad
        )

        output_grad = (residual_grad, probs_grad, routing_map_grad, l_aux_grad)

        output_grad = (
            (hidden_states_grad, *output_grad, hidden_states_grad_)
            if self.using_post_norm_recompute
            else (hidden_states_grad + hidden_states_grad_, *output_grad)
        )
        output_grad = (inputs_embeds_mtp_grad, *output_grad) if self.send_mtp_embed else output_grad

        if self.using_post_norm_recompute:
            with TemporaryVarContext(norm_out, invar):
                output_grad = self.attn_and_gate_node.backward(output_grad)
        else:
            output_grad = self.attn_and_gate_node.backward(output_grad)
        return output_grad

    def backward_for_fusion(self, output_grad, combine_bw_event_to_wait=None, pp_stream=None):
        paddle.base.core.nvprof_nvtx_push("backward")
        if combine_bw_event_to_wait is None:
            combine_bw_event_to_wait = deep_ep.get_event_from_calc_stream(self.moe_group.id)

        paddle.base.core.nvprof_nvtx_push("post_process_backward")
        output_grad = self.post_process_backward(output_grad, combine_bw_event_to_wait)
        paddle.base.core.nvprof_nvtx_pop()

        paddle.base.core.nvprof_nvtx_push("combine_backward")
        output_grad = self.combine_backward(
            output_grad, previous_event=combine_bw_event_to_wait, async_finish=True, allocate_on_comm_stream=True
        )
        combine_backward_event = deep_ep.get_event_from_comm_stream(self.moe_group.id)
        combine_backward_event.calc_stream_wait(self.moe_group.id)
        paddle.base.core.nvprof_nvtx_pop()

        if WeightGradStore.enabled:
            paddle.base.core.nvprof_nvtx_push("mlp_backward")
            output_grad = self.mlp_backward(output_grad)
            paddle.base.core.nvprof_nvtx_pop()

            paddle.base.core.nvprof_nvtx_push("dispatch_backward")
            output_grad = self.dispatch_backward(output_grad)
            paddle.base.core.nvprof_nvtx_pop()

            paddle.base.core.nvprof_nvtx_push("attn_backward")
            output_grad = self.attn_backward(output_grad)
            paddle.base.core.nvprof_nvtx_pop()

            event_to_wait = None

        else:
            paddle.base.core.nvprof_nvtx_push("mlp_backward_dx")
            assert WeightGradStore.funcs_queue.empty()
            WeightGradStore.enabled = True
            output_grad = self.mlp_backward(output_grad)
            WeightGradStore.enabled = False
            WeightGradStore.flush()
            output_grad_event = deep_ep.get_event_from_calc_stream(self.moe_group.id)
            paddle.base.core.nvprof_nvtx_pop()

            paddle.base.core.nvprof_nvtx_push("dispatch_backward")
            output_grad = self.dispatch_backward(
                output_grad, async_finish=True, previous_event=output_grad_event, allocate_on_comm_stream=True
            )
            dispatch_backward_event = deep_ep.get_event_from_comm_stream(self.moe_group.id)
            paddle.base.core.nvprof_nvtx_pop()

            paddle.base.core.nvprof_nvtx_push("mlp_backward_dw")
            WeightGradStore.pop()
            assert WeightGradStore.funcs_queue.empty()
            paddle.base.core.nvprof_nvtx_pop()

            paddle.base.core.nvprof_nvtx_push("attn_backward_dx")
            dispatch_backward_event.calc_stream_wait(self.moe_group.id)
            WeightGradStore.enabled = True
            output_grad = self.attn_backward(output_grad)
            WeightGradStore.enabled = False
            WeightGradStore.flush()
            event_to_wait = deep_ep.get_event_from_calc_stream(self.moe_group.id)
            paddle.base.core.nvprof_nvtx_pop()

            paddle.base.core.nvprof_nvtx_push("attn_backward_dw")
            WeightGradStore.pop()
            assert WeightGradStore.funcs_queue.empty()
            paddle.base.core.nvprof_nvtx_pop()

        paddle.base.core.nvprof_nvtx_pop()
        return output_grad, event_to_wait

    def forward(self, inputs):
        if self.stepped_recompute_fwd_gate_up:
            self.fp8_fusion_moe_node.mlp_node.set_recompute_fwd_gate_up(True)
        inputs = self.attn_forward(inputs)
        inputs = self.dispatch_forward(inputs)
        inputs = self.mlp_forward(inputs)
        inputs = self.combine_forward(inputs)
        inputs = self.post_process_forward(inputs)
        return inputs

    def backward(self, output_grad=None, scaler=None):
        assert (output_grad is not None) and (scaler is None)
        output_grad = self.post_process_backward(output_grad)
        output_grad = self.combine_backward(output_grad)
        output_grad = self.mlp_backward(output_grad)
        # todo(phlrain): overlap here
        output_grad = self.dispatch_backward(output_grad)
        output_grad = self.attn_backward(output_grad)
        return output_grad


class DenseDecoderLayerNode(ScheduleNode):
    def __init__(
        self,
        attn_node,
        mlp_node,
        name="DenseDecoderLayerNode",
    ):
        super().__init__(fwd_func=None, name=name)
        self.attn_node = attn_node
        self.mlp_node = mlp_node

    def forward(self, inputs):
        inputs = self.attn_node.forward(inputs)
        inputs = self.mlp_node.forward(inputs)
        return inputs

    def backward(self, output_grad=None, scaler=None):
        assert (output_grad is not None) and (scaler is None)
        output_grad = self.mlp_node.backward(output_grad)
        output_grad = self.attn_node.backward(output_grad)
        return output_grad


class OverlapedFUsionScheduleNode:
    def __init__(self, forward_node, backward_node, name=""):
        assert isinstance(forward_node, FusionFp8DecoderLayerNode) and isinstance(
            backward_node, FusionFp8DecoderLayerNode
        )
        self.forward_node = forward_node
        self.backward_node = backward_node
        self.name = name

    def forward_backward(self, inputs, output_grad, combine_bw_event_to_wait=None, pp_stream=None):
        paddle.base.core.nvprof_nvtx_push("forward_backward")

        combine_bwd_event = deep_ep.get_event_from_calc_stream(self.backward_node.moe_group.id)

        paddle.base.core.nvprof_nvtx_push("attn_forward")
        inputs = self.forward_node.attn_forward(inputs)
        paddle.base.core.nvprof_nvtx_pop()
        attn_compute_event = deep_ep.get_event_from_calc_stream(self.forward_node.moe_group.id)

        paddle.base.core.nvprof_nvtx_push("post_process_backward")
        output_grad = self.backward_node.post_process_backward(output_grad, combine_bw_event_to_wait)
        paddle.base.core.nvprof_nvtx_pop()

        paddle.base.core.nvprof_nvtx_push("combine_backward")
        if combine_bw_event_to_wait is not None:
            output_grad = self.backward_node.combine_backward(
                output_grad, previous_event=combine_bw_event_to_wait, async_finish=True, allocate_on_comm_stream=True
            )
        else:
            output_grad = self.backward_node.combine_backward(
                output_grad, previous_event=combine_bwd_event, async_finish=True, allocate_on_comm_stream=True
            )
        # get combine event
        combine_backward_event = deep_ep.get_event_from_comm_stream(self.backward_node.moe_group.id)
        paddle.base.core.nvprof_nvtx_pop()

        combine_backward_event.calc_stream_wait(self.backward_node.moe_group.id)
        paddle.base.core.nvprof_nvtx_push("mlp_backward_dx")
        assert WeightGradStore.funcs_queue.empty()
        WeightGradStore.enabled = True
        output_grad = self.backward_node.mlp_backward(output_grad)
        WeightGradStore.enabled = False
        WeightGradStore.flush()
        paddle.base.core.nvprof_nvtx_pop()

        output_grad_event = deep_ep.get_event_from_calc_stream(self.backward_node.moe_group.id)

        paddle.base.core.nvprof_nvtx_push("dispatch_forward")
        inputs = self.forward_node.dispatch_forward(
            inputs, previous_event=attn_compute_event, async_finish=True, allocate_on_comm_stream=True
        )
        paddle.base.core.nvprof_nvtx_pop()
        dispatch_forward_event = deep_ep.get_event_from_comm_stream(self.forward_node.moe_group.id)

        paddle.base.core.nvprof_nvtx_push("dispatch_backward")
        output_grad = self.backward_node.dispatch_backward(
            output_grad, async_finish=True, previous_event=output_grad_event, allocate_on_comm_stream=True
        )
        paddle.base.core.nvprof_nvtx_pop()
        # get dispatch backward event
        dispatch_backward_event = deep_ep.get_event_from_comm_stream(self.backward_node.moe_group.id)

        paddle.base.core.nvprof_nvtx_push("dispatch_backward_dw")
        WeightGradStore.pop()
        assert WeightGradStore.funcs_queue.empty()
        paddle.base.core.nvprof_nvtx_pop()

        dispatch_forward_event.calc_stream_wait(self.forward_node.moe_group.id)
        paddle.base.core.nvprof_nvtx_push("mlp_forward")
        inputs = self.forward_node.mlp_forward(inputs)
        paddle.base.core.nvprof_nvtx_pop()

        if pp_stream is not None:
            paddle.base.core.nvprof_nvtx_push("post_process_forward")

            final_out = self.forward_node.post_process_node.forward_without_residual(inputs)
            paddle.base.core.nvprof_nvtx_pop()

        final_out_event = deep_ep.get_event_from_calc_stream(self.forward_node.moe_group.id)

        paddle.base.core.nvprof_nvtx_push("combine_forward")
        inputs = self.forward_node.combine_forward(
            inputs, previous_event=final_out_event, async_finish=True, allocate_on_comm_stream=True
        )
        paddle.base.core.nvprof_nvtx_pop()

        combine_forward_event = deep_ep.get_event_from_comm_stream(self.forward_node.moe_group.id)

        combine_fwd_out = inputs[-2] if self.forward_node.using_post_norm_recompute else inputs[-1]

        if pp_stream is not None:
            send_recv_stream = paddle.device.Stream(stream_base=pp_stream)

            paddle.base.core.nvprof_nvtx_push("pp stream add")

            with paddle.device.stream_guard(send_recv_stream):
                combine_forward_event.current_stream_wait()
                final_out_event.current_stream_wait()

                # TODO: check correct
                # if final_out.shape[-1] != combine_fwd_out.shape[-1]:
                #     final_out[:, :, : combine_fwd_out.shape[-1]] += combine_fwd_out  # Directly broadcast and add
                # else:
                #     final_out += combine_fwd_out
                inputs = final_out + combine_fwd_out

                final_out._record_stream()
                combine_fwd_out._record_stream()

            paddle.base.core.nvprof_nvtx_pop()

        dispatch_backward_event.calc_stream_wait(self.backward_node.moe_group.id)

        paddle.base.core.nvprof_nvtx_push("attn_backward")
        assert WeightGradStore.funcs_queue.empty()
        WeightGradStore.enabled = True
        output_grad = self.backward_node.attn_backward(output_grad)
        event_to_wait = deep_ep.get_event_from_calc_stream(self.backward_node.moe_group.id)

        if EventStore is not None:
            EventStore.set(event_to_wait)

        WeightGradStore.enabled = False
        WeightGradStore.flush()
        WeightGradStore.pop()
        assert WeightGradStore.funcs_queue.empty()

        paddle.base.core.nvprof_nvtx_pop()

        # residual add
        if pp_stream is None:
            combine_forward_event.calc_stream_wait(self.forward_node.moe_group.id)

            final_out = self.forward_node.post_process_node.forward_without_residual(inputs)
            if final_out.shape[-1] != combine_fwd_out.shape[-1]:
                final_out[:, :, : combine_fwd_out.shape[-1]] += combine_fwd_out
            else:
                final_out += combine_fwd_out
            inputs = final_out
            combine_fwd_out._record_stream()

        paddle.base.core.nvprof_nvtx_pop()
        return inputs, output_grad, event_to_wait


class OverlapedDenseFusionScheduleNode:
    def __init__(self, forward_node, backward_node, name=""):
        assert isinstance(forward_node, FusionFp8DecoderLayerNode) or isinstance(
            backward_node, FusionFp8DecoderLayerNode
        )
        assert isinstance(forward_node, DenseDecoderLayerNode) or isinstance(backward_node, DenseDecoderLayerNode)
        self.forward_node = forward_node
        self.backward_node = backward_node
        self.name = name

    def forward_backward(self, inputs, output_grad, combine_bw_event_to_wait=None, pp_stream=None):
        # Dense forward + MoE backward
        if isinstance(self.forward_node, DenseDecoderLayerNode):
            paddle.base.core.nvprof_nvtx_push("dense_fw_moe_bw")

            paddle.base.core.nvprof_nvtx_push("dense_attn_moe_combine")
            # Note: the input combine_bw_event_to_wait is unreliable, we need to record a new event here.
            combine_bw_event_to_wait = deep_ep.get_event_from_calc_stream(self.backward_node.moe_group.id)
            output_grad = self.backward_node.post_process_backward(output_grad, combine_bw_event_to_wait)
            output_grad = self.backward_node.combine_backward(
                output_grad, previous_event=combine_bw_event_to_wait, async_finish=True, allocate_on_comm_stream=True
            )
            combine_bw_event = deep_ep.get_event_from_comm_stream(self.backward_node.moe_group.id)
            inputs = self.forward_node.attn_node.forward(inputs)
            combine_bw_event.calc_stream_wait(self.backward_node.moe_group.id)
            paddle.base.core.nvprof_nvtx_pop()  # dense_attn_moe_combine

            paddle.base.core.nvprof_nvtx_push("moe_mlp")
            assert WeightGradStore.funcs_queue.empty()
            WeightGradStore.enabled = True
            output_grad = self.backward_node.mlp_backward(output_grad)
            WeightGradStore.enabled = False
            WeightGradStore.flush()
            paddle.base.core.nvprof_nvtx_pop()  # moe_mlp

            paddle.base.core.nvprof_nvtx_push("dense_mlp_moe_dispatch")
            output_grad = self.backward_node.dispatch_backward(
                output_grad, async_finish=True, allocate_on_comm_stream=True
            )
            dispatch_bw_event = deep_ep.get_event_from_comm_stream(self.backward_node.moe_group.id)
            paddle.base.core.nvprof_nvtx_push("dispatch_backward_dw")
            WeightGradStore.pop()
            assert WeightGradStore.funcs_queue.empty()
            paddle.base.core.nvprof_nvtx_pop()
            inputs = self.forward_node.mlp_node.forward(inputs)
            dispatch_bw_event.calc_stream_wait(self.backward_node.moe_group.id)
            paddle.base.core.nvprof_nvtx_pop()  # dense_mlp_moe_dispatch

            paddle.base.core.nvprof_nvtx_push("moe_attn")
            output_grad = self.backward_node.attn_backward(output_grad)
            paddle.base.core.nvprof_nvtx_pop()  # moe_attn

            event_to_wait = deep_ep.get_event_from_calc_stream(self.backward_node.moe_group.id)
            paddle.base.core.nvprof_nvtx_pop()  # dense_fw_moe_bw

        # Dense backward + MoE forward
        else:
            paddle.base.core.nvprof_nvtx_push("dense_bw_moe_fw")

            paddle.base.core.nvprof_nvtx_push("moe_attn")
            inputs = self.forward_node.attn_forward(inputs)
            attn_fw_event = deep_ep.get_event_from_calc_stream(self.forward_node.moe_group.id)
            paddle.base.core.nvprof_nvtx_pop()  # moe_attn

            paddle.base.core.nvprof_nvtx_push("dense_mlp_moe_dispatch")
            if combine_bw_event_to_wait is not None:
                combine_bw_event_to_wait.calc_stream_wait(self.forward_node.moe_group.id)
            output_grad = self.backward_node.mlp_node.backward(output_grad)
            inputs = self.forward_node.dispatch_forward(
                inputs, previous_event=attn_fw_event, async_finish=True, allocate_on_comm_stream=True
            )
            dispatch_fw_event = deep_ep.get_event_from_comm_stream(self.forward_node.moe_group.id)
            dispatch_fw_event.calc_stream_wait(self.forward_node.moe_group.id)
            paddle.base.core.nvprof_nvtx_pop()  # dense_mlp_moe_dispatch

            paddle.base.core.nvprof_nvtx_push("moe_mlp")
            inputs = self.forward_node.mlp_forward(inputs)
            paddle.base.core.nvprof_nvtx_pop()  # moe_mlp

            paddle.base.core.nvprof_nvtx_push("dense_attn_moe_combine")
            inputs = self.forward_node.combine_forward(inputs, async_finish=True, allocate_on_comm_stream=True)
            combine_fw_event = deep_ep.get_event_from_comm_stream(self.forward_node.moe_group.id)
            output_grad = self.backward_node.attn_node.backward(output_grad)
            combine_fw_event.calc_stream_wait(self.forward_node.moe_group.id)
            paddle.base.core.nvprof_nvtx_pop()  # dense_attn_moe_combine

            paddle.base.core.nvprof_nvtx_push("moe_post")
            inputs = self.forward_node.post_process_forward(inputs)
            paddle.base.core.nvprof_nvtx_pop()  # moe_post

            event_to_wait = deep_ep.get_event_from_calc_stream(self.forward_node.moe_group.id)
            paddle.base.core.nvprof_nvtx_pop()  # dense_bw_moe_fw

        return inputs, output_grad, event_to_wait


def build_overlapped_nodes(config: DeepseekV2FastConfig, forward_chunk, backward_chunk):
    overlap_element_class = (
        FusionFp8DecoderLayerNode if config.dsv3_use_fp8_gemm else DecoderLayerNode,
        DenseDecoderLayerNode,
    )
    forward_decoder_layer_num = 0
    backward_decoder_layer_num = 0
    assert isinstance(forward_chunk, ScheduleChunk) and isinstance(backward_chunk, ScheduleChunk)
    for n in forward_chunk.nodes:
        if isinstance(n, overlap_element_class):
            forward_decoder_layer_num += 1
    for n in reversed(backward_chunk.nodes):
        if isinstance(n, overlap_element_class):
            backward_decoder_layer_num += 1

    overlap_layers_num = min(forward_decoder_layer_num, backward_decoder_layer_num)
    forward_pre_overlap_layers = []
    forward_post_overlap_layers = []
    forward_overlap_layers = []
    is_pre = True
    for n in forward_chunk.nodes:
        if not isinstance(n, overlap_element_class):
            if is_pre:
                forward_pre_overlap_layers.append(n)
            else:
                forward_post_overlap_layers.append(n)
        else:
            is_pre = False
            if len(forward_overlap_layers) == overlap_layers_num:
                forward_post_overlap_layers.append(n)
            else:
                forward_overlap_layers.append(n)
    forward_pre_node = ScheduleChunk(forward_pre_overlap_layers)
    forward_post_node = ScheduleChunk(forward_post_overlap_layers)

    backward_pre_overlap_layers = []
    backward_post_overlap_layers = []
    backward_overlap_layers = []
    is_pre = True
    for n in reversed(backward_chunk.nodes):
        if not isinstance(n, overlap_element_class):
            if is_pre:
                backward_pre_overlap_layers.append(n)
            else:
                backward_post_overlap_layers.append(n)
        else:
            is_pre = False
            if len(backward_overlap_layers) == overlap_layers_num:
                backward_post_overlap_layers.append(n)
            else:
                backward_overlap_layers.append(n)

    backward_pre_node = ScheduleChunk(list(reversed(backward_pre_overlap_layers)))
    backward_post_node = ScheduleChunk(list(reversed(backward_post_overlap_layers)))

    if not forward_chunk.nodes and all(isinstance(n, FusionFp8DecoderLayerNode) for n in backward_chunk.nodes):
        backward_post_node = DecoderBackwardScheduleChunk(backward_post_overlap_layers)

    overlap_node = OverlapedScheduleChunk(
        forward_overlap_layers, backward_overlap_layers, use_fuion=config.dsv3_use_fp8_gemm
    )
    return forward_pre_node, backward_pre_node, overlap_node, forward_post_node, backward_post_node


class EmbeddingFunction(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, x, weight):
        out = paddle.nn.functional.embedding(
            x, weight=weight, padding_idx=None, max_norm=None, norm_type=2.0, sparse=False, scale_grad_by_freq=False
        )

        ctx.save_for_backward(x, weight)
        return out

    @staticmethod
    def backward(ctx, dout):
        x, weight = ctx.saved_tensor()

        if hasattr(weight, "main_grad"):
            paddle.incubate.nn.functional.embedding_grad_add_to_(x, weight.main_grad, dout)
        else:
            paddle.incubate.nn.functional.embedding_grad_add_to_(x, weight.grad, dout)

        return None, None


class DeepseekV2EmbeddingPipe(nn.Layer):
    def __init__(self, config: DeepseekV2FastConfig):
        super(DeepseekV2EmbeddingPipe, self).__init__()
        self.config = config
        self.sequence_parallel = config.sequence_parallel
        self.hidden_size = config.hidden_size
        if config.tensor_model_parallel_size > 1 and config.vocab_size % config.tensor_model_parallel_size == 0:
            self.embed_tokens = fleet.meta_parallel.VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                weight_attr=paddle.ParamAttr(initializer=nn.initializer.XavierNormal()),
            )
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

    @property
    def embedding_weight(self):
        return get_attr(self.embed_tokens, "weight")

    def forward(self, args):
        """_summary_

        Args:
            input (_type_): _description_

        Returns:
            _type_: _description_
        """
        input_ids, attention_mask, attn_mask_startend_row_indices, position_ids = parse_args(args)
        inputs_embeds = EmbeddingFunction.apply(input_ids, self.embed_tokens.weight)

        batch_size, seq_length = input_ids.shape
        if self.config.num_nextn_predict_layers > 0:
            seq_length -= self.config.num_nextn_predict_layers

            if attention_mask is not None:
                attention_mask = attention_mask[
                    :, :, : -self.config.num_nextn_predict_layers, : -self.config.num_nextn_predict_layers
                ]

        if attention_mask is not None:
            assert (
                attn_mask_startend_row_indices is None
            ), "attention_mask and attn_mask_startend_row_indices can not be set at same time"

            attention_mask = DeepseekV2Model._prepare_decoder_attention_mask(
                attention_mask, (batch_size, seq_length), 0, inputs_embeds.dtype
            )
            attention_mask.stop_gradient = True
            if get_env_device() == "npu":
                attention_mask = attention_mask.astype("bool")
        elif get_env_device() == "npu":
            attention_mask = paddle.tril(paddle.ones((seq_length, seq_length), dtype="bool"))
            attention_mask.stop_gradient = True

        if self.config.num_nextn_predict_layers > 0:
            inputs_embeds_extra = inputs_embeds[:, -self.config.num_nextn_predict_layers :, :]  # [B, S, D]
            inputs_embeds = inputs_embeds[:, : -self.config.num_nextn_predict_layers, :]
            inputs_embeds_ori = inputs_embeds
            batch_size, seq_length, _ = inputs_embeds.shape

            if self.sequence_parallel:
                # [bs, seq_len, num_head * head_dim] -> [bs * seq_len, num_head * head_dim]
                inputs_embeds = paddle.reshape(inputs_embeds, [-1, inputs_embeds.shape[-1]])
                # [seq_len * bs / n, num_head * head_dim] (n is mp parallelism)
                inputs_embeds = ScatterOp.apply(inputs_embeds)
            embeds_res = [inputs_embeds]
            mtp_embeds = []
            for depth in range(self.config.num_nextn_predict_layers):
                inputs_embeds_mtp = paddle.cat(
                    [
                        inputs_embeds_ori[:, (depth + 1) :, :],
                        inputs_embeds_extra[:, : (depth + 1), :],
                    ],
                    axis=1,
                )
                if self.sequence_parallel:
                    inputs_embeds_mtp = inputs_embeds_mtp.reshape([-1, inputs_embeds_mtp.shape[-1]])
                    inputs_embeds_mtp = ScatterOp.apply(inputs_embeds_mtp)
                mtp_embeds.append(inputs_embeds_mtp)

            if self.config.send_mtp_embed:
                embeds_res.extend(mtp_embeds)
                # if not self.sequence_parallel
                # mtp_embeds: [B*num_nextn_predict_layers, seq_len, hidden_size]
                # else:
                # mtp_embeds: [B*seq_len*num_nextn_predict_layers, hidden_size]
                inputs_embeds = paddle.cat(embeds_res, axis=-1)
            else:
                global global_inputs_embeds_mtp_queue
                cloned_mtp_embeds = [t.detach() for t in mtp_embeds]
                global_inputs_embeds_mtp_queue.put(cloned_mtp_embeds)
            return return_args(inputs_embeds, attention_mask, attn_mask_startend_row_indices, position_ids)
        else:
            if self.sequence_parallel:
                inputs_embeds = inputs_embeds.reshape([-1, inputs_embeds.shape[-1]])
                inputs_embeds = ScatterOp.apply(inputs_embeds)
            return return_args(inputs_embeds, attention_mask, attn_mask_startend_row_indices, position_ids)

    def build_schedule_node(self):
        return ScheduleNode(self.forward, name="DeepseekV2EmbeddingPipe")


class DeepseekV2DecoderLayerPipe(DeepseekV2DecoderLayer):
    def forward(self, args):
        hidden_states, attention_mask, attn_mask_startend_row_indices, position_ids = parse_args(args)

        if self.config.send_mtp_embed:
            batch_size, _, hidden_size = hidden_states.shape
            batch_size_mtp = hidden_size // (self.config.num_nextn_predict_layers + 1)
            inputs_embeds_mtp = hidden_states[..., batch_size_mtp:]
            hidden_states = hidden_states[..., :batch_size_mtp]

        has_gradient = not hidden_states.stop_gradient

        if attention_mask is not None and attention_mask.dtype == paddle.int32:
            attention_mask, attn_mask_startend_row_indices, position_ids = (
                None,
                attention_mask,
                attn_mask_startend_row_indices,
            )
        elif attention_mask is not None and attention_mask.dtype == paddle.int64:
            attention_mask, attn_mask_startend_row_indices, position_ids = None, None, attention_mask
        elif attn_mask_startend_row_indices is not None and attn_mask_startend_row_indices.dtype == paddle.int64:
            attn_mask_startend_row_indices, position_ids = None, attn_mask_startend_row_indices

        if (
            self.config.recompute_granularity == "full"
            and self.config.recompute_method == "uniform"
            and self.config.recompute_num_layers == 1
            and has_gradient
        ):
            if attention_mask is not None or attn_mask_startend_row_indices is not None:
                hidden_states = recompute(
                    super().forward,
                    hidden_states,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                    use_reentrant=False,
                )
            else:
                # for pretrain
                hidden_states = recompute(
                    super().forward,
                    hidden_states,
                    position_ids=position_ids,
                    attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                    use_reentrant=self.config.recompute_use_reentrant,
                )
        else:
            hidden_states = super().forward(
                hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            )

        if self.config.send_mtp_embed:
            hidden_states = paddle.cat([hidden_states, inputs_embeds_mtp], axis=-1)

        return return_args(hidden_states, attention_mask, attn_mask_startend_row_indices, position_ids)

    def attn_compute(self, args):
        hidden_states, attention_mask, attn_mask_startend_row_indices, position_ids = parse_args(args)
        assert attention_mask is None
        assert attn_mask_startend_row_indices is None
        assert position_ids is None
        assert self.config.send_mtp_embed

        batch_size, _, hidden_size = hidden_states.shape
        batch_size_mtp = hidden_size // (self.config.num_nextn_predict_layers + 1)
        inputs_embeds_mtp = hidden_states[..., batch_size_mtp:]
        hidden_states = hidden_states[..., :batch_size_mtp]

        def attn_compute_func(hidden_states):
            hidden_states, residual = self.self_attn_compute(hidden_states)
            l_aux, _, intermediate_hidden_states, token_indices, token_probs = self.pre_dispatch_compute(hidden_states)
            return (hidden_states, residual, l_aux, intermediate_hidden_states, token_indices, token_probs)

        has_gradient = not hidden_states.stop_gradient
        if (
            self.config.recompute_granularity == "full"
            and self.config.recompute_method == "uniform"
            and self.config.recompute_num_layers == 1
            and has_gradient
        ):
            # for pretrain
            outputs = recompute(
                attn_compute_func,
                hidden_states,
                use_reentrant=self.config.recompute_use_reentrant,
            )
        else:
            outputs = attn_compute_func(hidden_states)

        return (inputs_embeds_mtp, *outputs)

    def attn_compute_for_fusion(self, args):
        hidden_states, attention_mask, attn_mask_startend_row_indices, position_ids = parse_args(args)
        assert attention_mask is None
        assert attn_mask_startend_row_indices is None
        assert position_ids is None

        send_mtp_embed = self.config.send_mtp_embed

        if send_mtp_embed:
            # slice from holy tensor
            batch_size, _, hidden_size = hidden_states.shape
            batch_size_mtp = hidden_size // (self.config.num_nextn_predict_layers + 1)
            inputs_embeds_mtp = hidden_states[..., batch_size_mtp:]
            hidden_states = hidden_states[..., :batch_size_mtp]

        hidden_states, residual = self.self_attn_compute(hidden_states)
        _, _, d_model = hidden_states.shape

        if self.using_post_norm_recompute:
            probs, routing_map, l_aux, _, norm_out = self.mlp.router(hidden_states)
        else:
            probs, routing_map, l_aux, _ = self.mlp.router(hidden_states)

        # common return values
        ret = (
            hidden_states,
            residual,
            probs,
            routing_map,
            l_aux,
        )
        # append mtp embed if needed
        ret = (inputs_embeds_mtp, *ret) if send_mtp_embed else ret
        # append norm_out if using post_norm recompute
        ret = (*ret, norm_out) if self.using_post_norm_recompute else ret

        return ret

    def mlp_compute(self, inputs):
        if isinstance(inputs, list):
            inputs = tuple(inputs)
        send_mtp_embed = self.config.send_mtp_embed

        if send_mtp_embed:
            (
                inputs_embeds_mtp,
                hidden_states,
                residual,
                l_aux,
                intermediate_hidden_states,
                dispatched_indices,
                dispatched_probs,
            ) = inputs
        else:
            (
                hidden_states,
                residual,
                l_aux,
                intermediate_hidden_states,
                dispatched_indices,
                dispatched_probs,
            ) = inputs
        has_gradient = not intermediate_hidden_states.stop_gradient
        if (
            self.config.recompute_granularity == "full"
            and self.config.recompute_method == "uniform"
            and self.config.recompute_num_layers == 1
            and has_gradient
        ):
            expert_output = recompute(
                self.expert_forward_compute,
                intermediate_hidden_states,
                dispatched_indices,
                dispatched_probs,
                use_reentrant=self.config.recompute_use_reentrant,
            )
        else:
            expert_output = self.expert_forward_compute(
                intermediate_hidden_states, dispatched_indices, dispatched_probs
            )
        if send_mtp_embed:
            return (inputs_embeds_mtp, hidden_states, residual, l_aux, expert_output)
        else:
            return (hidden_states, residual, l_aux, expert_output)

    def post_process_compute(self, inputs):
        send_mtp_embed = self.config.send_mtp_embed

        if isinstance(inputs, list):
            inputs = tuple(inputs)
        if send_mtp_embed:
            (inputs_embeds_mtp, hidden_states, residual, l_aux, combine_output) = inputs
        else:
            (hidden_states, residual, l_aux, combine_output) = inputs
        has_gradient = not hidden_states.stop_gradient
        if (
            self.config.recompute_granularity == "full"
            and self.config.recompute_method == "uniform"
            and self.config.recompute_num_layers == 1
            and has_gradient
        ):
            hidden_states = recompute(
                self.post_combine_compute,
                residual,
                hidden_states,
                combine_output,
                l_aux,
                use_reentrant=self.config.recompute_use_reentrant,
            )
        else:
            hidden_states = self.post_combine_compute(
                residual,
                hidden_states,
                combine_output,
                l_aux,
            )
        if send_mtp_embed:
            hidden_states = paddle.cat([hidden_states, inputs_embeds_mtp], axis=-1)

        return return_args(hidden_states)

    def post_process_compute_for_fusion(self, inputs):
        send_mtp_embed = self.config.send_mtp_embed

        if isinstance(inputs, list):
            inputs = tuple(inputs)

        if send_mtp_embed:
            (inputs_embeds_mtp, hidden_states, residual, l_aux, final_hidden_states) = inputs
        else:
            (hidden_states, residual, l_aux, final_hidden_states) = inputs

        final_hidden_states = self.mlp.post_process(hidden_states, final_hidden_states, l_aux)

        hidden_states = residual + final_hidden_states

        hidden_states = (hidden_states,)

        if type(hidden_states) is tuple and len(hidden_states) == 1:
            hidden_states = hidden_states[0]

        if send_mtp_embed:
            hidden_states = paddle.cat([hidden_states, inputs_embeds_mtp], axis=-1)

        return return_args(hidden_states)

    def attn_compute_dense(self, args):
        hidden_states, attention_mask, attn_mask_startend_row_indices, position_ids = parse_args(args)
        assert attention_mask is None
        assert attn_mask_startend_row_indices is None
        assert position_ids is None

        if self.config.send_mtp_embed:
            batch_size, _, hidden_size = hidden_states.shape
            batch_size_mtp = hidden_size // (self.config.num_nextn_predict_layers + 1)
            inputs_embeds_mtp = hidden_states[..., batch_size_mtp:]
            hidden_states = hidden_states[..., :batch_size_mtp]

        hidden_states, residual = self.self_attn_compute(hidden_states)

        ret = (hidden_states, residual)
        ret = (inputs_embeds_mtp, *ret) if self.config.send_mtp_embed else ret
        return ret

    def mlp_compute_dense(self, inputs):
        if self.config.send_mtp_embed:
            (inputs_embeds_mtp, hidden_states, residual) = inputs
        else:
            (hidden_states, residual) = inputs

        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        if self.config.send_mtp_embed:
            hidden_states = paddle.cat([hidden_states, inputs_embeds_mtp], axis=-1)

        return hidden_states

    def build_schedule_node(self):
        if isinstance(self.mlp, DeepseekV2MoE):
            self.mlp.update_flex_token()
            if self.mlp.using_flex_token:
                if self.config.dsv3_use_fp8_gemm:
                    attn_and_gate_node = ScheduleNode(self.attn_compute_for_fusion, name="attn_and_gate_node")

                    # recompute_fwd_gate_up_ may be 1, 0 or -1. 1 means recompute, 0 means disable recompute, -1 means adaptive recompute.
                    recompute_fwd_gate_up_ = 1 if self.layer_idx in self.config.recompute_fwd_gate_up_list else 0
                    if recompute_fwd_gate_up_ == 0 and self.config.adaptive_remained_O1_recompute_ratio:
                        recompute_fwd_gate_up_ = -1

                    fp8_fusion_moe_node = FusionMoeNode(
                        self.mlp,
                        recompute_fwd_gate_up=recompute_fwd_gate_up_,
                        is_split_group_gemm=self.config.is_split_group_gemm,
                        mlp_fwd_subbatch_rows=self.config.mlp_fwd_subbatch_rows,
                        mlp_bwd_subbatch_rows=self.config.mlp_bwd_subbatch_rows,
                        output_subbatch_rows=self.config.output_subbatch_rows,
                        dsv3_use_fp8_dispatch=self.config.dsv3_use_fp8_dispatch,
                        name="fp8_fusion_moe_node",
                    )
                    post_process_node = PostProcessNode(
                        self.config.send_mtp_embed,
                        self.mlp.training,
                        self.mlp.alpha,
                        self.config,
                        self.mlp.shared_experts,
                        self.config.using_post_norm_recompute,
                        output_mtp_embed_first=isinstance(self, DeepseekV2MTPLayer),
                        name="post_process_node",
                    )
                    return FusionFp8DecoderLayerNode(
                        attn_and_gate_node=attn_and_gate_node,
                        fp8_fusion_moe_node=fp8_fusion_moe_node,
                        post_process_node=post_process_node,
                        mlp_layer=self.mlp,
                        send_mtp_embed=self.config.send_mtp_embed,
                        using_post_norm_recompute=self.config.using_post_norm_recompute,
                        stepped_recompute_fwd_gate_up=self.config.stepped_recompute_fwd_gate_up,
                        dsv3_use_fp8_dispatch=self.config.dsv3_use_fp8_dispatch,
                        name="FusionFp8DecoderLayerNode",
                    )
                else:
                    attn_node = ScheduleNode(self.attn_compute, name="attn_node")
                    mlp_node = ScheduleNode(self.mlp_compute, name="mlp_node")
                    post_process_node = ScheduleNode(self.post_process_compute, name="post_process_node")
                    return DecoderLayerNode(
                        attn_node=attn_node,
                        dispatch_node=None,
                        mlp_node=mlp_node,
                        combine_node=None,
                        post_process_node=post_process_node,
                        mlp_layer=self.mlp,
                        name="DecoderLayerNode",
                    )

        attn_node = ScheduleNode(self.attn_compute_dense, name="attn_node")
        mlp_node = ScheduleNode(self.mlp_compute_dense, name="mlp_node")
        return DenseDecoderLayerNode(
            attn_node=attn_node,
            mlp_node=mlp_node,
            name="DenseDecoderLayerNode",
        )


class DeepseekV2MTPLayerPipe(DeepseekV2MTPLayer):
    def forward(self, args):
        hidden_states, attention_mask, attn_mask_startend_row_indices, position_ids = parse_args(args)

        if self.config.send_mtp_embed:
            hidden_states_list = paddle.split(hidden_states, self.config.num_nextn_predict_layers + 1, axis=-1)
            hidden_states_main_model = hidden_states_list[0]
            inputs_embeds_cur_depth_list = hidden_states_list[1:]
        else:
            hidden_states_main_model = hidden_states
            global global_inputs_embeds_mtp_queue
            inputs_embeds_cur_depth_list = global_inputs_embeds_mtp_queue.get()

        has_gradient = not hidden_states_main_model.stop_gradient

        if attention_mask is not None and attention_mask.dtype == paddle.int32:
            attention_mask, attn_mask_startend_row_indices, position_ids = (
                None,
                attention_mask,
                attn_mask_startend_row_indices,
            )
        elif attention_mask is not None and attention_mask.dtype == paddle.int64:
            attention_mask, attn_mask_startend_row_indices, position_ids = None, None, attention_mask
        elif attn_mask_startend_row_indices is not None and attn_mask_startend_row_indices.dtype == paddle.int64:
            attn_mask_startend_row_indices, position_ids = None, attn_mask_startend_row_indices

        output_list = [hidden_states_main_model]
        hidden_states = hidden_states_main_model
        for depth in range(self.config.num_nextn_predict_layers):
            inputs_embeds_cur_depth = inputs_embeds_cur_depth_list[depth]
            if (
                self.config.recompute_granularity == "full"
                and self.config.recompute_method == "uniform"
                and self.config.recompute_num_layers == 1
                and has_gradient
            ):
                if attention_mask is not None or attn_mask_startend_row_indices is not None:
                    hidden_states = recompute(
                        super().forward,
                        hidden_states,
                        inputs_embeds_cur_depth,
                        position_ids=position_ids,
                        attention_mask=attention_mask,
                        attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                        use_reentrant=False,
                    )
                else:
                    # for pretrain
                    hidden_states = recompute(
                        super().forward,
                        hidden_states,
                        inputs_embeds_cur_depth,
                        position_ids=position_ids,
                        attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                        use_reentrant=self.config.recompute_use_reentrant,
                    )
            else:
                hidden_states = super().forward(
                    hidden_states,
                    inputs_embeds_cur_depth,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                )
            output_list.append(hidden_states)

        hidden_states = paddle.cat(output_list, axis=-1)
        return return_args(hidden_states, attention_mask, attn_mask_startend_row_indices, position_ids)

    def attn_compute_for_fusion(self, args):
        hidden_states, attention_mask, attn_mask_startend_row_indices, position_ids = parse_args(args)
        assert attention_mask is None
        assert attn_mask_startend_row_indices is None
        assert position_ids is None
        assert self.config.num_nextn_predict_layers == 1

        if self.config.send_mtp_embed:
            hidden_states_list = paddle.split(hidden_states, self.config.num_nextn_predict_layers + 1, axis=-1)
            hidden_states_main_model = hidden_states_list[0]
            inputs_embeds_cur_depth_list = hidden_states_list[1:]
        else:
            hidden_states_main_model = hidden_states
            global global_inputs_embeds_mtp_queue
            inputs_embeds_cur_depth_list = global_inputs_embeds_mtp_queue.get()

        hidden_states = hidden_states_main_model
        nextn_hidden_state = inputs_embeds_cur_depth_list[0]

        # mtp compute
        hidden_states = self.hnorm(hidden_states)
        nextn_hidden_state = self.enorm(nextn_hidden_state)

        concat_h = paddle.cat([nextn_hidden_state, hidden_states], axis=-1)
        hidden_states = FP8LinearFunction.apply(concat_h, self.eh_proj)

        # attention compute
        hidden_states, residual = self.self_attn_compute(hidden_states)

        if self.using_post_norm_recompute:
            probs, routing_map, l_aux, _, norm_out = self.mlp.router(hidden_states)
        else:
            probs, routing_map, l_aux, _ = self.mlp.router(hidden_states)

        # common return values
        ret = (
            hidden_states_main_model,
            hidden_states,
            residual,
            probs,
            routing_map,
            l_aux,
        )
        ret = (*ret, norm_out) if self.using_post_norm_recompute else ret

        return ret

    def build_schedule_node(self):
        if isinstance(self.mlp, DeepseekV2MoE):
            self.mlp.update_flex_token()
            if (
                self.mlp.using_flex_token
                and self.config.dsv3_use_fp8_gemm
                and self.config.num_nextn_predict_layers == 1
            ):
                prev_send_mtp_embed = self.config.send_mtp_embed
                self.config.send_mtp_embed = True  # must be True in MTP node

                node = DeepseekV2DecoderLayerPipe.build_schedule_node(self)
                assert isinstance(node, FusionFp8DecoderLayerNode)

                self.config.send_mtp_embed = prev_send_mtp_embed
                return node
        return ScheduleNode(self.forward, name="DeepseekV2MTPLayerPipe")


class DeepseekV2RMSNormPipe(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.norm = DeepseekV2RMSNorm(config)

    def forward(self, args):
        hidden_states, attention_mask, attn_mask_startend_row_indices, position_ids = parse_args(args)

        if self.config.num_nextn_predict_layers > 0:
            hidden_states_list = paddle.split(hidden_states, self.config.num_nextn_predict_layers + 1, axis=-1)
            hidden_states = hidden_states_list[0]
            hidden_states_mtp = hidden_states_list[-self.config.num_nextn_predict_layers :]

            output_list = [self.norm(hidden_states)]
            for hidden_states in hidden_states_mtp:
                output_list.append(self.norm(hidden_states))
            return output_list
        else:
            return self.norm(hidden_states)

    def build_schedule_node(self):
        return ScheduleNode(self.forward, name="DeepseekV2RMSNormPipe")


class DeepseekV2LMHeadPipe(DeepseekV2LMHead):
    def __init__(self, config, embedding_weight=None):
        super(DeepseekV2LMHeadPipe, self).__init__(config, embedding_weight=embedding_weight)

    @property
    def embedding_weight(self):
        return get_attr(self, "weight")

    def forward(self, args: Union[Tuple, paddle.Tensor]):
        if self.config.num_nextn_predict_layers > 0:
            logits = []
            for _hidden_states in args:
                logits.append(super().forward(_hidden_states))
            return logits
        hidden_states = args
        logits = super().forward(hidden_states)
        return logits

    def build_schedule_node(self):
        return ScheduleNode(self.forward, name="DeepseekV2LMHeadPipe")


class DeepseekV2PretrainingCriterionPipe(DeepseekV2PretrainingCriterionFast):
    def forward(self, logits, labels):
        if self.config.num_nextn_predict_layers > 0:
            mtp_logits = logits[1:]
            logits = logits[0]
            loss = super().forward(logits, labels, mtp_logits=mtp_logits)
        else:
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            loss = super().forward(logits, labels)
        return loss

    def build_schedule_node(self):
        return ScheduleNode(self.forward, name="DeepseekV2PretrainingCriterionPipe")


class DeepseekV2ForCausalLMPipe(PipelinePretrainedModel, PipelineLayer):
    """DeepseekV2ForPretraining adapted for pipeline parallelism.

    The largest change is flattening the DeepseekV2Model class so we can express it as a
    sequence of layers including embedding, transformer layers, and output.
    """

    config_class = DeepseekV2FastConfig
    _base_model = DeepseekV2PretrainedModel
    _get_tensor_parallel_mappings = DeepseekV2PretrainedModel._get_tensor_parallel_mappings
    _init_weights = DeepseekV2PretrainedModel._init_weights
    _keys_to_ignore_on_load_unexpected = DeepseekV2PretrainedModel._keys_to_ignore_on_load_unexpected
    _get_model_flops = DeepseekV2PretrainedModel._get_model_flops
    _get_hardware_flops = DeepseekV2PretrainedModel._get_hardware_flops

    _tied_weights_keys = ["lm_head.weight"]

    # DONOT Add base_model_prefix !!!!

    def step_flex_token(self, cur_step):
        set_global_step(cur_step)

    @classmethod
    def _prepare_pipeline_inputs_func(cls, inputs):
        first_stage_keys = ["input_ids", "attention_mask", "attn_mask_startend_row_indices", "position_ids"]
        last_stage_keys = ["labels"]

        def get_expected_keys(inputs, keys):
            ret = tuple([inputs.pop(k) if k in inputs else None for k in keys])
            if len(ret) == 1:
                ret = ret[0]
            return ret

        if type(inputs) is dict or type(inputs) is OrderedDict:
            return [
                get_expected_keys(inputs, first_stage_keys),
                get_expected_keys(inputs, last_stage_keys),
            ]

        keys = list(inputs[0].keys())
        inputs_batch = {key: [data.pop(key) for data in inputs] for key in keys}
        return [
            get_expected_keys(inputs_batch, first_stage_keys),
            get_expected_keys(inputs_batch, last_stage_keys),
        ]

    def __init__(self, config: DeepseekV2FastConfig):
        self.config = config

        # Note that we will actually perform a recompute only if both enable_recompute and layerwise_recompute are set to True
        # Enable_recompute defaults to False and is controlled by Trainer
        self.enable_recompute = False
        self.recompute_granularity = self.config.recompute_granularity
        self.pp_recompute_interval = 1
        self.no_recompute_layers = config.no_recompute_layers if config.no_recompute_layers is not None else []
        if self.recompute_granularity == "full":
            assert len(self.no_recompute_layers) == 0, "for pp with full recompute, no_recompute_layers is not support"

        virtual_pipeline_model_parallel_size = getattr(self.config, "virtual_pipeline_model_parallel_size", 1)
        use_dualpipev = getattr(self.config, "use_dualpipev", False)
        if use_dualpipev:
            assert LocalSharedLayerDesc is not None, "LocalSharedLayerDesc is None, please update your paddle."
        shared_class = LocalSharedLayerDesc if use_dualpipev else SharedLayerDesc

        def get_hcg():
            return fleet.get_hybrid_communicate_group()

        hcg = get_hcg()
        tensor_model_parallel_size = max(hcg.get_model_parallel_world_size(), 1)
        tensor_parallel_rank = max(hcg.get_model_parallel_rank(), 0)

        # TODO: fix tensor_model_parallel_size rewrite in here
        config.tensor_model_parallel_size = tensor_model_parallel_size
        config.tensor_parallel_rank = tensor_parallel_rank

        if config.tie_word_embeddings:
            self.add_sequential_layer(
                shared_class(
                    "DeepseekV2_shared_weight",
                    DeepseekV2EmbeddingPipe,
                    shared_weight_attr="embedding_weight",
                    config=config,
                ),
                self._base_model.base_model_prefix,
            )
        else:
            self.add_sequential_layer(
                LayerDesc(DeepseekV2EmbeddingPipe, config=config), self._base_model.base_model_prefix
            )

        def compute_recompute_fwd_gate_up_list(pp_nums, all_dl_nums, dense_dl_nums, recompute_fwd_gate_up):
            all_layers_nums = all_dl_nums + 4  # embedding, rms, lm_head, mtp
            segment_size = all_layers_nums // pp_nums
            boundary = math.ceil((1 + dense_dl_nums) / segment_size) * segment_size
            recompute_fwd_gate_up_list = [dense_dl_nums]
            for idx in range(boundary - 1, all_dl_nums, segment_size):
                recompute_fwd_gate_up_list.append(idx)

            # If `recompute_fwd_gate_up` is a Boolean value and is True, means all O1 will be recomputed.
            # Otherwise `recompute_fwd_gate_up` should be an integer representing how many O1 are recomputed.
            assert isinstance(recompute_fwd_gate_up, (int, bool))
            if type(recompute_fwd_gate_up) is bool:
                enable_k_o1_rc = segment_size if recompute_fwd_gate_up is True else 0
            else:
                enable_k_o1_rc = recompute_fwd_gate_up

            ret = []
            for i in range(len(recompute_fwd_gate_up_list)):
                for k in range(min(segment_size, enable_k_o1_rc)):
                    ret.append(recompute_fwd_gate_up_list[i] + k)
            return ret

        def compute_recompute_fa3_list(pp_nums, all_dl_nums, recompute_fa3):
            all_layers_nums = all_dl_nums + 4  # embedding, rms, lm_head, mtp
            segment_size = all_layers_nums // pp_nums
            recompute_fa3_list = [0]
            for idx in range(segment_size - 1, all_dl_nums, segment_size):
                recompute_fa3_list.append(idx)

            # If `recompute_fa3` is a Boolean value and is True, means all O1 will be recomputed.
            # Otherwise `recompute_fa3` should be an integer representing how many O1 are recomputed.
            assert isinstance(recompute_fa3, (int, bool))
            if type(recompute_fa3) is bool:
                enable_k_o1_rc = segment_size if recompute_fa3 is True else 0
            else:
                enable_k_o1_rc = recompute_fa3

            ret = []
            for i in range(len(recompute_fa3_list)):
                for k in range(min(segment_size, enable_k_o1_rc)):
                    ret.append(recompute_fa3_list[i] + k)
            return ret

        pp_nums = (
            self.config["pipeline_model_parallel_size"] * 2
            if self.config.use_dualpipev
            else self.config["pipeline_model_parallel_size"]
        )
        recompute_fwd_gate_up_list = compute_recompute_fwd_gate_up_list(
            pp_nums,
            self.config.num_hidden_layers,
            self.config.first_k_dense_replace,
            self.config.recompute_fwd_gate_up,
        )
        recompute_fa3_list = compute_recompute_fa3_list(
            pp_nums, self.config.num_hidden_layers, self.config.recompute_fa3
        )

        logger.info(f"recompute_fa3_list: {recompute_fa3_list}")
        logger.info(f"recompute_fwd_gate_up_list: {recompute_fwd_gate_up_list}")
        config.recompute_fwd_gate_up_list = recompute_fwd_gate_up_list

        for i in range(config.num_hidden_layers):
            self.add_sequential_layer(
                LayerDesc(
                    DeepseekV2DecoderLayerPipe,
                    config=config,
                    layer_idx=i,
                    layerwise_recompute=i not in self.no_recompute_layers,
                    recompute_fa3=i in recompute_fa3_list,
                ),
                f"{self._base_model.base_model_prefix}.layers.{i}",
            )
        for i in range(config.num_nextn_predict_layers):
            self.add_sequential_layer(
                LayerDesc(DeepseekV2MTPLayerPipe, config=config, layer_idx=config.num_hidden_layers + i),
                f"{self._base_model.base_model_prefix}.layers.{config.num_hidden_layers + i}",
            )

        self.add_sequential_layer(LayerDesc(DeepseekV2RMSNormPipe, config=config), self._base_model.base_model_prefix)

        if config.tie_word_embeddings:
            self.add_sequential_layer(
                shared_class(
                    "DeepseekV2_shared_weight",
                    DeepseekV2LMHeadPipe,
                    shared_weight_attr="embedding_weight",
                    config=config,
                    **{"transpose_y": True},
                ),
                "lm_head",
            )
        else:
            self.add_sequential_layer(LayerDesc(DeepseekV2LMHeadPipe, config=config), "lm_head")

        recompute_interval = 0

        seg_method = "layer:DeepseekV2DecoderLayer|DeepseekV2MTPLayerPipe"
        if config.num_hidden_layers % get_hcg().topology().get_dim_size("pipe") != 0:
            seg_method = "uniform"

        PipelineLayer.__init__(
            self,
            layers=self.get_sequential_layers(),
            loss_fn=self.get_loss_fn(config),
            topology=get_hcg().topology(),
            seg_method=seg_method,
            recompute_interval=recompute_interval,
            recompute_ctx={
                "mp_group": get_hcg().get_model_parallel_group(),
                "offload": False,
                "partition": False,
            },
            num_virtual_pipeline_stages=virtual_pipeline_model_parallel_size,
            use_dualpipev=use_dualpipev,
        )
        # You should call init here, since there is a  diamond inheritance problem
        self.apply(self._init_weights)
        # DON'T init PipelinePretrainedModel
        # PipelinePretrainedModel.__init__(self.super(), config=config)

    def fp8_quant_weight(self, batch_mode=False, quant_transpose=True):
        """fp8_quant_weight"""
        with paddle.no_grad():
            for i, layer in self._sub_layers.items():
                if isinstance(
                    layer, paddle.distributed.fleet.meta_parallel.parallel_layers.pp_layers.PipelineLayerChunk
                ):
                    for i, sub_layer in layer.named_sublayers():
                        if isinstance(sub_layer, DeepseekV2DecoderLayer) and hasattr(sub_layer, "fp8_quant_weight"):
                            sub_layer.fp8_quant_weight(batch_mode, quant_transpose)
                if isinstance(layer, DeepseekV2DecoderLayer) and hasattr(layer, "fp8_quant_weight"):
                    layer.fp8_quant_weight(batch_mode, quant_transpose)

    def get_loss_fn(self, config):
        return DeepseekV2PretrainingCriterionPipe(config)

    def overlapped_forward_backward(
        self,
        forward_chunk,  # the module of the forward chunk
        forward_inputs,
        forward_loss_fn_node,
        backward_chunk,  # the module of the backward chunk, maybe not used
        backward_loss_fn_node,
        backward_input_grads,
        scaler,
        combine_bw_event_to_wait=None,
        pp_stream=None,
    ):
        if backward_loss_fn_node is not None:
            if scaler:
                backward_input_grads = backward_loss_fn_node.backward(scaler=scaler)
            else:
                backward_input_grads = backward_loss_fn_node.backward()

        (
            forward_pre_node,
            backward_pre_node,
            overlap_node,
            forward_post_node,
            backward_post_node,
        ) = build_overlapped_nodes(self.config, forward_chunk, backward_chunk)
        forward_inputs = forward_pre_node.forward(forward_inputs)
        backward_input_grads = backward_pre_node.backward(backward_input_grads)
        forward_inputs, backward_input_grads, _ = overlap_node.forward_backward(
            forward_inputs,
            backward_input_grads,
            combine_bw_event_to_wait=combine_bw_event_to_wait,
            pp_stream=pp_stream,
        )
        forward_inputs = forward_post_node.forward(forward_inputs)
        backward_input_grads = backward_post_node.backward(backward_input_grads)

        if forward_loss_fn_node is not None:
            forward_loss = forward_loss_fn_node.forward(forward_inputs)
        else:
            forward_loss = None

        forward_inputs = [forward_inputs] if isinstance(forward_inputs, paddle.Tensor) else forward_inputs
        return forward_inputs, forward_loss, backward_input_grads
