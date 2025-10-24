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

"""_summary_

Returns:
    _type_: _description_
"""

import inspect
import itertools
from typing import List, Optional, Tuple

import paddle
import paddle.distributed as dist
import paddle.nn.functional as F
from paddle import Tensor, _C_ops, framework, nn
from paddle.autograd import PyLayer
from paddle.distributed import fleet
from paddle.distributed.communication.group import Group
from paddle.distributed.fleet.utils import recompute

from paddleformers.utils.log import logger

try:
    from paddle.incubate.nn.functional import moe_combine, moe_gate_dispatch
except ImportError:
    logger.warning_once("Fail to import moe_combine and moe_gate_dispatch.")
    moe_combine = None
    moe_gate_dispatch = None

from .abstract import MOELayerBase
from .all_to_all import AlltoAll, AlltoAllAsync
from .utils import ScatterOp


class GateCombine(PyLayer):
    """
    Custom PyLayer for gate combination operations with backward pass.
    """

    @staticmethod
    def forward(ctx, x, combine_weights, scatter_index):
        """
        Forward pass for gate combination.

        Args:
            x: Input tensor
            combine_weights: Combination weights
            scatter_index: Scatter indices

        Returns:
            Tensor: Combined output
        """
        ctx.x = x
        ctx.combine_weights = combine_weights
        ctx.scatter_index = scatter_index
        ret = moe_combine(x, combine_weights, scatter_index)
        return ret

    @staticmethod
    def backward(ctx, grad_y, *_):
        """
        Backward pass for gate combination.

        Args:
            grad_y: Gradient of output [seqlen, hidden_size]

        Returns:
            tuple: (grad_x, grad_combine_weight, None)
        """
        grad_x, grad_combine_weight_helper = _C_ops.moe_combine_grad(
            ctx.x, ctx.combine_weights, ctx.scatter_index, grad_y
        )
        # grad_combine_weight_helper is the same shape with grad x [seqlen * K, dim]
        # reduce the hidden shape
        # TODO: implement reduce in cuda ops
        grad_combine_weight = grad_combine_weight_helper.sum(-1)
        return grad_x, grad_combine_weight.reshape(ctx.combine_weights.shape), None


def combining(x, combine_weights, scatter_index, hard_gate=False):
    """
    Fused version of combining operation.

    Args:
        x: Input tensor [seq, dim]
        combine_weights: Combination weights [s, k]
        scatter_index: Scatter indices [k, s]
        hard_gate: Whether to use hard gating

    Returns:
        Tensor: Combined output [s, dim]
    """
    if hard_gate:
        x_gatherd = F.embedding(scatter_index, x)  # [s,k,dim]
        return x_gatherd.squeeze(-2)
    if paddle.device.is_compiled_with_custom_device("npu"):
        from ernie.fusion_ops.npu_fusion_ops import npu_combining

        ret = npu_combining(x, combine_weights, scatter_index)
    else:
        ret = GateCombine.apply(x, combine_weights, scatter_index)
    ret.stop_gradient = False
    return ret


class MOEAlltoAllLayer(MOELayerBase):
    """
    Mixture of Experts layer implementation based on GShard paper.
    """

    def __init__(
        self,
        gate: nn.Layer,
        experts: List[nn.Layer],
        layer_idx,
        shared_experts: Optional[List[nn.Layer]] = None,
        group: Group = None,
        recompute=False,
        k=2,
        all_to_all_dropout=0,
        group_experts=False,
        moe_statics=None,
        moe_num_experts=None,
    ):
        """
        Initialize MoE layer.

        Args:
            gate: Gate network for expert selection
            experts: List of expert networks
            layer_idx: Index of this layer in the model
            group: Distributed communication group
            recompute: Whether to enable recomputation
            k: Number of experts to select per token
            all_to_all_dropout: Dropout rate for all-to-all communication
            group_experts: Whether to group experts
            moe_statics: MoE statistics tracking object
        """
        super().__init__()
        self.gate = gate
        self.layer_idx = layer_idx
        self.recompute = recompute
        for p in self.gate.parameters():
            p.is_gate = True
        if isinstance(experts, nn.LayerList):
            self.experts = experts
        else:
            logger.info(f"using fused experts, type={type(experts)}")
            self.experts = experts
        self.shared_experts = shared_experts

        self.group = group
        self.k = k
        self.all_to_all_dropout = all_to_all_dropout
        self.use_correction_bias = moe_statics is not None
        self.moe_statics = moe_statics
        if self.use_correction_bias:
            logger.info(f"using correction bias, aux-coef:{self.gate.config.moe_aux_loss_lambda}")
            assert self.gate.config.moe_use_aux_free

        self.is_mp_moe = (
            hasattr(fleet.fleet, "_hcg") and group is fleet.get_hybrid_communicate_group().get_model_parallel_group()
        )
        is_dummy_moe = dist.get_world_size(group) == 1

        for p in experts.parameters():
            p.expert = not (self.is_mp_moe or is_dummy_moe)  # type: ignore
            p.no_sync = not (self.is_mp_moe or is_dummy_moe)
            if self.is_mp_moe:
                p.is_distributed = True
                p.mp_moe = True

        self.world_size = dist.get_world_size(self.group)
        # assert self.world_size > 1, f'moe-group not found, world_size {self.world_size}'
        self.rank = dist.get_rank(self.group)
        if self.world_size < 1:
            self.world_size = 1
        if self.rank < 0:
            self.rank = 0

        self.multimodal_experts = isinstance(moe_num_experts, (tuple, list)) and len(moe_num_experts) > 1
        self.num_local_experts = len(self.experts) // self.world_size
        if self.multimodal_experts:
            self.num_local_multimodal_experts = [num // self.world_size for num in moe_num_experts]
            self.multimodal_expert_index = [0] + list(itertools.accumulate(moe_num_experts))

        self.input_preprocess = self.output_postprocess = None
        self.group_experts = group_experts
        self.config = self.gate.config
        self.zero = paddle.to_tensor(0, dtype=paddle.float32)

    def forward_experts(self, dispatched_input):
        """
        Forward pass through experts sequentially.

        Args:
            dispatched_input: Input tensor of shape [num_experts, capacity, dim]

        Returns:
            Tensor: Expert outputs of shape [num_experts, capacity, dim]
        """

        if not self.multimodal_experts:
            true_experts = self.experts[self.rank * self.num_local_experts : (self.rank + 1) * self.num_local_experts]
        else:
            true_experts = []
            for i, num in enumerate(self.num_local_multimodal_experts):
                current_modal_experts = self.experts[
                    self.multimodal_expert_index[i] : self.multimodal_expert_index[i + 1]
                ]
                true_experts.extend(current_modal_experts[self.rank * num : (self.rank + 1) * num])

        dispatched_input = dispatched_input.reshape(
            [self.world_size, self.num_local_experts, -1, dispatched_input.shape[-1]]
        )  # [e,1,c,m]
        expert_outputs = []
        if isinstance(self.experts, nn.LayerList):
            chunks = dispatched_input.transpose([1, 0, 2, 3]).contiguous().unbind(0)
            assert len(chunks) == len(true_experts), (len(chunks), len(true_experts))
            for chunk, expert in zip(chunks, true_experts):
                expert_outputs += [expert(chunk)]
        else:
            dispatched_input = dispatched_input.transpose([1, 0, 2, 3])
            dispatched_input.contiguous()
            orig_shape = dispatched_input.shape
            chunks = dispatched_input.reshape([orig_shape[0], -1, orig_shape[-1]])
            chunks = self.experts(chunks)
            chunks = chunks.reshape(orig_shape[:-1] + [chunks.shape[-1]]).unbind(0)
            expert_outputs += chunks
        expert_output = paddle.stack(expert_outputs, axis=1)  # [ecm]
        return expert_output

    def fused_gate_logits_process(self, gate_logits, token_type_ids=None, offload_helper=None):
        """
        Process and combine gate logits.

        Args:
            gate_logits: Raw gate logits

        Returns:
            tuple: (processed probabilities, max probabilities)
        """
        k = self.k
        experts_type_ids = self.gate.experts_type_ids
        use_hard_gate = self.config.moe_use_hard_gate
        max_prob = None

        if token_type_ids is not None and use_hard_gate:
            if offload_helper is None:
                offload_helper = dict()
                lm_mask = token_type_ids == 0
                is_lm = lm_mask.any()
                mm_mask = token_type_ids == 1
                is_mm = mm_mask.any()
                seq_lm = lm_mask.sum()
                seq_mm = mm_mask.sum()
                lm_mask = lm_mask.unsqueeze(1) & (experts_type_ids == 0).unsqueeze(0)
                mm_mask = mm_mask.unsqueeze(1) & (experts_type_ids == 1).unsqueeze(0)
                offload_helper["lm_mask"] = [lm_mask, is_lm, seq_lm]
                offload_helper["mm_mask"] = [mm_mask, is_mm, seq_mm]

            is_lm = offload_helper["lm_mask"][1]
            prob = paddle.zeros_like(gate_logits)
            # 处理 lm_prob
            if is_lm:
                lm_mask = offload_helper["lm_mask"][0]
                seq_lm_cpu = offload_helper["lm_mask"][2]
                lm_mask_nonzero = lm_mask.nonzero()
                lm_partial_gate_logits = gate_logits.gather_nd(lm_mask_nonzero).reshape([seq_lm_cpu, -1])
                if self.group_experts:
                    lm_prob = self.gate.act(lm_partial_gate_logits.reshape([lm_partial_gate_logits.shape[0], k, -1]))
                    max_prob = lm_prob.max(-1, keepdim=True)  # [s_l, k, 1]
                    lm_prob /= max_prob
                else:
                    lm_prob = self.gate.act(lm_partial_gate_logits)
                prob = paddle.scatter_nd_add(prob, lm_mask_nonzero, lm_prob.flatten())

            is_mm = offload_helper["mm_mask"][1]
            if is_mm:
                mm_mask = offload_helper["mm_mask"][0]
                seq_mm_cpu = offload_helper["mm_mask"][2]
                mm_mask_nonzero = paddle.nonzero(mm_mask)
                mm_partial_gate_logits = gate_logits.gather_nd(mm_mask_nonzero).reshape([seq_mm_cpu, -1])
                mm_prob = self.gate.act(mm_partial_gate_logits)
                prob = paddle.scatter_nd_add(prob, mm_mask_nonzero, mm_prob.flatten())
        else:
            if self.group_experts:
                prob = self.gate.act(gate_logits.reshape([gate_logits.shape[0], k, -1]))
                max_prob = prob.max(-1, keepdim=True)
                prob /= max_prob
                prob = prob.reshape([prob.shape[0], -1])
            else:
                prob = self.gate.act(gate_logits)
        return prob, max_prob

    def gate_and_dispatch(self, input, token_type_ids=None):
        """
        Calculate gate and dispatch inputs.

        Args:
            input: Input tensor of shape [seq, dim]

        Returns:
            tuple: (dispatched_input, combine_weights, dispatch_mask,
            scatter_index, router_loss, gate_logits, gate_prob)
        """
        seqlen, d_model = input.shape
        args = ()
        if token_type_ids is not None:
            token_type_ids = token_type_ids.reshape([-1])
            args = (token_type_ids,)

        (
            gate_logits,
            capacity,
            router_loss,
        ) = self.gate(input, *args)
        if self.input_preprocess is not None:
            input, gate_logits = self.input_preprocess(input, gate_logits, capacity)
        # capacity no use
        k = self.k
        prob, max_prob = self.fused_gate_logits_process(gate_logits, token_type_ids)

        if "corr_bias" in inspect.signature(moe_gate_dispatch).parameters:
            if self.use_correction_bias:
                compat_args = (self.moe_statics.e_score_correction_bias[0],)
            else:
                compat_args = (None,)
        else:
            assert not self.use_correction_bias, "correction bias not supported, rebuild moe-ops"
            compat_args = ()

        (
            dispatched_input,
            combine_weights_unnorm,
            scatter_index,
            dispatch_mask,
            _,
        ) = moe_gate_dispatch(input, prob, *compat_args, k=k, capacity=capacity, use_pad=True)
        dispatched_input = dispatched_input.astype(input.dtype)

        dispatch_mask = paddle.diff(F.pad(dispatch_mask, (1, 0)))
        if self.use_correction_bias:
            if self.use_multimodel_experts:
                for i in range(len(self.moe_statics.expert_usage)):
                    self.moe_statics.expert_usage[i] += dispatch_mask[self.gate.experts_type_mask[i]].detach()
            else:
                self.moe_statics.expert_usage[0] += dispatch_mask.detach()
        dispatched_input.stop_gradient = False
        combine_weights_unnorm.stop_gradient = False
        scatter_index.stop_gradient = True
        dispatch_mask.stop_gradient = True

        scatter_index = scatter_index.transpose([1, 0])  # [k,s] ->[s,k]
        if self.group_experts:
            if max_prob is not None:
                if token_type_ids is not None:
                    p = paddle.ones_like(combine_weights_unnorm.unsqueeze(-1))
                    p = paddle.scatter_nd_add(p, paddle.nonzero(token_type_ids == 0), -1 + max_prob)
                else:
                    p = max_prob
                combine_weights_unnorm = (combine_weights_unnorm.unsqueeze(-1) * p).squeeze(-1)
                prob = (prob.reshape([p.shape[0], k, -1]) * p).reshape([p.shape[0], -1])
        if self.gate.norm_gate_logits:
            combine_weights = combine_weights_unnorm / paddle.clip(
                combine_weights_unnorm.sum(-1, keepdim=True), min=1e-12
            )
        else:
            combine_weights = combine_weights_unnorm
        combine_weights = combine_weights.cast(dispatched_input.dtype)

        dispatched_input = dispatched_input.reshape([self.world_size * self.num_local_experts, capacity, d_model])
        dispatch_mask.stop_gradient = True
        scatter_index.stop_gradient = True
        return (
            dispatched_input,
            combine_weights,
            dispatch_mask,
            scatter_index,
            router_loss,
            gate_logits,
            prob,
        )

    def _calc_router_loss(
        self,
        dispatch_mask,
        gate_logits,
        gate_prob,
        num_experts,
        use_group,
        layer_idx,
        token_type=None,
        tokens_type_mask=None,
        dispatch_tokens_mask=None,
        prefix="",
    ):
        """
        Calculate router loss including auxiliary loss, z-loss and orthogonal loss.

        Args:
            dispatch_mask: Dispatch mask
            gate_logits: Gate logits
            gate_prob: Gate probabilities
            num_experts: Number of experts
            use_group: Whether to use expert groups
            layer_idx: Layer index
            token_type: Token type
            tokens_type_mask: Token type mask
            dispatch_tokens_mask: Dispatch tokens mask
            prefix: Prefix for logging

        Returns:
            Tensor: Total router loss
        """
        router_loss, l_aux, orthogonal_loss, zloss = 0.0, None, None, None
        if self.gate.config.moe_aux_loss_lambda:
            l_aux = self.gate._cal_aux_loss(
                gate_prob,
                dispatch_mask,
                num_experts,
                use_group,
                tokens_type_mask,
                dispatch_tokens_mask,
            )
            router_loss += self.gate.moe_aux_loss_lambda[token_type or 0] * l_aux
        else:
            router_loss += self.zero * gate_prob[0, 0]  # must use gate prob to avoid zero pointer
        if self.gate.config.moe_orthogonal_loss_lambda:
            orthogonal_loss = self.gate._cal_orthogonal_loss(token_type, use_group)
            router_loss += self.gate.moe_orthogonal_loss_lambda[token_type or 0] * orthogonal_loss
        if self.gate.config.moe_z_loss_lambda:
            zloss = self.gate._cal_z_loss(gate_logits, tokens_type_mask)
            router_loss += self.gate.moe_z_loss_lambda[token_type or 0] * zloss
        return router_loss

    def calc_router_loss_and_logging(
        self,
        router_loss,
        combine_weights,
        dispatch_mask,
        gate_logits,
        gate_prob,
        token_type_ids=None,
        dispatch_token_type_ids=None,
        offload_helper=None,
    ):
        """
        Calculate auxiliary losses and log statistics in fused expert case.

        Args:
            router_loss: Base router loss
            combine_weights: Combination weights
            dispatch_mask: Dispatch mask
            gate_logits: Gate logits
            gate_prob: Gate probabilities

        Returns:
            Tensor: Updated router loss
        """
        assert gate_prob is not None
        if token_type_ids is not None and self.gate.config.moe_use_hard_gate:  # true
            if not self.gate.weight.stop_gradient:
                lm_tokens_mask = token_type_ids == 0
                if offload_helper is not None:
                    is_lm = offload_helper["lm_mask"][1]
                else:
                    is_lm = lm_tokens_mask.any()
                if is_lm:
                    dispatch_tokens_mask = (
                        dispatch_token_type_ids == 0 if dispatch_token_type_ids is not None else None
                    )
                    router_loss += self._calc_router_loss(
                        (
                            dispatch_mask[self.gate.experts_type_mask[0]]
                            if hasattr(self.gate, "experts_type_mask")
                            else dispatch_mask
                        ),
                        (
                            gate_logits[:, self.gate.experts_type_mask[0]]
                            if hasattr(self.gate, "experts_type_mask")
                            else gate_logits
                        ),
                        (
                            gate_prob[:, self.gate.experts_type_mask[0]]
                            if hasattr(self.gate, "experts_type_mask")
                            else gate_prob
                        ),
                        (
                            self.gate.num_experts_list[0]
                            if hasattr(self.gate, "num_experts_list")
                            else self.gate.num_experts_tensor
                        ),
                        self.group_experts,
                        self.layer_idx,
                        0,
                        lm_tokens_mask,
                        dispatch_tokens_mask,
                        prefix="lm",
                    )
            mm_tokens_mask = token_type_ids == 1
            if offload_helper is not None:
                is_mm = offload_helper["mm_mask"][1]
            else:
                is_mm = mm_tokens_mask.any()
            if is_mm:
                dispatch_tokens_mask = dispatch_token_type_ids == 1 if dispatch_token_type_ids is not None else None
                router_loss += self._calc_router_loss(
                    dispatch_mask[self.gate.experts_type_mask[1]],
                    gate_logits[:, self.gate.experts_type_mask[1]],
                    gate_prob[:, self.gate.experts_type_mask[1]],
                    self.gate.num_experts_list[1],
                    False,
                    self.layer_idx,
                    1,
                    mm_tokens_mask,
                    dispatch_tokens_mask,
                    prefix="mm",
                )

        else:
            router_loss += self._calc_router_loss(
                dispatch_mask,
                gate_logits,
                gate_prob,
                self.gate.num_experts_tensor,
                self.group_experts,
                self.layer_idx,
            )

        return router_loss

    def combine_expert_output(self, expert_output, combine_weights, scatter_index):
        """
        Combine expert outputs using combination weights.

        Args:
            expert_output: Expert outputs [num_experts, capacity, dim]
            combine_weights: Combination weights
            scatter_index: Scatter indices

        Returns:
            Tensor: Combined output [seqlen, dim]
        """
        expert_output = expert_output.reshape([-1, expert_output.shape[-1]])  # [e*1,c,m]
        combined_output = combining(expert_output, combine_weights, scatter_index)

        if self.output_postprocess is not None:
            combined_output = self.output_postprocess(combined_output)

        return combined_output

    def forward_single_stage(self, dispatched_input, stage_id):
        """
        Forward pass for single expert stage.

        Args:
            dispatched_input: Dispatched input
            stage_id: Stage index

        Returns:
            Tensor: Expert output
        """
        assert isinstance(self.experts, nn.LayerList)
        return self.experts[stage_id](dispatched_input)

    def all2all_expert_overlap(self, x, group):
        """all2all_expert_overlap"""
        all2all_tasks = []
        all2all_ins = paddle.unbind(x, axis=0)
        for stage_id in range(1):
            stage_input = all2all_ins[stage_id]
            x_out, task = AlltoAll.apply(stage_input, group=self.group, sync_op=False)
            all2all_tasks.append((task, x_out))

        expert_outputs = []
        for stage_id in range(self.num_local_experts):
            if stage_id + 1 != self.num_local_experts:
                stage_input = all2all_ins[stage_id + 1]
                x_out, task = AlltoAll.apply(stage_input, group=self.group, sync_op=False)
                all2all_tasks.append((task, x_out))

            task, dispatched_input = all2all_tasks[stage_id]
            task.wait()
            expert_outputs_cur_stage = (
                recompute(self.forward_single_stage, dispatched_input, stage_id)
                if self.recompute and self.training
                else self.forward_single_stage(dispatched_input, stage_id)
            )
            expert_outputs.append(expert_outputs_cur_stage)

        expert_output = paddle.stack(expert_outputs, axis=1)
        return expert_output

    def forward(
        self,
        input: Tensor,
        token_type_ids=None,
        **kwargs,
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """
        Forward pass through MoE layer.

        Args:
            input: Input tensor of shape [s, d]

        Returns:
            tuple: (output, combine_weights, router_loss, gate_logits)
        """
        # assert len(input) == 1, "only single input Tensor supported"
        if input.ndim == 3:
            orig_shape = input.shape
            input = input.reshape([-1, input.shape[-1]])
        else:
            orig_shape = None
        assert len(input.shape) == 2, f"input Tensor must have dimensions: (s)equence, (d)im, got:{input.shape}"
        if token_type_ids is not None:
            token_type_ids = token_type_ids.clone()[:, :-1]
            if self.config.sequence_parallel:
                token_type_ids = token_type_ids.reshape([-1])
                token_type_ids = ScatterOp.apply(token_type_ids)
                token_type_ids.stop_gradient = True

        assert self.gate is not None
        if hasattr(self, "rng") and self.rng.random() < self.all_to_all_dropout:
            orig_shape_2 = input.shape
            output = self.forward_experts(input)
            output += self.gate.weight.sum() * 0.0  # hack for grad
            output = output.reshape(orig_shape or orig_shape_2)  # [e*1,c,m]
            return output, None, 0

        is_first_fwd = not framework._dygraph_tracer()._has_grad
        gate_input = input

        (
            dispatched_input,
            combine_weights,
            dispatch_mask,
            scatter_index,
            router_loss,
            gate_logits,
            gate_prob,
        ) = self.gate_and_dispatch(gate_input, token_type_ids)

        use_async = self.shared_experts is not None
        if use_async:
            dispatched_input, shared_out = AlltoAllAsync.apply(
                dispatched_input,
                input,  # args to shared-experts
                group=self.group,
                fn=self.shared_experts,
                is_first_fwd=is_first_fwd,
            )
        else:
            dispatched_input = AlltoAll.apply(dispatched_input, self.group)

        expert_out = (
            recompute(self.forward_experts, dispatched_input)
            if self.recompute and self.training
            else self.forward_experts(dispatched_input)
        )

        expert_out, router_loss2 = AlltoAllAsync.apply(
            expert_out,
            router_loss,
            combine_weights,
            dispatch_mask,
            gate_logits,
            gate_prob,
            token_type_ids,
            group=self.group,
            fn=self.calc_router_loss_and_logging,
            is_first_fwd=is_first_fwd,
        )

        combined_output = self.combine_expert_output(expert_out, combine_weights, scatter_index)

        if self.shared_experts is not None:
            combined_output += shared_out

        if orig_shape:
            combined_output = combined_output.clone().reshape(orig_shape[:-1] + [combined_output.shape[-1]])
        return combined_output, combine_weights, router_loss2, gate_logits
