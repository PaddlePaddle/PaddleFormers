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

import inspect
import logging
from collections import namedtuple

import numpy as np
import paddle
import paddle.distributed as dist
import paddle.nn.functional as F
from paddle import framework, nn
from paddle.autograd import PyLayer
from paddle.distributed import fleet
from paddle.distributed.communication import stream
from paddle.distributed.fleet.utils import recompute
from paddle.incubate.nn.functional import (
    moe_combine,
    moe_gate_dispatch,
    moe_gate_dispatch_permute,
)

from paddleformers.cli.train.ernie_pretrain.models.comm_utils import profile
from paddleformers.cli.train.ernie_pretrain.models.moe.token_dispatcher.fp8_utils import (
    ExpertsGroupGemmContiguousNode,
    ExpertsGroupGemmNode,
    ExpertsGroupGemmWLCHNode,
)
from paddleformers.cli.train.ernie_pretrain.models.moe.token_dispatcher.moe_utils import (
    UnZipNode,
    ZipNode,
)
from paddleformers.cli.train.ernie_pretrain.models.sequence_parallel_utils import (
    ScatterOp,
)
from paddleformers.cli.train.ernie_pretrain.models.utils import manual_backward

try:
    from paddle.incubate.nn.functional import moe_gate_dispatch_and_quant
except ImportError:
    moe_gate_dispatch_and_quant = None

try:
    from paddleformers.cli.train.ernie_pretrain.src.utils.misc import (
        global_training_logs,
    )
except ModuleNotFoundError:
    global_training_logs = {}

logger = logging.getLogger(__name__)

GateOutput = namedtuple(
    "GateOutput",
    [
        "aux",
        "z",
        "logits",
    ],
)


def set_grad_in_dtype_non_consistent(ctx):
    """Allow grad dtype not consistent with forward dtype"""
    if hasattr(ctx, "set_grad_in_dtype_consistent"):
        ctx.set_grad_in_dtype_consistent(False)


class Fp8MoeGateDispatchAndQuant(paddle.autograd.PyLayer):
    """Fp8MoeGateDispatchAndQuant"""

    @staticmethod
    def forward(
        ctx,
        x,
        gate_logtis,
        corr_bias,
        k,
        capacity,
        use_pad,
        use_pow2_scale=True,
    ):
        """forward"""
        set_grad_in_dtype_non_consistent(ctx)
        assert moe_gate_dispatch_and_quant is not None, "Please use new version Paddle."
        with paddle.amp.auto_cast(enable=False):
            (out_fp8, scale, combine_weights, scatter_index, expert_offset, expert_id,) = moe_gate_dispatch_and_quant(
                x,
                gate_logtis,
                corr_bias=corr_bias,
                k=k,
                capacity=capacity,
                use_pad=use_pad,
                use_pow2_scale=use_pow2_scale,
            )
        assert out_fp8.shape[0] == scale.shape[0]

        out_fp8.stop_gradient = False
        combine_weights.stop_gradient = False
        scatter_index.stop_gradient = True
        expert_offset.stop_gradient = True
        expert_id.stop_gradient = True
        scale.stop_gradient = True

        ctx.k = k
        ctx.capacity = capacity
        ctx.use_pad = use_pad
        ctx.combine_weights = combine_weights
        ctx.scatter_index = scatter_index
        ctx.expert_id = expert_id
        ctx.has_corr_bias = corr_bias is not None

        return (
            out_fp8,
            combine_weights,
            scatter_index,
            expert_offset,
            expert_id,
            {
                "scale": scale,
            },
        )

    @staticmethod
    def backward(ctx, *grads):
        """backward"""
        out_grad, combine_weights_grad = grads[0], grads[1]
        x_grad, gate_logits_grad = paddle._C_ops.moe_gate_dispatch_grad(
            ctx.combine_weights,
            ctx.scatter_index,
            ctx.expert_id,
            out_grad,
            combine_weights_grad,
            ctx.k,
            ctx.capacity,
            ctx.use_pad,
        )
        if ctx.has_corr_bias:
            return x_grad, gate_logits_grad, None
        else:
            return x_grad, gate_logits_grad


def recompute_fwd_gate_up_func(config, layer_idx):
    if "recompute_fwd_gate_up" in config.fp8_mem_configs:
        if isinstance(config.fp8_mem_configs["recompute_fwd_gate_up"], bool):
            return config.fp8_mem_configs["recompute_fwd_gate_up"]
        if isinstance(config.fp8_mem_configs["recompute_fwd_gate_up"], list):
            return layer_idx in config.fp8_mem_configs["recompute_fwd_gate_up"]

    return False


class MoEStatics(nn.Layer):
    def __init__(self, config, layer_idx):
        super().__init__()
        self._cast_to_low_precision = False
        self._cast_to_low_precision = False
        num_experts = config.moe_num_experts

        with paddle.utils.unique_name.guard(f"mm_layer_{layer_idx}_"):
            num_experts_groups = 1
            p = self.create_parameter(
                shape=[num_experts_groups, num_experts],
                dtype="float32",
                is_bias=True,
                attr=paddle.ParamAttr(name=paddle.utils.unique_name.generate("corr_bias")),
            )
            p.stop_gradient = False
            self.e_score_correction_bias = p
            self.e_score_correction_bias.is_distributed = True
            self.e_score_correction_bias.unused_param = True
            if getattr(config, "build_skip_comm_buffer", False):
                self.e_score_correction_bias.color = {
                    "color": "skip_comm",
                    "group": paddle.distributed.new_group([paddle.distributed.get_rank()]),
                }
            p = paddle.zeros(
                shape=[num_experts_groups, num_experts],
                dtype="int64",
            )
            p.stop_gradient = True
            self.expert_usage = p


class GateCombine(PyLayer):
    @staticmethod
    def forward(ctx, x, combine_weights, scatter_index):
        ctx.x = x
        ctx.combine_weights = combine_weights
        ctx.scatter_index = scatter_index
        ret = moe_combine(x, combine_weights, scatter_index)
        return ret

    @staticmethod
    def backward(ctx, grad_y, *_):
        # assert moe_combine is not None
        grad_x, grad_combine_weight_helper = paddle._C_ops.moe_combine_grad(
            ctx.x, ctx.combine_weights, ctx.scatter_index, grad_y
        )
        grad_combine_weight = grad_combine_weight_helper.sum(-1)
        return grad_x, grad_combine_weight.reshape(ctx.combine_weights.shape), None


class FusionFP8Expert(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, hidden_states, custom_map):
        ctx.node = ExpertsGroupGemmNode(None, custom_map)

        t1 = hidden_states.transpose([1, 0, 2, 3]).contiguous()
        expert_num = t1.shape[0]
        tokens_num = t1.shape[1] * t1.shape[2]
        tokens_per_expert = paddle.full([expert_num], fill_value=tokens_num, dtype="int32")

        t1 = t1.reshape([-1, hidden_states.shape[-1]])
        out = ctx.node.forward_no_prob(t1, tokens_per_expert)

        expert_output = out.reshape(hidden_states.shape).transpose([1, 0, 2, 3]).contiguous()

        ctx.save_for_backward(tokens_per_expert)
        return expert_output

    @staticmethod
    def backward(ctx, output_grad):
        (tokens_per_expert,) = ctx.saved_tensor()

        t1 = output_grad.transpose([1, 0, 2, 3]).contiguous()
        t1 = t1.reshape([-1, output_grad.shape[-1]])

        dx = ctx.node.backward_no_prob(t1, tokens_per_expert)
        dx = dx.reshape(output_grad.shape).transpose([1, 0, 2, 3]).contiguous()
        return dx


class AlltoAll(PyLayer):
    @staticmethod
    def forward(ctx, x, group, sync_op=True):
        ctx.group = group
        if dist.get_world_size(group) <= 1:
            return x
        output = paddle.empty_like(x)
        output.stop_gradient = False
        task = stream.alltoall_single(output, x, None, None, group, sync_op=sync_op, use_calc_stream=sync_op)
        if not sync_op:
            return output, task
        else:
            return output

    @staticmethod
    def backward(ctx, *dx):
        return AlltoAll.apply(*dx, group=ctx.group)


class AlltoAllExpertOverlap(PyLayer):
    @staticmethod
    def forward(ctx, input, group, num_local_experts, forward_func_dict, is_first_fwd=False):
        assert (
            dist.get_world_size(group) > 1
        ), "AlltoAllExpertOverlap is not supported for a world size less than or equal to 1."

        ctx.bw_funcs = {}
        ctx.group = group
        ctx.num_local_experts = num_local_experts

        assert isinstance(forward_func_dict, nn.LayerList)
        all2all_tasks = []
        all2all_ins = paddle.unbind(input, axis=0)
        for stage_id in range(1):
            stage_input = all2all_ins[stage_id]
            x_out, task = AlltoAll.apply(stage_input, group=group, sync_op=False)
            all2all_tasks.append((task, x_out))

        expert_outputs = []
        for stage_id in range(num_local_experts):
            if stage_id + 1 != num_local_experts:
                stage_input = all2all_ins[stage_id + 1]
                x_out, task = AlltoAll.apply(stage_input, group=group, sync_op=False)
                all2all_tasks.append((task, x_out))

            task, dispatched_input = all2all_tasks[stage_id]
            task.wait()
            bwf, (expert_outputs_cur_stage,) = manual_backward(
                forward_func_dict[stage_id], is_first_fwd, dispatched_input
            )
            ctx.bw_funcs[stage_id] = bwf
            expert_outputs.append(expert_outputs_cur_stage)

        expert_output = paddle.stack(expert_outputs, axis=1)
        return expert_output

    @staticmethod
    def backward(ctx, out_grad):
        all2all_tasks = []
        expert_outputs = []

        out_grad_list = paddle.split(out_grad, num_or_sections=out_grad.shape[1], axis=1)
        for stage_id in range(ctx.num_local_experts):
            (grad_cur_stage,) = ctx.bw_funcs[stage_id](out_grad_list[stage_id])

            x_out, task = AlltoAll.apply(grad_cur_stage, group=ctx.group, sync_op=False)
            all2all_tasks.append(task)
            expert_outputs.append(x_out)

        for task in all2all_tasks:
            task.wait()

        expert_output = paddle.stack(expert_outputs, axis=0)
        return expert_output


class AlltoAllAsync(PyLayer):
    @staticmethod
    def forward(ctx, x, *fn_args, group=None, fn=None, is_first_fwd=False):
        assert fn is not None, "use AlltoAll no async"
        ctx.group = group
        if dist.get_world_size(group) <= 1:
            ctx.bwf, fn_out = manual_backward(fn, is_first_fwd, *fn_args)
            return (x,) + fn_out
        x_out = paddle.empty_like(x)
        x_out.stop_gradient = False
        task = stream.alltoall_single(
            x_out,
            x,
            None,
            None,
            group,
            sync_op=False,
        )
        ctx.bwf, fn_out = manual_backward(fn, is_first_fwd, *fn_args)
        task.wait()
        return (x_out,) + fn_out

    @staticmethod
    def backward(ctx, dx_out, *fn_out_grads):
        if dist.get_world_size(ctx.group) <= 1:
            fn_args_grads = ctx.bwf(*fn_out_grads)
            return (dx_out,) + fn_args_grads

        dx = paddle.empty_like(dx_out)
        dx.stop_gradient = False
        task = stream.alltoall_single(
            dx,
            dx_out,
            None,
            None,
            ctx.group,
            sync_op=False,
        )
        fn_args_grads = ctx.bwf(*fn_out_grads)
        task.wait()
        return (dx,) + fn_args_grads


def dispatching(x, dispatch_mask, scatter_index, num_experts, capacity):
    output = None
    orig_dtype = x.dtype
    scatter_index = scatter_index.unbind(1)
    dispatch_mask = dispatch_mask.unbind(1)
    for i_scatter_index, i_dispatch_mask in zip(scatter_index, dispatch_mask):
        init_output = paddle.zeros([num_experts * capacity, x.shape[-1]], dtype="float32")
        updates = x * i_dispatch_mask.unsqueeze(-1).cast(x.dtype)
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


def combining_fused(x, combine_weights, scatter_index, hard_gate=False):
    if hard_gate:
        x_gatherd = F.embedding(scatter_index, x)
        return x_gatherd.squeeze(-2)
    ret = GateCombine.apply(x, combine_weights, scatter_index)
    ret.stop_gradient = False
    return ret


class ReshapeKeepGradDtype(PyLayer):
    @staticmethod
    def forward(ctx, x, shape):
        set_grad_in_dtype_non_consistent(ctx)
        ctx.orig_shape = x.shape
        return x.reshape(shape)

    @staticmethod
    def backward(ctx, grad):
        return grad.reshape(ctx.orig_shape)


class MOELayer(nn.Layer):
    """Mixture of Experts (MoE) Layer implementation.

    This layer dynamically routes input tokens to different expert networks
    based on a gating mechanism, allowing for conditional computation.

    """

    def __init__(
        self,
        gate,
        experts,
        layer_idx,
        shared_experts,
        group,
        recompute=False,
        enable_logging=False,
        k=2,
        enable_bpr=False,
        all_to_all_dropout=0,
        group_experts=False,
        moe_statics=None,
    ):
        """Initialize the MoE layer.

        Args:
            gate (nn.Layer): Gating network that outputs routing scores.
            experts (nn.LayerList, optional): List of expert networks.
            layer_idx (int): Identifier for this layer (used for logging).
            shared_experts (nn.Layer): Shared expert applied to all tokens (optional).
            group (dist.ProcessGroup): Process group for distributed expert parallelism.
            recompute (bool, optional): If True, enables gradient checkpointing. Defaults to False.
            enable_logging (bool, optional): If True, tracks expert usage statistics. Defaults to False.
            k (int, optional): Number of experts to route each token to. Defaults to 2.
            enable_bpr (bool, optional): If True, uses balanced positive routing. Defaults to False.
            all_to_all_dropout (float, optional): Dropout rate for cross-device communication. Defaults to 0.
            group_experts (bool, optional): If True, optimizes expert communication. Defaults to False.
        """

        super().__init__()
        self.gate = gate
        self.layer_idx = layer_idx
        self.recompute = recompute
        logger.info(f"using moe recompute={recompute}")
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
        self.enable_logging = enable_logging
        self.use_correction_bias = moe_statics is not None
        self.moe_statics = moe_statics
        if self.use_correction_bias:
            logger.info(f"using correction bias, aux-coef:{self.gate.config.moe_aux_loss_lambda}")
            assert self.gate.config.moe_use_aux_free

        self.is_mp_moe = (
            hasattr(fleet.fleet, "_hcg") and group is fleet.get_hybrid_communicate_group().get_model_parallel_group()
        )
        self.is_ep_moe = (
            hasattr(fleet.fleet, "_hcg")
            and hasattr(
                fleet.get_hybrid_communicate_group(),
                "get_moe_sharding_parallel_world_size",
            )
            and fleet.get_hybrid_communicate_group().get_moe_sharding_parallel_world_size() > 0
        )
        is_dummy_moe = dist.get_world_size(group) == 1

        for p in experts.parameters():
            p.expert = not (self.is_mp_moe or is_dummy_moe)
            p.no_sync = not (self.is_mp_moe or is_dummy_moe)
            logger.info(f"expert no-sync={p.no_sync}-{p.name}")
            if self.is_mp_moe or self.is_ep_moe:
                p.is_distributed = True

        expert_color = None
        if self.is_ep_moe:
            moe_grad_group = fleet.get_hybrid_communicate_group().get_moe_sharding_parallel_group()
            expert_color = {"color": "moe_expert", "group": moe_grad_group}
        elif self.config.offline_quant_expert_weight and self.config.clear_origin_weight_when_offline_quant:
            expert_color = {"color": "moe_expert"}

        if expert_color is not None:
            for p in self.experts.parameters():
                setattr(p, "color", expert_color)

        self.world_size = dist.get_world_size(self.group)
        self.rank = dist.get_rank(self.group)
        if self.world_size < 1:
            self.world_size = 1
        if self.rank < 0:
            self.rank = 0

        self.num_local_experts = len(self.experts)
        self.dispatch_by_task = hasattr(self.gate, "dispatch_by_task") and self.gate.dispatch_by_task

        assert not self.dispatch_by_task, "no dispatch_by_task for now"

        self.input_preprocess = self.output_postprocess = None
        self.group_experts = group_experts
        self.config = self.gate.config
        self.zero = paddle.to_tensor(0, dtype=paddle.float32)

    def forward_experts(self, dispatched_input):
        with profile("fwd-expert"):
            dispatched_input = dispatched_input.reshape(
                [
                    self.world_size,
                    self.num_local_experts,
                    -1,
                    dispatched_input.shape[-1],
                ]
            )
            expert_outputs = []
            if isinstance(self.experts, nn.LayerList):
                if self.config.use_fp8_fuse_node:
                    expert_output = FusionFP8Expert.apply(dispatched_input, self)
                else:
                    chunks = dispatched_input.transpose([1, 0, 2, 3]).contiguous().unbind(0)
                    assert len(chunks) == len(self.experts), (
                        len(chunks),
                        len(self.experts),
                    )
                    for chunk, expert in zip(chunks, self.experts):
                        expert_outputs += [expert(chunk)]

                    expert_output = paddle.stack(expert_outputs, axis=1)

            else:
                dispatched_input = dispatched_input.transpose([1, 0, 2, 3])
                dispatched_input.contiguous()
                orig_shape = dispatched_input.shape
                chunks = dispatched_input.reshape([orig_shape[0], -1, orig_shape[-1]])
                chunks = self.experts(chunks)
                chunks = chunks.reshape(orig_shape[:-1] + [chunks.shape[-1]]).unbind(0)
                expert_outputs += chunks
                expert_output = paddle.stack(expert_outputs, axis=1)
        return expert_output

    def fp8_quant_weight(self):
        expert_w1_list = [expert.up_gate_proj.weight for expert in self.experts if expert is not None]
        expert_w2_list = [expert.down_proj.weight for expert in self.experts if expert is not None]

        expert_w1 = expert_w1_list[0]
        expert_w2 = expert_w2_list[0]

        fp8_weight_stacked_w1, fp8_scale_stacked_w1 = paddle.incubate.nn.functional.fused_stack_transpose_quant(
            expert_w1_list, transpose=False
        )
        setattr(expert_w1, "fp8_weight_stacked", fp8_weight_stacked_w1)
        setattr(expert_w1, "fp8_scale_stacked", fp8_scale_stacked_w1)

        fp8_weight_stacked_w1_t, fp8_scale_stacked_w1_t = paddle.incubate.nn.functional.fused_stack_transpose_quant(
            expert_w1_list, transpose=True
        )
        setattr(expert_w1, "fp8_weight_stacked_transpose", fp8_weight_stacked_w1_t)
        setattr(expert_w1, "fp8_scale_stacked_transpose", fp8_scale_stacked_w1_t)

        fp8_weight_stacked_w2, fp8_scale_stacked_w2 = paddle.incubate.nn.functional.fused_stack_transpose_quant(
            expert_w2_list, transpose=False
        )
        setattr(expert_w2, "fp8_weight_stacked", fp8_weight_stacked_w2)
        setattr(expert_w2, "fp8_scale_stacked", fp8_scale_stacked_w2)

        fp8_weight_stacked_w2_t, fp8_scale_stacked_w2_t = paddle.incubate.nn.functional.fused_stack_transpose_quant(
            expert_w2_list, transpose=True
        )
        setattr(expert_w2, "fp8_weight_stacked_transpose", fp8_weight_stacked_w2_t)
        setattr(expert_w2, "fp8_scale_stacked_transpose", fp8_scale_stacked_w2_t)

    def fused_gate_logits_process(self, gate_logits, token_type_ids, offload_helper=None):
        k = self.k
        max_prob = None

        if self.group_experts:
            prob = self.gate.act(gate_logits.reshape([gate_logits.shape[0], k, -1]))
            max_prob = prob.max(-1, keepdim=True)
            prob /= max_prob
            prob = prob.reshape([prob.shape[0], -1])
        else:
            prob = self.gate.act(gate_logits)
        return prob, max_prob

    def gate_distpach_and_quant(self, input, token_type_ids):
        """
        Quantization is performed within the op
        """
        assert not self.config.use_ep_comm_overlap, "ep_comm_overlap is not supported"

        seqlen, d_model = input.shape
        args = ()
        assert token_type_ids is None

        (
            gate_logits,
            capacity,
            router_loss,
        ) = self.gate(input, *args)

        if self.input_preprocess is not None:
            input, gate_logits = self.input_preprocess(input, gate_logits, capacity)

        k = self.k
        prob, max_prob = self.fused_gate_logits_process(gate_logits, token_type_ids)

        with profile("dispatch_op"):
            corr_bias = self.moe_statics.e_score_correction_bias[0].detach() if self.use_correction_bias else None

            (
                dispatched_input,
                combine_weights_unnorm,
                scatter_index,
                dispatch_mask,
                _,
                fp8_dispatched_handle,
            ) = Fp8MoeGateDispatchAndQuant.apply(input, prob, corr_bias, k=k, capacity=capacity, use_pad=True)

        dispatch_mask = paddle.diff(F.pad(dispatch_mask, (1, 0)))
        if self.use_correction_bias:
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
        combine_weights = combine_weights.cast("bfloat16")

        def reshape_for_a2a(tensor):
            return ReshapeKeepGradDtype.apply(
                tensor,
                [
                    self.world_size * self.num_local_experts,
                    capacity,
                    -1,
                ],
            )

        dispatched_input = reshape_for_a2a(dispatched_input)
        fp8_dispatched_handle["scale"] = reshape_for_a2a(fp8_dispatched_handle["scale"])
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
            fp8_dispatched_handle,
        )

    def gate_and_distpach(self, input, token_type_ids):
        seqlen, d_model = input.shape
        args = ()
        assert token_type_ids is None

        (
            gate_logits,
            capacity,
            router_loss,
        ) = self.gate(input, *args)

        if self.input_preprocess is not None:
            input, gate_logits = self.input_preprocess(input, gate_logits, capacity)

        k = self.k
        prob, max_prob = self.fused_gate_logits_process(gate_logits, token_type_ids)

        with profile("dispatch_op"):
            if "corr_bias" in inspect.signature(moe_gate_dispatch).parameters:
                if self.use_correction_bias:
                    compat_args = (self.moe_statics.e_score_correction_bias[0],)
                else:
                    compat_args = (None,)
            else:
                assert not self.use_correction_bias, "correction bias not supported, rebuild moe-ops"
                compat_args = ()

            if not self.config.use_ep_comm_overlap:
                (dispatched_input, combine_weights_unnorm, scatter_index, dispatch_mask, _,) = moe_gate_dispatch(
                    input,
                    prob,
                    *compat_args,
                    k=k,
                    capacity=capacity,
                    use_pad=True,
                )
            else:
                (
                    dispatched_input,
                    combine_weights_unnorm,
                    scatter_index,
                    dispatch_mask,
                    _,
                ) = moe_gate_dispatch_permute(
                    input,
                    prob,
                    *compat_args,
                    k=k,
                    capacity=capacity,
                    world_size=self.group.nranks,
                )

            dispatched_input = dispatched_input.cast(input.dtype)

            dispatch_mask = paddle.diff(F.pad(dispatch_mask, (1, 0)))
            if self.use_correction_bias:
                self.moe_statics.expert_usage[0] += dispatch_mask.detach()
            dispatched_input.stop_gradient = False
            combine_weights_unnorm.stop_gradient = False
            scatter_index.stop_gradient = True
            dispatch_mask.stop_gradient = True

            scatter_index = scatter_index.transpose([1, 0])
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

        if not self.config.use_ep_comm_overlap:
            dispatched_input = dispatched_input.reshape(
                [
                    self.world_size * self.num_local_experts,
                    capacity,
                    (d_model),
                ]
            )
        else:
            assert (
                len(dispatched_input.shape) == 4
                and dispatched_input.shape[1] == self.world_size
                and dispatched_input.shape[0] == self.num_local_experts
            ), (
                f"When using ep_comm_overlap, moe_gate_dispatch_permute is needed. "
                f"Expected dispatched_input to have shape[1] == {self.world_size} "
                f"and shape[0] == {self.num_local_experts}, "
                f"but got shape {dispatched_input.shape}"
            )
            dispatched_input = dispatched_input
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
        router_loss, l_aux, orthogonal_loss = 0.0, None, None
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
            router_loss += self.zero * gate_prob[0, 0]
        if self.gate.config.moe_orthogonal_loss_lambda:
            orthogonal_loss = self.gate._cal_orthogonal_loss(token_type, use_group)
            router_loss += self.gate.moe_orthogonal_loss_lambda[token_type or 0] * orthogonal_loss
        return router_loss

    def calc_router_loss_and_logging(
        self,
        router_loss,
        combine_weights,
        dispatch_mask,
        gate_logits,
        gate_prob,
        token_type_ids,
        dispatch_token_type_ids=None,
        offload_helper=None,
    ):
        assert gate_prob is not None
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
        expert_output = expert_output.reshape([-1, expert_output.shape[-1]])
        combined_output = combining_fused(expert_output, combine_weights, scatter_index)

        if self.output_postprocess is not None:
            combined_output = self.output_postprocess(combined_output)
        return combined_output

    def forward_single_stage(self, dispatched_input, stage_id):
        assert isinstance(self.experts, nn.LayerList)
        return self.experts[stage_id](dispatched_input)

    def all2all_expert_overlap(self, x, group):
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
        input,
        token_type_ids=None,
    ):
        if input.ndim == 3:
            orig_shape = input.shape
            input = input.reshape([-1, input.shape[-1]])
        else:
            orig_shape = None
        assert len(input.shape) == 2, f"input Tensor must have dimensions: (s)equence, (d)im, got:{input.shape}"

        hidden_size = input.shape[1]
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
            output += self.gate.weight.sum() * 0.0
            output = output.reshape(orig_shape or orig_shape_2)
            return output, None, 0

        is_first_fwd = not framework._dygraph_tracer()._has_grad
        use_async = self.shared_experts is not None
        gate_input = input

        use_fp8_fuse_node = self.config.use_combine_before_a2a and self.config.use_fp8_fuse_node
        use_quant_before_a2a = self.config.use_quant_before_a2a and use_fp8_fuse_node

        with profile("fused_gate_and_dispatch"):
            fp8_dispatched_handle = None
            if use_quant_before_a2a:
                (
                    dispatched_input,
                    combine_weights,
                    dispatch_mask,
                    scatter_index,
                    router_loss,
                    gate_logits,
                    gate_prob,
                    fp8_dispatched_handle,
                ) = self.gate_distpach_and_quant(gate_input, token_type_ids)
            else:
                (
                    dispatched_input,
                    combine_weights,
                    dispatch_mask,
                    scatter_index,
                    router_loss,
                    gate_logits,
                    gate_prob,
                ) = self.gate_and_distpach(gate_input, token_type_ids)

        if self.config.use_combine_before_a2a:
            assert (
                not self.config.use_ep_comm_overlap
            ), "Dont support `use_ep_comm_overlap` when enable `use_combine_before_a2a`."
            cw_shape = combine_weights.shape
            si_shape = scatter_index.shape
            scatter_index = scatter_index.reshape([-1])

            token_combine_weights = paddle.zeros([cw_shape[0] * cw_shape[1]], dtype=combine_weights.dtype)
            token_combine_weights = paddle.scatter(
                token_combine_weights,
                scatter_index,
                combine_weights.reshape([-1]),
                overwrite=False,
            )

            token_combine_weights = token_combine_weights.reshape([cw_shape[0], cw_shape[1], 1])
            token_combine_weights = AlltoAll.apply(token_combine_weights, self.group)

        if not self.config.use_ep_comm_overlap:
            if use_quant_before_a2a:
                # To enable backward pass overlap, the all-to-all (a2a) operation is performed inside
                # FP8FusedWLCHFunc, eliminating the need for external a2a. However, be careful not
                # to skip the computation of shared_experts.
                shared_out = self.shared_experts(input) if self.shared_experts is not None else None
            else:
                with profile("moe_comm_and_shared_expert"):
                    if use_async:
                        dispatched_input, shared_out = AlltoAllAsync.apply(
                            dispatched_input,
                            input,
                            group=self.group,
                            fn=self.shared_experts,
                            is_first_fwd=is_first_fwd,
                        )
                    else:
                        dispatched_input = AlltoAll.apply(dispatched_input, self.group)

            if use_fp8_fuse_node:
                expert_out = FP8FusedWLCHFunc.apply(
                    dispatched_input,
                    token_combine_weights,
                    self,
                    recompute_fwd_gate_up=recompute_fwd_gate_up_func(self.config, self.layer_idx),
                    dequant_input=("dequant_input" in self.config.fp8_mem_configs)
                    and self.config.fp8_mem_configs["dequant_input"],
                    quant_before_a2a=use_quant_before_a2a,
                    async_a2a=self.config.use_async_a2a,
                    is_first_fwd=not framework._dygraph_tracer()._has_grad,
                    group=self.group,
                    fp8_dispatched_handle=fp8_dispatched_handle,
                )
            else:
                expert_out = (
                    recompute(self.forward_experts, dispatched_input)
                    if self.recompute and self.training
                    else self.forward_experts(dispatched_input)
                )

                if self.config.use_combine_before_a2a:
                    token_combine_weights = token_combine_weights.clone().reshape(expert_out.shape[:-1] + [1])
                    expert_out = expert_out * token_combine_weights
        else:
            assert (
                len(dispatched_input.shape) == 4
                and dispatched_input.shape[1] == self.world_size
                and dispatched_input.shape[0] == self.num_local_experts
            ), (
                f"When using ep_comm_overlap, moe_gate_dispatch_permute is needed. "
                f"Expected dispatched_input to have shape[1] == {self.world_size} "
                f"and shape[0] == {self.num_local_experts}, "
                f"but got shape {dispatched_input.shape}"
            )
            with profile("moe_comm_and_forward_expert"):
                expert_out = AlltoAllExpertOverlap.apply(
                    dispatched_input,
                    self.group,
                    self.num_local_experts,
                    self.experts,
                    is_first_fwd=is_first_fwd,
                )
                if self.shared_experts is not None:
                    shared_out = self.shared_experts(input)

        with profile("moe_comm_and_calc_routerloss"):
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

        with profile("combine"):
            if self.config.use_combine_before_a2a:
                expert_out = expert_out.reshape([-1, hidden_size])
                scatter_index = scatter_index.reshape(si_shape)
                combined_output = paddle.incubate.nn.functional.moe_combine_no_weight(
                    expert_out, combine_weights, scatter_index, epsilon=1e-15
                )
            else:
                combined_output = self.combine_expert_output(expert_out, combine_weights, scatter_index)

        if self.shared_experts is not None:
            combined_output += shared_out

        if orig_shape:
            combined_output = combined_output.clone().reshape(orig_shape[:-1] + [combined_output.shape[-1]])
        return combined_output, combine_weights, router_loss2, gate_logits

    def sharded_state_dict(
        self,
        structured_name_prefix: str = "",
    ):
        sharded_state_dict = super().sharded_state_dict(structured_name_prefix)
        global_expert_id_offset = self.group.rank * self.num_local_experts
        for k, v in sharded_state_dict.items():
            v.global_expert_id_offset = global_expert_id_offset
            sharded_state_dict[k] = v
        return sharded_state_dict


class FP8FusedWLCHFunc(paddle.autograd.PyLayer):
    @staticmethod
    def forward(
        ctx,
        hidden_states,
        dispatched_probs,
        custom_map,
        recompute_fwd_gate_up=False,
        dequant_input=False,
        quant_before_a2a=False,
        async_a2a=False,
        is_first_fwd=False,
        group=None,
        fp8_dispatched_handle=None,
    ):
        ctx.node = ExpertsGroupGemmWLCHNode(
            custom_map,
            recompute_fwd_gate_up=recompute_fwd_gate_up,
            dequant_input=dequant_input,
            group=group,
        )
        ctx.group = group
        ctx.quant_before_a2a = quant_before_a2a
        ctx.async_a2a = async_a2a
        num_local_experts = custom_map.num_local_experts

        def a2a_fn(input_fp8, input_scale):
            return AlltoAll.apply(input_fp8, group), AlltoAll.apply(input_scale, group)

        if quant_before_a2a:
            assert fp8_dispatched_handle is not None
            assert hidden_states.dtype == paddle.float8_e4m3fn
            hidden_states, scale = a2a_fn(hidden_states, fp8_dispatched_handle["scale"])
            scale = scale.reshape([-1, scale.shape[-1]])
        else:
            scale = None

        hidden_states = hidden_states.reshape([-1, hidden_states.shape[-1]])
        dispatched_probs = dispatched_probs.reshape([-1, dispatched_probs.shape[-1]])
        tokens_per_expert = [np.prod(hidden_states.shape[:-1]) // num_local_experts] * num_local_experts

        out = ctx.node.forward(
            hidden_states,
            dispatched_probs,
            tokens_per_expert,
            tokens_per_expert,
            scale=scale,
        )

        if is_first_fwd:
            ctx.node.reset_status()

        return out

    @staticmethod
    def backward(ctx, output_grad):
        if not ctx.quant_before_a2a:
            return ctx.node.backward(output_grad)

        if ctx.async_a2a:

            def a2a_async_fn(input):
                return AlltoAll.apply(input, ctx.group, sync_op=False)

            return ctx.node.backward(output_grad, a2a_async_fn=a2a_async_fn)
        else:
            dx, probs_grad = ctx.node.backward(output_grad)
            return AlltoAll.apply(dx, ctx.group), probs_grad


class MlpNode:
    def __init__(self, custom_map, max_topk, recompute_fwd_gate_up=False, dequant_input=False):
        self.token_dispatcher = custom_map.dispatcher
        self.experts = custom_map.experts
        self.experts_group_gemm_node = ExpertsGroupGemmContiguousNode(
            custom_map,
            recompute_fwd_gate_up=recompute_fwd_gate_up,
            dequant_input=dequant_input,
        )
        self.unzip_node = UnZipNode(self.token_dispatcher)
        self.zip_node = ZipNode(self.token_dispatcher)
        self.dispatched_indices = None
        self.dispatched_probs = None
        self.tokens_per_expert = self.token_dispatcher._comm_manager.tokens_per_expert_list
        self.router_topk = max_topk

    def reset_status(self):
        self.dispatched_indices = None
        self.dispatched_probs = None
        self.tokens_per_expert = None
        self.router_topk = None
        self.experts_group_gemm_node.reset_status()
        self.experts_group_gemm_node = None

    def release_mem(self):
        self.experts_group_gemm_node.reset_status()
        self.experts_group_gemm_node = None

    @paddle.no_grad()
    def forward(self, hs_2d_dispatched, dispatched_indices, dispatched_probs):
        num_experts = len(self.tokens_per_expert)

        self.dispatched_indices = dispatched_indices.to(paddle.int32)
        (unzipped_tokens, zipped_expertwise_rowmap, unzipped_probs) = self.unzip_node.forward(
            hs_2d_dispatched,
            self.dispatched_indices,
            dispatched_probs,
            topk=self.router_topk,
            num_experts=num_experts,
            tokens_per_expert=self.tokens_per_expert,
        )
        hs_2d_dispatched._record_stream()
        dispatched_indices._record_stream()
        dispatched_probs._record_stream()

        padding_token_per_experts = [(x + 127) // 128 * 128 for x in self.tokens_per_expert]
        expert_out = self.experts_group_gemm_node.forward(
            unzipped_tokens,
            unzipped_probs,
            padding_token_per_experts,
            self.tokens_per_expert,
        )

        expert_out_tmp = expert_out.reshape([-1, expert_out.shape[-1]])

        expert_out_zipped = self.zip_node.forward(
            expert_out_tmp,
            zipped_expertwise_rowmap,
            self.dispatched_indices,
            unzipped_probs,
            total_zipped_tokens=hs_2d_dispatched.shape[0],
            num_experts=num_experts,
        )

        self.dispatched_probs = dispatched_probs
        expert_out_zipped.stop_gradient = False

        return expert_out_zipped

    @paddle.no_grad()
    def backward(self, hidden_states_out_grad):
        unzipped_grad = self.zip_node.backward(
            hidden_states_out_grad,
            self.dispatched_indices,
            self.dispatched_probs,
            top_k=self.router_topk,
            num_experts=len(self.tokens_per_expert),
            tokens_per_expert=self.tokens_per_expert,
        )
        hidden_states_out_grad._record_stream()

        expert_out, probs_grad = self.experts_group_gemm_node.backward(unzipped_grad)

        hs_fp8_dispatched_grad, dispatched_probs_grad = self.unzip_node.backward(
            expert_out,
            hidden_states_out_grad,
            probs_grad,
            self.dispatched_indices,
            num_experts=len(self.tokens_per_expert),
        )
        self.reset_status()
        return hs_fp8_dispatched_grad, dispatched_probs_grad


class Fp8FusedMoeFunc(paddle.autograd.PyLayer):
    @staticmethod
    def forward(
        ctx,
        hidden_states,
        dispatched_probs,
        dispatched_indices,
        custom_map,
        max_topk,
        recompute_fwd_gate_up=False,
        dequant_input=False,
        is_first_fwd=False,
    ):
        ctx.node = MlpNode(
            custom_map,
            max_topk,
            recompute_fwd_gate_up=recompute_fwd_gate_up,
            dequant_input=dequant_input,
        )
        out = ctx.node.forward(hidden_states, dispatched_indices, dispatched_probs)

        if is_first_fwd:
            ctx.node.release_mem()
        return out

    @staticmethod
    def backward(ctx, output_grad):
        hidden_states_grad, dispatched_probs_grad = ctx.node.backward(output_grad)
        return hidden_states_grad, dispatched_probs_grad, None
