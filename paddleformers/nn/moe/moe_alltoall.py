import paddle
import paddle.distributed as dist
import paddle.nn.functional as F
from paddle import Tensor, _C_ops, framework, nn
from paddle.autograd import PyLayer
from paddle.distributed import fleet
from paddle.distributed.communication import stream
from paddle.distributed.communication.group import Group
from paddle.distributed.fleet.utils import recompute
from paddle.incubate.nn.functional import moe_combine, moe_gate_dispatch
from paddleformers.utils.log import logger
from paddleformers.transformers.ernie4_5.sequence_parallel_utils import ScatterOp

from .utils import combine_expert_output, _calc_router_loss
from .alltoall import AlltoAll, AlltoAllAsync

def moe_alltoall_forward(
    input: Tensor,
    token_type_ids=None,
    config,
    gate: nn.Layer,
    k,
    use_correction_bias,
    moe_statics,
    world_size,
    num_local_experts,
    shared_experts=None,
    group: Group = None,
    experts: List[nn.Layer],
    rank,
    isRecompute,
    isTraining,
    layer_idx,
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
    assert (
        len(input.shape) == 2
    ), f"input Tensor must have dimensions: (s)equence, (d)im, got:{input.shape}"
    if token_type_ids is not None:
        token_type_ids = token_type_ids.clone()[:, :-1]
        if config.sequence_parallel:
            token_type_ids = token_type_ids.reshape([-1])
            token_type_ids = ScatterOp.apply(token_type_ids)
            token_type_ids.stop_gradient = True

    assert gate is not None

    # if hasattr(self, "rng") and self.rng.random() < self.all_to_all_dropout:
    #     orig_shape_2 = input.shape
    #     output = self.forward_experts(input)
    #     output += self.gate.weight.sum() * 0.0  # hack for grad
    #     output = output.reshape(orig_shape or orig_shape_2)  # [e*1,c,m]
    #     return output, None, 0

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
    ) = gate_and_dispatch(gate_input, token_type_ids, gate, k, config, use_correction_bias, moe_statics, world_size, num_local_experts)

    use_async = shared_experts is not None
    if use_async:
        dispatched_input, shared_out = AlltoAllAsync.apply(
            dispatched_input,
            input,  # args to shared-experts
            group=group,
            fn=shared_experts,
            is_first_fwd=is_first_fwd,
        )
    else:
        dispatched_input = AlltoAll.apply(dispatched_input, group=group)

    expert_out = (
        recompute(forward_experts, dispatched_input, experts, rank, num_local_experts, world_size)
        if isRecompute and isTraining
        else forward_experts(dispatched_input, experts, rank, num_local_experts, world_size)
    )

    expert_out, router_loss2 = AlltoAllAsync.apply(
        expert_out,
        router_loss,
        combine_weights,
        dispatch_mask,
        gate_logits,
        gate_prob,
        token_type_ids,
        gate,
        layer_idx,
        group=group,
        fn=calc_router_loss_and_logging,
        is_first_fwd=is_first_fwd,
    )

    combined_output = combine_expert_output(
        expert_out, combine_weights, scatter_index
    )

    if shared_experts is not None:
        combined_output += shared_out

    if orig_shape:
        combined_output = combined_output.clone().reshape(
            orig_shape[:-1] + [combined_output.shape[-1]]
        )
    return combined_output, combine_weights, router_loss2, gate_logits

def calc_router_loss_and_logging(
        self,
        router_loss,
        combine_weights,
        dispatch_mask,
        gate_logits,
        gate_prob,
        token_type_ids=None,
        gate: nn.Layer,
        dispatch_token_type_ids=None,
        offload_helper=None,
        layer_idx,
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
    if token_type_ids is not None and gate.config.moe_use_hard_gate:  # true
        if not gate.weight.stop_gradient:
            lm_tokens_mask = token_type_ids == 0
            if offload_helper is not None:
                is_lm = offload_helper["lm_mask"][1]
            else:
                is_lm = lm_tokens_mask.any()
            if is_lm:
                dispatch_tokens_mask = (
                    dispatch_token_type_ids == 0
                    if dispatch_token_type_ids is not None
                    else None
                )
                router_loss += _calc_router_loss(
                    (
                        dispatch_mask[gate.experts_type_mask[0]]
                        if hasattr(gate, "experts_type_mask")
                        else dispatch_mask
                    ),
                    (
                        gate_logits[:, gate.experts_type_mask[0]]
                        if hasattr(gate, "experts_type_mask")
                        else gate_logits
                    ),
                    (
                        gate_prob[:, gate.experts_type_mask[0]]
                        if hasattr(gate, "experts_type_mask")
                        else gate_prob
                    ),
                    (
                        gate.num_experts_list[0]
                        if hasattr(gate, "num_experts_list")
                        else gate.num_experts_tensor
                    ),
                    False, # self.group_experts,
                    layer_idx,
                    0,
                    lm_tokens_mask,
                    dispatch_tokens_mask,
                    prefix="lm",
                    gate=gate
                )
        # mm_tokens_mask = token_type_ids == 1
        # if offload_helper is not None:
        #     is_mm = offload_helper["mm_mask"][1]
        # else:
        #     is_mm = mm_tokens_mask.any()
        # if is_mm:
        #     dispatch_tokens_mask = (
        #         dispatch_token_type_ids == 1
        #         if dispatch_token_type_ids is not None
        #         else None
        #     )
        #     router_loss += self._calc_router_loss(
        #         dispatch_mask[self.gate.experts_type_mask[1]],
        #         gate_logits[:, self.gate.experts_type_mask[1]],
        #         gate_prob[:, self.gate.experts_type_mask[1]],
        #         self.gate.num_experts_list[1],
        #         False,
        #         self.layer_idx,
        #         1,
        #         mm_tokens_mask,
        #         dispatch_tokens_mask,
        #         prefix="mm",
        #     )

    else:
        router_loss += _calc_router_loss(
            dispatch_mask,
            gate_logits,
            gate_prob,
            gate.num_experts_tensor,
            False,# self.group_experts,
            layer_idx,
            gate=gate
        )

    return router_loss

def forward_experts(
    dispatched_input, 
    experts: List[nn.Layer],
    rank,
    num_local_experts,
    world_size,
    ):
    """
    Forward pass through experts sequentially.

    Args:
        dispatched_input: Input tensor of shape [num_experts, capacity, dim]

    Returns:
        Tensor: Expert outputs of shape [num_experts, capacity, dim]
    """

    # if not self.multimodal_experts:
    #     true_experts = self.experts[
    #         self.rank
    #         * self.num_local_experts : (self.rank + 1)
    #         * self.num_local_experts
    #     ]
    # else:
    #     true_experts = []
    #     for i, num in enumerate(self.num_local_multimodal_experts):
    #         current_modal_experts = self.experts[
    #             self.multimodal_expert_index[i] : self.multimodal_expert_index[
    #                 i + 1
    #             ]
    #         ]
    #         true_experts.extend(
    #             current_modal_experts[self.rank * num : (self.rank + 1) * num]
    #         )
    true_experts = experts[
        rank
        * num_local_experts : (rank + 1)
        * num_local_experts
    ]

    dispatched_input = dispatched_input.reshape(
        [world_size, num_local_experts, -1, dispatched_input.shape[-1]]
    )  # [e,1,c,m]
    expert_outputs = []
    if isinstance(experts, nn.LayerList):
        chunks = dispatched_input.transpose([1, 0, 2, 3]).contiguous().unbind(0)
        assert len(chunks) == len(true_experts), (len(chunks), len(true_experts))
        for chunk, expert in zip(chunks, true_experts):
            expert_outputs += [expert(chunk)]
    else:
        dispatched_input = dispatched_input.transpose([1, 0, 2, 3])
        dispatched_input.contiguous()
        orig_shape = dispatched_input.shape
        chunks = dispatched_input.reshape([orig_shape[0], -1, orig_shape[-1]])
        chunks = experts(chunks)
        chunks = chunks.reshape(orig_shape[:-1] + [chunks.shape[-1]]).unbind(0)
        expert_outputs += chunks
    expert_output = paddle.stack(expert_outputs, axis=1)  # [ecm]
    return expert_output

def gate_and_dispatch(
    input,
    token_type_ids=None,
    gate: nn.Layer,
    k,
    config,
    use_correction_bias,
    moe_statics,
    world_size,
    num_local_experts
    ):
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
    ) = gate(input, *args)
    # if self.input_preprocess is not None:
    #     input, gate_logits = self.input_preprocess(input, gate_logits, capacity)
    # capacity no use
    # k = self.k
    prob, max_prob = fused_gate_logits_process(
        gate_logits=gate_logits, 
        token_type_ids=token_type_ids,
        k=k,
        gate=gate,
        config=config)

    if "corr_bias" in inspect.signature(moe_gate_dispatch).parameters:
        if use_correction_bias:
            compat_args = (moe_statics.e_score_correction_bias[0],)
        else:
            compat_args = (None,)
    else:
        assert (
            not use_correction_bias
        ), "correction bias not supported, rebuild moe-ops"
        compat_args = ()

    (
        dispatched_input,
        combine_weights_unnorm,
        scatter_index,
        dispatch_mask,
        _,
    ) = moe_gate_dispatch(
        input, prob, *compat_args, k=k, capacity=capacity, use_pad=True
    )
    dispatched_input = dispatched_input.astype(input.dtype)

    dispatch_mask = paddle.diff(F.pad(dispatch_mask, (1, 0)))
    if use_correction_bias:
        if gate.config.multimodel_experts:
            for i in range(len(moe_statics.expert_usage)):
                moe_statics.expert_usage[i] += dispatch_mask[
                    gate.experts_type_mask[i]
                ].detach()
        else:
            moe_statics.expert_usage[0] += dispatch_mask.detach()
    dispatched_input.stop_gradient = False
    combine_weights_unnorm.stop_gradient = False
    scatter_index.stop_gradient = True
    dispatch_mask.stop_gradient = True

    scatter_index = scatter_index.transpose([1, 0])  # [k,s] ->[s,k]
    # if self.group_experts:
    #     if max_prob is not None:
    #         if token_type_ids is not None:
    #             p = paddle.ones_like(combine_weights_unnorm.unsqueeze(-1))
    #             p = paddle.scatter_nd_add(
    #                 p, paddle.nonzero(token_type_ids == 0), -1 + max_prob
    #             )
    #         else:
    #             p = max_prob
    #         combine_weights_unnorm = (
    #             combine_weights_unnorm.unsqueeze(-1) * p
    #         ).squeeze(-1)
    #         # gate_prob 进行还原
    #         prob = (prob.reshape([p.shape[0], k, -1]) * p).reshape([p.shape[0], -1])
    if gate.norm_gate_logits:
        combine_weights = combine_weights_unnorm / paddle.clip(
            combine_weights_unnorm.sum(-1, keepdim=True), min=1e-12
        )
    else:
        combine_weights = combine_weights_unnorm
    combine_weights = combine_weights.cast(dispatched_input.dtype)

    dispatched_input = dispatched_input.reshape(
        [world_size * num_local_experts, capacity, d_model]
    )
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

def fused_gate_logits_process(
    gate_logits, 
    token_type_ids=None,
    offload_helper=None,
    k,
    gate:nn.Layer,
    config,

):
    """
    Process and combine gate logits.

    Args:
        gate_logits: Raw gate logits

    Returns:
        tuple: (processed probabilities, max probabilities)
    """
    experts_type_ids = gate.experts_type_ids
    use_hard_gate = config.moe_use_hard_gate
    max_prob = None

    if token_type_ids is not None and use_hard_gate:
        if offload_helper is None:
            offload_helper = dict()
            lm_mask = token_type_ids == 0
            is_lm = lm_mask.any()
            mm_mask = token_type_ids == 1
            # is_mm = mm_mask.any()
            seq_lm = lm_mask.sum()
            seq_mm = mm_mask.sum()
            lm_mask = lm_mask.unsqueeze(1) & (experts_type_ids == 0).unsqueeze(0)
            mm_mask = mm_mask.unsqueeze(1) & (experts_type_ids == 1).unsqueeze(0)
            offload_helper["lm_mask"] = [lm_mask, is_lm, seq_lm]
            # offload_helper["mm_mask"] = [mm_mask, is_mm, seq_mm]

        is_lm = offload_helper["lm_mask"][1]
        prob = paddle.zeros_like(gate_logits)
        # 处理 lm_prob
        if is_lm:
            lm_mask = offload_helper["lm_mask"][0]
            seq_lm_cpu = offload_helper["lm_mask"][2]
            lm_mask_nonzero = lm_mask.nonzero()
            lm_partial_gate_logits = gate_logits.gather_nd(lm_mask_nonzero).reshape(
                [seq_lm_cpu, -1]
            )
            # if self.group_experts:
            #     lm_prob = self.gate.act(
            #         lm_partial_gate_logits.reshape(
            #             [lm_partial_gate_logits.shape[0], k, -1]
            #         )
            #     )
            #     max_prob = lm_prob.max(-1, keepdim=True)  # [s_l, k, 1]
            #     lm_prob /= max_prob
            # else:
            lm_prob = gate.act(lm_partial_gate_logits)
            prob = paddle.scatter_nd_add(prob, lm_mask_nonzero, lm_prob.flatten())
        # 处理 mm_prob
        # is_mm = offload_helper["mm_mask"][1]
        # if is_mm:
        #     mm_mask = offload_helper["mm_mask"][0]
        #     seq_mm_cpu = offload_helper["mm_mask"][2]
        #     mm_mask_nonzero = paddle.nonzero(mm_mask)
        #     mm_partial_gate_logits = gate_logits.gather_nd(mm_mask_nonzero).reshape(
        #         [seq_mm_cpu, -1]
        #     )
        #     mm_prob = gate.act(mm_partial_gate_logits)
        #     prob = paddle.scatter_nd_add(prob, mm_mask_nonzero, mm_prob.flatten())
    else:
        # 处理非硬门和不需要token_type_ids的情况
        # if self.group_experts:
        #     prob = self.gate.act(gate_logits.reshape([gate_logits.shape[0], k, -1]))
        #     max_prob = prob.max(-1, keepdim=True)
        #     prob /= max_prob
        #     prob = prob.reshape([prob.shape[0], -1])
        # else:
        prob = gate.act(gate_logits)
    return prob, max_prob