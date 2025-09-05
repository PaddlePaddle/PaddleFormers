import inspect
from typing import Callable, Dict, List, Optional, Tuple

import paddle
import paddle.distributed as dist
from paddle import framework, nn
from paddle.autograd import PyLayer
from paddle.distributed import fleet
from paddle.distributed.communication.group import Group, _get_global_group
from paddle.distributed.fleet.utils import recompute
from paddle.incubate.nn.functional import (
    build_src_rank_and_local_expert_id,
    expand_modality_expert_id,
    moe_gate_dispatch_partial_nosoftmaxtopk,
)
from paddle.incubate.tensor.manipulation import async_offload
from paddleformers.peft.lora.lora_quantization_layers import QuantizationLoRALinear
from paddleformers.utils.log import logger

from paddleformers.transformers.ernie4_5.distributed.common_dist_utils import (
    AllGatherGroupOp,
    ReduceScatterGroupOp,
    all_gather_group,
    get_async_loader,
    hack_offload_wait,
    reduce_scatter_group,
)

from .utils import manual_backward, combine_expert_output, _calc_router_loss, ReshardCombineWeight
from .allgather import AllGatherAsync
from .alltoall_smart import AlltoAllSmart

def moe_allgather_forward(
    input: paddle.Tensor,
    token_type_ids=None,
    use_dense_expert=False,
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
    dense_token_type,
) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """Implements forward pass for Mixture-of-Experts (MoE) layer with distributed communication.

    Core Functionality:
        - Processes input through gating network to determine expert assignments
        - Performs distributed All-to-All communication for expert computation
        - Combines expert outputs and calculates routing loss

    Key Features:
        1. Supports both dense and sparse expert computation modes
        2. Implements fused gating and dispatch for performance optimization
        3. Handles sequence length padding/unpadding for irregular inputs
        4. Enables communication-computation overlap through asynchronous operations

    Args:
        input (Tensor): Input tensor of shape [seq_len, hidden_dim]
        token_type_ids: Optional segmentation markers for heterogeneous inputs
        use_dense_expert: Flag to enable dense expert computation bypass

    Returns:
        tuple: (
            combined_output: Aggregated expert outputs [seq_len, hidden_dim],
            combine_weights: Expert combination coefficients,
            router_loss: Calculated router balancing loss
        )
    """
    if input.ndim == 3:
        orig_shape = input.shape
        input = input.reshape([-1, input.shape[-1]])
    else:
        orig_shape = None

    assert (
        len(input.shape) == 2
    ), f"input Tensor must have dimensions: (s)equence, (d)im, got:{input.shape}"
    dispatch_token_type_ids = None
    global_dense_expert_mask = None
    if token_type_ids is not None:
        token_type_ids = token_type_ids[:, :-1].reshape([-1])
        dispatch_token_type_ids = token_type_ids
        if config.sequence_parallel:
            hcg = fleet.get_hybrid_communicate_group()
            rank = hcg.get_model_parallel_rank()
            interval = (
                token_type_ids.shape[0] // hcg.get_model_parallel_world_size()
            )
            token_type_ids = token_type_ids.slice(
                [0], rank * interval, (rank + 1) * interval
            )
            token_type_ids.stop_gradient = True

        if use_dense_expert:
            global_dense_expert_mask = (
                dispatch_token_type_ids == dense_token_type
            )

    assert gate is not None
    # if hasattr(self, "rng") and self.rng.random() < self.all_to_all_dropout:
    #     orig_shape_2 = input.shape
    #     output = self.forward_experts(input)
    #     output += self.gate.weight.sum() * 0.0  # hack for grad
    #     output = output.reshape(orig_shape or orig_shape_2)  # [e*1,c,m]
    #     return output, None, 0
    (
        dispatched_input,
        global_hidden_states,
        local_combine_weights,
        expert_num_global_no_token_drop,
        expert_num_global,
        expert_num_global_list,
        local_scatter_index,
        scatter_index_rev,
        router_loss,
        (gate_logits, gate_prob),
        (gate_logits_mm, gate_prob_mm),
        expert_num_local,
    ) = fused_gate_and_dispatch(
        input=input, 
        token_type_ids=token_type_ids, 
        global_dense_expert_mask=global_dense_expert_mask,
        gate=gate,
        k=k,
        use_correction_bias=use_correction_bias,
        moe_statics=moe_statics,
        world_size=world_size,
        num_local_experts=num_local_experts,
        shared_experts=shared_experts,
        group=group,
        experts=experts,
        rank=rank,
        isRecompute=isRecompute,
        isTraining=isTraining,
    )
    seqlen_this_mp = input.shape[0]
    if len(scatter_index_rev):
        recv_rank_local = scatter_index_rev // seqlen_this_mp
    else:
        recv_rank_local = scatter_index_rev

    # if self.use_padding:
    #     if self.send_rank is None:
    #         capacity = self.gate.get_capacity(
    #             input.shape[0] * self.config.moe_world_size
    #         )
    #         self.send_rank = (
    #             paddle.arange(self.config.moe_world_size)
    #             .repeat_interleave(capacity * self.num_local_experts)
    #             .astype("int32")  # cap
    #         )
    #         self.local_expert_id = (
    #             paddle.arange(self.num_local_experts)
    #             .repeat_interleave(capacity)
    #             .tile(self.config.moe_world_size)
    #             .astype(self.send_rank.dtype)
    #         )
    #     recv_rank, recv_rank_task = allgather_async(
    #         recv_rank_local, group=self.config.moe_group
    #     )
    #     send_rank = self.send_rank
    #     local_expert_id = self.local_expert_id

    # else:
    all_expert_num = sum(expert_num_global_list)
    # 非常慢
    if config.moe_group.nranks > 1:
        recv_rank = paddle.empty([all_expert_num], dtype=recv_rank_local.dtype)
        # 非常慢
        recv_rank_task = dist.stream.alltoall_single(
            recv_rank,
            recv_rank_local.tile(config.moe_world_size),
            [
                sum(
                    expert_num_global_list[
                        i
                        * num_local_experts : (i + 1)
                        * num_local_experts
                    ]
                )
                for i in range(config.moe_world_size)
            ],  # output-size
            [len(recv_rank_local)] * config.moe_world_size,  # input-size
            group=config.moe_group,
            sync_op=False,
            use_calc_stream=False,
        )
    else:
        recv_rank_task = None
        recv_rank = recv_rank_local.tile(config.moe_world_size)

    send_rank, local_expert_id = build_src_rank_and_local_expert_id(
        expert_num_global, expert_num_global_list, num_local_experts
    )

    # if not self.use_expert_out_alltoall:
    #     expert_outs = (
    #         recompute(self.forward_experts, *dispatched_input)
    #         if self.recompute and self.training
    #         else self.forward_experts(*dispatched_input)
    #     )
    #     expert_outs = paddle.concat(
    #         [e for e in expert_outs if e is not None], axis=0
    #     )  # [e*c,m]
    #     expert_out_to_combine = AllGatherGroupOp.apply(
    #         expert_outs, group=self.config.moe_group
    #     )  # for test
    #     router_loss2 = self.calc_router_loss_and_logging(
    #         router_loss,
    #         gate_logits,
    #         gate_prob,
    #         gate_logits_mm,
    #         gate_prob_mm,
    #         local_combine_weights,
    #         expert_num_global_no_token_drop,
    #         token_type_ids,
    #         dispatch_token_type_ids,
    #     )
    # else:
    recv_rank_task and recv_rank_task.wait()  # wait for recv_rank

    world_size = dist.get_world_size(config.moe_group)
    this_rank = dist.get_rank(config.moe_group)

    recv_size = paddle.count_nonzero(
        recv_rank == dist.get_rank(config.moe_group)
    )
    recv_size = paddle.maximum(
        recv_size, paddle.ones([], dtype=recv_size.dtype)
    )

    recv_size_cpu, recv_size_task = async_offload(recv_size, get_async_loader())

    send_rank_this_rank = paddle.count_nonzero(send_rank == this_rank)

    send_rank_this_rank_cpu, send_rank_this_rank_task = async_offload(
        send_rank_this_rank, get_async_loader()
    )

    recv_rank[recv_rank == -1] = world_size
    send_recv_count_global = paddle.scatter_nd_add(
        paddle.zeros(
            [num_local_experts, world_size + 1, world_size + 1],
            dtype="int32",
        ),
        paddle.stack([local_expert_id, send_rank, recv_rank], -1),
        paddle.ones([len(send_rank)], dtype="int32"),
    )  # [num_local_experts, world_size + 1 , world_size + 1]
    send_counts_cpu = send_recv_count_global[:, this_rank, :-1].numpy()
    recv_counts_cpu = send_recv_count_global[:, :-1, this_rank].numpy()
    send_counts_num_cpu = send_counts_cpu.sum(-1)
    recv_counts_num_cpu = recv_counts_cpu.sum(-1)

    dispatched_input = forward_experts(*dispatched_input, experts, rank, num_local_experts)

    if recv_size_task is not None:
        recv_size_task.cpu_wait()
    if send_rank_this_rank_task is not None:
        send_rank_this_rank_task.cpu_wait()

    input_size = sum([len(i) if i is not None else 0 for i in dispatched_input])
    # if self.use_padding or input_size > 1:
    if input_size > 1:
        assert send_rank_this_rank_cpu.item() == input_size, (
            send_rank,
            [len(i) if i is not None else 0 for i in dispatched_input],
        )

    expert_out_to_combine, router_loss2, distributed_input_to_alltoall_out = (
        AlltoAllSmart.apply(
            *dispatched_input,
            router_loss,
            gate_logits,
            gate_prob,
            gate_logits_mm,
            gate_prob_mm,
            local_combine_weights,
            expert_num_global_no_token_drop,
            token_type_ids,
            dispatch_token_type_ids,
            gate,
            layer_idx,
            forward_func_dict=None,
            router_loss_fn=calc_router_loss_and_logging,
            local_expert_id=local_expert_id,
            send_rank_global=send_rank,
            recv_rank_global=recv_rank,
            num_local_experts=num_local_experts,
            # capacity=dispatched_input[0].shape[1] if self.use_padding else None,
            capacity=None,
            use_padding=False,#self.use_padding,
            expert_num_global=expert_num_global_list,
            is_first_fwd=not framework._dygraph_tracer()._has_grad,
            group=config.moe_group,
            recv_size=recv_size_cpu,
            send_counts=send_counts_cpu,
            recv_counts=recv_counts_cpu,
            send_counts_num=send_counts_num_cpu,
            recv_counts_num=recv_counts_num_cpu,
        )
    )
    # /origin input -> distributed input/ => /origin-input -> alltoall out -input/
    local_scatter_index = distributed_input_to_alltoall_out[local_scatter_index]
    local_scatter_index.stop_gradient = True

    # global -> local
    combined_output = combine_expert_output(
        expert_out_to_combine, local_combine_weights, local_scatter_index
    )

    if shared_experts is not None:
        shared_out = shared_experts(input)
        combined_output += shared_out

    if orig_shape:
        combined_output = combined_output.reshape(
            orig_shape[:-1] + [combined_output.shape[-1]]
        )

    return combined_output, local_combine_weights, router_loss2, gate_logits

def forward_experts(
    dispatched_input, 
    experts: List[nn.Layer],
    rank,
    num_local_experts
    ):
    """Execute expert model computations in sequence for Mixture-of-Experts (MoE) layer.

    Core Functionality:
        - Distributes dispatched tokens to local expert models
        - Handles empty expert inputs with zero-initialized fallback
        - Maintains gradient flow for expert outputs
        - Aggregates outputs from all active experts

    Args:
        *dispatched_input: Variable-length expert-specific input tensors

    Returns:
        list: Expert output tensors (None for inactive experts)

    Implementation Details:
        1. Processes valid expert inputs through corresponding expert models
        2. Generates dummy inputs for inactive experts to preserve model structure
        3. Aggregates dummy outputs to first active expert to maintain gradient flow
    """
    expert_outputs = []
    assert isinstance(experts, nn.LayerList), type(experts)

    no_tokens_expert_outputs = []
    # if not self.multimodal_experts:
    true_experts = experts[
        rank
        * num_local_experts : (rank + 1)
        * num_local_experts
    ]
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

    assert len(dispatched_input) == len(true_experts), (
        len(dispatched_input),
        len(true_experts),
    )

    for iexpert, chunk in enumerate(dispatched_input):
        if chunk is None:
            # QuantizationLoRALinear can not call `.weight`.
            if not isinstance(
                true_experts[iexpert].up_gate_proj, QuantizationLoRALinear
            ):
                input_shape = [
                    1,
                    true_experts[iexpert].up_gate_proj.weight.shape[0],
                ]
                input_dtype = true_experts[iexpert].up_gate_proj.weight.dtype
            else:
                input_shape = [
                    1,
                    true_experts[iexpert].up_gate_proj.lora_A.shape[0],
                ]
                input_dtype = true_experts[iexpert].up_gate_proj.lora_A.dtype

            chunk = paddle.zeros(
                input_shape,
                input_dtype,
            )
            if true_experts[iexpert].training:
                chunk.stop_gradient = False
            expert_out = true_experts[iexpert](chunk.contiguous())
            no_tokens_expert_outputs.append(
                expert_out * 0.0
            )  # mutiply 0.0 to zero out and grad

            expert_outputs.append(None)
            continue

        expert_out = true_experts[iexpert](chunk.contiguous())
        expert_outputs.append(expert_out)

    # if self.config.moe_layer_feed_fake_token and len(no_tokens_expert_outputs) > 0:
    if len(no_tokens_expert_outputs) > 0:
        first_has_tokens_idx = 0
        for idx, expert_out in enumerate(expert_outputs):
            if expert_out is not None:
                first_has_tokens_idx = idx
                break
        for idx, expert_out in enumerate(no_tokens_expert_outputs):
            expert_outputs[first_has_tokens_idx] += expert_out

    return expert_outputs

def fused_gate_logits_process_fused(
    gate_logits_lm, 
    gate_logits_mm=None, 
    token_type_ids=None,
    k,
    gate,
    use_correction_bias,
    moe_statics
):
    """Process gating logits for expert selection in Mixture-of-Experts (MoE) layers.

    Core Functionality:
    - Transforms raw gating logits into expert selection weights and IDs
    - Supports both grouped and standard expert selection modes
    - Handles bias correction for improved expert load balancing

    Args:
        gate_logits_lm (Tensor): Raw gating scores of shape [batch_size, total_experts]

    Returns:
        tuple: (
            lm_weight_and_expert_id: Combined tensor containing selection weights
                    and expert IDs [batch_size, 2*top_k],
            prob_flat: Flattened expert probabilities [batch_size, total_experts]
        )
    """
    top_k = k
    num_expert_per_rank_per_modality = (
        gate_logits_lm.shape[-1] // gate.config.moe_world_size
    )
    group_size = gate_logits_lm.shape[-1] // top_k
    # if self.group_experts:
    #     assert not self.use_correction_bias
    #     gate_logits_lm = gate_logits_lm.reshape(
    #         [gate_logits_lm.shape[0], top_k, -1]
    #     )
    #     prob_lm = self.gate.act(gate_logits_lm)
    #     prob_lm_ = prob_lm
    #     weight_lm, expert_id_lm = prob_lm_.topk(k=1, axis=-1)
    #     weight_lm = weight_lm.reshape([gate_logits_lm.shape[0], -1])
    #     group_size = gate_logits_lm.shape[-1]
    #     expert_id_lm = expert_id_lm.squeeze(-1)
    # else:
    prob_lm = gate.act(gate_logits_lm)
    if use_correction_bias:
        prob_lm_ = (
            prob_lm + moe_statics.e_score_correction_bias[0].detach()
        )
    else:
        prob_lm_ = prob_lm
    weight_lm, expert_id_lm = prob_lm_.topk(k=top_k, axis=-1)

    if use_correction_bias:
        batch_idx = (
            paddle.arange(prob_lm_.shape[0]).unsqueeze(-1).expand_as(expert_id_lm)
        )
        weight_lm = prob_lm[batch_idx, expert_id_lm]  # use correct bias

    expert_id_lm = expand_modality_expert_id(
        expert_id_lm,
        num_expert_per_modality=(
            num_expert_per_rank_per_modality if token_type_ids is not None else 0
        ),
        group_size=group_size,
        modality_offset=0,
        is_group_expert=False, # self.group_experts,
    )
    expert_id_lm = expert_id_lm.reshape(weight_lm.shape)
    lm_weight_and_expert_id = paddle.concat(
        [weight_lm, expert_id_lm.astype("float32")], -1
    )

    if token_type_ids is None or gate_logits_mm is None:
        return (
            lm_weight_and_expert_id,
            prob_lm.reshape([prob_lm.shape[0], -1]),
            None,
        )

    prob_mm = gate.act(gate_logits_mm)
    if use_correction_bias:
        prob_mm_ = prob_mm + moe_statics.e_score_correction_bias[1].detach()
    else:
        prob_mm_ = prob_mm
    weight_mm, expert_id_mm = prob_mm_.topk(k=top_k, axis=-1)
    if use_correction_bias:
        batch_idx = (
            paddle.arange(prob_lm_.shape[0]).unsqueeze(-1).expand_as(expert_id_lm)
        )
        weight_mm = prob_mm[batch_idx, expert_id_mm]  # use correct bias

    expert_id_mm = expand_modality_expert_id(
        expert_id_mm,
        num_expert_per_modality=num_expert_per_rank_per_modality,
        group_size=group_size,
        modality_offset=1,
        is_group_expert=False,
    )
    expert_id_mm = expert_id_mm.reshape(weight_mm.shape)
    mm_weight_and_expert_id = paddle.concat(
        [weight_mm, expert_id_mm.astype("float32")], -1
    )
    weight_and_expert = paddle.where(
        (token_type_ids == 0).unsqueeze(-1),
        lm_weight_and_expert_id,
        mm_weight_and_expert_id,
    )
    return weight_and_expert, prob_lm.reshape([prob_lm.shape[0], -1]), prob_mm

def fused_gate_and_dispatch(
        input, 
        token_type_ids=None, 
        global_dense_expert_mask=None,
        gate=gate,
        k=k,
        use_correction_bias=use_correction_bias,
        moe_statics=moe_statics,
        world_size=world_size,
        num_local_experts=num_local_experts,
    ):
    """Implements fused expert gating and token dispatch logic for Mixture-of-Experts (MoE) layers.

    Core Functionality:
        - Computes expert selection probabilities and routing weights
        - Performs distributed token-to-expert assignment
        - Handles communication and synchronization in model-parallel environments

    Args:
        input (Tensor): Input tensor of shape [seq_len, hidden_dim]

    Returns:
        tuple: (
            dispatched_input: Expert-assigned tokens [num_experts, capacity, hidden_dim],
            global_hidden_states: Full sequence representations,
            local_combine_weights: Local expert combination weights,
            expert_num_global_notrunc: Global expert token counts (without capacity truncation),
            expert_num_global: Actual expert token counts,
            expert_num_global_list: Per-expert token counts,
            local_scatter_index: Local token reorganization indices,
            scatter_index_rev: Reverse scattering indices,
            router_loss: Calculated routing loss,
            gate_outputs: Raw gating network outputs,
            expert_num_local: Local expert utilization counts
        )
    """
    seqlen, d_model = input.shape
    args = ()
    if token_type_ids is not None:
        token_type_ids = token_type_ids.reshape([-1])
        args = (token_type_ids,)

    router_loss = paddle.zeros([1], dtype="float32")
    router_loss.stop_gradient = False
    top_k = k

    def build_weights_and_expert_id(
        input,
        gate,
        k,
        use_correction_bias,
        moe_statics):
        nonlocal token_type_ids, args
        logits, capacity, router_loss = gate(
            input, *args, transform_weight=False
        )
        # if gate.config.multimodel_experts:
        #     gate_logits_lm, gate_logits_mm = logits.chunk(2, axis=-1)
        # else:
        gate_logits_lm, gate_logits_mm = logits, None

        weigth_and_expert, gate_prob_lm, gate_prob_mm = (
            fused_gate_logits_process_fused(
                gate_logits_lm,
                gate_logits_mm,
                token_type_ids if global_dense_expert_mask is None else None,
                k=k,
                gate=gate,
                use_correction_bias=use_correction_bias,
                moe_statics=moe_statics,
            )
        )
        weigth_and_expert = AllGatherGroupOp.apply(
            weigth_and_expert, group=gate.config.moe_group
        )
        return (
            weigth_and_expert,
            gate_logits_lm,
            gate_logits_mm,
            gate_prob_lm,
            gate_prob_mm,
        )

    capacity = gate.get_capacity(input.shape[0]) * world_size
    (
        global_hidden_states,
        combine_weights_and_expert_id,
        gate_logits_lm,
        gate_logits_mm,
        gate_prob_lm,
        gate_prob_mm,
    ) = AllGatherAsync.apply(
        input,
        input,
        gate,
        k,
        use_correction_bias,
        moe_statics,
        fn=build_weights_and_expert_id,
        group=gate.config.moe_group,
        is_first_fwd=not framework._dygraph_tracer()._has_grad,
    )
    combine_weights_unnorm, expert_id = combine_weights_and_expert_id.chunk(
        2, axis=-1
    )
    expert_id = expert_id.cast("int32")
    expert_id.stop_gradient = True
    num_experts = (
        sum(gate.config.moe_num_experts)
        if isinstance(gate.config.moe_num_experts, (tuple, list))
        else gate.config.moe_num_experts
    )  # all-experts = 96
    if global_dense_expert_mask is not None:
        combine_weights_unnorm[global_dense_expert_mask] = 0.0
        expert_id[global_dense_expert_mask] = num_experts
        num_experts += 1

    if (
        "reverse_token_drop"
        in inspect.signature(moe_gate_dispatch_partial_nosoftmaxtopk).parameters
    ):
        # compat_kwargs = {"reverse_token_drop": self.enable_reverse_token_drop}
        compat_kwargs = {"reverse_token_drop": False}
    else:
        compat_kwargs = {}

    # Disable AMP because:
    # - combine_weights_unnorm is fp32, global_hidden_states is bf16
    # - AMP O2 would upcast global_hidden_states to fp32, making dispatched_input fp32
    # - This is a data movement op with no computation, so upcasting is unnecessary
    with paddle.amp.auto_cast(False):
        (
            dispatched_input,
            combine_weights_unnorm,
            scatter_index,  # input -> dispatched_input
            scatter_index_rev,  # dispatch-input -> input
            expert_num_global,
            expert_num_local,
        ) = moe_gate_dispatch_partial_nosoftmaxtopk(
            global_hidden_states,
            combine_weights_unnorm,
            expert_id,
            top_k,
            capacity,
            num_experts,
            False, # self.use_padding,
            expert_start_index=num_local_experts * gate.config.moe_rank,
            expert_end_index=num_local_experts * (gate.config.moe_rank + 1),
            **compat_kwargs,
        )

    if use_correction_bias:
        # if gate.config.multimodel_experts:
        #     # MLLM
        #     for i in range(len(self.moe_statics.expert_usage)):
        #         self.moe_statics.expert_usage[i] += expert_num_local[
        #             self.gate.experts_type_mask[i]
        #         ].detach()
        # else:
        #     # LLM
        moe_statics.expert_usage[0] += expert_num_local.detach()

    # When use unpad , `moe_ops_partial` output likes `scatter_index_rev==[]`.
    if scatter_index_rev.ndim == 0:
        # assert not self.use_padding
        scatter_index_rev = paddle.empty([0], dtype=scatter_index_rev.dtype)

    dispatched_input.stop_gradient = False
    combine_weights_unnorm.stop_gradient = False
    scatter_index.stop_gradient = True
    expert_num_global.stop_gradient = True
    expert_num_global_notrunc = expert_num_global
    capacity_tensor = paddle.to_tensor(capacity, dtype=expert_num_global.dtype)
    expert_num_global = paddle.minimum(expert_num_global, capacity_tensor)

    if global_dense_expert_mask is not None:
        expert_num_global = expert_num_global[:-1]
        expert_num_local = expert_num_local[:-1]
        expert_num_global_notrunc = expert_num_global_notrunc[:-1]

    scatter_index = scatter_index.transpose([1, 0])  # [k,s] ->[s,k]

    last_local_expert = num_local_experts * gate.config.moe_rank
    expert_offset_global = expert_num_global.cumsum()

    loader = get_async_loader()
    expert_num_global_list, offload_task = async_offload(expert_num_global, loader)
    # if self.use_padding:
    #     offset = last_local_expert * capacity
    # else:
    offset = (
        expert_offset_global[last_local_expert - 1]
        if gate.config.moe_rank > 0
        else 0
    )
    local_combine_weights_unnorm = ReshardCombineWeight.apply(
        combine_weights_unnorm.contiguous(), group=gate.config.moe_group
    )
    local_scatter_index = ReduceScatterGroupOp.apply(
        paddle.where(
            combine_weights_unnorm > 0.0,
            scatter_index + offset,
            scatter_index,
        ),
        group=gate.config.moe_group,
    )
    if gate.norm_gate_logits:
        local_combine_weights = local_combine_weights_unnorm / paddle.clip(
            local_combine_weights_unnorm.sum(-1, keepdim=True), min=1e-12
        )
    else:
        local_combine_weights = local_combine_weights_unnorm
    local_combine_weights = local_combine_weights.cast(dispatched_input.dtype)
    # if self.use_padding:
    #     dispatched_input = dispatched_input.reshape(
    #         [self.num_local_experts, -1, d_model]
    #     )
    #     dispatched_input = dispatched_input.unbind(0)
    # else:
    s = num_local_experts * gate.config.moe_rank
    e = num_local_experts * (gate.config.moe_rank + 1)
    expert_num_local = expert_num_local.tolist()[s:e]
    expert_num_local_valid = [i for i in expert_num_local if i > 0]
    valid_pos = [j for j, i in enumerate(expert_num_local) if i > 0]
    if expert_num_local_valid:
        dispatched_input_list = dispatched_input.split(expert_num_local_valid)
        dispatched_input = [None] * len(expert_num_local)
        for p, t in zip(valid_pos, dispatched_input_list):
            dispatched_input[p] = t
    else:
        dispatched_input = [dispatched_input] + (
            [None] * (len(expert_num_local) - 1)
        )

    scatter_index.stop_gradient = True
    scatter_index_rev.stop_gradient = True
    if offload_task is not None:
        hack_offload_wait(offload_task)
    expert_num_global_list = expert_num_global_list.tolist()

    return (
        dispatched_input,
        global_hidden_states,
        local_combine_weights,
        expert_num_global_notrunc,  # for auxloss calculation.
        expert_num_global,
        expert_num_global_list,
        local_scatter_index,
        scatter_index_rev,
        router_loss,
        (gate_logits_lm, gate_prob_lm),
        (gate_logits_mm, gate_prob_mm),
        expert_num_local,
    )

def calc_router_loss_and_logging(
        router_loss,
        gate_logits,
        gate_prob,
        gate_logits_mm,
        gate_prob_mm,
        combine_weights,
        dispatch_mask,
        token_type_ids,
        dispatch_token_type_ids,
        gate,
        layer_idx
    ):
    """Calculate and aggregate router auxiliary loss for Mixture-of-Experts training.

    Core Functionality:
    - Computes expert load balancing loss to prevent expert under-utilization
    - Integrates multiple loss components from different routing stages
    - Maintains gradient flow for routing mechanism optimization

    Args:
        router_loss (Tensor): Accumulated router loss tensor
        gate_logits (Tensor): Raw gating network outputs [batch_size, num_experts]
        gate_prob (Tensor): Activated gating probabilities [batch_size, num_experts]
        combine_weights (Tensor): Expert combination weights [batch_size, top_k]
        dispatch_mask (Tensor): Token dispatch mask indicating expert assignments

    Returns:
        Tensor: Updated router loss with new auxiliary components
    """
    dispatch_mask_3d = dispatch_mask.reshape([gate.config.moe_world_size, -1])
    if token_type_ids is not None and gate.config.moe_use_hard_gate:
        # MLLM
        if not gate.weight.stop_gradient:
            dispatch_tokens_mask = (
                dispatch_token_type_ids == 0
                if dispatch_token_type_ids is not None
                else None
            )
            lm_tokens_mask = (token_type_ids == 0).astype(gate_prob.dtype)
            # hard code
            lm_experts = (
                gate.num_experts[0]
                if isinstance(gate.num_experts, (tuple, list))
                else gate.num_experts
            )
            dispatch_mask_lm = dispatch_mask_3d[
                :, : lm_experts // gate.config.moe_world_size
            ].reshape([-1])
            router_loss += _calc_router_loss(
                dispatch_mask_lm,
                gate_logits * lm_tokens_mask.unsqueeze(-1),
                gate_prob * lm_tokens_mask.unsqueeze(-1),
                gate.num_experts_list[0],
                False, # self.group_experts,
                layer_idx,
                0,  # ortholoss
                lm_tokens_mask,
                dispatch_tokens_mask,
                prefix="lm",
                gate,
            )
        else:
            zero = paddle.to_tensor(0, dtype=paddle.float32)
            router_loss += zero * gate_logits[0, 0] * gate_prob[0, 0]
        if gate_prob_mm is not None:
            mm_tokens_mask = (token_type_ids == 1).astype(gate_prob_mm.dtype)
            dispatch_tokens_mask = (
                dispatch_token_type_ids == 1
                if dispatch_token_type_ids is not None
                else None
            )
            dispatch_mask_mm = dispatch_mask_3d[
                :, gate.num_experts[0] // gate.config.moe_world_size :
            ].reshape([-1])

            router_loss += _calc_router_loss(
                dispatch_mask_mm,
                gate_logits_mm * mm_tokens_mask.unsqueeze(-1),
                gate_prob_mm * mm_tokens_mask.unsqueeze(-1),
                gate.num_experts_list[1],
                False,
                layer_idx,
                1,
                mm_tokens_mask,
                dispatch_tokens_mask,
                prefix="mm",
                gate,
            )

    else:
        # LLM
        router_loss += _calc_router_loss(
            dispatch_mask,
            gate_logits,
            gate_prob,
            gate.num_experts_tensor,
            False, #self.group_experts,
            layer_idx,
            0,
            paddle.ones([gate_prob.shape[0]], "bool"),
            paddle.ones(
                [gate.config.moe_world_size * gate_prob.shape[0]], "bool"
            ),
            prefix="lm",
        )

    return router_loss