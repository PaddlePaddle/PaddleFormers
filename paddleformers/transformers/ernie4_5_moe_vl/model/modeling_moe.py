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

"""Paddle Ernie model"""

import contextlib
import math
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple, Union

import paddle
import paddle.distributed as dist
import paddle.distributed.communication.group
import paddle.nn.functional as F
from paddle import nn
from paddle.autograd import PyLayer
from paddle.distributed import fleet
from paddle.distributed.communication.group import _get_global_group
from paddle.distributed.fleet.layers.mpu.random import get_rng_state_tracker
from paddle.distributed.fleet.meta_parallel import VocabParallelEmbedding
from paddle.distributed.fleet.utils import recompute
from paddle.incubate.tensor.manipulation import async_offload

from paddleformers.transformers.model_outputs import (
    BaseModelOutputWithPastAndCrossAttentions as _BaseModelOutput,
)
from paddleformers.transformers.model_outputs import (
    CausalLMOutputWithCrossAttentions as _CausalLMOutput,
)
from paddleformers.transformers.model_utils import PretrainedModel, register_base_model
from paddleformers.utils.log import logger

from .configuration import Ernie4_5_MoeConfig
from .distributed import ScatterOp, mark_as_sequence_parallel_parameter
from .distributed.common_dist_utils import get_async_loader, hack_offload_wait
from .fusion_ops import fused_swiglu
from .loss.dpo import ErnieDPOCriterion
from .modeling import Ernie4_5_Attention, Ernie4_5_LMHead, Ernie4_5_MLP
from .modeling import ErniePretrainingCriterion as ErniePretrainingCriterionBase
from .modeling import FusedDropoutImpl, LayerNorm, RMSNorm
from .moe.moe_all_gather_layer import MOEAllGatherLayerV2
from .moe.moe_layer import MOELayer, MoEStatics
from .moe.topk_gate import TopKGate
from .refined_recompute.utils import create_skip_config_for_refined_recompute
from .sequence_parallel_utils import GatherOp

# Note: ProcessGroupNCCL do not support deepcopy protocol, we made modifications here.
paddle.distributed.communication.group.Group.__deepcopy__ = lambda self, _: self
paddle.distributed.communication.group.Group.to_json = lambda self: repr(self)


def mtp_hidden_states_set_zero(hidden_states, inbatch_pack_offset):
    # inbatch_pack_offset: [batch_size, seqlen]
    # hidden_states: [batch_size, seqlen, d_model]
    if len(hidden_states.shape) == 3:
        batch_size, seqlen, d_model = hidden_states.shape
        valid_indices = paddle.where(inbatch_pack_offset[0] > 0)[0]
        mask = paddle.ones_like(hidden_states[0])
        assert batch_size == 1, "only support batch_size=1 in inbatch sft training"

        if len(valid_indices) > 0:
            zeros = paddle.zeros([len(valid_indices), d_model], dtype=hidden_states.dtype)
            mask = paddle.scatter(mask, valid_indices.reshape([-1, 1]), zeros, overwrite=True)
        mask.stop_gradient = True
        hidden_states = hidden_states * mask.unsqueeze(0)

    elif len(hidden_states.shape) == 2:
        seqlen, d_model = hidden_states.shape
        valid_indices = paddle.where(inbatch_pack_offset > 0)[0]
        mask = paddle.ones_like(hidden_states)

        if len(valid_indices) > 0:
            zeros = paddle.zeros([len(valid_indices), d_model], dtype=hidden_states.dtype)
            mask = paddle.scatter(mask, valid_indices, zeros, overwrite=True)

        mask.stop_gradient = True
        hidden_states = hidden_states * mask
    return hidden_states


@dataclass
class BaseModelOutputWithPastAndCrossAttentions(_BaseModelOutput):
    """
    Base class for model outputs with past key values and cross attention layers,
    with additional support for router components in mixture-of-experts models.

    This extends the base model output to include:
    1. Router-related outputs for expert selection
    2. Maintains all existing functionality from the parent class

    Args:
        router_loss (Optional[paddle.Tensor]):
            The auxiliary loss computed by the router in mixture-of-experts models.
            None if not using router mechanism.

        gate_logits (Optional[Tuple[paddle.Tensor]]):
            The raw logits output by the gating network before softmax.
            None if not using router mechanism.
    """

    router_loss: Optional[paddle.Tensor] = None
    gate_logits: Optional[Tuple[paddle.Tensor]] = None
    mtp_outputs: Optional[paddle.Tensor] = None


@dataclass
class CausalLMOutputWithCrossAttentions(_CausalLMOutput):
    """
    Output class for causal language models with cross-attention mechanisms,
    extending the base causal LM output with additional routing components.

    This class inherits all attributes from _CausalLMOutput and adds:
    - Support for router-related outputs in mixture-of-experts architectures
    - Maintains cross-attention capabilities from parent class

    Args:
        router_loss (Optional[paddle.Tensor]):
            The routing loss computed by the gating network in mixture-of-experts models.
            This is typically the load balancing loss that encourages equal expert utilization.
            None when not using mixture-of-experts routing.
    """

    router_loss: Optional[paddle.Tensor] = None


ERNIE_PRETRAINED_MODEL_ARCHIVE_LIST = []

__all__ = [
    "Ernie4_5_ForCausalLM",
    "Ernie4_5_MoeForCausalLM",
    "ErniePretrainingCriterion",
    "CausalLMOutputWithCrossAttentions",
]

gate_class = dict(
    top2_fused=TopKGate,
    topk=TopKGate,
)


def get_gate(
    config: Ernie4_5_MoeConfig,
    expert: Tuple[Tuple[int, nn.Layer]],
    layer_idx: int,
) -> Tuple[nn.Layer, nn.LayerList]:
    """Initialize and distribute MoE (Mixture of Experts) components.

    Creates gate layer and distributed expert network for MoE architecture.

    Args:
        config (Ernie4_5_MoeConfig): Configuration for MoE architecture
        expert (nn.Layer): Prototype expert network to be replicated
        layer_idx (int): Index of current layer in transformer stack

    Returns:
        Tuple[nn.Layer, nn.LayerList]:
            - gate: Initialized gate layer for routing
            - experts: LayerList containing distributed expert networks
                      (each device gets moe_num_experts/moe_world_size experts)
    """
    moe_num_experts = sum(config.moe_num_experts) if config.multimodel_experts else config.moe_num_experts
    assert (
        moe_num_experts >= config.moe_world_size
    ), f"expert moe_num_experts={moe_num_experts} >= moe_world_size={config.moe_world_size}"
    assert (
        moe_num_experts % config.moe_world_size == 0
    ), f"expert moe_num_experts={moe_num_experts} % moe_world_size={config.moe_world_size} == 0"
    moe_num_experts_per_device = moe_num_experts // config.moe_world_size
    logger.debug(f"using moe-world-size: {config.moe_world_size} expert-per-device:{moe_num_experts_per_device},")
    logger.info(f"MOE-GATE:-{config.moe_gate}")

    experts = nn.LayerList([])
    moe_rank = paddle.distributed.get_rank(config.moe_group)
    if moe_rank < 0:
        moe_rank = 0

    if not config.multimodel_experts:
        # LLM
        for expert_id, (experts_num, fc) in enumerate(expert):
            for i in range(experts_num):
                if i // moe_num_experts_per_device == moe_rank:
                    experts.append(deepcopy(fc))
                else:
                    experts.append(None)
    else:
        # VL model
        for expert_id, (experts_num, fc) in enumerate(expert):
            assert experts_num % config.moe_world_size == 0
            num_experts_per_device = experts_num // config.moe_world_size
            experts_to_append = []
            if not hasattr(fc, "__len__"):  # run this
                experts_to_append.append(fc)
                if expert_id == 1:
                    with paddle.utils.unique_name.guard("_mm_deepcopy"):
                        for _ in range(num_experts_per_device - 1):
                            experts_to_append.append(deepcopy(fc))
                else:
                    for _ in range(num_experts_per_device - 1):
                        experts_to_append.append(deepcopy(fc))
            else:
                experts_to_append = fc

            for ex in experts_to_append:
                for p in ex.parameters():
                    p.expert_type = (
                        f"expert_type_{expert_id}"  # Different `expert_type` can have different intermediate-size
                    )

            # To compat with safetensors format.
            index = 0
            for i in range(experts_num):
                if i // num_experts_per_device == moe_rank:
                    experts.append(experts_to_append[index])
                    index += 1
                else:
                    experts.append(None)

    assert (
        len(experts) == moe_num_experts  # including None
    ), f"experts.len={len(experts)} != moe_num_experts={moe_num_experts}"

    gate = gate_class[config.moe_gate.lower()](config, layer_idx=layer_idx, group=config.moe_group)

    if not config.multimodel_experts:
        return gate, experts

    if config.multimodel_experts and config.moe_use_hard_gate and moe_num_experts > 2:
        lm_experts = experts[: config.moe_num_experts[0]]
        lm_cfg = deepcopy(config)
        lm_cfg.moe_num_experts = config.moe_num_experts[0]
        lm_gate = gate_class[config.moe_gate.lower()](
            lm_cfg, layer_idx=layer_idx, group=config.moe_group, gate_weight=gate.weight
        )
    else:
        if config.multimodel_experts and config.moe_use_hard_gate:
            lm_gate, lm_experts = gate, experts
        else:
            lm_gate, lm_experts = None, None

    logger.info(f"LM-experts-{lm_experts} -- experts-{experts}")

    return gate, experts, lm_gate, lm_experts


def _parse_moe_group(
    moe_group: str,
) -> Union[str, paddle.distributed.communication.group.Group]:
    """Parse and initialize the MoE (Mixture of Experts) communication group.

    Converts string representation of MoE group into actual process group
    for distributed expert parallelism.

    Args:
        moe_group (str): Specifies the type of parallel group to use for MoE.
            Supported values:
            - "data" or "dp": Data parallel group
            - "mp", "model" or "tp": Model parallel group
            - "dummy": Dummy group for single process
            - "none", "world" or "all": Global communication group

    Returns:
        Union[str, paddle.distributed.communication.group.Group]:
            The corresponding process group object, or dummy group string.
            Returns dummy group for single-process case.
    """
    moe_group = moe_group.lower()
    assert moe_group in {
        "data",
        "dp",
        "mp",
        "tp",
        "model",
        "dummy",  # 4.5t_mm infer run this
        "none",
        "world",
        "all",
    }, f"moe-group not supported, got: {moe_group}"
    logger.info(f"using moe-group: {moe_group}")
    if moe_group in {"data", "dp"}:
        moe_group = fleet.get_hybrid_communicate_group().get_data_parallel_group()
    elif moe_group in {"mp", "model", "tp"}:
        try:
            moe_group = fleet.get_hybrid_communicate_group().get_model_parallel_group()
            # (LiuTing): multi-gpu but tp=1
            # need use dummy group for `moe_gate_dispatch_partial_nosoftmaxtopk` kernel.
            if moe_group.nranks <= 1:
                moe_group = paddle.distributed.communication.group.Group(0, None, [0])
        except Exception:
            # (LiuTing): just single-gpu
            moe_group = paddle.distributed.communication.group.Group(0, None, [0])

    elif moe_group in {"dummy"}:  # 4.5t_mm infer run this
        dummy_group = paddle.distributed.communication.group.Group(0, None, [0])
        moe_group = dummy_group
    else:
        moe_group = _get_global_group()

    return moe_group


class Ernie4_5_MoeMLP(Ernie4_5_MLP):
    """Mixture of Experts (MoE) variant of ERNIE's MLP layer."""

    def __init__(self, config, layer_idx=0):
        """Initialize the MoE MLP layer.

        Args:
            config (Ernie4_5_MoeConfig): Configuration for MoE architecture.
            layer_idx (int): Index of current layer in transformer stack
        """

        if getattr(config, "disable_ffn_model_parallel", False):
            config = deepcopy(config)
            config.tensor_parallel_degree = 1

        super().__init__(config, layer_idx=layer_idx)
        self.moe_dropout_prob = config.moe_dropout_prob
        self.fuse_swiglu = config.fuse_swiglu
        if self.fuse_swiglu:
            assert fused_swiglu is not None, "fused_swiglu operator is not found."

    def forward(self, x):
        """Forward pass through MoE MLP layer.

        Args:
            x (paddle.Tensor): Input tensor of shape [batch_size, seq_len, hidden_size]
                              or [seq_len, hidden_size]

        Returns:
            paddle.Tensor: Output tensor with same shape as input
        """
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        if self.fuse_swiglu:
            x = paddle.concat([gate, up], axis=-1)
            x = fused_swiglu(x)
        else:
            x = F.silu(gate) * up
        if self.moe_dropout_prob > 0:
            with get_rng_state_tracker().rng_state("local_seed"):
                x = F.dropout(x=x, p=self.moe_dropout_prob)
        ret = self.down_proj(x)
        return ret


class FakeMoERouterLoss(PyLayer):
    """A gradient trick layer for MoE router loss computation.

    This layer artificially injects router loss gradients during backpropagation
    while passing through the original tensor during forward pass.
    """

    @staticmethod
    def forward(ctx, x, router_loss, num_acc_steps, enable_delay_scale_loss):
        """Forward pass that preserves input tensor while storing router loss context.

        Args:
            x (paddle.Tensor): The input hidden states tensor
            router_loss (paddle.Tensor): Computed router loss value
            num_acc_steps (int): Gradient accumulation steps
            enable_delay_scale_loss (bool): Whether to scale loss by accumulation steps

        Returns:
            paddle.Tensor: The unchanged input tensor x
        """
        ctx.num_acc_steps = num_acc_steps
        ctx.loss_shape = router_loss.shape
        ctx.loss_dtype = router_loss.dtype
        ctx.enable_delay_scale_loss = enable_delay_scale_loss
        return x

    @staticmethod
    def backward(ctx, out_grad):
        """Backward pass that injects router loss gradients.

        Args:
            out_grad (paddle.Tensor): Gradient from downstream layers

        Returns:
            Tuple[paddle.Tensor, paddle.Tensor]:
                - The original downstream gradient
                - Artificial router loss gradient
        """
        if ctx.enable_delay_scale_loss:
            router_loss_grad_value = 1.0
        else:
            router_loss_grad_value = 1.0 / ctx.num_acc_steps

        return out_grad, paddle.full(ctx.loss_shape, router_loss_grad_value, dtype=ctx.loss_dtype)


class Ernie4_5_DecoderLayer(nn.Layer):
    """A single transformer decoder layer in ERNIE-MoE model.

    Contains self-attention and feed-forward components with optional MoE (Mixture of Experts)
    support, residual connections, and layer normalization.
    """

    def __init__(self, config, layer_idx):
        """Initialize the decoder layer.

        Args:
            config (Ernie4_5_MoeConfig): Model configuration.
            layer_idx (int): Index of this layer in the transformer stack
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.config = config
        self.use_moe = config.use_moe
        self.self_attn = Ernie4_5_Attention(config, layer_idx)

        moe_layer_start_index = (
            min(config.moe_layer_start_index)
            if isinstance(config.moe_layer_start_index, (tuple, list))
            else config.moe_layer_start_index
        )
        moe_layer_end_index = (
            max(config.moe_layer_end_index)
            if isinstance(config.moe_layer_end_index, (tuple, list))
            else config.moe_layer_end_index
        )

        if (
            self.use_moe
            and ((layer_idx + 1) % config.moe_layer_interval == 0)
            and layer_idx >= moe_layer_start_index  # 3
            and layer_idx <= moe_layer_end_index  # 53
        ):
            gate, experts, lm_gate, lm_experts, moe_statics = self._init_gate_and_experts(layer_idx)
            shared_experts = self._init_shared_experts() if hasattr(config, "moe_num_shared_experts") else None
            dense_experts = None
            moe_cls = MOELayer
            if config.moe_multimodal_dispatch_use_allgather:  # v2
                logger.info("Enable MOEAllGatherLayerV2!")
                moe_cls = partial(
                    MOEAllGatherLayerV2,
                    use_expert_out_alltoall="alltoall" in config.moe_multimodal_dispatch_use_allgather,  # false
                    use_padding="unpad" not in config.moe_multimodal_dispatch_use_allgather,  # true
                    enable_reverse_token_drop=config.moe_reverse_token_drop,  # false
                    dense_token_type=config.moe_dense_experts_token_type_id,  # 3
                )
            else:
                assert dense_experts is None, "only `MOEAllGatherLayerV2` can process dense experts"

            logger.info(f"moe-logging: {config.moe_multimodal_dispatch_use_allgather} moe_cls={moe_cls}")

            self.mlp = moe_cls(
                gate=gate,
                experts=experts,
                layer_idx=layer_idx,
                shared_experts=shared_experts,
                group=config.moe_group,
                recompute=config.use_recompute_moe,
                k=config.moe_k,
                all_to_all_dropout=config.moe_all_to_all_dropout,
                group_experts=config.moe_group_experts,  # false
                moe_statics=moe_statics,
                moe_num_experts=config.moe_num_experts,
            )

            if config.multimodel_experts and config.moe_use_hard_gate:  # VL model
                _mlp_text = MOEAllGatherLayerV2(
                    gate=lm_gate,
                    experts=lm_experts,
                    layer_idx=layer_idx,
                    use_padding=False,
                    shared_experts=shared_experts,
                    group=config.moe_group,
                    recompute=config.use_recompute_moe,
                    k=config.moe_k,
                    all_to_all_dropout=config.moe_all_to_all_dropout,
                    group_experts=config.moe_group_experts,
                    moe_statics=moe_statics,
                    moe_num_experts=config.moe_num_experts,
                )
                self.mlp_text = (
                    lambda: _mlp_text
                )  # This lambda prevents the text parameter from being scanned into the state-dict

            if (
                config.sequence_parallel
            ):  # Under `mp-moe`, gate is effective in attn and is in the synchronization zone.
                for p in gate.parameters():
                    mark_as_sequence_parallel_parameter(p)
        else:
            self.mlp = Ernie4_5_MLP(config)

        Norm = RMSNorm if config.use_rmsnorm else LayerNorm

        self.input_layernorm = Norm(config)
        self.post_attention_layernorm = Norm(config)

        self.residual_add1 = FusedDropoutImpl(config.hidden_dropout_prob, mode="upscale_in_train")
        self.residual_add2 = FusedDropoutImpl(config.hidden_dropout_prob, mode="upscale_in_train")

        if config.sequence_parallel:
            mark_as_sequence_parallel_parameter(self.post_attention_layernorm.weight)
            # There is no Column/RowLinear in bias and expert in mp-moe. No hook is needed.
            if not hasattr(config, "disable_ffn_model_parallel"):
                mark_as_sequence_parallel_parameter(self.input_layernorm.weight)
                if config.use_bias:  # false
                    mark_as_sequence_parallel_parameter(self.self_attn.o_proj.bias)
                    if isinstance(self.mlp, MOELayer):
                        for m in self.mlp.experts:
                            mark_as_sequence_parallel_parameter(m.down_proj.bias)
                    else:
                        mark_as_sequence_parallel_parameter(self.mlp.down_proj.bias)

            if not config.use_rmsnorm and config.use_bias:
                mark_as_sequence_parallel_parameter(self.post_attention_layernorm.bias)
                mark_as_sequence_parallel_parameter(self.input_layernorm.bias)

    def _init_shared_experts(self):
        """init shared experts

        Returns:
            _type_: _description_
        """
        cfg = deepcopy(self.config)
        if cfg.moe_num_shared_experts > 0:
            if cfg.moe_intermediate_size:
                inter_size = (
                    next(iter(cfg.moe_intermediate_size))
                    if isinstance(cfg.moe_intermediate_size, (tuple, list))
                    else cfg.moe_intermediate_size
                )
                cfg.intermediate_size = inter_size * cfg.moe_num_shared_experts
            else:
                cfg.intermediate_size = cfg.intermediate_size * cfg.moe_num_shared_experts
            cfg.disable_ffn_model_parallel = False  # split shared epxert
            shared_experts = Ernie4_5_MoeMLP(cfg, True)
        else:
            shared_experts = None
        return shared_experts

    def _init_gate_and_experts(self, layer_idx):
        """Initialize MoE gate and expert networks.

        Args:
            layer_idx (int): Current layer index

        Returns:
            Tuple: Contains:
                - gate: MoE routing gate
                - experts: List of expert networks
                - moe_statics: Optional statistics tracker
        """
        cfg = deepcopy(self.config)
        fc_cls = Ernie4_5_MoeMLP
        if cfg.moe_intermediate_size:
            if isinstance(cfg.moe_intermediate_size, (tuple, list)):
                cfg.moe_num_experts = cfg.moe_num_experts
                assert isinstance(cfg.moe_num_experts, (tuple, list)) and len(cfg.moe_num_experts) == len(
                    cfg.moe_intermediate_size
                )
                fc = []
                for _i, (num_experts, intermediate_size) in enumerate(
                    zip(cfg.moe_num_experts, cfg.moe_intermediate_size)
                ):
                    ex_cfg = deepcopy(cfg)
                    ex_cfg.intermediate_size = intermediate_size
                    cur_modality_start_layer_idx = (
                        cfg.moe_layer_start_index[_i]
                        if isinstance(cfg.moe_layer_start_index, (tuple, list))
                        else cfg.moe_layer_start_index
                    )
                    cur_modality_end_layer_idx = (
                        cfg.moe_layer_end_index[_i]
                        if isinstance(cfg.moe_layer_end_index, (tuple, list))
                        else cfg.moe_layer_end_index
                    )
                    if layer_idx >= cur_modality_start_layer_idx and layer_idx <= cur_modality_end_layer_idx:
                        if _i == 1:
                            with paddle.utils.unique_name.guard(f"mm_expert_{layer_idx}_"):
                                fc.append((num_experts, fc_cls(ex_cfg)))
                        else:
                            fc.append((num_experts, fc_cls(ex_cfg)))
                    else:
                        logger.info(f"moe multimodal experts use Identity layer_idx: {layer_idx}")
                        fc.append((num_experts, nn.Identity()))
            else:
                cfg.intermediate_size = cfg.moe_intermediate_size
                fc = [(cfg.moe_num_experts, fc_cls(cfg, layer_idx))]
        else:
            fc = [(cfg.moe_num_experts, fc_cls(cfg, layer_idx))]

        if cfg.multimodel_experts:
            gate, experts, lm_gate, lm_experts = get_gate(self.config, fc, layer_idx)
        else:
            gate, experts = get_gate(self.config, fc, layer_idx)
            lm_gate, lm_experts = None, None

        # for AuxLoss Free Router:
        if cfg.moe_use_aux_free:
            moe_statics = MoEStatics(cfg, layer_idx)
        else:
            moe_statics = None
        return gate, experts, lm_gate, lm_experts, moe_statics

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        attn_mask_start_row_indices: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        token_type_ids: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = False,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        use_cache: Optional[bool] = False,
        output_gate_logits=True,  # PP model should not output gate logits,
    ) -> Tuple[paddle.Tensor, Optional[Tuple[paddle.Tensor, paddle.Tensor]]]:
        """Forward pass through the decoder layer.

        Args:
            hidden_states (paddle.Tensor): Input tensor [batch_size, seq_len, hidden_size]
            attention_mask (Optional[paddle.Tensor]): Attention mask tensor
            attn_mask_start_row_indices (Optional[paddle.Tensor]): Indices for variable length attention
            position_ids (Optional[paddle.Tensor]): Position indices for rotary embeddings
            output_attentions (Optional[bool]): Whether to return attention weights
            past_key_value (Optional[Tuple[paddle.Tensor]]): Cached key/value states
            use_cache (Optional[bool]): Whether to cache key/value states
            output_gate_logits (bool): Whether to return MoE gate logits

        Returns:
            Union: Various output combinations depending on arguments:
                - Base case: Hidden states tensor
                - With attention: Tuple of (hidden_states, attention_weights)
                - With cache: Tuple of (hidden_states, cached_key_value)
                - With MoE: May include gate logits in output tuple
        """
        residual = hidden_states

        if token_type_ids is not None:
            is_multimodel_token = token_type_ids.any()
            has_dense_experts_token = (token_type_ids == self.config.moe_dense_experts_token_type_id).any()
            async_loader = get_async_loader()
            is_multimodel_token_cpu, is_multimodel_token_task = async_offload(is_multimodel_token, async_loader)
            _, has_dense_experts_token_task = async_offload(has_dense_experts_token, async_loader)
        else:
            is_multimodel_token_task = None
            is_multimodel_token_cpu = None
            has_dense_experts_token_task = None
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        has_gradient = not hidden_states.stop_gradient
        if self.config.recompute and self.config.recompute_granularity == "full_attn" and has_gradient:
            hidden_states, self_attn_weights, present_key_value = recompute(
                self.self_attn,
                hidden_states,
                past_key_value,
                attention_mask,
                attn_mask_start_row_indices,
                position_ids,
                output_attentions,
                use_cache,
                use_reentrant=self.config.recompute_use_reentrant,
            )
        else:
            (hidden_states, self_attn_weights, present_key_value, *router_loss_attn) = self.self_attn(
                hidden_states=hidden_states,
                past_key_value=past_key_value,
                attention_mask=attention_mask,
                attn_mask_start_row_indices=attn_mask_start_row_indices,
                position_ids=position_ids,
                output_attentions=output_attentions,
                use_cache=use_cache,
                token_type_ids=token_type_ids,
            )

        with self.model_parallel_dropout():
            hidden_states = self.residual_add1(hidden_states, residual)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if isinstance(self.mlp, MOELayer):
            if is_multimodel_token_task is not None:
                hack_offload_wait(is_multimodel_token_task)
            if has_dense_experts_token_task is not None:
                hack_offload_wait(has_dense_experts_token_task)
            if (
                self.config.multimodel_experts
                and self.config.moe_use_hard_gate
                and token_type_ids is not None
                and not is_multimodel_token_cpu
            ):
                hidden_states, _, router_loss, gate_logits = self.mlp_text()(hidden_states, None)  # run this
            else:
                hidden_states, _, router_loss, gate_logits = self.mlp(hidden_states, token_type_ids)
        else:
            hidden_states = self.mlp(hidden_states)
            gate_logits, router_loss = None, None

        with self.model_parallel_dropout():
            hidden_states = self.residual_add2(hidden_states, residual)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if self.use_moe:
            # Non-empty only if `use_moe`
            if router_loss_attn:
                router_loss_attn = router_loss_attn[0]
                router_loss = router_loss + router_loss_attn

            # When use_moe is enabled, an additional return value will be added regardless of whether this layer has a moe layer or not
            if isinstance(
                self.mlp,
                (
                    MOELayer,
                    MOEAllGatherLayerV2,
                ),
            ):
                hidden_states = FakeMoERouterLoss.apply(
                    hidden_states,
                    router_loss,
                    self.config.num_acc_steps,
                    self.config.enable_delay_scale_loss,
                )
                if self.training:
                    hidden_states.stop_gradient = False

                outputs = (hidden_states,) + outputs[1:]

            if output_gate_logits:
                outputs += (gate_logits,)

        # remove empty tuple for pipeline parallel
        if type(outputs) is tuple and len(outputs) == 1:
            outputs = outputs[0]
        return outputs

    def model_parallel_dropout(self):
        """Get context manager for model-parallel dropout with proper seed control.

        Returns:
            Context manager for dropout operation
        """
        if self.config.tensor_parallel_degree > 1 and self.config.hidden_dropout_prob > 0.0:
            current_seed = "local_seed" if self.config.sequence_parallel else "global_seed"
            return get_rng_state_tracker().rng_state(current_seed)
        return contextlib.nullcontext()


class Ernie4_5_PretrainedModel(PretrainedModel):
    """Base class for ERNIE pretrained models."""

    config_class = Ernie4_5_MoeConfig
    base_model_prefix = "model"
    _keep_in_fp32_modules = ["mlp.gate", "e_score_correction_bias"]
    transpose_weight_keys = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        # "gate",
    ]

    @classmethod
    def _get_tensor_parallel_mappings(cls, config, is_split=True):
        """Generate tensor parallel mappings for model conversion."""

        from ...conversion_utils import split_or_merge_func

        fn = split_or_merge_func(
            is_split=is_split,
            tensor_parallel_degree=config.tensor_parallel_degree,
            tensor_parallel_rank=config.tensor_parallel_rank,
            num_attention_heads=config.num_attention_heads,
        )

        LAYER_COLWISE = [
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "mlp.up_proj.weight",
            "mlp.gate_proj.weight",
        ]
        LAYER_ROWWISE = ["self_attn.o_proj.weight", "mlp.down_proj.weight"]

        BIAS_KEYS = [
            "self_attn.q_proj.bias",
            "self_attn.k_proj.bias",
            "self_attn.v_proj.bias",
            "mlp.gate_proj.bias",
            "mlp.up_proj.bias",
            "self_attn.o_proj.bias",
            "mlp.down_proj.bias",
            "lm_head.bias",
        ]
        SHARED_EXPERTS_COLWISE_KEYS = ["up_proj.weight", "gate_proj.weight"]
        SHARED_EXPERTS_ROWWISE_KEYS = ["down_proj.weight"]

        def make_base_actions():
            actions = {
                "lm_head.weight": partial(fn, is_column=False),
                "embed_tokens.weight": partial(fn, is_column=False),
            }
            for layer_idx in range(config.num_hidden_layers):
                actions.update(
                    {
                        f"{cls.base_model_prefix}.layers.{layer_idx}.{k}": partial(fn, is_column=True)
                        for k in LAYER_COLWISE
                    }
                )
                actions.update(
                    {
                        f"{cls.base_model_prefix}.layers.{layer_idx}.{k}": partial(fn, is_column=False)
                        for k in LAYER_ROWWISE
                    }
                )
                # bias
                if config.use_bias:
                    actions.update(
                        {
                            f"{cls.base_model_prefix}.layers.{layer_idx}.{b}": partial(fn, is_column=True)
                            for b in BIAS_KEYS
                        }
                    )
            return actions

        def expand_actions(base_actions, num_layers):
            extend_action = {}
            moe_group = config.moe_group if isinstance(config.moe_group, str) else config.moe_group_origin
            moe_in_mp = moe_group in {"mp", "model", "tp"}

            extend_key_prefix = f"{cls.base_model_prefix}.layers.0"

            moe_layer_start_index = (
                min(config.moe_layer_start_index)
                if isinstance(config.moe_layer_start_index, (tuple, list))
                else config.moe_layer_start_index
            )
            moe_layer_end_index = (
                max(config.moe_layer_end_index)
                if isinstance(config.moe_layer_end_index, (tuple, list))
                else config.moe_layer_end_index
            )

            for i in range(num_layers):
                # skip non-moe layers
                if ((i + 1) % config.moe_layer_interval != 0) or i < moe_layer_start_index or i > moe_layer_end_index:
                    continue
                experts_newkey = extend_key_prefix.replace("layers.0", f"layers.{i}.mlp.experts")
                expert_type = ["text_moe.experts", "vision_moe.experts"]

                if isinstance(config.moe_num_experts, int):
                    moe_num_experts = [config.moe_num_experts, config.moe_num_experts]
                else:
                    moe_num_experts = config.moe_num_experts
                assert len(moe_num_experts) == 2, "moe_num_experts should be a list of length 2"
                for j in range(2):
                    if moe_num_experts[j] > 0:
                        moe_expert_newkey = experts_newkey.replace("experts", expert_type[j])
                        for eid in range(moe_num_experts[j]):
                            for key in LAYER_COLWISE:
                                exp_key = f"{moe_expert_newkey}.{eid}.{key}"
                                action = partial(fn, is_column=True)
                                if not moe_in_mp:
                                    extend_action[exp_key] = action

                            for key in LAYER_ROWWISE:
                                exp_key = f"{moe_expert_newkey}.{eid}.{key}"
                                action = partial(fn, is_column=False)
                                if not moe_in_mp:
                                    extend_action[exp_key] = action

                if config.moe_num_shared_experts > 0:
                    shared_expert_newkey = extend_key_prefix.replace("layers.0", f"layers.{i}.mlp.shared_experts")
                    for key in SHARED_EXPERTS_COLWISE_KEYS:
                        exp_key = f"{shared_expert_newkey}.{key}"
                        action = partial(fn, is_column=True)
                        extend_action[exp_key] = action

                    for key in SHARED_EXPERTS_ROWWISE_KEYS:
                        exp_key = f"{shared_expert_newkey}.{key}"
                        action = partial(fn, is_column=False)
                        extend_action[exp_key] = action
            extend_action.update(base_actions)
            return extend_action

        base_actions = make_base_actions()
        mappings = expand_actions(base_actions, config.num_hidden_layers)
        return mappings


@register_base_model
class Ernie4_5_Model(Ernie4_5_PretrainedModel):
    """The core ERNIE transformer model with MoE (Mixture of Experts) support."""

    def __init__(self, config: Ernie4_5_MoeConfig):
        """Initialize the ERNIE model architecture.

        Args:
            config (Ernie4_5_MoeConfig): Model configuration.
        """
        if config.moe_group in {"mp", "model", "tp"} and config.tensor_parallel_degree > 1:
            logger.info(f"disable FFN tensor model parallel, moe-group={config.moe_group}")
            config.disable_ffn_model_parallel = True

        config.moe_group_origin = config.moe_group
        config.moe_group = _parse_moe_group(config.moe_group)
        config.moe_world_size = dist.get_world_size(config.moe_group)
        if config.moe_world_size < 0:
            config.moe_world_size = 1
        config.moe_rank = dist.get_rank(config.moe_group)
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.config = config

        if config.tensor_parallel_degree > 1:
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                self.hidden_size,
            )
        else:
            self.embed_tokens = nn.Embedding(
                self.vocab_size,
                self.hidden_size,
            )

        self.layers = nn.LayerList(
            [
                Ernie4_5_DecoderLayer(create_skip_config_for_refined_recompute(i, config), i)
                for i in range(config.num_hidden_layers)
            ]
        )
        Norm = RMSNorm if config.use_rmsnorm else LayerNorm
        self.norm = Norm(config)

        # set externally in VL model
        self.resampler_model = None

        self.gradient_checkpointing = False

        if self.config.num_nextn_predict_layers > 0:
            self.mtp_block = paddle.nn.LayerList(
                [Ernie4_5_DecoderLayer(config, layer_idx) for layer_idx in range(self.config.num_nextn_predict_layers)]
            )
            Norm = RMSNorm if config.use_rmsnorm else LayerNorm
            self.mtp_hidden_norm = paddle.nn.LayerList(
                [Norm(config) for _ in range(self.config.num_nextn_predict_layers)]
            )
            self.mtp_emb_norm = paddle.nn.LayerList(
                [Norm(config) for _ in range(self.config.num_nextn_predict_layers)]
            )

            LinearFN = paddle.incubate.nn.FusedLinear if config.fuse_linear else paddle.nn.Linear
            self.mtp_linear_proj = paddle.nn.LayerList(
                [
                    LinearFN(
                        self.config.hidden_size * 2,
                        self.config.hidden_size,
                        bias_attr=config.use_bias,
                    )
                    for _ in range(self.config.num_nextn_predict_layers)
                ]
            )
            if config.sequence_parallel:
                for mtp_linear in self.mtp_linear_proj:
                    mark_as_sequence_parallel_parameter(mtp_linear.weight)
                    if config.use_bias:
                        mark_as_sequence_parallel_parameter(mtp_linear.bias)

    def get_input_embeddings(self):
        """Get the input embedding layer.

        Returns:
            nn.Embedding: The embedding layer for input tokens
        """
        return self.embed_tokens

    def set_input_embeddings(self, value):
        """Set new input embeddings.

        Args:
            value (nn.Embedding): New embedding layer to use
        """
        self.embed_tokens = value

    @paddle.jit.not_to_static
    def recompute_training(
        self,
        layer_module,
        hidden_states,
        attention_mask,
        attn_mask_start_row_indices,
        position_ids,
        token_type_ids,
        output_attentions,
        past_key_value,
        use_cache,
    ):
        """Perform gradient checkpointing for memory-efficient training.

        Args:
            layer_module (nn.Layer): Transformer layer to recompute
            hidden_states (paddle.Tensor): Input hidden states
            attention_mask (paddle.Tensor): Attention mask
            attn_mask_start_row_indices (paddle.Tensor): Variable length indices
            position_ids (paddle.Tensor): Position indices
            output_attentions (bool): Whether to output attention weights
            past_key_value (Optional[Tuple[paddle.Tensor]]): Cached key/value states
            use_cache (bool): Whether to cache key/value states

        Returns:
            paddle.Tensor: Output hidden states after recomputation
        """

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs, output_gate_logits=False)

            return custom_forward

        hidden_states = recompute(
            create_custom_forward(layer_module),
            hidden_states,
            attention_mask,
            attn_mask_start_row_indices,
            position_ids,
            token_type_ids,
            output_attentions,
            past_key_value,
            use_cache,
        )
        return hidden_states

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        token_type_ids=None,
        attention_mask=None,
        attn_mask_start_row_indices=None,
        inputs_embeds=None,
        use_cache=None,
        past_key_values=None,
        output_attentions=False,
        output_hidden_states=None,
        return_dict=False,
        **kwargs,
    ):
        """Forward pass through the ERNIE model.

        Args:
            input_ids (Optional[paddle.Tensor]): Input token IDs
            position_ids (Optional[paddle.Tensor]): Position indices
            attention_mask (Optional[paddle.Tensor]): Attention mask
            attn_mask_start_row_indices (Optional[paddle.Tensor]): Variable length attention indices
            inputs_embeds (Optional[paddle.Tensor]): Precomputed embeddings
            use_cache (Optional[bool]): Whether to cache key/value states
            past_key_values (Optional[Tuple[Tuple[paddle.Tensor]]]): Cached key/value states
            output_attentions (Optional[bool]): Whether to output attention weights
            output_hidden_states (Optional[bool]): Whether to output all hidden states
            return_dict (Optional[bool]): Whether to return dict or tuple

        Returns:
            Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
                Various outputs depending on configuration, including:
                - last_hidden_state: Final layer hidden states
                - past_key_values: Cached key/value states if use_cache=True
                - hidden_states: All hidden states if output_hidden_states=True
                - attentions: Attention weights if output_attentions=True
                - router_loss: MoE router loss if use_moe=True
                - gate_logits: MoE gate logits if use_moe=True
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            _, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            _, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.layers))

        seq_length -= self.config.num_nextn_predict_layers
        seq_length_with_past = seq_length
        cache_length = 0
        if past_key_values[0] is not None:
            cache_length = paddle.shape(past_key_values[0][0])[1]
            seq_length_with_past += cache_length
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        inputs_embeds = inputs_embeds.astype(self.embed_tokens.weight.dtype)
        if self.config.num_nextn_predict_layers > 0:
            inputs_embeds_extra = inputs_embeds[:, -self.config.num_nextn_predict_layers :, :]
            inputs_embeds = inputs_embeds[:, : -self.config.num_nextn_predict_layers, :]
            inputs_embeds_ori = inputs_embeds

            if position_ids is not None:
                position_ids_extra = position_ids[:, -self.config.num_nextn_predict_layers :]
                position_ids = position_ids[:, : -self.config.num_nextn_predict_layers]
                position_ids_ori = position_ids

            if attention_mask is not None:
                attention_mask_full = attention_mask
                attention_mask = attention_mask[
                    :,
                    :,
                    : -self.config.num_nextn_predict_layers,
                    : -self.config.num_nextn_predict_layers,
                ]

            if attn_mask_start_row_indices is not None:
                attn_mask_start_row_indices_extra = attn_mask_start_row_indices[
                    :, :, -self.config.num_nextn_predict_layers :
                ]
                attn_mask_start_row_indices = attn_mask_start_row_indices[
                    :, :, : -self.config.num_nextn_predict_layers
                ]
                attn_mask_start_row_indices_ori = attn_mask_start_row_indices

            nbatch_pack_offset = kwargs.get("nbatch_pack_offset", None)
            if nbatch_pack_offset is None:
                raise ValueError("nbatch_pack_offset is required in mtp train")

            nbatch_pack_offset_extra = nbatch_pack_offset[:, -self.config.num_nextn_predict_layers :]
            nbatch_pack_offset = nbatch_pack_offset[:, -self.config.num_nextn_predict_layers :]
            nbatch_pack_offset_ori = nbatch_pack_offset

        if self.config.sequence_parallel:
            inputs_embeds = inputs_embeds.reshape([-1, inputs_embeds.shape[-1]])
            inputs_embeds = ScatterOp.apply(inputs_embeds)

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        all_router_loss = 0.0 if self.config.use_moe else None
        all_gate_logits = ()
        mtp_outputs = []

        for idx, (decoder_layer) in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None
            has_gradient = not hidden_states.stop_gradient
            if self.config.recompute and self.config.recompute_granularity == "full" and has_gradient:
                layer_outputs = self.recompute_training(
                    decoder_layer,
                    hidden_states,
                    attention_mask,
                    attn_mask_start_row_indices,
                    position_ids,
                    token_type_ids,
                    output_attentions,
                    past_key_value,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask,
                    attn_mask_start_row_indices,
                    position_ids,
                    token_type_ids,
                    output_attentions,
                    past_key_value,
                    use_cache,
                )

            if isinstance(layer_outputs, (tuple, list)):
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
            if self.config.use_moe:
                if not (self.config.recompute and self.config.recompute_granularity == "full" and has_gradient):
                    layer_outputs, gate_logits = layer_outputs[:-1], layer_outputs[-1]
                    all_gate_logits = all_gate_logits + (gate_logits,)

        # Multi Token Prediction
        if self.config.num_nextn_predict_layers > 0:
            mtp_outputs.append(hidden_states)

            for depth in range(self.config.num_nextn_predict_layers):
                if self.config.sequence_parallel:
                    hidden_states = GatherOp.apply(hidden_states)
                    hidden_states = hidden_states.reshape([-1, seq_length, hidden_states.shape[-1]])

                inputs_embeds_cur_depth = paddle.concat(
                    [
                        inputs_embeds_ori[:, (depth + 1) :, :],
                        inputs_embeds_extra[:, : (depth + 1), :],
                    ],
                    axis=1,
                )

                if attention_mask is not None:
                    b, h, seqlen, seqlen = attention_mask.shape
                    attention_mask = attention_mask_full[
                        :,
                        :,
                        (depth + 1) : (seqlen + depth + 1),
                        (depth + 1) : (seqlen + depth + 1),
                    ]

                if attn_mask_start_row_indices is not None:
                    attn_mask_start_row_indices = paddle.concat(
                        [
                            attn_mask_start_row_indices_ori[:, :, (depth + 1) :],
                            attn_mask_start_row_indices_extra[:, :, : (depth + 1)],
                        ],
                        axis=-1,
                    )
                if position_ids is not None:
                    position_ids = paddle.concat(
                        [
                            position_ids_ori[:, (depth + 1) :],
                            position_ids_extra[:, : (depth + 1)],
                        ],
                        axis=1,
                    )
                nbatch_pack_offset_cur_depth = paddle.concat(
                    [
                        nbatch_pack_offset_ori[:, (depth + 1) :],
                        nbatch_pack_offset_extra[:, : (depth + 1)],
                    ],
                    axis=1,
                )
                hidden_states = mtp_hidden_states_set_zero(hidden_states, nbatch_pack_offset_cur_depth)

                # Norm&Concat
                inputs_embeds_cur_depth_norm = self.mtp_emb_norm[depth](inputs_embeds_cur_depth)
                hidden_states_norm = self.mtp_hidden_norm[depth](hidden_states)

                inputs_embeds_cur_depth = self.mtp_linear_proj[depth](
                    paddle.concat([inputs_embeds_cur_depth_norm, hidden_states_norm], axis=-1)
                )

                if self.config.sequence_parallel:
                    inputs_embeds_cur_depth = inputs_embeds_cur_depth.reshape([-1, inputs_embeds_cur_depth.shape[-1]])
                    inputs_embeds_cur_depth = ScatterOp.apply(inputs_embeds_cur_depth)

                decoder_layer = self.mtp_block[depth]
                past_key_value = None
                layer_outputs = decoder_layer(
                    inputs_embeds_cur_depth,
                    attention_mask,
                    attn_mask_start_row_indices,
                    position_ids,
                    token_type_ids,
                    output_attentions,
                    past_key_value,
                    use_cache,
                )
                if isinstance(layer_outputs, (tuple, list)):
                    hidden_states = layer_outputs[0]
                else:
                    hidden_states = layer_outputs

                if self.config.use_moe:
                    if not (self.config.recompute and has_gradient):
                        layer_outputs, gate_logits = (
                            layer_outputs[:-1],
                            layer_outputs[-1],
                        )
                        all_gate_logits = all_gate_logits + (gate_logits,)

                mtp_outputs.append(hidden_states)
            mtp_outputs = [self.norm(hidden_states) for hidden_states in mtp_outputs]
            hidden_states, mtp_outputs = mtp_outputs[0], mtp_outputs[1:]
        else:
            hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_self_attns,
                    all_router_loss,
                    all_gate_logits,
                    mtp_outputs,
                ]
                if v is not None
            )

        # assert all_router_loss is None, f'moe not support `return-dict`'
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=None,
            router_loss=all_router_loss,
            gate_logits=all_gate_logits,
            mtp_outputs=mtp_outputs,
        )


Ernie4_5_MoeLMHead = Ernie4_5_LMHead


class ErniePretrainingCriterion(ErniePretrainingCriterionBase):
    """Criterion for ERNIE pretraining task."""

    def __init__(self, config, return_tuple=True):
        """Initialize the pretraining criterion.

        Args:
            config (Ernie4_5_Config): Model configuration.
            return_tuple (bool): Whether to return loss as tuple (loss, loss_sum). Defaults to True.
        """
        super(ErniePretrainingCriterion, self).__init__(config, return_tuple=return_tuple)
        self.ignored_index = getattr(config, "ignored_index", -100)
        self.config = config
        self.return_tuple = return_tuple
        self.enable_parallel_cross_entropy = config.tensor_parallel_degree > 1 and config.tensor_parallel_output

        if self.enable_parallel_cross_entropy:  # and False: # and lm_head is distributed
            logger.info("using parallel cross entroy, take care")
            self.loss_func = fleet.meta_parallel.ParallelCrossEntropy()
        else:
            self.loss_func = paddle.nn.CrossEntropyLoss(
                reduction="none",
            )

    def forward(
        self,
        prediction_scores,
        masked_lm_labels,
        loss_mask,
        router_loss=None,
        mtp_logits=None,
    ):
        """Compute the combined pretraining loss.

        Args:
            prediction_scores: Prediction scores tensor, [batch_size, seq_len, vocab_size]
            masked_lm_labels: Target labels tensor [batch_size, seq_len]
            loss_mask: Optional mask for valid tokens
            router_loss: Optional MoE router loss tensor

        Returns:
            Union:
                - If return_tuple=True: Tuple of (combined_loss, mlm_loss_sum)
                - If return_tuple=False: Combined loss tensor
        """
        if self.config.num_nextn_predict_layers > 0:
            masked_lm_labels_ori = masked_lm_labels
            masked_lm_labels = masked_lm_labels[:, : -self.config.num_nextn_predict_layers]
            loss_mask = loss_mask[:, : -self.config.num_nextn_predict_layers]
            seq_length = masked_lm_labels.shape[1]

        res = super().forward(
            prediction_scores,
            masked_lm_labels,
            loss_mask,
        )

        if self.config.num_nextn_predict_layers > 0:
            mtp_loss_res = []
            for depth in range(self.config.num_nextn_predict_layers):
                prediction_scores_cur_depth = mtp_logits[depth]
                masked_lm_labels_cur_depth = masked_lm_labels_ori[:, (depth + 1) : (depth + 1 + seq_length)]
                res_cur_depth = super().forward(prediction_scores_cur_depth, masked_lm_labels_cur_depth, loss_mask)
                mtp_loss_res.append(res_cur_depth)

        def add_loss(main_loss, loss):
            return main_loss + loss - loss.detach()

        if self.return_tuple:
            loss, loss_sum = res
            if self.config.num_nextn_predict_layers > 0:
                loss = add_loss(
                    loss,
                    self.config.multi_token_pred_lambda * sum([x[0] for x in mtp_loss_res]) / len(mtp_loss_res),
                )
                loss_sum = loss_sum + self.config.multi_token_pred_lambda * sum(
                    [x[1].detach() for x in mtp_loss_res]
                ) / len(mtp_loss_res)
        else:
            loss, loss_sum = res, None
            if self.config.num_nextn_predict_layers > 0:
                loss = add_loss(
                    loss,
                    self.config.multi_token_pred_lambda * sum([x[0] for x in mtp_loss_res]) / len(mtp_loss_res),
                )

        if router_loss is not None and isinstance(router_loss, paddle.Tensor):
            loss = loss + router_loss - router_loss.detach()

        return loss, loss_sum


class Ernie4_5_MoeForCausalLM(Ernie4_5_PretrainedModel):
    """ERNIE Mixture of Experts (MoE) model for causal language modeling."""

    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config):
        """
        Initializes the ERNIE MoE model for causal language modeling.

        Args:
            config (dict): Model configuration.
        """
        super().__init__(config)

        # initialize-trick for big model,
        # see https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/README.md#std-init
        new_initializer_range = math.sqrt(0.3333 / config.hidden_size)
        logger.info(f"change initializer-range from {config.initializer_range} to {new_initializer_range}")
        config.initializer_range = new_initializer_range
        self.config = config
        self.model = Ernie4_5_Model(config)
        self.lm_head = Ernie4_5_MoeLMHead(config)
        if self.config.dpo_config is not None:
            self.criterion = ErnieDPOCriterion(config)
        else:
            self.criterion = ErniePretrainingCriterion(config)

        self.tie_weights()  # maybe weight share

        if self.config.use_rmsnorm:
            if self.config.fuse_rms_norm:
                logger.info("Use fusedRMSNorm")
            else:
                logger.info("Use normal RMSNorm")
        else:
            if self.config.fuse_ln:
                logger.info("Use fusedLN")
            else:
                logger.info("Use normal LayerNorm")

    @paddle.no_grad()
    def set_state_dict(self, state_dict, *args, **kwargs):
        """
        Loads the model state dictionary.

        Args:
            state_dict (dict): Model state dictionary.
        """
        ret = super().set_state_dict(state_dict)
        return ret

    def get_input_embeddings(self):
        """Returns the input embeddings layer."""
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        """Sets the input embeddings layer."""
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        """Returns the output embeddings (LM head)."""
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """Sets the output embeddings layer."""
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        """Sets the ERNIE decoder model."""
        self.model = decoder

    def get_decoder(self):
        """Get the transformer decoder.

        Returns:
            nn.Layer: The decoder module
        """
        return self.model

    def prepare_attention_mask_for_generation(self, input_ids, pad_token_id, eos_token_id):
        """Avoid using attention_mask with flash_attn on generation."""
        if self.config.use_flash_attention:
            return None
        return super().prepare_attention_mask_for_generation(input_ids, pad_token_id, eos_token_id)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        use_cache=False,
        past_key_values=None,
        inputs_embeds=None,
        **kwargs,
    ):
        """Prepares model inputs for generation in PaddlePaddle models.

        Args:
            input_ids (paddle.Tensor):
                The input token IDs with shape [batch_size, sequence_length].
            use_cache (bool, optional):
                Whether to use cached key-value states for faster generation.
                Defaults to False.
            past_key_values (Optional[Tuple[paddle.Tensor]]):
                Cached past key-value states from previous generation steps.
                If provided, the input_ids will be truncated to only keep the last token.
            inputs_embeds (Optional[paddle.Tensor]):
                Precomputed embeddings instead of token IDs.
                Only used in the first generation step when past_key_values is None.
            **kwargs:
                Additional keyword arguments including:
                - attention_mask (paddle.Tensor): Attention mask tensor
                - position_ids (paddle.Tensor): Position IDs (required if config.rope_3d=True)

        Returns:
            Dict[str, Union[paddle.Tensor, bool, Dict]]:
            A dictionary containing:
                - "input_ids" or "inputs_embeds": The main input tensors
                - "past_key_values": The cached key-value states
                - "use_cache": Flag indicating whether to use caching
                - "attention_mask": The attention mask tensor (if provided)
                - "position_ids": Position IDs (if config.rope_3d=True)
                - "return_dict": Always set to True for consistent output format

        """
        if past_key_values:
            input_ids = input_ids[:, -1:]

        attention_mask = kwargs.get("attention_mask", None)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": True,  # use_cache,
                "attention_mask": attention_mask,
                "return_dict": True,
            }
        )

        if self.config.get("rope_3d", False):
            model_inputs.update({"position_ids": kwargs["position_ids"]})

        return model_inputs

    # @staticmethod
    def update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder=False):
        """
        Updates model kwargs for generation.

        Args:
            outputs (Any): Model outputs.
            model_kwargs (dict): Current model kwargs.
            is_encoder_decoder (bool): Whether using encoder-decoder architecture.

        Returns:
            dict: Updated model kwargs.
        """
        # update cache
        if isinstance(outputs, tuple) and len(outputs) > 1 and not isinstance(outputs[1], paddle.Tensor):
            model_kwargs["past_key_values"] = outputs[1]

        if isinstance(outputs, CausalLMOutputWithCrossAttentions) and "past_key_values" in outputs:
            model_kwargs["past_key_values"] = outputs.past_key_values

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs and model_kwargs["token_type_ids"] is not None:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = paddle.concat([token_type_ids, token_type_ids[:, -1:]], axis=-1)

        if not is_encoder_decoder and model_kwargs.get("attention_mask", None) is not None:
            # update attention mask
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = paddle.concat(
                [
                    attention_mask,
                    paddle.ones([attention_mask.shape[0], 1], dtype="int64"),
                ],
                axis=-1,
            )
        # update role_ids
        if "role_ids" in model_kwargs and model_kwargs["role_ids"] is not None:
            role_ids = model_kwargs["role_ids"]
            model_kwargs["role_ids"] = paddle.concat([role_ids, role_ids[:, -1:]], axis=-1)

        if self.config.get("rope_3d", False):
            assert "position_ids" in model_kwargs, "position_ids must be provided if rope_3d is on"
            position_ids = model_kwargs["position_ids"]

            # becasue the model can only generate text.
            model_kwargs["position_ids"] = paddle.concat(
                [
                    position_ids,
                    position_ids.max(axis=(1, 2), keepdim=True).tile([1, 1, 3]) + 1,
                ],
                axis=1,
            )

        return model_kwargs

    def forward(
        self,
        input_ids,
        position_ids=None,
        attention_mask=None,
        attn_mask_start_row_indices=None,
        token_type_ids=None,  # for moe token-type routing
        inputs_embeds=None,
        labels=None,
        loss_mask=None,
        use_cache=False,
        past_key_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=False,  # true when decode, false when pretrain & eval
        **kwargs,
    ):
        """
        Forward pass for causal language modeling.

        Args:
            input_ids (paddle.Tensor): Input token IDs.
            position_ids (paddle.Tensor): Position IDs.
            attention_mask (paddle.Tensor): Attention mask.
            attn_mask_start_row_indices (paddle.Tensor): Attention mask start indices.
            inputs_embeds (paddle.Tensor): Optional embedded inputs.
            labels (paddle.Tensor): Target labels.
            loss_mask (paddle.Tensor): Loss mask.
            use_cache (bool): Whether to use cached hidden states.
            past_key_values (dict): Pre-computed hidden states.
            output_attentions (bool): Whether to output attentions.
            output_hidden_states (bool): Whether to output hidden states.
            return_dict (bool): Whether to return a dictionary.

        Returns:
            Union[tuple, CausalLMOutputWithCrossAttentions]: Model outputs.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if attention_mask is not None and attention_mask.dtype != paddle.bool:
            attention_mask = paddle.cast(attention_mask, paddle.bool)

        outputs = self.model(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            attn_mask_start_row_indices=attn_mask_start_row_indices,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            nbatch_pack_offset=kwargs.get("nbatch_pack_offset", None),
        )

        hidden_states = outputs.last_hidden_state
        mtp_outputs = outputs.mtp_outputs

        if isinstance(self.criterion, ErnieDPOCriterion):
            logits = (
                hidden_states,
                self.lm_head.weight,
                None,
                self.config.tie_word_embeddings,
            )
            chosen_labels = kwargs.get("chosen_labels", None)
            rejected_labels = kwargs.get("rejected_labels", None)
            response_indexs = kwargs.get("response_indexs", None)
            score_deltas = kwargs.get("score_deltas", None)
            reference_chosen_logps = kwargs.get("reference_chosen_logps", None)
            reference_rejected_logps = kwargs.get("reference_rejected_logps", None)
            if score_deltas:
                labels = (
                    chosen_labels,
                    rejected_labels,
                    response_indexs,
                    score_deltas,
                    reference_chosen_logps,
                    reference_rejected_logps,
                )
            else:
                labels = (
                    chosen_labels,
                    rejected_labels,
                    response_indexs,
                    reference_chosen_logps,
                    reference_rejected_logps,
                )
            return self.criterion(
                logits,
                labels,
            )

        # if labels is None，means we need full output, instead of tensor_parallel_output
        # tensor_parallel_output is togather with ParallelCrossEntropy
        logits = self.lm_head(hidden_states)
        mtp_logits = []
        if len(mtp_outputs) > 0:
            mtp_logits = [self.lm_head(_hidden_states) for _hidden_states in mtp_outputs]

        if return_dict:  # aka Generate Decoding
            if labels is not None:
                loss, _ = self.criterion(logits, labels, loss_mask)
            else:
                loss = None
            return CausalLMOutputWithCrossAttentions(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                router_loss=outputs.router_loss if self.config.use_moe else None,
            )
        if self.config.use_moe:
            router_loss = outputs.router_loss
        else:
            router_loss = None

        # Pretrain & Eval must have labels
        assert labels is not None

        return self.criterion(logits, labels, loss_mask, router_loss, mtp_logits)


Ernie4_5_ForCausalLM = Ernie4_5_MoeForCausalLM
