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

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Dict, Optional

import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.distributed import fleet
from paddle.distributed.fleet.utils.sequence_parallel_utils import GatherOp, ScatterOp

from ...transformers.configuration_utils import PretrainedConfig
from ...transformers.token_dispatcher import MoEFlexTokenDispatcher
from .moe_communication import AllToAllMoECommunication, DeepEPMoECommunication
from .moe_expert import StandardMLPExpert
from .moe_gate import StandardMoEGate
from .moe_loss import AddAuxiliaryLoss
from .moe_loss_instance import get_global_loss_registry

logger = logging.getLogger(__name__)
global_loss_registry = get_global_loss_registry()


class ModularMoELayer(nn.Layer):
    def __init__(
        self,
        hidden_size: int,
        moe_intermediate_size: int,
        num_experts: int,
        num_shared_experts: int,
        num_experts_per_tok: int,
        norm_topk_prob: int,
        expert_activation: str,
        moe_config: Dict,
        model_type: str,
        expert_class,
        transpose_gate_weight: bool,
        pretrained_config: Optional[PretrainedConfig] = None,
    ):

        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_shared_experts = num_shared_experts
        self.moe_intermediate_size = moe_intermediate_size
        self.expert_activation = expert_activation
        self.norm_topk_prob = norm_topk_prob
        self.model_type = model_type
        self.expert_class = expert_class
        self.transpose_gate_weight = transpose_gate_weight

        self.sequence_parallel = pretrained_config.get("sequence_parallel", False)
        self.tensor_parallel_degree = pretrained_config.get("tensor_parallel_degree", 1)
        self.seq_length = pretrained_config.get("seq_length", pretrained_config.get("max_seq_len", 1024))
        self.fuse_up_gate = pretrained_config.get("fuse_attention_ffn", False)
        self.ep_communication_type = pretrained_config.get("ep_communication_type", "deepep")
        self.n_group = pretrained_config.get("n_group", 1)
        self.topk_group = pretrained_config.get("topk_group", 1)
        self.routed_scaling_factor = pretrained_config.get("routed_scaling_factor", 1.0)
        self.aux_loss_alpha = pretrained_config.get("aux_loss_alpha", 0.0)
        self.moe_subbatch_token_num = pretrained_config.get("moe_subbatch_token_num", -1)
        try:
            moe_group = fleet.get_hybrid_communicate_group().get_expert_parallel_group()
        except Exception:
            moe_group = None
        self.expert_parallel_degree = dist.get_world_size(moe_group) if moe_group is not None else 1

        self.gate_activation = moe_config.get("gate_activation", "softmax")
        self.topk_method = (
            moe_config.get("train_topk_method", "greedy")
            if self.training
            else moe_config.get("inference_topk_method", "greedy")
        )
        self.drop_tokens = moe_config.get("drop_tokens", False)
        self.use_flexible_loss = moe_config.get(
            "use_flexible_loss", False
        )  # TODO: use customized loss system, not implemented yet
        self.expert_dropout = moe_config.get("expert_dropout", 0.0)
        self.loss_configs = moe_config.get("loss_configs", None)
        self.loss_combiner_name = moe_config.get("loss_combiner_name", "weighted_sum")

        self._init_expert_parallel()
        self.gate = StandardMoEGate(
            num_experts=self.num_experts,
            expert_hidden_size=self.hidden_size,
            drop_tokens=self.drop_tokens,
            topk_method=self.topk_method,
            num_experts_per_tok=self.num_experts_per_tok,
            norm_topk_prob=self.norm_topk_prob,
            moe_config=moe_config,
            seq_length=self.seq_length,
            n_group=self.n_group,
            topk_group=self.topk_group,
            routed_scaling_factor=self.routed_scaling_factor,
            moe_subbatch_token_num=self.moe_subbatch_token_num,
            tensor_parallel_degree=self.tensor_parallel_degree,
            sequence_parallel=self.sequence_parallel,
            transpose_gate_weight=self.transpose_gate_weight,
        )

        if self.expert_class is None:
            self.expert_class = StandardMLPExpert

        routed_expert_pretrained_config = deepcopy(pretrained_config)
        shared_expert_pretrained_config = deepcopy(pretrained_config)
        if self.expert_parallel_degree <= 1 and self.sequence_parallel and self.tensor_parallel_degree > 1:
            routed_expert_pretrained_config.sequence_parallel = False
            shared_expert_pretrained_config.sequence_parallel = False
        elif self.expert_parallel_degree > 1 and self.tensor_parallel_degree >= 1:
            routed_expert_pretrained_config.tensor_parallel_degree = 1

        expert_args = {}
        expert_args["config"] = routed_expert_pretrained_config
        expert_args["intermediate_size"] = self.moe_intermediate_size
        # Add more arguments for different models
        if self.model_type == "qwen3_moe":
            pass
        elif self.model_type == "glm4_moe":
            expert_args["fuse_up_gate"] = self.fuse_up_gate

        self.experts = nn.LayerList([])
        for i in range(self.num_experts):
            if i // self.num_experts_per_device == self.moe_rank:
                self.experts.append(self.expert_class(**expert_args))
            else:
                self.experts.append(None)

        if self.expert_parallel_degree > 1:
            self.token_dispatcher = MoEFlexTokenDispatcher(
                self.num_experts_per_device, self.num_experts_per_tok, self.num_experts, self.moe_group
            )
        else:
            self.token_dispatcher = None

        shared_expert_args = {}
        shared_expert_args["config"] = shared_expert_pretrained_config
        shared_expert_args["intermediate_size"] = self.moe_intermediate_size * self.num_shared_experts
        # Add more arguments for different models
        if self.model_type == "qwen3_moe":
            pass
        elif self.model_type == "glm4_moe":
            shared_expert_args["fuse_up_gate"] = self.fuse_up_gate

        if self.num_shared_experts > 0:
            self.shared_experts = self.expert_class(**shared_expert_args)
        else:
            self.shared_experts = None

        if self.ep_communication_type == "deepep":
            self.communication = DeepEPMoECommunication()
        elif self.ep_communication_type == "alltoall":
            self.communication = AllToAllMoECommunication()
        else:
            raise ValueError(
                f"Unsupported communication type: {self.ep_communication_type}, please choose from ['deepep', 'alltoall']"
            )

        if hasattr(dist, "fleet") and dist.is_initialized() and self.expert_parallel_degree > 1:
            self.is_mp_moe = False
            self.is_ep_moe = True
            for p in self.experts.parameters():
                setattr(p, "is_moe_param", True)
                setattr(p, "color", {"color": "moe_expert", "group": self.moe_grad_group})
                p.no_sync = not self.is_mp_moe
                p.expert = not self.is_mp_moe
                logger.info(f"expert no-sync={p.no_sync}-{p.name}")
                if self.is_mp_moe or self.is_ep_moe:
                    p.is_distributed = True

    def _init_expert_parallel(self):
        def _parse_moe_expert_parallel(num_experts: int, expert_parallel_degree: int) -> int:
            """
            Args:
                num_experts: Total number of experts
                expert_parallel_degree: Expert parallel groups

            Returns:
                moe_num_experts_per_device: Number of experts per device
            """
            assert (
                num_experts >= expert_parallel_degree
            ), f"expert num_experts={num_experts} >= moe_world_size={expert_parallel_degree}"
            assert (
                num_experts % expert_parallel_degree == 0
            ), f"expert num_experts={num_experts} % moe_world_size={expert_parallel_degree} == 0"

            moe_num_experts_per_device = num_experts // expert_parallel_degree
            return moe_num_experts_per_device

        try:
            dist.fleet.get_hybrid_communicate_group()
            is_fleet_init = True
        except AttributeError:
            is_fleet_init = False

        if is_fleet_init and self.expert_parallel_degree > 1:
            self.moe_group = dist.fleet.get_hybrid_communicate_group().get_expert_parallel_group()
            self.moe_grad_group = dist.fleet.get_hybrid_communicate_group().get_moe_sharding_parallel_group()
            self.moe_rank = dist.get_rank(self.moe_group)
            self.moe_rank = 0 if self.moe_rank < 0 else self.moe_rank
            new_expert_parallel_degree = dist.get_world_size(self.moe_group)
            assert (
                self.expert_parallel_degree == new_expert_parallel_degree
            ), f"self.expert_parallel_degree={self.expert_parallel_degree} != moe_world_size={new_expert_parallel_degree}"
            self.expert_parallel_degree = 1 if new_expert_parallel_degree < 0 else new_expert_parallel_degree
            self.num_experts_per_device = _parse_moe_expert_parallel(self.num_experts, self.expert_parallel_degree)
        else:
            self.moe_group = None
            self.moe_rank = 0
            self.expert_parallel_degree = 1
            self.num_experts_per_device = self.num_experts

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        """
        Args:
            hidden_states: Shape: [batch_size, seq_len, hidden_size]

        Returns:
            output: Shape: [batch_size, seq_len, hidden_size]
        """
        if self.expert_parallel_degree <= 1 and self.sequence_parallel:
            hidden_states = GatherOp.apply(hidden_states)
        orig_shape = hidden_states.shape
        residuals = hidden_states
        capacity, topk_weights, topk_indices, gates_masked, mask, priorities, aux_loss, z_loss = self.gate(
            hidden_states
        )
        # topk_weights, topk_indices will be used in AllToAllMoECommunication
        # gates_masked, mask will be used in DeepEPMoECommunication
        # capacity, priorities are not used currently

        if self.expert_parallel_degree > 1:
            output = self._forward_with_ep_parallel(
                hidden_states, topk_indices, topk_weights, gates_masked, mask, priorities
            )
        else:
            if len(hidden_states.shape) == 3:
                batch_size, seq_len, d_model = hidden_states.shape
                reshaped_input = hidden_states.reshape([-1, d_model])
            else:
                reshaped_input = hidden_states
            output = self._forward_traditional_moe(reshaped_input, topk_indices, topk_weights)

        if self.training and self.aux_loss_alpha > 0.0:
            aux_loss = aux_loss * self.aux_loss_alpha
            output = AddAuxiliaryLoss.apply(output, aux_loss)

        if self.shared_experts is not None:
            shared_output = self.shared_experts(residuals)
            output = output + shared_output

        output = output.reshape(orig_shape)

        if self.expert_parallel_degree <= 1 and self.sequence_parallel:
            output = ScatterOp.apply(output)

        return output

    def _forward_traditional_moe(
        self, hidden_states: paddle.Tensor, selected_experts: paddle.Tensor, topk_weights: paddle.Tensor
    ) -> paddle.Tensor:
        """
        Forward without expert parallelism

        Args:
            hidden_states: Input hidden states, shape: [batch_size*seq_len, hidden_size]
            selected_experts: TopK experts indices, shape: [seq_len, num_experts_per_tok]
            topk_weights: TopK weights, shape: [seq_len, num_experts_per_tok]

        Returns:
            output: Output hidden states, shape: [seq_len, hidden_size]
        """

        _, d_model = hidden_states.shape
        final_hidden_states = paddle.zeros_like(hidden_states, dtype=hidden_states.dtype)

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = paddle.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).transpose([2, 1, 0])
        tokens_per_expert = expert_mask.reshape([expert_mask.shape[0], -1]).sum(axis=-1)
        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            top_x, idx = paddle.where(expert_mask[expert_idx])
            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            if tokens_per_expert[expert_idx] <= 0.1:
                continue
            current_state = hidden_states[idx, None].reshape([-1, d_model])
            current_hidden_states = expert_layer(current_state) * topk_weights[idx, top_x].unsqueeze(-1)
            final_hidden_states.index_add_(
                index=idx.reshape([-1]), axis=0, value=current_hidden_states.to(hidden_states.dtype)
            )

        return final_hidden_states.cast(hidden_states.dtype)

    def _forward_with_ep_parallel(
        self,
        hidden_states: paddle.Tensor,
        topk_indices: paddle.Tensor,
        topk_weights: paddle.Tensor,
        gates_masked: paddle.Tensor,
        mask: paddle.Tensor,
        priorities: paddle.Tensor,
    ) -> paddle.Tensor:
        """
        Forward with expert parallelism

        Args:
            hidden_states: Input hidden states, shape: [seq_len, hidden_size]
            topk_indices: TopK experts indices, shape: [seq_len, num_experts_per_token]
            topk_weights: TopK weights, shape: [seq_len, num_experts_per_token]
            gates_masked: Masked hidden_states，形状: [seq_len, num_experts]
            mask: One-hot encoding of the selected experts for each token, shape: [seq_len, num_experts]

        Returns:
            output: Output hidden states, shape: [seq_len, hidden_size]
        """
        output = self.communication.forward(
            hidden_states,
            topk_indices,
            topk_weights,
            gates_masked,
            mask,
            priorities,
            self.expert_parallel_degree,
            self.moe_group,
            self.experts,
            self.moe_rank,
            self.num_experts_per_device,
            self.num_experts,
            self.num_experts_per_tok,
            self.token_dispatcher,
        )
        return output

    def get_auxiliary_loss(self) -> paddle.Tensor:
        return self.gate.get_auxiliary_loss()

    def get_z_loss(self) -> paddle.Tensor:
        return self.gate.get_z_loss()

    def get_all_losses(self) -> Dict[str, paddle.Tensor]:
        if hasattr(self.gate, "get_all_losses"):
            return self.gate.get_all_losses()
        else:
            return {"auxiliary": self.get_auxiliary_loss(), "z_loss": self.get_z_loss()}

    def get_total_loss(self) -> paddle.Tensor:
        if hasattr(self.gate, "get_total_loss"):
            return self.gate.get_total_loss()
        else:
            return self.get_auxiliary_loss() + self.get_z_loss()

    def remove_loss_function(self, name: str):
        if not self.use_flexible_loss:
            logger.warning("Current not open `use_flexible_loss`, cannot remove custom losses")
            return

        if hasattr(self.gate, "remove_loss_config"):
            self.gate.remove_loss_config(name)
        else:
            logger.warning("Current not open `remove_loss_config` on gate, cannot remove custom losses")

    def update_loss_weights(self, weights: Dict[str, float]):
        if not self.use_flexible_loss:
            logger.warning("Current not open `use_flexible_loss`, cannot update loss weights")
            return

        if hasattr(self.gate, "update_loss_weights"):
            self.gate.update_loss_weights(weights)
        else:
            logger.warning("Current not open `update_loss_weights` on gate, cannot update loss weights")

    def set_loss_combiner(self, combiner_name: str):
        if not self.use_flexible_loss:
            logger.warning("Current not open `use_flexible_loss`, cannot set loss combiner")
            return

        if hasattr(self.gate, "set_loss_combiner"):
            self.gate.set_loss_combiner(combiner_name)
        else:
            logger.warning("Current not open `set_loss_combiner` on gate, cannot set loss combiner")

    def get_expert_info(self) -> Dict[str, Any]:
        return {
            "num_experts": self.num_experts,
            "num_experts_per_device": self.num_experts_per_device,
            "expert_parallel_degree": self.expert_parallel_degree,
            "moe_rank": self.moe_rank,
            "is_parallel_enabled": self.expert_parallel_degree > 1,
            "use_flexible_loss": self.use_flexible_loss,
        }
