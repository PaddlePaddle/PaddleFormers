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
from paddle import nn
from paddle.distributed import fleet
from paddle.distributed.communication.group import Group
from paddleformers.utils.log import logger

from .moe_alltoall import moe_alltoall_forward
from .moe_allgather import moe_allgather_forward

class MOE(nn.Layer):
    _global_mapping = {
        "alltoall": moe_forward,
        "allgather": moe_allgather_forward,
    }
    
    def __init__(
        self,
        gate: nn.Layer,
        experts: List[nn.Layer],
        layer_idx,
        shared_experts: Optional[List[nn.Layer]] = None,
        group: Group = None,
        recompute=False,
        k=2,
        enable_reverse_token_drop=False,
        all_to_all_dropout=0,
        group_experts=False,
        use_expert_out_alltoall=True,  #
        use_padding=True,
        dense_token_type=3,  # considerd as dense tokens (no moe)
        moe_statics=None,
        moe_num_experts=None,
        moe_mode="allgather",
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
            logger.info(
                f"using correction bias, aux-coef:{self.gate.config.moe_aux_loss_lambda}"
            )
            assert self.gate.config.moe_use_aux_free

        self.is_mp_moe = (
            hasattr(fleet.fleet, "_hcg")
            and group is fleet.get_hybrid_communicate_group().get_model_parallel_group()
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

        # self.multimodal_experts = (
        #     isinstance(moe_num_experts, (tuple, list)) and len(moe_num_experts) > 1
        # )
        self.num_local_experts = len(self.experts) // self.world_size
        # if self.multimodal_experts:
        #     self.num_local_multimodal_experts = [
        #         num // self.world_size for num in moe_num_experts
        #     ]
        #     self.multimodal_expert_index = [0] + list(
        #         itertools.accumulate(moe_num_experts)
        #     )

        self.input_preprocess = self.output_postprocess = None
        self.group_experts = group_experts
        self.config = self.gate.config
        # self.zero = paddle.to_tensor(0, dtype=paddle.float32)
        self.moe_mode = moe_mode

        if (self.moe_mode == "allgather"):
            self.enable_reverse_token_drop = enable_reverse_token_drop
            self.is_allgather_moe_layer = is_allgather_moe_layer
            self.use_padding = use_padding

            # 全局 gate gather
            self.send_rank = None
            self.local_expert_id = None
            self.dense_experts = None
            self.dense_token_type = dense_token_type
            self.capacity_tensor = None
            self.use_expert_out_alltoall = use_expert_out_alltoall
            logger.info(
                f"uisng MOEAllGatherLayerV2, use_expert_out_alltoall={use_expert_out_alltoall}, "  # false
                f"use_padding={use_padding}, enable_reverse_token_drop={self.enable_reverse_token_drop}"  # true false
            )
            # self.two = paddle.to_tensor(2, dtype=paddle.float32)
                
    def forward(
        self,
        input: paddle.Tensor,
        token_type_ids=None,
        use_dense_expert=False,
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        if (self.moe_mode == "allgather"):           
            return moe_allgather_forward(
                input=input,
                token_type_ids=token_type_ids,
                use_dense_expert=use_dense_expert
                config=self.config,
                gate=self.gate,
                k=self.k,
                use_correction_bias=self.use_correction_bias,
                moe_statics=self.moe_statics,
                world_size=self.world_size,
                num_local_experts=self.num_local_experts,
                shared_experts=self.shared_experts,
                group=self.group,
                experts=self.experts,
                rank=self.rank,
                isRecompute=self.recompute,
                isTraining=self.training,
                layer_idx=self.layer_idx,
                dense_token_type=self.dense_token_type,)
        elif (self.moe_mode == "alltoall"):
            return moe_alltoall_forward(
                input=input, 
                token_type_ids=token_type_ids,
                config=self.config,
                gate=self.gate,
                k=self.k,
                use_correction_bias=self.use_correction_bias,
                moe_statics=self.moe_statics,
                world_size=self.world_size,
                num_local_experts=self.num_local_experts,
                shared_experts=self.shared_experts,
                group=self.group,
                experts=self.experts,
                rank=self.rank,
                isRecompute=self.recompute,
                isTraining=self.training,
                layer_idx=self.layer_idx,
                )
        else:
            raise ValueError("Unsupported MOE mode: {}".format(self.moe_mode))


class MoEStatics(nn.Layer):
    """
    Stores MoE (Mixture of Experts) statistics
    and expert usage information.
    """

    def __init__(self, config, layer_idx):
        """
        Initialize MoE statistics tracking.

        Args:
            config: Model configuration containing MoE parameters
            layer_idx: Index of the MoE layer in the model
        """
        super().__init__()
        self._cast_to_low_precision = False  # 兼容develop分支paddle
        self._cast_to_low_precison = False
        num_experts = (
            config.moe_num_experts[0]
            if config.multimodel_experts
            else config.moe_num_experts
        )
        # if config.multimodel_experts:
        #     assert (
        #         len(set(config.moe_num_experts)) == 1
        #     ), f"assume expert group has same size, got: {config.moe_num_experts}"

        with paddle.utils.unique_name.guard(f"mm_layer_{layer_idx}_"):
            num_experts_groups = (
                len(config.moe_num_experts) if config.multimodel_experts else 1
            )
            p = self.create_parameter(
                shape=[num_experts_groups, num_experts],
                dtype="float32",
                is_bias=True,
                attr=paddle.ParamAttr(
                    name=paddle.utils.unique_name.generate("corr_bias")
                ),
            )
            p.stop_gradient = True
            self.e_score_correction_bias = p
            self.e_score_correction_bias.is_distributed = True
            p = paddle.zeros(
                shape=[num_experts_groups, num_experts],
                dtype="int64",
            )
            p.stop_gradient = True
            self.expert_usage = p

