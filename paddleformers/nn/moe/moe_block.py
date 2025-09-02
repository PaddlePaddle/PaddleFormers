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


"""
moe_layer_all_gather
"""

from typing import List, Optional

import paddle
from paddle import nn
from paddle.distributed.communication.group import Group

from .abstract import MOELayerBase
from .moe_allgather_layer import MOEAllGatherLayerV2
from .moe_alltoall_layer import MOEAlltoAllLayer


def create_moe_block(
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
) -> MOELayerBase:
    if moe_mode == "allgather":
        model = MOEAllGatherLayerV2(
            gate,
            experts,
            layer_idx,
            shared_experts,
            group,
            recompute,
            k,
            enable_reverse_token_drop,
            all_to_all_dropout,
            group_experts,
            use_expert_out_alltoall,  #
            use_padding,
            dense_token_type,  # considerd as dense tokens (no moe)
            moe_statics,
            moe_num_experts,
        )
    elif moe_mode == "alltoall":
        model = MOEAlltoAllLayer(
            gate,
            experts,
            layer_idx,
            shared_experts,
            group,
            recompute,
            k,
            all_to_all_dropout,
            group_experts,
            moe_statics,
            moe_num_experts,
        )
    else:
        raise ValueError("Invalid moe_mode")

    return model


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
        use_multimodel_experts = config.get("multimodel_experts", False)

        num_experts = config.moe_num_experts[0] if use_multimodel_experts else config.moe_num_experts
        if use_multimodel_experts:
            assert (
                len(set(config.moe_num_experts)) == 1
            ), f"assume expert group has same size, got: {config.moe_num_experts}"

        with paddle.utils.unique_name.guard(f"mm_layer_{layer_idx}_"):
            num_experts_groups = len(config.moe_num_experts) if use_multimodel_experts else 1
            p = self.create_parameter(
                shape=[num_experts_groups, num_experts],
                dtype="float32",
                is_bias=True,
                attr=paddle.ParamAttr(name=paddle.utils.unique_name.generate("corr_bias")),
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
