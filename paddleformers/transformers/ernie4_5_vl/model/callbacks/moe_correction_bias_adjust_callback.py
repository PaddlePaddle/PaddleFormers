# !/usr/bin/env python3

# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
# limitations under the License."""
"""
SPGradSyncCallback
"""
import logging

import paddle
import paddle.distributed as dist
from paddle.distributed.fleet import fleet
from paddleformers.trainer.trainer_callback import TrainerCallback

from ernie.modeling_moe import Ernie4_5_DecoderLayer
from ernie.moe.moe_layer import MOELayer

logger = logging.getLogger(__name__)


def is_sequence_parallel_parameter(parameter):
    """
    Determines whether the passed parameter is a sequence-parallel parameter.

    Args:
        parameter (object): The parameter object to be checked.

    Returns:
        bool: Returns True if the parameter is a sequence-parallel parameter;
        otherwise, returns False.

    """
    return getattr(parameter, "sequence_parallel", False)


class MoECorrectionBiasAdjustCallback(TrainerCallback):
    """used for moe aux loss free balance"""

    def __init__(self, lr, use_sp):
        super().__init__()
        self.update_lr = lr
        self.use_sp = use_sp

    def on_optimizer_end(self, args, state, control, **kwargs):
        """on_optimizer_begin"""
        model = kwargs["model"]

        usages = {}
        biases = {}

        def get_stat(layer):
            nonlocal usages, biases
            if isinstance(layer, Ernie4_5_DecoderLayer):
                if not isinstance(layer.mlp, MOELayer):
                    return
                assert hasattr(
                    layer.mlp, "moe_statics"
                ), "make sure update to latest ernie-core, too use AuxFree Balance"
                usages[layer.layer_idx] = (
                    layer.mlp.moe_statics.expert_usage
                )  # usage list
                biases[layer.layer_idx] = layer.mlp.moe_statics.e_score_correction_bias

        model.apply(get_stat)
        keys, tensor_list = zip(*sorted(usages.items(), key=lambda x: x[0]))
        usages_tensor = paddle.stack(
            tensor_list, 0
        )  # [num_layers, 2, num_experts_per_modality]
        if not hasattr(fleet, "_hcg"):
            dist.all_reduce(usages_tensor)
            return

        hcg = fleet.get_hybrid_communicate_group()
        mp_group = hcg.get_model_parallel_group()
        dp_group = hcg.get_data_parallel_group()
        sd_group = hcg.get_sharding_parallel_group()
        logger.info("allgather moe expert statics")
        if self.use_sp and mp_group.nranks > 1:
            dist.all_reduce(usages_tensor, group=mp_group)
        if dp_group.nranks > 1:
            dist.all_reduce(usages_tensor, group=dp_group)
        if sd_group.nranks > 1:
            dist.all_reduce(usages_tensor, group=sd_group)
        logger.info("done allgather moe expert statics")

        usages_mean = usages_tensor.mean(-1, keepdim=True)
        update = paddle.sign(usages_mean - usages_tensor) * self.update_lr
        update_dict = dict(zip(keys, update))

        def update_bias(layer):
            nonlocal usages, biases
            if isinstance(layer, Ernie4_5_DecoderLayer):
                if not isinstance(layer.mlp, MOELayer):
                    return
                with paddle.no_grad():
                    if layer.mlp.gate.weight.stop_gradient:
                        update_dict[layer.layer_idx][0, :] = 0
                    biases[layer.layer_idx].add_(update_dict[layer.layer_idx])
                    usages[layer.layer_idx].data.zero_()

        model.apply(update_bias)
