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


import logging
import os

from paddleformers.trainer.trainer_callback import TrainerCallback

logger = logging.getLogger(__name__)
g_shard_bypass_dygraph_optimizer = int(os.environ.get("FLAGS_shard_bypass_dygraph_optimizer", 0))


def enable_in_dict_config(config, key):
    return key in config and config[key]


skip_count = 0


class FP8QuantWeightCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        model = kwargs["model"]
        optimizer = kwargs["optimizer"]
        global skip_count

        # offline quant fp8 weight
        if enable_in_dict_config(model.config.fp8_mem_configs, "offline_quant_expert_weight"):
            logger.info("offline quant expert weight from bf16 to fp8.")
            clear_origin_weight = enable_in_dict_config(
                model.config.fp8_mem_configs, "clear_origin_weight_when_offline_quant"
            )

            if not g_shard_bypass_dygraph_optimizer or skip_count == 0:
                model.fp8_quant_weight()

            if clear_origin_weight:
                logger.info("clear origin bf16 weight after fp8 quant.")
                optimizer.clear_param_storage("moe_expert")

        skip_count += 1
