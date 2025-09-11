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

import copy

import paddle.nn as nn
from paddle.distributed.fleet.meta_parallel import ParallelCrossEntropy

from ...utils.log import logger
from ..general import GeneralInterface
from .dpo_loss import dpo_loss_forward
from .kto_loss import kto_loss_forward
from .sft_loss import mtp_sft_loss_forward, sft_loss_forward


class LossInterface(GeneralInterface):

    _global_mapping = {
        "sft": sft_loss_forward,
        "dpo": dpo_loss_forward,
        "kto": kto_loss_forward,
        "mtp_sft": mtp_sft_loss_forward,
    }


ALL_LOSS_FUNCTIONS = LossInterface()


class CriterionLayer(nn.Layer):
    def __init__(self, config, return_tuple=True, ignore_eos_token=False, use_infohub=False, **kwargs):
        super().__init__()
        self.config = config
        self.dpo_config = copy.deepcopy(config.get("dpo_config", None))
        self.kto_config = copy.deepcopy(config.get("kto_config", None))
        self.ignored_index = getattr(config, "ignored_index", -100)
        self.use_filtered_label_loss = config.get("use_filtered_label_loss", False)
        self.loss_subbatch_sequence_length = config.get("loss_subbatch_sequence_length", -1)
        self.use_subbatch = self.loss_subbatch_sequence_length > 0
        self.sequence_parallel = config.get("sequence_parallel", False)
        self.tensor_parallel = config.tensor_parallel_degree > 1
        self.use_fused_head_and_loss_fn = config.get("use_fused_head_and_loss_fn", False)
        self.enable_parallel_cross_entropy = config.tensor_parallel_degree > 1 and config.tensor_parallel_output
        logger.info(
            f"loss_subbatch_sequence_length: {self.loss_subbatch_sequence_length} , use_fused_head_and_loss_fn: {self.use_fused_head_and_loss_fn}, use_filtered_label_loss: {self.use_filtered_label_loss}"
        )

        self.return_tuple = return_tuple
        self.tie_word_embeddings = config.get("tie_word_embeddings", False)
        self.use_infohub = use_infohub
        self.ignore_eos_token = ignore_eos_token

        if self.enable_parallel_cross_entropy:
            logger.info("using parallel cross entroy, take care")
            self.loss_func = ParallelCrossEntropy()
        else:
            self.loss_func = nn.CrossEntropyLoss(
                reduction="none",
            )

        assert not config.get("dpo_config", None) or not config.get(
            "kto_config", None
        ), "dpo_config and kto_config cannot be both set"

        if kwargs.get("loss_type", None):
            loss_type = kwargs["loss_type"]
        elif config.get("dpo_config", None):
            loss_type = "dpo"
        elif config.get("kto_config", None):
            loss_type = "kto"
        else:
            loss_type = "sft"

            if config.get("num_nextn_predict_layers", 0) > 0:
                loss_type = "mtp_sft"

        self.loss_foward_fn = ALL_LOSS_FUNCTIONS.get(loss_type)
        self.loss_type = loss_type

    def forward(self, logits, labels, loss_mask=None, **kwargs):
        loss = self.loss_foward_fn(self, logits, labels, loss_mask, **kwargs)
        return loss
