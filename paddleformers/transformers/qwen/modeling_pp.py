# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.distributed.fleet as fleet
import paddle.nn as nn
from paddle.distributed.fleet.meta_parallel import LayerDesc, PipelineLayer

from ..dpo_criterion import DPOCriterion
from ..model_utils import PipelinePretrainedModel
from ..refined_recompute import get_skip_recompute_ops
from .modeling import (
    QWenBlock,
    QWenConfig,
    QWenLMHead,
    QWenModel,
    QWenPretrainedModel,
    QWenPretrainingCriterion,
    QWenRMSNorm,
)

__all__ = [
    "QWenForCausalLMPipe",
]


def parse_args(args):
    if isinstance(args, tuple):
        if len(args) == 3:
            hidden_states, attention_mask, position_ids = args
        elif len(args) == 2:
            hidden_states, attention_mask = args
            position_ids = None
        elif len(args) == 1:
            hidden_states = args
            attention_mask, position_ids = None, None
    else:
        hidden_states = args
        attention_mask, position_ids = None, None

    if position_ids is not None:
        position_ids.stop_gradient = True

    if attention_mask is not None:
        attention_mask.stop_gradient = True

    return hidden_states, attention_mask, position_ids


def return_args(hidden_states, attention_mask=None, position_ids=None):
    ret = (hidden_states,)

    if attention_mask is not None:
        ret += (attention_mask.clone(),)
    if position_ids is not None:
        ret += (position_ids.clone(),)
    if len(ret) == 1:
        ret = ret[0]

    return ret


class QWenEmbeddingPipe(nn.Layer):
    """Extends QWenEmbeddings to forward attention_mask through the pipeline."""

    def __init__(self, config):
        super(QWenEmbeddingPipe, self).__init__()
        self.hidden_size = config.hidden_size
        self.sequence_parallel = config.sequence_parallel
        if config.tensor_parallel_degree > 1:
            self.wte = fleet.meta_parallel.VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                weight_attr=paddle.ParamAttr(initializer=nn.initializer.XavierNormal()),
            )
        else:
            self.wte = nn.Embedding(config.vocab_size, config.hidden_size)

    def forward(self, args):
        """_summary_

        Args:
            input (_type_): _description_

        Returns:
            _type_: _description_
        """
        input_ids, attention_mask, position_ids = parse_args(args)
        input_embeds = self.wte(input_ids)
        if self.sequence_parallel:
            from paddle.distributed.fleet.utils.sequence_parallel_utils import ScatterOp

            # [bs, seq_len, num_head * head_dim] -> [bs * seq_len, num_head * head_dim]
            bs, seq_len, hidden_size = input_embeds.shape
            input_embeds = paddle.reshape_(input_embeds, [bs * seq_len, hidden_size])
            # [seq_len * bs / n, num_head * head_dim] (n is mp parallelism)
            input_embeds = ScatterOp.apply(input_embeds)

        batch_size, seq_length = input_ids.shape
        if attention_mask is not None:
            attention_mask = QWenModel._prepare_decoder_attention_mask(
                attention_mask, (batch_size, seq_length), 0, input_embeds.dtype
            )
            attention_mask.stop_gradient = True

        return return_args(input_embeds, attention_mask, position_ids)


class QWenBlockPipe(QWenBlock):
    def forward(self, args):
        hidden_states, attention_mask, position_ids = parse_args(args)
        hidden_states = super().forward(hidden_states, attention_mask=attention_mask)
        return return_args(hidden_states, attention_mask, position_ids)


class QWenRMSNormPipe(QWenRMSNorm):
    def forward(self, args):
        hidden_states, attention_mask, position_ids = parse_args(args)
        return super().forward(hidden_states)


class QWenForCausalLMPipe(PipelinePretrainedModel, PipelineLayer):
    """QWenForPretraining adapted for pipeline parallelism.

    The largest change is flattening the QWenModel class so we can express it as a
    sequence of layers including embedding, transformer layers, and output.
    """

    config_class = QWenConfig

    _get_tensor_parallel_mappings = QWenPretrainedModel._get_tensor_parallel_mappings
    _init_weights = QWenPretrainedModel._init_weights
    _keys_to_ignore_on_load_unexpected = QWenPretrainedModel._keys_to_ignore_on_load_unexpected
    _get_model_flops = QWenPretrainedModel._get_model_flops
    _get_hardware_flops = QWenPretrainedModel._get_hardware_flops

    # DONOT Add base_model_prefix !!!!

    def __init__(self, config):
        self.config = config

        self.recompute = self.config.recompute
        self.recompute_granularity = self.config.recompute_granularity
        self.pp_recompute_interval = self.config.pp_recompute_interval
        self.no_recompute_layers = config.no_recompute_layers if config.no_recompute_layers is not None else []
        if self.recompute_granularity == "full":
            assert len(self.no_recompute_layers) == 0, "for pp with full recompute, no_recompute_layers is not support"

        virtual_pp_degree = getattr(self.config, "virtual_pp_degree", 1)

        def get_hcg():
            return fleet.get_hybrid_communicate_group()

        hcg = get_hcg()
        tensor_parallel_degree = max(hcg.get_model_parallel_world_size(), 1)
        tensor_parallel_rank = max(hcg.get_model_parallel_rank(), 0)

        # TODO: fix tensor_parallel_degree rewrite in here
        config.tensor_parallel_degree = tensor_parallel_degree
        config.tensor_parallel_rank = tensor_parallel_rank

        self.add_sequential_layer(LayerDesc(QWenEmbeddingPipe, config=config), "qwen")
        for i in range(config.num_hidden_layers):
            self.add_sequential_layer(
                LayerDesc(
                    QWenBlockPipe,
                    config=config,
                    skip_recompute_ops=get_skip_recompute_ops(config, i),
                ),
                f"qwen.h.{i}",
            )
        self.add_sequential_layer(LayerDesc(QWenRMSNormPipe, config=config), "qwen.ln_f")
        self.add_sequential_layer(LayerDesc(QWenLMHead, config=config), "lm_head")

        recompute_interval = 0
        if self.recompute and self.recompute_granularity == "full":
            assert self.config.pp_recompute_interval <= config.num_hidden_layers // (
                virtual_pp_degree * get_hcg().topology().get_dim_size("pipe")
            ), "pp recompute interval should smaller than num layers of each pp chunk"
            recompute_interval = self.config.pp_recompute_interval

        seg_method = "layer:QWenBlock"
        if config.num_hidden_layers % get_hcg().topology().get_dim_size("pipe") != 0:
            seg_method = "uniform"

        PipelineLayer.__init__(
            self,
            layers=self.get_sequential_layers(),
            loss_fn=self.get_loss_fn(config),
            topology=get_hcg().topology(),
            seg_method=seg_method,
            recompute_interval=recompute_interval,
            recompute_ctx={
                "mp_group": get_hcg().get_model_parallel_group(),
                "offload": False,
                "partition": False,
            },
            num_virtual_pipeline_stages=virtual_pp_degree,
        )
        # You should call init here, since there is a  diamond inheritance problem
        self.apply(self._init_weights)
        # DON'T init PipelinePretrainedModel
        # PipelinePretrainedModel.__init__(self.super(), config=config)

    def get_loss_fn(self, config):
        if config.dpo_config is not None:
            return DPOCriterion(config, use_infohub=True)
        else:
            return QWenPretrainingCriterion(config)
