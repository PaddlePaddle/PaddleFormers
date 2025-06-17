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


from typing import OrderedDict

import paddle
import paddle.distributed.fleet as fleet
import paddle.nn as nn
from paddle.distributed.fleet.meta_parallel import (
    LayerDesc,
    PipelineLayer,
    SharedLayerDesc,
)
from paddle.distributed.fleet.recompute.recompute import recompute

from ..model_utils import PipelinePretrainedModel
from .modeling import (
    Qwen3MoeConfig,
    Qwen3MoeDecoderLayer,
    Qwen3MoeLMHead,
    Qwen3MoePretrainedModel,
    Qwen3MoePretrainingCriterion,
    Qwen3MoeRMSNorm,
)

__all__ = [
    "Qwen3MoeForCausalLMPipe",
]

from ..qwen2_moe.modeling_pp import (
    Qwen2MoeEmbeddingPipe,
    get_attr,
    parse_args,
    return_args,
)


class Qwen3MoeEmbeddingPipe(Qwen2MoeEmbeddingPipe):
    pass


class Qwen3MoeDecoderLayerPipe(Qwen3MoeDecoderLayer):
    def forward(self, args):
        hidden_states, attention_mask, attn_mask_startend_row_indices, position_ids = parse_args(args)

        has_gradient = not hidden_states.stop_gradient

        if attention_mask is not None and attention_mask.dtype == paddle.int32:
            attention_mask, attn_mask_startend_row_indices, position_ids = (
                None,
                attention_mask,
                attn_mask_startend_row_indices,
            )
        elif attention_mask is not None and attention_mask.dtype == paddle.int64:
            attention_mask, attn_mask_startend_row_indices, position_ids = None, None, attention_mask
        elif attn_mask_startend_row_indices is not None and attn_mask_startend_row_indices.dtype == paddle.int64:
            attn_mask_startend_row_indices, position_ids = None, attn_mask_startend_row_indices

        if self.enable_recompute and self.config.recompute_granularity == "full" and has_gradient:
            if attention_mask is not None or attn_mask_startend_row_indices is not None:
                hidden_states = recompute(
                    super().forward,
                    hidden_states,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                    use_reentrant=False,
                )
            else:
                # for pretrain
                hidden_states = recompute(
                    super().forward,
                    hidden_states,
                    position_ids=position_ids,
                    attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                    use_reentrant=self.config.recompute_use_reentrant,
                )
        else:
            hidden_states = super().forward(
                hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            )

        return return_args(hidden_states, attention_mask, attn_mask_startend_row_indices, position_ids)


class Qwen3MoeRMSNormPipe(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.norm = Qwen3MoeRMSNorm(config)

    def forward(self, args):
        hidden_states, attention_mask, attn_mask_startend_row_indices, position_ids = parse_args(args)
        return self.norm(hidden_states)


class Qwen3MoeLMHeadPipe(Qwen3MoeLMHead):
    def __init__(self, config, transpose_y=False):
        super(Qwen3MoeLMHeadPipe, self).__init__(config)

    @property
    def embedding_weight(self):
        return get_attr(self, "weight")


class Qwen3MoeForCausalLMPipe(PipelinePretrainedModel, PipelineLayer):
    """QWenForPretraining adapted for pipeline parallelism.

    The largest change is flattening the QWenModel class so we can express it as a
    sequence of layers including embedding, transformer layers, and output.
    """

    config_class = Qwen3MoeConfig

    _get_tensor_parallel_mappings = Qwen3MoePretrainedModel._get_tensor_parallel_mappings
    _init_weights = Qwen3MoePretrainedModel._init_weights
    _keys_to_ignore_on_load_unexpected = Qwen3MoePretrainedModel._keys_to_ignore_on_load_unexpected
    _tied_weights_keys = ["lm_head.weight"]

    # DONOT Add base_model_prefix !!!!

    @classmethod
    def _prepare_pipeline_inputs_func(cls, inputs):
        first_stage_keys = ["input_ids", "attention_mask", "attn_mask_startend_row_indices", "position_ids"]
        last_stage_keys = ["labels"]

        def get_expected_keys(inputs, keys):
            ret = tuple([inputs.pop(k) if k in inputs else None for k in keys])
            if len(ret) == 1:
                ret = ret[0]
            return ret

        if type(inputs) is dict or type(inputs) is OrderedDict:
            return [
                get_expected_keys(inputs, first_stage_keys),
                get_expected_keys(inputs, last_stage_keys),
            ]

        keys = list(inputs[0].keys())
        inputs_batch = {key: [data.pop(key) for data in inputs] for key in keys}
        return [
            get_expected_keys(inputs_batch, first_stage_keys),
            get_expected_keys(inputs_batch, last_stage_keys),
        ]

    def __init__(self, config: Qwen3MoeConfig):
        self.config = config

        # Note that we will actually perform a recompute only if both enable_recompute and layerwise_recompute are set to True
        # Enable_recompute defaults to False and is controlled by Trainer
        self.enable_recompute = False
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

        if config.tie_word_embeddings:
            self.add_sequential_layer(
                SharedLayerDesc(
                    "model_moe_shared_weight",
                    Qwen3MoeEmbeddingPipe,
                    shared_weight_attr="embedding_weight",
                    config=config,
                ),
                "model",
            )
        else:
            self.add_sequential_layer(LayerDesc(Qwen3MoeEmbeddingPipe, config=config), "model")

        for i in range(config.num_hidden_layers):
            self.add_sequential_layer(
                LayerDesc(
                    Qwen3MoeDecoderLayerPipe,
                    config=config,
                    layerwise_recompute=i not in self.no_recompute_layers,
                ),
                f"model.layers.{i}",
            )
        self.add_sequential_layer(LayerDesc(Qwen3MoeRMSNormPipe, config=config), "model")

        if config.tie_word_embeddings:
            self.add_sequential_layer(
                SharedLayerDesc(
                    "model_moe_shared_weight",
                    Qwen3MoeLMHeadPipe,
                    shared_weight_attr="embedding_weight",
                    config=config,
                    **{"transpose_y": True},
                ),
                "lm_head",
            )
        else:
            self.add_sequential_layer(LayerDesc(Qwen3MoeLMHeadPipe, config=config), "lm_head")

        recompute_interval = 0
        if self.enable_recompute and self.recompute_granularity == "full":
            assert self.config.pp_recompute_interval <= config.num_hidden_layers // (
                virtual_pp_degree * get_hcg().topology().get_dim_size("pipe")
            ), "pp recompute interval should smaller than num layers of each pp chunk"
            recompute_interval = self.config.pp_recompute_interval

        seg_method = "layer:Qwen3MoeDecoderLayer"
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
        return Qwen3MoePretrainingCriterion(config)
