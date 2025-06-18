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

from ...utils.tools import get_env_device
from ..dpo_criterion import DPOCriterion
from ..model_utils import PipelinePretrainedModel
from ..refined_recompute import get_skip_recompute_ops
from ..refined_recompute import recompute as rr_recompute
from .modeling import (
    Qwen3Config,
    Qwen3DecoderLayer,
    Qwen3LMHead,
    Qwen3Model,
    Qwen3PretrainedModel,
    Qwen3PretrainingCriterion,
    Qwen3RMSNorm,
)

__all__ = [
    "Qwen3ForCausalLMPipe",
]


def parse_args(args):
    if isinstance(args, tuple):
        if len(args) == 4:
            hidden_states, attention_mask, attn_mask_startend_row_indices, position_ids = args
        elif len(args) == 3:
            hidden_states, attention_mask, attn_mask_startend_row_indices = args
            position_ids = None
        elif len(args) == 2:
            hidden_states, attention_mask = args
            attn_mask_startend_row_indices, position_ids = None, None
    else:
        hidden_states = args
        attention_mask, attn_mask_startend_row_indices, position_ids = None, None, None

    if position_ids is not None:
        position_ids.stop_gradient = True

    if attention_mask is not None:
        attention_mask.stop_gradient = True

    if attn_mask_startend_row_indices is not None:
        attn_mask_startend_row_indices.stop_gradient = True

    return hidden_states, attention_mask, attn_mask_startend_row_indices, position_ids


def return_args(hidden_states, attention_mask=None, attn_mask_startend_row_indices=None, position_ids=None):
    ret = (hidden_states,)

    if attention_mask is not None:
        ret += (attention_mask.clone(),)
    if attn_mask_startend_row_indices is not None:
        ret += (attn_mask_startend_row_indices.clone(),)
    if position_ids is not None:
        ret += (position_ids.clone(),)
    if len(ret) == 1:
        ret = ret[0]

    return ret


def get_attr(layer, name):
    if getattr(layer, name, None) is not None:
        return getattr(layer, name, None)
    else:
        return get_attr(layer._layer, name)


class Qwen3EmbeddingPipe(nn.Layer):
    """Extends QWenEmbeddings to forward attention_mask through the pipeline."""

    def __init__(self, config: Qwen3Config):
        super(Qwen3EmbeddingPipe, self).__init__()
        self.config = config
        self.sequence_parallel = config.sequence_parallel
        self.hidden_size = config.hidden_size
        if config.tensor_parallel_degree > 1 and config.vocab_size % config.tensor_parallel_degree == 0:
            self.embed_tokens = fleet.meta_parallel.VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                weight_attr=paddle.ParamAttr(initializer=nn.initializer.XavierNormal()),
            )
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

    @property
    def embedding_weight(self):
        return get_attr(self.embed_tokens, "weight")

    def forward(self, args):
        """_summary_

        Args:
            input (_type_): _description_

        Returns:
            _type_: _description_
        """
        input_ids, attention_mask, attn_mask_startend_row_indices, position_ids = parse_args(args)
        input_embeds = self.embed_tokens(input_ids)
        if self.config.sequence_parallel:
            from paddle.distributed.fleet.utils.sequence_parallel_utils import ScatterOp

            # [bs, seq_len, num_head * head_dim] -> [bs * seq_len, num_head * head_dim]
            bs, seq_len, hidden_size = input_embeds.shape
            input_embeds = paddle.reshape_(input_embeds, [bs * seq_len, hidden_size])
            # [seq_len * bs / n, num_head * head_dim] (n is mp parallelism)
            input_embeds = ScatterOp.apply(input_embeds)

        batch_size, seq_length = input_ids.shape

        if attention_mask is not None:
            assert (
                attn_mask_startend_row_indices is None
            ), "attention_mask and attn_mask_startend_row_indices can not be set at same time"

            attention_mask = Qwen3Model._prepare_decoder_attention_mask(
                attention_mask, (batch_size, seq_length), 0, input_embeds.dtype
            )
            attention_mask.stop_gradient = True
            if get_env_device() == "npu":
                attention_mask = attention_mask.astype("bool")
        elif get_env_device() == "npu":
            attention_mask = paddle.tril(paddle.ones((seq_length, seq_length), dtype="bool"))
            attention_mask.stop_gradient = True

        return return_args(input_embeds, attention_mask, attn_mask_startend_row_indices, position_ids)


class Qwen3DecoderLayerPipe(Qwen3DecoderLayer):
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
            recompute_fn = rr_recompute if any(self.skip_recompute_ops.values()) else recompute
            if attention_mask is not None or attn_mask_startend_row_indices is not None:
                hidden_states = recompute_fn(
                    super().forward,
                    hidden_states,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                    use_reentrant=False,
                )
            else:
                # for pretrain
                hidden_states = recompute_fn(
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


class Qwen3RMSNormPipe(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.norm = Qwen3RMSNorm(config)

    def forward(self, args):
        hidden_states, attention_mask, attn_mask_startend_row_indices, position_ids = parse_args(args)
        return self.norm(hidden_states)


class Qwen3LMHeadPipe(Qwen3LMHead):
    def __init__(self, config, transpose_y=False):
        super(Qwen3LMHeadPipe, self).__init__(config, transpose_y=transpose_y)

    @property
    def embedding_weight(self):
        return get_attr(self, "weight")


class Qwen3ForCausalLMPipe(PipelinePretrainedModel, PipelineLayer):
    """QWenForPretraining adapted for pipeline parallelism.

    The largest change is flattening the QWenModel class so we can express it as a
    sequence of layers including embedding, transformer layers, and output.
    """

    config_class = Qwen3Config

    _get_tensor_parallel_mappings = Qwen3PretrainedModel._get_tensor_parallel_mappings
    _init_weights = Qwen3PretrainedModel._init_weights
    _keys_to_ignore_on_load_unexpected = Qwen3PretrainedModel._keys_to_ignore_on_load_unexpected
    _get_model_flops = Qwen3PretrainedModel._get_model_flops
    _get_hardware_flops = Qwen3PretrainedModel._get_hardware_flops

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

    def __init__(self, config: Qwen3Config):
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
                    "model_shared_weight", Qwen3EmbeddingPipe, shared_weight_attr="embedding_weight", config=config
                ),
                "model",
            )
        else:
            self.add_sequential_layer(LayerDesc(Qwen3EmbeddingPipe, config=config), "model")

        for i in range(config.num_hidden_layers):
            self.add_sequential_layer(
                LayerDesc(
                    Qwen3DecoderLayerPipe,
                    config=config,
                    layerwise_recompute=i not in self.no_recompute_layers,
                    skip_recompute_ops=get_skip_recompute_ops(config, i),
                ),
                f"model.layers.{i}",
            )
        self.add_sequential_layer(LayerDesc(Qwen3RMSNormPipe, config=config), "model")

        if config.tie_word_embeddings:
            self.add_sequential_layer(
                SharedLayerDesc(
                    "model_shared_weight",
                    Qwen3LMHeadPipe,
                    shared_weight_attr="embedding_weight",
                    config=config,
                    **{"transpose_y": True},
                ),
                "lm_head",
            )
        else:
            self.add_sequential_layer(LayerDesc(Qwen3LMHeadPipe, config=config), "lm_head")

        recompute_interval = 0
        if self.enable_recompute and self.recompute_granularity == "full":
            assert self.config.pp_recompute_interval <= config.num_hidden_layers // (
                virtual_pp_degree * get_hcg().topology().get_dim_size("pipe")
            ), "pp recompute interval should smaller than num layers of each pp chunk"
            recompute_interval = self.config.pp_recompute_interval

        seg_method = "layer:Qwen3DecoderLayer"
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
            return Qwen3PretrainingCriterion(config)
