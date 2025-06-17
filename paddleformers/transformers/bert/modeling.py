# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import warnings
from typing import Optional, Tuple

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import Tensor
from paddle.nn import Layer

try:
    from paddle.incubate.nn import FusedTransformerEncoderLayer
except ImportError:
    FusedTransformerEncoderLayer = None
from dataclasses import dataclass

from ...utils.converter import StateDictNameMapping, init_name_mappings
from ...utils.env import CONFIG_NAME
from ..model_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
    ModelOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ..model_utils import PretrainedModel, register_base_model
from ..transposed_linear import TransposedLinear
from .configuration import (
    BERT_PRETRAINED_INIT_CONFIGURATION,
    BERT_PRETRAINED_RESOURCE_FILES_MAP,
    BertConfig,
)

__all__ = [
    "BertModel",
    "BertPretrainedModel",
    "BertForPretraining",
    "BertPretrainingCriterion",
    "BertPretrainingHeads",
    "BertForSequenceClassification",
    "BertForTokenClassification",
    "BertForQuestionAnswering",
    "BertForMultipleChoice",
    "BertForMaskedLM",
]


class BertEmbeddings(Layer):
    """
    Include embeddings from word, position and token_type embeddings
    """

    def __init__(self, config: BertConfig):
        super(BertEmbeddings, self).__init__()

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values_length: Optional[int] = None,
    ):

        if position_ids is None:
            ones = paddle.ones_like(input_ids, dtype="int64")
            seq_length = paddle.cumsum(ones, axis=-1)

            position_ids = seq_length - ones
            if past_key_values_length is not None:
                position_ids += past_key_values_length
            position_ids.stop_gradient = True
        if token_type_ids is None:
            token_type_ids = paddle.zeros_like(input_ids, dtype="int64")

        input_embedings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = input_embedings + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertPooler(Layer):
    """
    Pool the result of BertEncoder.
    """

    def __init__(self, config: BertConfig):
        """init the bert pooler with config & args/kwargs

        Args:
            config (BertConfig): BertConfig instance. Defaults to None.
        """
        super(BertPooler, self).__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.pool_act = config.pool_act

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        if self.pool_act == "tanh":
            pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPretrainedModel(PretrainedModel):
    """
    An abstract class for pretrained BERT models. It provides BERT related
    `model_config_file`, `resource_files_names`, `pretrained_resource_files_map`,
    `pretrained_init_configuration`, `base_model_prefix` for downloading and
    loading pretrained models.
    See :class:`~paddleformers.transformers.model_utils.PretrainedModel` for more details.
    """

    model_config_file = CONFIG_NAME
    config_class = BertConfig
    resource_files_names = {"model_state": "model_state.pdparams"}
    base_model_prefix = "bert"

    pretrained_init_configuration = BERT_PRETRAINED_INIT_CONFIGURATION
    pretrained_resource_files_map = BERT_PRETRAINED_RESOURCE_FILES_MAP

    @classmethod
    def _get_name_mappings(cls, config: BertConfig) -> list[StateDictNameMapping]:
        mappings: list[StateDictNameMapping] = []
        model_mappings = [
            "embeddings.word_embeddings.weight",
            "embeddings.position_embeddings.weight",
            "embeddings.token_type_embeddings.weight",
            ["embeddings.LayerNorm.weight", "embeddings.layer_norm.weight"],
            ["embeddings.LayerNorm.bias", "embeddings.layer_norm.bias"],
            ["pooler.dense.weight", None, "transpose"],
            "pooler.dense.bias",
            # for TokenClassification
        ]
        for layer_index in range(config.num_hidden_layers):
            layer_mappings = [
                [
                    f"encoder.layer.{layer_index}.attention.self.query.weight",
                    f"encoder.layers.{layer_index}.self_attn.q_proj.weight",
                    "transpose",
                ],
                [
                    f"encoder.layer.{layer_index}.attention.self.query.bias",
                    f"encoder.layers.{layer_index}.self_attn.q_proj.bias",
                ],
                [
                    f"encoder.layer.{layer_index}.attention.self.key.weight",
                    f"encoder.layers.{layer_index}.self_attn.k_proj.weight",
                    "transpose",
                ],
                [
                    f"encoder.layer.{layer_index}.attention.self.key.bias",
                    f"encoder.layers.{layer_index}.self_attn.k_proj.bias",
                ],
                [
                    f"encoder.layer.{layer_index}.attention.self.value.weight",
                    f"encoder.layers.{layer_index}.self_attn.v_proj.weight",
                    "transpose",
                ],
                [
                    f"encoder.layer.{layer_index}.attention.self.value.bias",
                    f"encoder.layers.{layer_index}.self_attn.v_proj.bias",
                ],
                [
                    f"encoder.layer.{layer_index}.attention.output.dense.weight",
                    f"encoder.layers.{layer_index}.self_attn.out_proj.weight",
                    "transpose",
                ],
                [
                    f"encoder.layer.{layer_index}.attention.output.dense.bias",
                    f"encoder.layers.{layer_index}.self_attn.out_proj.bias",
                ],
                [
                    f"encoder.layer.{layer_index}.intermediate.dense.weight",
                    f"encoder.layers.{layer_index}.linear1.weight",
                    "transpose",
                ],
                [f"encoder.layer.{layer_index}.intermediate.dense.bias", f"encoder.layers.{layer_index}.linear1.bias"],
                [
                    f"encoder.layer.{layer_index}.attention.output.LayerNorm.weight",
                    f"encoder.layers.{layer_index}.norm1.weight",
                ],
                [
                    f"encoder.layer.{layer_index}.attention.output.LayerNorm.bias",
                    f"encoder.layers.{layer_index}.norm1.bias",
                ],
                [
                    f"encoder.layer.{layer_index}.output.dense.weight",
                    f"encoder.layers.{layer_index}.linear2.weight",
                    "transpose",
                ],
                [f"encoder.layer.{layer_index}.output.dense.bias", f"encoder.layers.{layer_index}.linear2.bias"],
                [f"encoder.layer.{layer_index}.output.LayerNorm.weight", f"encoder.layers.{layer_index}.norm2.weight"],
                [f"encoder.layer.{layer_index}.output.LayerNorm.bias", f"encoder.layers.{layer_index}.norm2.bias"],
            ]
            model_mappings.extend(layer_mappings)

        init_name_mappings(model_mappings)

        # base-model prefix "BertModel"
        if "BertModel" not in config.architectures:
            for mapping in model_mappings:
                mapping[0] = "bert." + mapping[0]
                mapping[1] = "bert." + mapping[1]

        # downstream mappings
        if "BertForQuestionAnswering" in config.architectures:
            model_mappings.extend(
                [["qa_outputs.weight", "classifier.weight", "transpose"], ["qa_outputs.bias", "classifier.bias"]]
            )
        if (
            "BertForMultipleChoice" in config.architectures
            or "BertForSequenceClassification" in config.architectures
            or "BertForTokenClassification" in config.architectures
        ):
            model_mappings.extend([["classifier.weight", "classifier.weight", "transpose"]])

        mappings = [StateDictNameMapping(*mapping, index=index) for index, mapping in enumerate(model_mappings)]
        return mappings

    def _init_weights(self, layer):
        """Initialization hook"""
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=self.config.initializer_range,
                        shape=layer.weight.shape,
                    )
                )

        elif isinstance(layer, nn.LayerNorm):
            layer._epsilon = self.config.layer_norm_eps


@register_base_model
class BertModel(BertPretrainedModel):
    """
    The bare BERT Model transformer outputting raw hidden-states.

    This model inherits from :class:`~paddleformers.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.

    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/zh/api/paddle/nn/Layer_cn.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        config (:class:`BertConfig`):
            An instance of BertConfig used to construct BertModel.
    """

    def __init__(self, config: BertConfig):
        super(BertModel, self).__init__(config)

        self.pad_token_id = config.pad_token_id
        self.initializer_range = config.initializer_range
        self.embeddings = BertEmbeddings(config)
        if config.fuse and FusedTransformerEncoderLayer is None:
            warnings.warn(
                "FusedTransformerEncoderLayer is not supported by the running Paddle. "
                "The flag fuse_transformer will be ignored. Try Paddle >= 2.3.0"
            )
        self.fuse = config.fuse and FusedTransformerEncoderLayer is not None
        if self.fuse:
            self.encoder = nn.LayerList(
                [
                    FusedTransformerEncoderLayer(
                        config.hidden_size,
                        config.num_attention_heads,
                        config.intermediate_size,
                        dropout_rate=config.hidden_dropout_prob,
                        activation=config.hidden_act,
                        attn_dropout_rate=config.attention_probs_dropout_prob,
                        act_dropout_rate=0.0,
                    )
                    for _ in range(config.num_hidden_layers)
                ]
            )
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                config.hidden_size,
                config.num_attention_heads,
                config.intermediate_size,
                dropout=config.hidden_dropout_prob,
                activation=config.hidden_act,
                attn_dropout=config.attention_probs_dropout_prob,
                act_dropout=0,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, config.num_hidden_layers)
        self.pooler = BertPooler(config)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        The BertModel forward method, overrides the `__call__()` special method.

        Args:
            input_ids (Tensor):
                Indices of input sequence tokens in the vocabulary. They are
                numerical representations of tokens that build the input sequence.
                Its data type should be `int64` and it has a shape of [batch_size, sequence_length].
            token_type_ids (Tensor, optional):
                Segment token indices to indicate different portions of the inputs.
                Selected in the range ``[0, type_vocab_size - 1]``.
                If `type_vocab_size` is 2, which means the inputs have two portions.
                Indices can either be 0 or 1:

                - 0 corresponds to a *sentence A* token,
                - 1 corresponds to a *sentence B* token.

                Its data type should be `int64` and it has a shape of [batch_size, sequence_length].
                Defaults to `None`, which means we don't add segment embeddings.
            position_ids(Tensor, optional):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
                max_position_embeddings - 1]``.
                Shape as `(batch_size, num_tokens)` and dtype as int64. Defaults to `None`.
            attention_mask (Tensor, optional):
                Mask used in multi-head attention to avoid performing attention on to some unwanted positions,
                usually the paddings or the subsequent positions.
                Its data type can be int, float and bool.
                When the data type is bool, the `masked` tokens have `False` values and the others have `True` values.
                When the data type is int, the `masked` tokens have `0` values and the others have `1` values.
                When the data type is float, the `masked` tokens have `-INF` values and the others have `0` values.
                It is a tensor with shape broadcasted to `[batch_size, num_attention_heads, sequence_length, sequence_length]`.
                Defaults to `None`, which means nothing needed to be prevented attention to.
            past_key_values (tuple(tuple(Tensor)), optional):
                The length of tuple equals to the number of layers, and each inner
                tuple haves 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`)
                which contains precomputed key and value hidden states of the attention blocks.
                If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that
                don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
                `input_ids` of shape `(batch_size, sequence_length)`.
            use_cache (`bool`, optional):
                If set to `True`, `past_key_values` key value states are returned.
                Defaults to `None`.
            output_hidden_states (bool, optional):
                Whether to return the hidden states of all layers.
                Defaults to `None`.
            output_attentions (bool, optional):
                Whether to return the attentions tensors of all attention layers.
                Defaults to `None`.
            return_dict (bool, optional):
                Whether to return a :class:`~paddleformers.transformers.model_outputs.ModelOutput` object. If `False`, the output
                will be a tuple of tensors. Defaults to `None`.

        Returns:
            An instance of :class:`~paddleformers.transformers.model_outputs.BaseModelOutputWithPoolingAndCrossAttentions` if
            `return_dict=True`. Otherwise it returns a tuple of tensors corresponding
            to ordered and not None (depending on the input arguments) fields of
            :class:`~paddleformers.transformers.model_outputs.BaseModelOutputWithPoolingAndCrossAttentions`.

        Example:
            .. code-block::

                import paddle
                from paddleformers.transformers import BertModel, BertTokenizer

                tokenizer = BertTokenizer.from_pretrained('bert-wwm-chinese')
                model = BertModel.from_pretrained('bert-wwm-chinese')

                inputs = tokenizer("欢迎使用百度飞桨!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                output = model(**inputs)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        past_key_values_length = None
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
        if attention_mask is None:
            attention_mask = paddle.unsqueeze(
                (input_ids == self.pad_token_id).astype(self.pooler.dense.weight.dtype) * -1e4, axis=[1, 2]
            )
            if past_key_values is not None:
                batch_size = past_key_values[0][0].shape[0]
                past_mask = paddle.zeros([batch_size, 1, 1, past_key_values_length], dtype=attention_mask.dtype)
                attention_mask = paddle.concat([past_mask, attention_mask], axis=-1)
        else:
            if attention_mask.ndim == 2:
                # attention_mask [batch_size, sequence_length] -> [batch_size, 1, 1, sequence_length]
                attention_mask = attention_mask.unsqueeze(axis=[1, 2]).astype(paddle.get_default_dtype())
                attention_mask = (1.0 - attention_mask) * -1e4

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            past_key_values_length=past_key_values_length,
        )
        if self.fuse:
            assert not output_attentions, "Not support attentions output currently."
            assert past_key_values is None, "Not support past_key_values currently."
            hidden_states = embedding_output
            all_hidden_states = [] if output_hidden_states else None
            for layer in self.encoder:
                hidden_states = layer(hidden_states, attention_mask)
                if output_hidden_states:
                    all_hidden_states.append(hidden_states)
            pooled_output = self.pooler(hidden_states)

            if return_dict:
                return BaseModelOutputWithPoolingAndCrossAttentions(
                    last_hidden_state=hidden_states, pooler_output=pooled_output, hidden_states=all_hidden_states
                )
            else:
                return (
                    (hidden_states, pooled_output, all_hidden_states)
                    if output_hidden_states
                    else (hidden_states, pooled_output)
                )
        else:
            self.encoder._use_cache = use_cache  # To be consistent with HF
            encoder_outputs = self.encoder(
                embedding_output,
                src_mask=attention_mask,
                cache=past_key_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            if isinstance(encoder_outputs, type(embedding_output)):
                sequence_output = encoder_outputs
                pooled_output = self.pooler(sequence_output)
                return (sequence_output, pooled_output)
            else:
                sequence_output = encoder_outputs[0]
                pooled_output = self.pooler(sequence_output)
                if not return_dict:
                    return (sequence_output, pooled_output) + encoder_outputs[1:]
                return BaseModelOutputWithPoolingAndCrossAttentions(
                    last_hidden_state=sequence_output,
                    pooler_output=pooled_output,
                    past_key_values=encoder_outputs.past_key_values,
                    hidden_states=encoder_outputs.hidden_states,
                    attentions=encoder_outputs.attentions,
                )


class BertForQuestionAnswering(BertPretrainedModel):
    """
    Bert Model with a linear layer on top of the hidden-states output to compute `span_start_logits`
    and `span_end_logits`, designed for question-answering tasks like SQuAD.

    Args:
        config (:class:`BertConfig`):
            An instance of BertConfig used to construct BertForQuestionAnswering.
    """

    def __init__(self, config: BertConfig):
        super(BertForQuestionAnswering, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.classifier = nn.Linear(config.hidden_size, 2)

    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        start_positions: Optional[Tensor] = None,
        end_positions: Optional[Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        The BertForQuestionAnswering forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`BertModel`.
            token_type_ids (Tensor, optional):
                See :class:`BertModel`.
            position_ids(Tensor, optional):
                See :class:`BertModel`.
            attention_mask (Tensor, optional):
                See :class:`BertModel`.
            start_positions (Tensor of shape `(batch_size,)`, optional):
                Labels for position (index) of the start of the labelled span for computing the token classification loss.
                Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
                are not taken into account for computing the loss.
            end_positions (Tensor of shape `(batch_size,)`, optional):
                Labels for position (index) of the end of the labelled span for computing the token classification loss.
                Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
                are not taken into account for computing the loss.
            output_hidden_states (bool, optional):
                Whether to return the hidden states of all layers.
                Defaults to `None`.
            output_attentions (bool, optional):
                Whether to return the attentions tensors of all attention layers.
                Defaults to `None`.
            return_dict (bool, optional):
                Whether to return a :class:`~paddleformers.transformers.model_outputs.QuestionAnsweringModelOutput` object. If
                `False`, the output will be a tuple of tensors. Defaults to `None`.

        Returns:
            An instance of :class:`~paddleformers.transformers.model_outputs.QuestionAnsweringModelOutput` if `return_dict=True`.
            Otherwise it returns a tuple of tensors corresponding to ordered and
            not None (depending on the input arguments) fields of :class:`~paddleformers.transformers.model_outputs.QuestionAnsweringModelOutput`.

        Example:
            .. code-block::

                import paddle
                from paddleformers.transformers.bert.modeling import BertForQuestionAnswering
                from paddleformers.transformers.bert.tokenizer import BertTokenizer

                tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
                model = BertForQuestionAnswering.from_pretrained('bert-base-cased')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleFormers!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                outputs = model(**inputs)

                start_logits = outputs[0]
                end_logits = outputs[1]
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.classifier(sequence_output)
        logits = paddle.transpose(logits, perm=[2, 0, 1])
        start_logits, end_logits = paddle.unstack(x=logits, axis=0)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if start_positions.ndim > 1:
                start_positions = start_positions.squeeze(-1)
            if start_positions.ndim > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.shape[1]
            start_positions = start_positions.clip(0, ignored_index)
            end_positions = end_positions.clip(0, ignored_index)

            loss_fct = paddle.nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BertForSequenceClassification(BertPretrainedModel):
    """
    Bert Model with a linear layer on top of the output layer,
    designed for sequence classification/regression tasks like GLUE tasks.

    Args:
        config (:class:`BertConfig`):
            An instance of BertConfig used to construct BertForSequenceClassification.
    """

    def __init__(self, config: BertConfig):
        super(BertForSequenceClassification, self).__init__(config)

        self.bert = BertModel(config)
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        The BertForSequenceClassification forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`BertModel`.
            token_type_ids (Tensor, optional):
                See :class:`BertModel`.
            position_ids(Tensor, optional):
                See :class:`BertModel`.
            attention_mask (Tensor, optional):
                See :class:`BertModel`.
            labels (Tensor of shape `(batch_size,)`, optional):
                Labels for computing the sequence classification/regression loss.
                Indices should be in `[0, ..., num_labels - 1]`. If `num_labels == 1`
                a regression loss is computed (Mean-Square loss), If `num_labels > 1`
                a classification loss is computed (Cross-Entropy).
            output_hidden_states (bool, optional):
                Whether to return the hidden states of all layers.
                Defaults to `None`.
            output_attentions (bool, optional):
                Whether to return the attentions tensors of all attention layers.
                Defaults to `None`.
            return_dict (bool, optional):
                Whether to return a :class:`~paddleformers.transformers.model_outputs.SequenceClassifierOutput` object. If
                `False`, the output will be a tuple of tensors. Defaults to `None`.

        Returns:
            An instance of :class:`~paddleformers.transformers.model_outputs.SequenceClassifierOutput` if `return_dict=True`.
            Otherwise it returns a tuple of tensors corresponding to ordered and
            not None (depending on the input arguments) fields of :class:`~paddleformers.transformers.model_outputs.SequenceClassifierOutput`.

        Example:
            .. code-block::

                import paddle
                from paddleformers.transformers.bert.modeling import BertForSequenceClassification
                from paddleformers.transformers.bert.tokenizer import BertTokenizer

                tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
                model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=2)

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleFormers!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}

                logits = model(**inputs)
                print(logits.shape)
                # [1, 2]

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == paddle.int64 or labels.dtype == paddle.int32):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = paddle.nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = paddle.nn.CrossEntropyLoss()
                loss = loss_fct(logits.reshape((-1, self.num_labels)), labels.reshape((-1,)))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = paddle.nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else (output[0] if len(output) == 1 else output)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BertForTokenClassification(BertPretrainedModel):
    """
    Bert Model with a linear layer on top of the hidden-states output layer,
    designed for token classification tasks like NER tasks.

    Args:
        config (:class:`BertConfig`):
            An instance of BertConfig used to construct BertForTokenClassification.
    """

    def __init__(self, config: BertConfig):
        super().__init__(config)

        self.bert = BertModel(config)
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        The BertForTokenClassification forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`BertModel`.
            token_type_ids (Tensor, optional):
                See :class:`BertModel`.
            position_ids(Tensor, optional):
                See :class:`BertModel`.
            attention_mask (list, optional):
                See :class:`BertModel`.
            labels (Tensor of shape `(batch_size, sequence_length)`, optional):
                Labels for computing the token classification loss. Indices should be in `[0, ..., num_labels - 1]`.
            output_hidden_states (bool, optional):
                Whether to return the hidden states of all layers.
                Defaults to `None`.
            output_attentions (bool, optional):
                Whether to return the attentions tensors of all attention layers.
                Defaults to `None`.
            return_dict (bool, optional):
                Whether to return a :class:`~paddleformers.transformers.model_outputs.TokenClassifierOutput` object. If
                `False`, the output will be a tuple of tensors. Defaults to `None`.

        Returns:
            An instance of :class:`~paddleformers.transformers.model_outputs.TokenClassifierOutput` if `return_dict=True`.
            Otherwise it returns a tuple of tensors corresponding to ordered and
            not None (depending on the input arguments) fields of :class:`~paddleformers.transformers.model_outputs.TokenClassifierOutput`.

        Example:
            .. code-block::

                import paddle
                from paddleformers.transformers.bert.modeling import BertForTokenClassification
                from paddleformers.transformers.bert.tokenizer import BertTokenizer

                tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
                model = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=2)

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleFormers!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}

                logits = model(**inputs)
                print(logits.shape)
                # [1, 13, 2]

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = paddle.nn.CrossEntropyLoss()
            loss = loss_fct(logits.reshape((-1, self.num_labels)), labels.reshape((-1,)))
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else (output[0] if len(output) == 1 else output)

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BertLMPredictionHead(Layer):
    """
    Bert Model with a `language modeling` head on top for CLM fine-tuning.
    """

    def __init__(self, config: BertConfig):
        super(BertLMPredictionHead, self).__init__()

        self.transform = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = getattr(nn.functional, config.hidden_act)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.decoder = TransposedLinear(config.hidden_size, config.vocab_size)
        # link bias to load pretrained weights
        self.decoder_bias = self.decoder.bias

    def forward(self, hidden_states, masked_positions=None):
        if masked_positions is not None:
            hidden_states = paddle.reshape(hidden_states, [-1, hidden_states.shape[-1]])
            hidden_states = paddle.tensor.gather(hidden_states, masked_positions)
        # gather masked tokens might be more quick
        hidden_states = self.transform(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertPretrainingHeads(Layer):
    """
    Perform language modeling task and next sentence classification task.

    Args:
        config (:class:`BertConfig`):
            An instance of BertConfig used to construct BertForPretraining.

    """

    def __init__(self, config: BertConfig):
        super(BertPretrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output, masked_positions=None):
        """
        Args:
            sequence_output(Tensor):
                Sequence of hidden-states at the last layer of the model.
                It's data type should be float32 and its shape is [batch_size, sequence_length, hidden_size].
            pooled_output(Tensor):
                The output of first token (`[CLS]`) in sequence.
                We "pool" the model by simply taking the hidden state corresponding to the first token.
                Its data type should be float32 and its shape is [batch_size, hidden_size].
            masked_positions(Tensor, optional):
                A tensor indicates positions to be masked in the position embedding.
                Its data type should be int64 and its shape is [batch_size, mask_token_num].
                `mask_token_num` is the number of masked tokens. It should be no bigger than `sequence_length`.
                Defaults to `None`, which means we output hidden-states of all tokens in masked token prediction.

        Returns:
            tuple: Returns tuple (``prediction_scores``, ``seq_relationship_score``).

            With the fields:

            - `prediction_scores` (Tensor):
                The scores of masked token prediction. Its data type should be float32.
                If `masked_positions` is None, its shape is [batch_size, sequence_length, vocab_size].
                Otherwise, its shape is [batch_size, mask_token_num, vocab_size].

            - `seq_relationship_score` (Tensor):
                The scores of next sentence prediction.
                Its data type should be float32 and its shape is [batch_size, 2].

        """
        prediction_scores = self.predictions(sequence_output, masked_positions)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


@dataclass
class BertForPreTrainingOutput(ModelOutput):
    """
    Output type of [`BertForPreTraining`].

    Args:
        loss (*optional*, returned when `labels` is provided, `paddle.Tensor` of shape `(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction
            (classification) loss.
        prediction_logits (`paddle.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        seq_relationship_logits (`paddle.Tensor` of shape `(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
            before SoftMax).
        hidden_states (`tuple(paddle.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `paddle.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(paddle.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[paddle.Tensor] = None
    prediction_logits: paddle.Tensor = None
    seq_relationship_logits: paddle.Tensor = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None
    attentions: Optional[Tuple[paddle.Tensor]] = None


class BertForPretraining(BertPretrainedModel):
    """
    Bert Model with pretraining tasks on top.

    Args:
        config (:class:`BertConfig`):
            An instance of BertConfig used to construct BertForPretraining.

    """

    def __init__(self, config: BertConfig):
        super(BertForPretraining, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertPretrainingHeads(config)
        self.tie_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        masked_positions: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        next_sentence_label: Optional[Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""

        Args:
            input_ids (Tensor):
                See :class:`BertModel`.
            token_type_ids (Tensor, optional):
                See :class:`BertModel`.
            position_ids (Tensor, optional):
                See :class:`BertModel`.
            attention_mask (Tensor, optional):
                See :class:`BertModel`.
            masked_positions(Tensor, optional):
                See :class:`BertPretrainingHeads`.
            labels (Tensor of shape `(batch_size, sequence_length)`, optional):
                Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
                vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked),
                the loss is only computed for the tokens with labels in `[0, ..., vocab_size]`.
            next_sentence_label (Tensor of shape `(batch_size,)`, optional):
                Labels for computing the next sequence prediction (classification) loss. Input should be a sequence
                pair (see `input_ids` docstring) Indices should be in `[0, 1]`:

                - 0 indicates sequence B is a continuation of sequence A,
                - 1 indicates sequence B is a random sequence.
            output_hidden_states (bool, optional):
                Whether to return the hidden states of all layers.
                Defaults to `None`.
            output_attentions (bool, optional):
                Whether to return the attentions tensors of all attention layers.
                Defaults to `None`.
            return_dict (bool, optional):
                Whether to return a :class:`~paddleformers.transformers.bert.BertForPreTrainingOutput` object. If
                `False`, the output will be a tuple of tensors. Defaults to `None`.

        Returns:
            An instance of :class:`~paddleformers.transformers.bert.BertForPreTrainingOutput` if `return_dict=True`.
            Otherwise it returns a tuple of tensors corresponding to ordered and
            not None (depending on the input arguments) fields of :class:`~paddleformers.transformers.bert.BertForPreTrainingOutput`.

        """
        with paddle.static.amp.fp16_guard():
            outputs = self.bert(
                input_ids,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            sequence_output, pooled_output = outputs[:2]
            prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output, masked_positions)

            total_loss = None
            if labels is not None and next_sentence_label is not None:
                loss_fct = paddle.nn.CrossEntropyLoss()
                masked_lm_loss = loss_fct(
                    prediction_scores.reshape((-1, prediction_scores.shape[-1])), labels.reshape((-1,))
                )
                next_sentence_loss = loss_fct(
                    seq_relationship_score.reshape((-1, 2)), next_sentence_label.reshape((-1,))
                )
                total_loss = masked_lm_loss + next_sentence_loss
            if not return_dict:
                output = (prediction_scores, seq_relationship_score) + outputs[2:]
                return ((total_loss,) + output) if total_loss is not None else output

            return BertForPreTrainingOutput(
                loss=total_loss,
                prediction_logits=prediction_scores,
                seq_relationship_logits=seq_relationship_score,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )


class BertPretrainingCriterion(paddle.nn.Layer):
    """

    Args:
        vocab_size(int):
            Vocabulary size of `inputs_ids` in `BertModel`. Defines the number of different tokens that can
            be represented by the `inputs_ids` passed when calling `BertModel`.

    """

    def __init__(self, vocab_size):
        super(BertPretrainingCriterion, self).__init__()
        # CrossEntropyLoss is expensive since the inner reshape (copy)
        self.loss_fn = paddle.nn.loss.CrossEntropyLoss(ignore_index=-1)
        self.vocab_size = vocab_size

    def forward(
        self, prediction_scores, seq_relationship_score, masked_lm_labels, next_sentence_labels, masked_lm_scale
    ):
        """
        Args:
            prediction_scores(Tensor):
                The scores of masked token prediction. Its data type should be float32.
                If `masked_positions` is None, its shape is [batch_size, sequence_length, vocab_size].
                Otherwise, its shape is [batch_size, mask_token_num, vocab_size]
            seq_relationship_score(Tensor):
                The scores of next sentence prediction. Its data type should be float32 and
                its shape is [batch_size, 2]
            masked_lm_labels(Tensor):
                The labels of the masked language modeling, its dimensionality is equal to `prediction_scores`.
                Its data type should be int64. If `masked_positions` is None, its shape is [batch_size, sequence_length, 1].
                Otherwise, its shape is [batch_size, mask_token_num, 1]
            next_sentence_labels(Tensor):
                The labels of the next sentence prediction task, the dimensionality of `next_sentence_labels`
                is equal to `seq_relation_labels`. Its data type should be int64 and
                its shape is [batch_size, 1]
            masked_lm_scale(Tensor or int):
                The scale of masked tokens. Used for the normalization of masked language modeling loss.
                If it is a `Tensor`, its data type should be int64 and its shape is equal to `prediction_scores`.

        Returns:
            Tensor: The pretraining loss, equals to the sum of `masked_lm_loss` plus the mean of `next_sentence_loss`.
            Its data type should be float32 and its shape is [1].


        """
        with paddle.static.amp.fp16_guard():
            masked_lm_loss = F.cross_entropy(prediction_scores, masked_lm_labels, reduction="none", ignore_index=-1)
            masked_lm_loss = masked_lm_loss / masked_lm_scale
            next_sentence_loss = F.cross_entropy(seq_relationship_score, next_sentence_labels, reduction="none")
        return paddle.sum(masked_lm_loss) + paddle.mean(next_sentence_loss)


class BertForMultipleChoice(BertPretrainedModel):
    """
    Bert Model with a linear layer on top of the hidden-states output layer,
    designed for multiple choice tasks like RocStories/SWAG tasks.

    Args:
        config (:class:`BertConfig`):
            An instance of BertConfig used to construct BertForMultipleChoice.

    Examples:
        >>> model = BertForMultipleChoice(config, dropout=0.1)
        >>> # or
        >>> config.hidden_dropout_prob = 0.1
        >>> model = BertForMultipleChoice(config)
    """

    def __init__(self, config: BertConfig):
        super(BertForMultipleChoice, self).__init__(config)

        self.bert = BertModel(config)
        self.num_choices = config.num_choices
        self.dropout = nn.Dropout(
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.classifier = nn.Linear(config.hidden_size, 1)

    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        The BertForMultipleChoice forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`BertModel` and shape as [batch_size, num_choice, sequence_length].
            token_type_ids(Tensor, optional):
                See :class:`BertModel` and shape as [batch_size, num_choice, sequence_length].
            position_ids(Tensor, optional):
                See :class:`BertModel` and shape as [batch_size, num_choice, sequence_length].
            attention_mask (list, optional):
                See :class:`BertModel` and shape as [batch_size, num_choice, sequence_length].
            labels (Tensor of shape `(batch_size, )`, optional):
                Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
                num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
                `input_ids` above)
            output_hidden_states (bool, optional):
                Whether to return the hidden states of all layers.
                Defaults to `None`.
            output_attentions (bool, optional):
                Whether to return the attentions tensors of all attention layers.
                Defaults to `None`.
            return_dict (bool, optional):
                Whether to return a :class:`~paddleformers.transformers.model_outputs.MultipleChoiceModelOutput` object. If
                `False`, the output will be a tuple of tensors. Defaults to `None`.

        Returns:
            An instance of :class:`~paddleformers.transformers.model_outputs.MultipleChoiceModelOutput` if `return_dict=True`.
            Otherwise it returns a tuple of tensors corresponding to ordered and
            not None (depending on the input arguments) fields of :class:`~paddleformers.transformers.model_outputs.MultipleChoiceModelOutput`.

        Example:
            .. code-block::

                import paddle
                from paddleformers.transformers import BertForMultipleChoice, BertTokenizer
                from paddleformers.data import Pad, Dict

                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                model = BertForMultipleChoice.from_pretrained('bert-base-uncased', num_choices=2)

                data = [
                    {
                        "question": "how do you turn on an ipad screen?",
                        "answer1": "press the volume button.",
                        "answer2": "press the lock button.",
                        "label": 1,
                    },
                    {
                        "question": "how do you indent something?",
                        "answer1": "leave a space before starting the writing",
                        "answer2": "press the spacebar",
                        "label": 0,
                    },
                ]

                text = []
                text_pair = []
                for d in data:
                    text.append(d["question"])
                    text_pair.append(d["answer1"])
                    text.append(d["question"])
                    text_pair.append(d["answer2"])

                inputs = tokenizer(text, text_pair)
                batchify_fn = lambda samples, fn=Dict(
                    {
                        "input_ids": Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
                        "token_type_ids": Pad(
                            axis=0, pad_val=tokenizer.pad_token_type_id
                        ),  # token_type_ids
                    }
                ): fn(samples)
                inputs = batchify_fn(inputs)

                reshaped_logits = model(
                    input_ids=paddle.to_tensor(inputs[0], dtype="int64"),
                    token_type_ids=paddle.to_tensor(inputs[1], dtype="int64"),
                )
                print(reshaped_logits.shape)
                # [2, 2]

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # input_ids: [bs, num_choice, seq_l]
        input_ids = input_ids.reshape(shape=(-1, input_ids.shape[-1]))  # flat_input_ids: [bs*num_choice,seq_l]

        if position_ids is not None:
            position_ids = position_ids.reshape(shape=(-1, position_ids.shape[-1]))
        if token_type_ids is not None:
            token_type_ids = token_type_ids.reshape(shape=(-1, token_type_ids.shape[-1]))

        if attention_mask is not None:
            attention_mask = attention_mask.reshape(shape=(-1, attention_mask.shape[-1]))

        outputs = self.bert(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)  # logits: (bs*num_choice,1)
        reshaped_logits = logits.reshape(shape=(-1, self.num_choices))  # logits: (bs, num_choice)

        loss = None
        if labels is not None:
            loss_fct = paddle.nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else (output[0] if len(output) == 1 else output)

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BertOnlyMLMHead(nn.Layer):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.predictions = BertLMPredictionHead(config=config)

    def forward(self, sequence_output, masked_positions=None):
        prediction_scores = self.predictions(sequence_output, masked_positions)
        return prediction_scores


class BertForMaskedLM(BertPretrainedModel):
    """
    Bert Model with a `masked language modeling` head on top.

    Args:
        config (:class:`BertConfig`):
            An instance of BertConfig used to construct BertForMaskedLM.

    """

    def __init__(self, config: BertConfig):
        super(BertForMaskedLM, self).__init__(config)
        self.bert = BertModel(config)

        self.cls = BertOnlyMLMHead(config=config)
        self.tie_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        masked_positions: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""

        Args:
            input_ids (Tensor):
                See :class:`BertModel`.
            token_type_ids (Tensor, optional):
                See :class:`BertModel`.
            position_ids (Tensor, optional):
                See :class:`BertModel`.
            attention_mask (Tensor, optional):
                See :class:`BertModel`.
            labels (Tensor of shape `(batch_size, sequence_length)`, optional):
                Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
                vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
                loss is only computed for the tokens with labels in `[0, ..., vocab_size]`
            output_hidden_states (bool, optional):
                Whether to return the hidden states of all layers.
                Defaults to `None`.
            output_attentions (bool, optional):
                Whether to return the attentions tensors of all attention layers.
                Defaults to `None`.
            return_dict (bool, optional):
                Whether to return a :class:`~paddleformers.transformers.model_outputs.MaskedLMOutput` object. If
                `False`, the output will be a tuple of tensors. Defaults to `None`.

        Returns:
            An instance of :class:`~paddleformers.transformers.model_outputs.MaskedLMOutput` if `return_dict=True`.
            Otherwise it returns a tuple of tensors corresponding to ordered and
            not None (depending on the input arguments) fields of :class:`~paddleformers.transformers.model_outputs.MaskedLMOutput`.

        Example:
            .. code-block::

                import paddle
                from paddleformers.transformers import BertForMaskedLM, BertTokenizer

                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                model = BertForMaskedLM.from_pretrained('bert-base-uncased')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleFormers!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}

                logits = model(**inputs)
                print(logits.shape)
                # [1, 13, 30522]

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output, masked_positions=masked_positions)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = paddle.nn.CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(
                prediction_scores.reshape((-1, prediction_scores.shape[-1])), labels.reshape((-1,))
            )
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return (
                ((masked_lm_loss,) + output)
                if masked_lm_loss is not None
                else (output[0] if len(output) == 1 else output)
            )

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
