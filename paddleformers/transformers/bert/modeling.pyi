from typing import Dict, Optional, Tuple, Union, overload

import paddle
import paddle.nn as nn
from _typeshed import Incomplete
from paddle import Tensor
from paddle.nn import Embedding, Layer, Linear

from ..model_outputs import ModelOutput
from ..model_utils import PretrainedModel
from .configuration import BertConfig

class BertEmbeddings(Layer):
    word_embeddings: Embedding
    position_embeddings: Embedding
    token_type_embeddings: Embedding
    layer_norm: Layer
    dropout: float
    def __init__(self, config: BertConfig) -> None: ...
    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Tensor | None = ...,
        position_ids: Tensor | None = ...,
        past_key_values_length: int = 0,
    ): ...

class BertPooler(Layer):
    dense: Linear
    activation: Layer
    pool_act: Layer
    def __init__(self, config: BertConfig) -> None: ...
    def forward(self, hidden_states): ...

class BertPretrainedModel(PretrainedModel):
    model_config_file: str
    config_class: Incomplete
    resource_files_names: Dict[str, str]
    base_model_prefix: str
    pretrained_init_configuration: Dict[str, dict]
    pretrained_resource_files_map: Dict[str, str]
    def init_weights(self, layer) -> None: ...

class BertModel(BertPretrainedModel):
    pad_token_id: int
    initializer_range: float
    embeddings: Embedding
    fuse: bool
    encoder: nn.TransformerDecoder
    pooler: BertPooler

    def __init__(self, config: BertConfig) -> None: ...
    def get_input_embeddings(self): ...
    def set_input_embeddings(self, value) -> None: ...
    def forward(
        self,
        input_ids,
        token_type_ids: Tensor | None = ...,
        position_ids: Tensor | None = ...,
        attention_mask: Tensor | None = ...,
        past_key_values: Tensor | None = ...,
        use_cache: Tensor | None = ...,
        output_hidden_states: bool = ...,
        output_attentions: bool = ...,
        return_dict: bool = ...,
    ): ...
    @staticmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        cache_dir: str | None = None,
        config: Optional[BertConfig] = None,
        *args,
        **kwargs
    ) -> BertModel: ...

class BertForQuestionAnswering(BertPretrainedModel):
    bert: BertModel
    dropout: nn.Dropout
    classifier: Linear
    def __init__(self, config: BertConfig): ...
    def forward(
        self,
        input_ids,
        token_type_ids: Tensor | None = ...,
        position_ids: Tensor | None = ...,
        attention_mask: Tensor | None = ...,
        start_positions: Tensor | None = ...,
        end_positions: Tensor | None = ...,
        output_hidden_states: bool = ...,
        output_attentions: bool = ...,
        return_dict: bool = ...,
    ): ...
    def __call__(
        self,
        input_ids,
        token_type_ids: Tensor | None = ...,
        position_ids: Tensor | None = ...,
        attention_mask: Tensor | None = ...,
        start_positions: Tensor | None = ...,
        end_positions: Tensor | None = ...,
        output_hidden_states: bool = ...,
        output_attentions: bool = ...,
        return_dict: bool = ...,
    ): ...
    @staticmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        cache_dir: str | None = None,
        classifier_dropout: float | None = None,
        config: Optional[BertConfig] = None,
        *args,
        **kwargs
    ) -> BertForQuestionAnswering: ...

class BertForSequenceClassification(BertPretrainedModel):
    bert: BertModel
    num_labels: int
    dropout: nn.Dropout
    classifier: Linear
    def __init__(self, config: BertConfig): ...
    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Tensor | None = ...,
        position_ids: Tensor | None = ...,
        attention_mask: Tensor | None = ...,
        labels: Tensor | None = ...,
        output_hidden_states: bool = ...,
        output_attentions: bool = ...,
        return_dict: bool = ...,
    ): ...
    def __call__(
        self,
        input_ids: Tensor,
        token_type_ids: Tensor | None = ...,
        position_ids: Tensor | None = ...,
        attention_mask: Tensor | None = ...,
        labels: Tensor | None = ...,
        output_hidden_states: bool = ...,
        output_attentions: bool = ...,
        return_dict: bool = ...,
    ): ...
    @staticmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        cache_dir: str | None = None,
        num_labels: int | None = 2,
        classifier_dropout: float | None = None,
        config: Optional[BertConfig] = None,
        *args,
        **kwargs
    ) -> BertForSequenceClassification: ...

class BertForTokenClassification(BertPretrainedModel):
    bert: BertModel
    num_labels: int
    dropout: nn.Dropout
    classifier: Linear
    def __init__(self, config: BertConfig): ...
    def forward(
        self,
        input_ids,
        token_type_ids: Tensor | None = ...,
        position_ids: Tensor | None = ...,
        attention_mask: Tensor | None = ...,
        labels: Tensor | None = ...,
        output_hidden_states: bool = ...,
        output_attentions: bool = ...,
        return_dict: bool = ...,
    ): ...
    def __call__(
        self,
        input_ids,
        token_type_ids: Tensor | None = ...,
        position_ids: Tensor | None = ...,
        attention_mask: Tensor | None = ...,
        labels: Tensor | None = ...,
        output_hidden_states: bool = ...,
        output_attentions: bool = ...,
        return_dict: bool = ...,
    ): ...
    @staticmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        cache_dir: str | None = None,
        num_labels: int | None = 2,
        classifier_dropout: float | None = None,
        config: Optional[BertConfig] = None,
        *args,
        **kwargs
    ) -> BertForTokenClassification: ...

class BertLMPredictionHead(Layer):
    transform: Incomplete
    activation: Incomplete
    layer_norm: nn.LayerNorm
    decoder_weight: paddle.ParamAttr
    decoder_bias: paddle.ParamAttr
    def __init__(self, config: BertConfig, embedding_weights: Tensor | None = ...) -> None: ...
    def forward(self, hidden_states, masked_positions: Tensor | None = ...): ...

class BertPretrainingHeads(Layer):
    predictions: Incomplete
    seq_relationship: Incomplete
    def __init__(self, config: BertConfig, embedding_weights: Tensor | None = ...) -> None: ...
    def forward(self, sequence_output, pooled_output, masked_positions: Tensor | None = ...): ...

class BertForPreTrainingOutput(ModelOutput):
    loss: Optional[paddle.Tensor]
    prediction_logits: paddle.Tensor
    seq_relationship_logits: paddle.Tensor
    hidden_states: Optional[Tuple[paddle.Tensor]]
    attentions: Optional[Tuple[paddle.Tensor]]
    def __init__(
        self,
        loss: Tensor | None,
        prediction_logits: Tensor | None,
        seq_relationship_logits: Tensor | None,
        hidden_states: Tensor | None,
        attentions: Tensor | None,
    ) -> None: ...

class BertForPretraining(BertPretrainedModel):
    bert: BertModel
    cls: Incomplete
    def __init__(self, config: BertConfig) -> None: ...
    def forward(
        self,
        input_ids,
        token_type_ids: Tensor | None = ...,
        position_ids: Tensor | None = ...,
        attention_mask: Tensor | None = ...,
        masked_positions: Tensor | None = ...,
        labels: Tensor | None = ...,
        next_sentence_label: Tensor | None = ...,
        output_hidden_states: bool = ...,
        output_attentions: bool = ...,
        return_dict: bool = ...,
    ): ...
    def __call__(
        self,
        input_ids,
        token_type_ids: Tensor | None = ...,
        position_ids: Tensor | None = ...,
        attention_mask: Tensor | None = ...,
        masked_positions: Tensor | None = ...,
        labels: Tensor | None = ...,
        next_sentence_label: Tensor | None = ...,
        output_hidden_states: bool = ...,
        output_attentions: bool = ...,
        return_dict: bool = ...,
    ): ...
    @staticmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        cache_dir: str | None = None,
        config: Optional[BertConfig] = None,
        *args,
        **kwargs
    ) -> BertForQuestionAnswering: ...

class BertPretrainingCriterion(paddle.nn.Layer):
    loss_fn: nn.Layer
    vocab_size: int
    def __init__(self, vocab_size) -> None: ...
    def forward(
        self, prediction_scores, seq_relationship_score, masked_lm_labels, next_sentence_labels, masked_lm_scale
    ): ...
    def __call__(
        self, prediction_scores, seq_relationship_score, masked_lm_labels, next_sentence_labels, masked_lm_scale
    ): ...

class BertForMultipleChoice(BertPretrainedModel):
    bert: BertModel
    num_choices: int
    dropout: nn.Dropout
    classifier: Linear
    @overload
    def __init__(self, config: BertConfig) -> None: ...
    def forward(
        self,
        input_ids,
        token_type_ids: Tensor | None = ...,
        position_ids: Tensor | None = ...,
        attention_mask: Tensor | None = ...,
        labels: Tensor | None = ...,
        output_hidden_states: bool = ...,
        output_attentions: bool = ...,
        return_dict: bool = ...,
    ): ...
    @staticmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        cache_dir: str | None = None,
        num_choices: int | None = 2,
        classifier_dropout: float | None = None,
        config: Optional[BertConfig] = None,
        *args,
        **kwargs
    ) -> BertForMultipleChoice: ...

class BertOnlyMLMHead(nn.Layer):
    predictions: BertLMPredictionHead
    def __init__(self, config: BertConfig, embedding_weights: Tensor | None = ...) -> None: ...
    def forward(self, sequence_output, masked_positions: Tensor | None = ...): ...

class BertForMaskedLM(BertPretrainedModel):
    bert: BertModel
    cls: BertOnlyMLMHead
    def __init__(self, config: BertConfig) -> None: ...
    def forward(
        self,
        input_ids,
        token_type_ids: Tensor | None = ...,
        position_ids: Tensor | None = ...,
        attention_mask: Tensor | None = ...,
        labels: Tensor | None = ...,
        output_hidden_states: bool = ...,
        output_attentions: bool = ...,
        return_dict: bool = ...,
    ): ...
    def __call__(
        self,
        input_ids,
        token_type_ids: Tensor | None = ...,
        position_ids: Tensor | None = ...,
        attention_mask: Tensor | None = ...,
        labels: Tensor | None = ...,
        output_hidden_states: bool = ...,
        output_attentions: bool = ...,
        return_dict: bool = ...,
    ): ...
    @staticmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        cache_dir: str | None = None,
        config: Optional[BertConfig] = None,
        *args,
        **kwargs
    ) -> BertForMaskedLM: ...
