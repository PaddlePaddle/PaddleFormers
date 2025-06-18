# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
""" BERT model configuration"""
from __future__ import annotations

from typing import Dict

from ..configuration_utils import PretrainedConfig

__all__ = ["BERT_PRETRAINED_INIT_CONFIGURATION", "BertConfig", "BERT_PRETRAINED_RESOURCE_FILES_MAP"]

BERT_PRETRAINED_INIT_CONFIGURATION = {
    "bert-base-uncased": {
        "vocab_size": 30522,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "type_vocab_size": 2,
        "initializer_range": 0.02,
        "pad_token_id": 0,
    },
    "bert-large-uncased": {
        "vocab_size": 30522,
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "intermediate_size": 4096,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "type_vocab_size": 2,
        "initializer_range": 0.02,
        "pad_token_id": 0,
    },
    "bert-base-multilingual-uncased": {
        "vocab_size": 105879,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "type_vocab_size": 2,
        "initializer_range": 0.02,
        "pad_token_id": 0,
    },
    "bert-base-cased": {
        "vocab_size": 28996,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "type_vocab_size": 2,
        "initializer_range": 0.02,
        "pad_token_id": 0,
    },
    "bert-base-chinese": {
        "vocab_size": 21128,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "type_vocab_size": 2,
        "initializer_range": 0.02,
        "pad_token_id": 0,
    },
    "bert-base-multilingual-cased": {
        "vocab_size": 119547,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "type_vocab_size": 2,
        "initializer_range": 0.02,
        "pad_token_id": 0,
    },
    "bert-large-cased": {
        "vocab_size": 28996,
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "intermediate_size": 4096,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "type_vocab_size": 2,
        "initializer_range": 0.02,
        "pad_token_id": 0,
    },
    "bert-wwm-chinese": {
        "vocab_size": 21128,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "type_vocab_size": 2,
        "initializer_range": 0.02,
        "pad_token_id": 0,
    },
    "bert-wwm-ext-chinese": {
        "vocab_size": 21128,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "type_vocab_size": 2,
        "initializer_range": 0.02,
        "pad_token_id": 0,
    },
    "macbert-base-chinese": {
        "vocab_size": 21128,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "type_vocab_size": 2,
        "initializer_range": 0.02,
        "pad_token_id": 0,
    },
    "macbert-large-chinese": {
        "vocab_size": 21128,
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "intermediate_size": 4096,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "type_vocab_size": 2,
        "initializer_range": 0.02,
        "pad_token_id": 0,
    },
    "simbert-base-chinese": {
        "vocab_size": 13685,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "type_vocab_size": 2,
        "initializer_range": 0.02,
        "pad_token_id": 0,
    },
    "uer/chinese-roberta-base": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "max_position_embeddings": 512,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "type_vocab_size": 2,
        "vocab_size": 21128,
        "pad_token_id": 0,
    },
    "uer/chinese-roberta-medium": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 512,
        "initializer_range": 0.02,
        "intermediate_size": 2048,
        "max_position_embeddings": 512,
        "num_attention_heads": 8,
        "num_hidden_layers": 8,
        "type_vocab_size": 2,
        "vocab_size": 21128,
        "pad_token_id": 0,
    },
    "uer/chinese-roberta-6l-768h": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "max_position_embeddings": 512,
        "num_attention_heads": 12,
        "num_hidden_layers": 6,
        "type_vocab_size": 2,
        "vocab_size": 21128,
        "pad_token_id": 0,
    },
    "uer/chinese-roberta-small": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 512,
        "initializer_range": 0.02,
        "intermediate_size": 2048,
        "max_position_embeddings": 512,
        "num_attention_heads": 8,
        "num_hidden_layers": 4,
        "type_vocab_size": 2,
        "vocab_size": 21128,
        "pad_token_id": 0,
    },
    "uer/chinese-roberta-mini": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 256,
        "initializer_range": 0.02,
        "intermediate_size": 1024,
        "max_position_embeddings": 512,
        "num_attention_heads": 4,
        "num_hidden_layers": 4,
        "type_vocab_size": 2,
        "vocab_size": 21128,
        "pad_token_id": 0,
    },
    "uer/chinese-roberta-tiny": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 128,
        "initializer_range": 0.02,
        "intermediate_size": 512,
        "max_position_embeddings": 512,
        "num_attention_heads": 2,
        "num_hidden_layers": 2,
        "type_vocab_size": 2,
        "vocab_size": 21128,
        "pad_token_id": 0,
    },
}

BERT_PRETRAINED_RESOURCE_FILES_MAP = {
    "model_state": {
        "bert-base-uncased": "https://bj.bcebos.com/paddlenlp/models/transformers/bert-base-uncased.pdparams",
        "bert-large-uncased": "https://bj.bcebos.com/paddlenlp/models/transformers/bert-large-uncased.pdparams",
        "bert-base-multilingual-uncased": "http://bj.bcebos.com/paddlenlp/models/transformers/bert-base-multilingual-uncased.pdparams",
        "bert-base-cased": "http://bj.bcebos.com/paddlenlp/models/transformers/bert/bert-base-cased.pdparams",
        "bert-base-chinese": "http://bj.bcebos.com/paddlenlp/models/transformers/bert/bert-base-chinese.pdparams",
        "bert-base-multilingual-cased": "http://bj.bcebos.com/paddlenlp/models/transformers/bert/bert-base-multilingual-cased.pdparams",
        "bert-large-cased": "http://bj.bcebos.com/paddlenlp/models/transformers/bert/bert-large-cased.pdparams",
        "bert-wwm-chinese": "http://bj.bcebos.com/paddlenlp/models/transformers/bert/bert-wwm-chinese.pdparams",
        "bert-wwm-ext-chinese": "http://bj.bcebos.com/paddlenlp/models/transformers/bert/bert-wwm-ext-chinese.pdparams",
        "macbert-base-chinese": "https://bj.bcebos.com/paddlenlp/models/transformers/macbert/macbert-base-chinese.pdparams",
        "macbert-large-chinese": "https://bj.bcebos.com/paddlenlp/models/transformers/macbert/macbert-large-chinese.pdparams",
        "simbert-base-chinese": "https://bj.bcebos.com/paddlenlp/models/transformers/simbert/simbert-base-chinese-v1.pdparams",
        "uer/chinese-roberta-base": "https://bj.bcebos.com/paddlenlp/models/transformers/uer/chinese_roberta_base.pdparams",
        "uer/chinese-roberta-medium": "https://bj.bcebos.com/paddlenlp/models/transformers/uer/chinese_roberta_medium.pdparams",
        "uer/chinese-roberta-6l-768h": "https://bj.bcebos.com/paddlenlp/models/transformers/uer/chinese_roberta_6l_768h.pdparams",
        "uer/chinese-roberta-small": "https://bj.bcebos.com/paddlenlp/models/transformers/uer/chinese_roberta_small.pdparams",
        "uer/chinese-roberta-mini": "https://bj.bcebos.com/paddlenlp/models/transformers/uer/chinese_roberta_mini.pdparams",
        "uer/chinese-roberta-tiny": "https://bj.bcebos.com/paddlenlp/models/transformers/uer/chinese_roberta_tiny.pdparams",
    }
}


class BertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`BertModel`] or a [`TFBertModel`]. It is used to
    instantiate a BERT model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the BERT
    bert-base-uncased architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the BERT model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`BertModel`] or [`TFBertModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`BertModel`] or [`TFBertModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
            Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
            positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
            [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155).
            For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models
            with Better Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658).
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        classifier_dropout (`float`, *optional*):
            The dropout ratio for the classification head.

    Examples:

    ```python
    >>> from paddleformers.transformers import BertModel, BertConfig

    >>> # Initializing a BERT bert-base-uncased style configuration
    >>> configuration = BertConfig()

    >>> # Initializing a model from the bert-base-uncased style configuration
    >>> model = BertModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "bert"
    attribute_map: Dict[str, str] = {"dropout": "classifier_dropout", "num_classes": "num_labels"}
    pretrained_init_configuration = BERT_PRETRAINED_INIT_CONFIGURATION

    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 16,
        initializer_range: float = 0.02,
        pad_token_id: int = 0,
        pool_act: str = "tanh",
        fuse: bool = False,
        layer_norm_eps=1e-12,
        use_cache=False,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.pool_act = pool_act
        self.fuse = fuse

        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
