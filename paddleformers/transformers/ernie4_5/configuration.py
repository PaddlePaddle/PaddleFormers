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

""" Ernie4_5 model configuration."""
from ..configuration_utils import PretrainedConfig


class Ernie4_5Config(PretrainedConfig):
    """
    Configuration class for Ernie4_5 model.

    This class stores the configuration of an Ernie4_5 model, defining the model architecture.
    It inherits from PretrainedConfig and can be used to control model outputs.
    """

    model_type = "ernie4_5"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=768,
        intermediate_size=11008,
        max_position_embeddings=32768,
        num_hidden_layers=2,
        num_attention_heads=2,
        head_dim=128,
        scale_qk_coeff=1.0,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=False,
        use_flash_attention=False,
        recompute=False,
        recompute_granularity="core_attn",
        recompute_use_reentrant=False,
        tie_word_embeddings=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        use_bias=False,
        rope_theta=10000,
        fuse_rope=False,
        fuse_softmax_mask=False,
        fuse_linear=False,
        max_sequence_length=None,
        ignored_index=-100,
        attention_dropout_prob=0.0,
        hidden_act="silu",
        hidden_dropout_prob=0.0,
        num_key_value_heads=None,
        micro_batch_size=-1,
        pp_seg_method="layer:Ernie4_5DecoderLayer|EmptyLayer",
        dpo_config=None,
        kto_config=None,
        **kwargs,
    ):
        """
        Initialize Ernie4_5 model configuration with default or specified parameters.

        Args:
            vocab_size (int): Size of the vocabulary (number of unique tokens)
            hidden_size (int): Dimensionality of the encoder layers and the pooler layer
            intermediate_size (int): Dimensionality of the "intermediate" (feed-forward) layer
            max_position_embeddings (int): Maximum sequence length the model can handle
            num_hidden_layers (int): Number of hidden layers in the Transformer encoder
            num_attention_heads (int): Number of attention heads for each attention layer
            rms_norm_eps (float): The epsilon used by the RMS normalization layers
            use_cache (bool): Whether to use caching for faster generation (decoding)
            use_flash_attention (bool): Whether to use FlashAttention for optimized attention computation
            recompute (bool): Whether to use gradient checkpointing to save memory
            recompute_granularity (str): Granularity of recomputation ("core_attn", "full", etc.)
            recompute_use_reentrant (bool): Whether to use reentrant checkpointing
            tie_word_embeddings (bool):  Whether the input and output word embeddings should be tied
            Whether the model's input and output word embeddings should be tied. Note that this is only relevant if the
            model has a output word embedding layer.
            pad_token_id (int): Token ID used for padding sequences
            bos_token_id (int): Token ID used for beginning-of-sequence
            eos_token_id (int): Token ID used for end-of-sequence
            use_bias (bool): Whether to use bias terms in linear layers
            rope_theta (float): The base period of the RoPE embeddings
            fuse_rope (bool): Whether to fuse RoPE operations
            fuse_linear (bool): Whether to fuse linear operations
            fuse_up_gate (bool): Whether to fuse up_proj and gate_proj to a single linear layer
            max_sequence_length (int): Maximum sequence length for positional embeddings
            ignored_index (int): Target value that is ignored during loss computation
            attention_dropout_prob (float): Dropout probability for attention weights
            hidden_act (str): Activation function for MLP layers
            hidden_dropout_prob (float): Dropout probability for hidden layers
            num_key_value_heads (int): Number of key/value heads (for Grouped Query Attention)
            micro_batch_size (int): Size of micro batches (-1 for automatic)
            pp_seg_method (str): Method for pipeline parallel segmentation
            dpo_config (DPOConfig | None): DPO training configuration
            kto_config (KTOConfig | None): KTO training configuration
            **kwargs: Additional keyword arguments passed to parent class
        """
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.scale_qk_coeff = scale_qk_coeff
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.recompute = recompute
        self.recompute_granularity = recompute_granularity
        self.use_flash_attention = use_flash_attention
        self.recompute_use_reentrant = recompute_use_reentrant
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.micro_batch_size = micro_batch_size

        self.max_sequence_length = max_sequence_length
        self.use_bias = use_bias
        self.rope_theta = rope_theta
        self.tie_word_embeddings = tie_word_embeddings
        self.fuse_rope = fuse_rope
        self.fuse_softmax_mask = fuse_softmax_mask
        self.fuse_linear = fuse_linear
        self.ignored_index = ignored_index
        self.attention_dropout_prob = attention_dropout_prob
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.num_key_value_heads = num_key_value_heads
        self.pp_seg_method = pp_seg_method
        self.dpo_config = dpo_config
        self.kto_config = kto_config
        self.register_unsavable_keys(
            [
                "attention_dropout_prob",
                "hidden_dropout_prob",
                "ignored_index",
                "scale_qk_coeff",
                "recompute",
                "recompute_use_reentrant",
                "recompute_granularity",
                "pp_seg_method",
                "micro_batch_size",
                "fuse_softmax_mask",
                "max_sequence_length",
                "dpo_config",
                "kto_config",
            ]
        )


__all__ = ["Ernie4_5Config"]
