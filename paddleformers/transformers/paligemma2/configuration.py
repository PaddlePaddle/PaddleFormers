# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""PaliGemma2 model configuration"""

from typing import Optional, Union

from paddleformers.transformers.configuration_utils import PretrainedConfig


class SiglipVisionConfig(PretrainedConfig):
    """Configuration for the SigLIP vision encoder used in PaliGemma2."""

    model_type = "siglip_vision_model"

    def __init__(
        self,
        hidden_size: int = 1152,
        image_size: int = 448,
        intermediate_size: int = 4304,
        num_hidden_layers: int = 27,
        num_attention_heads: int = 16,
        num_channels: int = 3,
        patch_size: int = 14,
        projection_dim: int = 3584,
        num_image_tokens: int = 1024,
        num_positions: int = 1024,
        layer_norm_eps: float = 1e-6,
        vision_use_head: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.image_size = image_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.projection_dim = projection_dim
        self.num_image_tokens = num_image_tokens
        self.num_positions = num_positions
        self.layer_norm_eps = layer_norm_eps
        self.vision_use_head = vision_use_head


class Gemma2TextConfig(PretrainedConfig):
    """Configuration for the Gemma2 text decoder used in PaliGemma2."""

    model_type = "gemma2"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 257216,
        hidden_size: int = 3584,
        intermediate_size: int = 14336,
        num_hidden_layers: int = 42,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        head_dim: Optional[int] = None,
        hidden_activation: str = "gelu_pytorch_tanh",
        max_position_embeddings: int = 8192,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: int = 0,
        bos_token_id: int = 2,
        eos_token_id: int = 1,
        tie_word_embeddings: bool = True,
        rope_theta: float = 10000.0,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        query_pre_attn_scalar: float = 256.0,
        sliding_window: int = 4096,
        final_logit_softcapping: float = 30.0,
        attn_logit_softcapping: float = 50.0,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = 256 if head_dim is None else head_dim
        self.hidden_activation = hidden_activation
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.query_pre_attn_scalar = query_pre_attn_scalar
        self.sliding_window = sliding_window
        self.final_logit_softcapping = final_logit_softcapping
        self.attn_logit_softcapping = attn_logit_softcapping


class PaliGemma2Config(PretrainedConfig):
    """Configuration for the PaliGemma2 multimodal model.

    PaliGemma2 combines a SigLIP vision encoder with a Gemma2 text decoder.

    Example:
        ```python
        >>> from paddleformers.transformers.paligemma2 import PaliGemma2Config
        >>> config = PaliGemma2Config()
        ```
    """

    model_type = "paligemma2"
    attribute_map = {"image_token_id": "image_token_index"}
    sub_configs = {"text_config": Gemma2TextConfig, "vision_config": SiglipVisionConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vision_config: Optional[Union[SiglipVisionConfig, dict]] = None,
        text_config: Optional[Union[Gemma2TextConfig, dict]] = None,
        image_token_index: int = 257152,
        projection_dim: int = 3584,
        hidden_size: int = 3584,
        vocab_size: int = 257152,
        tie_word_embeddings: bool = True,
        **kwargs,
    ):
        if vision_config is None:
            vision_config = SiglipVisionConfig()
        elif isinstance(vision_config, dict):
            vision_config = SiglipVisionConfig(**vision_config)

        if text_config is None:
            text_config = Gemma2TextConfig()
        elif isinstance(text_config, dict):
            text_config = Gemma2TextConfig(**text_config)

        self.vision_config = vision_config
        self.text_config = text_config
        self.image_token_index = image_token_index
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.tie_word_embeddings = tie_word_embeddings

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


__all__ = ["PaliGemma2Config", "SiglipVisionConfig", "Gemma2TextConfig"]
