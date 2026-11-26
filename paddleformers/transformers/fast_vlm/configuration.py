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

"""Native Transformers FastVLM composite configuration."""

from ..configuration_utils import PretrainedConfig
from ..qwen2.configuration import Qwen2Config


class FastVlmVisionConfig(PretrainedConfig):
    """Configuration of the FastViT tower wrapped by timm in Transformers."""

    model_type = "timm_wrapper"

    def __init__(self, architecture="fastvit_mci3", hidden_size=3072, model_args=None, **kwargs):
        super().__init__(**kwargs)
        if architecture != "fastvit_mci3":
            raise ValueError("FastVLM currently supports the fastvit_mci3 vision architecture")
        self.architecture = architecture
        self.hidden_size = hidden_size
        self.model_args = {"inference_mode": True} if model_args is None else dict(model_args)
        if not self.model_args.get("inference_mode", False):
            raise ValueError("FastVLM requires the reparameterized FastViT checkpoint (inference_mode=True)")


class FastVlmConfig(PretrainedConfig):
    model_type = "fast_vlm"
    sub_configs = {"text_config": Qwen2Config, "vision_config": FastVlmVisionConfig}
    attribute_map = {"image_token_id": "image_token_index"}

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        image_token_id=151646,
        image_seq_length=256,
        projector_hidden_act="gelu",
        vision_feature_select_strategy="full",
        vision_feature_layer=-1,
        multimodal_projector_bias=True,
        tie_word_embeddings=False,
        **kwargs,
    ):
        if vision_feature_select_strategy != "full" or vision_feature_layer != -1:
            raise ValueError("FastVLM supports only vision_feature_select_strategy='full' and vision_feature_layer=-1")
        if text_config is None:
            text_config = Qwen2Config(
                hidden_size=3584,
                vocab_size=152128,
                intermediate_size=18944,
                num_attention_heads=28,
                num_key_value_heads=4,
                num_hidden_layers=28,
            )
        elif isinstance(text_config, dict):
            if text_config.get("model_type", "qwen2") != "qwen2":
                raise ValueError("FastVLM currently supports the Qwen2 text backbone")
            text_config = Qwen2Config(**text_config)
        if vision_config is None:
            vision_config = FastVlmVisionConfig()
        elif isinstance(vision_config, dict):
            vision_config = FastVlmVisionConfig(**vision_config)
        self.text_config = text_config
        self.vision_config = vision_config
        self.image_token_index = kwargs.pop("image_token_index", image_token_id)
        self.image_seq_length = image_seq_length
        self.projector_hidden_act = projector_hidden_act
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer
        self.multimodal_projector_bias = multimodal_projector_bias
        tie_word_embeddings = kwargs.pop("tie_embedding", tie_word_embeddings) or text_config.tie_word_embeddings
        text_config.tie_word_embeddings = tie_word_embeddings
        if kwargs.get("architectures") is None:
            kwargs["architectures"] = ["FastVlmForConditionalGeneration"]
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


__all__ = ["FastVlmConfig", "FastVlmVisionConfig"]
