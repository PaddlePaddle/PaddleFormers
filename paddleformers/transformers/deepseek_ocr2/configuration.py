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

""" DeepSeekOCR2 model configuration"""
from ...utils.log import logger
from ..configuration_utils import PretrainedConfig
from ..deepseek_v3.configuration import DeepseekV3Config

__all__ = ["DeepseekOCR2Config"]


class DeepseekOCR2VisionConfig(PretrainedConfig):
    model_type = "deepencoderv2"
    base_config_key = "vision_config"

    def __init__(
        self,
        # SAM config
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        prompt_embed_dim=256,
        image_size=1024,
        vit_patch_size=16,
        mlp_ratio=4,
        window_size=14,
        layer_norm_eps=1e-6,
        # Qwen2 config
        decoder_layer=24,
        hidden_dimension=896,
        num_attention_heads=14,
        num_key_value_heads=2,
        intermediate_size=4864,
        max_query=400,
        **kwargs
    ):
        super().__init__(**kwargs)
        # SAM config
        self.encoder_embed_dim = encoder_embed_dim
        self.encoder_depth = encoder_depth
        self.encoder_num_heads = encoder_num_heads
        self.encoder_global_attn_indexes = encoder_global_attn_indexes
        self.prompt_embed_dim = prompt_embed_dim
        self.image_size = image_size
        self.vit_patch_size = vit_patch_size
        if not isinstance(mlp_ratio, int) or mlp_ratio != int(mlp_ratio):
            import math

            logger.warning(
                f"mlp_ratio should be an integer, but got {mlp_ratio} (type={type(mlp_ratio).__name__}). "
                f"Ceiling to {math.ceil(mlp_ratio)}."
            )
            mlp_ratio = math.ceil(mlp_ratio)
        self.mlp_ratio = int(mlp_ratio)
        self.window_size = window_size
        self.layer_norm_eps = layer_norm_eps

        # Qwen2 config
        self.decoder_layer = decoder_layer
        self.hidden_dimension = hidden_dimension
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.max_query = max_query

    def __setattr__(self, key, value):
        if key == "_attn_implementation" and value not in ["sdpa", "eager"]:
            logger.warning(
                f"Deepencoderv2 with attention mask needs 'sdpa' or 'eager' as {key}, but got {value}. Fallback to 'sdpa'."
            )

            super().__setattr__(key, "sdpa")

        else:
            super().__setattr__(key, value)


class DeepseekOCR2Config(DeepseekV3Config):
    model_type = "deepseek_ocr2"
    sub_configs = {"vision_config": DeepseekOCR2VisionConfig}

    def __init__(
        self,
        aux_loss_alpha=0.001,
        use_mla=True,
        vision_config=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.aux_loss_alpha = aux_loss_alpha
        self.use_mla = use_mla
        # DeepseekOCR2 requires rms_norm_eps to be strictly float
        self.rms_norm_eps = float(self.rms_norm_eps)

        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        if self.model_type != "deepseek_ocr2":
            logger.warning(
                f"Receive model type '{self.model_type}' for DeepseekOCR2Config. Change it to 'deepseek_ocr2'"
            )
            self.model_type = "deepseek_ocr2"


__all__ = ["DeepseekOCR2Config"]
