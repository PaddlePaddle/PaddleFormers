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

from ..configuration_utils import PretrainedConfig

__all__ = ["Florence2Config", "Florence2LanguageConfig", "Florence2VisionConfig"]


class Florence2VisionConfig(PretrainedConfig):
    model_type = "davit"
    base_config_key = "vision_config"

    def __init__(
        self,
        drop_path_rate=0.1,
        patch_size=(7, 3, 3, 3),
        patch_stride=(4, 2, 2, 2),
        patch_padding=(3, 1, 1, 1),
        patch_prenorm=(False, True, True, True),
        enable_checkpoint=False,
        dim_embed=(128, 256, 512, 1024),
        num_heads=(4, 8, 16, 32),
        num_groups=(4, 8, 16, 32),
        depths=(1, 1, 9, 1),
        window_size=12,
        projection_dim=768,
        visual_temporal_embedding=None,
        image_pos_embed=None,
        image_feature_source=("spatial_avg_pool", "temporal_avg_pool"),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.drop_path_rate = drop_path_rate
        self.patch_size = list(patch_size)
        self.patch_stride = list(patch_stride)
        self.patch_padding = list(patch_padding)
        self.patch_prenorm = list(patch_prenorm)
        self.enable_checkpoint = enable_checkpoint
        self.dim_embed = list(dim_embed)
        self.num_heads = list(num_heads)
        self.num_groups = list(num_groups)
        self.depths = list(depths)
        self.window_size = window_size
        self.projection_dim = projection_dim
        self.visual_temporal_embedding = visual_temporal_embedding or {
            "type": "COSINE",
            "max_temporal_embeddings": 100,
        }
        self.image_pos_embed = image_pos_embed or {"type": "learned_abs_2d", "max_pos_embeddings": 50}
        self.image_feature_source = list(image_feature_source)


class Florence2LanguageConfig(PretrainedConfig):
    model_type = "florence2_language"
    base_config_key = "text_config"
    attribute_map = {"num_attention_heads": "encoder_attention_heads", "hidden_size": "d_model"}

    def __init__(
        self,
        vocab_size=51289,
        max_position_embeddings=1024,
        encoder_layers=6,
        encoder_ffn_dim=3072,
        encoder_attention_heads=12,
        decoder_layers=6,
        decoder_ffn_dim=3072,
        decoder_attention_heads=12,
        encoder_layerdrop=0.0,
        decoder_layerdrop=0.0,
        activation_function="gelu",
        d_model=768,
        dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.1,
        init_std=0.02,
        classifier_dropout=0.0,
        scale_embedding=False,
        use_cache=True,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        decoder_start_token_id=2,
        forced_eos_token_id=2,
        is_encoder_decoder=True,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            decoder_start_token_id=decoder_start_token_id,
            forced_eos_token_id=forced_eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.encoder_layers = encoder_layers
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_layers = decoder_layers
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_attention_heads = decoder_attention_heads
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.activation_function = activation_function
        self.d_model = d_model
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.init_std = init_std
        self.classifier_dropout = classifier_dropout
        self.scale_embedding = scale_embedding
        self.use_cache = use_cache
        self.num_hidden_layers = encoder_layers


class Florence2Config(PretrainedConfig):
    model_type = "florence2"
    tokenizer_class = "BartTokenizer"
    keys_to_ignore_at_inference = ["past_key_values"]
    sub_configs = {"vision_config": Florence2VisionConfig, "text_config": Florence2LanguageConfig}

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        ignore_index=-100,
        vocab_size=51289,
        projection_dim=768,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        is_encoder_decoder=True,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            **kwargs,
        )
        self.vision_config = (
            Florence2VisionConfig(**vision_config) if isinstance(vision_config, dict) else vision_config
        ) or Florence2VisionConfig()
        self.text_config = (
            Florence2LanguageConfig(**text_config) if isinstance(text_config, dict) else text_config
        ) or Florence2LanguageConfig()
        self.ignore_index = ignore_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.tokenizer_class = "BartTokenizer"
        self.decoder_start_token_id = self.text_config.decoder_start_token_id
        self.forced_eos_token_id = self.text_config.forced_eos_token_id
        self.use_cache = self.text_config.use_cache
