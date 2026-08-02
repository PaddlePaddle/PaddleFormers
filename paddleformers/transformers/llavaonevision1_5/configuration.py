# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2026 The LLaVA-OneVision-1.5 Authors and The HuggingFace Inc. team. All rights reserved.
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
"""LLaVA-OneVision-1.5 model configuration."""

from ..configuration_utils import PretrainedConfig, layer_type_validation
from ..modeling_rope_utils import rope_config_validation, standardize_rope_params


class RiceConfig(PretrainedConfig):
    """Configuration for the Rice ViT vision tower used by LLaVA-OneVision-1.5."""

    model_type = "rice_vit"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth=24,
        embed_dim=1024,
        hidden_size=1024,
        hidden_act="gelu",
        intermediate_size=4096,
        num_heads=16,
        in_channels=3,
        patch_size=14,
        spatial_merge_size=2,
        temporal_patch_size=1,
        initializer_range=0.02,
        layer_norm_eps=1e-05,
        text_hidden_size=2560,
        _attn_implementation="sdpa",
        **kwargs,
    ):
        super().__init__(_attn_implementation=_attn_implementation, **kwargs)
        self.depth = depth
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.text_hidden_size = text_hidden_size


class LLaVAOneVision1_5TextConfig(PretrainedConfig):
    """Text backbone configuration for LLaVA-OneVision-1.5."""

    model_type = "llavaonevision1_5_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=2560,
        intermediate_size=9728,
        num_hidden_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=262144,
        initializer_range=0.02,
        rms_norm_eps=1e-06,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=1000000.0,
        attention_bias=False,
        use_sliding_window=False,
        sliding_window=None,
        max_window_layers=36,
        attention_dropout=0.0,
        rope_scaling=None,
        layer_types=None,
        image_token_id=None,
        video_token_id=None,
        fuse_rms_norm=True,
        _attn_implementation="sdpa",
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.tie_word_embeddings = tie_word_embeddings
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.fuse_rms_norm = fuse_rms_norm

        if self.rope_scaling is not None and "type" in self.rope_scaling:
            if self.rope_scaling["type"] == "mrope":
                self.rope_scaling["type"] = "default"
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        self.rope_parameters = self.rope_scaling

        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if self.sliding_window is not None and i >= self.max_window_layers
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.layer_types, self.num_hidden_layers)

        standardize_rope_params(self, rope_theta=rope_theta)
        rope_config_validation(self, ignore_keys={"mrope_section"})

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            _attn_implementation=_attn_implementation,
            **kwargs,
        )


class Llavaonevision1_5Config(PretrainedConfig):
    """Top-level configuration for LLaVA-OneVision-1.5."""

    model_type = "llavaonevision1_5"
    sub_configs = {"vision_config": RiceConfig, "text_config": LLaVAOneVision1_5TextConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
        vocab_size=152064,
        tie_word_embeddings=None,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()
        else:
            self.vision_config = vision_config

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"](**kwargs)
        else:
            self.text_config = text_config

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vocab_size = vocab_size

        if tie_word_embeddings is None:
            tie_word_embeddings = self.text_config.tie_word_embeddings

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


__all__ = ["Llavaonevision1_5Config", "LLaVAOneVision1_5TextConfig", "RiceConfig"]
