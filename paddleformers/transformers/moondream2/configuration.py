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

"""Configuration classes for the Moondream2 vision-language model."""

from ..configuration_utils import PretrainedConfig

__all__ = [
    "Moondream2Config",
    "Moondream2RegionConfig",
    "Moondream2TextConfig",
    "Moondream2VisionConfig",
]

# The reference checkpoint declares ``model_type`` as ``moondream1``; the
# released weights are the second generation of that family.
MODEL_TYPE = "moondream1"


class Moondream2VisionConfig(PretrainedConfig):
    """Configuration of Moondream2's ViT encoder and image projector."""

    model_type = MODEL_TYPE
    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_size=1152,
        patch_size=14,
        num_hidden_layers=27,
        intermediate_size=4304,
        num_attention_heads=16,
        projection_dim=2048,
        crop_size=378,
        num_channels=3,
        max_crops=12,
        overlap_margin=4,
        projection_inner_dim=8192,
        layer_norm_eps=1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.projection_dim = projection_dim
        self.crop_size = crop_size
        self.num_channels = num_channels
        self.max_crops = max_crops
        self.overlap_margin = overlap_margin
        self.projection_inner_dim = projection_inner_dim
        self.layer_norm_eps = layer_norm_eps
        self._validate()

    def _validate(self):
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("vision hidden_size must be divisible by num_attention_heads")
        if self.crop_size % self.patch_size:
            raise ValueError("vision crop_size must be divisible by patch_size")
        if self.overlap_margin * 2 >= self.crop_size // self.patch_size:
            raise ValueError("overlap_margin leaves no usable local crop area")


class Moondream2TextConfig(PretrainedConfig):
    """Configuration of Moondream2's causal text decoder."""

    model_type = MODEL_TYPE
    base_config_key = "text_config"

    def __init__(
        self,
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=24,
        vocab_size=51200,
        max_position_embeddings=2048,
        num_attention_heads=32,
        num_key_value_heads=32,
        prefix_attn=730,
        rotary_dim=32,
        rope_theta=10000.0,
        layer_norm_eps=1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.prefix_attn = prefix_attn
        self.rotary_dim = rotary_dim
        self.rope_theta = rope_theta
        self.layer_norm_eps = layer_norm_eps
        self._validate()

    def _validate(self):
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("text hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.rotary_dim % 2 or self.rotary_dim > self.hidden_size // self.num_attention_heads:
            raise ValueError("rotary_dim must be even and no greater than head_dim")
        if not 0 <= self.prefix_attn <= self.max_position_embeddings:
            raise ValueError("prefix_attn must be in [0, max_position_embeddings]")


class Moondream2RegionConfig(PretrainedConfig):
    """Configuration of the optional Moondream2 visual-grounding heads."""

    model_type = MODEL_TYPE
    base_config_key = "region_config"

    def __init__(
        self,
        hidden_size=2048,
        coordinate_feature_dim=256,
        coordinate_output_dim=1024,
        size_feature_dim=512,
        size_output_dim=2048,
        intermediate_size=8192,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.coordinate_feature_dim = coordinate_feature_dim
        self.coordinate_output_dim = coordinate_output_dim
        self.size_feature_dim = size_feature_dim
        self.size_output_dim = size_output_dim
        self.intermediate_size = intermediate_size


class Moondream2Config(PretrainedConfig):
    """Top-level configuration for Moondream2.

    The defaults reproduce the public 2025-01-09 Moondream2 architecture.  The
    nested configuration objects are intentionally serializable so converted
    checkpoints can be loaded through the standard PaddleFormers Auto classes.

    The reference checkpoint ships a flat layout (``text.dim``,
    ``vision.enc_dim``, ...) and declares ``model_type="moondream1"``.
    ``from_dict`` translates that layout into the conventional PaddleFormers
    field names (``text_config.hidden_size``, ...) so the rest of the library
    only ever sees one naming scheme.
    """

    model_type = MODEL_TYPE
    keys_to_ignore_at_inference = ["past_key_values"]
    sub_configs = {
        "vision_config": Moondream2VisionConfig,
        "text_config": Moondream2TextConfig,
        "region_config": Moondream2RegionConfig,
    }

    # Reference ``config.py`` field names -> PaddleFormers configuration names.
    _OFFICIAL_TEXT_KEYS = {
        "dim": "hidden_size",
        "ff_dim": "intermediate_size",
        "n_layers": "num_hidden_layers",
        "vocab_size": "vocab_size",
        "max_context": "max_position_embeddings",
        "n_heads": "num_attention_heads",
        "n_kv_heads": "num_key_value_heads",
        "prefix_attn": "prefix_attn",
    }
    _OFFICIAL_VISION_KEYS = {
        "enc_dim": "hidden_size",
        "enc_patch_size": "patch_size",
        "enc_n_layers": "num_hidden_layers",
        "enc_ff_dim": "intermediate_size",
        "enc_n_heads": "num_attention_heads",
        "proj_out_dim": "projection_dim",
        "crop_size": "crop_size",
        "in_channels": "num_channels",
        "max_crops": "max_crops",
        "overlap_margin": "overlap_margin",
        "proj_inner_dim": "projection_inner_dim",
    }
    _OFFICIAL_REGION_KEYS = {
        "dim": "hidden_size",
        "coord_feat_dim": "coordinate_feature_dim",
        "coord_out_dim": "coordinate_output_dim",
        "size_feat_dim": "size_feature_dim",
        "size_out_dim": "size_output_dim",
        "inner_dim": "intermediate_size",
    }

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        region_config=None,
        pad_token_id=0,
        bos_token_id=0,
        eos_token_id=0,
        tie_word_embeddings=False,
        use_cache=True,
        lm_head_bias=True,
        image_token_id=None,
        tokenizer_name_or_path="moondream/starmie-v1",
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vision_config = self._make_sub_config("vision_config", vision_config)
        self.text_config = self._make_sub_config("text_config", text_config)
        self.region_config = self._make_sub_config("region_config", region_config)
        self.vocab_size = self.text_config.vocab_size
        self.hidden_size = self.text_config.hidden_size
        self.use_cache = use_cache
        self.lm_head_bias = lm_head_bias
        # Placeholder token expanded in place by the model when the template
        # keeps ``<image>`` in the prompt. ``None`` keeps the legacy behaviour
        # of prepending the image prefix to every sequence.
        self.image_token_id = image_token_id
        # The released checkpoint is paired with ``moondream/starmie-v1``; the
        # tokenizer files inside the HF repository produce different ids.
        self.tokenizer_name_or_path = tokenizer_name_or_path

    def _make_sub_config(self, name, value):
        config_class = self.sub_configs[name]
        if value is None:
            return config_class()
        if isinstance(value, dict):
            return config_class(**value)
        if isinstance(value, config_class):
            return value
        raise TypeError(f"{name} must be a dict or {config_class.__name__}, got {type(value)!r}")

    @classmethod
    def _convert_official_section(cls, section, key_mapping):
        if not isinstance(section, dict):
            return section
        return {key_mapping.get(key, key): value for key, value in section.items()}

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        """Accept both the reference flat layout and the PaddleFormers layout."""
        if isinstance(config_dict, dict) and any(
            section in config_dict for section in ("text", "vision", "region", "tokenizer")
        ):
            config_dict = dict(config_dict)
            tokenizer_section = config_dict.pop("tokenizer", None) or {}
            converted = {
                "text_config": cls._convert_official_section(config_dict.pop("text", None), cls._OFFICIAL_TEXT_KEYS),
                "vision_config": cls._convert_official_section(
                    config_dict.pop("vision", None), cls._OFFICIAL_VISION_KEYS
                ),
                "region_config": cls._convert_official_section(
                    config_dict.pop("region", None), cls._OFFICIAL_REGION_KEYS
                ),
            }
            config_dict.update({key: value for key, value in converted.items() if value is not None})
            if "bos_id" in tokenizer_section:
                config_dict.setdefault("bos_token_id", tokenizer_section["bos_id"])
            if "eos_id" in tokenizer_section:
                config_dict.setdefault("eos_token_id", tokenizer_section["eos_id"])
        return super().from_dict(config_dict, **kwargs)
