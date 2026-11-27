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

"""Composite configs for the native Transformers DeepSeek-OCR-2 format."""

from ..configuration_utils import PretrainedConfig
from ..llama.configuration import LlamaConfig
from ..qwen2.configuration import Qwen2Config


class DeepseekOcr2SamVisionConfig(PretrainedConfig):
    model_type = "deepseek_ocr2_sam_vision_model"

    def __init__(
        self,
        hidden_size=768,
        output_channels=256,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=1024,
        patch_size=16,
        hidden_act="gelu",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        initializer_range=1e-10,
        qkv_bias=True,
        mlp_ratio=4.0,
        mlp_dim=None,
        use_abs_pos=True,
        use_rel_pos=True,
        window_size=14,
        global_attn_indexes=(2, 5, 8, 11),
        downsample_channels=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.output_channels = output_channels
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.qkv_bias = qkv_bias
        self.mlp_ratio = mlp_ratio
        self.mlp_dim = int(hidden_size * mlp_ratio) if mlp_dim is None else mlp_dim
        self.use_abs_pos = use_abs_pos
        self.use_rel_pos = use_rel_pos
        self.window_size = window_size
        self.global_attn_indexes = list(global_attn_indexes)
        self.downsample_channels = [512, 896] if downsample_channels is None else list(downsample_channels)
        if len(self.downsample_channels) != 2:
            raise ValueError("SAM requires exactly two downsampling channel sizes")


class DeepseekOcr2VisionEncoderConfig(Qwen2Config):
    model_type = "deepseek_ocr2_encoder"

    def __init__(self, **kwargs):
        rope = kwargs.pop("rope_parameters", None)
        if rope is not None:
            kwargs["rope_theta"] = rope.get("rope_theta", 10000.0)
            kwargs["rope_scaling"] = rope
        super().__init__(**kwargs)


class DeepseekOcr2VisionConfig(PretrainedConfig):
    model_type = "deepseek_ocr2_vision"
    sub_configs = {"sam_config": DeepseekOcr2SamVisionConfig, "encoder_config": DeepseekOcr2VisionEncoderConfig}

    def __init__(self, sam_config=None, encoder_config=None, **kwargs):
        super().__init__(**kwargs)
        self.sam_config = (
            DeepseekOcr2SamVisionConfig(**(sam_config or {}))
            if isinstance(sam_config, dict) or sam_config is None
            else sam_config
        )
        self.encoder_config = (
            DeepseekOcr2VisionEncoderConfig(**(encoder_config or {}))
            if isinstance(encoder_config, dict) or encoder_config is None
            else encoder_config
        )


class DeepseekOcr2TextConfig(LlamaConfig):
    model_type = "deepseek_ocr2_text"
    attribute_map = {"num_experts": "n_routed_experts"}

    def __init__(
        self,
        n_routed_experts=64,
        n_shared_experts=2,
        routed_scaling_factor=1.0,
        n_group=None,
        topk_group=None,
        topk_method="greedy",
        num_experts_per_tok=None,
        moe_intermediate_size=1407,
        mlp_layer_types=None,
        **kwargs
    ):
        rope = kwargs.pop("rope_parameters", None)
        if rope is not None:
            kwargs["rope_theta"] = rope.get("rope_theta", 10000.0)
            kwargs["rope_scaling"] = rope
        super().__init__(**kwargs)
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.n_group = n_group
        self.topk_group = topk_group
        self.topk_method = topk_method
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.mlp_layer_types = (
            list(mlp_layer_types) if mlp_layer_types is not None else ["dense"] * self.num_hidden_layers
        )
        if len(self.mlp_layer_types) != self.num_hidden_layers or any(
            x not in ("dense", "sparse") for x in self.mlp_layer_types
        ):
            raise ValueError("mlp_layer_types must specify dense/sparse for every text layer")
        if "sparse" in self.mlp_layer_types:
            if not num_experts_per_tok or not 0 < num_experts_per_tok <= n_routed_experts:
                raise ValueError("Sparse layers require 0 < num_experts_per_tok <= n_routed_experts")
            if topk_method not in ("greedy", "group_limited_greedy"):
                raise ValueError("Unsupported DeepSeek-OCR-2 routing method")
            if topk_method == "group_limited_greedy" and (
                not n_group or n_routed_experts % n_group or not topk_group or not 0 < topk_group <= n_group
            ):
                raise ValueError("Invalid grouped expert routing configuration")


class DeepseekOcr2Config(PretrainedConfig):
    model_type = "deepseek_ocr2"
    sub_configs = {"vision_config": DeepseekOcr2VisionConfig, "text_config": DeepseekOcr2TextConfig}

    def __init__(
        self, vision_config=None, text_config=None, image_token_id=128815, tie_word_embeddings=False, **kwargs
    ):
        self.vision_config = (
            DeepseekOcr2VisionConfig(**(vision_config or {}))
            if isinstance(vision_config, dict) or vision_config is None
            else vision_config
        )
        self.text_config = (
            DeepseekOcr2TextConfig(**(text_config or {}))
            if isinstance(text_config, dict) or text_config is None
            else text_config
        )
        self.image_token_id = image_token_id
        tie_word_embeddings = kwargs.pop("tie_embedding", tie_word_embeddings) or self.text_config.tie_word_embeddings
        self.text_config.tie_word_embeddings = tie_word_embeddings
        kwargs.setdefault("architectures", ["DeepseekOcr2ForConditionalGeneration"])
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    def to_dict(self, saving_file=False):
        result = super().to_dict(saving_file=saving_file)

        def normalize_dtype(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key in ("dtype", "torch_dtype") and item == "None":
                        value[key] = None
                    else:
                        normalize_dtype(item)
            elif isinstance(value, list):
                for item in value:
                    normalize_dtype(item)

        # HF's strict native configs require JSON null, not the string "None".
        normalize_dtype(result)
        return result


__all__ = [
    "DeepseekOcr2Config",
    "DeepseekOcr2TextConfig",
    "DeepseekOcr2VisionConfig",
    "DeepseekOcr2SamVisionConfig",
    "DeepseekOcr2VisionEncoderConfig",
]
