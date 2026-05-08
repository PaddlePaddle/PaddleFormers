# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2024 HuggingFace Inc. team. All rights reserved.
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
"""Molmo model configuration."""

from ..configuration_utils import PretrainedConfig
from ..modeling_rope_utils import rope_config_validation, standardize_rope_params


class MolmoConfig(PretrainedConfig):
    r"""
    Configuration for Molmo (text-only LLM backbone).

    Molmo is an OLMo-family model with:
    - Post-norm architecture (``norm_after=True``)
    - Optional QK-Norm (``attention_layer_norm=True``)
    - Extended embedding size (``embedding_size`` may exceed ``vocab_size``)
    - SwiGLU activation with fused gate+up projection

    Args:
        vocab_size (int): Vocabulary size. The model's actual embedding table may
            be larger than this (up to ``embedding_size``) to pad to a multiple of 128.
        embedding_size (int): Size of the embedding table. Defaults to ``vocab_size``
            rounded up to the next multiple of 128.
        hidden_size (int): Dimensionality of the model hidden states.
        intermediate_size (int): Dimensionality of the MLP intermediate layer
            (after the fused gate+up projection, before SwiGLU splits it in half).
        num_hidden_layers (int): Number of transformer decoder layers.
        num_attention_heads (int): Number of attention heads.
        num_key_value_heads (int): Number of KV heads for GQA. Defaults to
            ``num_attention_heads`` (no GQA).
        attention_layer_norm (bool): Whether to apply QK-Norm (RMSNorm on Q and K).
        norm_after (bool): If True, use post-norm (OLMo2 style). If False, use pre-norm.
        layer_norm_type (str): Type of layer norm: "rms", "default", or "low_precision".
        layer_norm_eps (float): Epsilon for layer norm.
        qkv_bias (bool): Whether to use bias in Q/K/V projections.
        clip_qkv (float, optional): Clip Q/K/V values to this range.
        rope_theta (float): RoPE base frequency.
        max_position_embeddings (int): Maximum sequence length.
        weight_tying (bool): Whether to tie input embedding and LM head weights.
            Maps to HF's ``tie_word_embeddings``.
        use_position_ids (bool): Whether to use position IDs for RoPE.
    """

    model_type = "molmo"

    def __init__(
        self,
        vocab_size: int = 100278,
        embedding_size: int = 100352,
        hidden_size: int = 4096,
        intermediate_size: int = 22016,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int = None,
        attention_layer_norm: bool = True,
        norm_after: bool = True,
        layer_norm_type: str = "rms",
        layer_norm_eps: float = 1e-6,
        additional_vocab_size: int = 128,
        qkv_bias: bool = False,
        clip_qkv: float = None,
        rope_theta: float = 500000.0,
        rope_impl: str = "llama",
        max_position_embeddings: int = 4096,
        use_position_ids: bool = True,
        weight_tying: bool = False,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        pad_token_id: int = None,
        bos_token_id: int = None,
        eos_token_id: int = None,
        attention_dropout: float = 0.0,
        head_dim: int = None,
        # Vision backbone fields used by Molmo-7B-O-0924.
        vision_backbone: dict | None = None,
        image_padding_embed: str | None = "pad_and_partial_pad",
        vit_layers: list | tuple | None = (-2, -9),
        image_pooling_h: int = 2,
        image_pooling_w: int = 2,
        image_pooling_2d: str = "attention-meanq",
        image_projector: str = "mlp",
        image_feature_dropout: float = 0.0,
        vision_attention_type: str = "sdpa",
        float32_attention: bool = True,
        activation_type: str = "swiglu",
        **kwargs,
    ):
        self.vocab_size = vocab_size
        # embedding_size: size of the base embedding table (padded to multiple of 128)
        self.embedding_size = embedding_size if embedding_size is not None else vocab_size
        # additional_vocab_size: number of extra tokens (e.g. image tokens) stored in new_embedding
        self.additional_vocab_size = additional_vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        self.attention_layer_norm = attention_layer_norm
        self.norm_after = norm_after
        self.layer_norm_type = layer_norm_type
        self.layer_norm_eps = layer_norm_eps
        self.qkv_bias = qkv_bias
        self.clip_qkv = clip_qkv
        self.max_position_embeddings = max_position_embeddings
        self.use_position_ids = use_position_ids
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.activation_type = activation_type

        if vision_backbone is None:
            vision_backbone = {
                "image_default_input_size": (336, 336),
                "image_patch_size": 14,
                "image_pos_patch_size": 14,
                "image_emb_dim": 1024,
                "image_num_heads": 16,
                "image_num_key_value_heads": 16,
                "image_num_layers": 23,
                "image_head_dim": 64,
                "image_mlp_dim": 4096,
                "image_mlp_activations": "quick_gelu",
                "image_dropout_rate": 0.0,
                "image_num_pos": 577,
                "image_norm_eps": 1e-5,
                "attention_dropout": 0.0,
                "residual_dropout": 0.0,
                "initializer_range": 0.02,
            }
        self.vision_backbone = vision_backbone
        if self.vision_backbone is not None and "image_default_input_size" in self.vision_backbone:
            self.vision_backbone["image_default_input_size"] = tuple(self.vision_backbone["image_default_input_size"])
        self.image_padding_embed = image_padding_embed
        self.vit_layers = tuple(vit_layers) if vit_layers is not None else None
        self.image_pooling_h = image_pooling_h
        self.image_pooling_w = image_pooling_w
        self.image_pooling_2d = image_pooling_2d
        self.image_projector = image_projector
        self.image_feature_dropout = image_feature_dropout
        self.vision_attention_type = vision_attention_type
        self.float32_attention = float32_attention

        # HF uses tie_word_embeddings; molmo ref uses weight_tying — normalize both
        # kwargs may contain tie_word_embeddings from HF checkpoint configs
        tie_word_embeddings = kwargs.pop("tie_word_embeddings", weight_tying)
        self.weight_tying = weight_tying or tie_word_embeddings

        self.rope_theta = rope_theta
        self.rope_impl = rope_impl
        self.rope_scaling = kwargs.pop("rope_scaling", None)
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        self.rope_parameters = self.rope_scaling
        standardize_rope_params(self, rope_theta=self.rope_theta)
        rope_config_validation(self)

        # Default to SDPA attention
        kwargs.setdefault("_attn_implementation", "sdpa")

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=self.weight_tying,
            **kwargs,
        )

    @property
    def image_num_patch(self):
        size = self.vision_backbone["image_default_input_size"]
        patch = self.vision_backbone["image_patch_size"]
        return size[0] // patch, size[1] // patch

    def llm_patches_per_crop(self):
        h, w = self.image_num_patch
        h = (h + self.image_pooling_h - 1) // self.image_pooling_h
        w = (w + self.image_pooling_w - 1) // self.image_pooling_w
        return h, w
