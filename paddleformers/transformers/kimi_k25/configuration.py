# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2026 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
from ..modeling_rope_utils import rope_config_validation, standardize_rope_params


class Kimi25TextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Kimi25TextModel`]. It is used to instantiate a
    Kimi-K2.5 model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 102400):
            Vocabulary size of the KimiK25 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Kimi25TextModel`]
        hidden_size (`int`, *optional*, defaults to 7168):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 18432):
            Dimension of the MLP representations.
        moe_intermediate_size (`int`, *optional*, defaults to 1536):
            Dimension of the MoE representations.
        num_hidden_layers (`int`, *optional*, defaults to 64):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 128):
            Number of attention heads for each attention layer in the Transformer decoder.
        n_shared_experts (`int`, *optional*, defaults to None):
            Number of shared experts, None means dense model.
        n_routed_experts (`int`, *optional*, defaults to None):
            Number of routed experts, None means dense model.
        routed_scaling_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for routed experts.
        kv_lora_rank (`int`, *optional*, defaults to 512):
            Rank for key/value LoRA projection (for MLA attention).
        q_lora_rank (`int`, *optional*, defaults to 1536):
            Rank for query LoRA projection (for MLA attention).
        qk_rope_head_dim (`int`, *optional*, defaults to 64):
            Head dimension for QK rope (for MLA attention).
        v_head_dim (`int`, *optional*, defaults to 128):
            Head dimension for value projection (for MLA attention).
        qk_nope_head_dim (`int`, *optional*, defaults to 128):
            Head dimension for QK non-rope projection (for MLA attention).
        multi_latent_attention (`bool`, *optional*, defaults to True):
            Whether to use multi-latent attention (MLA) mechanism.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 131072):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings.
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
    """

    model_type = "kimi_k25_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=102400,
        hidden_size=7168,
        intermediate_size=18432,
        moe_intermediate_size=1536,
        num_hidden_layers=64,
        num_attention_heads=128,
        num_key_value_heads=128,
        n_shared_experts=None,
        n_routed_experts=None,
        ep_size=1,
        routed_scaling_factor=1.0,
        kv_lora_rank=512,
        q_lora_rank=1536,
        qk_rope_head_dim=64,
        v_head_dim=128,
        qk_nope_head_dim=128,
        multi_latent_attention=True,
        topk_method="greedy",
        n_group=None,
        topk_group=None,
        num_experts_per_tok=None,
        moe_layer_freq=1,
        first_k_dense_replace=0,
        norm_topk_prob=False,
        scoring_func="softmax",
        seq_aux=True,
        hidden_act="silu",
        max_position_embeddings=131072,
        seq_length=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.ep_size = ep_size
        self.routed_scaling_factor = routed_scaling_factor
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.head_dim = qk_rope_head_dim
        self.multi_latent_attention = multi_latent_attention
        self.topk_method = topk_method
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_layer_freq = moe_layer_freq
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob
        self.scoring_func = scoring_func
        self.seq_aux = seq_aux
        self.attention_dropout = attention_dropout

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias

        self.rope_parameters = rope_scaling
        standardize_rope_params(self, rope_theta=rope_theta)
        rope_config_validation(self)

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class KimiK25VisionConfig(PretrainedConfig):
    def __init__(
        self,
        patch_size: int = 14,
        init_pos_emb_height: int = 64,
        init_pos_emb_width: int = 64,
        init_pos_emb_time: int = 4,
        pos_emb_type: str = "divided_fixed",
        vt_num_attention_heads: int = 16,
        vt_num_hidden_layers: int = 27,
        vt_hidden_size: int = 1152,
        vt_intermediate_size: int = 4304,
        merge_kernel_size: tuple = (2, 2),
        video_attn_type: str = "spatial_temporal",
        merge_type: str = "sd2_tpool",
        _attn_implementation: str = "flash_attention_2",
        # MM Projector parameters
        mm_projector_type: str = "patchmerger",
        mm_hidden_size: int | None = None,
        projector_hidden_act: str = "gelu",
        projector_ln_eps: float = 1e-5,
        # Other parameters
        ignore_index: int = -100,
        media_placeholder_token_id: int = 163605,
        pad_token_id: int = 0,
        use_unified_vision_chunk: bool = True,
        video_placeholder="<|kimi_k25_video_placeholder|>",
        text_hidden_size=7168,
        **vision_config_kwargs
    ):

        super().__init__(**vision_config_kwargs)
        self.patch_size = patch_size
        self.init_pos_emb_height = init_pos_emb_height
        self.init_pos_emb_width = init_pos_emb_width
        self.init_pos_emb_time = init_pos_emb_time
        self.pos_emb_type = pos_emb_type
        self.vt_num_attention_heads = vt_num_attention_heads
        self.vt_num_hidden_layers = vt_num_hidden_layers
        self.vt_hidden_size = vt_hidden_size
        self.vt_intermediate_size = vt_intermediate_size
        self.merge_kernel_size = merge_kernel_size
        self.video_attn_type = video_attn_type
        self.merge_type = merge_type
        self._attn_implementation = _attn_implementation

        # MM Projector config
        self.mm_projector_type = mm_projector_type
        self.mm_hidden_size = mm_hidden_size if mm_hidden_size is not None else vt_hidden_size
        self.projector_hidden_act = projector_hidden_act
        self.projector_ln_eps = projector_ln_eps
        self.text_hidden_size = text_hidden_size


class KimiK25Config(PretrainedConfig):
    """Kimi-K2.5 model configuration.
    Args:
        text_config (dict | DeepseekV3Config): Configuration for the text model.

        Vision Tower Parameters (from MoonViT3dConfig):
            patch_size (int): Patch size for vision tower.
            init_pos_emb_height (int): Initial position embedding height.
            init_pos_emb_width (int): Initial position embedding width.
            init_pos_emb_time (int): Initial position embedding time dimension.
            pos_emb_type (str): Type of position embedding.
            vt_num_attention_heads (int): Number of attention heads in vision tower.
            vt_num_hidden_layers (int): Number of hidden layers in vision tower.
            vt_hidden_size (int): Hidden size of vision tower.
            vt_intermediate_size (int): Intermediate size in vision tower FFN.
            merge_kernel_size (tuple): Kernel size for patch merging.
            video_attn_type (str): Type of video attention.
            merge_type (str): Type of merge operation.
            _attn_implementation (str): Attention implementation type.

        MM Projector Parameters (from MultiModalProjectorConfig):
            mm_projector_type (str): Type of multimodal projector.
            mm_hidden_size (int): Hidden size from vision tower (should match vt_hidden_size).
            projector_hidden_act (str): Activation function for projector.
            projector_ln_eps (float): Layer norm epsilon for projector.

        Other Parameters:
            ignore_index (int): The ignore index for the loss function.
            media_placeholder_token_id (int): The token ID to use for media placeholders.
            pad_token_id (int): The token ID to use for padding.
    """

    model_type = "kimi_k25"

    def __init__(
        self,
        text_config: dict | Kimi25TextConfig = None,
        vision_config: dict | KimiK25VisionConfig = None,
        # Other parameters
        ignore_index: int = -100,
        media_placeholder_token_id: int = 163605,
        pad_token_id: int = 0,
        use_unified_vision_chunk: bool = True,
        video_placeholder="<|kimi_k25_video_placeholder|>",
        **kwargs,
    ):
        if isinstance(text_config, dict):
            text_config = Kimi25TextConfig(**text_config)
        if isinstance(vision_config, dict):
            vision_config = KimiK25VisionConfig(**vision_config)
        self.text_config = text_config
        self.vision_config = vision_config
        # Other config
        self.ignore_index = ignore_index
        self.media_placeholder_token_id = media_placeholder_token_id
        self.use_unified_vision_chunk = use_unified_vision_chunk
        self.video_placeholder = video_placeholder
        if getattr(self.text_config, "quantization_config", None) is not None:
            self.quantization_config = self.text_config.quantization_config

        super().__init__(pad_token_id=pad_token_id, **kwargs)


__all__ = ["KimiK25Config", "KimiK25VisionConfig"]
