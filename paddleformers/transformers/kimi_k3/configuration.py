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


class KimiK3TextConfig(PretrainedConfig):
    r"""
    Configuration class for the Kimi-K3 text model.

    Kimi-K3 combines a Kimi Delta Attention (KDA) and gated Multi-Latent
    Attention (MLA) schedule with block attention residuals, SiTU feed-forward
    activations, and Stable LatentMoE experts.

    Args:
        vocab_size (`int`, *optional*, defaults to 163840):
            Vocabulary size of the model.
        hidden_size (`int`, *optional*, defaults to 7168):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 33792):
            Intermediate size of the dense feed-forward layers.
        num_hidden_layers (`int`, *optional*, defaults to 93):
            Number of decoder layers.
        num_nextn_predict_layers (`int`, *optional*, defaults to 0):
            Number of Multi-Token Prediction layers.
        hidden_act (`str`, *optional*, defaults to `"situ"`):
            Activation function used by dense, shared, and routed experts.
        max_sequence_length (`int`, *optional*, defaults to 1048576):
            Maximum sequence length supported by the model.
        rms_norm_eps (`float`, *optional*, defaults to 1e-5):
            Epsilon used by RMSNorm.
        linear_attn_config (`dict`, *optional*):
            KDA configuration and the one-based KDA/MLA layer schedule. The
            dictionary contains `kda_layers`, `full_attn_layers`, KDA head
            dimensions, and short-convolution settings.
        attn_res_block_size (`int`, *optional*, defaults to 12):
            Number of decoder layers in each block attention residual group.
        num_attention_heads (`int`, *optional*, defaults to 96):
            Number of attention heads in gated MLA layers.
        num_key_value_heads (`int`, *optional*, defaults to 96):
            Number of key/value heads.
        q_lora_rank (`int`, *optional*, defaults to 1536):
            Low-rank dimension of the MLA query projection.
        kv_lora_rank (`int`, *optional*, defaults to 512):
            Low-rank dimension of the MLA key/value projection.
        qk_nope_head_dim (`int`, *optional*, defaults to 128):
            Non-positional dimension of each MLA query/key head.
        qk_rope_head_dim (`int`, *optional*, defaults to 64):
            RoPE dimension of each MLA query/key head.
        v_head_dim (`int`, *optional*, defaults to 128):
            Value dimension of each MLA head.
        multi_latent_attention (`bool`, *optional*, defaults to `True`):
            Whether to use Multi-Latent Attention for full-attention layers.
        mla_use_nope (`bool`, *optional*, defaults to `True`):
            Whether MLA uses the non-positional query/key projection.
        gated_attention (`bool`, *optional*, defaults to `True`):
            Whether MLA applies its learned output gate.
        use_qk_norm (`bool`, *optional*, defaults to `True`):
            Whether to normalize the MLA query and key latent projections.
        moe_intermediate_size (`int`, *optional*, defaults to 3072):
            Intermediate size of each routed expert.
        n_shared_experts (`int`, *optional*, defaults to 2):
            Number of shared experts.
        n_routed_experts (`int`, *optional*, defaults to 896):
            Number of routed experts.
        num_experts_per_tok (`int`, *optional*, defaults to 16):
            Number of routed experts selected for each token.
        routed_scaling_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor applied to routed expert outputs.
        topk_method (`str`, *optional*, defaults to `"quantile_balancing"`):
            Expert selection method used by the MoE router.
        n_group (`int`, *optional*, defaults to 1):
            Number of router expert groups.
        topk_group (`int`, *optional*, defaults to 1):
            Number of expert groups selected for each token.
        moe_layer_freq (`int`, *optional*, defaults to 1):
            Frequency of MoE layers after the initial dense layers.
        first_k_dense_replace (`int`, *optional*, defaults to 1):
            Number of leading decoder layers that use a dense feed-forward
            network instead of MoE.
        norm_topk_prob (`bool`, *optional*, defaults to `True`):
            Whether to normalize the selected expert weights.
        scoring_func (`str`, *optional*, defaults to `"sigmoid"`):
            Activation function used to compute router scores.
        qb_n_bins (`int`, *optional*, defaults to 256):
            Number of histogram bins used by Quantile Balancing.
        moe_router_load_balancing_type (`str`, *optional*, defaults to `"none"`):
            Auxiliary load-balancing strategy. Quantile Balancing requires
            `"none"` because it balances expert load directly.
        router_aux_loss_coef (`float`, *optional*, defaults to 0.0):
            Router auxiliary-loss coefficient. Quantile Balancing requires it
            to be disabled.
        seq_aux (`bool`, *optional*, defaults to `False`):
            Whether to compute sequence-level router auxiliary loss.
        moe_topk_fusion (`bool`, *optional*, defaults to `False`):
            Whether to use the fused MoE TopK kernel. Quantile Balancing does
            not support this fusion.
        moe_latent_size (`int`, *optional*, defaults to 3584):
            Stable LatentMoE latent projection size.
        latent_moe_use_norm (`bool`, *optional*, defaults to `True`):
            Whether to apply RMSNorm to the Stable LatentMoE output latent.
        activation_situ_beta (`float`, *optional*, defaults to 4.0):
            Scale of the tanh term in the SiTU gate activation.
        activation_situ_linear_beta (`float`, *optional*, defaults to 25.0):
            Tanh scale applied to the linear branch of SiTU-GLU.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            Base period of the RoPE embeddings.
        rope_scaling (`dict`, *optional*):
            RoPE scaling configuration.

    Official Kimi/HuggingFace field names are translated to PaddleFleet fields
    by `_HF_TO_FLEET_FIELD_MAP` when loading existing configuration files.
    """

    model_type = "kimi_linear"
    keys_to_ignore_at_inference = ["past_key_values"]

    # HuggingFace config.json field name -> PaddleFleet TransformerConfig field name.
    _HF_TO_FLEET_FIELD_MAP = {
        "max_position_embeddings": "max_sequence_length",
        "mla_use_output_gate": "gated_attention",
        "num_shared_experts": "n_shared_experts",
        "num_experts": "n_routed_experts",
        "num_experts_per_token": "num_experts_per_tok",
        "num_expert_group": "n_group",
        "moe_renormalize": "norm_topk_prob",
        "moe_router_activation_func": "scoring_func",
        "routed_expert_hidden_size": "moe_latent_size",
    }

    def __init__(
        self,
        # === Basic architecture ===
        vocab_size=163840,
        hidden_size=7168,
        intermediate_size=33792,
        num_hidden_layers=93,
        num_nextn_predict_layers=0,
        hidden_act="situ",
        max_sequence_length=1048576,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        tie_word_embeddings=False,
        normalization="RMSNorm",
        # === KDA/MLA hybrid attention schedule ===
        linear_attn_config=None,
        attn_res_block_size=12,
        # === MLA (Multi-Latent Attention) ===
        num_attention_heads=96,
        num_key_value_heads=96,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        multi_latent_attention=True,
        mla_use_nope=True,
        gated_attention=True,
        use_qk_norm=True,
        attention_bias=False,
        attention_dropout=0.0,
        # === MoE ===
        moe_intermediate_size=3072,
        n_shared_experts=2,
        n_routed_experts=896,
        num_experts_per_tok=16,
        routed_scaling_factor=1.0,
        n_group=1,
        topk_group=1,
        moe_layer_freq=1,
        first_k_dense_replace=1,
        norm_topk_prob=True,
        scoring_func="sigmoid",
        # === Quantile Balancing ===
        topk_method="quantile_balancing",
        qb_n_bins=256,
        moe_router_load_balancing_type="none",
        router_aux_loss_coef=0.0,
        seq_aux=False,
        moe_topk_fusion=False,
        # === Stable LatentMoE ===
        moe_latent_size=3584,
        latent_moe_use_norm=True,
        # === SiTU ===
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
        # === RoPE ===
        rope_theta=10000.0,
        rope_scaling=None,
        **kwargs,
    ):
        # Remap HF-style field names passed via kwargs to Fleet-internal names.
        for hf_name, fleet_name in self._HF_TO_FLEET_FIELD_MAP.items():
            if hf_name in kwargs:
                val = kwargs.pop(hf_name)
                setattr(self, f"_hf_{hf_name}", val)

        # Basic architecture
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.hidden_act = hidden_act
        self.max_sequence_length = max_sequence_length
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache

        # KDA/MLA hybrid attention schedule
        self.linear_attn_config = linear_attn_config
        self.layer_types = self._build_layer_types()
        self._flatten_linear_attn_config()
        self.attn_res_block_size = attn_res_block_size

        # MLA
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.head_dim = linear_attn_config.get("head_dim", v_head_dim) if linear_attn_config else v_head_dim
        self.v_head_dim = self.head_dim
        self.multi_latent_attention = multi_latent_attention
        self.mla_use_nope = mla_use_nope
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.gated_attention = gated_attention
        self.use_qk_norm = use_qk_norm

        # MoE
        self.moe_intermediate_size = moe_intermediate_size
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.n_group = n_group
        self.topk_group = topk_group
        self.moe_layer_freq = moe_layer_freq
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob
        self.scoring_func = scoring_func

        # Quantile Balancing
        self.topk_method = topk_method
        self.qb_n_bins = qb_n_bins
        self.moe_router_load_balancing_type = moe_router_load_balancing_type
        self.router_aux_loss_coef = router_aux_loss_coef
        self.seq_aux = seq_aux
        self.moe_topk_fusion = moe_topk_fusion

        # Stable LatentMoE
        self.latent_moe_use_norm = latent_moe_use_norm
        self.moe_latent_size = moe_latent_size

        # SiTU
        self.activation_situ_beta = activation_situ_beta
        self.activation_situ_linear_beta = activation_situ_linear_beta

        # RoPE
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling

        self.normalization = normalization

        # Apply HF->Fleet field mappings after setting Fleet defaults so values
        # loaded from official Kimi/HuggingFace configs take precedence.
        for hf_name, fleet_name in self._HF_TO_FLEET_FIELD_MAP.items():
            hf_attr = f"_hf_{hf_name}"
            if hasattr(self, hf_attr):
                val = getattr(self, hf_attr)
                if not fleet_name.startswith("_"):
                    setattr(self, fleet_name, val)

        self.seq_length = min(self.max_sequence_length, 1024)

        super().__init__(
            num_nextn_predict_layers=num_nextn_predict_layers,
            tie_word_embeddings=tie_word_embeddings,
            # Keep fields managed by LlmMetaConfig from being reset to its
            # defaults by PretrainedConfig.__init__().
            moe_router_load_balancing_type=moe_router_load_balancing_type,
            multi_latent_attention=multi_latent_attention,
            normalization=normalization,
            router_aux_loss_coef=router_aux_loss_coef,
            **kwargs,
        )

    def _build_layer_types(self):
        """Turn the one-based KDA/MLA schedule into a per-layer type list.

        Shared by the model provider (Fleet build) and the checkpoint AOA
        rules so both derive the KDA/MLA layout from a single source.
        """
        if not isinstance(self.linear_attn_config, dict):
            raise ValueError("Kimi-K3 requires layer_types or linear_attn_config.")
        kda_layers = self._layer_numbers("kda_layers")
        full_attn_layers = self._layer_numbers("full_attn_layers")
        overlap = kda_layers & full_attn_layers
        if overlap:
            raise ValueError(
                "Kimi-K3 kda_layers and full_attn_layers must be disjoint; " f"overlap={sorted(overlap)}."
            )
        expected_layers = set(range(1, self.num_hidden_layers + 1))
        actual_layers = kda_layers | full_attn_layers
        if actual_layers != expected_layers:
            raise ValueError(
                "Kimi-K3 attention schedule must cover every decoder layer "
                f"exactly once; missing={sorted(expected_layers - actual_layers)}, "
                f"out_of_range={sorted(actual_layers - expected_layers)}."
            )
        return [
            "kimi_delta_attention" if layer_number in kda_layers else "multi_latent_attention"
            for layer_number in range(1, self.num_hidden_layers + 1)
        ]

    def _layer_numbers(self, name):
        """Parse and validate a list of one-based layer numbers."""
        values = self.linear_attn_config.get(name)
        if not isinstance(values, (list, tuple)):
            raise ValueError(f"Kimi-K3 {name} must be a list of layer numbers.")
        if any(type(value) is not int for value in values):
            raise ValueError(f"Kimi-K3 {name} must contain only integers.")
        if len(values) != len(set(values)):
            raise ValueError(f"Kimi-K3 {name} contains duplicate layer numbers.")
        return set(values)

    def _flatten_linear_attn_config(self):
        """Expose the nested KDA config as flat ``linear_*`` fields.

        The model provider forwards these to the Fleet ``TransformerConfig``
        and the checkpoint AOA rules read them back, so keep them on the config
        as the single source of the KDA layout.
        """
        cfg = self.linear_attn_config
        if not isinstance(cfg, dict):
            raise ValueError("Kimi-K3 requires linear_attn_config to expose KDA fields.")
        head_dim = cfg["head_dim"]
        num_heads = cfg["num_heads"]
        self.linear_conv_kernel_dim = cfg["short_conv_kernel_size"]
        self.linear_key_head_dim = head_dim
        self.linear_value_head_dim = head_dim
        self.linear_num_key_heads = num_heads
        self.linear_num_value_heads = num_heads
        self.linear_gate_lora_rank = head_dim
        self.linear_use_full_rank_gate = cfg.get("use_full_rank_gate", False)
        self.linear_gate_lower_bound = cfg.get("gate_lower_bound")


class KimiK3VisionConfig(PretrainedConfig):
    r"""
    Configuration class for the Kimi-K3 MoonViT3d vision tower.

    Field names follow the official `vision_config` block in
    `Kimi-K3/config.json`. The `vt_*` names are the HuggingFace spelling of the
    encoder geometry; `_HF_TO_FLEET_FIELD_MAP` translates them to the
    PaddleFleet `TransformerConfig` names consumed by
    `paddlefleet.models.kimi_k3.build_kimi_k3_vision_config`.

    Args:
        patch_size (`int`, *optional*, defaults to 14):
            Spatial size of each square image patch.
        in_channels (`int`, *optional*, defaults to 3):
            Number of input image channels.
        vt_hidden_size (`int`, *optional*, defaults to 1024):
            Hidden size of the vision encoder.
        vt_intermediate_size (`int`, *optional*, defaults to 4096):
            Feed-forward size of the vision encoder.
        vt_num_hidden_layers (`int`, *optional*, defaults to 27):
            Number of vision encoder layers.
        vt_num_attention_heads (`int`, *optional*, defaults to 12):
            Number of vision attention heads.
        qkv_hidden_size (`int`, *optional*, defaults to 1536):
            Packed QKV projection width. Note this is not
            `vt_hidden_size`, so the per-head dim is
            `qkv_hidden_size // vt_num_attention_heads` (K3: 1536/12 = 128).
        init_pos_emb_height (`int`, *optional*, defaults to 64):
            Height of the learnable absolute position embedding table.
        init_pos_emb_width (`int`, *optional*, defaults to 64):
            Width of the learnable absolute position embedding table.
        init_pos_emb_time (`int`, *optional*, defaults to 4):
            Length of the time position embedding table.
        pos_emb_type (`str`, *optional*, defaults to `"divided_fixed"`):
            Position embedding variant; `divided_fixed` keeps the spatial and
            time tables separate.
        pos_emb_interpolation_mode (`str`, *optional*, defaults to `"bilinear"`):
            Interpolation used when resizing the spatial table to the actual grid.
        patch_embed_proj_bias (`bool`, *optional*, defaults to `False`):
            Whether the patch-embedding convolution has a bias.
        merge_kernel_size (`tuple`, *optional*, defaults to `(2, 2)`):
            Spatial merge kernel of `sd2_tpool`. One 2x2 merge means the LLM
            sees `(H/2) * (W/2)` visual tokens per media, with the time axis
            mean-pooled away.
        mm_hidden_size (`int`, *optional*, defaults to 1024):
            Per-patch width entering the projector; the projector consumes
            `merge_kernel_size[0] * merge_kernel_size[1] * mm_hidden_size`.
        text_hidden_size (`int`, *optional*, defaults to 7168):
            Output width of the projector. Must equal the text backbone
            `hidden_size`, otherwise the merged visual tokens cannot be spliced
            into the text embedding stream.
        projector_ln_eps (`float`, *optional*, defaults to 1e-5):
            Epsilon of the projector post-RMSNorm.
        max_height (`int`, *optional*, defaults to 512):
            Upper bound of the interpolated position grid height.
        max_width (`int`, *optional*, defaults to 512):
            Upper bound of the interpolated position grid width.
    """

    model_type = "kimi_k3_vision"

    # HuggingFace vision_config field name -> PaddleFleet TransformerConfig name.
    _HF_TO_FLEET_FIELD_MAP = {
        "vt_hidden_size": "hidden_size",
        "vt_intermediate_size": "intermediate_size",
        "vt_num_hidden_layers": "num_hidden_layers",
        "vt_num_attention_heads": "num_attention_heads",
    }

    def __init__(
        self,
        patch_size=14,
        in_channels=3,
        vt_hidden_size=1024,
        vt_intermediate_size=4096,
        vt_num_hidden_layers=27,
        vt_num_attention_heads=12,
        qkv_hidden_size=1536,
        init_pos_emb_height=64,
        init_pos_emb_width=64,
        init_pos_emb_time=4,
        pos_emb_type="divided_fixed",
        pos_emb_interpolation_mode="bilinear",
        patch_embed_proj_bias=False,
        merge_kernel_size=(2, 2),
        mm_hidden_size=1024,
        text_hidden_size=7168,
        projector_ln_eps=1e-5,
        max_height=512,
        max_width=512,
        **kwargs,
    ):
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.vt_hidden_size = vt_hidden_size
        self.vt_intermediate_size = vt_intermediate_size
        self.vt_num_hidden_layers = vt_num_hidden_layers
        self.vt_num_attention_heads = vt_num_attention_heads
        self.qkv_hidden_size = qkv_hidden_size
        self.init_pos_emb_height = init_pos_emb_height
        self.init_pos_emb_width = init_pos_emb_width
        self.init_pos_emb_time = init_pos_emb_time
        self.pos_emb_type = pos_emb_type
        self.pos_emb_interpolation_mode = pos_emb_interpolation_mode
        self.patch_embed_proj_bias = patch_embed_proj_bias
        self.merge_kernel_size = tuple(merge_kernel_size)
        self.mm_hidden_size = mm_hidden_size
        self.text_hidden_size = text_hidden_size
        self.projector_ln_eps = projector_ln_eps
        self.max_height = max_height
        self.max_width = max_width

        super().__init__(**kwargs)

    def to_fleet_vision_overrides(self):
        """Keyword arguments for `build_kimi_k3_vision_config`."""
        overrides = {
            "patch_size": self.patch_size,
            "in_channels": self.in_channels,
            "qkv_hidden_size": self.qkv_hidden_size,
            "init_pos_emb_height": self.init_pos_emb_height,
            "init_pos_emb_width": self.init_pos_emb_width,
            "init_pos_emb_time": self.init_pos_emb_time,
            "pos_emb_type": self.pos_emb_type,
            "pos_emb_interpolation_mode": self.pos_emb_interpolation_mode,
            "patch_embed_proj_bias": self.patch_embed_proj_bias,
            "merge_kernel_size": self.merge_kernel_size,
            "mm_hidden_size": self.mm_hidden_size,
            "text_hidden_size": self.text_hidden_size,
            "projector_ln_eps": self.projector_ln_eps,
            "max_height": self.max_height,
            "max_width": self.max_width,
        }
        for hf_name, fleet_name in self._HF_TO_FLEET_FIELD_MAP.items():
            overrides[fleet_name] = getattr(self, hf_name)
        return overrides


class KimiK3Config(PretrainedConfig):
    r"""
    Configuration class for the Kimi-K3 model wrapper.

    This class preserves the official nested configuration layout while
    exposing the Kimi-K3 text backbone through [`KimiK3TextConfig`] and the
    MoonViT3d vision tower through [`KimiK3VisionConfig`].

    Without `vision_config` the model is text-only ([`KimiK3Model`] /
    [`KimiK3ForCausalLM`]); with `vision_config` it is the full multimodal
    model ([`KimiK3ForConditionalGeneration`]).

    Args:
        text_config (`KimiK3TextConfig` or `dict`, *optional*):
            Configuration of the Kimi-K3 text backbone. A dictionary is
            converted to [`KimiK3TextConfig`]. When omitted, the default text
            configuration is created.
        vision_config (`KimiK3VisionConfig` or `dict`, *optional*):
            Configuration of the MoonViT3d vision tower. When omitted the model
            is text-only.
        media_placeholder_token_id (`int`, *optional*, defaults to 163605):
            Token id of the single media placeholder. Exactly one placeholder is
            emitted per media; the model expands it to the actual number of
            visual tokens.
        image_placeholder (`str`, *optional*, defaults to `"<|kimi_image_placeholder|>"`):
            Textual form of the media placeholder.
    """

    model_type = "kimi_k3"
    sub_configs = {"text_config": KimiK3TextConfig, "vision_config": KimiK3VisionConfig}
    is_composition = True
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        media_placeholder_token_id=163605,
        image_placeholder="<|kimi_image_placeholder|>",
        **kwargs,
    ):
        if text_config is None:
            text_config = KimiK3TextConfig()
        elif isinstance(text_config, dict):
            text_config = KimiK3TextConfig(**text_config)

        if isinstance(vision_config, dict):
            vision_config = KimiK3VisionConfig(**vision_config)

        self.text_config = text_config
        self.vision_config = vision_config
        self.media_placeholder_token_id = media_placeholder_token_id
        self.image_placeholder = image_placeholder

        if vision_config is None:
            kwargs["architectures"] = ["KimiK3ForCausalLM"]
        else:
            # Coerced, not validated: the projector output has to land in the text
            # embedding space, and a reduced text_config is a legitimate setup
            # (the official value is 7168). This is the single source of truth for
            # the field, so downstream assembly does not re-check it.
            vision_config.text_hidden_size = text_config.hidden_size
            kwargs["architectures"] = ["KimiK3ForConditionalGeneration"]

        super().__init__(**kwargs)


__all__ = [
    "KimiK3Config",
    "KimiK3TextConfig",
    "KimiK3VisionConfig",
]
