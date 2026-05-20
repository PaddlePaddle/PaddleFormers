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

"""
Base class for MoE models' AOA (Auto-Optimized Architecture) config generation.

This module provides a reusable base class for generating weight conversion
configurations in MoE (Mixture of Experts) models, supporting various features
like shared experts, dense-MoE hybrid layers, and MTP (Multi-Token Prediction).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class MoEAOAConfigParams:
    """Parameters for MoE AOA config generation.

    This dataclass holds all the configuration parameters needed to generate
    AOA (Auto-Optimized Architecture) statements for weight conversion.
    """

    # Basic model config
    num_hidden_layers: int = 0
    num_attention_heads: int = 0
    num_key_value_heads: int = 0

    # MoE specific config
    num_experts: int = 0
    using_sonic_moe: bool = False
    moe_expert_fusion: bool = False
    fp8: bool = False
    fd_fallback: bool = False

    # Embedding config
    tie_word_embeddings: bool = False

    # Layer offset config
    num_head_empty_layers: int = 0
    first_k_dense_replace: int = 0
    num_nextn_predict_layers: int = 0

    # Attention config
    attention_bias: bool = False
    multi_latent_attention: bool = False
    use_qk_norm: bool = False

    # Shared experts config
    has_shared_experts: bool = True

    # Runtime config
    model_prefix: str = "model."

    # Extra statements to add
    extra_statements: List[str] = field(default_factory=list)


class MoEAOAConfigGenerator:
    """Base class for MoE AOA config generation.

    This class provides a modular and extensible framework for generating
    weight conversion configurations. Subclasses can override specific methods
    to customize behavior for different model architectures.

    Example:
        class GlmMoeDsaAOAGenerator(MoEAOAConfigGenerator):
            def _get_attention_statements(self, params, layer_idx, prefix, prefix_offset):
                if params.multi_latent_attention:
                    return self._get_mla_attention_statements(params, prefix, prefix_offset)
                return super()._get_attention_statements(params, layer_idx, prefix, prefix_offset)
    """

    @classmethod
    def gen_aoa_config(cls, config: Any) -> Dict[str, List[str]]:
        """Main entry point for generating AOA config.

        Args:
            config: Model configuration object with necessary attributes.

        Returns:
            Dictionary with 'aoa_statements' key containing list of conversion statements.
        """
        params = cls._extract_params(config)
        return cls._build_aoa_config(params)

    @classmethod
    def _extract_params(cls, config: Any) -> MoEAOAConfigParams:
        """Extract parameters from config object.

        Subclasses can override this to add custom parameter extraction.
        """
        # Get num_experts from config
        if hasattr(config, "n_routed_experts"):
            num_experts = config.n_routed_experts
        else:
            num_experts = getattr(config, "num_experts", 0)

        return MoEAOAConfigParams(
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            num_experts=num_experts,
            using_sonic_moe=getattr(config, "using_sonic_moe", False),
            moe_expert_fusion=getattr(config, "moe_expert_fusion", False),
            fp8=getattr(config, "fp8", False),
            fd_fallback=config.get("fd_fallback", False) if hasattr(config, "get") else False,
            tie_word_embeddings=getattr(config, "tie_word_embeddings", False),
            num_head_empty_layers=(
                config.num_empty_layers_add_in_head
                if hasattr(config, "num_empty_layers_add_in_head") and config.num_empty_layers_add_in_head
                else 0
            ),
            first_k_dense_replace=getattr(config, "first_k_dense_replace", 0),
            num_nextn_predict_layers=getattr(config, "num_nextn_predict_layers", 0) or 0,
            attention_bias=getattr(config, "attention_bias", False),
            multi_latent_attention=getattr(config, "multi_latent_attention", False),
            use_qk_norm=getattr(config, "use_qk_norm", False),
            has_shared_experts=cls._has_shared_experts(config),
            model_prefix=cls._get_model_prefix(config),
        )

    @classmethod
    def _get_model_prefix(cls, config: Any) -> str:
        """Get model prefix based on class type."""
        if hasattr(cls, "base_model_class") and cls == cls.base_model_class:
            return ""
        return "model."

    @classmethod
    def _has_shared_experts(cls, config: Any) -> bool:
        """Check if model has shared experts. Override for models without shared experts."""
        return True

    @classmethod
    def _build_aoa_config(cls, params: MoEAOAConfigParams) -> Dict[str, List[str]]:
        """Build the complete AOA config from parameters."""
        aoa_statements = []

        # 1. Basic weights (norm, embed_tokens, lm_head)
        aoa_statements.extend(cls._get_basic_weight_statements(params))

        # 2. MoE layers
        aoa_statements.extend(cls._get_moe_layer_statements(params))

        # 3. MTP layers (if any)
        aoa_statements.extend(cls._get_mtp_layer_statements(params))

        # 4. Dense layers (if any)
        aoa_statements.extend(cls._get_dense_layer_statements(params))

        # 5. Grouped GEMM (if enabled)
        aoa_statements.extend(cls._get_grouped_gemm_statements(params))

        # 6. Extra statements from subclasses
        aoa_statements.extend(params.extra_statements)

        return {"aoa_statements": aoa_statements}

    # ==================== Basic Weights ====================

    @classmethod
    def _get_basic_weight_statements(cls, params: MoEAOAConfigParams) -> List[str]:
        """Generate statements for basic weights: norm, embeddings, lm_head."""
        statements = [
            f"model.norm.weight -> {params.model_prefix}norm.weight",
        ]

        # Embeddings
        statements.append(f"model.embed_tokens.weight -> {params.model_prefix}embedding.embed_tokens.weight")

        # lm_head
        if params.tie_word_embeddings:
            statements.append(f"model.embed_tokens.weight -> {params.model_prefix}lm_head.weight")
        else:
            statements.append(f"lm_head.weight -> {params.model_prefix}lm_head.weight")

        return statements

    # ==================== Dense Layers ====================

    @classmethod
    def _get_dense_layer_statements(cls, params: MoEAOAConfigParams) -> List[str]:
        """Generate statements for dense (non-MoE) layers.

        Override this method to customize dense layer handling.
        Default implementation handles first_k_dense_replace layers.
        """
        statements = []

        if params.first_k_dense_replace <= 0:
            return statements

        for layer_idx in reversed(range(0, params.first_k_dense_replace)):
            layer_idx_offset = layer_idx + params.num_head_empty_layers
            statements.extend(cls._get_single_dense_layer_statements(params, layer_idx, layer_idx_offset))

        return statements

    @classmethod
    def _get_single_dense_layer_statements(
        cls, params: MoEAOAConfigParams, layer_idx: int, layer_idx_offset: int
    ) -> List[str]:
        """Generate statements for a single dense layer."""
        prefix = f"model.layers.{layer_idx}"
        prefix_offset = f"{params.model_prefix}layers.{layer_idx_offset}"
        statements = []
        # Layer norms and attention output
        statements.extend(
            [
                f"{prefix}.input_layernorm.weight -> {prefix_offset}.input_layernorm.weight",
                f"{prefix}.post_attention_layernorm.weight -> {prefix_offset}.post_attention_layernorm.weight",
                f"{prefix}.self_attn.o_proj.weight^T -> {prefix_offset}.self_attn.o_proj.weight",
            ]
        )

        # Attention QKV (can be standard or MLA)
        statements.extend(cls._get_attention_statements(params, layer_idx, prefix, prefix_offset))

        # MLP
        statements.extend(
            [
                f"{prefix}.mlp.down_proj.weight^T -> {prefix_offset}.mlp.down_proj.weight",
                f"{prefix}.mlp.gate_proj.weight^T, {prefix}.mlp.up_proj.weight^T -> {prefix_offset}.mlp.up_gate_proj.weight, fused_ffn",
            ]
        )

        return statements

    # ==================== MTP (Multi-Token Prediction) ====================

    @classmethod
    def _get_mtp_layer_statements(cls, params: MoEAOAConfigParams) -> List[str]:
        """Generate statements for MTP layers."""
        statements = []

        if params.num_nextn_predict_layers <= 0:
            return statements

        num_hidden_layers = params.num_hidden_layers
        for layer_idx in reversed(range(num_hidden_layers, num_hidden_layers + params.num_nextn_predict_layers)):
            layer_idx_offset = layer_idx + params.num_head_empty_layers
            statements.extend(cls._get_single_mtp_layer_statements(params, layer_idx, layer_idx_offset))

        return statements

    @classmethod
    def _get_single_mtp_layer_statements(
        cls, params: MoEAOAConfigParams, layer_idx: int, layer_idx_offset: int
    ) -> List[str]:
        """Generate statements for a single MTP layer. Override for customization."""
        prefix = f"model.layers.{layer_idx}"
        prefix_offset = f"{params.model_prefix}layers.{layer_idx_offset}"

        return [
            f"{prefix}.eh_proj.weight^T -> {prefix_offset}.eh_proj.weight",
            f"{prefix}.enorm.weight -> {prefix_offset}.enorm.weight",
            f"{prefix}.hnorm.weight -> {prefix_offset}.hnorm.weight",
            f"{prefix}.shared_head.norm.weight -> {prefix_offset}.norm.weight",
        ]

    # ==================== MoE Layers ====================

    @classmethod
    def _get_moe_layer_statements(cls, params: MoEAOAConfigParams) -> List[str]:
        """Generate statements for MoE layers."""
        statements = []

        # Determine layer range
        start_layer = params.first_k_dense_replace
        end_layer = params.num_hidden_layers + params.num_nextn_predict_layers

        for layer_idx in reversed(range(start_layer, end_layer)):
            layer_idx_offset = layer_idx + params.num_head_empty_layers
            statements.extend(cls._get_single_moe_layer_statements(params, layer_idx, layer_idx_offset))

        return statements

    @classmethod
    def _get_single_moe_layer_statements(
        cls, params: MoEAOAConfigParams, layer_idx: int, layer_idx_offset: int
    ) -> List[str]:
        """Generate statements for a single MoE layer."""
        statements = []

        prefix = f"model.layers.{layer_idx}"
        prefix_offset = f"{params.model_prefix}layers.{layer_idx_offset}"

        # Handle MTP transformer layer
        if layer_idx >= params.num_hidden_layers:
            prefix_offset += ".transformer_layer"

        # Layer norms and attention output
        statements.extend(
            [
                f"{prefix}.input_layernorm.weight -> {prefix_offset}.input_layernorm.weight",
                f"{prefix}.post_attention_layernorm.weight -> {prefix_offset}.post_attention_layernorm.weight",
                f"{prefix}.self_attn.o_proj.weight^T -> {prefix_offset}.self_attn.o_proj.weight",
            ]
        )

        # Attention QKV (can be standard or MLA)
        statements.extend(cls._get_attention_statements(params, layer_idx, prefix, prefix_offset))

        # MoE specific weights
        statements.extend(cls._get_moe_expert_statements(params, prefix, prefix_offset))

        return statements

    # ==================== Attention ====================

    @classmethod
    def _get_attention_statements(
        cls, params: MoEAOAConfigParams, layer_idx: int, prefix: str, prefix_offset: str
    ) -> List[str]:
        """Generate attention-related statements.

        Override this method for different attention types (standard QKV vs MLA).
        """
        if params.multi_latent_attention:
            return cls._get_mla_attention_statements(params, prefix, prefix_offset)
        return cls._get_standard_attention_statements(params, prefix, prefix_offset)

    @classmethod
    def _get_standard_attention_statements(
        cls, params: MoEAOAConfigParams, prefix: str, prefix_offset: str
    ) -> List[str]:
        """Generate standard QKV attention statements."""
        statements = [
            f"{prefix}.self_attn.q_proj.weight^T, {prefix}.self_attn.k_proj.weight^T, {prefix}.self_attn.v_proj.weight^T -> {prefix_offset}.self_attn.qkv_proj.weight, fused_qkv, num_heads={params.num_attention_heads}, num_key_value_groups={params.num_key_value_heads}",
        ]

        if params.attention_bias:
            statements.append(
                f"{prefix}.self_attn.q_proj.bias, {prefix}.self_attn.k_proj.bias, {prefix}.self_attn.v_proj.bias -> {prefix_offset}.self_attn.qkv_proj.bias, fused_qkv, num_heads={params.num_attention_heads}, num_key_value_groups={params.num_key_value_heads}, axis=0"
            )

        return statements

    @classmethod
    def _get_mla_attention_statements(cls, params: MoEAOAConfigParams, prefix: str, prefix_offset: str) -> List[str]:
        """Generate Multi-Latent Attention (MLA) statements.

        MLA uses compressed KV representation with separate projections.
        """
        statements = [
            f"{prefix}.self_attn.kv_a_proj_with_mqa.weight^T -> {prefix_offset}.self_attn.kv_a_proj_with_mqa.weight",
            f"{prefix}.self_attn.kv_b_proj.weight^T -> {prefix_offset}.self_attn.kv_b_proj.weight",
            f"{prefix}.self_attn.q_a_proj.weight^T -> {prefix_offset}.self_attn.q_a_proj.weight",
            f"{prefix}.self_attn.q_b_proj.weight^T -> {prefix_offset}.self_attn.q_b_proj.weight",
        ]

        if params.use_qk_norm:
            statements.extend(
                [
                    f"{prefix}.self_attn.q_a_layernorm.weight -> {prefix_offset}.self_attn.q_a_layernorm.weight",
                    f"{prefix}.self_attn.kv_a_layernorm.weight -> {prefix_offset}.self_attn.kv_a_layernorm.weight",
                ]
            )

        return statements

    # ==================== MoE Expert Weights ====================

    @classmethod
    def _get_moe_expert_statements(cls, params: MoEAOAConfigParams, prefix: str, prefix_offset: str) -> List[str]:
        """Generate MoE expert weight statements."""
        statements = []

        # Gate weights
        statements.append(
            f"{prefix}.mlp.gate.e_score_correction_bias -> {prefix_offset}.mlp.gate.e_score_correction_bias"
        )
        statements.append(f"{prefix}.mlp.gate.weight -> {prefix_offset}.mlp.gate.weight, dtype='float32'")

        # Shared experts (if model has them)
        if params.has_shared_experts:
            statements.extend(cls._get_shared_expert_statements(params, prefix, prefix_offset))

        # Routed experts
        statements.extend(cls._get_routed_expert_statements(params, prefix, prefix_offset))

        return statements

    @classmethod
    def _get_shared_expert_statements(cls, params: MoEAOAConfigParams, prefix: str, prefix_offset: str) -> List[str]:
        """Generate shared expert weight statements."""
        return [
            f"{prefix}.mlp.shared_experts.down_proj.weight^T -> {prefix_offset}.mlp.shared_experts.down_proj.weight",
            f"{prefix}.mlp.shared_experts.gate_proj.weight^T, {prefix}.mlp.shared_experts.up_proj.weight^T -> {prefix_offset}.mlp.shared_experts.up_gate_proj.weight, fused_ffn",
        ]

    @classmethod
    def _get_routed_expert_statements(cls, params: MoEAOAConfigParams, prefix: str, prefix_offset: str) -> List[str]:
        """Generate routed expert weight statements."""
        statements = []

        # Down projection
        if params.using_sonic_moe:
            statements.append(
                f"{prefix}.mlp.experts.$EXPERT_ID.down_proj.weight -> {prefix_offset}.mlp.experts.$EXPERT_ID.down_proj.weight"
            )
        else:
            statements.append(
                f"{prefix}.mlp.experts.$EXPERT_ID.down_proj.weight^T -> {prefix_offset}.mlp.experts.$EXPERT_ID.down_proj.weight"
            )

        # Up and gate projection fusion
        if params.using_sonic_moe:
            statements.append(
                f"{prefix}.mlp.experts.$EXPERT_ID.gate_proj.weight, {prefix}.mlp.experts.$EXPERT_ID.up_proj.weight -> {prefix_offset}.mlp.experts.$EXPERT_ID.up_gate_proj.weight, axis=0"
            )
        else:
            statements.append(
                f"{prefix}.mlp.experts.$EXPERT_ID.gate_proj.weight^T, {prefix}.mlp.experts.$EXPERT_ID.up_proj.weight^T -> {prefix_offset}.mlp.experts.$EXPERT_ID.up_gate_proj.weight, axis=1"
            )

        return statements

    # ==================== Grouped GEMM ====================

    @classmethod
    def _get_grouped_gemm_statements(cls, params: MoEAOAConfigParams) -> List[str]:
        """Generate grouped GEMM statements for efficient MoE computation."""
        if not (params.moe_expert_fusion or params.using_sonic_moe) and not params.fp8:
            return cls._get_fd_fallback_statements(params)

        statements = []

        start_layer = params.first_k_dense_replace
        end_layer = params.num_hidden_layers + params.num_nextn_predict_layers

        for layer_idx in range(start_layer, end_layer):
            layer_idx_offset = layer_idx + params.num_head_empty_layers
            prefix_offset = f"{params.model_prefix}layers.{layer_idx_offset}"

            if layer_idx >= params.num_hidden_layers:
                prefix_offset += ".transformer_layer"

            ep_weight1 = []
            ep_weight2 = []
            for expert_id in range(params.num_experts):
                ep_weight1.append(f"{prefix_offset}.mlp.experts.{expert_id}.up_gate_proj.weight")
                ep_weight2.append(f"{prefix_offset}.mlp.experts.{expert_id}.down_proj.weight")

            group_gemm1 = ",".join(ep_weight1)
            group_gemm2 = ",".join(ep_weight2)

            statements.extend(
                [
                    f"{group_gemm1} -> {prefix_offset}.mlp.grouped_gemm_experts.weight1, axis=0",
                    f"{group_gemm2} -> {prefix_offset}.mlp.grouped_gemm_experts.weight2, axis=0",
                ]
            )

        return statements

    @classmethod
    def _get_fd_fallback_statements(cls, params: MoEAOAConfigParams) -> List[str]:
        """Generate fallback statements when grouped GEMM is not available."""
        if not params.fd_fallback:
            return []

        statements = []

        start_layer = params.first_k_dense_replace
        end_layer = params.num_hidden_layers + params.num_nextn_predict_layers

        for layer_idx in range(start_layer, end_layer):
            layer_idx_offset = layer_idx + params.num_head_empty_layers
            prefix_offset = f"{params.model_prefix}layers.{layer_idx_offset}"

            if layer_idx >= params.num_hidden_layers:
                prefix_offset += ".transformer_layer"

            ep_weight1 = []
            ep_weight2 = []
            for expert_id in range(params.num_experts):
                ep_weight1.append(f"{prefix_offset}.mlp.experts.{expert_id}.up_gate_proj.weight")
                ep_weight2.append(f"{prefix_offset}.mlp.experts.{expert_id}.down_proj.weight")

            group1 = ",".join(ep_weight1)
            group2 = ",".join(ep_weight2)

            statements.extend(
                [
                    f"{group1} -> {prefix_offset}.mlp.experts.up_gate_proj, axis=0",
                    f"{group2} -> {prefix_offset}.mlp.experts.down_proj, axis=0",
                ]
            )

        return statements

    # ==================================================================
    # Inverse AOA Config Generation (PaddleFleet -> HuggingFace)
    # ==================================================================

    @classmethod
    def gen_inv_aoa_config(cls, config: Any) -> Dict[str, List[str]]:
        """Main entry point for generating inverse AOA config.

        The inverse AOA maps PaddleFleet weight names back to HuggingFace format,
        used during save_pretrained to convert weights back to HF convention.

        Args:
            config: Model configuration object with necessary attributes.

        Returns:
            Dictionary with 'aoa_statements' key containing list of inverse conversion statements.
        """
        params = cls._extract_params(config)
        return cls._build_inv_aoa_config(params)

    @classmethod
    def _build_inv_aoa_config(cls, params: MoEAOAConfigParams) -> Dict[str, List[str]]:
        """Build the complete inverse AOA config from parameters."""
        aoa_statements = []

        # 1. Basic weights (norm, embed_tokens, lm_head)
        aoa_statements.extend(cls._get_inv_basic_weight_statements(params))

        # 2. MoE layers (attention + experts)
        aoa_statements.extend(cls._get_inv_moe_layer_statements(params))

        # 3. MTP layers (if any)
        aoa_statements.extend(cls._get_inv_mtp_layer_statements(params))

        # 4. Dense layers (if any)
        aoa_statements.extend(cls._get_inv_dense_layer_statements(params))

        # 5. Extra statements from subclasses
        aoa_statements.extend(params.extra_statements)

        return {"aoa_statements": aoa_statements}

    # ==================== Inverse Basic Weights ====================

    @classmethod
    def _get_inv_basic_weight_statements(cls, params: MoEAOAConfigParams) -> List[str]:
        """Generate inverse statements for basic weights: norm, embeddings, lm_head."""
        statements = [
            f"{params.model_prefix}norm.weight -> model.norm.weight",
            "model.embedding.embed_tokens.weight -> model.embed_tokens.weight",
        ]

        if params.tie_word_embeddings:
            statements.append(f"{params.model_prefix}lm_head.weight -> _")
        else:
            statements.append(f"{params.model_prefix}lm_head.weight -> lm_head.weight")

        return statements

    # ==================== Inverse Dense Layers ====================

    @classmethod
    def _get_inv_dense_layer_statements(cls, params: MoEAOAConfigParams) -> List[str]:
        """Generate inverse statements for dense (non-MoE) layers.

        Only handles MLP weights for dense layers. Attention weights are handled
        in _get_inv_moe_layer_statements which covers all layers uniformly.
        """
        statements = []

        if params.first_k_dense_replace <= 0:
            return statements

        for layer_idx in reversed(range(0, params.first_k_dense_replace)):
            layer_idx_offset = layer_idx + params.num_head_empty_layers
            statements.extend(cls._get_inv_single_dense_layer_statements(params, layer_idx, layer_idx_offset))

        return statements

    @classmethod
    def _get_inv_single_dense_layer_statements(
        cls, params: MoEAOAConfigParams, layer_idx: int, layer_idx_offset: int
    ) -> List[str]:
        """Generate inverse statements for a single dense layer (MLP only)."""
        prefix = f"model.layers.{layer_idx}"
        prefix_offset = f"{params.model_prefix}layers.{layer_idx_offset}"

        # MLP: un-fuse up_gate_proj -> gate_proj + up_proj, then transpose each
        return [
            f"{prefix_offset}.mlp.down_proj.weight^T -> {prefix}.mlp.down_proj.weight",
            f"{prefix_offset}.mlp.up_gate_proj.weight -> {prefix_offset}.mlp.gate_proj.weight, {prefix_offset}.mlp.up_proj.weight, fused_ffn",
            f"{prefix_offset}.mlp.gate_proj.weight^T -> {prefix}.mlp.gate_proj.weight",
            f"{prefix_offset}.mlp.up_proj.weight^T -> {prefix}.mlp.up_proj.weight",
        ]

    # ==================== Inverse MTP Layers ====================

    @classmethod
    def _get_inv_mtp_layer_statements(cls, params: MoEAOAConfigParams) -> List[str]:
        """Generate inverse statements for MTP layers."""
        statements = []

        if params.num_nextn_predict_layers <= 0:
            return statements

        num_hidden_layers = params.num_hidden_layers
        for layer_idx in reversed(range(num_hidden_layers, num_hidden_layers + params.num_nextn_predict_layers)):
            layer_idx_offset = layer_idx + params.num_head_empty_layers
            statements.extend(cls._get_inv_single_mtp_layer_statements(params, layer_idx, layer_idx_offset))

        return statements

    @classmethod
    def _get_inv_single_mtp_layer_statements(
        cls, params: MoEAOAConfigParams, layer_idx: int, layer_idx_offset: int
    ) -> List[str]:
        """Generate inverse statements for a single MTP layer. Override for customization."""
        prefix = f"model.layers.{layer_idx}"
        prefix_offset = f"{params.model_prefix}layers.{layer_idx_offset}"

        return [
            f"{prefix_offset}.eh_proj.weight^T -> {prefix}.eh_proj.weight",
            f"{prefix_offset}.enorm.weight -> {prefix}.enorm.weight",
            f"{prefix_offset}.hnorm.weight -> {prefix}.hnorm.weight",
            f"{prefix_offset}.norm.weight -> {prefix}.shared_head.norm.weight",
        ]

    # ==================== Inverse MoE Layers ====================

    @classmethod
    def _get_inv_moe_layer_statements(cls, params: MoEAOAConfigParams) -> List[str]:
        """Generate inverse statements for MoE layers (attention + experts)."""
        statements = []

        start_layer = params.first_k_dense_replace
        end_layer = params.num_hidden_layers + params.num_nextn_predict_layers

        # Attention for all layers (including dense layer 0 if first_k_dense_replace > 0)
        for layer_idx in range(0, end_layer):
            layer_idx_offset = layer_idx + params.num_head_empty_layers
            prefix = f"model.layers.{layer_idx}"
            prefix_offset = f"{params.model_prefix}layers.{layer_idx_offset}"
            if layer_idx >= params.num_hidden_layers:
                prefix_offset += ".transformer_layer"

            statements.extend(
                [
                    f"{prefix_offset}.input_layernorm.weight -> {prefix}.input_layernorm.weight",
                    f"{prefix_offset}.post_attention_layernorm.weight -> {prefix}.post_attention_layernorm.weight",
                    f"{prefix_offset}.self_attn.o_proj.weight^T -> {prefix}.self_attn.o_proj.weight",
                ]
            )
            statements.extend(cls._get_inv_attention_statements(params, layer_idx, prefix, prefix_offset))

        # MoE expert weights for layers from start_layer onward
        for layer_idx in range(start_layer, end_layer):
            layer_idx_offset = layer_idx + params.num_head_empty_layers
            prefix = f"model.layers.{layer_idx}"
            prefix_offset = f"{params.model_prefix}layers.{layer_idx_offset}"
            if layer_idx >= params.num_hidden_layers:
                prefix_offset += ".transformer_layer"

            # Grouped GEMM un-grouping (if applicable)
            statements.extend(cls._get_inv_grouped_gemm_layer_statements(params, prefix_offset))

            # MoE expert weight inversion
            statements.extend(cls._get_inv_moe_expert_statements(params, prefix, prefix_offset))

        return statements

    # ==================== Inverse Attention ====================

    @classmethod
    def _get_inv_attention_statements(
        cls, params: MoEAOAConfigParams, layer_idx: int, prefix: str, prefix_offset: str
    ) -> List[str]:
        """Generate inverse attention-related statements."""
        if params.multi_latent_attention:
            return cls._get_inv_mla_attention_statements(params, prefix, prefix_offset)
        return cls._get_inv_standard_attention_statements(params, prefix, prefix_offset)

    @classmethod
    def _get_inv_standard_attention_statements(
        cls, params: MoEAOAConfigParams, prefix: str, prefix_offset: str
    ) -> List[str]:
        """Generate inverse standard QKV attention statements.

        Un-fuse qkv_proj back to separate q/k/v projections and transpose each.
        """
        statements = [
            f"{prefix_offset}.self_attn.qkv_proj.weight -> {prefix}.self_attn.q_proj.weight, {prefix}.self_attn.k_proj.weight, {prefix}.self_attn.v_proj.weight, fused_qkv, num_heads={params.num_attention_heads}, num_key_value_groups={params.num_key_value_heads}",
        ]
        statements.extend(
            f"{prefix}.self_attn.{x}_proj.weight^T -> {prefix}.self_attn.{x}_proj.weight" for x in ("q", "k", "v")
        )

        if params.attention_bias:
            statements.append(
                f"{prefix_offset}.self_attn.qkv_proj.bias -> {prefix}.self_attn.q_proj.bias, {prefix}.self_attn.k_proj.bias, {prefix}.self_attn.v_proj.bias, fused_qkv, num_heads={params.num_attention_heads}, num_key_value_groups={params.num_key_value_heads}, axis=0"
            )

        return statements

    @classmethod
    def _get_inv_mla_attention_statements(
        cls, params: MoEAOAConfigParams, prefix: str, prefix_offset: str
    ) -> List[str]:
        """Generate inverse Multi-Latent Attention (MLA) statements."""
        statements = [
            f"{prefix_offset}.self_attn.kv_a_proj_with_mqa.weight^T -> {prefix}.self_attn.kv_a_proj_with_mqa.weight",
            f"{prefix_offset}.self_attn.kv_b_proj.weight^T -> {prefix}.self_attn.kv_b_proj.weight",
            f"{prefix_offset}.self_attn.q_a_proj.weight^T -> {prefix}.self_attn.q_a_proj.weight",
            f"{prefix_offset}.self_attn.q_b_proj.weight^T -> {prefix}.self_attn.q_b_proj.weight",
        ]

        if params.use_qk_norm:
            statements.extend(
                [
                    f"{prefix_offset}.self_attn.q_a_layernorm.weight -> {prefix}.self_attn.q_a_layernorm.weight",
                    f"{prefix_offset}.self_attn.kv_a_layernorm.weight -> {prefix}.self_attn.kv_a_layernorm.weight",
                ]
            )

        return statements

    # ==================== Inverse MoE Expert Weights ====================

    @classmethod
    def _get_inv_moe_expert_statements(cls, params: MoEAOAConfigParams, prefix: str, prefix_offset: str) -> List[str]:
        """Generate inverse MoE expert weight statements."""
        statements = []

        # Gate weights (cast back to bfloat16)
        statements.append(f"{prefix_offset}.mlp.gate.weight -> {prefix}.mlp.gate.weight, dtype='bfloat16'")
        statements.append(
            f"{prefix_offset}.mlp.gate.e_score_correction_bias -> {prefix}.mlp.gate.e_score_correction_bias"
        )

        # Shared experts (if model has them)
        if params.has_shared_experts:
            statements.extend(cls._get_inv_shared_expert_statements(params, prefix, prefix_offset))

        # Routed experts
        statements.extend(cls._get_inv_routed_expert_statements(params, prefix, prefix_offset))

        return statements

    @classmethod
    def _get_inv_shared_expert_statements(
        cls, params: MoEAOAConfigParams, prefix: str, prefix_offset: str
    ) -> List[str]:
        """Generate inverse shared expert weight statements.

        Un-fuse up_gate_proj back to gate_proj + up_proj, then transpose each.
        """
        return [
            f"{prefix_offset}.mlp.shared_experts.down_proj.weight^T -> {prefix}.mlp.shared_experts.down_proj.weight",
            f"{prefix_offset}.mlp.shared_experts.up_gate_proj.weight -> {prefix_offset}.mlp.shared_experts.gate_proj.weight, {prefix_offset}.mlp.shared_experts.up_proj.weight, fused_ffn",
            f"{prefix_offset}.mlp.shared_experts.gate_proj.weight^T -> {prefix}.mlp.shared_experts.gate_proj.weight",
            f"{prefix_offset}.mlp.shared_experts.up_proj.weight^T -> {prefix}.mlp.shared_experts.up_proj.weight",
        ]

    @classmethod
    def _get_inv_routed_expert_statements(
        cls, params: MoEAOAConfigParams, prefix: str, prefix_offset: str
    ) -> List[str]:
        """Generate inverse routed expert weight statements.

        Un-fuse up_gate_proj back to gate_proj + up_proj per expert,
        then transpose each (unless using_sonic_moe).
        """
        statements = []

        # Un-fuse up_gate_proj per expert
        if params.using_sonic_moe:
            statements.append(
                f"{prefix_offset}.mlp.experts.$EXPERT_ID.up_gate_proj.weight -> {prefix_offset}.mlp.experts.$EXPERT_ID.gate_proj.weight, {prefix_offset}.mlp.experts.$EXPERT_ID.up_proj.weight, axis=0"
            )
        else:
            statements.append(
                f"{prefix_offset}.mlp.experts.$EXPERT_ID.up_gate_proj.weight -> {prefix_offset}.mlp.experts.$EXPERT_ID.gate_proj.weight, {prefix_offset}.mlp.experts.$EXPERT_ID.up_proj.weight, axis=1"
            )

        # Transpose back (not needed for sonic_moe)
        if not params.using_sonic_moe:
            statements.extend(
                [
                    f"{prefix_offset}.mlp.experts.$EXPERT_ID.down_proj.weight^T -> {prefix}.mlp.experts.$EXPERT_ID.down_proj.weight",
                    f"{prefix_offset}.mlp.experts.$EXPERT_ID.gate_proj.weight^T -> {prefix}.mlp.experts.$EXPERT_ID.gate_proj.weight",
                    f"{prefix_offset}.mlp.experts.$EXPERT_ID.up_proj.weight^T -> {prefix}.mlp.experts.$EXPERT_ID.up_proj.weight",
                ]
            )

        return statements

    # ==================== Inverse Grouped GEMM ====================

    @classmethod
    def _get_inv_grouped_gemm_layer_statements(cls, params: MoEAOAConfigParams, prefix_offset: str) -> List[str]:
        """Generate inverse grouped GEMM statements for a single layer.

        Un-groups the consolidated weight tensors back to per-expert weights.
        """
        if (params.moe_expert_fusion or params.using_sonic_moe) and not params.fp8:
            ep_weight1 = []
            ep_weight2 = []
            for expert_id in range(params.num_experts):
                ep_weight1.append(f"{prefix_offset}.mlp.experts.{expert_id}.up_gate_proj.weight")
                ep_weight2.append(f"{prefix_offset}.mlp.experts.{expert_id}.down_proj.weight")
            group_gemm1 = ",".join(ep_weight1)
            group_gemm2 = ",".join(ep_weight2)
            return [
                f"{prefix_offset}.mlp.grouped_gemm_experts.weight1 -> {group_gemm1}, axis=0",
                f"{prefix_offset}.mlp.grouped_gemm_experts.weight2 -> {group_gemm2}, axis=0",
            ]

        if params.fd_fallback:
            ep_weight1 = []
            ep_weight2 = []
            for expert_id in range(params.num_experts):
                ep_weight1.append(f"{prefix_offset}.mlp.experts.{expert_id}.up_gate_proj.weight")
                ep_weight2.append(f"{prefix_offset}.mlp.experts.{expert_id}.down_proj.weight")
            group1 = ",".join(ep_weight1)
            group2 = ",".join(ep_weight2)
            return [
                f"{prefix_offset}.mlp.experts.up_gate_proj -> {group1}, axis=0",
                f"{prefix_offset}.mlp.experts.down_proj -> {group2}, axis=0",
            ]

        return []


__all__ = [
    "MoEAOAConfigParams",
    "MoEAOAConfigGenerator",
]
