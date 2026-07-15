# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
"""Gemma4 MoE model provider and ForCausalLM entry.

"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import paddle

from paddleformers.transformers.gpt_provider import GPTModelProvider
from paddleformers.transformers.model_utils import PretrainedModel

logger = logging.getLogger(__name__)

from paddlefleet.models.common.embeddings import Gemma4DualRotaryEmbedding
from paddlefleet.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from paddlefleet.transformer.transformer_layer import Gemma4TransformerLayer

from .configuration import Gemma4MoeConfig


def _patch_embedding_scale(embedding_layer, embed_scale):
    """Monkey-patch LanguageModelEmbedding.forward to multiply by embed_scale.

    This is needed because PipelineLayer stores GPTEmbedding under numeric
    keys, making it impossible to replace the embedding via attribute assignment.
    Instead, we patch the forward method of the inner LanguageModelEmbedding.
    """
    import types

    orig_forward = embedding_layer.forward.__func__

    def _scaled_forward(self_inner, input_ids, position_ids, tokentype_ids=None):
        result = orig_forward(self_inner, input_ids, position_ids, tokentype_ids)
        return result * embed_scale

    embedding_layer.forward = types.MethodType(_scaled_forward, embedding_layer)


class Gemma4MoePreTrainedModel(PretrainedModel):
    config_class = Gemma4MoeConfig
    base_model_prefix = "gemma4_moe"


@dataclass
class Gemma4MoeModelProvider(GPTModelProvider):
    """Provider for Gemma4 MoE model. Aligns with Megatron Gemma4ModelProvider."""

    # Override defaults for Gemma4 26B-A4B
    num_layers: int = 30
    hidden_size: int = 2816
    ffn_hidden_size: int = 2112
    num_attention_heads: int = 16
    num_query_groups: int = 8
    kv_channels: int = 256

    # Gemma4-specific
    global_head_dim: int = 512
    num_global_key_value_heads: int = 2
    layer_types: list = None

    # MoE
    n_routed_experts: int = 128
    num_moe_experts: int = 128
    moe_router_topk: int = 8
    moe_ffn_hidden_size: int = 704
    moe_shared_expert_intermediate_size: int = 2112
    moe_token_dispatcher_type: str = "alltoall"
    moe_grouped_gemm: bool = True
    moe_layer_freq: int = 1
    scoring_func: str = "sigmoid"

    # RoPE
    rotary_base: float = 10000
    sliding_window_rope_base: float = 10000.0
    full_attention_rope_base: float = 1000000.0
    global_rotary_percent: float = 0.25
    rotary_percent: float = 1.0
    rope_scaling: object = None

    # Model structure
    share_embeddings_and_output_weights: bool = True
    normalization: str = "RMSNorm"
    layernorm_epsilon: float = 1e-6
    rms_norm_eps: float = 1e-6
    gated_linear_unit: bool = True
    activation_func: str = "gelu_pytorch_tanh"
    attention_k_eq_v: bool = True
    final_logit_softcapping: float = 30.0
    scale_embeddings_by_hidden_size: bool = True
    add_swa_attention_sink_bias: bool = False
    add_full_attention_sink_bias: bool = False

    # Layer spec
    transformer_layer_spec: Callable = None

    transform_rules = {
        **GPTModelProvider.transform_rules,
        "dtype": "params_dtype",
        "num_experts": "n_routed_experts",
        "top_k_experts": "num_experts_per_tok",
        "num_hidden_layers": "num_layers",
        "num_key_value_heads": "num_query_groups",
        "head_dim": "kv_channels",
        "moe_ffn_hidden_size": "moe_intermediate_size",
    }

    def __post_init__(self):
        # Set head_dim and num_key_value_heads to sliding-layer base values
        # BEFORE super().__post_init__(), so TransformerConfig doesn't default them
        # to hidden_size//num_heads and num_attention_heads.
        # Global layers override these per-layer in Gemma4SelfAttention.__init__.
        self.head_dim = self.kv_channels  # 256 (sliding)
        self.num_key_value_heads = self.num_query_groups  # 8 (sliding)

        super().__post_init__()
        if self.transformer_layer_spec is None:
            self.transformer_layer_spec = self._get_decoder_layers_spec
        if not hasattr(self, "num_experts_per_tok") or self.num_experts_per_tok == 2:
            self.num_experts_per_tok = self.moe_router_topk
        # Gemma4 controls shared expert via moe_shared_expert_intermediate_size directly.
        # MoELayer needs n_shared_experts > 0 to create shared_experts.
        if not getattr(self, "n_shared_experts", None):
            self.n_shared_experts = 1
        # Sync num_hidden_layers from num_layers (transform_rules maps HF num_hidden_layers→num_layers)
        if self.num_hidden_layers == 1 and self.num_layers > 1:
            self.num_hidden_layers = self.num_layers
        # Convert sliding_window from int (HF config) to tuple (PaddleFleet expects tuple[int, int])
        sw = getattr(self, "sliding_window", None)
        if isinstance(sw, int):
            self.sliding_window = (sw, 0)

    def _get_decoder_layers_spec(self, config):
        """Generate layer specs for all Gemma4 layers via standard GPT path."""
        config.specific_layer = Gemma4TransformerLayer
        num_layers = getattr(config, "num_layers", 30)
        return [
            get_gpt_layer_local_spec(
                config=config,
                num_experts=None,
                use_qk_norm=True,
                normalization=getattr(config, "normalization", "RMSNorm"),
                layer_number=i,
                attention_layer_type="gemma4",
            )
            for i in range(num_layers)
        ]

    def provide(self, pre_process=None, post_process=None, vp_stage=None, loss_fn=None):
        """Build Gemma4 model using standard GPT spec path with gemma4 attention type."""
        from paddle.distributed.fleet.meta_parallel import LayerSpec, build_spec_layer
        from paddlefleet.models.common.empty_layer import EmptyLayer
        from paddlefleet.models.common.language_loss.language_loss import LanguageLoss
        from paddlefleet.models.gpt.gpt_layer_specs import get_gpt_spec

        # Build layers via standard get_gpt_layer_local_spec path
        transformer_layers_spec = self._get_decoder_layers_spec(self)

        head_empty_layers_spec = [
            LayerSpec(layer=EmptyLayer, extra_kwargs={"config": self})
            for _ in range(self.num_empty_layers_add_in_head)
        ]
        tail_empty_layers_spec = [
            LayerSpec(layer=EmptyLayer, extra_kwargs={"config": self})
            for _ in range(self.num_empty_layers_add_in_tail)
        ]

        gpt_spec = get_gpt_spec(
            config=self,
            transformer_layers_spec=transformer_layers_spec,
            mtp_layers_spec=None,
            vocab_size=self.vocab_size,
            head_empty_layers_spec=head_empty_layers_spec,
            tail_empty_layers_spec=tail_empty_layers_spec,
            max_sequence_length=self.max_sequence_length,
            position_embedding_type=self.position_embedding_type,
            rotary_percent=self.rotary_percent,
            rotary_base=self.rotary_base,
            swa_rotary_base=getattr(self, "swa_rope_theta", None),
            rope_scaling=self.rope_scaling,
            parallel_output=self.parallel_output,
            tie_word_embeddings=self.tie_word_embeddings,
        )

        pp_size = self.pipeline_model_parallel_size
        fleet_model = build_spec_layer(
            gpt_spec,
            loss_fn=loss_fn if loss_fn else LanguageLoss(self),
            num_stages=pp_size,
            seg_method="layer:Gemma4TransformerLayer|EmptyLayer",
        )

        # Convert FleetLayer GPTModel → PaddleFormers GPTModel (PretrainedModel)
        from paddleformers.transformers.gpt_provider import GPTModel

        model = GPTModel.__new__(GPTModel)
        for attr_name in dir(fleet_model):
            if not attr_name.startswith("__"):
                try:
                    setattr(model, attr_name, getattr(fleet_model, attr_name))
                except:
                    pass

        # TODO(xingmingyyj) Support context parallel
        # Replace RoPE with Dual RoPE.
        # rotary_pos_emb lives inside GPTEmbedding (model.embedding.rotary_pos_emb),
        # NOT as a top-level model attribute.
        has_emb = hasattr(model, "embedding")
        has_rpe_top = hasattr(model, "rotary_pos_emb")
        has_rpe_emb = has_emb and hasattr(model.embedding, "rotary_pos_emb")
        logger.info(
            f"[Gemma4] RoPE replacement check: has_embedding={has_emb}, "
            f"has_model.rotary_pos_emb={has_rpe_top}, "
            f"has_model.embedding.rotary_pos_emb={has_rpe_emb}"
        )
        if has_rpe_emb:
            old_rpe = model.embedding.rotary_pos_emb
            model.embedding.rotary_pos_emb = Gemma4DualRotaryEmbedding(self)
            logger.info(
                f"[Gemma4] Replaced model.embedding.rotary_pos_emb: "
                f"{type(old_rpe).__name__} -> {type(model.embedding.rotary_pos_emb).__name__}"
            )
        elif has_rpe_top:
            old_rpe = model.rotary_pos_emb
            model.rotary_pos_emb = Gemma4DualRotaryEmbedding(self)
            logger.info(
                f"[Gemma4] Replaced model.rotary_pos_emb: "
                f"{type(old_rpe).__name__} -> {type(model.rotary_pos_emb).__name__}"
            )
        else:
            # Fallback: search sublayers for GPTEmbedding with rotary_pos_emb
            logger.warning(
                "[Gemma4] Could not find rotary_pos_emb via top-level or model.embedding. " "Searching sublayers..."
            )
            for name, sublayer in model.named_sublayers():
                if hasattr(sublayer, "rotary_pos_emb") and sublayer.rotary_pos_emb is not None:
                    old_rpe = sublayer.rotary_pos_emb
                    sublayer.rotary_pos_emb = Gemma4DualRotaryEmbedding(self)
                    logger.info(
                        f"[Gemma4] Replaced {name}.rotary_pos_emb: "
                        f"{type(old_rpe).__name__} -> {type(sublayer.rotary_pos_emb).__name__}"
                    )
                    break
            else:
                logger.error("[Gemma4] FAILED to find any rotary_pos_emb to replace!")

        # Logit Softcapping: patch GPTLMHead._forward to apply tanh softcapping.
        # Cannot use Gemma4OutputLayer wrapper because GPTModel (PipelineLayer)
        # stores the LM head in its internal pipeline registry, not as a direct
        # `output_layer` attribute. Instead, find the GPTLMHead sublayer and
        # monkey-patch its _forward method.
        if self.final_logit_softcapping > 0:
            import types

            from paddlefleet.models.gpt.lm_head import GPTLMHead

            softcap = self.final_logit_softcapping

            def _make_softcapped_forward(orig_fwd, cap):
                def _forward_with_softcap(self_inner, hidden_states):
                    result = orig_fwd(self_inner, hidden_states)
                    if isinstance(result, tuple):
                        self_inner.config.fused_linear_ce_loss_chunk = 0
                        result = orig_fwd(self_inner, hidden_states)
                    return paddle.tanh(result / cap) * cap

                return _forward_with_softcap

            for name, sublayer in model.named_sublayers():
                if isinstance(sublayer, GPTLMHead):
                    orig_fwd = sublayer._forward.__func__
                    sublayer._forward = types.MethodType(_make_softcapped_forward(orig_fwd, softcap), sublayer)
                    break

        # Embedding scale: Gemma4 multiplies embeddings by sqrt(hidden_size).
        # GPTModel (PipelineLayer) stores GPTEmbedding under numeric keys in _sub_layers,
        # NOT as model.embedding. Must search via named_sublayers().
        if self.scale_embeddings_by_hidden_size:
            embed_scale = self.hidden_size**0.5
            found_emb = False

            # Try direct access first (works if PipelineLayer exposes it)
            if hasattr(model, "embedding"):
                gpt_emb = model.embedding
                if hasattr(gpt_emb, "embedding"):
                    _patch_embedding_scale(gpt_emb.embedding, embed_scale)
                    found_emb = True
                    logger.info(f"[Gemma4] Applied embedding scale √{self.hidden_size} via model.embedding")

            # Fallback: search sublayers for GPTEmbedding with .embedding
            if not found_emb:
                from paddlefleet.models.gpt.gpt_embedding import GPTEmbedding

                for name, sublayer in model.named_sublayers():
                    if isinstance(sublayer, GPTEmbedding) and hasattr(sublayer, "embedding"):
                        _patch_embedding_scale(sublayer.embedding, embed_scale)
                        found_emb = True
                        logger.info(f"[Gemma4] Applied embedding scale √{self.hidden_size} via sublayer {name}")
                        break

            if not found_emb:
                logger.error("[Gemma4] FAILED to find embedding layer for √hidden_size scaling!")

        return model


class Gemma4MoeForCausalLM(Gemma4MoePreTrainedModel):
    """Gemma4 MoE ForCausalLM entry using PaddleFleet spec mode."""

    @classmethod
    def _gen_aoa_config(cls, config):
        model_prefix = "model."
        num_hidden_layers = config.num_hidden_layers
        num_head_empty_layers = (
            config.num_empty_layers_add_in_head
            if hasattr(config, "num_empty_layers_add_in_head") and config.num_empty_layers_add_in_head
            else 0
        )
        aoa_config = {
            "aoa_statements": [
                f"model.language_model.norm.weight -> {model_prefix}norm.weight",
                f"model.language_model.embed_tokens.weight -> {model_prefix}embedding.embed_tokens.weight",
            ]
        }
        if config.tie_word_embeddings:
            aoa_config["aoa_statements"].append(
                f"model.language_model.embed_tokens.weight -> {model_prefix}lm_head.weight"
            )
        for layer_idx in range(num_hidden_layers):
            lo = layer_idx + num_head_empty_layers
            hf = f"model.language_model.layers.{layer_idx}"
            pf = f"{model_prefix}layers.{lo}"
            # Heterogeneous attention: global layers have different kv_heads
            layer_types = getattr(config, "layer_types", None)
            is_global = layer_types is not None and layer_types[layer_idx] == "full_attention"
            kv_heads = (
                getattr(config, "num_global_key_value_heads", config.num_key_value_heads)
                if is_global
                else config.num_key_value_heads
            )
            aoa_config["aoa_statements"] += [
                f"{hf}.input_layernorm.weight -> {pf}.input_layernorm.weight",
                f"{hf}.post_attention_layernorm.weight -> {pf}.post_self_attn_layernorm.weight",
                f"{hf}.pre_feedforward_layernorm.weight -> {pf}.pre_mlp_layernorm.weight",
                f"{hf}.post_feedforward_layernorm.weight -> {pf}.post_mlp_layernorm.weight",
                f"{hf}.layer_scalar -> {pf}.layer_scalar, dtype='float32'",
            ]
            # Attention: fused QKV
            # Global layers have K=V tying: HF checkpoint has no v_proj, use k_proj as V
            if is_global:
                aoa_config["aoa_statements"].append(
                    f"{hf}.self_attn.q_proj.weight^T, {hf}.self_attn.k_proj.weight^T, {hf}.self_attn.k_proj.weight^T -> {pf}.self_attn.qkv_proj.weight, fused_qkv, num_heads={config.num_attention_heads}, num_key_value_groups={kv_heads}"
                )
            else:
                aoa_config["aoa_statements"].append(
                    f"{hf}.self_attn.q_proj.weight^T, {hf}.self_attn.k_proj.weight^T, {hf}.self_attn.v_proj.weight^T -> {pf}.self_attn.qkv_proj.weight, fused_qkv, num_heads={config.num_attention_heads}, num_key_value_groups={kv_heads}"
                )
            aoa_config["aoa_statements"] += [
                f"{hf}.self_attn.o_proj.weight^T -> {pf}.self_attn.o_proj.weight",
                f"{hf}.self_attn.q_norm.weight -> {pf}.self_attn.q_norm.weight",
                f"{hf}.self_attn.k_norm.weight -> {pf}.self_attn.k_norm.weight",
                f"{hf}.mlp.gate_proj.weight^T, {hf}.mlp.up_proj.weight^T -> {pf}.mlp.shared_experts.up_gate_proj.weight, fused_ffn",
                f"{hf}.mlp.down_proj.weight^T -> {pf}.mlp.shared_experts.down_proj.weight",
                f"{hf}.post_feedforward_layernorm_1.weight -> {pf}.mlp.post_shared_expert_layernorm.weight",
                f"{hf}.pre_feedforward_layernorm_2.weight -> {pf}.mlp.pre_feedforward_layernorm_2.weight",
                f"{hf}.post_feedforward_layernorm_2.weight -> {pf}.mlp.post_moe_layernorm.weight",
                f"{hf}.router.proj.weight -> {pf}.mlp.gate.weight, dtype='float32'",
                f"{hf}.router.per_expert_scale -> {pf}.mlp.gate.routed_scaling_factor_param, dtype='float32'",
                f"{hf}.router.scale -> {pf}.mlp.gate.router_input_scale, dtype='float32'",
            ]
            # Routed experts
            # HF: experts.gate_up_proj [E, I*2, H]  ->  PF: grouped_gemm_experts.weight1 [E, H, I*2]
            # HF: experts.down_proj    [E, H, I]    ->  PF: grouped_gemm_experts.weight2 [E, I, H]
            # Both need permute="[0,2,1]" (swap last two dims within each expert slice)
            # TODO(xingmingyyj) add assert
            aoa_config["aoa_statements"] += [
                f"{hf}.experts.gate_up_proj -> {pf}.mlp.grouped_gemm_experts.weight1, permute='[0,2,1]'",
                f"{hf}.experts.down_proj -> {pf}.mlp.grouped_gemm_experts.weight2, permute='[0,2,1]'",
            ]
        return aoa_config

    @classmethod
    def _gen_inv_aoa_config(cls, config):
        """PF -> HF weight mapping for saving checkpoints."""
        model_prefix = "model."
        num_hidden_layers = config.num_hidden_layers
        num_head_empty_layers = (
            config.num_empty_layers_add_in_head
            if hasattr(config, "num_empty_layers_add_in_head") and config.num_empty_layers_add_in_head
            else 0
        )
        aoa_statements = [
            f"{model_prefix}norm.weight -> model.language_model.norm.weight",
            f"{model_prefix}embedding.embed_tokens.weight -> model.language_model.embed_tokens.weight",
        ]
        if config.tie_word_embeddings:
            aoa_statements.append(f"{model_prefix}lm_head.weight -> _")

        for layer_idx in range(num_hidden_layers):
            lo = layer_idx + num_head_empty_layers
            hf = f"model.language_model.layers.{layer_idx}"
            pf = f"{model_prefix}layers.{lo}"
            layer_types = getattr(config, "layer_types", None)
            is_global = layer_types is not None and layer_types[layer_idx] == "full_attention"
            kv_heads = (
                getattr(config, "num_global_key_value_heads", config.num_key_value_heads)
                if is_global
                else config.num_key_value_heads
            )

            # Norms
            aoa_statements += [
                f"{pf}.input_layernorm.weight -> {hf}.input_layernorm.weight",
                f"{pf}.post_self_attn_layernorm.weight -> {hf}.post_attention_layernorm.weight",
                f"{pf}.pre_mlp_layernorm.weight -> {hf}.pre_feedforward_layernorm.weight",
                f"{pf}.post_mlp_layernorm.weight -> {hf}.post_feedforward_layernorm.weight",
            ]

            # layer_scalar
            aoa_statements.append(f"{pf}.layer_scalar -> {hf}.layer_scalar, dtype='bfloat16'")

            # Attention: qkv_proj -> split q/k/v + transpose
            # Global layers (K=V tying): HF has no v_proj, skip v output
            aoa_statements.append(
                f"{pf}.self_attn.qkv_proj.weight -> {pf}.self_attn.q_proj.weight, {pf}.self_attn.k_proj.weight, {pf}.self_attn.v_proj.weight, fused_qkv, num_heads={config.num_attention_heads}, num_key_value_groups={kv_heads}"
            )
            aoa_statements += [
                f"{pf}.self_attn.q_proj.weight^T -> {hf}.self_attn.q_proj.weight",
                f"{pf}.self_attn.k_proj.weight^T -> {hf}.self_attn.k_proj.weight",
            ]
            if not is_global:
                aoa_statements.append(f"{pf}.self_attn.v_proj.weight^T -> {hf}.self_attn.v_proj.weight")
            aoa_statements += [
                f"{pf}.self_attn.o_proj.weight^T -> {hf}.self_attn.o_proj.weight",
                f"{pf}.self_attn.q_norm.weight -> {hf}.self_attn.q_norm.weight",
                f"{pf}.self_attn.k_norm.weight -> {hf}.self_attn.k_norm.weight",
            ]

            # Shared expert: up_gate_proj -> gate + up + transpose
            aoa_statements += [
                f"{pf}.mlp.shared_experts.up_gate_proj.weight -> {pf}.mlp.shared_experts.gate_proj.weight, {pf}.mlp.shared_experts.up_proj.weight, fused_ffn",
                f"{pf}.mlp.shared_experts.gate_proj.weight^T -> {hf}.mlp.gate_proj.weight",
                f"{pf}.mlp.shared_experts.up_proj.weight^T -> {hf}.mlp.up_proj.weight",
                f"{pf}.mlp.shared_experts.down_proj.weight^T -> {hf}.mlp.down_proj.weight",
            ]

            # MoE norms
            aoa_statements += [
                f"{pf}.mlp.post_shared_expert_layernorm.weight -> {hf}.post_feedforward_layernorm_1.weight",
                f"{pf}.mlp.pre_feedforward_layernorm_2.weight -> {hf}.pre_feedforward_layernorm_2.weight",
                f"{pf}.mlp.post_moe_layernorm.weight -> {hf}.post_feedforward_layernorm_2.weight",
            ]

            # Router
            aoa_statements += [
                f"{pf}.mlp.gate.weight -> {hf}.router.proj.weight, dtype='bfloat16'",
                f"{pf}.mlp.gate.routed_scaling_factor_param -> {hf}.router.per_expert_scale, dtype='bfloat16'",
                f"{pf}.mlp.gate.router_input_scale -> {hf}.router.scale, dtype='bfloat16'",
            ]

            # Routed experts (inverse)
            # PF: grouped_gemm_experts.weight1 [E, H, I*2] -> HF: experts.gate_up_proj [E, I*2, H]
            # PF: grouped_gemm_experts.weight2 [E, I, H]   -> HF: experts.down_proj    [E, H, I]
            # TODO(xingmingyyj) add assert
            aoa_statements += [
                f"{pf}.mlp.grouped_gemm_experts.weight1 -> {hf}.experts.gate_up_proj, permute='[0,2,1]'",
                f"{pf}.mlp.grouped_gemm_experts.weight2 -> {hf}.experts.down_proj, permute='[0,2,1]'",
            ]

        return {"aoa_statements": aoa_statements}

    is_fleet = True

    def __new__(cls, config):
        config.tensor_model_parallel_size = max(getattr(config, "tensor_model_parallel_size", 1), 1)
        config.pipeline_model_parallel_size = max(getattr(config, "pipeline_model_parallel_size", 1), 1)
        config.expert_model_parallel_size = max(getattr(config, "expert_model_parallel_size", 1), 1)
        config.context_parallel_size = max(getattr(config, "context_parallel_size", 1), 1)
        config.virtual_pipeline_model_parallel_size = max(
            getattr(config, "virtual_pipeline_model_parallel_size", 1), 1
        )

        model_provider = Gemma4MoeModelProvider.from_config(config)
        gpt_model = model_provider.provide()
        gpt_model._gen_aoa_config = cls._gen_aoa_config
        gpt_model._gen_inv_aoa_config = cls._gen_inv_aoa_config
        gpt_model.config_to_save = config
        gpt_model.is_fleet = cls.is_fleet
        return gpt_model
