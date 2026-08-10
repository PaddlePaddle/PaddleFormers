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

import types
from dataclasses import dataclass

import paddle
from paddle.distributed import fleet
from paddlefleet.models.kimi_k3 import (
    build_kimi_k3_vision_config,
    kimi_k3_vision_builder,
)

from ...nn.criterion.interface import CriterionLayer
from ...nn.pp_model import GeneralModelForCausalLMPipe
from ..gpt_provider import GPTModelProvider
from ..model_utils import PretrainedModel
from .configuration import KimiK3Config, KimiK3TextConfig


@dataclass
class KimiK3ModelProvider(GPTModelProvider):
    """Kimi-K3 configuration provider for PaddleFleet GPTModel.

    Consumes the KDA/MLA schedule and flat ``linear_*`` KDA fields resolved by
    ``KimiK3TextConfig`` and adapts the block attention residual size to the
    per-sublayer count Fleet expects.
    """

    # === Kimi-K3 required defaults ===
    multi_latent_attention: bool = True
    gated_attention: bool = True
    use_qk_norm: bool = True
    qk_norm_eps: float = 1e-6
    gated_linear_unit: bool = True
    normalization: str = "RMSNorm"

    # KDA/MLA hybrid attention schedule
    linear_attn_config: dict | None = None
    block_attention_residuals: bool = True

    # General defaults
    share_embeddings_and_output_weights: bool = False

    transform_rules = {
        **GPTModelProvider.transform_rules,
        "dtype": "params_dtype",
        # HF config.json -> Fleet TransformerConfig field mappings
        **KimiK3TextConfig._HF_TO_FLEET_FIELD_MAP,
    }


class KimiK3PretrainedModel(PretrainedModel):
    config_class = KimiK3Config
    base_model_prefix = "model"

    @staticmethod
    def _is_moe_layer(config, layer_idx):
        """Whether decoder layer ``layer_idx`` (zero-based) uses a MoE MLP."""
        frequency = getattr(config, "moe_layer_freq", 1)
        if isinstance(frequency, (list, tuple)):
            return bool(frequency[layer_idx])
        first_dense = getattr(config, "first_k_dense_replace", 0) or 0
        if layer_idx < first_dense:
            return False
        if first_dense:
            return not frequency or (layer_idx - first_dense + 1) % frequency == 0
        return layer_idx % frequency == 0

    @classmethod
    def _gen_aoa_config(cls, config):
        """Map the official Kimi-K3 HuggingFace checkpoint to Fleet GPT."""
        if hasattr(config, "get_text_config"):
            config = config.get_text_config()

        num_layers = config.num_hidden_layers
        num_experts = config.n_routed_experts
        num_mtp_layers = getattr(config, "num_nextn_predict_layers", 0) or 0
        params_dtype = getattr(config, "params_dtype", getattr(config, "dtype", "bfloat16"))
        layer_types = config.layer_types

        src_model = "language_model.model"
        statements = [
            f"{src_model}.embed_tokens.weight -> model.embedding.embed_tokens.weight",
            f"{src_model}.norm.weight -> model.norm.weight",
            "language_model.lm_head.weight -> model.lm_head.weight",
            f"{src_model}.output_attn_res_proj.weight -> model.output_attn_res.block_attn_res.proj_weight",
            f"{src_model}.output_attn_res_norm.weight -> model.output_attn_res.block_attn_res.norm.weight",
        ]

        def add_attention(src, dst, attention_type):
            statements.extend(
                [
                    f"{src}.input_layernorm.weight -> {dst}.input_layernorm.weight",
                    f"{src}.post_attention_layernorm.weight -> {dst}.post_attention_layernorm.weight",
                ]
            )
            if attention_type == "kimi_delta_attention":
                in_proj_sources = [
                    f"{src}.self_attn.q_proj.weight^T",
                    f"{src}.self_attn.k_proj.weight^T",
                    f"{src}.self_attn.v_proj.weight^T",
                    f"{src}.self_attn.b_proj.weight^T",
                ]
                if config.linear_use_full_rank_gate:
                    in_proj_sources.append(f"{src}.self_attn.g_proj.weight^T")
                else:
                    statements.extend(
                        [
                            f"{src}.self_attn.g_a_proj.weight^T -> {dst}.self_attn.g_a_proj.weight",
                            f"{src}.self_attn.g_b_proj.weight^T -> {dst}.self_attn.g_b_proj.weight",
                        ]
                    )
                statements.extend(
                    [
                        f"{','.join(in_proj_sources)} -> {dst}.self_attn.in_proj.weight, axis=1",
                        f"{src}.self_attn.f_a_proj.weight^T -> {dst}.self_attn.f_a_proj.weight",
                        f"{src}.self_attn.f_b_proj.weight^T -> {dst}.self_attn.f_b_proj.weight",
                        f"{src}.self_attn.q_conv1d.weight,{src}.self_attn.k_conv1d.weight,"
                        f"{src}.self_attn.v_conv1d.weight -> {src}.self_attn.conv1d_fused, axis=0",
                        f"{src}.self_attn.conv1d_fused -> {dst}.self_attn.conv1d.weight, dtype='float32'",
                        f"{src}.self_attn.A_log -> {dst}.self_attn.A_log, dtype='float32'",
                        f"{src}.self_attn.dt_bias -> {dst}.self_attn.dt_bias, dtype='float32'",
                        f"{src}.self_attn.o_norm.weight -> {dst}.self_attn.out_norm.weight, dtype='{params_dtype}'",
                        f"{src}.self_attn.o_proj.weight^T -> {dst}.self_attn.out_proj.weight",
                    ]
                )
            elif attention_type == "multi_latent_attention":
                statements.extend(
                    [
                        f"{src}.self_attn.q_a_proj.weight^T -> {dst}.self_attn.q_a_proj.weight",
                        f"{src}.self_attn.q_b_proj.weight^T -> {dst}.self_attn.q_b_proj.weight",
                        f"{src}.self_attn.kv_a_proj_with_mqa.weight^T -> {dst}.self_attn.kv_a_proj_with_mqa.weight",
                        f"{src}.self_attn.kv_b_proj.weight^T -> {dst}.self_attn.kv_b_proj.weight",
                        f"{src}.self_attn.q_a_layernorm.weight -> {dst}.self_attn.q_a_layernorm.weight",
                        f"{src}.self_attn.kv_a_layernorm.weight -> {dst}.self_attn.kv_a_layernorm.weight",
                        f"{src}.self_attn.g_proj.weight^T -> {dst}.self_attn.gate_proj.weight",
                        f"{src}.self_attn.o_proj.weight^T -> {dst}.self_attn.o_proj.weight",
                    ]
                )
            else:
                raise ValueError(f"Unsupported Kimi-K3 attention layer type: {attention_type}")

        def add_attention_residual(src, dst):
            statements.extend(
                [
                    f"{src}.self_attention_res_proj.weight -> {dst}.block_attn_res_before_attention.proj_weight",
                    f"{src}.self_attention_res_norm.weight -> {dst}.block_attn_res_before_attention.norm.weight",
                    f"{src}.mlp_res_proj.weight -> {dst}.block_attn_res_before_mlp.proj_weight",
                    f"{src}.mlp_res_norm.weight -> {dst}.block_attn_res_before_mlp.norm.weight",
                ]
            )

        def add_dense_mlp(src, dst):
            statements.extend(
                [
                    f"{src}.mlp.gate_proj.weight^T,{src}.mlp.up_proj.weight^T "
                    f"-> {dst}.mlp.up_gate_proj.weight, fused_ffn",
                    f"{src}.mlp.down_proj.weight^T -> {dst}.mlp.down_proj.weight",
                ]
            )

        def add_moe(src, dst):
            src_moe = f"{src}.block_sparse_moe"
            dst_moe = f"{dst}.mlp"
            statements.extend(
                [
                    f"{src_moe}.gate.weight -> {dst_moe}.gate.weight, dtype='float32'",
                    f"{src_moe}.gate.e_score_correction_bias -> {dst_moe}.gate.e_score_correction_bias",
                    f"{src_moe}.routed_expert_down_proj.weight^T -> {dst_moe}.fc1_latent_proj.weight",
                    f"{src_moe}.routed_expert_up_proj.weight^T -> {dst_moe}.fc2_latent_proj.weight",
                    f"{src_moe}.routed_expert_norm.weight -> {dst_moe}.latent_norm.weight",
                ]
            )
            if getattr(config, "topk_method", None) == "quantile_balancing":
                statements.extend(
                    [
                        f"_ -> {dst_moe}.gate.qb_bin_min",
                        f"_ -> {dst_moe}.gate.qb_bin_max",
                    ]
                )
            for expert_idx in range(num_experts):
                src_expert = f"{src_moe}.experts.{expert_idx}"
                dst_expert = f"{dst_moe}.experts.{expert_idx}"
                statements.extend(
                    [
                        f"{src_expert}.w1.weight^T,{src_expert}.w3.weight^T "
                        f"-> {dst_expert}.up_gate_proj.weight, axis=1",
                        f"{src_expert}.w2.weight^T -> {dst_expert}.down_proj.weight",
                    ]
                )
            if getattr(config, "n_shared_experts", 0) > 0:
                statements.extend(
                    [
                        f"{src_moe}.shared_experts.gate_proj.weight^T,"
                        f"{src_moe}.shared_experts.up_proj.weight^T "
                        f"-> {dst_moe}.shared_experts.up_gate_proj.weight, fused_ffn",
                        f"{src_moe}.shared_experts.down_proj.weight^T -> {dst_moe}.shared_experts.down_proj.weight",
                    ]
                )
            if getattr(config, "moe_expert_fusion", False):
                weight1 = ",".join(
                    f"{dst_moe}.experts.{expert_idx}.up_gate_proj.weight" for expert_idx in range(num_experts)
                )
                weight2 = ",".join(
                    f"{dst_moe}.experts.{expert_idx}.down_proj.weight" for expert_idx in range(num_experts)
                )
                statements.extend(
                    [
                        f"{weight1} -> {dst_moe}.grouped_gemm_experts.weight1, axis=0",
                        f"{weight2} -> {dst_moe}.grouped_gemm_experts.weight2, axis=0",
                    ]
                )

        for layer_idx, attention_type in enumerate(layer_types):
            src = f"{src_model}.layers.{layer_idx}"
            dst = f"model.layers.{layer_idx}"
            add_attention(src, dst, attention_type)
            add_attention_residual(src, dst)
            if cls._is_moe_layer(config, layer_idx):
                add_moe(src, dst)
            else:
                add_dense_mlp(src, dst)

        # The released HF checkpoint has no MTP weights. Initialise its
        # projection/norms normally and seed compatible transformer weights
        # from the final decoder layer. Fleet builds the MTP attention from
        # the global attention setting, so a hybrid Kimi config may produce a
        # regular self-attention block that has no compatible HF source.
        if num_mtp_layers:
            src = f"{src_model}.layers.{num_layers - 1}"
            for mtp_idx in range(num_mtp_layers):
                layer_idx = num_layers + mtp_idx
                mtp = f"model.layers.{layer_idx}"
                dst = f"{mtp}.transformer_layer"
                statements.extend(
                    [
                        f"_ -> {mtp}.enorm.weight",
                        f"_ -> {mtp}.hnorm.weight",
                        f"_ -> {mtp}.eh_proj.weight",
                        f"_ -> {mtp}.norm.weight",
                    ]
                )
                if getattr(config, "multi_latent_attention", False):
                    add_attention(src, dst, "multi_latent_attention")
                else:
                    statements.extend(
                        [
                            f"{src}.input_layernorm.weight -> {dst}.input_layernorm.weight",
                            f"{src}.post_attention_layernorm.weight -> {dst}.post_attention_layernorm.weight",
                            f"_ -> {dst}.self_attn.qkv_proj.weight",
                            f"_ -> {dst}.self_attn.q_norm.weight",
                            f"_ -> {dst}.self_attn.k_norm.weight",
                            f"_ -> {dst}.self_attn.o_proj.weight",
                        ]
                    )
                if cls._is_moe_layer(config, num_layers - 1):
                    add_moe(src, dst)
                else:
                    add_dense_mlp(src, dst)

        return {"aoa_statements": statements}

    @classmethod
    def _gen_inv_aoa_config(cls, config):
        """Map Fleet GPT weights back to the official Kimi-K3 HF schema."""
        if hasattr(config, "get_text_config"):
            config = config.get_text_config()

        num_layers = config.num_hidden_layers
        num_experts = config.n_routed_experts
        num_mtp_layers = getattr(config, "num_nextn_predict_layers", 0) or 0
        layer_types = config.layer_types
        if getattr(config, "moe_expert_fusion", False):
            raise ValueError("Kimi-K3 HF export does not support fused expert weights.")

        hf_model = "language_model.model"
        statements = [
            f"model.embedding.embed_tokens.weight -> {hf_model}.embed_tokens.weight",
            f"model.norm.weight -> {hf_model}.norm.weight",
            "model.lm_head.weight -> language_model.lm_head.weight",
            f"model.output_attn_res.block_attn_res.proj_weight -> {hf_model}.output_attn_res_proj.weight",
            f"model.output_attn_res.block_attn_res.norm.weight -> {hf_model}.output_attn_res_norm.weight",
        ]

        def add_kda(src, dst):
            head_dim = config.linear_key_head_dim
            use_full_rank_gate = config.linear_use_full_rank_gate
            num_chunks = (4 if use_full_rank_gate else 3) * head_dim + 1
            chunks = [f"aoa_tmp.kda.{src}.in_proj.{idx}" for idx in range(num_chunks)]
            statements.append(f"{src}.self_attn.in_proj.weight -> {','.join(chunks)}, axis=1")

            offset = 0
            for name in ("q", "k", "v"):
                component = f"aoa_tmp.kda.{src}.{name}_proj.weight"
                statements.extend(
                    [
                        f"{','.join(chunks[offset : offset + head_dim])} -> {component}, axis=1",
                        f"{component}^T -> {dst}.self_attn.{name}_proj.weight",
                    ]
                )
                offset += head_dim
            statements.append(f"{chunks[offset]}^T -> {dst}.self_attn.b_proj.weight")
            offset += 1
            if use_full_rank_gate:
                component = f"aoa_tmp.kda.{src}.g_proj.weight"
                statements.extend(
                    [
                        f"{','.join(chunks[offset:])} -> {component}, axis=1",
                        f"{component}^T -> {dst}.self_attn.g_proj.weight",
                    ]
                )
            else:
                statements.extend(
                    [
                        f"{src}.self_attn.g_a_proj.weight^T -> {dst}.self_attn.g_a_proj.weight",
                        f"{src}.self_attn.g_b_proj.weight^T -> {dst}.self_attn.g_b_proj.weight",
                    ]
                )

            conv_parts = [f"aoa_tmp.kda.{src}.{name}_conv1d.weight" for name in ("q", "k", "v")]
            statements.extend(
                [
                    f"{src}.self_attn.f_a_proj.weight^T -> {dst}.self_attn.f_a_proj.weight",
                    f"{src}.self_attn.f_b_proj.weight^T -> {dst}.self_attn.f_b_proj.weight",
                    f"{src}.self_attn.conv1d.weight -> {','.join(conv_parts)}, axis=0",
                    f"{conv_parts[0]} -> {dst}.self_attn.q_conv1d.weight",
                    f"{conv_parts[1]} -> {dst}.self_attn.k_conv1d.weight",
                    f"{conv_parts[2]} -> {dst}.self_attn.v_conv1d.weight",
                    f"{src}.self_attn.A_log -> {dst}.self_attn.A_log, dtype='float32'",
                    f"{src}.self_attn.dt_bias -> {dst}.self_attn.dt_bias, dtype='float32'",
                    f"{src}.self_attn.out_norm.weight -> {dst}.self_attn.o_norm.weight, dtype='float32'",
                    f"{src}.self_attn.out_proj.weight^T -> {dst}.self_attn.o_proj.weight",
                ]
            )

        def add_attention(src, dst, attention_type):
            statements.extend(
                [
                    f"{src}.input_layernorm.weight -> {dst}.input_layernorm.weight",
                    f"{src}.post_attention_layernorm.weight -> {dst}.post_attention_layernorm.weight",
                ]
            )
            if attention_type == "kimi_delta_attention":
                add_kda(src, dst)
            elif attention_type == "multi_latent_attention":
                statements.extend(
                    [
                        f"{src}.self_attn.q_a_proj.weight^T -> {dst}.self_attn.q_a_proj.weight",
                        f"{src}.self_attn.q_b_proj.weight^T -> {dst}.self_attn.q_b_proj.weight",
                        f"{src}.self_attn.kv_a_proj_with_mqa.weight^T -> {dst}.self_attn.kv_a_proj_with_mqa.weight",
                        f"{src}.self_attn.kv_b_proj.weight^T -> {dst}.self_attn.kv_b_proj.weight",
                        f"{src}.self_attn.q_a_layernorm.weight -> {dst}.self_attn.q_a_layernorm.weight",
                        f"{src}.self_attn.kv_a_layernorm.weight -> {dst}.self_attn.kv_a_layernorm.weight",
                        f"{src}.self_attn.gate_proj.weight^T -> {dst}.self_attn.g_proj.weight",
                        f"{src}.self_attn.o_proj.weight^T -> {dst}.self_attn.o_proj.weight",
                    ]
                )
            else:
                raise ValueError(f"Unsupported Kimi-K3 attention layer type: {attention_type}")

        def add_attention_residual(src, dst):
            statements.extend(
                [
                    f"{src}.block_attn_res_before_attention.proj_weight -> {dst}.self_attention_res_proj.weight",
                    f"{src}.block_attn_res_before_attention.norm.weight -> {dst}.self_attention_res_norm.weight",
                    f"{src}.block_attn_res_before_mlp.proj_weight -> {dst}.mlp_res_proj.weight",
                    f"{src}.block_attn_res_before_mlp.norm.weight -> {dst}.mlp_res_norm.weight",
                ]
            )

        def add_dense_mlp(src, dst):
            gate = f"aoa_tmp.dense.{src}.gate_proj.weight"
            up = f"aoa_tmp.dense.{src}.up_proj.weight"
            statements.extend(
                [
                    f"{src}.mlp.up_gate_proj.weight -> {gate},{up}, axis=1",
                    f"{gate}^T -> {dst}.mlp.gate_proj.weight",
                    f"{up}^T -> {dst}.mlp.up_proj.weight",
                    f"{src}.mlp.down_proj.weight^T -> {dst}.mlp.down_proj.weight",
                ]
            )

        def add_moe(src, dst):
            src_moe = f"{src}.mlp"
            dst_moe = f"{dst}.block_sparse_moe"
            statements.extend(
                [
                    f"{src_moe}.gate.weight -> {dst_moe}.gate.weight, dtype='bfloat16'",
                    f"{src_moe}.gate.e_score_correction_bias -> {dst_moe}.gate.e_score_correction_bias",
                    f"{src_moe}.fc1_latent_proj.weight^T -> {dst_moe}.routed_expert_down_proj.weight",
                    f"{src_moe}.fc2_latent_proj.weight^T -> {dst_moe}.routed_expert_up_proj.weight",
                    f"{src_moe}.latent_norm.weight -> {dst_moe}.routed_expert_norm.weight",
                ]
            )
            if getattr(config, "topk_method", None) == "quantile_balancing":
                statements.extend(
                    [
                        f"{src_moe}.gate.qb_bin_min -> _",
                        f"{src_moe}.gate.qb_bin_max -> _",
                    ]
                )
            for expert_idx in range(num_experts):
                src_expert = f"{src_moe}.experts.{expert_idx}"
                dst_expert = f"{dst_moe}.experts.{expert_idx}"
                w1 = f"aoa_tmp.moe.{src_expert}.w1.weight"
                w3 = f"aoa_tmp.moe.{src_expert}.w3.weight"
                statements.extend(
                    [
                        f"{src_expert}.up_gate_proj.weight -> {w1},{w3}, axis=1",
                        f"{w1}^T -> {dst_expert}.w1.weight",
                        f"{w3}^T -> {dst_expert}.w3.weight",
                        f"{src_expert}.down_proj.weight^T -> {dst_expert}.w2.weight",
                    ]
                )
            if getattr(config, "n_shared_experts", 0) > 0:
                shared = f"{src_moe}.shared_experts"
                gate = f"aoa_tmp.moe.{shared}.gate_proj.weight"
                up = f"aoa_tmp.moe.{shared}.up_proj.weight"
                statements.extend(
                    [
                        f"{shared}.up_gate_proj.weight -> {gate},{up}, axis=1",
                        f"{gate}^T -> {dst_moe}.shared_experts.gate_proj.weight",
                        f"{up}^T -> {dst_moe}.shared_experts.up_proj.weight",
                        f"{shared}.down_proj.weight^T -> {dst_moe}.shared_experts.down_proj.weight",
                    ]
                )

        for layer_idx, attention_type in reversed(list(enumerate(layer_types))):
            src = f"model.layers.{layer_idx}"
            dst = f"{hf_model}.layers.{layer_idx}"
            add_attention(src, dst, attention_type)
            add_attention_residual(src, dst)
            if cls._is_moe_layer(config, layer_idx):
                add_moe(src, dst)
            else:
                add_dense_mlp(src, dst)

        # MTP is a training-only extension for this integration; the released
        # Kimi-K3 HF schema has no MTP tensors, so do not leak Fleet names into
        # an otherwise reloadable HF checkpoint.
        for mtp_idx in range(num_mtp_layers):
            layer_idx = num_layers + mtp_idx
            mtp = f"model.layers.{layer_idx}"
            transformer = f"{mtp}.transformer_layer"
            mtp_keys = [
                f"{mtp}.enorm.weight",
                f"{mtp}.hnorm.weight",
                f"{mtp}.eh_proj.weight",
                f"{mtp}.norm.weight",
                f"{transformer}.input_layernorm.weight",
                f"{transformer}.post_attention_layernorm.weight",
                f"{transformer}.block_attn_res_before_attention.proj_weight",
                f"{transformer}.block_attn_res_before_attention.norm.weight",
                f"{transformer}.block_attn_res_before_mlp.proj_weight",
                f"{transformer}.block_attn_res_before_mlp.norm.weight",
            ]
            if getattr(config, "multi_latent_attention", False):
                mtp_keys.extend(
                    f"{transformer}.self_attn.{name}"
                    for name in (
                        "q_a_proj.weight",
                        "q_b_proj.weight",
                        "kv_a_proj_with_mqa.weight",
                        "kv_b_proj.weight",
                        "q_a_layernorm.weight",
                        "kv_a_layernorm.weight",
                        "gate_proj.weight",
                        "o_proj.weight",
                    )
                )
            else:
                mtp_keys.extend(
                    f"{transformer}.self_attn.{name}"
                    for name in ("qkv_proj.weight", "q_norm.weight", "k_norm.weight", "o_proj.weight")
                )
            if cls._is_moe_layer(config, num_layers - 1):
                mlp = f"{transformer}.mlp"
                mtp_keys.extend(
                    [
                        f"{mlp}.gate.weight",
                        f"{mlp}.gate.e_score_correction_bias",
                        f"{mlp}.fc1_latent_proj.weight",
                        f"{mlp}.fc2_latent_proj.weight",
                        f"{mlp}.latent_norm.weight",
                    ]
                )
                if getattr(config, "topk_method", None) == "quantile_balancing":
                    mtp_keys.extend(
                        [
                            f"{mlp}.gate.qb_bin_min",
                            f"{mlp}.gate.qb_bin_max",
                        ]
                    )
                for expert_idx in range(num_experts):
                    mtp_keys.extend(
                        [
                            f"{mlp}.experts.{expert_idx}.up_gate_proj.weight",
                            f"{mlp}.experts.{expert_idx}.down_proj.weight",
                        ]
                    )
                if getattr(config, "n_shared_experts", 0) > 0:
                    mtp_keys.extend(
                        [
                            f"{mlp}.shared_experts.up_gate_proj.weight",
                            f"{mlp}.shared_experts.down_proj.weight",
                        ]
                    )
            else:
                mtp_keys.extend(
                    [
                        f"{transformer}.mlp.up_gate_proj.weight",
                        f"{transformer}.mlp.down_proj.weight",
                    ]
                )
            statements.extend(f"{key} -> _" for key in mtp_keys)

        return {"aoa_statements": statements}


def _build_text_model(model_class, config):
    text_config = config.get_text_config()

    # Parallelism config safeguards
    text_config.tensor_model_parallel_size = max(getattr(text_config, "tensor_model_parallel_size", 1), 1)
    text_config.context_parallel_size = max(getattr(text_config, "context_parallel_size", 1), 1)
    text_config.pipeline_model_parallel_size = max(getattr(text_config, "pipeline_model_parallel_size", 1), 1)
    text_config.virtual_pipeline_model_parallel_size = max(
        getattr(text_config, "virtual_pipeline_model_parallel_size", 1), 1
    )
    text_config.expert_model_parallel_size = max(getattr(text_config, "expert_model_parallel_size", 1), 1)

    model_provider = KimiK3ModelProvider.from_config(text_config)
    gpt_model = model_provider.provide()
    gpt_model.config_to_save = config
    gpt_model.is_fleet = model_class.is_fleet
    gpt_model._gen_aoa_config = model_class._gen_aoa_config
    gpt_model._gen_inv_aoa_config = model_class._gen_inv_aoa_config
    return gpt_model


class KimiK3Model(KimiK3PretrainedModel):
    """AutoModel-compatible alias for the Kimi-K3 text decoder."""

    is_fleet = True

    def __new__(cls, config):
        return _build_text_model(cls, config)


class KimiK3ForCausalLM(KimiK3PretrainedModel):
    """Kimi-K3 text-only causal language model."""

    is_fleet = True

    def __new__(cls, config):
        return _build_text_model(cls, config)


class KimiK3ForCausalLMPipe(KimiK3PretrainedModel, GeneralModelForCausalLMPipe):
    """Pipeline alias for the Kimi-K3 text-only model."""

    is_fleet = True

    def __new__(cls, config):
        return _build_text_model(cls, config)


def build_kimi_k3_vision_tower(vision_config, params_dtype=None):
    """Build the MoonViT3d tower from a :class:`KimiK3VisionConfig`.

    ``params_dtype`` must match the text backbone: the projector output is
    spliced into the text embedding stream, and the patch-embed conv would
    otherwise hit a dtype mismatch against ``pixel_values``.
    """
    overrides = vision_config.to_fleet_vision_overrides()
    if params_dtype is not None:
        overrides["params_dtype"] = params_dtype
    fleet_config = build_kimi_k3_vision_config(**overrides)
    tower = kimi_k3_vision_builder(
        fleet_config,
        seg_method="layer:TransformerLayer|EmptyLayer",
        num_stages=fleet_config.pipeline_model_parallel_size,
    )
    return tower, fleet_config


# Batch keys forwarded to pipeline stage 0. The scheduler only ships what is
# declared here and a missing key does not raise: without ``pixel_values`` the
# vision tower never runs and training silently degrades to text-only.
_PIPELINE_FIRST_STAGE_KEYS = [
    "input_ids",
    "attention_mask",
    "attn_mask_startend_row_indices",
    "position_ids",
    "pixel_values",
    "image_grid_thw",
]


class KimiK3CriterionPipe(CriterionLayer):
    """``CriterionLayer`` for the pipeline last stage.

    ``GPTLMHead.forward`` returns ``[main_logits, mtp_logits...]`` with MTP enabled,
    but the scheduler asserts ``loss_fn`` returns a single Tensor and never forwards
    ``mtp_logits`` itself.
    """

    def forward(self, logits, labels, loss_mask=None, **kwargs):
        if isinstance(logits, list):
            return super().forward(logits[0], labels, loss_mask, mtp_logits=logits[1:], **kwargs)
        return super().forward(logits, labels, loss_mask, **kwargs)


def _prepare_kimi_k3_pipeline_inputs(inputs, gather_pp_need_data=True):
    """Split a batch into stage-0 inputs and last-stage labels.

    ``gather_pp_need_data`` is accepted for signature compatibility only; this
    function always returns the ``(inputs, labels)`` tuple.
    """
    if isinstance(inputs, dict):
        first_stage_batch = {k: inputs[k] for k in _PIPELINE_FIRST_STAGE_KEYS if k in inputs}
        return (first_stage_batch, inputs.get("labels", None))

    first_stage_batch = {}
    for key in _PIPELINE_FIRST_STAGE_KEYS:
        values = [data.get(key, None) for data in inputs]
        if any(value is not None for value in values):
            first_stage_batch[key] = values
    return (first_stage_batch, [data.get("labels", None) for data in inputs])


class KimiK3VisionMergeLayer(paddle.nn.Layer):
    """Stage-0 layer that runs the MoonViT3d tower and hands the features to
    ``GPTEmbedding`` as ``image_embeds``.

    The attribute names are load bearing: Fleet's ``is_vision_merge_key`` matches the
    ``vision_merge.vision_model.`` prefix to keep these parameters out of the pipeline
    stage numbering and remap them in ``state_dict`` / ``sharded_state_dict``.
    """

    def __init__(self, vision_model):
        super().__init__()
        self.vision_model = vision_model

    def forward(self, dict_args):
        pixel_values = dict_args.get("pixel_values", None)
        if pixel_values is not None:
            grid_thws = dict_args.get("image_grid_thw", None)
            if grid_thws is None:
                raise ValueError(
                    "pixel_values were provided without `image_grid_thw`; the Kimi-K3 "
                    "vision tower needs the per-image [T, H, W] patch grid."
                )
            output = self.vision_model.forward({"pixel_values": pixel_values, "grid_thws": grid_thws})
            features = output["hidden_states"]
            if not isinstance(features, (list, tuple)):
                features = [features]
            dict_args["image_embeds"] = paddle.concat(
                [feature.reshape([-1, feature.shape[-1]]) for feature in features], axis=0
            )
        for key in ("pixel_values", "image_grid_thw"):
            dict_args.pop(key, None)
        return dict_args


def _build_vl_model(config, criterion):
    """Wire the PaddleFleet vision tower and fusion helper
    (``paddlefleet.models.kimi_k3``) to the :class:`KimiK3ModelProvider` backbone.
    """
    text_config = config.get_text_config()
    vision_config = config.vision_config

    for name in (
        "tensor_model_parallel_size",
        "context_parallel_size",
        "pipeline_model_parallel_size",
        "virtual_pipeline_model_parallel_size",
        "expert_model_parallel_size",
    ):
        setattr(text_config, name, max(getattr(text_config, name, 1), 1))

    pp_size = getattr(text_config, "pipeline_model_parallel_size", 1) or 1
    is_first_stage = pp_size == 1 or fleet.get_hybrid_communicate_group().get_stage_id() == 0

    vision_model = None
    if is_first_stage:
        vision_model, _ = build_kimi_k3_vision_tower(
            vision_config,
            params_dtype=getattr(text_config, "params_dtype", None) or getattr(text_config, "dtype", None),
        )
    language_provider = KimiK3ModelProvider.from_config(text_config)
    language_provider.multimodal_embedding = True
    language_provider.image_token_id = config.media_placeholder_token_id
    language_provider.video_token_id = -1

    if getattr(language_provider, "enable_mtp_magic_send", False):
        raise ValueError(
            "enable_mtp_magic_send re-embeds input_ids on the last pipeline "
            "stage and is incompatible with multimodal inputs."
        )
    language_model = language_provider.provide(loss_fn=criterion)

    if is_first_stage:
        vision_merge = KimiK3VisionMergeLayer(vision_model=vision_model)
        language_model.vision_merge = vision_merge
        if getattr(language_model, "_model_chunks", None):
            language_model._model_chunks[0].run_function.insert(0, vision_merge)
        else:
            language_model.run_function.insert(0, vision_merge)
    else:
        language_model.vision_merge = None

    language_model._prepare_pipeline_inputs_func = _prepare_kimi_k3_pipeline_inputs
    language_model.config_to_save = config
    language_model.is_fleet = True
    language_model._gen_aoa_config = lambda _=None: KimiK3ForConditionalGeneration._gen_aoa_config(config)
    language_model._gen_inv_aoa_config = lambda _=None: KimiK3ForConditionalGeneration._gen_inv_aoa_config(config)
    language_model.can_generate = lambda: False
    language_model.save_pretrained = types.MethodType(PretrainedModel.save_pretrained, language_model)
    return language_model


class KimiK3ForConditionalGeneration(KimiK3PretrainedModel):
    """Kimi-K3 multimodal model: MoonViT3d vision tower + KDA/MLA text backbone."""

    is_fleet = True

    @classmethod
    def _gen_aoa_config(cls, config):
        """Vision statements plus the text statements of the base class, retargeted for VL.

        Shadows the text-only mapping of :class:`KimiK3PretrainedModel` so both halves load
        from one checkpoint.
        """
        text_config = config.get_text_config()
        vision_config = config.vision_config
        dtype = getattr(text_config, "dtype", None) or getattr(config, "dtype", None)
        cast = f", dtype='{dtype}'" if dtype else ""
        vt_layers = vision_config.vt_num_hidden_layers
        vt_heads = vision_config.vt_num_attention_heads
        visual_prefix = "model.vision_model."
        aoa_config = {"aoa_statements": list(KimiK3PretrainedModel._gen_aoa_config(config)["aoa_statements"])}
        if (getattr(vision_config, "pipeline_model_parallel_size", 1) or 1) != 1:
            raise NotImplementedError(
                "Kimi-K3 vision AOA statements only cover the single-stage tower; "
                "pipeline-parallel vision re-numbers the child layers."
            )
        aoa_config["aoa_statements"] += [
            f"vision_tower.patch_embed.proj.weight -> {visual_prefix}patch_embed.embedding.proj.weight{cast}",
            f"vision_tower.patch_embed.pos_emb.weight -> {visual_prefix}patch_embed.embedding.pos_emb.weight{cast}",
            f"vision_tower.encoder.final_layernorm.weight -> {visual_prefix}final_layernorm.norm.weight{cast}",
            f"mm_projector.proj.0.weight^T -> {visual_prefix}mm_projector.proj.up_gate_proj.weight{cast}",
            f"mm_projector.proj.2.weight^T -> {visual_prefix}mm_projector.proj.down_proj.weight{cast}",
            f"mm_projector.post_norm.weight -> {visual_prefix}mm_projector.post_norm.weight{cast}",
        ]
        aoa_config["aoa_statements"] += [
            f"vision_tower.encoder.blocks.{layer_id}.{hf}{'^T' if transpose else ''} -> "
            f"{visual_prefix}layers.{layer_id}.{fleet}{cast}"
            for layer_id in range(vt_layers)
            for hf, fleet, transpose in (
                ("norm0.weight", "input_layernorm.weight", False),
                ("wo.weight", "self_attn.o_proj.weight", True),
                ("norm1.weight", "post_attention_layernorm.weight", False),
                ("mlp.fc0.weight", "mlp.up_gate_proj.weight", True),
                ("mlp.fc1.weight", "mlp.down_proj.weight", True),
            )
        ]
        # visual attention qkv: HF fuses as [all Q | all K | all V] while Fleet expects the
        # interleaved [q0 k0 v0 ...] layout, so a plain ^T would load and compute garbage.
        aoa_config["aoa_statements"] += [
            stmt
            for layer_id in range(vt_layers)
            for stmt in (
                f"vision_tower.encoder.blocks.{layer_id}.wqkv.weight -> k3vqkv{layer_id}{cast}",
                f"k3vqkv{layer_id} -> k3vqkv{layer_id}q,k3vqkv{layer_id}k,k3vqkv{layer_id}v, axis=0",
                f"k3vqkv{layer_id}q^T,k3vqkv{layer_id}k^T,k3vqkv{layer_id}v^T -> "
                f"{visual_prefix}layers.{layer_id}.self_attn.qkv_proj.weight, fused_qkv, "
                f"num_heads={vt_heads}, num_key_value_groups={vt_heads}",
            )
        ]
        return aoa_config

    @classmethod
    def _gen_inv_aoa_config(cls, config):
        """Inverse of :meth:`_gen_aoa_config`: VL weights back to the official HF schema."""
        vision_config = config.vision_config
        vt_layers = vision_config.vt_num_hidden_layers
        vt_heads = vision_config.vt_num_attention_heads
        visual_prefix = "model.vision_model."

        # language model: the Fleet names on the left are already the text-only names.
        aoa_config = {"aoa_statements": list(KimiK3PretrainedModel._gen_inv_aoa_config(config)["aoa_statements"])}

        # visual model
        aoa_config["aoa_statements"] += [
            f"{visual_prefix}patch_embed.embedding.proj.weight -> vision_tower.patch_embed.proj.weight",
            f"{visual_prefix}patch_embed.embedding.pos_emb.weight -> vision_tower.patch_embed.pos_emb.weight",
            f"{visual_prefix}final_layernorm.norm.weight -> vision_tower.encoder.final_layernorm.weight",
            f"{visual_prefix}mm_projector.proj.up_gate_proj.weight^T -> mm_projector.proj.0.weight",
            f"{visual_prefix}mm_projector.proj.down_proj.weight^T -> mm_projector.proj.2.weight",
            f"{visual_prefix}mm_projector.post_norm.weight -> mm_projector.post_norm.weight",
        ]
        aoa_config["aoa_statements"] += [
            f"{visual_prefix}layers.{layer_id}.{fleet}{'^T' if transpose else ''} -> "
            f"vision_tower.encoder.blocks.{layer_id}.{hf}"
            for layer_id in range(vt_layers)
            for hf, fleet, transpose in (
                ("norm0.weight", "input_layernorm.weight", False),
                ("wo.weight", "self_attn.o_proj.weight", True),
                ("norm1.weight", "post_attention_layernorm.weight", False),
                ("mlp.fc0.weight", "mlp.up_gate_proj.weight", True),
                ("mlp.fc1.weight", "mlp.down_proj.weight", True),
            )
        ]
        # visual attention qkv: unfuse the interleaved layout, then concatenate as HF stores it
        aoa_config["aoa_statements"] += [
            stmt
            for layer_id in range(vt_layers)
            for stmt in (
                f"{visual_prefix}layers.{layer_id}.self_attn.qkv_proj.weight -> "
                f"k3vqkv{layer_id}q,k3vqkv{layer_id}k,k3vqkv{layer_id}v, fused_qkv, "
                f"num_heads={vt_heads}, num_key_value_groups={vt_heads}",
                f"k3vqkv{layer_id}q^T,k3vqkv{layer_id}k^T,k3vqkv{layer_id}v^T -> "
                f"vision_tower.encoder.blocks.{layer_id}.wqkv.weight, axis=0",
            )
        ]
        return aoa_config

    def __new__(cls, config, have_criterion=True):
        if getattr(config, "vision_config", None) is None:
            raise ValueError(
                "KimiK3ForConditionalGeneration requires config.vision_config; "
                "use KimiK3ForCausalLM for the text-only model."
            )

        text_config = config.get_text_config()
        for name in (
            "tensor_model_parallel_size",
            "context_parallel_size",
            "pipeline_model_parallel_size",
            "virtual_pipeline_model_parallel_size",
            "expert_model_parallel_size",
        ):
            value = max(getattr(config, name, 1) or 1, 1)
            setattr(config, name, value)
            setattr(text_config, name, value)
        text_config.sequence_parallel = getattr(config, "sequence_parallel", False)

        criterion = None
        if have_criterion:
            criterion = KimiK3CriterionPipe(text_config, return_tuple=False)
        return _build_vl_model(config, criterion)


KimiK3ForConditionalGenerationPipe = KimiK3ForConditionalGeneration

__all__ = [
    "KimiK3Model",
    "KimiK3ForCausalLM",
    "KimiK3ForCausalLMPipe",
    "KimiK3ForConditionalGeneration",
    "KimiK3ForConditionalGenerationPipe",
    "KimiK3ModelProvider",
]
