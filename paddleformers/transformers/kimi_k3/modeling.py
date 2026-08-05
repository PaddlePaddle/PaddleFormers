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

from dataclasses import dataclass

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
    gated_linear_unit: bool = True
    normalization: str = "RMSNorm"

    # KDA/MLA hybrid attention schedule
    linear_attn_config: dict | None = None

    # General defaults
    share_embeddings_and_output_weights: bool = False

    transform_rules = {
        **GPTModelProvider.transform_rules,
        "dtype": "params_dtype",
        # HF config.json -> Fleet TransformerConfig field mappings
        **KimiK3TextConfig._HF_TO_FLEET_FIELD_MAP,
    }

    def __post_init__(self):
        if self.attn_res_block_size is None or self.attn_res_block_size <= 0:
            raise ValueError("Kimi-K3 attn_res_block_size must be a positive integer.")
        self.block_attention_residuals = True
        # Fleet counts attention and MLP as two residual sublayers, while the
        # source value counts decoder layers, so double it.
        self.attn_res_block_size *= 2
        super().__post_init__()


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


__all__ = [
    "KimiK3Model",
    "KimiK3ForCausalLM",
    "KimiK3ForCausalLMPipe",
    "KimiK3ModelProvider",
]
