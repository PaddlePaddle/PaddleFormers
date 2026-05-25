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

import logging
from dataclasses import dataclass

from ...nn.pp_model import CriterionLayerPipe, GeneralModelForCausalLMPipe
from ..gpt_provider import GPTModelProvider
from ..model_utils import PretrainedModel
from .configuration import DeepseekV4Config

logger = logging.getLogger(__name__)


@dataclass
class DeepseekV4ModelProvider(GPTModelProvider):
    """DeepSeek-V4 configuration provider for PaddleFleet GPTModel.

    Activates DSv4 Hybrid Attention (CSA + MLA + Grouped LoRA Output),
    mHC (multi-stream residual), MoE with hash routing, and MTP.
    """

    # === DSv4 required defaults ===
    multi_latent_attention: bool = True
    experimental_attention_variant: str = "dsv4_hybrid"
    enable_hyper_connections: bool = True
    gated_linear_unit: bool = True
    bias_activation_fusion: bool = True

    # MoE defaults
    moe_router_load_balancing_type: str = "seq_aux_loss"
    moe_shared_expert_overlap: bool = True
    moe_router_pre_softmax: bool = False
    moe_permute_fusion: bool = True
    moe_router_dtype: str = "fp32"

    # General defaults
    share_embeddings_and_output_weights: bool = False
    persist_layer_norm: bool = True
    apply_rope_fusion: bool = True
    bias_dropout_fusion: bool = True

    # MTP
    mtp_loss_scaling_factor: float = 0.1

    # Misc
    recompute_granularity: str = None
    virtual_pipeline_model_parallel_size: int = None

    transform_rules = {
        **GPTModelProvider.transform_rules,
        "dtype": "params_dtype",
        # HF config.json -> Fleet TransformerConfig field mappings
        "compress_ratios": "csa_compress_ratios",
        "num_hash_layers": "moe_n_hash_layers",
        "compress_rope_theta": "csa_compress_rotary_base",
        "sliding_window": "csa_window_size",
        "hc_mult": "num_residual_streams",
        "hc_sinkhorn_iters": "mhc_sinkhorn_iterations",
        "head_dim": "v_head_dim",
        "index_n_heads": "dsa_index_n_heads",
        "index_head_dim": "dsa_index_head_dim",
        "index_topk": "dsa_index_topk",
    }


class DeepseekV4PreTrainedModel(PretrainedModel):
    config: DeepseekV4Config

    @classmethod
    def _gen_aoa_config(cls, config: DeepseekV4Config):
        """Weight conversion: HuggingFace DSv4 checkpoint -> PaddleFleet internal format.

        Maps open-source DeepSeek-V4 HuggingFace parameter names to PaddleFleet names.
        Handles: Embedding/LM Head, DSv4 Hybrid Attention (MLA + grouped LoRA output),
        mHC (multi-stream HyperConnection), MoE experts, CSA Compressor, DSA Indexer,
        MTP (Multi-Token Prediction) layers.

        HF naming convention: layers.{L}.attn.*, layers.{L}.ffn.*, embed.weight, etc.
        PF naming convention: model.layers.{L}.self_attn.*, model.layers.{L}.mlp.*, etc.
        """
        num_hidden_layers = config.num_hidden_layers
        num_experts = config.n_routed_experts
        n_shared_experts = getattr(config, "n_shared_experts", 1)
        moe_n_hash_layers = getattr(config, "moe_n_hash_layers", 3)
        moe_expert_fusion = getattr(config, "moe_expert_fusion", False)
        csa_compress_ratios = config.csa_compress_ratios
        num_head_empty_layers = (
            config.num_empty_layers_add_in_head
            if hasattr(config, "num_empty_layers_add_in_head") and config.num_empty_layers_add_in_head
            else 0
        )
        mtp_num_layers = getattr(config, "num_nextn_predict_layers", 0)
        # Note: num_hidden_layers in PaddleFormers config is the decoder layer count (NOT bumped by MTP).
        # MTP layers are appended AFTER the decoder layers, so MTP layer i is at index num_hidden_layers + i.
        num_decoder_layers = num_hidden_layers

        stmts = []

        # === 1. Embedding, Final Norm, LM Head ===
        stmts += [
            "embed.weight -> model.embedding.embed_tokens.weight",
            "norm.weight -> model.norm.weight",
        ]
        if config.tie_word_embeddings:
            stmts += ["embed.weight -> model.lm_head.weight"]
        else:
            stmts += ["head.weight -> model.lm_head.weight"]

        # === 2. Per-layer mappings (layer 0 to num_decoder_layers-1) ===
        for L in range(num_decoder_layers):
            src = f"layers.{L}"
            tgt = f"model.layers.{L + num_head_empty_layers}"

            # --- LayerNorm ---
            stmts += [
                f"{src}.attn_norm.weight -> {tgt}.input_layernorm.weight",
                f"{src}.ffn_norm.weight -> {tgt}.post_attention_layernorm.weight",
            ]

            # --- DSv4 Hybrid Attention ---
            # Q path: low-rank decomposition (q_down -> q_norm -> q_up)
            stmts += [
                f"{src}.attn.wq_a.weight^T -> {tgt}.self_attn.linear_q_down_proj.weight",
                f"{src}.attn.wq_b.weight^T -> {tgt}.self_attn.linear_q_up_proj.weight",
                f"{src}.attn.q_norm.weight -> {tgt}.self_attn.q_layernorm.weight",
            ]
            # KV path: single-head MQA projection
            stmts += [
                f"{src}.attn.wkv.weight^T -> {tgt}.self_attn.linear_kv_proj.weight",
                f"{src}.attn.kv_norm.weight -> {tgt}.self_attn.kv_layernorm.weight",
            ]
            # Output projection: grouped LoRA (wo_a has same shape, no transpose)
            stmts += [
                f"{src}.attn.wo_a.weight -> {tgt}.self_attn.linear_o_group_proj",
                f"{src}.attn.wo_b.weight^T -> {tgt}.self_attn.o_proj.weight",
            ]
            # Attention sink (learnable bias per head)
            stmts += [
                f"{src}.attn.attn_sink -> {tgt}.self_attn.core_attention.attn_sink, dtype='bfloat16'",
            ]

            # --- mHC: Self-Attention HyperConnection ---
            # scale [3] -> split into alpha_pre[1], alpha_post[1], alpha_res[1]
            stmts += [
                f"{src}.hc_attn_scale -> {tgt}.self_attention_hyper_connection.alpha_pre_t, "
                f"{tgt}.self_attention_hyper_connection.alpha_post_t, "
                f"{tgt}.self_attention_hyper_connection.alpha_res_t, axis=0",
                f"{tgt}.self_attention_hyper_connection.alpha_pre_t -> {tgt}.self_attention_hyper_connection.alpha_pre, dtype='bfloat16'",
                f"{tgt}.self_attention_hyper_connection.alpha_post_t -> {tgt}.self_attention_hyper_connection.alpha_post, dtype='bfloat16'",
                f"{tgt}.self_attention_hyper_connection.alpha_res_t -> {tgt}.self_attention_hyper_connection.alpha_res, dtype='bfloat16'",
                f"{src}.hc_attn_base -> {tgt}.self_attention_hyper_connection.bias, dtype='bfloat16'",
                f"{src}.hc_attn_fn^T -> {tgt}.self_attention_hyper_connection.mapping_proj.weight, dtype='bfloat16'",
            ]

            # --- mHC: MLP HyperConnection ---
            stmts += [
                f"{src}.hc_ffn_scale -> {tgt}.mlp_hyper_connection.alpha_pre_t, "
                f"{tgt}.mlp_hyper_connection.alpha_post_t, "
                f"{tgt}.mlp_hyper_connection.alpha_res_t, axis=0",
                f"{tgt}.mlp_hyper_connection.alpha_pre_t -> {tgt}.mlp_hyper_connection.alpha_pre, dtype='bfloat16'",
                f"{tgt}.mlp_hyper_connection.alpha_post_t -> {tgt}.mlp_hyper_connection.alpha_post, dtype='bfloat16'",
                f"{tgt}.mlp_hyper_connection.alpha_res_t -> {tgt}.mlp_hyper_connection.alpha_res, dtype='bfloat16'",
                f"{src}.hc_ffn_base -> {tgt}.mlp_hyper_connection.bias, dtype='bfloat16'",
                f"{src}.hc_ffn_fn^T -> {tgt}.mlp_hyper_connection.mapping_proj.weight, dtype='bfloat16'",
            ]

            # --- CSA Compressor (present when compress_ratio > 0) ---
            if csa_compress_ratios[L] > 0:
                comp_src = f"{src}.attn.compressor"
                comp_tgt = f"{tgt}.self_attn.core_attention.compressor"
                stmts += [
                    f"{comp_src}.ape -> {comp_tgt}.ape, dtype='bfloat16'",
                    f"{comp_src}.norm.weight -> {comp_tgt}.norm.weight",
                    f"{comp_src}.wgate.weight^T -> {comp_tgt}.linear_wgate.weight",
                    f"{comp_src}.wkv.weight^T -> {comp_tgt}.linear_wkv.weight",
                ]

            # --- DSA Indexer (present on layers with compress_ratio > 0 and <= 4) ---
            if csa_compress_ratios[L] > 0 and csa_compress_ratios[L] <= 4:
                idx_src = f"{src}.attn.indexer"
                idx_tgt = f"{tgt}.self_attn.core_attention.indexer"
                stmts += [
                    f"{idx_src}.compressor.ape -> {idx_tgt}.compressor.ape, dtype='bfloat16'",
                    f"{idx_src}.compressor.norm.weight -> {idx_tgt}.compressor.norm.weight",
                    f"{idx_src}.compressor.wgate.weight^T -> {idx_tgt}.compressor.linear_wgate.weight",
                    f"{idx_src}.compressor.wkv.weight^T -> {idx_tgt}.compressor.linear_wkv.weight",
                    f"{idx_src}.weights_proj.weight^T -> {idx_tgt}.linear_weights_proj.weight",
                    f"{idx_src}.wq_b.weight^T -> {idx_tgt}.linear_wq_b.weight",
                ]

            # --- MoE Gate ---
            stmts += [f"{src}.ffn.gate.weight -> {tgt}.mlp.gate.weight, dtype='float32'"]
            # Non-hash layers have e_score_correction_bias; hash layers use tid2eid
            if L >= moe_n_hash_layers:
                stmts += [f"{src}.ffn.gate.bias -> {tgt}.mlp.gate.e_score_correction_bias"]
            else:
                stmts += [f"{src}.ffn.gate.tid2eid -> {tgt}.mlp.gate.tid2eid"]

            # --- Routed Experts (loop-expanded, per-expert mapping) ---
            for E in range(num_experts):
                stmts += [
                    f"{src}.ffn.experts.{E}.w1.weight^T, "
                    f"{src}.ffn.experts.{E}.w3.weight^T "
                    f"-> {tgt}.mlp.experts.{E}.up_gate_proj.weight, axis=1",
                    f"{src}.ffn.experts.{E}.w2.weight^T " f"-> {tgt}.mlp.experts.{E}.down_proj.weight",
                ]

            # --- GroupGEMM fusion: stack all experts into single tensors ---
            if moe_expert_fusion:
                ep_weight1 = []
                ep_weight2 = []
                for E in range(num_experts):
                    ep_weight1.append(f"{tgt}.mlp.experts.{E}.up_gate_proj.weight")
                    ep_weight2.append(f"{tgt}.mlp.experts.{E}.down_proj.weight")
                stmts += [
                    f"{','.join(ep_weight1)} -> {tgt}.mlp.grouped_gemm_experts.weight1, axis=0",
                    f"{','.join(ep_weight2)} -> {tgt}.mlp.grouped_gemm_experts.weight2, axis=0",
                ]

            # --- Shared Expert ---
            if n_shared_experts > 0:
                stmts += [
                    f"{src}.ffn.shared_experts.w1.weight^T, "
                    f"{src}.ffn.shared_experts.w3.weight^T "
                    f"-> {tgt}.mlp.shared_experts.up_gate_proj.weight, fused_ffn",
                    f"{src}.ffn.shared_experts.w2.weight^T " f"-> {tgt}.mlp.shared_experts.down_proj.weight",
                ]

        # === 3. Top-level mHC head contraction (output head HyperConnection) ===
        if mtp_num_layers > 0:
            stmts += [
                "hc_head_base -> model.mhc_contract.hc_head_base, dtype='bfloat16'",
                "hc_head_fn^T -> model.mhc_contract.hc_head_fn, dtype='bfloat16'",
                "hc_head_scale -> model.mhc_contract.hc_head_scale, dtype='bfloat16'",
            ]

        # === 4. MTP (Multi-Token Prediction) layers ===
        for i in range(mtp_num_layers):
            mtp_src = f"mtp.{i}"
            mtp_tgt = f"model.layers.{num_decoder_layers + num_head_empty_layers + i}"
            tl = f"{mtp_tgt}.transformer_layer"  # transformer_layer prefix in MTP

            # --- MTP-specific projections ---
            stmts += [
                f"{mtp_src}.e_proj.weight^T -> {mtp_tgt}.e_proj.weight",
                f"{mtp_src}.enorm.weight -> {mtp_tgt}.enorm.weight",
                f"{mtp_src}.h_proj.weight^T -> {mtp_tgt}.h_proj.weight",
                f"{mtp_src}.hnorm.weight -> {mtp_tgt}.hnorm.weight",
                f"{mtp_src}.norm.weight -> {mtp_tgt}.norm.weight",
            ]

            # --- MTP head HyperConnection ---
            stmts += [
                f"{mtp_src}.hc_head_base -> {mtp_tgt}.hc_head_base, dtype='bfloat16'",
                f"{mtp_src}.hc_head_fn^T -> {mtp_tgt}.hc_head_fn, dtype='bfloat16'",
                f"{mtp_src}.hc_head_scale -> {mtp_tgt}.hc_head_scale, dtype='bfloat16'",
            ]

            # --- LayerNorm (inside transformer_layer) ---
            stmts += [
                f"{mtp_src}.attn_norm.weight -> {tl}.input_layernorm.weight",
                f"{mtp_src}.ffn_norm.weight -> {tl}.post_attention_layernorm.weight",
            ]

            # --- DSv4 Hybrid Attention (inside transformer_layer) ---
            stmts += [
                f"{mtp_src}.attn.wq_a.weight^T -> {tl}.self_attn.linear_q_down_proj.weight",
                f"{mtp_src}.attn.wq_b.weight^T -> {tl}.self_attn.linear_q_up_proj.weight",
                f"{mtp_src}.attn.q_norm.weight -> {tl}.self_attn.q_layernorm.weight",
                f"{mtp_src}.attn.wkv.weight^T -> {tl}.self_attn.linear_kv_proj.weight",
                f"{mtp_src}.attn.kv_norm.weight -> {tl}.self_attn.kv_layernorm.weight",
                f"{mtp_src}.attn.wo_a.weight -> {tl}.self_attn.linear_o_group_proj",
                f"{mtp_src}.attn.wo_b.weight^T -> {tl}.self_attn.o_proj.weight",
                f"{mtp_src}.attn.attn_sink -> {tl}.self_attn.core_attention.attn_sink, dtype='bfloat16'",
            ]

            # --- mHC: Self-Attention HyperConnection (inside transformer_layer) ---
            stmts += [
                f"{mtp_src}.hc_attn_scale -> {tl}.self_attention_hyper_connection.alpha_pre_t, "
                f"{tl}.self_attention_hyper_connection.alpha_post_t, "
                f"{tl}.self_attention_hyper_connection.alpha_res_t, axis=0",
                f"{tl}.self_attention_hyper_connection.alpha_pre_t -> {tl}.self_attention_hyper_connection.alpha_pre, dtype='bfloat16'",
                f"{tl}.self_attention_hyper_connection.alpha_post_t -> {tl}.self_attention_hyper_connection.alpha_post, dtype='bfloat16'",
                f"{tl}.self_attention_hyper_connection.alpha_res_t -> {tl}.self_attention_hyper_connection.alpha_res, dtype='bfloat16'",
                f"{mtp_src}.hc_attn_base -> {tl}.self_attention_hyper_connection.bias, dtype='bfloat16'",
                f"{mtp_src}.hc_attn_fn^T -> {tl}.self_attention_hyper_connection.mapping_proj.weight, dtype='bfloat16'",
            ]

            # --- mHC: MLP HyperConnection (inside transformer_layer) ---
            stmts += [
                f"{mtp_src}.hc_ffn_scale -> {tl}.mlp_hyper_connection.alpha_pre_t, "
                f"{tl}.mlp_hyper_connection.alpha_post_t, "
                f"{tl}.mlp_hyper_connection.alpha_res_t, axis=0",
                f"{tl}.mlp_hyper_connection.alpha_pre_t -> {tl}.mlp_hyper_connection.alpha_pre, dtype='bfloat16'",
                f"{tl}.mlp_hyper_connection.alpha_post_t -> {tl}.mlp_hyper_connection.alpha_post, dtype='bfloat16'",
                f"{tl}.mlp_hyper_connection.alpha_res_t -> {tl}.mlp_hyper_connection.alpha_res, dtype='bfloat16'",
                f"{mtp_src}.hc_ffn_base -> {tl}.mlp_hyper_connection.bias, dtype='bfloat16'",
                f"{mtp_src}.hc_ffn_fn^T -> {tl}.mlp_hyper_connection.mapping_proj.weight, dtype='bfloat16'",
            ]

            # --- MTP CSA Compressor (if compress_ratio > 0 for this layer) ---
            mtp_layer_idx = num_decoder_layers + i
            if mtp_layer_idx < len(csa_compress_ratios) and csa_compress_ratios[mtp_layer_idx] > 0:
                comp_src = f"{mtp_src}.attn.compressor"
                comp_tgt = f"{tl}.self_attn.core_attention.compressor"
                stmts += [
                    f"{comp_src}.ape -> {comp_tgt}.ape, dtype='bfloat16'",
                    f"{comp_src}.norm.weight -> {comp_tgt}.norm.weight",
                    f"{comp_src}.wgate.weight^T -> {comp_tgt}.linear_wgate.weight",
                    f"{comp_src}.wkv.weight^T -> {comp_tgt}.linear_wkv.weight",
                ]
                if csa_compress_ratios[mtp_layer_idx] <= 4:
                    idx_src = f"{mtp_src}.attn.indexer"
                    idx_tgt = f"{tl}.self_attn.core_attention.indexer"
                    stmts += [
                        f"{idx_src}.compressor.ape -> {idx_tgt}.compressor.ape, dtype='bfloat16'",
                        f"{idx_src}.compressor.norm.weight -> {idx_tgt}.compressor.norm.weight",
                        f"{idx_src}.compressor.wgate.weight^T -> {idx_tgt}.compressor.linear_wgate.weight",
                        f"{idx_src}.compressor.wkv.weight^T -> {idx_tgt}.compressor.linear_wkv.weight",
                        f"{idx_src}.weights_proj.weight^T -> {idx_tgt}.linear_weights_proj.weight",
                        f"{idx_src}.wq_b.weight^T -> {idx_tgt}.linear_wq_b.weight",
                    ]

            # --- MoE Gate (MTP layers are always non-hash, so always have bias) ---
            stmts += [
                f"{mtp_src}.ffn.gate.weight -> {tl}.mlp.gate.weight, dtype='float32'",
                f"{mtp_src}.ffn.gate.bias -> {tl}.mlp.gate.e_score_correction_bias",
            ]

            # --- Routed Experts ---
            for E in range(num_experts):
                stmts += [
                    f"{mtp_src}.ffn.experts.{E}.w1.weight^T, "
                    f"{mtp_src}.ffn.experts.{E}.w3.weight^T "
                    f"-> {tl}.mlp.experts.{E}.up_gate_proj.weight, axis=1",
                    f"{mtp_src}.ffn.experts.{E}.w2.weight^T " f"-> {tl}.mlp.experts.{E}.down_proj.weight",
                ]

            # --- GroupGEMM fusion for MTP experts ---
            if moe_expert_fusion:
                ep_weight1 = []
                ep_weight2 = []
                for E in range(num_experts):
                    ep_weight1.append(f"{tl}.mlp.experts.{E}.up_gate_proj.weight")
                    ep_weight2.append(f"{tl}.mlp.experts.{E}.down_proj.weight")
                stmts += [
                    f"{','.join(ep_weight1)} -> {tl}.mlp.grouped_gemm_experts.weight1, axis=0",
                    f"{','.join(ep_weight2)} -> {tl}.mlp.grouped_gemm_experts.weight2, axis=0",
                ]

            # --- Shared Expert ---
            if n_shared_experts > 0:
                stmts += [
                    f"{mtp_src}.ffn.shared_experts.w1.weight^T, "
                    f"{mtp_src}.ffn.shared_experts.w3.weight^T "
                    f"-> {tl}.mlp.shared_experts.up_gate_proj.weight, fused_ffn",
                    f"{mtp_src}.ffn.shared_experts.w2.weight^T " f"-> {tl}.mlp.shared_experts.down_proj.weight",
                ]

        return {"aoa_statements": stmts}

    @classmethod
    def _gen_inv_aoa_config(cls, config: DeepseekV4Config):
        """Inverse weight conversion: PaddleFleet -> HuggingFace DSv4 format.

        Maps PaddleFleet internal parameter names back to HuggingFace naming.
        This is the reverse of _gen_aoa_config, used for save_pretrained.
        """
        num_hidden_layers = config.num_hidden_layers
        num_experts = config.n_routed_experts
        n_shared_experts = getattr(config, "n_shared_experts", 1)
        moe_n_hash_layers = getattr(config, "moe_n_hash_layers", 3)
        moe_expert_fusion = getattr(config, "moe_expert_fusion", False)
        csa_compress_ratios = config.csa_compress_ratios
        num_head_empty_layers = (
            config.num_empty_layers_add_in_head
            if hasattr(config, "num_empty_layers_add_in_head") and config.num_empty_layers_add_in_head
            else 0
        )
        mtp_num_layers = getattr(config, "num_nextn_predict_layers", 0)
        # Note: num_hidden_layers in PaddleFormers config is the decoder layer count (NOT bumped by MTP).
        # MTP layers are appended AFTER the decoder layers, so MTP layer i is at index num_hidden_layers + i.
        num_decoder_layers = num_hidden_layers

        stmts = []

        # === 1. Embedding, Final Norm, LM Head ===
        stmts += [
            "model.embedding.embed_tokens.weight -> embed.weight",
            "model.norm.weight -> norm.weight",
        ]
        if config.tie_word_embeddings:
            stmts += ["model.lm_head.weight -> _"]
        else:
            stmts += ["model.lm_head.weight -> head.weight"]

        # === 2. MTP layers (inverse, reversed order) ===
        for i in reversed(range(mtp_num_layers)):
            mtp_tgt = f"mtp.{i}"
            mtp_src = f"model.layers.{num_decoder_layers + num_head_empty_layers + i}"
            tl = f"{mtp_src}.transformer_layer"

            # --- MTP-specific projections ---
            stmts += [
                f"{mtp_src}.e_proj.weight^T -> {mtp_tgt}.e_proj.weight",
                f"{mtp_src}.enorm.weight -> {mtp_tgt}.enorm.weight",
                f"{mtp_src}.h_proj.weight^T -> {mtp_tgt}.h_proj.weight",
                f"{mtp_src}.hnorm.weight -> {mtp_tgt}.hnorm.weight",
                f"{mtp_src}.norm.weight -> {mtp_tgt}.norm.weight",
            ]

            # --- MTP head HyperConnection ---
            stmts += [
                f"{mtp_src}.hc_head_base -> {mtp_tgt}.hc_head_base, dtype='float32'",
                f"{mtp_src}.hc_head_fn^T -> {mtp_tgt}.hc_head_fn, dtype='float32'",
                f"{mtp_src}.hc_head_scale -> {mtp_tgt}.hc_head_scale, dtype='float32'",
            ]

            # --- LayerNorm ---
            stmts += [
                f"{tl}.input_layernorm.weight -> {mtp_tgt}.attn_norm.weight",
                f"{tl}.post_attention_layernorm.weight -> {mtp_tgt}.ffn_norm.weight",
            ]

            # --- DSv4 Hybrid Attention ---
            stmts += [
                f"{tl}.self_attn.linear_q_down_proj.weight^T -> {mtp_tgt}.attn.wq_a.weight",
                f"{tl}.self_attn.linear_q_up_proj.weight^T -> {mtp_tgt}.attn.wq_b.weight",
                f"{tl}.self_attn.q_layernorm.weight -> {mtp_tgt}.attn.q_norm.weight",
                f"{tl}.self_attn.linear_kv_proj.weight^T -> {mtp_tgt}.attn.wkv.weight",
                f"{tl}.self_attn.kv_layernorm.weight -> {mtp_tgt}.attn.kv_norm.weight",
                f"{tl}.self_attn.linear_o_group_proj -> {mtp_tgt}.attn.wo_a.weight",
                f"{tl}.self_attn.o_proj.weight^T -> {mtp_tgt}.attn.wo_b.weight",
                f"{tl}.self_attn.core_attention.attn_sink -> {mtp_tgt}.attn.attn_sink, dtype='float32'",
            ]

            # --- mHC: Self-Attention HyperConnection ---
            stmts += [
                f"{tl}.self_attention_hyper_connection.alpha_pre, "
                f"{tl}.self_attention_hyper_connection.alpha_post, "
                f"{tl}.self_attention_hyper_connection.alpha_res "
                f"-> {mtp_tgt}.hc_attn_scale, axis=0",
                f"{mtp_tgt}.hc_attn_scale -> {mtp_tgt}.hc_attn_scale, dtype='float32'",
                f"{tl}.self_attention_hyper_connection.bias -> {mtp_tgt}.hc_attn_base, dtype='float32'",
                f"{tl}.self_attention_hyper_connection.mapping_proj.weight^T -> {mtp_tgt}.hc_attn_fn, dtype='float32'",
            ]

            # --- mHC: MLP HyperConnection ---
            stmts += [
                f"{tl}.mlp_hyper_connection.alpha_pre, "
                f"{tl}.mlp_hyper_connection.alpha_post, "
                f"{tl}.mlp_hyper_connection.alpha_res "
                f"-> {mtp_tgt}.hc_ffn_scale, axis=0",
                f"{mtp_tgt}.hc_ffn_scale -> {mtp_tgt}.hc_ffn_scale, dtype='float32'",
                f"{tl}.mlp_hyper_connection.bias -> {mtp_tgt}.hc_ffn_base, dtype='float32'",
                f"{tl}.mlp_hyper_connection.mapping_proj.weight^T -> {mtp_tgt}.hc_ffn_fn, dtype='float32'",
            ]

            # --- MTP CSA Compressor ---
            mtp_layer_idx = num_decoder_layers + i
            if mtp_layer_idx < len(csa_compress_ratios) and csa_compress_ratios[mtp_layer_idx] > 0:
                comp_src = f"{tl}.self_attn.core_attention.compressor"
                comp_tgt = f"{mtp_tgt}.attn.compressor"
                stmts += [
                    f"{comp_src}.ape -> {comp_tgt}.ape, dtype='float32'",
                    f"{comp_src}.norm.weight -> {comp_tgt}.norm.weight",
                    f"{comp_src}.linear_wgate.weight^T -> {comp_tgt}.wgate.weight",
                    f"{comp_src}.linear_wkv.weight^T -> {comp_tgt}.wkv.weight",
                ]
                if csa_compress_ratios[mtp_layer_idx] <= 4:
                    idx_src = f"{tl}.self_attn.core_attention.indexer"
                    idx_tgt = f"{mtp_tgt}.attn.indexer"
                    stmts += [
                        f"{idx_src}.compressor.ape -> {idx_tgt}.compressor.ape, dtype='float32'",
                        f"{idx_src}.compressor.norm.weight -> {idx_tgt}.compressor.norm.weight",
                        f"{idx_src}.compressor.linear_wgate.weight^T -> {idx_tgt}.compressor.wgate.weight",
                        f"{idx_src}.compressor.linear_wkv.weight^T -> {idx_tgt}.compressor.wkv.weight",
                        f"{idx_src}.linear_weights_proj.weight^T -> {idx_tgt}.weights_proj.weight",
                        f"{idx_src}.linear_wq_b.weight^T -> {idx_tgt}.wq_b.weight",
                    ]

            # --- MoE Gate ---
            stmts += [
                f"{tl}.mlp.gate.weight -> {mtp_tgt}.ffn.gate.weight, dtype='bfloat16'",
                f"{tl}.mlp.gate.e_score_correction_bias -> {mtp_tgt}.ffn.gate.bias",
            ]

            # --- GroupGEMM de-fusion ---
            if moe_expert_fusion:
                ep_weight1 = []
                ep_weight2 = []
                for E in range(num_experts):
                    ep_weight1.append(f"{tl}.mlp.experts.{E}.up_gate_proj.weight")
                    ep_weight2.append(f"{tl}.mlp.experts.{E}.down_proj.weight")
                stmts += [
                    f"{tl}.mlp.grouped_gemm_experts.weight1 -> {','.join(ep_weight1)}, axis=0",
                    f"{tl}.mlp.grouped_gemm_experts.weight2 -> {','.join(ep_weight2)}, axis=0",
                ]

            # --- Routed Experts (split then transpose) ---
            for E in range(num_experts):
                stmts += [
                    f"{tl}.mlp.experts.{E}.up_gate_proj.weight "
                    f"-> {tl}.mlp.experts.{E}.w1.weight, "
                    f"{tl}.mlp.experts.{E}.w3.weight, axis=1",
                ]
                stmts += [
                    f"{tl}.mlp.experts.{E}.w1.weight^T -> {mtp_tgt}.ffn.experts.{E}.w1.weight",
                    f"{tl}.mlp.experts.{E}.w3.weight^T -> {mtp_tgt}.ffn.experts.{E}.w3.weight",
                    f"{tl}.mlp.experts.{E}.down_proj.weight^T -> {mtp_tgt}.ffn.experts.{E}.w2.weight",
                ]

            # --- Shared Expert ---
            if n_shared_experts > 0:
                stmts += [
                    f"{tl}.mlp.shared_experts.up_gate_proj.weight "
                    f"-> {tl}.mlp.shared_experts.w1.weight, "
                    f"{tl}.mlp.shared_experts.w3.weight, fused_ffn",
                    f"{tl}.mlp.shared_experts.w1.weight^T -> {mtp_tgt}.ffn.shared_experts.w1.weight",
                    f"{tl}.mlp.shared_experts.w3.weight^T -> {mtp_tgt}.ffn.shared_experts.w3.weight",
                    f"{tl}.mlp.shared_experts.down_proj.weight^T " f"-> {mtp_tgt}.ffn.shared_experts.w2.weight",
                ]

        # === 3. Top-level mHC head contraction (inverse) ===
        if mtp_num_layers > 0:
            stmts += [
                "model.mhc_contract.hc_head_base -> hc_head_base, dtype='float32'",
                "model.mhc_contract.hc_head_fn^T -> hc_head_fn, dtype='float32'",
                "model.mhc_contract.hc_head_scale -> hc_head_scale, dtype='float32'",
            ]

        # === 4. Per-layer mappings (reversed to avoid intermediate tensor name collisions) ===
        for L in reversed(range(num_decoder_layers)):
            src = f"model.layers.{L + num_head_empty_layers}"
            tgt = f"layers.{L}"

            # --- LayerNorm ---
            stmts += [
                f"{src}.input_layernorm.weight -> {tgt}.attn_norm.weight",
                f"{src}.post_attention_layernorm.weight -> {tgt}.ffn_norm.weight",
            ]

            # --- DSv4 Hybrid Attention ---
            stmts += [
                f"{src}.self_attn.linear_q_down_proj.weight^T -> {tgt}.attn.wq_a.weight",
                f"{src}.self_attn.linear_q_up_proj.weight^T -> {tgt}.attn.wq_b.weight",
                f"{src}.self_attn.q_layernorm.weight -> {tgt}.attn.q_norm.weight",
            ]
            stmts += [
                f"{src}.self_attn.linear_kv_proj.weight^T -> {tgt}.attn.wkv.weight",
                f"{src}.self_attn.kv_layernorm.weight -> {tgt}.attn.kv_norm.weight",
            ]
            stmts += [
                f"{src}.self_attn.linear_o_group_proj -> {tgt}.attn.wo_a.weight",
                f"{src}.self_attn.o_proj.weight^T -> {tgt}.attn.wo_b.weight",
            ]
            stmts += [
                f"{src}.self_attn.core_attention.attn_sink -> {tgt}.attn.attn_sink, dtype='float32'",
            ]

            # --- mHC: Self-Attention HyperConnection (merge alpha_pre/post/res -> scale) ---
            stmts += [
                f"{src}.self_attention_hyper_connection.alpha_pre, "
                f"{src}.self_attention_hyper_connection.alpha_post, "
                f"{src}.self_attention_hyper_connection.alpha_res "
                f"-> {tgt}.hc_attn_scale, axis=0",
                f"{tgt}.hc_attn_scale -> {tgt}.hc_attn_scale, dtype='float32'",
                f"{src}.self_attention_hyper_connection.bias -> {tgt}.hc_attn_base, dtype='float32'",
                f"{src}.self_attention_hyper_connection.mapping_proj.weight^T -> {tgt}.hc_attn_fn, dtype='float32'",
            ]

            # --- mHC: MLP HyperConnection ---
            stmts += [
                f"{src}.mlp_hyper_connection.alpha_pre, "
                f"{src}.mlp_hyper_connection.alpha_post, "
                f"{src}.mlp_hyper_connection.alpha_res "
                f"-> {tgt}.hc_ffn_scale, axis=0",
                f" {tgt}.hc_ffn_scale ->  {tgt}.hc_ffn_scale, dtype='float32'"
                f"{src}.mlp_hyper_connection.bias -> {tgt}.hc_ffn_base, dtype='float32'",
                f"{src}.mlp_hyper_connection.mapping_proj.weight^T -> {tgt}.hc_ffn_fn, dtype='float32'",
            ]

            # --- CSA Compressor ---
            if csa_compress_ratios[L] > 0:
                comp_src = f"{src}.self_attn.core_attention.compressor"
                comp_tgt = f"{tgt}.attn.compressor"
                stmts += [
                    f"{comp_src}.ape -> {comp_tgt}.ape, dtype='float32'",
                    f"{comp_src}.norm.weight -> {comp_tgt}.norm.weight",
                    f"{comp_src}.linear_wgate.weight^T -> {comp_tgt}.wgate.weight",
                    f"{comp_src}.linear_wkv.weight^T -> {comp_tgt}.wkv.weight",
                ]

            # --- DSA Indexer (present on layers with compress_ratio > 0 and <= 4) ---
            if csa_compress_ratios[L] > 0 and csa_compress_ratios[L] <= 4:
                idx_src = f"{src}.self_attn.core_attention.indexer"
                idx_tgt = f"{tgt}.attn.indexer"
                stmts += [
                    f"{idx_src}.compressor.ape -> {idx_tgt}.compressor.ape, dtype='float32'",
                    f"{idx_src}.compressor.norm.weight -> {idx_tgt}.compressor.norm.weight",
                    f"{idx_src}.compressor.linear_wgate.weight^T -> {idx_tgt}.compressor.wgate.weight",
                    f"{idx_src}.compressor.linear_wkv.weight^T -> {idx_tgt}.compressor.wkv.weight",
                    f"{idx_src}.linear_weights_proj.weight^T -> {idx_tgt}.weights_proj.weight",
                    f"{idx_src}.linear_wq_b.weight^T -> {idx_tgt}.wq_b.weight",
                ]

            # --- MoE Gate ---
            stmts += [f"{src}.mlp.gate.weight -> {tgt}.ffn.gate.weight,dtype='bfloat16'"]
            if L >= moe_n_hash_layers:
                stmts += [f"{src}.mlp.gate.e_score_correction_bias -> {tgt}.ffn.gate.bias"]
            else:
                stmts += [f"{src}.mlp.gate.tid2eid -> {tgt}.ffn.gate.tid2eid"]

            # --- GroupGEMM de-fusion: split stacked tensor back to per-expert ---
            if moe_expert_fusion:
                ep_weight1 = []
                ep_weight2 = []
                for E in range(num_experts):
                    ep_weight1.append(f"{src}.mlp.experts.{E}.up_gate_proj.weight")
                    ep_weight2.append(f"{src}.mlp.experts.{E}.down_proj.weight")
                stmts += [
                    f"{src}.mlp.grouped_gemm_experts.weight1 -> {','.join(ep_weight1)}, axis=0",
                    f"{src}.mlp.grouped_gemm_experts.weight2 -> {','.join(ep_weight2)}, axis=0",
                ]

            # --- Routed Experts (inverse: split fused_ffn, then transpose) ---
            for E in range(num_experts):
                # Step 1: split up_gate_proj back to w1/w3 (intermediate, no transpose yet)
                stmts += [
                    f"{src}.mlp.experts.{E}.up_gate_proj.weight "
                    f"-> {src}.mlp.experts.{E}.w1.weight, "
                    f"{src}.mlp.experts.{E}.w3.weight, axis=1",
                ]
                # Step 2: transpose each piece to HF shape
                stmts += [
                    f"{src}.mlp.experts.{E}.w1.weight^T -> {tgt}.ffn.experts.{E}.w1.weight",
                    f"{src}.mlp.experts.{E}.w3.weight^T -> {tgt}.ffn.experts.{E}.w3.weight",
                    f"{src}.mlp.experts.{E}.down_proj.weight^T -> {tgt}.ffn.experts.{E}.w2.weight",
                ]

            # --- Shared Expert ---
            if n_shared_experts > 0:
                stmts += [
                    f"{src}.mlp.shared_experts.up_gate_proj.weight "
                    f"-> {src}.mlp.shared_experts.w1.weight, "
                    f"{src}.mlp.shared_experts.w3.weight, fused_ffn",
                    f"{src}.mlp.shared_experts.w1.weight^T -> {tgt}.ffn.shared_experts.w1.weight",
                    f"{src}.mlp.shared_experts.w3.weight^T -> {tgt}.ffn.shared_experts.w3.weight",
                    f"{src}.mlp.shared_experts.down_proj.weight^T " f"-> {tgt}.ffn.shared_experts.w2.weight",
                ]

        return {"aoa_statements": stmts}


class DeepseekV4ForCausalLM(DeepseekV4PreTrainedModel):
    is_fleet = True

    def __new__(cls, config):
        # Parallelism config safeguards
        config.tensor_model_parallel_size = max(getattr(config, "tensor_model_parallel_size", 1), 1)
        config.context_parallel_size = max(getattr(config, "context_parallel_size", 1), 1)
        config.pipeline_model_parallel_size = max(getattr(config, "pipeline_model_parallel_size", 1), 1)
        config.virtual_pipeline_model_parallel_size = max(
            getattr(config, "virtual_pipeline_model_parallel_size", 1), 1
        )
        config.expert_model_parallel_size = max(getattr(config, "expert_model_parallel_size", 1), 1)
        config.fuse_rms_norm = True

        # Ensure DSv4 critical switches are on
        config.multi_latent_attention = True
        config.experimental_attention_variant = "dsv4_hybrid"
        config.enable_hyper_connections = True

        model_provider = DeepseekV4ModelProvider.from_config(config)
        loss_fn = None
        if getattr(config, "dpo_config", None):
            loss_fn = CriterionLayerPipe(config, use_infohub=True)
        gpt_model = model_provider.provide(loss_fn=loss_fn)
        gpt_model._gen_aoa_config = cls._gen_aoa_config
        gpt_model._gen_inv_aoa_config = cls._gen_inv_aoa_config
        gpt_model.config_to_save = config
        gpt_model.is_fleet = cls.is_fleet

        return gpt_model


class DeepseekV4ForCausalLMPipe(DeepseekV4PreTrainedModel, GeneralModelForCausalLMPipe):
    is_fleet = True

    def __new__(cls, config):
        # Parallelism config safeguards
        config.tensor_model_parallel_size = max(getattr(config, "tensor_model_parallel_size", 1), 1)
        config.context_parallel_size = max(getattr(config, "context_parallel_size", 1), 1)
        config.pipeline_model_parallel_size = max(getattr(config, "pipeline_model_parallel_size", 1), 1)
        config.virtual_pipeline_model_parallel_size = max(
            getattr(config, "virtual_pipeline_model_parallel_size", 1), 1
        )
        config.expert_model_parallel_size = max(getattr(config, "expert_model_parallel_size", 1), 1)
        config.fuse_rms_norm = True

        # Ensure DSv4 critical switches are on
        config.multi_latent_attention = True
        config.experimental_attention_variant = "dsv4_hybrid"
        config.enable_hyper_connections = True

        model_provider = DeepseekV4ModelProvider.from_config(config)
        loss_fn = None
        if getattr(config, "dpo_config", None):
            loss_fn = CriterionLayerPipe(config, use_infohub=True)
        gpt_model = model_provider.provide(loss_fn=loss_fn)
        gpt_model._gen_aoa_config = cls._gen_aoa_config
        gpt_model._gen_inv_aoa_config = cls._gen_inv_aoa_config
        if not hasattr(config, "architectures"):
            config.architectures = [cls.__name__.replace("Pipe", "")]
        gpt_model.config_to_save = config
        gpt_model.is_fleet = cls.is_fleet

        return gpt_model


__all__ = [
    "DeepseekV4ForCausalLMPipe",
    "DeepseekV4ForCausalLM",
]
