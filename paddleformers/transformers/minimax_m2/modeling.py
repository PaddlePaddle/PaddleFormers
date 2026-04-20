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

from ...nn.pp_model import CriterionLayerPipe, GeneralModelForCausalLMPipe
from ..glm4_moe.modeling import GLMMoEModelProvider
from ..model_utils import PretrainedModel
from .configuration import MiniMaxM2Config


class MiniMaxM2PreTrainedModel(PretrainedModel):
    config: MiniMaxM2Config

    @classmethod
    def _gen_aoa_config(cls, config: MiniMaxM2Config):
        model_prefix = "model."  # "" if cls == cls.base_model_class else
        using_sonic_moe = config.using_sonic_moe
        if hasattr(config, "n_routed_experts"):
            num_experts = config.n_routed_experts
        else:
            num_experts = config.num_experts
        aoa_config = {
            "aoa_statements": [
                f"model.norm.weight -> {model_prefix}norm.weight",
            ]
        }

        aoa_config["aoa_statements"] += [
            f"model.embed_tokens.weight -> {model_prefix}embedding.embed_tokens.weight",
        ]
        if config.tie_word_embeddings:
            aoa_config["aoa_statements"] += [f"model.embed_tokens.weight -> {model_prefix}lm_head.weight"]
        else:
            aoa_config["aoa_statements"] += [f"lm_head.weight -> {model_prefix}lm_head.weight"]

        num_hidden_layers = config.num_hidden_layers
        num_head_empty_layers = (
            config.num_empty_layers_add_in_head
            if hasattr(config, "num_empty_layers_add_in_head") and config.num_empty_layers_add_in_head
            else 0
        )

        # NOTE: MiniMax-M2 has no dense layers (first_k_dense_replace=0)

        if config.mtp_num_layers > 0:
            num_nextn_predict_layers = config.mtp_num_layers
        else:
            num_nextn_predict_layers = config.num_nextn_predict_layers if config.num_nextn_predict_layers else 0

        for layer_idx in reversed(range(num_hidden_layers, num_hidden_layers + num_nextn_predict_layers)):
            layer_idx_offset = layer_idx + num_head_empty_layers
            prefix = f"model.layers.{layer_idx}"
            prefix_offset = f"{model_prefix}layers.{layer_idx_offset}"
            aoa_config["aoa_statements"] += [
                f"{prefix}.eh_proj.weight^T -> {prefix_offset}.eh_proj.weight",
                f"{prefix}.enorm.weight -> {prefix_offset}.enorm.weight",
                f"{prefix}.hnorm.weight -> {prefix_offset}.hnorm.weight",
                f"{prefix}.shared_head.norm.weight -> {prefix_offset}.norm.weight",
            ]

        # layer0 - layer_num_hidden_layers
        for layer_idx in reversed(range(0, num_hidden_layers + num_nextn_predict_layers)):
            layer_idx_offset = layer_idx + num_head_empty_layers
            prefix = f"model.layers.{layer_idx}"
            prefix_offset = f"{model_prefix}layers.{layer_idx_offset}"
            if layer_idx >= num_hidden_layers:
                # for mtp
                prefix_offset += ".transformer_layer"
            aoa_config["aoa_statements"] += [
                f"{prefix}.input_layernorm.weight -> {prefix_offset}.input_layernorm.weight",
                f"{prefix}.post_attention_layernorm.weight -> {prefix_offset}.post_attention_layernorm.weight",
                f"{prefix}.self_attn.o_proj.weight^T -> {prefix_offset}.self_attn.o_proj.weight",
            ]
            if config.use_qk_norm:
                aoa_config["aoa_statements"] += [
                    f"{prefix}.self_attn.q_norm.weight -> {prefix_offset}.self_attn.q_norm.weight",
                    f"{prefix}.self_attn.k_norm.weight -> {prefix_offset}.self_attn.k_norm.weight",
                ]

            # attention qkv
            aoa_config["aoa_statements"] += [
                f"{prefix}.self_attn.q_proj.weight^T, {prefix}.self_attn.k_proj.weight^T, {prefix}.self_attn.v_proj.weight^T -> {prefix_offset}.self_attn.qkv_proj.weight, fused_qkv, num_heads={config.num_attention_heads}, num_key_value_groups={config.num_key_value_heads}",
            ]
            if config.attention_bias:
                aoa_config["aoa_statements"] += [
                    f"{prefix}.self_attn.q_proj.bias, {prefix}.self_attn.k_proj.bias, {prefix}.self_attn.v_proj.bias -> {prefix_offset}.self_attn.qkv_proj.bias, fused_qkv, num_heads={config.num_attention_heads}, num_key_value_groups={config.num_key_value_heads}, axis=0",
                ]

        # All layers are MoE (first_k_dense_replace=0)
        for layer_idx in reversed(range(config.first_k_dense_replace, num_hidden_layers + num_nextn_predict_layers)):
            layer_idx_offset = layer_idx + num_head_empty_layers
            prefix = f"model.layers.{layer_idx}"
            prefix_offset = f"{model_prefix}layers.{layer_idx_offset}"
            if layer_idx >= num_hidden_layers:
                # for mtp
                prefix_offset += ".transformer_layer"
            aoa_config["aoa_statements"] += [
                f"{prefix}.block_sparse_moe.e_score_correction_bias -> {prefix_offset}.mlp.gate.e_score_correction_bias",
                f"{prefix}.block_sparse_moe.gate.weight -> {prefix_offset}.mlp.gate.weight",
            ]
            if using_sonic_moe:
                aoa_config["aoa_statements"] += [
                    f"{prefix}.block_sparse_moe.experts.$EXPERT_ID.w2.weight -> {prefix_offset}.mlp.experts.$EXPERT_ID.down_proj.weight",
                ]
            else:
                aoa_config["aoa_statements"] += [
                    f"{prefix}.block_sparse_moe.experts.$EXPERT_ID.w2.weight^T -> {prefix_offset}.mlp.experts.$EXPERT_ID.down_proj.weight",
                ]

            for expert_id in range(config.n_routed_experts):
                if using_sonic_moe:
                    aoa_config["aoa_statements"] += [
                        f"{prefix}.block_sparse_moe.experts.{expert_id}.w1.weight, {prefix}.block_sparse_moe.experts.{expert_id}.w3.weight -> {prefix_offset}.mlp.experts.{expert_id}.up_gate_proj.weight, axis=0",
                    ]
                else:
                    aoa_config["aoa_statements"] += [
                        f"{prefix}.block_sparse_moe.experts.{expert_id}.w1.weight^T, {prefix}.block_sparse_moe.experts.{expert_id}.w3.weight^T -> {prefix_offset}.mlp.experts.{expert_id}.up_gate_proj.weight, axis=1",
                    ]

            if (config.moe_grouped_gemm or using_sonic_moe) and not config.fp8:
                ep_weight1 = []
                ep_weight2 = []
                for expert_id in range(num_experts):
                    ep_weight1.append(f"{prefix_offset}.mlp.experts.{expert_id}.up_gate_proj.weight")
                    ep_weight2.append(f"{prefix_offset}.mlp.experts.{expert_id}.down_proj.weight")
                group_gemm1 = ",".join(ep_weight1)
                group_gemm2 = ",".join(ep_weight2)
                aoa_config["aoa_statements"] += [
                    f"{group_gemm1} -> {prefix_offset}.mlp.grouped_gemm_experts.weight1, axis=0"
                    f"{group_gemm2} -> {prefix_offset}.mlp.grouped_gemm_experts.weight2, axis=0"
                ]
            else:
                if config.get("fd_fallback", False):
                    ep_weight1 = []
                    ep_weight2 = []
                    for expert_id in range(num_experts):
                        ep_weight1.append(f"{prefix_offset}.mlp.experts.{expert_id}.up_gate_proj.weight")
                        ep_weight2.append(f"{prefix_offset}.mlp.experts.{expert_id}.down_proj.weight")
                    group1 = ",".join(ep_weight1)
                    group2 = ",".join(ep_weight2)
                    aoa_config["aoa_statements"] += [
                        f"{group1} -> {prefix_offset}.mlp.experts.up_gate_proj, axis=0"
                        f"{group2} -> {prefix_offset}.mlp.experts.down_proj, axis=0"
                    ]

        return aoa_config

    # NOTE: These aoa_config items will be removed later. The subsequent AOA parsing module will automatically generate the reverse AOA based on the forward (from_pretrained) AOA.
    @classmethod
    def _gen_inv_aoa_config(cls, config: MiniMaxM2Config):
        model_prefix = "" if cls == getattr(cls, "base_model_class", None) else "model."
        using_sonic_moe = config.using_sonic_moe
        if hasattr(config, "n_routed_experts"):
            num_experts = config.n_routed_experts
        else:
            num_experts = config.num_experts
        aoa_statements = [
            f"{model_prefix}norm.weight -> model.norm.weight",
        ]

        aoa_statements += [
            "model.embedding.embed_tokens.weight -> model.embed_tokens.weight",
        ]
        if config.tie_word_embeddings:
            aoa_statements += [f"{model_prefix}lm_head.weight -> _"]
        else:
            aoa_statements += [f"{model_prefix}lm_head.weight -> lm_head.weight"]

        num_hidden_layers = config.num_hidden_layers
        num_head_empty_layers = (
            config.num_empty_layers_add_in_head
            if hasattr(config, "num_empty_layers_add_in_head") and config.num_empty_layers_add_in_head
            else 0
        )

        # NOTE: MiniMax-M2 has no dense layers (first_k_dense_replace=0)

        num_nextn_predict_layers = config.num_nextn_predict_layers if config.num_nextn_predict_layers else 0

        for layer_idx in reversed(range(num_hidden_layers, num_hidden_layers + num_nextn_predict_layers)):
            layer_idx_offset = layer_idx + num_head_empty_layers
            prefix = f"model.layers.{layer_idx}"
            prefix_offset = f"{model_prefix}layers.{layer_idx_offset}"
            aoa_statements += [
                f"{prefix_offset}.eh_proj.weight^T -> {prefix}.eh_proj.weight",
                f"{prefix_offset}.enorm.weight -> {prefix}.enorm.weight",
                f"{prefix_offset}.hnorm.weight -> {prefix}.hnorm.weight",
                f"{prefix_offset}.norm.weight -> {prefix}.shared_head.norm.weight",
            ]

        # layer 0 -> layer num_hidden_layers-1
        for layer_idx in range(0, num_hidden_layers + num_nextn_predict_layers):
            layer_idx_offset = layer_idx + num_head_empty_layers
            prefix_offset = f"{model_prefix}layers.{layer_idx_offset}"
            prefix = f"model.layers.{layer_idx}"
            if layer_idx >= num_hidden_layers:
                # for mtp
                prefix_offset += ".transformer_layer"

            aoa_statements += [
                f"{prefix_offset}.input_layernorm.weight -> {prefix}.input_layernorm.weight",
                f"{prefix_offset}.post_attention_layernorm.weight -> {prefix}.post_attention_layernorm.weight",
                f"{prefix_offset}.self_attn.o_proj.weight^T -> {prefix}.self_attn.o_proj.weight",
            ]

            if config.q_lora_rank:
                # MLA attention
                aoa_statements += [
                    f"{prefix_offset}.self_attn.o_proj.weight^T -> {prefix}.self_attn.o_proj.weight",
                    f"{prefix_offset}.self_attn.q_a_proj.weight^T -> {prefix}.self_attn.q_a_proj.weight",
                    f"{prefix_offset}.self_attn.q_b_proj.weight^T -> {prefix}.self_attn.q_b_proj.weight",
                    f"{prefix_offset}.self_attn.kv_a_proj_with_mqa.weight^T -> {prefix}.self_attn.kv_a_proj_with_mqa.weight",
                    f"{prefix_offset}.self_attn.kv_b_proj.weight^T -> {prefix}.self_attn.kv_b_proj.weight",
                    f"{prefix_offset}.self_attn.q_a_layernorm.weight -> {prefix}.self_attn.q_a_layernorm.weight",
                    f"{prefix_offset}.self_attn.kv_a_layernorm.weight -> {prefix}.self_attn.kv_a_layernorm.weight",
                ]
            else:
                if config.use_qk_norm:
                    aoa_statements += [
                        f"{prefix_offset}.self_attn.q_norm.weight -> {prefix}.self_attn.q_norm.weight",
                        f"{prefix_offset}.self_attn.k_norm.weight -> {prefix}.self_attn.k_norm.weight",
                    ]

                aoa_statements += [
                    f"{prefix_offset}.self_attn.qkv_proj.weight -> {prefix}.self_attn.q_proj.weight, {prefix}.self_attn.k_proj.weight, {prefix}.self_attn.v_proj.weight , fused_qkv, num_heads={config.num_attention_heads}, num_key_value_groups = {config.num_key_value_heads}",
                ]
                aoa_statements += [
                    f"{prefix}.self_attn.{x}_proj.weight^T -> {prefix}.self_attn.{x}_proj.weight"
                    for x in ("q", "k", "v")
                ]
                if config.attention_bias:
                    aoa_statements += [
                        f"{prefix_offset}.self_attn.qkv_proj.bias -> {prefix}.self_attn.q_proj.bias, {prefix}.self_attn.k_proj.bias, {prefix}.self_attn.v_proj.bias , fused_qkv, num_heads={config.num_attention_heads}, num_key_value_groups = {config.num_key_value_heads}, axis = 0",
                    ]

        # All layers are MoE (first_k_dense_replace=0)
        for layer_idx in range(config.first_k_dense_replace, num_hidden_layers + num_nextn_predict_layers):
            layer_idx_offset = layer_idx + num_head_empty_layers
            prefix_offset = f"{model_prefix}layers.{layer_idx_offset}"
            prefix = f"model.layers.{layer_idx}"
            if layer_idx >= num_hidden_layers:
                # for mtp
                prefix_offset += ".transformer_layer"

            if (config.moe_grouped_gemm or using_sonic_moe) and not config.fp8:
                ep_weight1 = []
                ep_weight2 = []
                for expert_id in range(config.n_routed_experts):
                    ep_weight1.append(f"{prefix_offset}.mlp.experts.{expert_id}.up_gate_proj.weight")
                    ep_weight2.append(f"{prefix_offset}.mlp.experts.{expert_id}.down_proj.weight")
                group_gemm1 = ",".join(ep_weight1)
                group_gemm2 = ",".join(ep_weight2)
                aoa_statements += [
                    f"{prefix_offset}.mlp.grouped_gemm_experts.weight1 -> {group_gemm1}, axis=0"
                    f"{prefix_offset}.mlp.grouped_gemm_experts.weight2 -> {group_gemm2}, axis=0"
                ]
            else:
                if config.get("fd_fallback", False):
                    ep_weight1 = []
                    ep_weight2 = []
                    for expert_id in range(num_experts):
                        ep_weight1.append(f"{prefix_offset}.mlp.experts.{expert_id}.up_gate_proj.weight")
                        ep_weight2.append(f"{prefix_offset}.mlp.experts.{expert_id}.down_proj.weight")
                    group1 = ",".join(ep_weight1)
                    group2 = ",".join(ep_weight2)
                    aoa_statements += [
                        f"{prefix_offset}.mlp.experts.up_gate_proj -> {group1}, axis=0"
                        f"{prefix_offset}.mlp.experts.down_proj -> {group2}, axis=0"
                    ]

            aoa_statements += [
                # do cast
                f"{prefix_offset}.mlp.gate.weight -> {prefix}.block_sparse_moe.gate.weight",
                # do transpose
                f"{prefix_offset}.mlp.gate.e_score_correction_bias -> {prefix}.block_sparse_moe.e_score_correction_bias",
            ]

            if using_sonic_moe:
                aoa_statements += [
                    f"{prefix_offset}.mlp.experts.{expert_id}.up_gate_proj.weight -> {prefix_offset}.block_sparse_moe.experts.{expert_id}.w1.weight, {prefix_offset}.block_sparse_moe.experts.{expert_id}.w3.weight, axis=0"
                    for expert_id in range(config.n_routed_experts)
                ]
            else:
                aoa_statements += [
                    f"{prefix_offset}.mlp.experts.{expert_id}.up_gate_proj.weight -> {prefix_offset}.block_sparse_moe.experts.{expert_id}.w1.weight, {prefix_offset}.block_sparse_moe.experts.{expert_id}.w3.weight, axis=1"
                    for expert_id in range(config.n_routed_experts)
                ]

            if not using_sonic_moe:
                aoa_statements += (
                    [
                        f"{prefix_offset}.block_sparse_moe.experts.{expert_id}.w1.weight^T -> {prefix}.block_sparse_moe.experts.{expert_id}.w1.weight"
                        for expert_id in range(config.n_routed_experts)
                    ]
                    + [
                        f"{prefix_offset}.block_sparse_moe.experts.{expert_id}.w3.weight^T -> {prefix}.block_sparse_moe.experts.{expert_id}.w3.weight"
                        for expert_id in range(config.n_routed_experts)
                    ]
                    + [
                        f"{prefix_offset}.mlp.experts.{expert_id}.down_proj.weight^T-> {prefix}.block_sparse_moe.experts.{expert_id}.w2.weight"
                        for expert_id in range(config.n_routed_experts)
                    ]
                )

        aoa_config = {"aoa_statements": aoa_statements}
        return aoa_config


class MiniMaxM2ForCausalLM(MiniMaxM2PreTrainedModel):
    is_fleet = True

    def __new__(cls, config):
        # Hybrid parallel config convert.
        config.tensor_model_parallel_size = max(config.tensor_model_parallel_size, 1)
        config.context_parallel_size = max(config.context_parallel_size, 1)
        config.pipeline_model_parallel_size = max(config.pipeline_model_parallel_size, 1)
        config.virtual_pipeline_model_parallel_size = max(config.virtual_pipeline_model_parallel_size, 1)
        config.expert_model_parallel_size = max(config.expert_model_parallel_size, 1)
        config.fuse_rms_norm = True
        # config.multi_latent_attention = True
        # config.rotary_interleaved = True

        model_provider_class = GLMMoEModelProvider
        model_provider = model_provider_class.from_config(config)
        loss_fn = None
        if getattr(config, "dpo_config", None):
            loss_fn = CriterionLayerPipe(config, use_infohub=True)
        gpt_model = model_provider.provide(loss_fn=loss_fn)
        gpt_model._gen_aoa_config = cls._gen_aoa_config
        gpt_model._gen_inv_aoa_config = cls._gen_inv_aoa_config
        gpt_model.config_to_save = config
        gpt_model.is_fleet = cls.is_fleet
        return gpt_model


class MiniMaxM2ForCausalLMPipe(MiniMaxM2PreTrainedModel, GeneralModelForCausalLMPipe):
    is_fleet = True

    def __new__(cls, config):
        # Hybrid parallel config convert.
        config.tensor_model_parallel_size = max(config.tensor_model_parallel_size, 1)
        config.context_parallel_size = max(config.context_parallel_size, 1)
        config.pipeline_model_parallel_size = max(config.pipeline_model_parallel_size, 1)
        config.virtual_pipeline_model_parallel_size = max(config.virtual_pipeline_model_parallel_size, 1)
        config.expert_model_parallel_size = max(config.expert_model_parallel_size, 1)
        config.fuse_rms_norm = True
        # config.multi_latent_attention = True
        # config.rotary_interleaved = True
        model_provider_class = GLMMoEModelProvider
        model_provider = model_provider_class.from_config(config)
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
    "MiniMaxM2ForCausalLMPipe",
    "MiniMaxM2ForCausalLM",
]
