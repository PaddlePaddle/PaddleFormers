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

from ...nn.pp_model import CriterionLayerPipe, GeneralModelForCausalLMPipe
from ..glm4_moe.modeling import GLMMoEModelProvider
from ..model_utils import PretrainedModel
from .configuration import GlmMoeDsaConfig


class GlmMoeDsaPreTrainedModel(PretrainedModel):
    config: GlmMoeDsaConfig
    config_class = GlmMoeDsaConfig
    base_model_prefix = "model"
    _keep_in_fp32_modules = ["mlp.gate.weight", "e_score_correction_bias"]
    transpose_weight_keys = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    @classmethod
    def _gen_aoa_config(cls, config: GlmMoeDsaConfig):
        model_prefix = "model."  # if cls == cls.base_model_class else "model."
        is_fleet = getattr(cls, "is_fleet", False)
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
        if is_fleet:
            aoa_config["aoa_statements"] += [
                f"model.embed_tokens.weight -> {model_prefix}embedding.embed_tokens.weight",
            ]
            if config.tie_word_embeddings:
                aoa_config["aoa_statements"] += [f"model.embed_tokens.weight -> {model_prefix}lm_head.weight"]
            else:
                aoa_config["aoa_statements"] += [f"lm_head.weight -> {model_prefix}lm_head.weight"]
        else:
            aoa_config["aoa_statements"] += [
                f"model.embed_tokens.weight -> {model_prefix}embed_tokens.weight",
            ]

        num_hidden_layers = config.num_hidden_layers
        num_head_empty_layers = (
            config.num_empty_layers_add_in_head
            if hasattr(config, "num_empty_layers_add_in_head") and config.num_empty_layers_add_in_head
            else 0
        )

        num_nextn_predict_layers = config.num_nextn_predict_layers if config.num_nextn_predict_layers else 0

        # dense layer dense2moe
        for layer_idx in reversed(range(0, config.first_k_dense_replace)):
            layer_idx_offset = layer_idx + num_head_empty_layers
            aoa_config["aoa_statements"] += [
                f"model.layers.{layer_idx}.mlp.down_proj.weight^T -> {model_prefix}layers.{layer_idx_offset}.mlp.down_proj.weight"
            ]
            aoa_config["aoa_statements"] += [
                f"model.layers.{layer_idx}.mlp.gate_proj.weight^T, model.layers.{layer_idx}.mlp.up_proj.weight^T -> {model_prefix}layers.{layer_idx_offset}.mlp.up_gate_proj.weight, fused_ffn",
            ]

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
            # attention qkv
            if config.multi_latent_attention:
                aoa_config["aoa_statements"] += [
                    f"{prefix}.self_attn.kv_a_proj_with_mqa.weight^T -> {prefix_offset}.self_attn.kv_a_proj_with_mqa.weight",
                    f"{prefix}.self_attn.kv_b_proj.weight^T -> {prefix_offset}.self_attn.kv_b_proj.weight",
                    f"{prefix}.self_attn.q_a_proj.weight^T -> {prefix_offset}.self_attn.q_a_proj.weight",
                    f"{prefix}.self_attn.q_b_proj.weight^T -> {prefix_offset}.self_attn.q_b_proj.weight",
                ]
                if config.use_qk_norm:
                    aoa_config["aoa_statements"] += [
                        f"{prefix}.self_attn.q_a_layernorm.weight -> {prefix_offset}.self_attn.q_a_layernorm.weight",
                        f"{prefix}.self_attn.kv_a_layernorm.weight -> {prefix_offset}.self_attn.kv_a_layernorm.weight",
                    ]
        # layer1 - layer_num_hidden_layers
        for layer_idx in reversed(range(config.first_k_dense_replace, num_hidden_layers + num_nextn_predict_layers)):
            layer_idx_offset = layer_idx + num_head_empty_layers
            prefix = f"model.layers.{layer_idx}"
            prefix_offset = f"{model_prefix}layers.{layer_idx_offset}"
            if layer_idx >= num_hidden_layers:
                # for mtp
                prefix_offset += ".transformer_layer"
            aoa_config["aoa_statements"] += [
                f"{prefix}.mlp.gate.e_score_correction_bias -> {prefix_offset}.mlp.gate.e_score_correction_bias",
                f"{prefix}.mlp.gate.weight -> {prefix_offset}.mlp.gate.weight, dtype='float32'",
                f"{prefix}.mlp.shared_experts.down_proj.weight^T -> {prefix_offset}.mlp.shared_experts.down_proj.weight",
            ]
            if using_sonic_moe:
                aoa_config["aoa_statements"] += [
                    f"{prefix}.mlp.experts.$EXPERT_ID.down_proj.weight -> {prefix_offset}.mlp.experts.$EXPERT_ID.down_proj.weight",
                ]
            else:
                aoa_config["aoa_statements"] += [
                    f"{prefix}.mlp.experts.$EXPERT_ID.down_proj.weight^T -> {prefix_offset}.mlp.experts.$EXPERT_ID.down_proj.weight",
                ]

            # FFN
            aoa_config["aoa_statements"] += [
                f"{prefix}.mlp.shared_experts.gate_proj.weight^T, {prefix}.mlp.shared_experts.up_proj.weight^T -> {prefix_offset}.mlp.shared_experts.up_gate_proj.weight, fused_ffn",
            ]
            if is_fleet:
                if using_sonic_moe:
                    aoa_config["aoa_statements"] += [
                        f"{prefix}.mlp.experts.$EXPERT_ID.gate_proj.weight, {prefix}.mlp.experts.$EXPERT_ID.up_proj.weight -> {prefix_offset}.mlp.experts.$EXPERT_ID.up_gate_proj.weight, axis=0",
                    ]
                else:
                    aoa_config["aoa_statements"] += [
                        f"{prefix}.mlp.experts.$EXPERT_ID.gate_proj.weight^T, {prefix}.mlp.experts.$EXPERT_ID.up_proj.weight^T -> {prefix_offset}.mlp.experts.$EXPERT_ID.up_gate_proj.weight, axis=1",
                    ]

            else:
                aoa_config["aoa_statements"] += [
                    f"{prefix}.mlp.experts.$EXPERT_ID.gate_proj.weight^T, {prefix}.mlp.experts.$EXPERT_ID.up_proj.weight^T -> {prefix_offset}.mlp.experts.$EXPERT_ID.up_gate_proj.weight, fused_ffn",
                ]

            # attention qkv
            aoa_config["aoa_statements"] += [
                f"{prefix}.input_layernorm.weight -> {prefix_offset}.input_layernorm.weight",
                f"{prefix}.post_attention_layernorm.weight -> {prefix_offset}.post_attention_layernorm.weight",
                f"{prefix}.self_attn.o_proj.weight^T -> {prefix_offset}.self_attn.o_proj.weight",
            ]
            if config.multi_latent_attention:
                aoa_config["aoa_statements"] += [
                    f"{prefix}.self_attn.kv_a_proj_with_mqa.weight^T -> {prefix_offset}.self_attn.kv_a_proj_with_mqa.weight",
                    f"{prefix}.self_attn.kv_b_proj.weight^T -> {prefix_offset}.self_attn.kv_b_proj.weight",
                    f"{prefix}.self_attn.q_a_proj.weight^T -> {prefix_offset}.self_attn.q_a_proj.weight",
                    f"{prefix}.self_attn.q_b_proj.weight^T -> {prefix_offset}.self_attn.q_b_proj.weight",
                ]
                if config.use_qk_norm:
                    aoa_config["aoa_statements"] += [
                        f"{prefix}.self_attn.q_a_layernorm.weight -> {prefix_offset}.self_attn.q_a_layernorm.weight",
                        f"{prefix}.self_attn.kv_a_layernorm.weight -> {prefix_offset}.self_attn.kv_a_layernorm.weight",
                    ]

            if is_fleet and (config.moe_grouped_gemm or using_sonic_moe) and not config.fp8:
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


class GlmMoeDsaForCausalLM(GlmMoeDsaPreTrainedModel):
    is_fleet = True

    def __new__(cls, config):
        # Hybrid parallel config convert.
        config.tensor_model_parallel_size = max(config.tensor_model_parallel_size, 1)
        config.context_parallel_size = max(config.context_parallel_size, 1)
        config.pipeline_model_parallel_size = max(config.pipeline_model_parallel_size, 1)
        config.virtual_pipeline_model_parallel_size = max(config.virtual_pipeline_model_parallel_size, 1)
        config.expert_model_parallel_size = max(config.expert_model_parallel_size, 1)
        config.fuse_rms_norm = True

        model_provider_class = GLMMoEModelProvider
        model_provider = model_provider_class.from_config(config)
        loss_fn = None
        if getattr(config, "dpo_config", None):
            loss_fn = CriterionLayerPipe(config, use_infohub=True)
        gpt_model = model_provider.provide(loss_fn=loss_fn)
        gpt_model._gen_aoa_config = cls._gen_aoa_config
        gpt_model.config_to_save = config
        gpt_model.is_fleet = cls.is_fleet
        return gpt_model


class GlmMoeDsaForCausalLMPipe(GlmMoeDsaPreTrainedModel, GeneralModelForCausalLMPipe):
    is_fleet = True

    def __new__(cls, config):
        # Hybrid parallel config convert.
        config.tensor_model_parallel_size = max(config.tensor_model_parallel_size, 1)
        config.context_parallel_size = max(config.context_parallel_size, 1)
        config.pipeline_model_parallel_size = max(config.pipeline_model_parallel_size, 1)
        config.virtual_pipeline_model_parallel_size = max(config.virtual_pipeline_model_parallel_size, 1)
        config.expert_model_parallel_size = max(config.expert_model_parallel_size, 1)
        config.fuse_rms_norm = True

        model_provider_class = GLMMoEModelProvider
        model_provider = model_provider_class.from_config(config)
        loss_fn = None
        if getattr(config, "dpo_config", None):
            loss_fn = CriterionLayerPipe(config, use_infohub=True)
        gpt_model = model_provider.provide(loss_fn=loss_fn)
        gpt_model._gen_aoa_config = cls._gen_aoa_config
        if not hasattr(config, "architectures"):
            config.architectures = [cls.__name__.replace("Pipe", "")]
        gpt_model.config_to_save = config
        gpt_model.is_fleet = cls.is_fleet
        return gpt_model


__all__ = [
    "GlmMoeDsaForCausalLMPipe",
    "GlmMoeDsaForCausalLM",
]
