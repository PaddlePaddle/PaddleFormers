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
from .configuration import KimiK2Config


@dataclass
class KimiK2Provider(GPTModelProvider):
    """
    Base config for Kimi-K2 Models.
    """

    transform_rules = {
        "dtype": "params_dtype",
    }

    def __post_init__(config):
        super().__post_init__()


class KimiK2PretrainedModel(PretrainedModel):
    config_class = KimiK2Config

    transpose_weight_keys = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "proj",
        "up_gate_proj",
        "qkv_proj",
    ]

    @classmethod
    def _gen_aoa_config(cls, config: KimiK2Config):
        # language model
        aoa_config = {"aoa_statements": []}
        aoa_config["aoa_statements"] += [
            "model.embed_tokens.weight -> model.embedding.embed_tokens.weight",
            "lm_head.weight -> model.lm_head.weight ",
            "model.layers.1.mlp.gate.weight -> model.layers.1.mlp.gate.weight, src_dtype='bfloat16',dst_dtype='float32'",
        ]
        # MLA
        for layer_id in range(config.num_hidden_layers):
            for mla_atten in ["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"]:
                aoa_config["aoa_statements"] += [
                    f"model.layers.{layer_id}.self_attn.{mla_atten}.weight^T -> model.layers.{layer_id}.self_attn.{mla_atten}.weight",
                ]
        # MLP
        # layer 0
        aoa_config["aoa_statements"] += [
            "model.layers.0.mlp.down_proj.weight^T -> model.layers.0.mlp.down_proj.weight",
            "model.layers.0.mlp.gate_proj.weight^T ,model.layers.0.mlp.up_proj.weight^T ->  model.layers.0.mlp.up_gate_proj.weight, axis=1",
        ]
        # layer 1 -> num_hidden_layers
        for layer_id in range(1, config.num_hidden_layers):
            aoa_config["aoa_statements"] += [
                f"model.layers.{layer_id}.mlp.experts.$EXPERT_ID.down_proj.weight^T -> model.layers.{layer_id}.mlp.experts.$EXPERT_ID.down_proj.weight",
                f"model.layers.{layer_id}.mlp.experts.$EXPERT_ID.gate_proj.weight^T, model.layers.{layer_id}.mlp.experts.$EXPERT_ID.up_proj.weight^T -> model.layers.{layer_id}.mlp.experts.$EXPERT_ID.up_gate_proj.weight , axis=1",
                f"model.layers.{layer_id}.mlp.shared_experts.down_proj.weight^T -> model.layers.{layer_id}.mlp.shared_experts.down_proj.weight",
                f"model.layers.{layer_id}.mlp.shared_experts.gate_proj.weight^T, model.layers.{layer_id}.mlp.shared_experts.up_proj.weight^T -> model.layers.{layer_id}.mlp.shared_experts.up_gate_proj.weight , axis=1",
            ]

        return aoa_config


class KimiK2ForCausalLM(KimiK2PretrainedModel):
    config_class = KimiK2Config

    def __new__(cls, config, have_criterion=True):
        config.tensor_model_parallel_size = max(config.tensor_model_parallel_size, 1)
        config.context_parallel_size = max(config.context_parallel_size, 1)
        config.pipeline_model_parallel_size = max(config.pipeline_model_parallel_size, 1)
        config.virtual_pipeline_model_parallel_size = max(config.virtual_pipeline_model_parallel_size, 1)
        config.expert_model_parallel_size = max(config.expert_model_parallel_size, 1)

        if hasattr(config, "rope_scaling") and config.rope_scaling:
            if "type" in config.rope_scaling:
                config.rope_type = config.rope_scaling["type"]

            if "beta_fast" in config.rope_scaling:
                config.beta_fast = config.rope_scaling["beta_fast"]
            if "beta_slow" in config.rope_scaling:
                config.beta_slow = config.rope_scaling["beta_slow"]
            if "factor" in config.rope_scaling:
                config.rotary_scaling_factor = config.rope_scaling["factor"]
            if "mscale" in config.rope_scaling:
                config.mscale = config.rope_scaling["mscale"]
            if "mscale_all_dim" in config.rope_scaling:
                config.mscale_all_dim = config.rope_scaling["mscale_all_dim"]
            if "original_max_position_embeddings" in config.rope_scaling:
                config.original_max_position_embeddings = config.rope_scaling["original_max_position_embeddings"]
        # Check if mtp_block_spec parameter is supported
        config.multi_latent_attention = True
        config.use_qk_norm = True
        model_provider_class = KimiK2Provider

        model_provider = model_provider_class.from_config(config)
        KimiK25_model = model_provider.provide()
        KimiK25_model._gen_aoa_config = cls._gen_aoa_config

        KimiK25_model.config_to_save = config

        return KimiK25_model


class KimiK2ForCausalLMPipe(KimiK2PretrainedModel, GeneralModelForCausalLMPipe):
    is_fleet = True

    def __new__(cls, config):
        # Hybrid parallel config convert.
        config.tensor_model_parallel_size = max(config.tensor_model_parallel_size, 1)
        config.context_parallel_size = max(config.context_parallel_size, 1)
        config.pipeline_model_parallel_size = max(config.pipeline_model_parallel_size, 1)
        config.virtual_pipeline_model_parallel_size = max(config.virtual_pipeline_model_parallel_size, 1)
        config.expert_model_parallel_size = max(config.expert_model_parallel_size, 1)

        if hasattr(config, "rope_scaling") and config.rope_scaling:
            if "type" in config.rope_scaling:
                config.rope_type = config.rope_scaling["type"]

            if "beta_fast" in config.rope_scaling:
                config.beta_fast = config.rope_scaling["beta_fast"]
            if "beta_slow" in config.rope_scaling:
                config.beta_slow = config.rope_scaling["beta_slow"]
            if "factor" in config.rope_scaling:
                config.rotary_scaling_factor = config.rope_scaling["factor"]
            if "mscale" in config.rope_scaling:
                config.mscale = config.rope_scaling["mscale"]
            if "mscale_all_dim" in config.rope_scaling:
                config.mscale_all_dim = config.rope_scaling["mscale_all_dim"]
            if "original_max_position_embeddings" in config.rope_scaling:
                config.original_max_position_embeddings = config.rope_scaling["original_max_position_embeddings"]
        # Check if mtp_block_spec parameter is supported
        config.multi_latent_attention = True
        config.use_qk_norm = True

        model_provider_class = KimiK2Provider
        model_provider = model_provider_class.from_config(config)

        gpt_model = model_provider.provide()
        gpt_model._gen_aoa_config = cls._gen_aoa_config

        if not hasattr(config, "architectures"):
            config.architectures = [cls.__name__.replace("Pipe", "")]
        gpt_model.config_to_save = config
        gpt_model.is_fleet = cls.is_fleet
        return gpt_model
