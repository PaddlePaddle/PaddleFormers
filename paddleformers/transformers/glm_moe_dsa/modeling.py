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
from ..aoa_config_base import MoEAOAConfigGenerator
from ..glm4_moe.modeling import GLMMoEModelProvider
from ..model_utils import PretrainedModel
from .configuration import GlmMoeDsaConfig


class GlmMoeDsaPreTrainedModel(PretrainedModel):
    config: GlmMoeDsaConfig

    @classmethod
    def _gen_aoa_config(cls, config: GlmMoeDsaConfig):
        """Generate AOA config using the base class for minimal code duplication.

        GLM MoE DSA features:
        - Multi-Latent Attention (MLA) for efficient KV caching
        - Dense-to-MoE hybrid layers (first_k_dense_replace)
        - MTP (Multi-Token Prediction) support
        - Shared experts for routing efficiency

        Args:
            config: GlmMoeDsaConfig configuration object

        Returns:
            Dictionary with 'aoa_statements' key containing conversion statements
        """
        return MoEAOAConfigGenerator.gen_aoa_config(config)

    @classmethod
    def _gen_inv_aoa_config(cls, config: GlmMoeDsaConfig):
        """Generate inverse AOA config using the base class.

        Maps PaddleFleet weight names back to HuggingFace format,
        used during save_pretrained to convert weights back to HF convention.

        Args:
            config: GlmMoeDsaConfig configuration object

        Returns:
            Dictionary with 'aoa_statements' key containing inverse conversion statements
        """
        return MoEAOAConfigGenerator.gen_inv_aoa_config(config)


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
        config.multi_latent_attention = True
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
        config.multi_latent_attention = True
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
    "GlmMoeDsaForCausalLMPipe",
    "GlmMoeDsaForCausalLM",
]
