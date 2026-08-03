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

    Derives the mixed KDA and gated MLA attention schedule, block attention
    residuals, and flattened KDA fields required by PaddleFleet.
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
        if not isinstance(self.linear_attn_config, dict):
            raise ValueError("Kimi-K3 requires linear_attn_config.")
        self._build_layer_types()
        self._expand_attn_res_block_size()
        self._flatten_linear_attn_config()
        super().__post_init__()

    def _build_layer_types(self):
        """Turn the one-based KDA/MLA schedule into a per-layer type list."""
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
        self.layer_types = [
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

    def _expand_attn_res_block_size(self):
        """Fleet counts attention and MLP as two residual sublayers; the source
        value counts decoder layers, so double it."""
        if self.attn_res_block_size is None or self.attn_res_block_size <= 0:
            raise ValueError("Kimi-K3 attn_res_block_size must be a positive integer.")
        self.block_attention_residuals = True
        self.attn_res_block_size *= 2

    def _flatten_linear_attn_config(self):
        """Flatten the nested KDA config into Fleet TransformerConfig fields."""
        cfg = self.linear_attn_config
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


class KimiK3PretrainedModel(PretrainedModel):
    config_class = KimiK3Config
    base_model_prefix = "model"


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
