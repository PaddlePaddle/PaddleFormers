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

# Refer to NVIDIA Megatron-Bridge https://github.com/NVIDIA-NeMo/Megatron-Bridge
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

import contextlib
import inspect
import logging
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Literal, Optional, Union

import paddle

from ..utils.import_utils import is_paddlefleet_available

# This module requires paddlefleet to be installed
if not is_paddlefleet_available():
    raise ImportError(
        "paddlefleet is required for gpt_provider. "
        "Please install paddlefleet to use this module. "
        "You can install it with: pip install paddlefleet"
    )

from paddlefleet import LayerSpec
from paddlefleet.models.gpt import GPTModel as FleetGPTModel
from paddlefleet.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec

try:
    from paddlefleet.models.gpt.gpt_config import GPTConfig
except ImportError:
    from paddlefleet.transformer.transformer_config import (
        TransformerConfig as GPTConfig,
    )

from paddlefleet.gpt_builders import gpt_builder

from paddleformers.transformers.model_utils import PretrainedModel

from .model_provider import ModelProviderMixin

logger = logging.getLogger(__name__)


class GPTModel(FleetGPTModel, PretrainedModel):
    """
    GPTModel class that inherits from FleetGPTModel.
    This class requires paddlefleet to be installed.
    """


# GPTModel = FleetGPTModel


def local_layer_spec(config: "GPTModelProvider") -> LayerSpec:
    """Create a local layer specification without Transformer Engine.

    Args:
        config: GPT configuration object

    Returns:
        LayerSpec: Module specification for local implementation layers
    """
    return get_gpt_layer_local_spec(
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
        qk_layernorm=config.qk_layernorm,
        normalization=config.normalization,
    )


@dataclass
class GPTModelProvider(GPTConfig, ModelProviderMixin[GPTModel]):
    """Configuration and provider for PaddleFleet GPT models.

    This class extends TransformerConfig with GPT-specific parameters and
    provides a method to instantiate configured GPT models.
    """

    # Model configuration
    fp16_lm_cross_entropy: bool = False
    parallel_output: bool = True
    tie_word_embeddings: bool = True
    make_vocab_size_divisible_by: int = 128
    position_embedding_type: Literal["learned_absolute", "rope"] = "rope"
    rotary_base: int = 10000
    rotary_percent: float = 1.0
    seq_len_interpolation_factor: Optional[float] = None
    seq_length: int = 1024

    max_sequence_length: int = 1024

    attention_softmax_in_fp32: bool = False
    deallocate_pipeline_outputs: bool = True
    scatter_embedding_sequence_parallel: bool = True
    tp_only_amax_red: bool = False
    tp_comm_overlap_cfg: Optional[Union[str, dict[str, Any]]] = None
    """Config file when tp_comm_overlap is enabled."""

    generation_config: Optional[Any] = None

    # This represents the unpadded vocab size
    # The padded vocab size is automatically calculated in the provide() method.
    vocab_size: Optional[int] = None
    # Set if the tokenizer provides the vocab size. In this case, the vocab size will be padded
    # Controls whether vocab size should be padded for tensor parallelism
    should_pad_vocab: bool = False

    # MoE / FP8
    n_routed_experts: Optional[int] = None
    moe_grouped_gemm: bool = False
    use_qk_norm: bool = False
    fp8: Optional[str] = None
    normalization: str = "RMSNorm"

    # Multi-token prediction
    mtp_enabled: bool = False

    # Additional parameters that might be needed
    init_model_with_meta_device: bool = False
    use_te_rng_tracker: bool = False
    virtual_pipeline_model_parallel_size: Optional[int] = None
    account_for_embedding_in_pipeline_split: bool = False
    account_for_loss_in_pipeline_split: bool = False

    # TODO: Support fusions
    # Fusions
    # masked_softmax_fusion: bool = True
    # cross_entropy_loss_fusion: bool = True  # Generally beneficial, no specific dependencies
    # gradient_accumulation_fusion: bool = field(default_factory=fusions.can_enable_gradient_accumulation_fusion)

    # If True, restore the modelopt_state that contains quantization, sparsity, speculative decoding transformation state.
    # When resuming modelopt_state, we also change the transformer_layer_spec to `paddlefleet.post_training.modelopt.gpt.model_specs` which is a combination of local spec + TEDotProductAttention.
    restore_modelopt_state: bool = False

    def provide(self, pre_process=None, post_process=None, vp_stage=None, loss_fn=None) -> GPTModel:
        """Configure and instantiate a PaddleFleet GPT model based on this configuration.

        Args:
            pre_process: Whether to include pre-processing in the model, defaults to first pipeline stage
            post_process: Whether to include post-processing in the model, defaults to last pipeline stage
            vp_stage: Virtual pipeline stage

        Returns:
            GPTModel: Configured PaddleFleet GPT model instance
        """
        pp_size = self.pipeline_model_parallel_size

        is_pipeline_asymmetric = getattr(self, "account_for_embedding_in_pipeline_split", False) or getattr(
            self, "account_for_loss_in_pipeline_split", False
        )
        is_pipeline_asymmetric |= (
            getattr(self, "num_empty_layers_add_in_head", None) or getattr(self, "num_empty_layers_add_in_tail", None)
        ) is not None

        # Initialize model as meta data instead of allocating data on a device
        model_init_device_context = contextlib.nullcontext
        if self.init_model_with_meta_device:
            model_init_device_context = partial(paddle.device, device="meta")

        # Check if mtp_block_spec parameter is supported
        kwargs = {}
        if "mtp_block_spec" in inspect.signature(GPTModel.__init__).parameters:
            kwargs["mtp_block_spec"] = mtp_block_spec(self, vp_stage=vp_stage)

        """
        if self.attention_backend == AttnBackend.local:
            if hasattr(transformer_layer_spec, "submodules"):
                transformer_layer_spec.submodules.self_attention.submodules.core_attention = DotProductAttention
        """

        with model_init_device_context():
            fleet_model = gpt_builder(
                self, num_stages=pp_size, seg_method="layer:TransformerLayer|EmptyLayer", loss_fn=loss_fn
            )
            # Convert original FleetGPTModel to our GPTModel to correctly inherit PretrainedModel methods
            model = GPTModel.__new__(GPTModel)
            # Manually copy all attributes
            for attr_name in dir(fleet_model):
                if not attr_name.startswith("__"):
                    try:
                        attr_value = getattr(fleet_model, attr_name)
                        setattr(model, attr_name, attr_value)
                    except:
                        pass

        return model


def mtp_block_spec(config: "GPTModelProvider", vp_stage: Optional[int] = None) -> Optional[LayerSpec]:
    """Pass in the MTP block spec if model has MTP layers.

    Args:
        config: GPT configuration object

    Returns:
        LayerSpec: The MTP module specification
    """
    if getattr(config, "mtp_num_layers", None):
        from paddlefleet.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec

        if isinstance(config.transformer_layer_spec, Callable):
            if "vp_stage" in inspect.signature(config.transformer_layer_spec).parameters:
                spec = config.transformer_layer_spec(config, vp_stage=vp_stage)
            else:
                spec = config.transformer_layer_spec(config)
        else:
            spec = config.transformer_layer_spec
        if hasattr(spec, "layer_specs") and len(spec.layer_specs) == 0:
            # Get the decoder layer spec explicitly if no decoder layer in the last stage,
            # Only happens with block spec (TransformerBlockSubmodules) when using MoE.
            spec = local_layer_spec(config)
        return get_gpt_mtp_block_spec(config, spec, vp_stage=vp_stage)
    else:
        return None
