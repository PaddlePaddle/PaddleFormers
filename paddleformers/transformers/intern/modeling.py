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

"""
InternLM2 Common Modeling

This module provides unified model classes that automatically route to the correct
implementation (2.0 or 2.5) based on the model configuration.
"""

from paddleformers.transformers.model_utils import PretrainedModel
from paddleformers.utils.log import logger

from .configuration import InternLM2Config


class InternLM2PretrainedModel(PretrainedModel):
    """
    Base class for all InternLM2 models.

    This is a proxy that routes to the actual implementation (2.0 or 2.5).
    """

    config_class = InternLM2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["InternLM2DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def __init__(self, config: InternLM2Config):
        """
        Initialize the appropriate model implementation based on config.

        Args:
            config: InternLM2Config with version detection
        """
        super().__init__(config)

        # Detect version and load appropriate implementation
        if config.is_version_2_5:
            logger.info("Detected InternLM2 2.5, loading 2.5 implementation")
            from ..intern_lm2_5.modeling import InternLM25PretrainedModel as ImplModel
        else:
            logger.error("Detected InternLM2 2.0, but 2.0 implementation is not supported!")
            raise NotImplementedError(
                "InternLM2 2.0 is not supported in PaddleFormers. "
                "Please use InternLM2 2.5 or later versions. "
                "If you need to use 2.0, please implement `paddleformers/transformers/internlm2/` module first."
            )

        # Store the actual implementation
        self._impl = ImplModel(config)

        # Copy all attributes from implementation to self
        # This makes the proxy transparent
        for key, value in self._impl.__dict__.items():
            if key not in self.__dict__:
                self.__dict__[key] = value

    def forward(self, *args, **kwargs):
        """Forward to the actual implementation."""
        return self._impl(*args, **kwargs)

    def __getattr__(self, name):
        """Proxy all attribute access to the actual implementation."""
        if name.startswith("_") or name in ["_impl", "config"]:
            return object.__getattribute__(self, name)
        return getattr(self._impl, name)

    def __setattr__(self, name, value):
        """Proxy all attribute setting to the actual implementation."""
        if name in ["_impl", "config"] or name.startswith("_"):
            object.__setattr__(self, name, value)
        elif hasattr(self, "_impl") and self._impl is not None:
            setattr(self._impl, name, value)
        else:
            object.__setattr__(self, name, value)


class InternLM2Model(InternLM2PretrainedModel):
    """
    The bare InternLM2 Model outputting raw hidden-states without any specific head.

    This is a proxy that routes to InternLM2 2.0 or 2.5 implementation.
    """

    _auto_class = "AutoModel"

    def __init__(self, config: InternLM2Config):
        super().__init__(config)


class InternLM2ForCausalLM(InternLM2PretrainedModel):
    """
    InternLM2 Model with a language modeling head on top.

    This is a proxy that routes to InternLM2 2.0 or 2.5 implementation.
    """

    _auto_class = "AutoModelForCausalLM"
    _tied_weights_keys = ["output.weight"]

    def __init__(self, config: InternLM2Config):
        super().__init__(config)


class InternLM2ForSequenceClassification(InternLM2PretrainedModel):
    """
    InternLM2 Model with a sequence classification head on top.

    This is a proxy that routes to InternLM2 2.0 or 2.5 implementation.
    """

    _auto_class = "AutoModelForSequenceClassification"

    def __init__(self, config: InternLM2Config):
        super().__init__(config)


class InternLM2ForQuestionAnswering(InternLM2PretrainedModel):
    """
    InternLM2 Model with a question answering head on top.

    This is a proxy that routes to InternLM2 2.0 or 2.5 implementation.
    """

    _auto_class = "AutoModelForQuestionAnswering"

    def __init__(self, config: InternLM2Config):
        super().__init__(config)


class InternLM2ForTokenClassification(InternLM2PretrainedModel):
    """
    InternLM2 Model with a token classification head on top.

    This is a proxy that routes to InternLM2 2.0 or 2.5 implementation.
    """

    _auto_class = "AutoModelForTokenClassification"

    def __init__(self, config: InternLM2Config):
        super().__init__(config)
