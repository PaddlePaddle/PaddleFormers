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
    _no_split_modules = ["InternLM2DecoderLayer", "InternLM25DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    transpose_weight_keys = ["wqkv", "wo", "w1", "w2", "w3", "output"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def __init__(self, config: InternLM2Config):
        super().__init__(config)

        if config.is_version_2_5:
            logger.info("Detected InternLM2 2.5, loading 2.5 implementation")
            from ..intern_lm2_5 import modeling as _impl_module
        else:
            logger.info("Detected InternLM2 2.0, loading 2.0 implementation")
            from ..intern_lm2 import modeling as _impl_module

        _cls_name = self.__class__.__name__
        if not hasattr(_impl_module, _cls_name):
            raise NotImplementedError(
                f"{_cls_name} is not implemented for InternLM2 "
                f"{'2.5' if config.is_version_2_5 else '2.0'} in PaddleFormers."
            )
        ImplModel = getattr(_impl_module, _cls_name)

        impl = ImplModel(config)
        self.add_sublayer("_impl", impl)
        object.__setattr__(self, "_impl", impl)

    @classmethod
    def _gen_aoa_config(cls, config):
        if config.is_version_2_5:
            from ..intern_lm2_5 import modeling as impl_module
        else:
            from ..intern_lm2 import modeling as impl_module
        impl_cls = getattr(impl_module, cls.__name__)
        return impl_cls._gen_aoa_config(config)

    @classmethod
    def _gen_inv_aoa_config(cls, config):
        if config.is_version_2_5:
            from ..intern_lm2_5 import modeling as impl_module
        else:
            from ..intern_lm2 import modeling as impl_module
        impl_cls = getattr(impl_module, cls.__name__)
        return impl_cls._gen_inv_aoa_config(config)

    def forward(self, *args, **kwargs):
        return self._impl(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        return self._impl.state_dict(*args, **kwargs)

    def set_state_dict(self, state_dict, *args, **kwargs):
        return self._impl.set_state_dict(state_dict, *args, **kwargs)

    def parameters(self, include_sublayers=True):
        return self._impl.parameters(include_sublayers=include_sublayers)

    def named_parameters(self, prefix="", include_sublayers=True):
        return self._impl.named_parameters(prefix=prefix, include_sublayers=include_sublayers)

    def sharded_state_dict(self, *args, **kwargs):
        return self._impl.sharded_state_dict(*args, **kwargs)

    def get_input_embeddings(self):
        return self._impl.get_input_embeddings()

    def set_input_embeddings(self, value):
        return self._impl.set_input_embeddings(value)

    def __getattr__(self, name):
        """Proxy all attribute access to the actual implementation."""
        if name in ["_impl", "config"]:
            return object.__getattribute__(self, name)
        if name.startswith("_"):
            try:
                return object.__getattribute__(self, name)
            except AttributeError:
                return getattr(self._impl, name)
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

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self._impl.prepare_inputs_for_generation(*args, **kwargs)

    def get_output_embeddings(self):
        return self._impl.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        return self._impl.set_output_embeddings(new_embeddings)


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
