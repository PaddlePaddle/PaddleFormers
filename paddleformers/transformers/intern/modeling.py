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

Factory-routing entry point. The classes here are never instantiated as
themselves: `__new__` (direct construction) and `from_pretrained` (loading)
both return a real implementation-class instance from `intern_lm2` (2.0) or
`intern_lm2_5` (2.5), selected via `config.is_version_2_5`.

"""

from paddleformers.transformers.model_utils import PretrainedModel
from paddleformers.utils.log import logger

from .configuration import InternLM2Config


def _select_impl_cls(cls_name, config):
    """Pick the real implementation class by version. Returns the class object."""
    if config.is_version_2_5:
        logger.info("Detected InternLM2 2.5, loading 2.5 implementation")
        from ..intern_lm2_5 import modeling as _impl_module
    else:
        logger.info("Detected InternLM2 2.0, loading 2.0 implementation")
        from ..intern_lm2 import modeling as _impl_module

    impl_cls = getattr(_impl_module, cls_name, None)
    if impl_cls is None:
        raise NotImplementedError(
            f"{cls_name} is not implemented for InternLM2 "
            f"{'2.5' if config.is_version_2_5 else '2.0'} in PaddleFormers."
        )
    return impl_cls


class InternLM2PretrainedModel(PretrainedModel):
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

    def __new__(cls, config, *args, **kwargs):
        impl_cls = _select_impl_cls(cls.__name__, config)
        return impl_cls(config, *args, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        # Read config first (without instantiating) to decide the version,
        # then delegate to the real implementation class's from_pretrained.
        cfg_kwargs = {}
        for k in ("download_hub", "cache_dir", "subfolder", "trust_remote_code"):
            if k in kwargs:
                cfg_kwargs[k] = kwargs[k]
        config = InternLM2Config.from_pretrained(pretrained_model_name_or_path, **cfg_kwargs)
        impl_cls = _select_impl_cls(cls.__name__, config)
        return impl_cls.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

    @classmethod
    def _gen_aoa_config(cls, config):
        impl_cls = _select_impl_cls(cls.__name__, config)
        return impl_cls._gen_aoa_config(config)

    @classmethod
    def _gen_inv_aoa_config(cls, config):
        impl_cls = _select_impl_cls(cls.__name__, config)
        return impl_cls._gen_inv_aoa_config(config)


class InternLM2Model(InternLM2PretrainedModel):
    _auto_class = "AutoModel"


class InternLM2ForCausalLM(InternLM2PretrainedModel):
    _auto_class = "AutoModelForCausalLM"
    _tied_weights_keys = ["output.weight"]


class InternLM2ForSequenceClassification(InternLM2PretrainedModel):
    _auto_class = "AutoModelForSequenceClassification"


class InternLM2ForQuestionAnswering(InternLM2PretrainedModel):
    _auto_class = "AutoModelForQuestionAnswering"


class InternLM2ForTokenClassification(InternLM2PretrainedModel):
    _auto_class = "AutoModelForTokenClassification"
