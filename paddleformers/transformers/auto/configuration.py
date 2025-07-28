# Copyright 2018 Google AI, Google Brain and the HuggingFace Inc. team.
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
from __future__ import annotations

import importlib
import inspect
import io
import json
import os
from collections import OrderedDict, defaultdict
from typing import Dict, List, Type

from ...utils.download import resolve_file_path
from ...utils.import_utils import import_module
from ...utils.log import logger
from ..configuration_utils import PretrainedConfig
from ..model_utils import PretrainedModel

__all__ = [
    "AutoConfig",
]

CONFIG_MAPPING_NAMES = OrderedDict(
    [
        ("bert", "BertConfig"),
        ("deepseek_v2", "DeepseekV2Config"),
        ("deepseek_v3", "DeepseekV3Config"),
        ("ernie4_5", "Ernie4_5Config"),
        ("llama", "LlamaConfig"),
        ("qwen", "QWenConfig"),
        ("qwen2", "Qwen2Config"),
        ("qwen2_moe", "Qwen2MoeConfig"),
        ("qwen3", "Qwen3Config"),
        ("qwen3_moe", "Qwen3MoeConfig"),
    ]
)


MODEL_NAMES_MAPPING = OrderedDict(
    # Base model mapping
    [
        ("bert", "Bert"),
        ("deepseek_v2", "DeepseekV2"),
        ("deepseek_v3", "DeepseekV3"),
        ("llama", "Llama"),
        ("qwen", "QWen"),
        ("qwen2", "Qwen2"),
        ("qwen2_moe", "Qwen2Moe"),
        ("qwen3", "Qwen3"),
        ("qwen3_moe", "Qwen3Moe"),
    ]
)

MULTI_MODELS_MAPPING = OrderedDict(
    # multi models mapping
    []
)


def config_class_to_model_type(config):
    """Converts a config class name to the corresponding model type"""
    for key, cls in CONFIG_MAPPING_NAMES.items():
        if cls == config:
            return key
    # if key not found check in extra content
    for key, cls in CONFIG_MAPPING._extra_content.items():
        if cls.__name__ == config:
            return key
    return None


class _LazyConfigMapping(OrderedDict):
    """
    A dictionary that lazily load its values when they are requested.
    """

    def __init__(self, mapping):
        self._mapping = mapping
        self._extra_content = {}
        self._modules = {}

    def __getitem__(self, key):
        # NOTE: (changwenbin) This is to enable the qwen2_vl language model to use qwen2 reasoning optimization
        for model_type, model_key in MULTI_MODELS_MAPPING.items():
            if key == model_type:
                key = model_key
        if key in self._extra_content:
            return self._extra_content[key]
        if key not in self._mapping:
            raise KeyError(key)
        value = self._mapping[key]
        module_name = model_type_to_module_name(key)
        if module_name not in self._modules:
            self._modules[module_name] = importlib.import_module(
                f".{module_name}.configuration", "paddleformers.transformers"
            )
        if hasattr(self._modules[module_name], value):
            return getattr(self._modules[module_name], value)

        # Some of the mappings have entries model_type -> config of another model type. In that case we try to grab the
        # object at the top level.
        transformers_module = importlib.import_module("paddleformers")
        return getattr(transformers_module, value)

    def keys(self):
        return list(self._mapping.keys()) + list(self._extra_content.keys())

    def values(self):
        return [self[k] for k in self._mapping.keys()] + list(self._extra_content.values())

    def items(self):
        return [(k, self[k]) for k in self._mapping.keys()] + list(self._extra_content.items())

    def __iter__(self):
        return iter(list(self._mapping.keys()) + list(self._extra_content.keys()))

    def __contains__(self, item):
        return item in self._mapping or item in self._extra_content

    def register(self, key, value, exist_ok=False):
        """
        Register a new configuration in this mapping.
        """
        if key in self._mapping.keys() and not exist_ok:
            raise ValueError(f"'{key}' is already used by a Transformers config, pick another name.")
        self._extra_content[key] = value


CONFIG_MAPPING = _LazyConfigMapping(CONFIG_MAPPING_NAMES)


def get_configurations() -> Dict[str, List[Type[PretrainedConfig]]]:
    """load the configurations of PretrainedConfig mapping: {<model-name>: [<class-name>, <class-name>, ...], }

    Returns:
        dict[str, str]: the mapping of model-name to model-classes
    """
    # 1. search the subdir<model-name> to find model-names
    transformers_dir = os.path.dirname(os.path.dirname(__file__))
    exclude_models = ["auto"]

    mappings = defaultdict(list)
    for model_name in os.listdir(transformers_dir):
        if model_name in exclude_models:
            continue

        model_dir = os.path.join(transformers_dir, model_name)
        if not os.path.isdir(model_dir):
            continue

        # 2. find the `configuration.py` file as the identifier of PretrainedConfig class
        configuration_path = os.path.join(model_dir, "configuration.py")
        if not os.path.exists(configuration_path):
            continue

        configuration_module = import_module(f"paddleformers.transformers.{model_name}.configuration")
        for key in dir(configuration_module):
            value = getattr(configuration_module, key)
            if inspect.isclass(value) and issubclass(value, PretrainedConfig):
                mappings[model_name].append(value)

    return mappings


def model_type_to_module_name(key):
    """Converts a config key to the corresponding module."""
    key = key.replace("-", "_")
    return key


class AutoConfig(PretrainedConfig):
    """
    AutoConfig is a generic config class that will be instantiated as one of the
    base PretrainedConfig classes when created with the AutoConfig.from_pretrained() classmethod.
    """

    MAPPING_NAMES: Dict[str, List[Type[PretrainedConfig]]] = get_configurations()

    # cache the builtin pretrained-model-name to Model Class
    name2class = None
    config_file = "config.json"

    # TODO(wj-Mcat): the supporting should be removed after v2.6
    legacy_config_file = "config.json"

    @classmethod
    def _get_config_class_from_config(
        cls, pretrained_model_name_or_path: str, config_file_path: str
    ) -> PretrainedConfig:
        with io.open(config_file_path, encoding="utf-8") as f:
            config = json.load(f)

        # add support for legacy config
        if "init_class" in config:
            architectures = [config.pop("init_class")]
        else:
            architectures = config.pop("architectures", None)
            if architectures is None:
                return cls

        model_name = architectures[0]
        model_class = import_module(f"paddleformers.transformers.{model_name}")

        # To make AutoConfig support loading config with custom model_class
        # which is not in paddleformers.transformers. Using "model_type" to load
        # here actually conforms to what PretrainedConfig doc describes.
        if model_class is None and "model_type" in config:
            model_type = config["model_type"]
            # MAPPING_NAMES is a dict with item like ('llama', [LlamaConfig, PretrainedConfig])
            for config_class in cls.MAPPING_NAMES[model_type]:
                if config_class is not PretrainedConfig:
                    model_config_class = config_class
                    return model_config_class

        assert inspect.isclass(model_class) and issubclass(
            model_class, PretrainedModel
        ), f"<{model_class}> should be a PretarinedModel class, but <{type(model_class)}>"

        return cls if model_class.config_class is None else model_class.config_class

    @classmethod
    def from_file(cls, config_file: str, **kwargs) -> AutoConfig:
        """construct configuration with AutoConfig class to enable normal loading

        Args:
            config_file (str): the path of config file

        Returns:
            AutoConfig: the instance of AutoConfig
        """
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)

        config.update(kwargs)
        return cls(**config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *model_args, **kwargs):
        """
        Creates an instance of `AutoConfig`. Related resources are loaded by
        specifying name of a built-in pretrained model, or a community-contributed
        pretrained model, or a local file directory path.

        Args:
            pretrained_model_name_or_path (str): Name of pretrained model or dir path
                to load from. The string can be:

                - Name of built-in pretrained model
                - Name of a community-contributed pretrained model.
                - Local directory path which contains processor related resources
                  and processor config file ("processor_config.json").
            *args (tuple): position arguments for model `__init__`. If provided,
                use these as position argument values for processor initialization.
            **kwargs (dict): keyword arguments for model `__init__`. If provided,
                use these to update pre-defined keyword argument values for processor
                initialization.

        Returns:
            PretrainedConfig: An instance of `PretrainedConfig`.


        Example:
            .. code-block::
            from paddleformers.transformers import AutoConfig
            config = AutoConfig.from_pretrained("bert-base-uncased")
            config.save_pretrained('./bert-base-uncased')
        """

        if not cls.name2class:
            cls.name2class = {}
            for model_classes in cls.MAPPING_NAMES.values():
                for model_class in model_classes:
                    cls.name2class.update(
                        {model_name: model_class for model_name in model_class.pretrained_init_configuration.keys()}
                    )

        # From built-in pretrained models
        if pretrained_model_name_or_path in cls.name2class:
            return cls.name2class[pretrained_model_name_or_path].from_pretrained(
                pretrained_model_name_or_path, *model_args, **kwargs
            )

        subfolder = kwargs.get("subfolder", "")
        if subfolder is None:
            subfolder = ""
        from_hf_hub = kwargs.get("from_hf_hub", False)
        from_aistudio = kwargs.get("from_aistudio", False)
        from_modelscope = kwargs.get("from_modelscope", False)
        cache_dir = kwargs.pop("cache_dir", None)

        config_file = resolve_file_path(
            pretrained_model_name_or_path,
            [cls.config_file, cls.legacy_config_file],
            subfolder,
            cache_dir=cache_dir,
            from_hf_hub=from_hf_hub,
            from_aistudio=from_aistudio,
            from_modelscope=from_modelscope,
        )
        config_dict, unused_kwargs = PretrainedConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)
        if "model_type" in config_dict:
            try:
                config_class = CONFIG_MAPPING[config_dict["model_type"]]
            except KeyError:
                raise ValueError(
                    f"The checkpoint you are trying to load has model type `{config_dict['model_type']}` "
                    "but Transformers does not recognize this architecture. This could be because of an "
                    "issue with the checkpoint, or because your version of Transformers is out of date."
                )
            return config_class.from_dict(config_dict, **unused_kwargs)
        elif "model_type" not in config_dict and config_file is not None and os.path.exists(config_file):
            config_class = cls._get_config_class_from_config(pretrained_model_name_or_path, config_file)
            logger.info("We are using %s to load '%s'." % (config_class, pretrained_model_name_or_path))
            if config_class is cls:
                return cls.from_file(config_file)
            return config_class.from_pretrained(config_file, *model_args, **kwargs)
        elif config_file is None:
            # Fallback: use pattern matching on the string.
            # We go from longer names to shorter names to catch roberta before bert (for instance)
            for pattern in sorted(CONFIG_MAPPING.keys(), key=len, reverse=True):
                if pattern in str(pretrained_model_name_or_path):
                    return CONFIG_MAPPING[pattern].from_dict(config_dict, **unused_kwargs)
        else:
            raise RuntimeError(
                f"Can't load config for '{pretrained_model_name_or_path}'.\n"
                f"Please make sure that '{pretrained_model_name_or_path}' is:\n"
                "- a correct model-identifier of built-in pretrained models,\n"
                "- or a correct model-identifier of community-contributed pretrained models,\n"
                "- or the correct path to a directory containing relevant config files.\n"
            )

    @staticmethod
    def register(model_type, config, exist_ok=False):
        """
        Register a new configuration for this class.

        Args:
            model_type (`str`): The model type like "bert" or "gpt".
            config ([`PretrainedConfig`]): The config to register.
        """
        if issubclass(config, PretrainedConfig) and config.model_type != model_type:
            raise ValueError(
                "The config you are passing has a `model_type` attribute that is not consistent with the model type "
                f"you passed (config has {config.model_type} and you passed {model_type}. Fix one of those so they "
                "match!"
            )
        CONFIG_MAPPING.register(model_type, config, exist_ok=exist_ok)
