# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import importlib
import io
import json
import os
from collections import OrderedDict
from copy import deepcopy

from ...utils.download import resolve_file_path
from ...utils.log import logger
from .. import *  # noqa
from ..configuration_utils import is_standard_config
from .configuration import (
    CONFIG_MAPPING_NAMES,
    MODEL_NAMES_MAPPING,
    AutoConfig,
    PretrainedConfig,
)
from .factory import _LazyAutoMapping

__all__ = [
    "AutoBackbone",
    "AutoModel",
    "AutoModelForPretraining",
    "AutoModelForSequenceClassification",
    "AutoModelForTokenClassification",
    "AutoModelForQuestionAnswering",
    "AutoModelForMultipleChoice",
    "AutoModelForMaskedLM",
    "AutoModelForCausalLM",
    "AutoModelForCausalLMPipe",
    "AutoEncoder",
    "AutoDecoder",
    "AutoGenerator",
    "AutoDiscriminator",
    "AutoModelForConditionalGeneration",
]

MAPPING_NAMES = OrderedDict(
    [
        ("Bert", "bert"),
        ("DeepseekV2", "deepseek_v2"),
        ("DeepseekV3", "deepseek_v3"),
        ("Ernie4_5", "ernie4_5"),
        ("Llama", "llama"),
        ("QWen", "qwen"),
        ("Qwen2", "qwen2"),
        ("Qwen3", "qwen3"),
        ("Qwen2Moe", "qwen2_moe"),
        ("Qwen3Moe", "qwen3_moe"),
    ]
)

MAPPING_TASKS = OrderedDict(
    [
        ("Backbone", "AutoBackbone"),
        ("Model", "AutoModel"),
        ("ForPretraining", "AutoModelForPretraining"),
        ("ForSequenceClassification", "AutoModelForSequenceClassification"),
        ("ForTokenClassification", "AutoModelForTokenClassification"),
        ("ForQuestionAnswering", "AutoModelForQuestionAnswering"),
        ("ForMultipleChoice", "AutoModelForMultipleChoice"),
        ("ForMaskedLM", "AutoModelForMaskedLM"),
        ("ForCausalLM", "AutoModelForCausalLM"),
        ("ForCausalLMPipe", "AutoModelForCausalLMPipe"),
        ("Encoder", "AutoEncoder"),
        ("Decoder", "AutoDecoder"),
        ("Generator", "AutoGenerator"),
        ("Discriminator", "AutoDiscriminator"),
        ("ForConditionalGeneration", "AutoModelForConditionalGeneration"),
    ]
)


MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = OrderedDict([])

MODEL_FOR_CAUSAL_LM_INFERENCE_MAPPING_NAMES = OrderedDict([])

MODEL_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_NAMES_MAPPING)


def get_name_mapping(task="Model"):
    """
    Task can be 'Backbone', 'Model', 'ForPretraining', 'ForSequenceClassification', 'ForTokenClassification',
    'ForQuestionAnswering', 'ForMultipleChoice', 'ForMaskedLM', 'ForCausalLM', 'Encoder', 'Decoder',
    'Generator', 'Discriminator', 'ForConditionalGeneration'
    """
    NAME_MAPPING = OrderedDict()
    for key, value in MAPPING_NAMES.items():
        import_class = key + task
        new_key = key + "Model_Import_Class"
        NAME_MAPPING[new_key] = import_class
        NAME_MAPPING[import_class] = value

    return NAME_MAPPING


def get_task_name(model_class):
    for key, value in MAPPING_TASKS.items():
        if model_class.endswith(key):
            return value
    return None


def get_init_configurations():
    CONFIGURATION_MODEL_MAPPING = OrderedDict()
    for key, class_name in MAPPING_NAMES.items():
        import_class = importlib.import_module(f"paddleformers.transformers.{class_name}.modeling")
        model_name = getattr(import_class, key + "Model")
        if key == "ErnieGen":
            name = tuple(model_name.ernie_gen_pretrained_init_configuration.keys())
        else:
            name = tuple(model_name.pretrained_init_configuration.keys())
        CONFIGURATION_MODEL_MAPPING[name] = key + "Model"

    return CONFIGURATION_MODEL_MAPPING


class _BaseAutoModelClass:
    # Base class for auto models.
    _pretrained_model_dict = None
    _name_mapping = None
    _task_choice = False
    model_config_file = "config.json"
    legacy_model_config_file = "model_config.json"

    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path).`"
        )

    # TODO: Refactor into AutoConfig when available
    @classmethod
    def _get_model_class_from_config(cls, pretrained_model_name_or_path, config_file_path, config=None):
        if config is None:
            with io.open(config_file_path, encoding="utf-8") as f:
                config = json.load(f)

        # Get class name corresponds to this configuration
        if is_standard_config(config):
            architectures = deepcopy(config["architectures"])
            init_class = architectures.pop() if architectures is not None and len(architectures) > 0 else None
        else:
            init_class = config.pop("init_class", None)
        init_class = init_class[:-5] if init_class is not None and init_class.endswith("Model") else init_class

        # Sort the MAPPING_NAMES to reorder the model class names with longest-first rule
        # thus the names with same prefix can be correctly inferred
        # such as QWen and QWen2MOE, QWen2MOE is the longest prefix of QWen2MOEModel
        model_name = None
        SORTED_MAPPING_NAMES = dict(sorted(MAPPING_NAMES.items(), key=lambda x: len(x[0]), reverse=True))
        if init_class:
            for model_flag, name in SORTED_MAPPING_NAMES.items():
                if model_flag in init_class:
                    model_name = model_flag + "Model"
                    break
        else:
            # From pretrained_model_name_or_path
            for model_flag, name in SORTED_MAPPING_NAMES.items():
                if type(pretrained_model_name_or_path) is str and name in pretrained_model_name_or_path.lower():
                    model_name = model_flag + "Model"
                    break
        if model_name is None:
            # Try to get model class from config class
            if not isinstance(config, PretrainedConfig) and pretrained_model_name_or_path is not None:
                config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
            if type(config) in MODEL_MAPPING.keys():
                model_class = MODEL_MAPPING[type(config)]
                if not isinstance(model_class, (list, tuple)):
                    return model_class
            raise AttributeError(
                f"Unable to parse 'architectures' or 'init_class' from {config_file_path}. Also unable to infer model class from 'pretrained_model_name_or_path'"
            )
        init_class = cls._name_mapping[model_name + "_Import_Class"]
        class_name = cls._name_mapping[init_class]
        import_class = importlib.import_module(f"paddleformers.transformers.{class_name}.modeling")
        try:
            model_class = getattr(import_class, init_class)
            return model_class
        except AttributeError as err:
            try:
                new_import_class = importlib.import_module(f"paddleformers.transformers.{class_name}")
                model_class = getattr(new_import_class, init_class)
                return model_class
            except AttributeError:
                logger.error(err)
                all_model_classes = import_class.__all__
                all_tasks = {get_task_name(m) for m in all_model_classes if get_task_name(m) is not None}
                raise AttributeError(
                    f"module '{import_class.__name__}' only supports the following classes: "
                    + ", ".join(m for m in all_model_classes)
                    + "\n"
                    "Hint: you can use interface "
                    + " or ".join(task + ".from_pretrained" for task in all_tasks)
                    + f" to load '{pretrained_model_name_or_path}'\n"
                )

    @classmethod
    def from_config(cls, config, **kwargs):
        model_class = cls._get_model_class_from_config(None, None, config)
        return model_class._from_config(config, **kwargs)

    @classmethod
    def _from_pretrained(cls, pretrained_model_name_or_path, task=None, *model_args, **kwargs):
        if task:
            if cls._task_choice:
                cls._name_mapping = get_name_mapping(task)
            else:
                print("We only support task choice for AutoModel.")
        cache_dir = kwargs.get("cache_dir", None)
        from_hf_hub = kwargs.get("from_hf_hub", False)
        from_aistudio = kwargs.get("from_aistudio", False)
        from_modelscope = kwargs.get("from_modelscope", False)
        subfolder = kwargs.get("subfolder", "")
        if subfolder is None:
            subfolder = ""
        kwargs["cache_dir"] = cache_dir
        kwargs["subfolder"] = subfolder
        all_model_names = []
        for pretrained_model_names, model_name in cls._pretrained_model_dict.items():
            for name in pretrained_model_names:
                all_model_names.append(name)

        # From built-in pretrained models
        if pretrained_model_name_or_path in all_model_names:
            for pretrained_model_names, model_name in cls._pretrained_model_dict.items():
                # From built-in pretrained models
                for pattern in pretrained_model_names:
                    if pattern == pretrained_model_name_or_path:
                        init_class = cls._name_mapping[model_name + "_Import_Class"]
                        class_name = cls._name_mapping[init_class]
                        import_class = importlib.import_module(f"paddleformers.transformers.{class_name}.modeling")
                        try:
                            model_class = getattr(import_class, init_class)
                        except AttributeError as err:
                            try:
                                import_class2 = importlib.import_module(f"paddleformers.transformers.{class_name}")
                                model_class = getattr(import_class2, init_class)
                            except AttributeError:
                                logger.error(err)
                                all_model_classes = import_class.__all__
                                all_tasks = {
                                    get_task_name(m) for m in all_model_classes if get_task_name(m) is not None
                                }
                                raise AttributeError(
                                    f"module '{import_class.__name__}' only supports the following classes: "
                                    + ", ".join(m for m in all_model_classes)
                                    + "\n"
                                    "Hint: you can use interface "
                                    + " or ".join(task + ".from_pretrained" for task in all_tasks)
                                    + f" to load '{pretrained_model_name_or_path}'\n"
                                )
                        logger.info(f"We are using {model_class} to load '{pretrained_model_name_or_path}'.")
                        return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        config_file = resolve_file_path(
            pretrained_model_name_or_path,
            [cls.model_config_file, cls.legacy_model_config_file],
            subfolder,
            cache_dir=cache_dir,
            from_hf_hub=from_hf_hub,
            from_aistudio=from_aistudio,
            from_modelscope=from_modelscope,
        )
        if config_file is not None and os.path.exists(config_file):
            model_class = cls._get_model_class_from_config(pretrained_model_name_or_path, config_file)
            logger.info(f"We are using {model_class} to load '{pretrained_model_name_or_path}'.")
            return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        else:
            raise RuntimeError(
                f"Can't load model for '{pretrained_model_name_or_path}'.\n"
                f"Please make sure that '{pretrained_model_name_or_path}' is:\n"
                "- a correct model-identifier of built-in pretrained models,\n"
                "- or a correct model-identifier of community-contributed pretrained models,\n"
                "- or the correct path to a directory containing relevant model files.\n"
            )

    @classmethod
    def register(cls, config_class, model_class, exist_ok=False):
        """
        Register a new model for this class.
        Args:
            config_class ([`PretrainedConfig`]):
                The configuration corresponding to the model to register.
            model_class ([`PreTrainedModel`]):
                The model to register.
        """
        if hasattr(model_class, "config_class") and model_class.config_class.__name__ != config_class.__name__:
            raise ValueError(
                "The model class you are passing has a `config_class` attribute that is not consistent with the "
                f"config class you passed (model has {model_class.config_class} and you passed {config_class}. Fix "
                "one of those so they match!"
            )
        MODEL_MAPPING.register(config_class, model_class, exist_ok=exist_ok)


class AutoBackbone(_BaseAutoModelClass):
    """
    AutoBackbone.
    """

    CONFIGURATION_MODEL_MAPPING = get_init_configurations()
    _pretrained_model_dict = CONFIGURATION_MODEL_MAPPING
    _name_mapping = get_name_mapping("Backbone")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Creates an instance of `AutoBackbone`. Model weights are loaded
        by specifying name of a built-in pretrained model, or a community contributed model,
        or a local file directory path.

        Args:
            pretrained_model_name_or_path (str): See :class:`AutoModel`.
            *args (tuple): See :class:`AutoModel`.
            **kwargs (dict): See :class:`AutoModel`.

        Returns:
            PretrainedModel: An instance of `AutoBackbone`.

        Example:
            .. code-block::

                from paddleformers.transformers import AutoBackbone

                # Name of built-in pretrained model
                model = AutoBackbone.from_pretrained("google/bit-50")
                print(type(model))
                # <class 'paddleformers.transformers.bit.modeling.BitBackbone'>


                # Load from local directory path
                model = AutoBackbone.from_pretrained("./bit-50")
                print(type(model))
                # <class 'paddleformers.transformers.bit.modeling.BitBackbone'>
        """
        return cls._from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)


class AutoModel(_BaseAutoModelClass):
    """
    AutoClass can help you automatically retrieve the relevant model given the provided
    pretrained weights/vocabulary.
    AutoModel is a generic model class that will be instantiated as one of the base model classes
    when created with the from_pretrained() classmethod.
    """

    CONFIGURATION_MODEL_MAPPING = get_init_configurations()
    _pretrained_model_dict = CONFIGURATION_MODEL_MAPPING
    _name_mapping = get_name_mapping("Model")
    _task_choice = True

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, task=None, *model_args, **kwargs):
        """
        Creates an instance of `AutoModel`. Model weights are loaded
        by specifying name of a built-in pretrained model, a pretrained model on HF, a community contributed model,
        or a local file directory path.

        Args:
            pretrained_model_name_or_path (str): Name of pretrained model or dir path
                to load from. The string can be:

                - Name of a built-in pretrained model
                - Name of a community-contributed pretrained model.
                - Local directory path which contains model weights file("model_state.pdparams")
                  and model config file ("model_config.json").
            task (str): Specify a downstream task. Task can be 'Model', 'ForPretraining',
                'ForSequenceClassification', 'ForTokenClassification', 'ForQuestionAnswering',
                'ForMultipleChoice', 'ForMaskedLM', 'ForCausalLM', 'Encoder', 'Decoder',
                'Generator', 'Discriminator', 'ForConditionalGeneration'.
                We only support specify downstream tasks in AutoModel. Defaults to `None`.
            *args (tuple): Position arguments for model `__init__`. If provided,
                use these as position argument values for model initialization.
            **kwargs (dict): Keyword arguments for model `__init__`. If provided,
                use these to update pre-defined keyword argument values for model
                initialization. If the keyword is in `__init__` argument names of
                base model, update argument values of the base model; else update
                argument values of derived model.

        Returns:
            PretrainedModel: An instance of `AutoModel`.

        Example:
            .. code-block::

                from paddleformers.transformers import AutoModel

                # Name of built-in pretrained model
                model = AutoModel.from_pretrained('bert-base-uncased')
                print(type(model))
                # <class 'paddleformers.transformers.bert.modeling.BertModel'>

                # Name of community-contributed pretrained model
                model = AutoModel.from_pretrained('yingyibiao/bert-base-uncased-sst-2-finetuned')
                print(type(model))
                # <class 'paddleformers.transformers.bert.modeling.BertModel'>

                # Load from local directory path
                model = AutoModel.from_pretrained('./my_bert/')
                print(type(model))
                # <class 'paddleformers.transformers.bert.modeling.BertModel'>

                # choose task
                model = AutoModel.from_pretrained('bert-base-uncased', task='ForPretraining')
                print(type(model))
                # <class 'paddleformers.transformers.bert.modeling.BertForPretraining'>
        """
        return cls._from_pretrained(pretrained_model_name_or_path, task, *model_args, **kwargs)


class AutoModelForPretraining(_BaseAutoModelClass):
    """
    AutoModelForPretraining.
    """

    CONFIGURATION_MODEL_MAPPING = get_init_configurations()
    _pretrained_model_dict = CONFIGURATION_MODEL_MAPPING
    _name_mapping = get_name_mapping("ForPretraining")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Creates an instance of `AutoModelForPretraining`. Model weights are loaded
        by specifying name of a built-in pretrained model, or a community contributed model,
        or a local file directory path.

        Args:
            pretrained_model_name_or_path (str): See :class:`AutoModel`.
            *args (tuple): See :class:`AutoModel`.
            **kwargs (dict): See :class:`AutoModel`.

        Returns:
            PretrainedModel: An instance of `AutoModelForPretraining`.

        Example:
            .. code-block::

                from paddleformers.transformers import AutoModelForPretraining

                # Name of built-in pretrained model
                model = AutoModelForPretraining.from_pretrained('bert-base-uncased')
                print(type(model))
                # <class 'paddleformers.transformers.bert.modeling.BertModelForPretraining'>

                # Name of community-contributed pretrained model
                model = AutoModelForPretraining.from_pretrained('iverxin/bert-base-japanese')
                print(type(model))
                # <class 'paddleformers.transformers.bert.modeling.BertModelForPretraining'>

                # Load from local directory path
                model = AutoModelForPretraining.from_pretrained('./my_bert/')
                print(type(model))
                # <class 'paddleformers.transformers.bert.modeling.BertModelForPretraining'>
        """
        return cls._from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)


class AutoModelForSequenceClassification(_BaseAutoModelClass):
    """
    AutoModelForSequenceClassification.
    """

    CONFIGURATION_MODEL_MAPPING = get_init_configurations()
    _pretrained_model_dict = CONFIGURATION_MODEL_MAPPING
    _name_mapping = get_name_mapping("ForSequenceClassification")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Creates an instance of `AutoModelForSequenceClassification`. Model weights are loaded
        by specifying name of a built-in pretrained model, or a community contributed model,
        or a local file directory path.

        Args:
            pretrained_model_name_or_path (str): See :class:`AutoModel`.
            *args (tuple): See :class:`AutoModel`.
            **kwargs (dict): See :class:`AutoModel`.

        Returns:
            PretrainedModel: An instance of `AutoModelForSequenceClassification`.

        Example:
            .. code-block::

                from paddleformers.transformers import AutoModelForSequenceClassification

                # Name of built-in pretrained model
                model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')
                print(type(model))
                # <class 'paddleformers.transformers.bert.modeling.BertModelForSequenceClassification'>

                # Name of community-contributed pretrained model
                model = AutoModelForSequenceClassification.from_pretrained('iverxin/bert-base-japanese')
                print(type(model))
                # <class 'paddleformers.transformers.bert.modeling.BertModelForSequenceClassification'>

                # Load from local directory path
                model = AutoModelForSequenceClassification.from_pretrained('./my_bert/')
                print(type(model))
                # <class 'paddleformers.transformers.bert.modeling.BertModelForSequenceClassification'>
        """
        return cls._from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)


class AutoModelForTokenClassification(_BaseAutoModelClass):
    """
    AutoModelForTokenClassification.
    """

    CONFIGURATION_MODEL_MAPPING = get_init_configurations()
    _pretrained_model_dict = CONFIGURATION_MODEL_MAPPING
    _name_mapping = get_name_mapping("ForTokenClassification")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Creates an instance of `AutoModelForTokenClassification`. Model weights are loaded
        by specifying name of a built-in pretrained model, or a community contributed model,
        or a local file directory path.

        Args:
            pretrained_model_name_or_path (str): See :class:`AutoModel`.
            *args (tuple): See :class:`AutoModel`.
            **kwargs (dict): See :class:`AutoModel`.

        Returns:
            PretrainedModel: An instance of `AutoModelForTokenClassification`.

        Example:
            .. code-block::

                from paddleformers.transformers import AutoModelForTokenClassification

                # Name of built-in pretrained model
                model = AutoModelForTokenClassification.from_pretrained('bert-base-uncased')
                print(type(model))
                # <class 'paddleformers.transformers.bert.modeling.BertModelForTokenClassification'>

                # Name of community-contributed pretrained model
                model = AutoModelForTokenClassification.from_pretrained('iverxin/bert-base-japanese')
                print(type(model))
                # <class 'paddleformers.transformers.bert.modeling.BertModelForTokenClassification'>

                # Load from local directory path
                model = AutoModelForTokenClassification.from_pretrained('./my_bert/')
                print(type(model))
                # <class 'paddleformers.transformers.bert.modeling.BertModelForTokenClassification'>
        """
        return cls._from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)


class AutoModelForQuestionAnswering(_BaseAutoModelClass):
    """
    AutoModelForQuestionAnswering.
    """

    CONFIGURATION_MODEL_MAPPING = get_init_configurations()
    _pretrained_model_dict = CONFIGURATION_MODEL_MAPPING
    _name_mapping = get_name_mapping("ForQuestionAnswering")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Creates an instance of `AutoModelForQuestionAnswering`. Model weights are loaded
        by specifying name of a built-in pretrained model, or a community contributed model,
        or a local file directory path.

        Args:
            pretrained_model_name_or_path (str): See :class:`AutoModel`.
            *args (tuple): See :class:`AutoModel`.
            **kwargs (dict): See :class:`AutoModel`.

        Returns:
            PretrainedModel: An instance of `AutoModelForQuestionAnswering`.

        Example:
            .. code-block::

                from paddleformers.transformers import AutoModelForQuestionAnswering

                # Name of built-in pretrained model
                model = AutoModelForQuestionAnswering.from_pretrained('bert-base-uncased')
                print(type(model))
                # <class 'paddleformers.transformers.bert.modeling.BertModelForQuestionAnswering'>

                # Name of community-contributed pretrained model
                model = AutoModelForQuestionAnswering.from_pretrained('iverxin/bert-base-japanese')
                print(type(model))
                # <class 'paddleformers.transformers.bert.modeling.BertModelForQuestionAnswering'>

                # Load from local directory path
                model = AutoModelForQuestionAnswering.from_pretrained('./my_bert/')
                print(type(model))
                # <class 'paddleformers.transformers.bert.modeling.BertModelForQuestionAnswering'>
        """
        return cls._from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)


class AutoModelForMultipleChoice(_BaseAutoModelClass):
    """
    AutoModelForMultipleChoice.
    """

    CONFIGURATION_MODEL_MAPPING = get_init_configurations()
    _pretrained_model_dict = CONFIGURATION_MODEL_MAPPING
    _name_mapping = get_name_mapping("ForMultipleChoice")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Creates an instance of `AutoModelForMultipleChoice`. Model weights are loaded
        by specifying name of a built-in pretrained model, or a community contributed model,
        or a local file directory path.

        Args:
            pretrained_model_name_or_path (str): See :class:`AutoModel`.
            *args (tuple): See :class:`AutoModel`.
            **kwargs (dict): See :class:`AutoModel`.

        Returns:
            PretrainedModel: An instance of `AutoModelForMultipleChoice`.

        Example:
            .. code-block::

                from paddleformers.transformers import AutoModelForMultipleChoice

                # Name of built-in pretrained model
                model = AutoModelForMultipleChoice.from_pretrained('bert-base-uncased')
                print(type(model))
                # <class 'paddleformers.transformers.bert.modeling.BertModelForMultipleChoice'>

                # Name of community-contributed pretrained model
                model = AutoModelForMultipleChoice.from_pretrained('iverxin/bert-base-japanese')
                print(type(model))
                # <class 'paddleformers.transformers.bert.modeling.BertModelForMultipleChoice'>

                # Load from local directory path
                model = AutoModelForMultipleChoice.from_pretrained('./my_bert/')
                print(type(model))
                # <class 'paddleformers.transformers.bert.modeling.BertModelForMultipleChoice'>
        """
        return cls._from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)


class AutoModelForMaskedLM(_BaseAutoModelClass):
    """
    AutoModelForMaskedLM.
    """

    CONFIGURATION_MODEL_MAPPING = get_init_configurations()
    _pretrained_model_dict = CONFIGURATION_MODEL_MAPPING
    _name_mapping = get_name_mapping("ForMaskedLM")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Creates an instance of `AutoModelForMaskedLM`. Model weights are loaded
        by specifying name of a built-in pretrained model, or a community contributed model,
        or a local file directory path.

        Args:
            pretrained_model_name_or_path (str): See :class:`AutoModel`.
            *args (tuple): See :class:`AutoModel`.
            **kwargs (dict): See :class:`AutoModel`.

        Returns:
            PretrainedModel: An instance of `AutoModelForMaskedLM`.

        Example:
            .. code-block::

                from paddleformers.transformers import AutoModelForMaskedLM

                # Name of built-in pretrained model
                model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')
                print(type(model))
                # <class 'paddleformers.transformers.bert.modeling.BertModelForMaskedLM'>

                # Name of community-contributed pretrained model
                model = AutoModelForMaskedLM.from_pretrained('iverxin/bert-base-japanese')
                print(type(model))
                # <class 'paddleformers.transformers.bert.modeling.BertModelForMaskedLM'>

                # Load from local directory path
                model = AutoModelForMaskedLM.from_pretrained('./my_bert/')
                print(type(model))
                # <class 'paddleformers.transformers.bert.modeling.BertModelForMaskedLM'>
        """
        return cls._from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)


class AutoModelForCausalLM(_BaseAutoModelClass):
    """
    AutoModelForCausalLM.
    """

    CONFIGURATION_MODEL_MAPPING = get_init_configurations()
    _pretrained_model_dict = CONFIGURATION_MODEL_MAPPING
    _name_mapping = get_name_mapping("ForCausalLM")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Creates an instance of `AutoModelForCausalLM`. Model weights are loaded
        by specifying name of a built-in pretrained model, or a community contributed model,
        or a local file directory path.

        Args:
            pretrained_model_name_or_path (str): See :class:`AutoModel`.
            *args (tuple): See :class:`AutoModel`.
            **kwargs (dict): See :class:`AutoModel`.

        Returns:
            PretrainedModel: An instance of `AutoModelForCausalLM`.

        Example:
            .. code-block::

                from paddleformers.transformers import AutoModelForCausalLM

                # Name of built-in pretrained model
                model = AutoModelForCausalLM.from_pretrained('gpt2-en')
                print(type(model))
                # <class 'paddleformers.transformers.gpt.modeling.GPTLMHeadModel'>

                # Name of community-contributed pretrained model
                model = AutoModelForCausalLM.from_pretrained('junnyu/distilgpt2')
                print(type(model))
                # <class 'paddleformers.transformers.gpt.modeling.GPTLMHeadModel'>

                # Load from local directory path
                model = AutoModelForCausalLM.from_pretrained('./my_gpt/')
                print(type(model))
                # <class 'paddleformers.transformers.gpt.modeling.GPTLMHeadModel'>
        """
        return cls._from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)


class AutoModelForCausalLMPipe(_BaseAutoModelClass):
    """
    Pipeline model for AutoModelForCausalLM.
    """

    CONFIGURATION_MODEL_MAPPING = get_init_configurations()
    _pretrained_model_dict = CONFIGURATION_MODEL_MAPPING
    _name_mapping = get_name_mapping("ForCausalLMPipe")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        return cls._from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)


class AutoEncoder(_BaseAutoModelClass):
    """
    AutoEncoder.
    """

    CONFIGURATION_MODEL_MAPPING = get_init_configurations()
    _pretrained_model_dict = CONFIGURATION_MODEL_MAPPING
    _name_mapping = get_name_mapping("Encoder")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Creates an instance of `AutoEncoder`. Model weights are loaded
        by specifying name of a built-in pretrained model, or a community contributed model,
        or a local file directory path.

        Args:
            pretrained_model_name_or_path (str): See :class:`AutoModel`.
            *args (tuple): See :class:`AutoModel`.
            **kwargs (dict): See :class:`AutoModel`.

        Returns:
            PretrainedModel: An instance of `AutoEncoder`.

        Example:
            .. code-block::

                from paddleformers.transformers import AutoEncoder

                # Name of built-in pretrained model
                model = AutoEncoder.from_pretrained('bart-base',vocab_size=20000)
                print(type(model))
                # <class 'paddleformers.transformers.bart.modeling.BartEncoder'>

                # Load from local directory path
                model = AutoEncoder.from_pretrained('./my_bart/')
                print(type(model))
                # <class 'paddleformers.transformers.bart.modeling.BartEncoder'>
        """
        return cls._from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)


class AutoDecoder(_BaseAutoModelClass):
    """
    AutoDecoder.
    """

    CONFIGURATION_MODEL_MAPPING = get_init_configurations()
    _pretrained_model_dict = CONFIGURATION_MODEL_MAPPING
    _name_mapping = get_name_mapping("Decoder")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Creates an instance of `AutoDecoder`. Model weights are loaded
        by specifying name of a built-in pretrained model, or a community contributed model,
        or a local file directory path.

        Args:
            pretrained_model_name_or_path (str): See :class:`AutoModel`.
            *args (tuple): See :class:`AutoModel`.
            **kwargs (dict): See :class:`AutoModel`.

        Returns:
            PretrainedModel: An instance of `AutoDecoder`.

        Example:
            .. code-block::

                from paddleformers.transformers import AutoDecoder

                # Name of built-in pretrained model
                model = AutoDecoder.from_pretrained('bart-base', vocab_size=20000)
                print(type(model))
                # <class 'paddleformers.transformers.bart.modeling.BartEncoder'>

                # Load from local directory path
                model = AutoDecoder.from_pretrained('./my_bart/')
                print(type(model))
                # <class 'paddleformers.transformers.bart.modeling.BartEncoder'>
        """
        return cls._from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)


class AutoGenerator(_BaseAutoModelClass):
    """
    AutoGenerator.
    """

    CONFIGURATION_MODEL_MAPPING = get_init_configurations()
    _pretrained_model_dict = CONFIGURATION_MODEL_MAPPING
    _name_mapping = get_name_mapping("Generator")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Creates an instance of `AutoGenerator`. Model weights are loaded
        by specifying name of a built-in pretrained model, or a community contributed model,
        or a local file directory path.

        Args:
            pretrained_model_name_or_path (str): See :class:`AutoModel`.
            *args (tuple): See :class:`AutoModel`.
            **kwargs (dict): See :class:`AutoModel`.

        Returns:
            PretrainedModel: An instance of `AutoGenerator`.

        Example:
            .. code-block::

                from paddleformers.transformers import AutoGenerator

                # Name of built-in pretrained model
                model = AutoGenerator.from_pretrained('electra-small')
                print(type(model))
                # <class 'paddleformers.transformers.electra.modeling.ElectraGenerator'>

                # Name of community-contributed pretrained model
                model = AutoGenerator.from_pretrained('junnyu/hfl-chinese-legal-electra-small-generator')
                print(type(model))
                # <class 'paddleformers.transformers.electra.modeling.ElectraGenerator'>

                # Load from local directory path
                model = AutoGenerator.from_pretrained('./my_electra/')
                print(type(model))
                # <class 'paddleformers.transformers.electra.modeling.ElectraGenerator'>
        """
        return cls._from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)


class AutoDiscriminator(_BaseAutoModelClass):
    """
    AutoDiscriminator.
    """

    CONFIGURATION_MODEL_MAPPING = get_init_configurations()
    _pretrained_model_dict = CONFIGURATION_MODEL_MAPPING
    _name_mapping = get_name_mapping("Discriminator")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Creates an instance of `AutoDiscriminator`. Model weights are loaded
        by specifying name of a built-in pretrained model, or a community contributed model,
        or a local file directory path.

        Args:
            pretrained_model_name_or_path (str): See :class:`AutoModel`.
            *args (tuple): See :class:`AutoModel`.
            **kwargs (dict): See :class:`AutoModel`.

        Returns:
            PretrainedModel: An instance of `AutoDiscriminator`.

        Example:
            .. code-block::

                from paddleformers.transformers import AutoDiscriminator

                # Name of built-in pretrained model
                model = AutoDiscriminator.from_pretrained('electra-small')
                print(type(model))
                # <class 'paddleformers.transformers.electra.modeling.ElectraDiscriminator'>

                # Name of community-contributed pretrained model
                model = AutoDiscriminator.from_pretrained('junnyu/hfl-chinese-legal-electra-small-generator')
                print(type(model))
                # <class 'paddleformers.transformers.electra.modeling.ElectraDiscriminator'>

                # Load from local directory path
                model = AutoDiscriminator.from_pretrained('./my_electra/')
                print(type(model))
                # <class 'paddleformers.transformers.electra.modeling.ElectraDiscriminator'>
        """
        return cls._from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)


class AutoModelForConditionalGeneration(_BaseAutoModelClass):
    """
    AutoModelForConditionalGeneration.
    """

    CONFIGURATION_MODEL_MAPPING = get_init_configurations()
    _pretrained_model_dict = CONFIGURATION_MODEL_MAPPING
    _name_mapping = get_name_mapping("ForConditionalGeneration")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Creates an instance of `AutoModelForConditionalGeneration`. Model weights are loaded
        by specifying name of a built-in pretrained model, or a community contributed model,
        or a local file directory path.

        Args:
            pretrained_model_name_or_path (str): See :class:`AutoModel`.
            *args (tuple): See :class:`AutoModel`.
            **kwargs (dict): See :class:`AutoModel`.

        Returns:
            PretrainedModel: An instance of `AutoModelForConditionalGeneration`.

        Example:
            .. code-block::

                from paddleformers.transformers import AutoModelForConditionalGeneration

                # Name of built-in pretrained model
                model = AutoModelForConditionalGeneration.from_pretrained('bart-base')
                print(type(model))
                # <class 'paddleformers.transformers.bart.modeling.BartForConditionalGeneration'>


                # Load from local directory path
                model = AutoModelForConditionalGeneration.from_pretrained('./my_bart/')
                print(type(model))
                # <class 'paddleformers.transformers.bart.modeling.BartForConditionalGeneration'>
        """
        return cls._from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
