# coding=utf-8
# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2021 The HuggingFace Inc. team.
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
import inspect
import json
import os
from collections import OrderedDict

from transformers import AutoConfig, PretrainedConfig
from transformers.dynamic_module_utils import (
    get_class_from_dynamic_module,
    resolve_trust_remote_code,
)
from transformers.models.auto.configuration_auto import (
    CONFIG_MAPPING_NAMES,
    model_type_to_module_name,
    replace_list_option_in_docstrings,
)
from transformers.utils import (
    FEATURE_EXTRACTOR_NAME,
    PROCESSOR_NAME,
    VIDEO_PROCESSOR_NAME,
)

from ...utils.download import resolve_file_path
from ..image_processing_utils import ImageProcessingMixin
from ..processing_utils import ProcessorMixin
from ..tokenizer_utils import TOKENIZER_CONFIG_FILE
from ..video_processing_utils import BaseVideoProcessor
from .factory import _LazyAutoMapping
from .image_processing import AutoImageProcessor
from .tokenizer import AutoTokenizer

PROCESSOR_MAPPING_NAMES = OrderedDict(
    [
        ("qwen2_5_vl", "Qwen2_5_VLProcessor"),
        ("qwen2_vl", "Qwen2VLProcessor"),
    ]
)

PROCESSOR_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, PROCESSOR_MAPPING_NAMES)


def processor_class_from_name(class_name: str):
    for module_name, extractors in PROCESSOR_MAPPING_NAMES.items():
        if class_name in extractors:
            module_name = model_type_to_module_name(module_name)

            module = importlib.import_module(f".{module_name}", "paddleformers.transformers")
            try:
                return getattr(module, class_name)
            except AttributeError:
                continue

    for extractor in PROCESSOR_MAPPING._extra_content.values():
        if getattr(extractor, "__name__", None) == class_name:
            return extractor

    # We did not find the class, but maybe it's because a dep is missing. In that case, the class will be in the main
    # init and we return the proper dummy to get an appropriate error message.
    main_module = importlib.import_module("paddleformers.transformers")
    if hasattr(main_module, class_name):
        return getattr(main_module, class_name)

    return None


class AutoProcessor:
    """
    Smart AutoProcessor that automatically adapts based on available dependencies:

    1. **Multi-source support**: Supports HuggingFace, PaddleFormers, and other download sources
    2. **Conditional Paddle integration**: Automatically detects PaddlePaddle availability
    3. **Fallback compatibility**: Works seamlessly with or without Paddle dependencies
    4. **Enhanced functionality**: Extends HuggingFace's standard processor loading logic

    Features:
    - Maintains full compatibility with all HuggingFace processors
    - Supports custom download sources through environment variables
    """

    @classmethod
    @replace_list_option_in_docstrings(PROCESSOR_MAPPING_NAMES)
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        download_hub = kwargs.get("download_hub", None)
        if download_hub is None:
            download_hub = os.environ.get("DOWNLOAD_SOURCE", "huggingface")
            kwargs["download_hub"] = download_hub

        config = kwargs.pop("config", None)
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        kwargs["_from_auto"] = True

        processor_class = None
        processor_auto_map = None

        resolve_file_path_kwargs = {
            key: kwargs[key] for key in inspect.signature(resolve_file_path).parameters if key in kwargs
        }
        resolve_file_path_kwargs.update({"force_return": True})  # do not raise error when file not found

        # Checking whether the processor class is saved in a processor config
        processor_config_file = resolve_file_path(
            pretrained_model_name_or_path,
            PROCESSOR_NAME,
            **resolve_file_path_kwargs,
        )
        if processor_config_file is not None:
            config_dict, _ = ProcessorMixin.get_processor_dict(pretrained_model_name_or_path, **kwargs)
            processor_class = config_dict.get("processor_class")
            if "AutoProcessor" in config_dict.get("auto_map", {}):
                processor_auto_map = config_dict["auto_map"]["AutoProcessor"]

        if processor_class is None:
            # Checking whether the processor class is saved in an image processor config
            preprocessor_config_file = resolve_file_path(
                pretrained_model_name_or_path,
                FEATURE_EXTRACTOR_NAME,
                **resolve_file_path_kwargs,
            )
            if preprocessor_config_file is not None:
                config_dict, _ = ImageProcessingMixin.get_image_processor_dict(pretrained_model_name_or_path, **kwargs)
                processor_class = config_dict.get("processor_class", None)
                if "AutoProcessor" in config_dict.get("auto_map", {}):
                    processor_auto_map = config_dict["auto_map"]["AutoProcessor"]

            # Saved as video processor
            if preprocessor_config_file is None:
                preprocessor_config_file = resolve_file_path(
                    pretrained_model_name_or_path,
                    VIDEO_PROCESSOR_NAME,
                    **resolve_file_path_kwargs,
                )
                if preprocessor_config_file is not None:
                    config_dict, _ = BaseVideoProcessor.get_video_processor_dict(
                        pretrained_model_name_or_path, **kwargs
                    )
                    processor_class = config_dict.get("processor_class", None)
                    if "AutoProcessor" in config_dict.get("auto_map", {}):
                        processor_auto_map = config_dict["auto_map"]["AutoProcessor"]

        if processor_class is None:
            # Checking whether the processor class is saved in a tokenizer
            tokenizer_config_file = resolve_file_path(
                pretrained_model_name_or_path,
                TOKENIZER_CONFIG_FILE,
                **resolve_file_path_kwargs,
            )
            if tokenizer_config_file is not None:
                with open(tokenizer_config_file, encoding="utf-8") as reader:
                    config_dict = json.load(reader)

                processor_class = config_dict.get("processor_class", None)
                if "AutoProcessor" in config_dict.get("auto_map", {}):
                    processor_auto_map = config_dict["auto_map"]["AutoProcessor"]

        if processor_class is None:
            # Otherwise, load config, if it can be loaded.
            if not isinstance(config, PretrainedConfig):
                config = AutoConfig.from_pretrained(
                    pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
                )

            # And check if the config contains the processor class.
            processor_class = getattr(config, "processor_class", None)
            if hasattr(config, "auto_map") and "AutoProcessor" in config.auto_map:
                processor_auto_map = config.auto_map["AutoProcessor"]

        if processor_class is not None:
            processor_class = processor_class_from_name(processor_class)

        has_remote_code = processor_auto_map is not None
        has_local_code = processor_class is not None or type(config) in PROCESSOR_MAPPING

        if has_remote_code:
            if "--" in processor_auto_map:
                upstream_repo = processor_auto_map.split("--")[0]
            else:
                upstream_repo = None
            trust_remote_code = resolve_trust_remote_code(
                trust_remote_code, pretrained_model_name_or_path, has_local_code, has_remote_code, upstream_repo
            )

        if has_remote_code and trust_remote_code:
            processor_class = get_class_from_dynamic_module(
                processor_auto_map, pretrained_model_name_or_path, **kwargs
            )
            _ = kwargs.pop("code_revision", None)
            processor_class.register_for_auto_class()
            return processor_class.from_pretrained(
                pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
            )
        elif processor_class is not None:
            return processor_class.from_pretrained(
                pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
            )
        # Last try: we use the PROCESSOR_MAPPING.
        elif type(config) in PROCESSOR_MAPPING:
            return PROCESSOR_MAPPING[type(config)].from_pretrained(pretrained_model_name_or_path, **kwargs)

        # At this stage, there doesn't seem to be a `Processor` class available for this model, so let's try a
        # tokenizer.
        try:
            return AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
            )
        except Exception:
            try:
                return AutoImageProcessor.from_pretrained(
                    pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
                )
            except Exception:
                pass

        raise ValueError(
            f"Unrecognized processing class in {pretrained_model_name_or_path}. Can't instantiate a processor, a "
            "tokenizer or an image processorfor this model. Make sure the repository contains "
            "the files of at least one of those processing classes."
        )


__all__ = ["PROCESSOR_MAPPING", "AutoProcessor"]
