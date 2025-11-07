# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import json
import os
from collections import OrderedDict
from typing import Optional, Union

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
from transformers.models.auto.video_processing_auto import (
    get_video_processor_config as get_video_processor_config_hf,
)
from transformers.utils import CONFIG_NAME, VIDEO_PROCESSOR_NAME

from ...utils.download import DownloadSource, resolve_file_path
from ...utils.log import logger
from ..video_processing_utils import BaseVideoProcessor
from .factory import _LazyAutoMapping

VIDEO_PROCESSOR_MAPPING_NAMES = OrderedDict(
    [
        ("qwen2_5_vl", "Qwen2VLVideoProcessor"),
        ("qwen2_vl", "Qwen2VLVideoProcessor"),
    ]
)

VIDEO_PROCESSOR_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, VIDEO_PROCESSOR_MAPPING_NAMES)


def video_processor_class_from_name(class_name: str):
    for module_name, extractors in VIDEO_PROCESSOR_MAPPING_NAMES.items():
        if class_name in extractors:
            module_name = model_type_to_module_name(module_name)

            module = importlib.import_module(f".{module_name}", "paddleformers.transformers")
            try:
                return getattr(module, class_name)
            except AttributeError:
                continue

    for extractor in VIDEO_PROCESSOR_MAPPING._extra_content.values():
        if getattr(extractor, "__name__", None) == class_name:
            return extractor

    # We did not find the class, but maybe it's because a dep is missing. In that case, the class will be in the main
    # init and we return the proper dummy to get an appropriate error message.
    main_module = importlib.import_module("paddleformers.transformers")
    if hasattr(main_module, class_name):
        return getattr(main_module, class_name)

    return None


def get_video_processor_config(
    pretrained_model_name_or_path: Union[str, os.PathLike],
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    force_download: bool = False,
    proxies: Optional[dict[str, str]] = None,
    token: Optional[Union[bool, str]] = None,
    revision: Optional[str] = None,
    local_files_only: bool = False,
    **kwargs,
):
    """
    Loads the video processor configuration from a pretrained model video processor configuration.

    Args:
        pretrained_model_name_or_path (`str` or `os.PathLike`):
            This can be either:

            - a string, the *model id* of a pretrained model configuration hosted inside a model repo on
              huggingface.co.
            - a path to a *directory* containing a configuration file saved using the
              [`~PreTrainedTokenizer.save_pretrained`] method, e.g., `./my_model_directory/`.

        cache_dir (`str` or `os.PathLike`, *optional*):
            Path to a directory in which a downloaded pretrained model configuration should be cached if the standard
            cache should not be used.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether or not to force to (re-)download the configuration files and override the cached versions if they
            exist.
        proxies (`dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
        token (`str` or *bool*, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `hf auth login` (stored in `~/.huggingface`).
        revision (`str`, *optional*, defaults to `"main"`):
            The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
            git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
            identifier allowed by git.
        local_files_only (`bool`, *optional*, defaults to `False`):
            If `True`, will only try to load the video processor configuration from local files.

    <Tip>

    Passing `token=True` is required when you want to use a private model.

    </Tip>

    Returns:
        `Dict`: The configuration of the video processor.

    Examples:

    ```python
    # Download configuration from Hugging Face, ModelScope, or AI Studio depending on `download_hub` and cache.
    # By default, `download_hub="huggingface"` will download from huggingface.co.
    video_processor_config = get_video_processor_config("google-bert/bert-base-uncased", download_hub="huggingface")
    # This model does not have an video processor config, so the result will be an empty dict.
    video_processor_config = get_video_processor_config("FacebookAI/xlm-roberta-base")

    # Save a pretrained video processor locally and you can reload its config
    from paddleformers.transformers import AutoVideoProcessor

    video_processor = AutoVideoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-0.5b-ov-hf")
    video_processor.save_pretrained("video-processor-test")
    video_processor = get_video_processor_config("video-processor-test")
    ```"""
    download_hub = kwargs.get("download_hub", None)
    if download_hub is None:
        download_hub = os.environ.get("DOWNLOAD_SOURCE", "huggingface")

    if download_hub == DownloadSource.HUGGINGFACE:
        return get_video_processor_config_hf(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            token=token,
            revision=revision,
            local_files_only=local_files_only,
        )

    try:
        resolved_config_file = resolve_file_path(
            pretrained_model_name_or_path,
            VIDEO_PROCESSOR_NAME,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            token=token,
            revision=revision,
            local_files_only=local_files_only,
            download_hub=download_hub,
        )
    except Exception as e:
        if any(
            keyword in str(e).lower()
            for keyword in ["not exist", "not found", "entrynotfound", "notexist", "does not appear"]
        ):
            hf_link = f"https://huggingface.co/{pretrained_model_name_or_path}"
            modelscope_link = f"https://modelscope.cn/models/{pretrained_model_name_or_path}"
            encoded_model_name = pretrained_model_name_or_path.replace("/", "%2F")
            aistudio_link = f"https://aistudio.baidu.com/modelsoverview?sortBy=weight&q={encoded_model_name}"

            raise ValueError(
                f"Unable to find {VIDEO_PROCESSOR_NAME} in the model repository '{pretrained_model_name_or_path}'. Please check:\n"
                f"The model repository ID is correct for your chosen source:\n"
                f"   - Hugging Face Hub: {hf_link}\n"
                f"   - ModelScope: {modelscope_link}\n"
                f"   - AI Studio: {aistudio_link}\n"
                f"Note: The repository ID may differ between ModelScope, AI Studio, and Hugging Face Hub.\n"
                f"You are currently using the download source: {download_hub}. Please check the repository ID on the official website."
            ) from None
        else:
            raise
    if resolved_config_file is None:
        logger.info(
            "Could not locate the video processor configuration file, will try to use the model config instead."
        )
        return {}

    with open(resolved_config_file, encoding="utf-8") as reader:
        return json.load(reader)


class AutoVideoProcessor:
    """
    Smart AutoVideoProcessor that automatically adapts based on available dependencies:

    1. **Multi-source support**: Supports HuggingFace, PaddleFormers, and other download sources
    2. **Conditional Paddle integration**: Automatically detects PaddlePaddle availability
    3. **Fallback compatibility**: Works seamlessly with or without Paddle dependencies
    4. **Enhanced functionality**: Extends HuggingFace's standard video processor loading logic

    Features:
    - Maintains full compatibility with all HuggingFace videoprocessors
    - Supports custom download sources through environment variables
    """

    @classmethod
    @replace_list_option_in_docstrings(VIDEO_PROCESSOR_MAPPING_NAMES)
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        download_hub = kwargs.get("download_hub", None)
        if download_hub is None:
            download_hub = os.environ.get("DOWNLOAD_SOURCE", "huggingface")
            kwargs["download_hub"] = download_hub

        config = kwargs.pop("config", None)
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        kwargs["_from_auto"] = True

        config_dict, _ = BaseVideoProcessor.get_video_processor_dict(pretrained_model_name_or_path, **kwargs)
        video_processor_class = config_dict.get("video_processor_type", None)
        video_processor_auto_map = None
        if "AutoVideoProcessor" in config_dict.get("auto_map", {}):
            video_processor_auto_map = config_dict["auto_map"]["AutoVideoProcessor"]

        if video_processor_class is None and video_processor_auto_map is None:
            image_processor_class = config_dict.pop("image_processor_type", None)
            if image_processor_class is not None:
                video_processor_class_inferred = image_processor_class.replace("ImageProcessor", "VideoProcessor")

                # Some models have different image processors, e.g. InternVL uses GotOCRImageProcessor
                # We cannot use GotOCRVideoProcessor when falling back for BC and should try to infer from config later on
                if video_processor_class_inferred in VIDEO_PROCESSOR_MAPPING_NAMES.values():
                    video_processor_class = video_processor_class_inferred
            if "AutoImageProcessor" in config_dict.get("auto_map", {}):
                image_processor_auto_map = config_dict["auto_map"]["AutoImageProcessor"]
                video_processor_auto_map = image_processor_auto_map.replace("ImageProcessor", "VideoProcessor")

        # If we don't find the video processor class in the video processor config, let's try the model config.
        if video_processor_class is None and video_processor_auto_map is None:
            if not isinstance(config, PretrainedConfig):
                config = AutoConfig.from_pretrained(
                    pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
                )
            # It could be in `config.video_processor_type``
            video_processor_class = getattr(config, "video_processor_type", None)
            if hasattr(config, "auto_map") and "AutoVideoProcessor" in config.auto_map:
                video_processor_auto_map = config.auto_map["AutoVideoProcessor"]

        if video_processor_class is not None:
            video_processor_class = video_processor_class_from_name(video_processor_class)

        has_remote_code = video_processor_auto_map is not None
        has_local_code = video_processor_class is not None or type(config) in VIDEO_PROCESSOR_MAPPING
        trust_remote_code = resolve_trust_remote_code(
            trust_remote_code, pretrained_model_name_or_path, has_local_code, has_remote_code
        )

        if has_remote_code and trust_remote_code:
            class_ref = video_processor_auto_map
            video_processor_class = get_class_from_dynamic_module(class_ref, pretrained_model_name_or_path, **kwargs)
            _ = kwargs.pop("code_revision", None)
            video_processor_class.register_for_auto_class()
            return video_processor_class.from_dict(config_dict, **kwargs)
        elif video_processor_class is not None:
            return video_processor_class.from_dict(config_dict, **kwargs)
        # Last try: we use the VIDEO_PROCESSOR_MAPPING.
        elif type(config) in VIDEO_PROCESSOR_MAPPING:
            video_processor_class = VIDEO_PROCESSOR_MAPPING[type(config)]

            if video_processor_class is not None:
                return video_processor_class.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
            else:
                raise ValueError("This video processor cannot be instantiated.")

        raise ValueError(
            f"Unrecognized video processor in {pretrained_model_name_or_path}. Should have a "
            f"`video_processor_type` key in its {VIDEO_PROCESSOR_NAME} of {CONFIG_NAME}, or one of the following "
            f"`model_type` keys in its {CONFIG_NAME}: {', '.join(c for c in VIDEO_PROCESSOR_MAPPING_NAMES)}"
        )


__all__ = ["VIDEO_PROCESSOR_MAPPING", "AutoVideoProcessor"]
