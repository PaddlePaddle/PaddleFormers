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

import json
import os
from typing import Optional, Union

import transformers as hf
from transformers import AutoConfig, ImageProcessingMixin, PretrainedConfig
from transformers.dynamic_module_utils import (
    get_class_from_dynamic_module,
    resolve_trust_remote_code,
)
from transformers.models.auto.configuration_auto import (
    CONFIG_MAPPING_NAMES,
    replace_list_option_in_docstrings,
)
from transformers.models.auto.image_processing_auto import (
    IMAGE_PROCESSOR_MAPPING_NAMES,
    get_image_processor_class_from_name,
)
from transformers.models.auto.image_processing_auto import (
    get_image_processor_config as get_image_processor_config_hf,
)
from transformers.utils import (
    CONFIG_NAME,
    IMAGE_PROCESSOR_NAME,
    is_timm_config_dict,
    is_timm_local_checkpoint,
)

from ...utils.download import DownloadSource, resolve_file_path
from ...utils.log import logger
from ..image_processing_utils import PaddleImageProcessingMixin
from .factory import _LazyAutoMapping

IMAGE_PROCESSOR_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, IMAGE_PROCESSOR_MAPPING_NAMES)


def get_image_processor_config(
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
    Loads the image processor configuration from a pretrained model image processor configuration.

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
            If `True`, will only try to load the image processor configuration from local files.

    <Tip>

    Passing `token=True` is required when you want to use a private model.

    </Tip>

    Returns:
        `Dict`: The configuration of the image processor.

    Examples:

    ```python
    # Download configuration from Hugging Face, ModelScope, or AI Studio depending on `download_hub` and cache.
    # By default, `download_hub="huggingface"` will download from huggingface.co.
    image_processor_config = get_image_processor_config("google-bert/bert-base-uncased", download_hub="huggingface")
    # This model does not have an image processor config, so the result will be an empty dict.
    image_processor_config = get_image_processor_config("FacebookAI/xlm-roberta-base")

    # Save a pretrained image processor locally and you can reload its config
    from transformers import AutoTokenizer

    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k", download_hub="huggingface")
    image_processor.save_pretrained("image-processor-test")
    image_processor_config = get_image_processor_config("image-processor-test")
    ```"""
    download_hub = kwargs.get("download_hub", None)
    if download_hub is None:
        download_hub = os.environ.get("DOWNLOAD_SOURCE", "huggingface")

    if download_hub == DownloadSource.HUGGINGFACE:
        return get_image_processor_config_hf(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            token=token,
            revision=revision,
            local_files_only=local_files_only,
            **kwargs,
        )

    try:
        resolved_config_file = resolve_file_path(
            pretrained_model_name_or_path,
            IMAGE_PROCESSOR_NAME,
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
                f"Unable to find {IMAGE_PROCESSOR_NAME} in the model repository '{pretrained_model_name_or_path}'. Please check:\n"
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
            "Could not locate the image processor configuration file, will try to use the model config instead."
        )
        return {}

    with open(resolved_config_file, encoding="utf-8") as reader:
        return json.load(reader)


def _bind_paddle_mixin_if_available(image_processor_class):
    """
    Bind the PaddleImageProcessingMixin if Paddle is available; otherwise, return the original class.

    Args:
        image_processor_class: The original image processor class.

    Returns:
        The tokenizer class bound with PaddleImageProcessingMixin, or the original class.
    """
    return type(image_processor_class.__name__, (PaddleImageProcessingMixin, image_processor_class), {})


class AutoImageProcessor(hf.AutoImageProcessor):
    """
    Smart AutoImageProcessor that automatically adapts based on available dependencies:

    1. **Multi-source support**: Supports HuggingFace, PaddleFormers, and other download sources
    2. **Conditional Paddle integration**: Automatically detects PaddlePaddle availability
    3. **Fallback compatibility**: Works seamlessly with or without Paddle dependencies
    4. **Enhanced functionality**: Extends HuggingFace's standard tokenizer loading logic

    Features:
    - Automatically binds PaddleImageProcessingMixin when PaddlePaddle is available
    - Falls back to pure Transformers mode when PaddlePaddle is not available
    - Maintains full compatibility with all HuggingFace tokenizers
    - Supports custom download sources through environment variables
    """

    @classmethod
    @replace_list_option_in_docstrings(IMAGE_PROCESSOR_MAPPING_NAMES)
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        download_hub = kwargs.get("download_hub", None)
        if download_hub is None:
            download_hub = os.environ.get("DOWNLOAD_SOURCE", "huggingface")
            kwargs["download_hub"] = download_hub

        config = kwargs.pop("config", None)
        use_fast = kwargs.pop("use_fast", True)
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        kwargs["_from_auto"] = True

        # Resolve the image processor config filename
        if "image_processor_filename" in kwargs:
            image_processor_filename = kwargs.pop("image_processor_filename")
        elif is_timm_local_checkpoint(pretrained_model_name_or_path):
            image_processor_filename = CONFIG_NAME
        else:
            image_processor_filename = IMAGE_PROCESSOR_NAME

        # Load the image processor config
        try:
            # Main path for all transformers models and local TimmWrapper checkpoints
            if download_hub == DownloadSource.HUGGINGFACE:
                config_dict, _ = ImageProcessingMixin.get_image_processor_dict(
                    pretrained_model_name_or_path, image_processor_filename=image_processor_filename, **kwargs
                )
            else:
                config_dict, _ = PaddleImageProcessingMixin.get_image_processor_dict(
                    pretrained_model_name_or_path, image_processor_filename=image_processor_filename, **kwargs
                )
        except Exception as initial_exception:
            # Fallback path for Hub TimmWrapper checkpoints. Timm models' image processing is saved in `config.json`
            # instead of `preprocessor_config.json`. Because this is an Auto class and we don't have any information
            # except the model name, the only way to check if a remote checkpoint is a timm model is to try to
            # load `config.json` and if it fails with some error, we raise the initial exception.
            try:
                if download_hub == DownloadSource.HUGGINGFACE:
                    config_dict, _ = ImageProcessingMixin.get_image_processor_dict(
                        pretrained_model_name_or_path, image_processor_filename=CONFIG_NAME, **kwargs
                    )
                else:
                    config_dict, _ = PaddleImageProcessingMixin.get_image_processor_dict(
                        pretrained_model_name_or_path, image_processor_filename=CONFIG_NAME, **kwargs
                    )
            except Exception:
                raise initial_exception

            # In case we have a config_dict, but it's not a timm config dict, we raise the initial exception,
            # because only timm models have image processing in `config.json`.
            if not is_timm_config_dict(config_dict):
                raise initial_exception

        image_processor_type = config_dict.get("image_processor_type", None)
        image_processor_auto_map = None
        if "AutoImageProcessor" in config_dict.get("auto_map", {}):
            image_processor_auto_map = config_dict["auto_map"]["AutoImageProcessor"]

        # If we still don't have the image processor class, check if we're loading from a previous feature extractor config
        # and if so, infer the image processor class from there.
        if image_processor_type is None and image_processor_auto_map is None:
            feature_extractor_class = config_dict.pop("feature_extractor_type", None)
            if feature_extractor_class is not None:
                image_processor_type = feature_extractor_class.replace("FeatureExtractor", "ImageProcessor")
            if "AutoFeatureExtractor" in config_dict.get("auto_map", {}):
                feature_extractor_auto_map = config_dict["auto_map"]["AutoFeatureExtractor"]
                image_processor_auto_map = feature_extractor_auto_map.replace("FeatureExtractor", "ImageProcessor")

        # If we don't find the image processor class in the image processor config, let's try the model config.
        if image_processor_type is None and image_processor_auto_map is None:
            if not isinstance(config, PretrainedConfig):
                config = AutoConfig.from_pretrained(
                    pretrained_model_name_or_path,
                    trust_remote_code=trust_remote_code,
                    **kwargs,
                )
            # It could be in `config.image_processor_type``
            image_processor_type = getattr(config, "image_processor_type", None)
            if hasattr(config, "auto_map") and "AutoImageProcessor" in config.auto_map:
                image_processor_auto_map = config.auto_map["AutoImageProcessor"]

        image_processor_class = None
        if image_processor_type is not None:
            if use_fast is None or use_fast:
                if use_fast:
                    if image_processor_type.endswith("Fast"):
                        logger.warning(
                            f"The model's image processor only supports the slow version (`use_fast=False`). "
                            f"Detected `use_fast=True` but will fall back to the slow version: "
                            f"'{image_processor_type}' will be loaded as '{image_processor_type[:-4]}'."
                        )
                    else:
                        logger.warning(
                            f"The requested image processor `{image_processor_type}` does not have a fast version. "
                            f"Detected `use_fast=True` but the slow version will be loaded instead (`use_fast=False`)."
                        )

                use_fast = False
            if not use_fast:
                image_processor_type_slow = image_processor_type.removesuffix("Fast")
                image_processor_class = get_image_processor_class_from_name(image_processor_type_slow)

                if image_processor_class is None:
                    if image_processor_type.endswith("Fast"):
                        raise ValueError(
                            f"The slow version of `{image_processor_type}` (i.e., "
                            f"`{image_processor_type_slow}`) could not be found."
                        )
                    else:
                        raise ValueError(
                            f"The image processor class `{image_processor_type_slow}` could not be found."
                        )

        has_remote_code = image_processor_auto_map is not None
        has_local_code = image_processor_class is not None or type(config) in IMAGE_PROCESSOR_MAPPING
        if has_remote_code:
            if image_processor_auto_map is not None and not isinstance(image_processor_auto_map, tuple):
                # In some configs, only the slow image processor class is stored
                image_processor_auto_map = (image_processor_auto_map, None)
            if use_fast and image_processor_auto_map[1] is not None:
                class_ref = image_processor_auto_map[1]
            else:
                class_ref = image_processor_auto_map[0]
            if "--" in class_ref:
                upstream_repo = class_ref.split("--")[0]
            else:
                upstream_repo = None
            trust_remote_code = resolve_trust_remote_code(
                trust_remote_code, pretrained_model_name_or_path, has_local_code, has_remote_code, upstream_repo
            )

        if has_remote_code and trust_remote_code:
            if not use_fast and image_processor_auto_map[1] is not None:
                logger.warning(
                    f"Fast image processor class {image_processor_auto_map[1]} is available for this model. "
                    "Using slow image processor class. To use the fast image processor class set `use_fast=True`."
                )

            image_processor_class = get_class_from_dynamic_module(class_ref, pretrained_model_name_or_path, **kwargs)
            _ = kwargs.pop("code_revision", None)
            image_processor_class.register_for_auto_class()
            # Bind PaddleImageProcessingMixin
            image_processor_class = _bind_paddle_mixin_if_available(image_processor_class)
            return image_processor_class.from_dict(config_dict, **kwargs)
        elif image_processor_class is not None:
            # Bind PaddleImageProcessingMixin
            image_processor_class = _bind_paddle_mixin_if_available(image_processor_class)
            return image_processor_class.from_dict(config_dict, **kwargs)
        # Last try: we use the IMAGE_PROCESSOR_MAPPING.
        elif type(config) in IMAGE_PROCESSOR_MAPPING:
            image_processor_tuple = IMAGE_PROCESSOR_MAPPING[type(config)]

            image_processor_class_py, image_processor_class_fast = image_processor_tuple

            if not use_fast and image_processor_class_fast is not None:
                logger.warning(
                    f"Fast image processor class {image_processor_class_fast} is available for this model. "
                    "Using slow image processor class. To use the fast image processor class set `use_fast=True`."
                )

            if image_processor_class_fast and (use_fast or image_processor_class_py is None):

                # Bind PaddleImageProcessingMixin
                image_processor_class_fast = _bind_paddle_mixin_if_available(image_processor_class_fast)
                return image_processor_class_fast.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
            else:
                if image_processor_class_py is not None:
                    # Bind PaddleImageProcessingMixin
                    image_processor_class_py = _bind_paddle_mixin_if_available(image_processor_class_py)
                    return image_processor_class_py.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
                else:
                    raise ValueError(
                        "This image processor cannot be instantiated. Please make sure you have `Pillow` installed."
                    )
        raise ValueError(
            f"Unrecognized image processor in {pretrained_model_name_or_path}. Should have a "
            f"`image_processor_type` key in its {IMAGE_PROCESSOR_NAME} of {CONFIG_NAME}, or one of the following "
            f"`model_type` keys in its {CONFIG_NAME}: {', '.join(c for c in IMAGE_PROCESSOR_MAPPING_NAMES)}"
        )


__all__ = ["IMAGE_PROCESSOR_MAPPING", "AutoImageProcessor"]
