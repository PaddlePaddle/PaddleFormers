# coding=utf-8
# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2022 The HuggingFace Inc. team.
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
from functools import wraps
from typing import Any, Union

import numpy as np
import paddle
from transformers.feature_extraction_utils import BatchFeature as BatchFeature_hf
from transformers.image_processing_base import IMAGE_PROCESSOR_NAME
from transformers.image_processing_base import (
    ImageProcessingMixin as ImageProcessingMixin_hf,
)
from transformers.image_processing_utils import (
    BaseImageProcessor as BaseImageProcessor_hf,
)
from transformers.image_processing_utils import get_size_dict  # noqa: F401
from transformers.utils import PROCESSOR_NAME

from ..utils.download import DownloadSource, resolve_file_path
from ..utils.log import logger
from .feature_extraction_utils import BatchFeature


class PaddleImageProcessingMixin:
    """
    A mixin that extends Hugging Face image processor classes to support
    PaddlePaddle tensor return types.

    This mixin dynamically wraps common preprocessing and transformation
    methods, enabling them to return Paddle tensors when the argument
    `return_tensors="pd"` is specified.

    The wrapping is non-intrusive — methods behave exactly the same as
    before unless Paddle tensor output is explicitly requested.
    """

    # Define the key methods that should support Paddle tensor return types.
    # Only methods that actually exist in a subclass will be wrapped.
    methods_to_wrap = [
        "__call__",
        "preprocess",
        "rescale",
        "normalize",
        "center_crop",
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._wrap_return_tensor_methods()

    def _wrap_return_tensor_methods(self):
        """Dynamically wraps class methods to add Paddle tensor return support.

        This function iterates through `self.methods_to_wrap` and, for each method
        that exists in the current instance, replaces it with a wrapped version
        that automatically converts outputs to Paddle tensors when
        `return_tensors="pd"` is provided.

        Purpose:
            - Ensure consistent Paddle tensor output across preprocessing steps.
            - Maintain compatibility with existing Hugging Face-style APIs.
            - Allow seamless switching between NumPy/PIL outputs and Paddle tensors.

        Typical wrapped methods include:
            - Core call interface (`__call__`)
            - Data preprocessing and normalization
            (`preprocess`, `rescale`, `normalize`, `center_crop`)
        Returns:
            None: Methods are replaced in-place within the current instance.
        """
        for method_name in self.methods_to_wrap:
            if hasattr(self, method_name):
                self._wrap_single_method(method_name)

    def _wrap_single_method(self, method_name):
        """Wrap a single method of the class to convert its output to Paddle tensors when requested.

        This decorator modifies the specified method to optionally convert its return value to
        PaddlePaddle tensors when the 'return_tensors="pd"' parameter is provided.

        Args:
            method_name (str): The name of the method to be wrapped.

        Returns:
            None: This method modifies the class instance in-place by replacing the original method
            with the wrapped version.
        """
        original_method = getattr(self, method_name)

        def convert_to_paddle(inputs):
            """Convert various input types to Paddle tensors recursively.

            Handles conversion of:
            - Lists (both single and nested)
            - BatchFeature objects (converts values recursively)
            - Other types (returns unchanged)

            Args:
                inputs: The input data to be converted

            Returns:
                The converted Paddle tensor or the original input if no conversion was needed
            """

            if isinstance(inputs, list):
                if isinstance(inputs[0], int):
                    return paddle.to_tensor([inputs])
                else:
                    return paddle.to_tensor(inputs)
            elif isinstance(inputs, int):
                return paddle.to_tensor(inputs)
            elif isinstance(inputs, np.ndarray):
                return paddle.to_tensor(inputs)
            elif isinstance(inputs, BatchFeature) or isinstance(inputs, dict):
                for key, value in inputs.items():
                    inputs[key] = convert_to_paddle(value)
                return inputs
            else:
                return inputs

        @wraps(original_method)
        def wrapper(*args, **kwargs):
            return_tensors = kwargs.get("return_tensors", None)
            if return_tensors == "pd":
                return_tensors = kwargs.pop("return_tensors", None)
            result = original_method(*args, **kwargs)
            if return_tensors == "pd":
                result = convert_to_paddle(result)
            return result

        setattr(self, method_name, wrapper)

    def __call__(self, images, *args, **kwargs) -> BatchFeature:
        original_output: BatchFeature_hf = super().__call__(images, *args, **kwargs)
        return BatchFeature(data=original_output.data, tensor_type=kwargs["return_tensors"])

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        *args,
        **kwargs,
    ):
        image_processor_dict, kwargs = cls.get_image_processor_dict(pretrained_model_name_or_path, **kwargs)
        return super().from_dict(image_processor_dict, **kwargs)

    @classmethod
    def get_image_processor_dict(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        From a `pretrained_model_name_or_path`, resolve to a dictionary of parameters, to be used for instantiating a
        image processor of type [`~image_processor_utils.ImageProcessingMixin`] using `from_dict`.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.
            subfolder (`str`, *optional*, defaults to `""`):
                In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
                specify the folder name here.
            image_processor_filename (`str`, *optional*, defaults to `"config.json"`):
                The name of the file in the model directory to use for the image processor config.

        Returns:
            `tuple[Dict, Dict]`: The dictionary(ies) that will be used to instantiate the image processor object.
        """
        download_hub = kwargs.get("download_hub", None)
        local_files_only = kwargs.pop("local_files_only", False)

        if download_hub is None:
            download_hub = os.environ.get("DOWNLOAD_SOURCE", "huggingface")
        logger.info(f"Using download source: {download_hub}")

        # If downloaded from hf, use the native hf from pretrained
        if download_hub == DownloadSource.HUGGINGFACE:
            return super().get_image_processor_dict(
                pretrained_model_name_or_path,
                **kwargs,
            )

        cache_dir = kwargs.pop("cache_dir", None)
        subfolder = kwargs.pop("subfolder", "")
        image_processor_filename = kwargs.pop("image_processor_filename", IMAGE_PROCESSOR_NAME)

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        is_local = os.path.isdir(pretrained_model_name_or_path)
        if os.path.isdir(pretrained_model_name_or_path):
            image_processor_file = os.path.join(pretrained_model_name_or_path, image_processor_filename)
        if os.path.isfile(pretrained_model_name_or_path):
            resolved_image_processor_file = pretrained_model_name_or_path
            is_local = True
        else:
            image_processor_file = image_processor_filename
            try:
                resolved_image_processor_file = resolve_file_path(
                    pretrained_model_name_or_path,
                    [image_processor_file, PROCESSOR_NAME],
                    subfolder,
                    cache_dir=cache_dir,
                    download_hub=download_hub,
                    local_files_only=local_files_only,
                )
            except Exception:
                hf_link = f"https://huggingface.co/{pretrained_model_name_or_path}"
                modelscope_link = f"https://modelscope.cn/models/{pretrained_model_name_or_path}"
                encoded_model_name = pretrained_model_name_or_path.replace("/", "%2F")
                aistudio_link = f"https://aistudio.baidu.com/modelsoverview?sortBy=weight&q={encoded_model_name}"

                raise ValueError(
                    f"No image processor for model '{pretrained_model_name_or_path}'. "
                    f"Please check:\n"
                    f"1. The model repository ID is correct for your chosen source:\n"
                    f"   - Hugging Face Hub: {hf_link}\n"
                    f"   - ModelScope: {modelscope_link}\n"
                    f"   - AI Studio: {aistudio_link}\n"
                    f"2. You have permission to access this model repository\n"
                    f"3. Network connection is working properly\n"
                    f"4. Try clearing cache and downloading again\n"
                    f"Expected image processor files: {image_processor_filename}\n"
                    f"Note: The repository ID may differ between ModelScope, AI Studio, and Hugging Face Hub.\n"
                    f"You are currently using the download source: {download_hub}. Please check the repository ID on the official website."
                )

        try:
            # Load image_processor dict
            with open(resolved_image_processor_file, encoding="utf-8") as reader:
                text = reader.read()
            image_processor_dict = json.loads(text)
            image_processor_dict = image_processor_dict.get("image_processor", image_processor_dict)

        except json.JSONDecodeError:
            raise OSError(
                f"It looks like the config file at '{resolved_image_processor_file}' is not a valid JSON file."
            )

        if is_local:
            logger.info(f"loading configuration file {resolved_image_processor_file}")
        else:
            logger.info(
                f"loading configuration file {image_processor_file} from cache at {resolved_image_processor_file}"
            )
        return image_processor_dict, kwargs

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary.

        Returns:
            `dict[str, Any]`: Dictionary of all the attributes that make up this image processor instance.
        """
        output = super().to_dict()
        for method_name in self.methods_to_wrap:
            output.pop(method_name, None)

        return output


def warp_image_processormixin(hf_image_processormixin_class: ImageProcessingMixin_hf):
    return type(
        hf_image_processormixin_class.__name__, (PaddleImageProcessingMixin, hf_image_processormixin_class), {}
    )


def warp_base_image_processor(hf_base_image_processor_class: BaseImageProcessor_hf):
    return type(
        hf_base_image_processor_class.__name__, (PaddleImageProcessingMixin, hf_base_image_processor_class), {}
    )


class ImageProcessingMixin(PaddleImageProcessingMixin, ImageProcessingMixin_hf):
    def init(self, *args, **kwargs):
        super().init(*args, **kwargs)


class BaseImageProcessor(PaddleImageProcessingMixin, BaseImageProcessor_hf):
    def init(self, *args, **kwargs):
        super().init(*args, **kwargs)
