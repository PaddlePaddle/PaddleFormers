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
"""
 Processing saving/loading class for common processors.
"""

import bisect
import copy
import inspect
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Optional, TypedDict, Union

import numpy as np
from transformers.processing_utils import AUTO_TO_BASE_CLASS_MAPPING
from transformers.processing_utils import (
    AllKwargsForChatTemplate as AllKwargsForChatTemplate_hf,
)
from transformers.processing_utils import ProcessingKwargs as ProcessingKwargs_hf
from transformers.processing_utils import ProcessorMixin as ProcessorMixin_hf
from transformers.processing_utils import transformers_module
from transformers.utils import (
    CHAT_TEMPLATE_FILE,
    LEGACY_PROCESSOR_CHAT_TEMPLATE_FILE,
    PROCESSOR_NAME,
    PushToHubMixin,
)
from transformers.utils.chat_template_utils import render_jinja_template

from ..utils.download import resolve_file_path
from ..utils.import_utils import direct_paddleformers_import
from ..utils.log import logger
from ..utils.type_validators import (
    device_validator,
    image_size_validator,
    positive_any_number,
    positive_int,
    resampling_validator,
    tensor_type_validator,
    video_metadata_validator,
)
from .configuration_utils import custom_object_save
from .feature_extraction_utils import BatchFeature
from .image_utils import ChannelDimension, ImageInput
from .tokenizer_utils import TensorType
from .tokenizer_utils_base import PreTokenizedInput, TextInput
from .video_utils import VideoInput, VideoMetadataType

paddleformers_module = direct_paddleformers_import(Path(__file__).parent)


AUTO_TO_BASE_CLASS_MAPPING.update(
    {
        "AutoImageProcessor": "PaddleImageProcessingMixin",
        "AutoVideoProcessor": "BaseVideoProcessor",
    }
)

try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack


class VideosKwargs(TypedDict, total=False):
    """
    Keyword arguments for video processing.

    Attributes:
        do_convert_rgb (`bool`):
            Whether to convert the video to RGB format.
        do_resize (`bool`):
            Whether to resize the video.
        size (`dict[str, int]`, *optional*):
            Resize the shorter side of the input to `size["shortest_edge"]`.
        default_to_square (`bool`, *optional*, defaults to `self.default_to_square`):
            Whether to default to a square when resizing, if size is an int.
        resample (`PILImageResampling`, *optional*):
            Resampling filter to use if resizing the video.
        do_rescale (`bool`, *optional*):
            Whether to rescale the video by the specified scale `rescale_factor`.
        rescale_factor (`int` or `float`, *optional*):
            Scale factor to use if rescaling the video.
        do_normalize (`bool`, *optional*):
            Whether to normalize the video.
        image_mean (`float` or `list[float] or tuple[float, float, float]`, *optional*):
            Mean to use if normalizing the video.
        image_std (`float` or `list[float] or tuple[float, float, float]`, *optional*):
            Standard deviation to use if normalizing the video.
        do_center_crop (`bool`, *optional*):
            Whether to center crop the video.
        do_pad (`bool`, *optional*):
            Whether to pad the images in the batch.
        do_sample_frames (`bool`, *optional*):
            Whether to sample frames from the video before processing or to process the whole video.
        video_metadata (`Union[VideoMetadata, dict]`, *optional*):
            Metadata of the video containing information about total duration, fps and total number of frames.
        num_frames (`int`, *optional*):
            Maximum number of frames to sample when `do_sample_frames=True`.
        fps (`int` or `float`, *optional*):
            Target frames to sample per second when `do_sample_frames=True`.
        crop_size (`dict[str, int]`, *optional*):
            Desired output size when applying center-cropping.
        data_format (`ChannelDimension` or `str`, *optional*):
            The channel dimension format for the output video.
        input_data_format (`ChannelDimension` or `str`, *optional*):
            The channel dimension format for the input video.
        device (`Union[str, paddle.Tensor]`, *optional*):
            The device to use for processing (e.g. "cpu", "cuda"), only relevant for fast image processing.
        return_metadata (`bool`, *optional*):
            Whether to return video metadata or not.
        return_tensors (`str` or [`~utils.TensorType`], *optional*):
            If set, will return tensors of a particular framework. Acceptable values are:
            - `'pd'`: Return Paddle `paddle.Tensor` objects.
            - `'np'`: Return NumPy `np.ndarray` objects.
    """

    do_convert_rgb: Optional[bool]
    do_resize: Optional[bool]
    size: Annotated[Optional[Union[int, list[int], tuple[int, ...], dict[str, int]]], image_size_validator()]
    default_to_square: Optional[bool]
    resample: Annotated[int, resampling_validator()]
    do_rescale: Optional[bool]
    rescale_factor: Optional[float]
    do_normalize: Optional[bool]
    image_mean: Optional[Union[float, list[float], tuple[float, ...]]]
    image_std: Optional[Union[float, list[float], tuple[float, ...]]]
    do_center_crop: Optional[bool]
    do_pad: Optional[bool]
    crop_size: Annotated[Optional[Union[int, list[int], tuple[int, ...], dict[str, int]]], image_size_validator()]
    data_format: Optional[Union[str, ChannelDimension]]
    input_data_format: Optional[Union[str, ChannelDimension]]
    device: Annotated[Optional[str], device_validator()]
    do_sample_frames: Optional[bool]
    video_metadata: Annotated[Optional[VideoMetadataType], video_metadata_validator()]
    fps: Annotated[Optional[Union[int, float]], positive_any_number()]
    num_frames: Annotated[Optional[int], positive_int()]
    return_metadata: Optional[bool]
    return_tensors: Annotated[Optional[Union[str, TensorType]], tensor_type_validator()]


class ProcessingKwargs(ProcessingKwargs_hf):

    videos_kwargs: VideosKwargs = {
        **VideosKwargs.__annotations__,
    }


class AllKwargsForChatTemplate(AllKwargsForChatTemplate_hf):
    processor_kwargs: ProcessingKwargs


@dataclass
class MultiModalData:
    """
    Dataclass that holds extra useful data for processing
    multimodal data. Processors currently cannot return keys,
    unless it is used in model's forward. Thus we have helper
    methods that calculate and return useful data from processing
    input multimodals (images/videos).
    Note that this dataclass is aimed to be used only in vLLM
    and we might change its API in the future.
    """

    num_image_tokens: Optional[list[int]] = None
    num_video_tokens: Optional[list[int]] = None
    num_image_patches: Optional[list[int]] = None

    def __contains__(self, key):
        return hasattr(self, key) and getattr(self, key) is not None

    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        raise AttributeError(f"{self.__class__.__name__} has no attribute {key}")


class PaddleProcessorMixin:

    _auto_class = None
    valid_processor_kwargs = ProcessingKwargs

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(
        self,
        images: Optional[ImageInput] = None,
        text: Optional[Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]]] = None,
        videos: Optional[VideoInput] = None,
        **kwargs: Unpack[ProcessingKwargs],
    ):
        original_output = super().__call__(images, text, videos, **kwargs)
        return BatchFeature(data=original_output.data, tensor_type=kwargs["return_tensors"])

    def check_argument_for_proper_class(self, argument_name, argument):
        """
        Checks the passed argument's class against the expected transformers class. In case of an unexpected
        mismatch between expected and actual class, an error is raise. Otherwise, the proper retrieved class
        is returned.
        """
        class_name = getattr(self, f"{argument_name}_class")
        # Nothing is ever going to be an instance of "AutoXxx", in that case we check the base class.
        class_name = AUTO_TO_BASE_CLASS_MAPPING.get(class_name, class_name)
        if isinstance(class_name, tuple):
            proper_class = tuple(self.get_possibly_dynamic_module(n) for n in class_name if n is not None)
        else:
            proper_class = self.get_possibly_dynamic_module(class_name)

        if not isinstance(argument, proper_class):
            raise TypeError(
                f"Received a {type(argument).__name__} for argument {argument_name}, but a {class_name} was expected."
            )

        return proper_class

    def to_dict(self, legacy_serialization=True) -> dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            `dict[str, Any]`: Dictionary of all the attributes that make up this processor instance.
        """
        output = copy.deepcopy(self.__dict__)

        # Get the kwargs in `__init__`.
        sig = inspect.signature(self.__init__)
        # Only save the attributes that are presented in the kwargs of `__init__`.
        # or in the attributes
        attrs_to_save = list(sig.parameters) + self.__class__.attributes
        # extra attributes to be kept
        attrs_to_save += ["auto_map"]

        if "tokenizer" in output:
            del output["tokenizer"]
        if "qformer_tokenizer" in output:
            del output["qformer_tokenizer"]
        if "protein_tokenizer" in output:
            del output["protein_tokenizer"]
        if "char_tokenizer" in output:
            del output["char_tokenizer"]
        if "chat_template" in output:
            del output["chat_template"]

        def save_public_processor_class(dictionary):
            # make sure private name "_processor_class" is correctly
            # saved as "processor_class"
            _processor_class = dictionary.pop("_processor_class", None)
            if _processor_class is not None:
                dictionary["processor_class"] = _processor_class
            for value in dictionary.values():
                if isinstance(value, dict):
                    save_public_processor_class(value)
            return dictionary

        def cast_array_to_list(dictionary):
            """
            Numpy arrays are not serialiazable but can be in pre-processing dicts.
            This function casts arrays to list, recusring through the nested configs as well.
            """
            for key, value in dictionary.items():
                if isinstance(value, np.ndarray):
                    dictionary[key] = value.tolist()
                elif isinstance(value, dict):
                    dictionary[key] = cast_array_to_list(value)
            return dictionary

        # Serialize attributes as a dict
        output = {
            k: v.to_dict() if isinstance(v, PushToHubMixin) else v
            for k, v in output.items()
            if (
                k in attrs_to_save  # keep all attributes that have to be serialized
                and v.__class__.__name__ != "BeamSearchDecoderCTC"  # remove attributes with that are objects
            )
        }
        output = cast_array_to_list(output)
        output = save_public_processor_class(output)
        output["processor_class"] = self.__class__.__name__

        return output

    def save_pretrained(self, save_directory, push_to_hub: bool = False, **kwargs):
        """
        Saves the attributes of this processor (feature extractor, tokenizer...) in the specified directory so that it
        can be reloaded using the [`~ProcessorMixin.from_pretrained`] method.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the feature extractor JSON file and the tokenizer files will be saved (directory will
                be created if it does not exist).
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs (`dict[str, Any]`, *optional*):
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        save_jinja_files = kwargs.pop("save_jinja_files", True)

        os.makedirs(save_directory, exist_ok=True)

        for attribute_name in self.attributes:
            attribute = getattr(self, attribute_name)
            if hasattr(attribute, "_set_processor_class"):
                attribute._set_processor_class(self.__class__.__name__)

            # Save the tokenizer in its own vocab file. The other attributes are saved as part of `processor_config.json`
            if attribute_name == "tokenizer":
                # Propagate save_jinja_files to tokenizer to ensure we don't get conflicts
                attribute.save_pretrained(save_directory, save_jinja_files=save_jinja_files)
            elif attribute._auto_class is not None:
                custom_object_save(attribute, save_directory, config=attribute)

        # If we save using the predefined names, we can load using `from_pretrained`
        # plus we save chat_template in its own file
        output_processor_file = os.path.join(save_directory, PROCESSOR_NAME)
        output_chat_template_file_jinja = os.path.join(save_directory, CHAT_TEMPLATE_FILE)
        output_chat_template_file_legacy = os.path.join(save_directory, LEGACY_PROCESSOR_CHAT_TEMPLATE_FILE)

        # Save `chat_template` in its own file. We can't get it from `processor_dict` as we popped it in `to_dict`
        # to avoid serializing chat template in json config file. So let's get it from `self` directly
        if self.chat_template is not None:
            is_single_template = isinstance(self.chat_template, str)
            if save_jinja_files and is_single_template:
                # New format for single templates is to save them as chat_template.jinja
                with open(output_chat_template_file_jinja, "w", encoding="utf-8") as f:
                    f.write(self.chat_template)
                logger.info(f"chat template saved in {output_chat_template_file_jinja}")
            elif save_jinja_files and not is_single_template:
                # New format for multiple templates is to save the default as chat_template.jinja
                # and the other templates in the chat_templates/ directory
                for template_name, template in self.chat_template.items():
                    if template_name == "default":
                        with open(output_chat_template_file_jinja, "w", encoding="utf-8") as f:
                            f.write(self.chat_template["default"])
                        logger.info(f"chat template saved in {output_chat_template_file_jinja}")
            elif is_single_template:
                # Legacy format for single templates: Put them in chat_template.json
                chat_template_json_string = (
                    json.dumps({"chat_template": self.chat_template}, indent=2, sort_keys=True) + "\n"
                )
                with open(output_chat_template_file_legacy, "w", encoding="utf-8") as writer:
                    writer.write(chat_template_json_string)
                logger.info(f"chat template saved in {output_chat_template_file_legacy}")
            elif self.chat_template is not None:
                # At this point we have multiple templates in the legacy format, which is not supported
                # chat template dicts are saved to chat_template.json as lists of dicts with fixed key names.
                raise ValueError(
                    "Multiple chat templates are not supported in the legacy format. Please save them as "
                    "separate files using the `save_jinja_files` argument."
                )

        # Create a unified `preprocessor_config.json` and save all attributes as a composite config, except for tokenizers
        self.to_json_file(output_processor_file)
        logger.info(f"processor saved in {output_processor_file}")
        return_files = [output_processor_file]

        return return_files

    @classmethod
    def get_processor_dict(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        From a `pretrained_model_name_or_path`, resolve to a dictionary of parameters, to be used for instantiating a
        processor of type [`~processing_utils.ProcessingMixin`] using `from_args_and_dict`.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.
            subfolder (`str`, *optional*, defaults to `""`):
                In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
                specify the folder name here.

        Returns:
            `tuple[Dict, Dict]`: The dictionary(ies) that will be used to instantiate the processor object.
        """
        download_hub = kwargs.get("download_hub", None)
        if download_hub is None:
            download_hub = os.environ.get("DOWNLOAD_SOURCE", "huggingface")
        logger.info(f"Using download source: {download_hub}")

        cache_dir = kwargs.pop("cache_dir", None)
        local_files_only = kwargs.pop("local_files_only", False)
        subfolder = kwargs.pop("subfolder", "")

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        is_local = os.path.isdir(pretrained_model_name_or_path)
        if os.path.isdir(pretrained_model_name_or_path):
            processor_file = os.path.join(pretrained_model_name_or_path, PROCESSOR_NAME)

        resolved_additional_chat_template_files = {}
        if os.path.isfile(pretrained_model_name_or_path):
            resolved_processor_file = pretrained_model_name_or_path
            # can't load chat-template when given a file as pretrained_model_name_or_path
            resolved_chat_template_file = None
            resolved_raw_chat_template_file = None
            is_local = True
        else:
            processor_file = PROCESSOR_NAME
            try:
                # Load from local folder or from cache or download from model Hub and cache
                resolved_processor_file = resolve_file_path(
                    pretrained_model_name_or_path,
                    processor_file,
                    subfolder,
                    cache_dir=cache_dir,
                    download_hub=download_hub,
                    local_files_only=local_files_only,
                    force_return=True,
                )

                # chat_template.json is a legacy file used by the processor class
                # a raw chat_template.jinja is preferred in future
                resolved_chat_template_file = resolve_file_path(
                    pretrained_model_name_or_path,
                    LEGACY_PROCESSOR_CHAT_TEMPLATE_FILE,
                    subfolder,
                    cache_dir=cache_dir,
                    download_hub=download_hub,
                    local_files_only=local_files_only,
                    force_return=True,
                )

                resolved_raw_chat_template_file = resolve_file_path(
                    pretrained_model_name_or_path,
                    CHAT_TEMPLATE_FILE,
                    subfolder,
                    cache_dir=cache_dir,
                    download_hub=download_hub,
                    local_files_only=local_files_only,
                    force_return=True,
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
                    f"Expected image processor files: {PROCESSOR_NAME}\n"
                    f"Note: The repository ID may differ between ModelScope, AI Studio, and Hugging Face Hub.\n"
                    f"You are currently using the download source: {download_hub}. Please check the repository ID on the official website."
                )

        # Add chat template as kwarg before returning because most models don't have processor config
        if resolved_chat_template_file is not None:
            # This is the legacy path
            with open(resolved_chat_template_file, encoding="utf-8") as reader:
                chat_template_json = json.loads(reader.read())
                chat_templates = {"default": chat_template_json["chat_template"]}
                if resolved_additional_chat_template_files:
                    raise ValueError(
                        "Cannot load chat template due to conflicting files - this checkpoint combines "
                        "a legacy chat_template.json file with separate template files, which is not "
                        "supported. To resolve this error, replace the legacy chat_template.json file "
                        "with a modern chat_template.jinja file."
                    )
        else:
            chat_templates = {
                template_name: open(template_file, "r", encoding="utf-8").read()
                for template_name, template_file in resolved_additional_chat_template_files.items()
            }
            if resolved_raw_chat_template_file is not None:
                with open(resolved_raw_chat_template_file, "r", encoding="utf-8") as reader:
                    chat_templates["default"] = reader.read()
        if isinstance(chat_templates, dict) and "default" in chat_templates and len(chat_templates) == 1:
            chat_templates = chat_templates["default"]  # Flatten when we just have a single template/file

        if chat_templates:
            kwargs["chat_template"] = chat_templates

        # Existing processors on the Hub created before #27761 being merged don't have `processor_config.json` (if not
        # updated afterward), and we need to keep `from_pretrained` work. So here it fallbacks to the empty dict.
        # (`cached_file` called using `_raise_exceptions_for_missing_entries=False` to avoid exception)
        # However, for models added in the future, we won't get the expected error if this file is missing.
        if resolved_processor_file is None:
            # In any case we need to pass `chat_template` if it is available
            processor_dict = {}
        else:
            try:
                # Load processor dict
                with open(resolved_processor_file, encoding="utf-8") as reader:
                    text = reader.read()
                processor_dict = json.loads(text)

            except json.JSONDecodeError:
                raise OSError(
                    f"It looks like the config file at '{resolved_processor_file}' is not a valid JSON file."
                )

        if is_local:
            logger.info(f"loading configuration file {resolved_processor_file}")
        else:
            logger.info(f"loading configuration file {processor_file} from cache at {resolved_processor_file}")

        if "chat_template" in processor_dict and processor_dict["chat_template"] is not None:
            logger.warning_once(
                "Chat templates should be in a 'chat_template.jinja' file but found key='chat_template' "
                "in the processor's config. Make sure to move your template to its own file."
            )

        if "chat_template" in kwargs:
            processor_dict["chat_template"] = kwargs.pop("chat_template")

        return processor_dict, kwargs

    @classmethod
    def from_args_and_dict(cls, args, processor_dict: dict[str, Any], **kwargs):
        """
        Instantiates a type of [`~processing_utils.ProcessingMixin`] from a Python dictionary of parameters.

        Args:
            processor_dict (`dict[str, Any]`):
                Dictionary that will be used to instantiate the processor object. Such a dictionary can be
                retrieved from a pretrained checkpoint by leveraging the
                [`~processing_utils.ProcessingMixin.to_dict`] method.
            kwargs (`dict[str, Any]`):
                Additional parameters from which to initialize the processor object.

        Returns:
            [`~processing_utils.ProcessingMixin`]: The processor object instantiated from those
            parameters.
        """
        processor_dict = processor_dict.copy()
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

        # We have to pop up some unused (but specific) kwargs and then validate that it doesn't contain unused kwargs
        # If we don't pop, some specific kwargs will raise a warning or error
        for unused_kwarg in cls.attributes + ["auto_map", "processor_class"]:
            processor_dict.pop(unused_kwarg, None)

        # override processor_dict with given kwargs
        processor_dict.update(kwargs)

        # check if there is an overlap between args and processor_dict
        accepted_args_and_kwargs = cls.__init__.__code__.co_varnames[: cls.__init__.__code__.co_argcount][1:]

        # validate both processor_dict and given kwargs
        unused_kwargs, valid_kwargs = cls.validate_init_kwargs(
            processor_config=processor_dict, valid_kwargs=accepted_args_and_kwargs
        )

        # update args that are already in processor_dict to avoid duplicate arguments
        args_to_update = {
            i: valid_kwargs.pop(arg)
            for i, arg in enumerate(accepted_args_and_kwargs)
            if (arg in valid_kwargs and i < len(args))
        }
        args = [args_to_update.get(i, arg) for i, arg in enumerate(args)]

        # instantiate processor with used (and valid) kwargs only
        processor = cls(*args, **valid_kwargs)

        # logger.info(f"Processor {processor}")
        if return_unused_kwargs:
            return processor, unused_kwargs
        else:
            return processor

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        **kwargs,
    ):
        processor_dict, kwargs = cls.get_processor_dict(pretrained_model_name_or_path, **kwargs)
        args = cls._get_arguments_from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls.from_args_and_dict(args, processor_dict, **kwargs)

    @classmethod
    def _get_arguments_from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        Identify and instantiate the subcomponents of Processor classes, like image processors and
        tokenizers. This method uses the Processor attributes like `tokenizer_class` to figure out what class those
        subcomponents should be. Note that any subcomponents must either be library classes that are accessible in
        the `transformers` root, or they must be custom code that has been registered with the relevant autoclass,
        via methods like `AutoTokenizer.register()`. If neither of these conditions are fulfilled, this method
        will be unable to find the relevant subcomponent class and will raise an error.
        """
        args = []
        for attribute_name in cls.attributes:
            class_name = getattr(cls, f"{attribute_name}_class")
            if isinstance(class_name, tuple):
                classes = tuple(cls.get_possibly_dynamic_module(n) if n is not None else None for n in class_name)
                if attribute_name == "image_processor":
                    use_fast = kwargs.get("use_fast", None)
                    if use_fast is None or use_fast:
                        logger.warning(
                            "The model's image processor only supports the slow version (`use_fast=False`). "
                            "Detected `use_fast=True` but will fall back to the slow version."
                        )
                else:
                    use_fast = kwargs.get("use_fast", True)
                if use_fast and classes[1] is not None and "Image" not in attribute_name:
                    attribute_class = classes[1]
                else:
                    attribute_class = classes[0]
            else:
                attribute_class = cls.get_possibly_dynamic_module(class_name)

            args.append(attribute_class.from_pretrained(pretrained_model_name_or_path, **kwargs))

        return args

    @staticmethod
    def get_possibly_dynamic_module(module_name):
        if hasattr(paddleformers_module, module_name):
            return getattr(paddleformers_module, module_name)
        lookup_locations = [
            paddleformers_module.IMAGE_PROCESSOR_MAPPING,
            paddleformers_module.VIDEO_PROCESSOR_MAPPING,
            paddleformers_module.TOKENIZER_MAPPING,
            transformers_module.TOKENIZER_MAPPING,
        ]
        for lookup_location in lookup_locations:
            for custom_class in lookup_location._extra_content.values():
                if isinstance(custom_class, tuple):
                    for custom_subclass in custom_class:
                        if custom_subclass is not None and custom_subclass.__name__ == module_name:
                            return custom_subclass
                elif custom_class is not None and custom_class.__name__ == module_name:
                    return custom_class
        raise ValueError(f"Could not find module {module_name} in `paddleformers`.")

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        if not hasattr(self, "tokenizer"):
            raise ValueError(f"Cannot decode text: {self.__class__.__name__} has no tokenizer.")
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        model_input_names = []
        for attribute_name in self.attributes:
            attribute = getattr(self, attribute_name, None)
            attr_input_names = getattr(attribute, "model_input_names")
            model_input_names.extend(attr_input_names)
        return model_input_names

    def apply_chat_template(
        self,
        conversation: Union[list[dict[str, str]], list[list[dict[str, str]]]],
        chat_template: Optional[str] = None,
        **kwargs: Unpack[AllKwargsForChatTemplate],
    ) -> str:
        """
        Similar to the `apply_chat_template` method on tokenizers, this method applies a Jinja template to input
        conversations to turn them into a single tokenizable string.

        The input is expected to be in the following format, where each message content is a list consisting of text and
        optionally image or video inputs. One can also provide an image, video, URL or local path which will be used to form
        `pixel_values` when `return_dict=True`. If not provided, one will get only the formatted text, optionally tokenized text.

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
                    {"type": "text", "text": "Please describe this image in detail."},
                ],
            },
        ]

        Args:
            conversation (`Union[list[Dict, [str, str]], list[list[dict[str, str]]]]`):
                The conversation to format.
            chat_template (`Optional[str]`, *optional*):
                The Jinja template to use for formatting the conversation. If not provided, the tokenizer's
                chat template is used.
        """
        if chat_template is None:
            if isinstance(self.chat_template, dict) and "default" in self.chat_template:
                chat_template = self.chat_template["default"]
            elif isinstance(self.chat_template, dict):
                raise ValueError(
                    'The processor has multiple chat templates but none of them are named "default". You need to specify'
                    " which one to use by passing the `chat_template` argument. Available templates are: "
                    f"{', '.join(self.chat_template.keys())}"
                )
            elif self.chat_template is not None:
                chat_template = self.chat_template
            else:
                raise ValueError(
                    "Cannot use apply_chat_template because this processor does not have a chat template."
                )
        else:
            if isinstance(self.chat_template, dict) and chat_template in self.chat_template:
                # It's the name of a template, not a full template string
                chat_template = self.chat_template[chat_template]
            else:
                # It's a template string, render it directly
                pass

        is_tokenizers_fast = hasattr(self, "tokenizer") and self.tokenizer.__class__.__name__.endswith("Fast")

        if kwargs.get("continue_final_message", False):
            if kwargs.get("add_generation_prompt", False):
                raise ValueError(
                    "continue_final_message and add_generation_prompt are not compatible. Use continue_final_message when you want the model to continue the final message, and add_generation_prompt when you want to add a header that will prompt it to start a new assistant message instead."
                )
            if kwargs.get("return_assistant_tokens_mask", False):
                raise ValueError("continue_final_message is not compatible with return_assistant_tokens_mask.")

        if kwargs.get("return_assistant_tokens_mask", False):
            if not is_tokenizers_fast:
                raise ValueError(
                    "`return_assistant_tokens_mask` is not possible with slow tokenizers. Make sure you have `tokenizers` installed. "
                    "If the error persists, open an issue to support a Fast tokenizer for your model."
                )
            else:
                kwargs["return_offsets_mapping"] = True  # force offset mapping so we can infer token boundaries

        # Fill sets of kwargs that should be used by different parts of template
        processed_kwargs = {
            "template_kwargs": {},
        }

        for kwarg_type in processed_kwargs:
            for key in AllKwargsForChatTemplate.__annotations__[kwarg_type].__annotations__:
                kwarg_type_defaults = AllKwargsForChatTemplate.__annotations__[kwarg_type]
                default_value = getattr(kwarg_type_defaults, key, None)
                value = kwargs.pop(key, default_value)
                if value is not None and not isinstance(value, dict):
                    processed_kwargs[kwarg_type][key] = value

        # pop unused and deprecated kwarg
        kwargs.pop("video_load_backend", None)

        # Pass unprocessed custom kwargs
        processed_kwargs["template_kwargs"].update(kwargs)

        if isinstance(conversation, (list, tuple)) and (
            isinstance(conversation[0], (list, tuple)) or hasattr(conversation[0], "content")
        ):
            is_batched = True
            conversations = conversation
        else:
            is_batched = False
            conversations = [conversation]

        tokenize = processed_kwargs["template_kwargs"].pop("tokenize", False)
        return_dict = processed_kwargs["template_kwargs"].pop("return_dict", False)

        if tokenize:
            batch_images, batch_videos = [], []
            for conversation in conversations:
                images, videos = [], []
                for message in conversation:
                    visuals = [content for content in message["content"] if content["type"] in ["image", "video"]]
                    image_fnames = [
                        vision_info[key]
                        for vision_info in visuals
                        for key in ["image", "url", "path", "base64"]
                        if key in vision_info and vision_info["type"] == "image"
                    ]
                    images.extend(image_fnames)
                    video_fnames = [
                        vision_info[key]
                        for vision_info in visuals
                        for key in ["video", "url", "path"]
                        if key in vision_info and vision_info["type"] == "video"
                    ]
                    videos.extend(video_fnames)

                # Currently all processors can accept nested list of batches, but not flat list of visuals
                # So we'll make a batched list of images and let the processor handle it
                batch_images.append(images)
                batch_videos.append(videos)

        prompt, generation_indices = render_jinja_template(
            conversations=conversations,
            chat_template=chat_template,
            **processed_kwargs["template_kwargs"],  # different flags such as `return_assistant_mask`
            **self.tokenizer.special_tokens_map,  # tokenizer special tokens are used by some templates
        )

        if not is_batched:
            prompt = prompt[0]

        if tokenize:
            # Tokenizer's `apply_chat_template` never adds special tokens when tokenizing
            # But processor's `apply_chat_template` didn't have an option to tokenize, so users had to format the prompt
            # and pass it to the processor. Users thus never worried about special tokens relying on processor handling
            # everything internally. The below line is to keep BC for that and be able to work with model that have
            # special tokens in the template (consistent with tokenizers). We dont want to raise warning, it will flood command line
            # without actionable solution for users
            single_prompt = prompt[0] if is_batched else prompt
            if self.tokenizer.bos_token is not None and single_prompt.startswith(self.tokenizer.bos_token):
                kwargs["add_special_tokens"] = False

            # Always sample frames by default unless explicitly set to `False` by users. If users do not pass `num_frames`/`fps`
            # sampling should not done for BC.
            if "do_sample_frames" not in kwargs and (
                kwargs.get("fps") is not None or kwargs.get("num_frames") is not None
            ):
                kwargs["do_sample_frames"] = True

            images_exist = any((im is not None) for im_list in batch_images for im in im_list)
            videos_exist = any((vid is not None) for vid_list in batch_videos for vid in vid_list)
            out = self(
                text=prompt,
                images=batch_images if images_exist else None,
                videos=batch_videos if videos_exist else None,
                **kwargs,
            )

            if return_dict:
                if processed_kwargs["template_kwargs"].get("return_assistant_tokens_mask", False):
                    assistant_masks = []
                    offset_mapping = out.pop("offset_mapping")
                    input_ids = out["input_ids"]
                    for i in range(len(input_ids)):
                        current_mask = [0] * len(input_ids[i])
                        offsets = offset_mapping[i]
                        offset_starts = [start for start, end in offsets]
                        for assistant_start_char, assistant_end_char in generation_indices[i]:
                            start_pos = bisect.bisect_left(offset_starts, assistant_start_char)
                            end_pos = bisect.bisect_left(offset_starts, assistant_end_char)

                            if not (
                                start_pos >= 0
                                and offsets[start_pos][0] <= assistant_start_char < offsets[start_pos][1]
                            ):
                                # start_token is out of bounds maybe due to truncation.
                                continue
                            for token_id in range(start_pos, end_pos if end_pos else len(input_ids[i])):
                                current_mask[token_id] = 1
                        assistant_masks.append(current_mask)
                    out["assistant_masks"] = assistant_masks
                    out.convert_to_tensors(tensor_type=kwargs.get("return_tensors"))
                return out
            else:
                return out["input_ids"]
        return prompt


def warp_processormixin(hf_processormixin_class: ProcessorMixin_hf):
    return type(hf_processormixin_class.__name__, (PaddleProcessorMixin, hf_processormixin_class), {})


class ProcessorMixin(PaddleProcessorMixin, ProcessorMixin_hf):
    def init(self, *args, **kwargs):
        super().init(*args, **kwargs)
