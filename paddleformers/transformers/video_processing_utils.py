# coding=utf-8
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
from collections.abc import Callable
from copy import deepcopy
from functools import partial
from typing import Any, Optional, Union

import numpy as np
import paddle
from huggingface_hub.dataclasses import validated_field
from transformers.dynamic_module_utils import custom_object_save
from transformers.image_processing_utils import get_size_dict
from transformers.utils import (
    IMAGE_PROCESSOR_NAME,
    PROCESSOR_NAME,
    VIDEO_PROCESSOR_NAME,
    add_start_docstrings,
)

from ..utils.download import resolve_file_path
from ..utils.log import logger
from .feature_extraction_utils import BatchFeature
from .image_processing_utils_fast import BaseImageProcessorFast
from .image_utils import ChannelDimension, validate_kwargs
from .paddle_vision_utils import grayscale_to_rgb, pil_to_tensor
from .processing_utils import Unpack, VideosKwargs
from .video_utils import (
    VideoInput,
    VideoMetadata,
    group_videos_by_shape,
    infer_channel_dimension_format,
    is_valid_video,
    load_video,
    make_batched_metadata,
    make_batched_videos,
    reorder_videos,
)

BASE_VIDEO_PROCESSOR_DOCSTRING = r"""
    Args:
        do_resize (`bool`, *optional*, defaults to `self.do_resize`):
            Whether to resize the video's (height, width) dimensions to the specified `size`. Can be overridden by the
            `do_resize` parameter in the `preprocess` method.
        size (`dict`, *optional*, defaults to `self.size`):
            Size of the output video after resizing. Can be overridden by the `size` parameter in the `preprocess`
            method.
        size_divisor (`int`, *optional*, defaults to `self.size_divisor`):
            The size by which to make sure both the height and width can be divided.
        default_to_square (`bool`, *optional*, defaults to `self.default_to_square`):
            Whether to default to a square video when resizing, if size is an int.
        resample (`PILImageResampling`, *optional*, defaults to `self.resample`):
            Resampling filter to use if resizing the video. Only has an effect if `do_resize` is set to `True`. Can be
            overridden by the `resample` parameter in the `preprocess` method.
        do_center_crop (`bool`, *optional*, defaults to `self.do_center_crop`):
            Whether to center crop the video to the specified `crop_size`. Can be overridden by `do_center_crop` in the
            `preprocess` method.
        crop_size (`dict[str, int]` *optional*, defaults to `self.crop_size`):
            Size of the output video after applying `center_crop`. Can be overridden by `crop_size` in the `preprocess`
            method.
        do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
            Whether to rescale the video by the specified scale `rescale_factor`. Can be overridden by the
            `do_rescale` parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `self.rescale_factor`):
            Scale factor to use if rescaling the video. Only has an effect if `do_rescale` is set to `True`. Can be
            overridden by the `rescale_factor` parameter in the `preprocess` method.
        do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
            Whether to normalize the video. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method. Can be overridden by the `do_normalize` parameter in the `preprocess` method.
        image_mean (`float` or `list[float]`, *optional*, defaults to `self.image_mean`):
            Mean to use if normalizing the video. This is a float or list of floats the length of the number of
            channels in the video. Can be overridden by the `image_mean` parameter in the `preprocess` method. Can be
            overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `list[float]`, *optional*, defaults to `self.image_std`):
            Standard deviation to use if normalizing the video. This is a float or list of floats the length of the
            number of channels in the video. Can be overridden by the `image_std` parameter in the `preprocess` method.
            Can be overridden by the `image_std` parameter in the `preprocess` method.
        do_convert_rgb (`bool`, *optional*, defaults to `self.image_std`):
            Whether to convert the video to RGB.
        video_metadata (`VideoMetadata`, *optional*):
            Metadata of the video containing information about total duration, fps and total number of frames.
        do_sample_frames (`int`, *optional*, defaults to `self.do_sample_frames`):
            Whether to sample frames from the video before processing or to process the whole video.
        num_frames (`int`, *optional*, defaults to `self.num_frames`):
            Maximum number of frames to sample when `do_sample_frames=True`.
        fps (`int` or `float`, *optional*, defaults to `self.fps`):
            Target frames to sample per second when `do_sample_frames=True`.
        return_tensors (`str` or `TensorType`, *optional*):
            Returns stacked tensors if set to `pt, otherwise returns a list of tensors.
        data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
            The channel dimension format for the output video. Can be one of:
            - `"channels_first"` or `ChannelDimension.FIRST`: video in (num_channels, height, width) format.
            - `"channels_last"` or `ChannelDimension.LAST`: video in (height, width, num_channels) format.
            - Unset: Use the channel dimension format of the input video.
        input_data_format (`ChannelDimension` or `str`, *optional*):
            The channel dimension format for the input video. If unset, the channel dimension format is inferred
            from the input video. Can be one of:
            - `"channels_first"` or `ChannelDimension.FIRST`: video in (num_channels, height, width) format.
            - `"channels_last"` or `ChannelDimension.LAST`: video in (height, width, num_channels) format.
            - `"none"` or `ChannelDimension.NONE`: video in (height, width) format.
        device (`paddle.device`, *optional*):
            The device to process the videos on. If unset, the device is inferred from the input videos.
        return_metadata (`bool`, *optional*):
            Whether to return video metadata or not.
        """


@add_start_docstrings(
    "Constructs a base VideoProcessor.",
    BASE_VIDEO_PROCESSOR_DOCSTRING,
)
class BaseVideoProcessor(BaseImageProcessorFast):
    _auto_class = None

    resample = None
    image_mean = None
    image_std = None
    size = None
    size_divisor = None
    default_to_square = True
    crop_size = None
    do_resize = None
    do_center_crop = None
    do_rescale = None
    rescale_factor = 1 / 255
    do_normalize = None
    do_convert_rgb = None
    do_sample_frames = None
    fps = None
    num_frames = None
    video_metadata = None
    return_metadata = False
    valid_kwargs = VideosKwargs
    model_input_names = ["pixel_values_videos"]
    video_backend = "paddlecodec"

    def __init__(self, **kwargs: Unpack[VideosKwargs]) -> None:
        super().__init__()

        # Dynamically wraps class methods to add Paddle tensor return support.
        self._wrap_return_tensor_methods()

        self._processor_class = kwargs.pop("processor_class", None)

        # Additional attributes without default values
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error(f"Can't set {key} with value {value} for {self}")
                raise err

        # Prepare size related keys and turn then into `SizeDict`
        size = kwargs.pop("size", self.size)
        self.size = (
            get_size_dict(size=size, default_to_square=kwargs.pop("default_to_square", self.default_to_square))
            if size is not None
            else None
        )
        crop_size = kwargs.pop("crop_size", self.crop_size)
        self.crop_size = get_size_dict(crop_size, param_name="crop_size") if crop_size is not None else None

        # Save valid kwargs in a list for further processing
        self.model_valid_processing_keys = list(self.valid_kwargs.__annotations__.keys())
        for key in self.model_valid_processing_keys:
            if kwargs.get(key) is not None:
                setattr(self, key, kwargs[key])
            else:
                setattr(self, key, deepcopy(getattr(self, key, None)))

    def __call__(self, videos, **kwargs) -> BatchFeature:
        return self.preprocess(videos, **kwargs)

    def convert_to_rgb(
        self,
        video: "paddle.Tensor",
    ) -> VideoInput:
        """
        Converts a video to RGB format.
        """
        video = grayscale_to_rgb(video)
        if video.shape[-3] == 3 or not (video[..., 3, :, :] < 255).any():
            return video

        alpha = video[..., 3, :, :] / 255.0
        video = (1 - alpha[..., None, :, :]) * 255 + alpha[..., None, :, :] * video[..., :3, :, :]
        return video

    def sample_frames(
        self,
        metadata: VideoMetadata,
        num_frames: Optional[int] = None,
        fps: Optional[Union[int, float]] = None,
        **kwargs,
    ):
        """
        Default sampling function which uniformly samples the desired number of frames between 0 and total number of frames.
        If `fps` is passed along with metadata, `fps` frames per second are sampled uniformty. Arguments `num_frames`
        and `fps` are mutually exclusive.

        Args:
            metadata (`VideoMetadata`):
                Metadata of the video containing information about total duration, fps and total number of frames.
            num_frames (`int`, *optional*):
                Maximum number of frames to sample. Defaults to `self.num_frames`.
            fps (`int` or `float`, *optional*):
                Target frames to sample per second. Defaults to `self.fps`.

        Returns:
            np.ndarray:
                Indices to sample video frames.
        """
        if fps is not None and num_frames is not None:
            raise ValueError(
                "`num_frames`, `fps`, and `sample_indices_fn` are mutually exclusive arguments, please use only one!"
            )

        num_frames = num_frames if num_frames is not None else self.num_frames
        fps = fps if fps is not None else self.fps
        total_num_frames = metadata.total_num_frames

        # If num_frames is not given but fps is, calculate num_frames from fps
        if num_frames is None and fps is not None:
            if metadata is None or metadata.fps is None:
                raise ValueError(
                    "Asked to sample `fps` frames per second but no video metadata was provided which is required when sampling with `fps`. "
                    "Please pass in `VideoMetadata` object or use a fixed `num_frames` per input video"
                )
            num_frames = int(total_num_frames / metadata.fps * fps)

        if num_frames > total_num_frames:
            raise ValueError(
                f"Video can't be sampled. The `num_frames={num_frames}` exceeds `total_num_frames={total_num_frames}`. "
            )

        if num_frames is not None:
            indices = paddle.arange(0, total_num_frames, total_num_frames / num_frames).int()
        else:
            indices = paddle.arange(total_num_frames).int()
        return indices

    def _decode_and_sample_videos(
        self,
        videos: VideoInput,
        video_metadata: Union[VideoMetadata, dict],
        do_sample_frames: Optional[bool] = None,
        sample_indices_fn: Optional[Callable] = None,
        **kwargs,
    ) -> list["paddle.Tensor"]:
        """
        Decode input videos and sample frames if needed.
        """
        videos = make_batched_videos(videos)
        video_metadata = make_batched_metadata(videos, video_metadata=video_metadata)

        # Only sample frames if an array video is passed, otherwise first decode -> then sample
        if is_valid_video(videos[0]) and do_sample_frames:
            sampled_videos = []
            sampled_metadata = []
            for video, metadata in zip(videos, video_metadata):
                indices = sample_indices_fn(metadata=metadata)
                metadata.frames_indices = indices
                sampled_videos.append(video[indices])
                sampled_metadata.append(metadata)
            videos = sampled_videos
            video_metadata = sampled_metadata
        elif not is_valid_video(videos[0]):
            if isinstance(videos[0], list):
                # Videos sometimes are passed as a list of image URLs, especially through templates
                videos = [
                    paddle.stack([pil_to_tensor(image) for image in images], dim=0)
                    for images in self.fetch_images(videos)
                ]
                if do_sample_frames:
                    raise ValueError(
                        "Sampling frames from a list of images is not supported! Set `do_sample_frames=False`."
                    )
            else:
                videos, video_metadata = self.fetch_videos(videos, sample_indices_fn=sample_indices_fn, **kwargs)

        return videos, video_metadata

    def _prepare_input_videos(
        self,
        videos: VideoInput,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> list["paddle.Tensor"]:
        """
        Prepare the input videos for processing.
        """
        processed_videos = []
        for video in videos:
            # `make_batched_videos` always returns a 4D array per video
            if isinstance(video, np.ndarray):
                # not using F.to_tensor as it doesn't handle (C, H, W) numpy arrays
                video = paddle.to_tensor(video).contiguous()

            # Infer the channel dimension format if not provided
            if input_data_format is None:
                input_data_format = infer_channel_dimension_format(video)

            if input_data_format == ChannelDimension.LAST:
                video = video.permute(0, 3, 1, 2).contiguous()

            processed_videos.append(video)
        return processed_videos

    @add_start_docstrings(
        BASE_VIDEO_PROCESSOR_DOCSTRING,
    )
    def preprocess(
        self,
        videos: VideoInput,
        **kwargs: Unpack[VideosKwargs],
    ) -> BatchFeature:
        validate_kwargs(
            captured_kwargs=kwargs.keys(),
            valid_processor_keys=list(self.valid_kwargs.__annotations__.keys()) + ["return_tensors"],
        )

        # Perform type validation on received kwargs
        validated_field(self.valid_kwargs, kwargs)

        # Set default kwargs from self. This ensures that if a kwarg is not provided
        # by the user, it gets its default value from the instance, or is set to None.
        for kwarg_name in self.valid_kwargs.__annotations__:
            kwargs.setdefault(kwarg_name, getattr(self, kwarg_name, None))

        input_data_format = kwargs.pop("input_data_format")
        do_sample_frames = kwargs.pop("do_sample_frames")
        video_metadata = kwargs.pop("video_metadata")

        sample_indices_fn = partial(self.sample_frames, **kwargs) if do_sample_frames else None
        videos, video_metadata = self._decode_and_sample_videos(
            videos,
            video_metadata=video_metadata,
            do_sample_frames=do_sample_frames,
            sample_indices_fn=sample_indices_fn,
            **kwargs,
        )
        videos = self._prepare_input_videos(videos=videos, input_data_format=input_data_format)

        kwargs = self._further_process_kwargs(**kwargs)

        # Pop kwargs that are not needed in _preprocess
        kwargs.pop("data_format")
        return_metadata = kwargs.pop("return_metadata")

        preprocessed_videos = self._preprocess(videos=videos, **kwargs)
        if return_metadata:
            preprocessed_videos["video_metadata"] = video_metadata
        return preprocessed_videos

    def _preprocess(
        self,
        videos,
        do_convert_rgb,
        do_resize,
        size,
        interpolation,
        do_center_crop,
        crop_size,
        do_rescale,
        rescale_factor,
        do_normalize,
        image_mean,
        image_std,
        return_tensors=None,
        **kwargs,
    ) -> BatchFeature:
        # Group videos by size for batched resizing
        grouped_videos, grouped_videos_index = group_videos_by_shape(videos)
        resized_videos_grouped = {}
        for shape, stacked_videos in grouped_videos.items():
            if do_convert_rgb:
                stacked_videos = self.convert_to_rgb(stacked_videos)
            if do_resize:
                stacked_videos = self.resize(stacked_videos, size=size, interpolation=interpolation)
            resized_videos_grouped[shape] = stacked_videos
        resized_videos = reorder_videos(resized_videos_grouped, grouped_videos_index)

        # Group videos by size for further processing
        # Needed in case do_resize is False, or resize returns videos with different sizes
        grouped_videos, grouped_videos_index = group_videos_by_shape(resized_videos)
        processed_videos_grouped = {}
        for shape, stacked_videos in grouped_videos.items():
            if do_center_crop:
                stacked_videos = self.center_crop(stacked_videos, crop_size)
            # Fused rescale and normalize
            stacked_videos = self.rescale_and_normalize(
                stacked_videos, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_videos_grouped[shape] = stacked_videos

        processed_videos = reorder_videos(processed_videos_grouped, grouped_videos_index)
        processed_videos = paddle.stack(processed_videos, dim=0) if return_tensors else processed_videos

        return BatchFeature(data={"pixel_values_videos": processed_videos}, tensor_type=return_tensors)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        *args,
        **kwargs,
    ):
        image_processor_dict, kwargs = cls.get_video_processor_dict(pretrained_model_name_or_path, **kwargs)
        return cls.from_dict(image_processor_dict, **kwargs)

    def save_pretrained(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs):
        """
        Save an video processor object to the directory `save_directory`, so that it can be re-loaded using the
        [`~video_processing_utils.VideoProcessorBase.from_pretrained`] class method.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the video processor JSON file will be saved (will be created if it does not exist).
            kwargs (`dict[str, Any]`, *optional*):
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        os.makedirs(save_directory, exist_ok=True)

        # If we have a custom config, we copy the file defining it in the folder and set the attributes so it can be
        # loaded from the Hub.
        if self._auto_class is not None:
            custom_object_save(self, save_directory, config=self)

        # If we save using the predefined names, we can load using `from_pretrained`
        output_video_processor_file = os.path.join(save_directory, VIDEO_PROCESSOR_NAME)

        self.to_json_file(output_video_processor_file)
        logger.info(f"Video processor saved in {output_video_processor_file}")

        return [output_video_processor_file]

    @classmethod
    def get_video_processor_dict(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        download_hub = kwargs.get("download_hub", None)
        local_files_only = kwargs.pop("local_files_only", False)

        if download_hub is None:
            download_hub = os.environ.get("DOWNLOAD_SOURCE", "huggingface")
        logger.info(f"Using download source: {download_hub}")

        cache_dir = kwargs.pop("cache_dir", None)
        subfolder = kwargs.pop("subfolder", "")

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        is_local = os.path.isdir(pretrained_model_name_or_path)
        if os.path.isfile(pretrained_model_name_or_path):
            resolved_video_processor_file = pretrained_model_name_or_path
            is_local = True
        else:
            video_processor_file = VIDEO_PROCESSOR_NAME
            try:
                resolved_video_processor_file = resolve_file_path(
                    pretrained_model_name_or_path,
                    [video_processor_file, IMAGE_PROCESSOR_NAME, PROCESSOR_NAME],
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
                    f"Expected image processor files: {VIDEO_PROCESSOR_NAME}\n"
                    f"Note: The repository ID may differ between ModelScope, AI Studio, and Hugging Face Hub.\n"
                    f"You are currently using the download source: {download_hub}. Please check the repository ID on the official website."
                )

        try:
            # Load image_processor dict
            with open(resolved_video_processor_file, encoding="utf-8") as reader:
                text = reader.read()
            video_processor_dict = json.loads(text)
            video_processor_dict = video_processor_dict.get("video_processor", video_processor_dict)

        except json.JSONDecodeError:
            raise OSError(
                f"It looks like the config file at '{resolved_video_processor_file}' is not a valid JSON file."
            )

        if is_local:
            logger.info(f"loading configuration file {resolved_video_processor_file}")
        else:
            logger.info(
                f"loading configuration file {video_processor_file} from cache at {resolved_video_processor_file}"
            )
        return video_processor_dict, kwargs

    @classmethod
    def from_dict(cls, video_processor_dict, **kwargs):
        video_processor_dict = video_processor_dict.copy()
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

        if "size" in kwargs and "size" in video_processor_dict:
            video_processor_dict["size"] = kwargs.pop("size")
        if "crop_size" in kwargs and "crop_size" in video_processor_dict:
            video_processor_dict["crop_size"] = kwargs.pop("crop_size")

        video_processor = cls(**video_processor_dict)

        # Update video_processor with kwargs if needed
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(video_processor, key):
                setattr(video_processor, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        # logger.info(f"Video processor {video_processor}")
        if return_unused_kwargs:
            return video_processor, kwargs
        else:
            return video_processor

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary.

        Returns:
            `dict[str, Any]`: Dictionary of all the attributes that make up this image processor instance.
        """
        output = deepcopy(self.__dict__)
        for method_name in self.methods_to_wrap:
            output.pop(method_name, None)
        output.pop("model_valid_processing_keys", None)
        output.pop("_valid_kwargs_names", None)
        output["video_processor_type"] = self.__class__.__name__

        return output

    def to_json_string(self) -> str:
        """
        Serializes this instance to a JSON string.

        Returns:
            `str`: String containing all the attributes that make up this feature_extractor instance in JSON format.
        """
        dictionary = self.to_dict()

        for key, value in dictionary.items():
            if isinstance(value, np.ndarray):
                dictionary[key] = value.tolist()

        # make sure private name "_processor_class" is correctly
        # saved as "processor_class"
        _processor_class = dictionary.pop("_processor_class", None)
        if _processor_class is not None:
            dictionary["processor_class"] = _processor_class

        return json.dumps(dictionary, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this image_processor instance's parameters will be saved.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())

    def fetch_videos(
        self, video_url_or_urls: Union[str, list[str], list[list[str]]], sample_indices_fn=None, **kwargs
    ):
        """
        Convert a single or a list of urls into the corresponding `np.array` objects.

        If a single url is passed, the return value will be a single object. If a list is passed a list of objects is
        returned.
        """
        video_backend = kwargs.get("video_backend", "paddlecodec")

        if isinstance(video_url_or_urls, list):
            return list(
                zip(*[self.fetch_videos(x, sample_indices_fn=sample_indices_fn, **kwargs) for x in video_url_or_urls])
            )
        else:
            return load_video(video_url_or_urls, video_backend=video_backend, sample_indices_fn=sample_indices_fn)
