# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's Transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/llava/processing_llava.py
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

# The file has been adapted from hiyouga LLaMA-Factory project
# Copyright (c) 2025 LLaMA-Factory
# Licensed under the Apache License - https://github.com/hiyouga/LLaMA-Factory/blob/main/LICENSE

import copy
import inspect
import io
import math
import os
import random
from copy import deepcopy
from dataclasses import dataclass
from typing import BinaryIO, Optional

# import librosa
import numpy as np
import requests
from decord import VideoReader, cpu
from PIL import Image
from PIL.Image import Image as ImageObject
from transformers.image_utils import is_valid_image
from typing_extensions import override

from ...utils.log import logger
from .augment_utils import (
    JpegCompression,
    RandomApply,
    RandomDiscreteRotation,
    RandomScale,
    RandomSingleSidePadding,
    transforms,
)

IMAGE_PLACEHOLDER = os.getenv("IMAGE_PLACEHOLDER", "<image>")
VIDEO_PLACEHOLDER = os.getenv("VIDEO_PLACEHOLDER", "<video>")
AUDIO_PLACEHOLDER = os.getenv("AUDIO_PLACEHOLDER", "<audio>")


def _make_batched_images(images, imglens: list[int]):
    r"""Make nested list of images."""
    batch_images = []
    for imglen in imglens:
        batch_images.append(images[:imglen])
        images = images[imglen:]

    return batch_images


def _check_video_is_nested_images(video) -> bool:
    r"""Check if the video is nested images."""
    return isinstance(video, list) and all(isinstance(frame, (str, BinaryIO, dict, ImageObject)) for frame in video)


@dataclass
class MMPluginMixin:
    image_token: Optional[str]
    video_token: Optional[str]
    audio_token: Optional[str]
    expand_mm_tokens: bool = True

    def _validate_input(
        self,
        processor,
        images,
        videos,
        audios,
    ) -> None:
        r"""Validate if this model accepts the input modalities."""
        image_processor = getattr(processor, "image_processor", None)
        video_processor = getattr(processor, "video_processor", getattr(processor, "image_processor", None))
        feature_extractor = getattr(processor, "feature_extractor", None)
        if len(images) != 0 and self.image_token is None:
            raise ValueError(
                "This model does not support image input. Please check whether the correct `template` is used."
            )

        if len(videos) != 0 and self.video_token is None:
            raise ValueError(
                "This model does not support video input. Please check whether the correct `template` is used."
            )

        if len(audios) != 0 and self.audio_token is None:
            raise ValueError(
                "This model does not support audio input. Please check whether the correct `template` is used."
            )

        if self.image_token is not None and processor is None:
            raise ValueError("Processor was not found, please check and update your model file.")

        if self.image_token is not None and image_processor is None:
            raise ValueError("Image processor was not found, please check and update your model file.")

        if self.video_token is not None and video_processor is None:
            raise ValueError("Video processor was not found, please check and update your model file.")

        if self.audio_token is not None and feature_extractor is None:
            raise ValueError("Audio feature extractor was not found, please check and update your model file.")

    def _validate_messages(
        self,
        messages,
        images,
        videos,
        audios,
    ):
        r"""Validate if the number of images, videos and audios match the number of placeholders in messages."""
        num_image_tokens, num_video_tokens, num_audio_tokens = 0, 0, 0
        for message in messages:
            num_image_tokens += message["content"].count(IMAGE_PLACEHOLDER)
            num_video_tokens += message["content"].count(VIDEO_PLACEHOLDER)
            num_audio_tokens += message["content"].count(AUDIO_PLACEHOLDER)

        if len(images) != num_image_tokens:
            raise ValueError(
                f"The number of images does not match the number of {IMAGE_PLACEHOLDER} tokens in {messages}."
            )

        if len(videos) != num_video_tokens:
            raise ValueError(
                f"The number of videos does not match the number of {VIDEO_PLACEHOLDER} tokens in {messages}."
            )

        if len(audios) != num_audio_tokens:
            raise ValueError(
                f"The number of audios does not match the number of {AUDIO_PLACEHOLDER} tokens in {messages}."
            )

    def _file_download(self, url: str) -> bytes:
        os.environ["https_proxy"] = os.environ.get("HTTPS_PROXY", "")
        os.environ["http_proxy"] = os.environ.get("HTTP_PROXY", "")
        if url.startswith("http"):
            response = requests.get(url)
            bytes_data = response.content
        elif os.path.isfile(url):
            bytes_data = open(url, "rb").read()
        else:
            raise ValueError(f"{url} is not a valid url or file path.")
        bytes_content = io.BytesIO(bytes_data)

        return bytes_content

    def _img_download(self, url: str) -> Image.Image:
        bytes_content = self._file_download(url)
        img = Image.open(bytes_content)

        return img

    def _video_download(self, url: str) -> VideoReader:
        bytes_content = self._file_download(url)
        video_reader = VideoReader(bytes_content, ctx=cpu(0), num_threads=1)

        return video_reader

    def _preprocess_image(self, image, image_max_pixels, image_min_pixels, **kwargs):
        r"""Pre-process a single image."""
        if (image.width * image.height) > image_max_pixels:
            resize_factor = math.sqrt(image_max_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if (image.width * image.height) < image_min_pixels:
            resize_factor = math.sqrt(image_min_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    def _get_video_sample_indices(self, video_reader, video_fps, video_maxlen, **kwargs):
        r"""Compute video sample indices according to fps."""
        total_frames = len(video_reader)
        if total_frames == 0:  # infinite video
            return np.linspace(0, video_maxlen - 1, video_maxlen).astype(np.int32)

        sample_frames = max(1, math.floor(float(total_frames / video_reader.get_avg_fps()) * video_fps))
        sample_frames = min(total_frames, video_maxlen, sample_frames)
        start_frame, end_frame = 0, total_frames - 1
        frame_indices = np.linspace(start_frame, end_frame, sample_frames).round()

        return frame_indices

    def _regularize_images(self, images, **kwargs):
        r"""Regularize images to avoid error. Including reading and pre-processing."""
        results = []
        for image in images:
            image = self._img_download(image)
            results.append(self._preprocess_image(image, **kwargs))

        return {"images": results}

    def _regularize_videos(self, videos, **kwargs):
        r"""Regularizes videos to avoid error. Including reading, resizing and converting."""
        results = []
        for video in videos:
            frames = []
            if _check_video_is_nested_images(video):
                for frame in video:
                    if not is_valid_image(frame) and not isinstance(frame, dict) and not os.path.exists(frame):
                        raise ValueError("Invalid image found in video frames.")
                frames = video
            else:
                video_reader = self._video_download(video)
                sample_indices = self._get_video_sample_indices(video_reader, **kwargs)
                try:
                    frames = video_reader.get_batch(sample_indices)
                    video_reader.seek(0)
                except Exception:
                    logger.info(f"get {sample_indices} frames error")

            regularized_frames = []
            for frame in frames:
                regularized_frames.append(self._preprocess_image(frame, **kwargs))
            results.append(regularized_frames)

        return {"videos": results}

    def _regularize_audios(self, audios, sampling_rate: float, **kwargs):
        r"""Regularizes audios to avoid error. Including reading and resampling."""
        results, sampling_rates = [], []
        for audio in audios:
            if not isinstance(audio, np.ndarray):
                # audio, sampling_rate = librosa.load(audio, sr=sampling_rate)
                audio, sampling_rate = None, None

            results.append(audio)
            sampling_rates.append(sampling_rate)

        return {"audios": results, "sampling_rates": sampling_rates}

    def _get_mm_inputs(
        self,
        images,
        videos,
        audios,
        processor,
        imglens=None,
    ):
        mm_inputs = {}
        if len(images) != 0:
            image_processor = getattr(processor, "image_processor", None)
            images = self._regularize_images(
                images,
                image_max_pixels=getattr(processor, "image_max_pixels", 768 * 768),
                image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32),
            )["images"]
            if imglens is not None:  # if imglens are provided, make batched images
                images = _make_batched_images(images, imglens)

            mm_inputs.update(image_processor(images, return_tensors="pd"))

        if len(videos) != 0:
            video_processor = getattr(processor, "video_processor", getattr(processor, "image_processor", None))
            videos = self._regularize_videos(
                videos,
                image_max_pixels=getattr(processor, "video_max_pixels", 256 * 256),
                image_min_pixels=getattr(processor, "video_min_pixels", 16 * 16),
                video_fps=getattr(processor, "video_fps", 2.0),
                video_maxlen=getattr(processor, "video_maxlen", 128),
            )["videos"]
            if "videos" in inspect.signature(video_processor.preprocess).parameters:  # for qwen2_vl and video_llava
                mm_inputs.update(video_processor(images=None, videos=videos, return_tensors="pd"))
            else:  # for llava_next_video
                mm_inputs.update(video_processor(videos, return_tensors="pd"))

        if len(audios) != 0:
            feature_extractor = getattr(processor, "feature_extractor", None)
            audios = self._regularize_audios(
                audios,
                sampling_rate=getattr(processor, "audio_sampling_rate", 16000),
            )["audios"]
            mm_inputs.update(
                feature_extractor(
                    audios,
                    sampling_rate=getattr(processor, "audio_sampling_rate", 16000),
                    return_attention_mask=True,
                    padding="max_length",
                    return_tensors="pd",
                )
            )
            mm_inputs["feature_attention_mask"] = mm_inputs.pop("attention_mask", None)  # prevent conflicts

        return mm_inputs


@dataclass
class BasePlugin(MMPluginMixin):
    def process_messages(
        self,
        messages,
        images,
        videos,
        audios,
        mm_inputs,
        processor,
    ):
        r"""Pre-process input messages before tokenization for VLMs."""
        self._validate_input(processor, images, videos, audios)
        return messages

    def get_mm_inputs(
        self,
        images,
        videos,
        audios,
        imglens,
        vidlens,
        audlens,
        batch_ids,
        processor,
    ):
        r"""Build batched multimodal inputs for VLMs."""
        self._validate_input(processor, images, videos, audios)
        return self._get_mm_inputs(images, videos, audios, processor)


@dataclass
class PaddleOCRVLPlugin(BasePlugin):
    image_bos_token: str = "<|IMAGE_START|>"
    image_eos_token: str = "<|IMAGE_END|>"

    def __init__(self, image_token, video_token, audio_token, **kwargs):
        super().__init__(image_token, video_token, audio_token, **kwargs)
        self.image_augmentation = self.get_ocr_augmentations(
            rotation_degrees=[90, 270],
            rotation_p=0.1,
            jpeg_quality_range=(60, 100),
            jpeg_p=0.3,
            scale_range=(0.5, 1.5),
            scale_p=0.5,
            padding_range=(0, 15),
            padding_p=0.1,
            color_jitter_p=0.1,
        )

    def get_ocr_augmentations(
        self,
        scale_range=(0.8, 1.2),
        scale_p=0.5,
        padding_range=(0, 15),
        padding_p=0.5,
        rotation_degrees=[0],
        rotation_p=0.5,
        color_jitter_p=0.5,
        jpeg_quality_range=(40, 90),
        jpeg_p=0.5,
    ):

        augmentations = []

        if scale_p > 0:
            scale_transform = RandomScale(scale_range=scale_range)
            augmentations.append(RandomApply([scale_transform], p=scale_p))

        if padding_p > 0:
            padding_transform = RandomSingleSidePadding(padding_range=padding_range, fill="white")
            augmentations.append(RandomApply([padding_transform], p=padding_p))

        if rotation_p > 0 and rotation_degrees:
            rotation_transform = RandomDiscreteRotation(degrees=rotation_degrees, interpolation="nearest", expand=True)
            augmentations.append(RandomApply([rotation_transform], p=rotation_p))

        if color_jitter_p > 0:
            color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1)
            augmentations.append(RandomApply([color_jitter], p=color_jitter_p))

        if jpeg_p > 0:
            jpeg_transform = JpegCompression(quality_range=jpeg_quality_range)
            augmentations.append(RandomApply([jpeg_transform], p=jpeg_p))

        return transforms.Compose(augmentations)

    @override
    def _preprocess_image(self, image, **kwargs):

        width, height = image.size
        image_max_pixels = kwargs["image_max_pixels"]
        image_min_pixels = kwargs["image_min_pixels"]
        image_processor = kwargs["image_processor"]

        # pre-resize before augmentation
        resized_height, resized_width = image_processor.get_smarted_resize(
            height,
            width,
            min_pixels=image_min_pixels,
            max_pixels=image_max_pixels,
        )[0]

        image = image.resize((resized_width, resized_height))

        if image and hasattr(self, "image_augmentation"):
            image = self.image_augmentation(image)

        return image

    @override
    def _get_mm_inputs(
        self,
        images,
        videos,
        audios,
        processor,
    ):
        image_processor = getattr(processor, "image_processor", None)
        mm_inputs = {}
        if len(images) != 0:
            images = self._regularize_images(
                images,
                image_max_pixels=getattr(processor, "max_pixels", 2822400),
                image_min_pixels=getattr(processor, "min_pixels", 147384),
                image_processor=image_processor,
            )["images"]
            mm_inputs.update(image_processor(images, return_tensors="pd"))

        return mm_inputs

    @override
    def process_messages(
        self,
        messages,
        images,
        videos,
        audios,
        mm_inputs,
        processor,
    ):
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens = 0
        messages = deepcopy(messages)
        image_processor = getattr(processor, "image_processor")

        merge_length = getattr(image_processor, "merge_size") ** 2
        if self.expand_mm_tokens:
            image_grid_thw = mm_inputs.get("image_grid_thw", [])
        else:
            image_grid_thw = [None] * len(images)

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                image_seqlen = (
                    image_grid_thw[num_image_tokens].prod().item() // merge_length if self.expand_mm_tokens else 1
                )
                content = content.replace(
                    IMAGE_PLACEHOLDER,
                    f"{self.image_bos_token}{self.image_token * image_seqlen}{self.image_eos_token}",
                    1,
                )
                num_image_tokens += 1

            message["content"] = content

        return messages


@dataclass
class ErnieVLPlugin(BasePlugin):
    image_bos_token: str = "<|IMAGE_START|>"
    image_eos_token: str = "<|IMAGE_END|>"
    vision_bos_token: str = "<|VIDEO_START|>"
    vision_eos_token: str = "<|VIDEO_END|>"

    def convert_to_rgb(self, image: Image.Image) -> Image.Image:
        def has_transparent_background(img):
            """has_transparent_background"""
            if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                # Check for any pixel with alpha channel less than 255 (fully opaque)
                alpha = img.convert("RGBA").split()[-1]
                if alpha.getextrema()[0] < 255:
                    return True
            return False

        def add_white_background(img):
            """
            Add a white background to a transparent background image
            """
            if img.mode != "RGBA":
                img = img.convert("RGBA")
            # Create an image with a white background and the same size as the original image
            img_white_background = Image.new("RGBA", img.size, (255, 255, 255))

            # Paste the original image onto a white background
            img_white_background.paste(img, (0, 0), img)

            return img_white_background

        def change_I16_to_L(img):
            """
            Convert image from I;16 mode to L mode
            """
            # Since the point function in I mode only supports addition, subtraction, and multiplication, the following * (1 / 256) cannot be changed to division.
            return img.point(lambda i: i * (1 / 256)).convert("L")

        try:
            if image.mode == "I;16":
                image = change_I16_to_L(image)
            if has_transparent_background(image):
                image = add_white_background(image)
        except Exception:
            pass
        return image.convert("RGB")

    @override
    def _preprocess_image(self, image, **kwargs):
        image = self.convert_to_rgb(image)
        return image

    @override
    def _get_video_sample_indices(self, video_reader, video_fps, video_maxlen, frames_sample, **kwargs):
        r"""Compute video sample indices according to fps."""
        total_frames = len(video_reader)
        duration = total_frames / video_reader.get_avg_fps()
        if total_frames == 0:  # infinite video
            return np.linspace(0, video_maxlen - 1, video_maxlen).astype(np.int32)

        sample_frames = max(1, math.floor(float(duration) * video_fps))
        sample_frames = min(total_frames, video_maxlen, sample_frames)

        assert frames_sample in ["rand", "middle", "leading"]
        intervals = np.linspace(start=0, stop=total_frames, num=sample_frames + 1).astype(int)

        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if frames_sample == "rand":
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except Exception:
                frame_indices = np.random.permutation(total_frames)[:sample_frames]
                frame_indices.sort()
                frame_indices = list(frame_indices)
        elif frames_sample == "leading":
            frame_indices = [x[0] for x in ranges]
        elif frames_sample == "middle":
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError
        time_stamps = [frame_idx * duration / total_frames for frame_idx in frame_indices]

        return frame_indices, time_stamps

    @override
    def _regularize_videos(self, videos, **kwargs):
        results = []
        processor = kwargs.get("processor", None)
        for video in videos:
            frames = []
            if _check_video_is_nested_images(video):
                for frame in video:
                    if not is_valid_image(frame) and not isinstance(frame, dict) and not os.path.exists(frame):
                        raise ValueError("Invalid image found in video frames.")

                frames = video
                time_stamps = [idx / kwargs.get("video_fps", 2.0) for idx in range(len(frames))]
            else:
                video_reader = self._video_download(video)
                sample_indices, time_stamps = self._get_video_sample_indices(video_reader, **kwargs)
                try:
                    frames = video_reader.get_batch(sample_indices).asnumpy()
                    video_reader.seek(0)
                except Exception:
                    logger.info(f"get {sample_indices} frames error")

            if len(frames) % 2 != 0:
                padded_image = copy.deepcopy(frames[-1])
                padded_stamp = copy.deepcopy(time_stamps[-1])
                frames = np.concatenate([frames, padded_image[np.newaxis, ...]], axis=0)
                time_stamps.append(padded_stamp)

            rendered_frames = []
            for frame, time_stamp in zip(frames, time_stamps):
                frame = Image.fromarray(frame, "RGB")
                try:
                    frame = processor.render_frame_timestamp(frame, time_stamp)
                except:
                    rendered_frames = frames
                    break
                rendered_frames.append(np.array(frame.convert("RGB")))

            results.append(rendered_frames)

        return {"videos": results}

    @override
    def _get_mm_inputs(
        self,
        images,
        videos,
        audios,
        processor,
    ):
        image_processor = getattr(processor, "image_processor", None)
        mm_inputs = {}
        if len(images) != 0:
            images = self._regularize_images(
                images,
                image_max_pixels=getattr(processor, "max_pixels", 28 * 28 * 1280),
                image_min_pixels=getattr(processor, "min_pixels", 56 * 56),
            )["images"]
            mm_inputs.update(image_processor(images=images, return_tensors="pd"))

        if len(videos) != 0:
            videos = self._regularize_videos(
                videos,
                image_max_pixels=getattr(processor, "video_max_pixels", 28 * 28 * 1280),
                image_min_pixels=getattr(processor, "video_min_pixels", 56 * 56),
                video_fps=getattr(processor, "video_fps", 2.0),
                video_maxlen=getattr(processor, "video_maxlen", 180),
                frames_sample=getattr(processor, "frames_sample", "middle"),
                processor=processor,
            )["videos"]
            mm_inputs.update(image_processor(images=None, videos=videos, return_tensors="pd"))

        return mm_inputs

    @override
    def process_messages(
        self,
        messages,
        images,
        videos,
        audios,
        mm_inputs,
        processor,
    ):
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens, num_video_tokens = 0, 0
        messages = deepcopy(messages)
        image_processor = getattr(processor, "image_processor")

        merge_length = getattr(image_processor, "merge_size") ** 2
        temporal_conv_size = getattr(image_processor, "temporal_conv_size")
        if self.expand_mm_tokens:
            image_grid_thw = mm_inputs.get("image_grid_thw", [])
            video_grid_thw = mm_inputs.get("video_grid_thw", [])
        else:
            image_grid_thw = [None] * len(images)
            video_grid_thw = [None] * len(videos)

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                image_seqlen = (
                    image_grid_thw[num_image_tokens].prod().item() // merge_length if self.expand_mm_tokens else 1
                )
                content = content.replace(
                    IMAGE_PLACEHOLDER,
                    f"Picture {num_image_tokens + 1}:{self.image_bos_token}{self.image_token * image_seqlen}{self.image_eos_token}",
                    1,
                )
                num_image_tokens += 1

            while VIDEO_PLACEHOLDER in content:
                video_seqlen = (
                    video_grid_thw[num_video_tokens].prod().item() // merge_length // temporal_conv_size
                    if self.expand_mm_tokens
                    else 1
                )
                content = content.replace(
                    VIDEO_PLACEHOLDER,
                    f"Video {num_video_tokens + 1}:{self.vision_bos_token}{self.video_token * video_seqlen}{self.vision_eos_token}",
                    1,
                )
                num_video_tokens += 1

            message["content"] = content

        return messages


@dataclass
class Qwen2VLPlugin(BasePlugin):
    vision_bos_token: str = "<|vision_start|>"
    vision_eos_token: str = "<|vision_end|>"

    @override
    def _preprocess_image(self, image, **kwargs):
        image = super()._preprocess_image(image, **kwargs)
        if min(image.width, image.height) < 28:
            width, height = max(image.width, 28), max(image.height, 28)
            image = image.resize((width, height))

        if image.width / image.height > 200:
            width, height = image.height * 180, image.height
            image = image.resize((width, height))

        if image.height / image.width > 200:
            width, height = image.width, image.width * 180
            image = image.resize((width, height))

        return image

    @override
    def _regularize_videos(self, videos, **kwargs):
        results, fps_per_video = [], []
        for video in videos:
            frames = []
            if _check_video_is_nested_images(video):
                for frame in video:
                    if not is_valid_image(frame) and not isinstance(frame, dict) and not os.path.exists(frame):
                        raise ValueError("Invalid image found in video frames.")

                frames = video
                fps_per_video.append(kwargs.get("video_fps", 2.0))
            else:
                video_reader = self._video_download(video)
                sample_indices = self._get_video_sample_indices(video_reader, **kwargs)
                try:
                    frames = video_reader.get_batch(sample_indices).asnumpy()
                    video_reader.seek(0)
                except Exception:
                    logger.info(f"get {sample_indices} frames error")

                try:
                    fps_per_video.append(video_reader.get_avg_fps())
                except:
                    fps_per_video.append(kwargs.get("video_fps", 2.0))

            if len(frames) % 2 != 0:
                padded_image = copy.deepcopy(frames[-1])
                frames = np.concatenate([frames, padded_image[np.newaxis, ...]], axis=0)

            regularized_frames = []
            for frame in frames:
                if isinstance(frame, np.ndarray):
                    frame = Image.fromarray(frame, "RGB")
                regularized_frames.append(self._preprocess_image(frame, **kwargs))
            results.append(regularized_frames)

        return {"videos": results, "fps_per_video": fps_per_video}

    @override
    def _get_mm_inputs(
        self,
        images,
        videos,
        audios,
        processor,
    ):
        image_processor = getattr(processor, "image_processor", None)
        mm_inputs = {}
        if len(images) != 0:
            images = self._regularize_images(
                images,
                image_max_pixels=getattr(processor, "image_max_pixels", 768 * 768),
                image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32),
            )["images"]
            mm_inputs.update(image_processor(images, return_tensors="pd"))

        if len(videos) != 0:
            video_data = self._regularize_videos(
                videos,
                image_max_pixels=getattr(processor, "video_max_pixels", 256 * 256),
                image_min_pixels=getattr(processor, "video_min_pixels", 16 * 16),
                video_fps=getattr(processor, "video_fps", 2.0),
                video_maxlen=getattr(processor, "video_maxlen", 128),
            )
            mm_inputs.update(image_processor(images=None, videos=video_data["videos"], return_tensors="pd"))
            temporal_patch_size: int = getattr(image_processor, "temporal_patch_size", 2)
            if "second_per_grid_ts" in processor.model_input_names:
                mm_inputs["second_per_grid_ts"] = [temporal_patch_size / fps for fps in video_data["fps_per_video"]]

        return mm_inputs

    @override
    def process_messages(
        self,
        messages,
        images,
        videos,
        audios,
        mm_inputs,
        processor,
    ):
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens, num_video_tokens = 0, 0
        messages = deepcopy(messages)
        image_processor = getattr(processor, "image_processor")

        merge_length = getattr(image_processor, "merge_size") ** 2
        if self.expand_mm_tokens:
            image_grid_thw = mm_inputs.get("image_grid_thw", [])
            video_grid_thw = mm_inputs.get("video_grid_thw", [])
        else:
            image_grid_thw = [None] * len(images)
            video_grid_thw = [None] * len(videos)

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                image_seqlen = (
                    image_grid_thw[num_image_tokens].prod().item() // merge_length if self.expand_mm_tokens else 1
                )
                content = content.replace(
                    IMAGE_PLACEHOLDER,
                    f"{self.vision_bos_token}{self.image_token * image_seqlen}{self.vision_eos_token}",
                    1,
                )
                num_image_tokens += 1

            while VIDEO_PLACEHOLDER in content:
                video_seqlen = (
                    video_grid_thw[num_video_tokens].prod().item() // merge_length if self.expand_mm_tokens else 1
                )
                content = content.replace(
                    VIDEO_PLACEHOLDER,
                    f"{self.vision_bos_token}{self.video_token * video_seqlen}{self.vision_eos_token}",
                    1,
                )
                num_video_tokens += 1

            message["content"] = content

        return messages


@dataclass
class Qwen3VLPlugin(Qwen2VLPlugin):
    @override
    def _get_mm_inputs(
        self,
        images,
        videos,
        audios,
        processor,
    ):
        image_processor = getattr(processor, "image_processor", None)
        video_processor = getattr(processor, "video_processor", None)
        mm_inputs = {}
        if len(images) != 0:
            images = self._regularize_images(
                images,
                image_max_pixels=getattr(processor, "image_max_pixels", 768 * 768),
                image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32),
            )["images"]
            mm_inputs.update(image_processor(images, return_tensors="pd"))

        if len(videos) != 0:
            videos = self._regularize_videos(
                videos,
                image_max_pixels=getattr(processor, "video_max_pixels", 256 * 256),
                image_min_pixels=getattr(processor, "video_min_pixels", 16 * 16),
                video_fps=getattr(processor, "video_fps", 2.0),
                video_maxlen=getattr(processor, "video_maxlen", 128),
            )
            video_metadata = [
                {"fps": getattr(processor, "video_fps", 24.0), "duration": len(video), "total_num_frames": len(video)}
                for video in videos["videos"]
            ]
            mm_inputs.update(
                video_processor(videos=videos["videos"], video_metadata=video_metadata, return_metadata=True)
            )
            temporal_patch_size = getattr(image_processor, "temporal_patch_size", 2)
            if "second_per_grid_ts" in processor.model_input_names:
                mm_inputs["second_per_grid_ts"] = [temporal_patch_size / fps for fps in videos["fps_per_video"]]

        return mm_inputs

    @override
    def process_messages(
        self,
        messages,
        images,
        videos,
        audios,
        mm_inputs,
        processor,
    ):
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens, num_video_tokens = 0, 0
        messages = deepcopy(messages)
        image_processor = getattr(processor, "image_processor")
        video_processor = getattr(processor, "video_processor")

        image_merge_length = getattr(image_processor, "merge_size") ** 2
        video_merge_length = getattr(video_processor, "merge_size") ** 2
        if self.expand_mm_tokens:
            image_grid_thw = mm_inputs.get("image_grid_thw", [])
            video_grid_thw = mm_inputs.get("video_grid_thw", [])
            num_frames = video_grid_thw[0][0] if len(video_grid_thw) > 0 else 0
            video_metadata = mm_inputs.get("video_metadata", {})

        else:
            image_grid_thw = [None] * len(images)
            video_grid_thw = [None] * len(videos)
            num_frames = 0
            timestamps = [0]

        for idx, message in enumerate(messages):
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                if num_image_tokens >= len(image_grid_thw):
                    raise ValueError(f"Found more {IMAGE_PLACEHOLDER} tags than actual images provided.")

                image_seqlen = (
                    image_grid_thw[num_image_tokens].prod().item() // image_merge_length
                    if self.expand_mm_tokens
                    else 1
                )
                content = content.replace(
                    IMAGE_PLACEHOLDER,
                    f"{self.vision_bos_token}{self.image_token * image_seqlen}{self.vision_eos_token}",
                    1,
                )
                num_image_tokens += 1

            while VIDEO_PLACEHOLDER in content:
                if num_video_tokens >= len(video_grid_thw):
                    raise ValueError(f"Found more {VIDEO_PLACEHOLDER} tags than actual videos provided.")

                metadata = video_metadata[idx]
                timestamps = processor._calculate_timestamps(
                    metadata.frames_indices,
                    metadata.fps,
                    video_processor.merge_size,
                )
                video_structure = ""
                for frame_index in range(num_frames):
                    video_seqlen = (
                        video_grid_thw[num_video_tokens][1:].prod().item() // video_merge_length
                        if self.expand_mm_tokens
                        else 1
                    )
                    timestamp_sec = timestamps[frame_index]
                    frame_structure = (
                        f"<{timestamp_sec:.1f} seconds>"
                        f"{self.vision_bos_token}{self.video_token * video_seqlen}{self.vision_eos_token}"
                    )
                    video_structure += frame_structure

                if not self.expand_mm_tokens:
                    video_structure = f"{self.vision_bos_token}{self.video_token}{self.vision_eos_token}"

                content = content.replace(VIDEO_PLACEHOLDER, video_structure, 1)
                num_video_tokens += 1

            message["content"] = content

        return messages


@dataclass
class GLM4VPlugin(Qwen2VLPlugin):
    @override
    def _get_mm_inputs(
        self,
        images,
        videos,
        audios,
        processor,
    ):
        image_processor = getattr(processor, "image_processor", None)
        video_processor = getattr(processor, "video_processor", None)
        mm_inputs = {}
        if len(images) != 0:
            images = self._regularize_images(
                images,
                image_max_pixels=getattr(processor, "image_max_pixels", 768 * 768),
                image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32),
            )["images"]
            mm_inputs.update(image_processor(images, return_tensors="pd"))

        if len(videos) != 0:
            video_data = self._regularize_videos(
                videos,
                image_max_pixels=getattr(processor, "video_max_pixels", 256 * 256),
                image_min_pixels=getattr(processor, "video_min_pixels", 16 * 16),
                video_fps=getattr(processor, "video_fps", 2.0),
                video_maxlen=getattr(processor, "video_maxlen", 128),
            )
            # prepare video metadata
            video_metadata = [
                {"fps": 2, "duration": len(video), "total_frames": len(video)} for video in video_data["videos"]
            ]
            mm_inputs.update(video_processor(images=None, videos=video_data["videos"], video_metadata=video_metadata))

        return mm_inputs

    @override
    def process_messages(
        self,
        messages,
        images,
        videos,
        audios,
        mm_inputs,
        processor,
    ):
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens, num_video_tokens = 0, 0
        messages = deepcopy(messages)
        image_processor = getattr(processor, "image_processor")

        merge_length = getattr(image_processor, "merge_size") ** 2
        if self.expand_mm_tokens:
            image_grid_thw = mm_inputs.get("image_grid_thw", [])
            video_grid_thw = mm_inputs.get("video_grid_thw", [])
            num_frames = video_grid_thw[0][0] if len(video_grid_thw) > 0 else 0  # hard code for now
            timestamps = mm_inputs.get("timestamps", [])

            if hasattr(timestamps, "tolist"):
                timestamps = timestamps.tolist()

            if not timestamps:
                timestamps_list = []
            elif isinstance(timestamps[0], list):
                timestamps_list = timestamps[0]
            else:
                timestamps_list = timestamps

            unique_timestamps = timestamps_list.copy()
            selected_timestamps = unique_timestamps[:num_frames]
            while len(selected_timestamps) < num_frames:
                selected_timestamps.append(selected_timestamps[-1] if selected_timestamps else 0)

        else:
            image_grid_thw = [None] * len(images)
            video_grid_thw = [None] * len(videos)
            num_frames = 0
            selected_timestamps = [0]

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                image_seqlen = (
                    image_grid_thw[num_image_tokens].prod().item() // merge_length if self.expand_mm_tokens else 1
                )
                content = content.replace(
                    IMAGE_PLACEHOLDER, f"<|begin_of_image|>{self.image_token * image_seqlen}<|end_of_image|>", 1
                )
                num_image_tokens += 1

            while VIDEO_PLACEHOLDER in content:
                video_structure = ""
                for frame_index in range(num_frames):
                    video_seqlen = (
                        video_grid_thw[num_video_tokens][1:].prod().item() // merge_length
                        if self.expand_mm_tokens
                        else 1
                    )
                    timestamp_sec = selected_timestamps[frame_index]
                    frame_structure = (
                        f"<|begin_of_image|>{self.image_token * video_seqlen}<|end_of_image|>{timestamp_sec}"
                    )
                    video_structure += frame_structure

                if not self.expand_mm_tokens:
                    video_structure = self.video_token

                content = content.replace(VIDEO_PLACEHOLDER, f"<|begin_of_video|>{video_structure}<|end_of_video|>", 1)
                num_video_tokens += 1

            message["content"] = content

        return messages

    @override
    def get_mm_inputs(
        self,
        images,
        videos,
        audios,
        imglens,
        vidlens,
        audlens,
        batch_ids,
        processor,
    ):
        self._validate_input(processor, images, videos, audios)
        mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
        mm_inputs.pop("timestamps", None)
        return mm_inputs


@dataclass
class Gemma3Plugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages,
        images,
        videos,
        audios,
        mm_inputs,
        processor,
    ):
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens = 0
        messages = deepcopy(messages)
        boi_token = getattr(processor, "boi_token")
        full_image_sequence = getattr(processor, "full_image_sequence")
        image_str = full_image_sequence if self.expand_mm_tokens else boi_token

        do_pan_and_scan = getattr(processor, "image_do_pan_and_scan", False)

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                if do_pan_and_scan:
                    image_placeholder_str = (
                        "Here is the original image {{image}} and here are some crops to help you see better "
                        + " ".join(["{{image}}"] * mm_inputs["num_crops"][0][num_image_tokens])
                    )
                else:
                    image_placeholder_str = "{{image}}"

                content = content.replace(IMAGE_PLACEHOLDER, image_placeholder_str, 1)
                num_image_tokens += 1

            message["content"] = content.replace("{{image}}", image_str)

        return messages

    @override
    def get_mm_inputs(
        self,
        images,
        videos,
        audios,
        imglens,
        vidlens,
        audlens,
        batch_ids,
        processor,
    ):
        self._validate_input(processor, images, videos, audios)
        mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
        mm_inputs.pop("num_crops", None)
        return mm_inputs


PLUGINS = {
    "base": BasePlugin,
    "ernie_vl": ErnieVLPlugin,
    "qwen2_vl": Qwen2VLPlugin,
    "paddleocr_vl": PaddleOCRVLPlugin,
    "qwen3_vl": Qwen3VLPlugin,
    "glm4v": GLM4VPlugin,
    "gemma3": Gemma3Plugin,
}


def register_mm_plugin(name: str, plugin_class: type["BasePlugin"]) -> None:
    r"""Register a multimodal plugin."""
    if name in PLUGINS:
        raise ValueError(f"Multimodal plugin {name} already exists.")

    PLUGINS[name] = plugin_class


def get_mm_plugin(
    name: str,
    image_token: Optional[str] = None,
    video_token: Optional[str] = None,
    audio_token: Optional[str] = None,
    **kwargs,
) -> "BasePlugin":
    r"""Get plugin for multimodal inputs."""
    if name not in PLUGINS:
        raise ValueError(f"Multimodal plugin `{name}` not found.")

    return PLUGINS[name](image_token, video_token, audio_token, **kwargs)
