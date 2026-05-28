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
import sys
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, BinaryIO, Optional

import librosa
import numpy as np
import paddle
import requests
from PIL import Image
from PIL.Image import Image as ImageObject
from transformers.image_utils import is_valid_image
from typing_extensions import override

from paddleformers.transformers.qwen2_vl.vision_process import fetch_image, fetch_video
from paddleformers.transformers.qwen3_omni_moe.processor import (
    Qwen3OmniMoeProcessorKwargs,
)

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
os.environ["https_proxy"] = os.environ.get("HTTPS_PROXY", "")
os.environ["http_proxy"] = os.environ.get("HTTP_PROXY", "")


def _create_video_decoder(video_src):
    """Create a paddlecodec VideoDecoder via the Paddle proxy import."""
    try:
        del sys.modules["torchcodec"]
        paddle.enable_compat(scope={"torchcodec"})
        from torchcodec.decoders import VideoDecoder

        sys.modules["torchcodec"] = None
    except (ImportError, RuntimeError) as e:
        logger.error(
            f"Failed to load 'torchcodec' backend via Paddle proxy.\n"
            f"  - Recommended Fix Steps:\n"
            f"    1. Install dependencies: `conda install ffmpeg -c conda-forge`\n"
            f"    2. Reinstall packages: `pip install paddlecodec --force-reinstall`\n"
            f"  - Original Error: {e}"
        )
        raise
    PADDLECODEC_NUM_THREADS = int(os.environ.get("PADDLECODEC_NUM_THREADS", 0))
    return VideoDecoder(video_src, num_ffmpeg_threads=PADDLECODEC_NUM_THREADS)


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
        if url.startswith("http"):
            response = requests.get(url)
            bytes_data = response.content
        elif os.path.isfile(url):
            with open(url, "rb") as f:
                bytes_data = f.read()
        else:
            raise ValueError(f"{url} is not a valid url or file path.")
        bytes_content = io.BytesIO(bytes_data)

        return bytes_content

    def _img_download(self, url: str) -> Image.Image:
        bytes_content = self._file_download(url)
        img = Image.open(bytes_content)

        return img

    def _video_download(self, url: str):
        if os.path.isfile(url):
            return _create_video_decoder(url)
        # For URLs, download bytes and pass as paddle tensor (raw encoded bytes)
        response = requests.get(url)
        video_tensor = paddle.to_tensor(np.frombuffer(response.content, dtype="uint8"))
        return _create_video_decoder(video_tensor)

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
        total_frames = video_reader.metadata.num_frames
        if total_frames == 0:  # infinite video
            return np.linspace(0, video_maxlen - 1, video_maxlen).astype(np.int32)

        sample_frames = max(1, math.floor(float(total_frames / video_reader.metadata.average_fps) * video_fps))
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
                    frames = (
                        video_reader.get_frames_at(indices=[int(i) for i in sample_indices])
                        .data.contiguous()
                        .cpu()
                        .numpy()
                        .transpose(0, 2, 3, 1)  # [N,C,H,W] -> [N,H,W,C]
                    )
                    paddle.disable_compat()
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
                audio, _ = librosa.load(audio, sr=sampling_rate, mono=True)
            results.append(audio)
            sampling_rates.append(sampling_rate)

        return {"audios": results, "sampling_rates": sampling_rates}

    def _get_mm_inputs(
        self,
        images,
        videos,
        audios,
        processor,
        **kwargs,
    ):
        mm_inputs = {}
        if len(images) != 0:
            image_processor = getattr(processor, "image_processor", None)
            images = self._regularize_images(
                images,
                image_max_pixels=getattr(processor, "image_max_pixels", 768 * 768),
                image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32),
            )["images"]
            imglens = kwargs.get("imglens", None)
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

    def process_tokens(self, tokens, processor):
        r"""Pre-process input tokens for VLMs."""

        labels = deepcopy(tokens)

        tokenizer = getattr(processor, "tokenizer")

        masked_tokens = getattr(self, "masked_tokens", None)
        if masked_tokens:
            masked_tokens_ids = tokenizer.convert_tokens_to_ids(masked_tokens)

            if len(masked_tokens) != len(masked_tokens_ids):
                raise ValueError(
                    f"The number of masked tokens {masked_tokens} does not match the number of masked tokens ids {masked_tokens_ids} tokens."
                )

            # Mask tokens that should be ignored in loss calculation
            for i, token in enumerate(labels):
                if token in masked_tokens_ids:
                    labels[i] = -100

        return labels

    def get_mm_inputs(
        self,
        images,
        videos,
        audios,
        processor,
        **kwargs,
    ):
        r"""Build batched multimodal inputs for VLMs."""
        # imglens = kwargs.get("imglens", None)
        # vidlens = kwargs.get("vidlens", None)
        # audlens = kwargs.get("audlens", None)
        # batch_ids = kwargs.get("batch_ids", None)

        self._validate_input(processor, images, videos, audios)
        return self._get_mm_inputs(images, videos, audios, processor, **kwargs)


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
        **kwargs,
    ):
        image_processor = getattr(processor, "image_processor", None)
        mm_inputs = {}
        if len(images) != 0:
            images = self._regularize_images(
                images,
                image_max_pixels=getattr(image_processor, "max_pixels", 2822400),
                image_min_pixels=getattr(image_processor, "min_pixels", 147384),
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

        self.masked_tokens = [self.image_token, self.image_bos_token, self.image_eos_token]

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
        total_frames = video_reader.metadata.num_frames
        duration = total_frames / video_reader.metadata.average_fps
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
                    frames = (
                        video_reader.get_frames_at(indices=[int(i) for i in sample_indices])
                        .data.contiguous()
                        .cpu()
                        .numpy()
                        .transpose(0, 2, 3, 1)  # [N,C,H,W] -> [N,H,W,C]
                    )
                    paddle.disable_compat()
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
                except Exception:
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
        **kwargs,
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
                    frames = (
                        video_reader.get_frames_at(indices=[int(i) for i in sample_indices])
                        .data.contiguous()
                        .cpu()
                        .numpy()
                        .transpose(0, 2, 3, 1)  # [N,C,H,W] -> [N,H,W,C]
                    )
                    paddle.disable_compat()
                except Exception:
                    logger.info(f"get {sample_indices} frames error")

                try:
                    fps_per_video.append(video_reader.metadata.average_fps)
                except Exception:
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
        **kwargs,
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
class Qwen2OmniPlugin(Qwen2VLPlugin):
    audio_bos_token: str = "<|audio_start|>"
    audio_eos_token: str = "<|audio_end|>"

    @override
    def _get_mm_inputs(
        self,
        images,
        videos,
        audios,
        processor,
        **kwargs,
    ) -> None:

        image_processor = getattr(processor, "image_processor", None)
        video_processor = getattr(processor, "video_processor", None)
        feature_extractor = getattr(processor, "feature_extractor", None)
        patch_size = getattr(image_processor, "patch_size", None)
        mm_inputs = {}

        if len(images) != 0:
            processed_images = []
            for image in images:
                _image = fetch_image({"image": image}, image_patch_size=patch_size)
                processed_images.append(_image)
            mm_inputs.update(image_processor(processed_images, return_tensors="pd"))

        if len(videos) != 0:
            if processor.__class__.__name__ == "Qwen3OmniMoeProcessor":  # for qwen3omni
                videos_kwargs = Qwen3OmniMoeProcessorKwargs._defaults.get("videos_kwargs")
                fps = videos_kwargs.get("fps", 1.0)
                processed_videos = []
                for video in videos:
                    _video = fetch_video({"video": video}, image_patch_size=patch_size)
                    if isinstance(_video, paddle.Tensor):
                        _video = paddle.cast(_video, "uint8")
                    processed_videos.append(_video)
                video_inputs = video_processor(videos=processed_videos, **videos_kwargs, return_tensors="pd")
                mm_inputs.update(video_inputs)
                fps = [fps] * len(processed_videos)
            else:
                video_data = self._regularize_videos(
                    videos,
                    image_max_pixels=getattr(processor, "video_max_pixels", 256 * 256),
                    image_min_pixels=getattr(processor, "video_min_pixels", 16 * 16),
                    video_fps=getattr(processor, "video_fps", 2.0),
                    video_maxlen=getattr(processor, "video_maxlen", 128),
                )
                mm_inputs.update(video_processor(videos=video_data["videos"], return_tensors="pd"))
            mm_inputs["video_second_per_grid"] = paddle.to_tensor(
                [video_processor.temporal_patch_size / fps[i] for i in range(len(fps))]
            )
        if len(audios) != 0:
            audios = self._regularize_audios(
                audios,
                sampling_rate=getattr(processor, "audio_sampling_rate", 16000),
            )["audios"]
            mm_inputs.update(
                feature_extractor(
                    audios,
                    sampling_rate=getattr(processor, "audio_sampling_rate", 16000),
                    return_attention_mask=True,
                    padding=False,
                    return_tensors="pd",
                )
            )
            mm_inputs["feature_attention_mask"] = mm_inputs.pop("attention_mask", None)

        # Convert floating point tensors to target dtype if specified
        target_dtype = kwargs.get("dtype", None)
        if target_dtype:
            mm_inputs = self._to_float_dtype(mm_inputs, target_dtype)
        else:
            logger.warning("Not specified dtype, use float32 by default.")
        return mm_inputs

    @staticmethod
    def _to_float_dtype(data: Any, dtype: str) -> Any:
        """Change the float inputs to a dtype (e.g., 'bfloat16').

        Args:
            data: Input data which can be a nested structure containing Paddle tensors.
            dtype: Target dtype string (e.g., 'bfloat16', 'float32', 'float16').

        Returns:
            Data with float tensors converted to the target dtype.
        """
        if paddle is None:
            return data

        if isinstance(data, dict):
            return {k: Qwen2OmniPlugin._to_float_dtype(v, dtype) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return type(data)(Qwen2OmniPlugin._to_float_dtype(v, dtype) for v in data)
        elif isinstance(data, paddle.Tensor):
            if data.dtype in [paddle.float32, paddle.float64, paddle.float16, paddle.bfloat16]:
                return paddle.cast(data, dtype)
        return data

    @override
    def process_messages(
        self,
        messages,
        images,
        videos,
        audios,
        mm_inputs,
        processor,
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens, num_video_tokens, num_audio_tokens = 0, 0, 0
        messages = deepcopy(messages)
        image_processor = getattr(processor, "image_processor")

        merge_length = getattr(image_processor, "merge_size") ** 2
        use_audio_in_video = getattr(processor, "use_audio_in_video", False)

        if self.expand_mm_tokens:
            image_grid_thw = mm_inputs.get("image_grid_thw", [])
            video_grid_thw = mm_inputs.get("video_grid_thw", [])
            if "feature_attention_mask" in mm_inputs:
                if processor.__class__.__name__ == "Qwen3OmniMoeProcessor":  # for qwen3omni
                    input_lengths = mm_inputs["feature_attention_mask"].sum(-1)
                    input_lengths_leave = input_lengths % 100
                    feature_lengths = (input_lengths_leave - 1) // 2 + 1
                    audio_lengths = ((feature_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
                else:
                    input_lengths = (mm_inputs["feature_attention_mask"].sum(-1).numpy() - 1) // 2 + 1
                    audio_lengths = (input_lengths - 2) // 2 + 1
        else:
            image_grid_thw = [None] * len(images)
            video_grid_thw = [None] * len(videos)
            audio_lengths = [None] * len(audios)

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
            if use_audio_in_video and len(audios) and len(videos):
                raise NotImplementedError
            else:
                while AUDIO_PLACEHOLDER in content:
                    audio_seqlen = audio_lengths[num_audio_tokens].prod().item() if self.expand_mm_tokens else 1
                    content = content.replace(
                        AUDIO_PLACEHOLDER,
                        f"{self.audio_bos_token}{self.audio_token * audio_seqlen}{self.audio_eos_token}",
                        1,
                    )
                    num_audio_tokens += 1

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
        **kwargs,
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
        **kwargs,
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
        processor,
        **kwargs,
    ):
        self._validate_input(processor, images, videos, audios)
        mm_inputs = self._get_mm_inputs(images, videos, audios, processor, **kwargs)
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
        processor,
        **kwargs,
    ):
        self._validate_input(processor, images, videos, audios)
        mm_inputs = self._get_mm_inputs(images, videos, audios, processor, **kwargs)
        mm_inputs.pop("num_crops", None)
        return mm_inputs


@dataclass
class GlmOcrPlugin(BasePlugin):
    """
    GLM-OCR 专用插件：
    - messages 里用 IMAGE_PLACEHOLDER(默认 <image>) 做占位符
    - 展开后插入：<|begin_of_image|> + N个<|image|> + <|end_of_image|>
    - N 来自 image_grid_thw.prod() // (merge_size**2)
    """

    # 这些 token 必须在 tokenizer special tokens 里存在
    image_bos_token: str = "<|begin_of_image|>"
    image_eos_token: str = "<|end_of_image|>"

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
        # 1) 基本校验：是否支持 image input、processor/image_processor 是否存在等
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)

        # 2) 取 image_processor / merge_length
        image_processor = getattr(processor, "image_processor", None)
        if image_processor is None:
            raise ValueError("image_processor was not found in processor.")

        merge_size = getattr(image_processor, "merge_size", None)
        if merge_size is None:
            raise ValueError("image_processor.merge_size was not found.")
        merge_length = int(merge_size) ** 2

        # 3) 取 image_grid_thw（expand_mm_tokens 时必须有）
        if self.expand_mm_tokens:
            image_grid_thw = mm_inputs.get("image_grid_thw", None)
            if image_grid_thw is None or len(image_grid_thw) == 0:
                raise ValueError(
                    "expand_mm_tokens=True but mm_inputs has no valid image_grid_thw. "
                    "Please ensure image_processor returns image_grid_thw."
                )
        else:
            # 不展开时，每张图就 1 个 token（不会用到 grid）
            image_grid_thw = None

        # 4) 展开：把每个 <image> 依次替换为 BOS + N*image_token + EOS
        # 关键点：IMAGE_PLACEHOLDER 必须 != self.image_token，否则会死循环
        if self.image_token is None:
            raise ValueError("GlmOcrPlugin requires image_token to be set (e.g., '<|image|>').")

        if IMAGE_PLACEHOLDER == self.image_token:
            raise ValueError(
                f"IMAGE_PLACEHOLDER ({IMAGE_PLACEHOLDER}) must be different from image_token ({self.image_token}). "
                "Otherwise placeholder replacement will not terminate."
            )

        num_image_tokens = 0
        messages = deepcopy(messages)

        for msg in messages:
            content = msg["content"]

            while IMAGE_PLACEHOLDER in content:
                # 越界保护（你现在遇到的 OutOfRange 就是这里本该被挡住）
                if num_image_tokens >= len(images):
                    raise ValueError(
                        f"Found more {IMAGE_PLACEHOLDER} placeholders than provided images: "
                        f"placeholders_so_far={num_image_tokens+1}, len(images)={len(images)}"
                    )

                if self.expand_mm_tokens:
                    # image_grid_thw shape: [num_images, 3]
                    # 每张图的 token 数 = prod(thw) // (merge_size**2)
                    seqlen = int(image_grid_thw[num_image_tokens].prod().item()) // merge_length
                    seqlen = max(1, seqlen)
                else:
                    seqlen = 1

                repl = f"{self.image_bos_token}{self.image_token * seqlen}{self.image_eos_token}"
                content = content.replace(IMAGE_PLACEHOLDER, repl, 1)
                num_image_tokens += 1

            msg["content"] = content
        # 5) mask：这些 token 不参与 loss（和你原先 PaddleOCRVLPlugin 一致）
        self.masked_tokens = [self.image_token, self.image_bos_token, self.image_eos_token]
        return messages


PLUGINS = {
    "base": BasePlugin,
    "ernie_vl": ErnieVLPlugin,
    "qwen2_vl": Qwen2VLPlugin,
    "paddleocr_vl": PaddleOCRVLPlugin,
    "qwen3_vl": Qwen3VLPlugin,
    "glm4v": GLM4VPlugin,
    "gemma3": Gemma3Plugin,
    "qwen2_omni": Qwen2OmniPlugin,
    "glm_ocr": GlmOcrPlugin,
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
