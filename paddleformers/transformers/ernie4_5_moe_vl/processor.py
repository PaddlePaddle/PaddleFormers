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
"""Processor class for ERNIE4.5-MOE-VL."""

import copy
import io
from collections import defaultdict
from typing import Any, Dict, List, Union

import numpy as np
from PIL import Image

from ..feature_extraction_utils import BatchFeature
from ..image_utils import ChannelDimension
from ..processing_utils import ProcessorMixin
from .vision_process import (
    RAW_IMAGE_DIR,
    get_downloadable,
    read_frames_decord,
    read_video_decord,
    render_frame_timestamp,
)

IDS_TYPE_FLAG = {"text": 0, "image": 1, "video": 2}


class Ernie4_5_VLProcessor(ProcessorMixin):
    """
    Processes multimodal chat messages into model-ready inputs,
    handling text, images, and videos with 3D positional embeddings.
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = [
        "chat_template",
        "spatial_conv_size",
        "temporal_conv_size",
        "image_min_pixels",
        "image_max_pixels",
        "video_min_pixels",
        "video_max_pixels",
        "video_target_frames",
        "video_frames_sample",
        "video_max_frames",
        "video_min_frames",
        "video_fps",
    ]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    CLS_TOKEN = "<|begin_of_sentence|>"
    SEP_TOKEN = "<|end_of_sentence|>"
    IMG_START = "<|IMAGE_START|>"
    IMG_END = "<|IMAGE_END|>"
    VID_START = "<|VIDEO_START|>"
    VID_END = "<|VIDEO_END|>"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        spatial_conv_size: int = 2,
        temporal_conv_size: int = 2,
        image_min_pixels: int = 4 * 28 * 28,
        image_max_pixels: int = 6177 * 28 * 28,
        video_min_pixels: int = 299 * 28 * 28,
        video_max_pixels: int = 1196 * 28 * 28,
        video_target_frames: int = -1,
        video_frames_sample: str = "middle",
        video_max_frames: int = 180,
        video_min_frames: int = 16,
        video_fps: int = 2,
        **kwargs,
    ):
        super().__init__(image_processor, tokenizer, chat_template=chat_template)
        self.tokenizer.ignored_index = -100

        # Convolution sizes for patch aggregation
        self.spatial_conv_size = spatial_conv_size
        self.temporal_conv_size = temporal_conv_size

        # Pixel constraints
        self.image_min_pixels = image_min_pixels
        self.image_max_pixels = image_max_pixels
        self.video_min_pixels = video_min_pixels
        self.video_max_pixels = video_max_pixels

        # Video sampling parameters
        self.target_frames = video_target_frames
        self.frames_sample = video_frames_sample
        self.max_frames = video_max_frames
        self.min_frames = video_min_frames
        self.fps = video_fps

        # Special tokens and IDs
        self.cls_token = self.CLS_TOKEN
        self.sep_token = self.SEP_TOKEN
        self.image_start = self.IMG_START
        self.image_end = self.IMG_END
        self.video_start = self.VID_START
        self.video_end = self.VID_END
        self.image_patch_id = self.tokenizer.convert_tokens_to_ids("<|IMAGE_PLACEHOLDER|>")

        self.token_type_mapping = self._build_token_type_mapping()
        self.is_training = True
        self.role_prefixes = {"system": "", "user": "User: ", "bot": "Assistant: "}

    def _build_token_type_mapping(self) -> Dict[Any, int]:
        mapping = defaultdict(lambda: IDS_TYPE_FLAG["text"])
        for token in (self.IMG_START, self.IMG_END, self.VID_START, self.VID_END):
            mapping[token] = IDS_TYPE_FLAG["image"]
        mapping[self.image_patch_id] = IDS_TYPE_FLAG["image"]
        return mapping

    def _download_image(
        self,
        item: Dict,
    ):
        """Download image from url and resize it to the specified size."""
        url_info = item.get("image_url", {})
        url = url_info.get("url")
        w = url_info.get("image_width", None)
        h = url_info.get("image_height", None)
        data = get_downloadable(url, download_dir=RAW_IMAGE_DIR, save_to_disk=False)

        img = Image.open(io.BytesIO(data) if isinstance(data, bytes) else data)
        if w and h:
            img = img.resize((w, h))
        return img

    def _download_video(self, item: Dict):
        """Download video from url and resize it to the specified size."""
        url_info = item.get("video_url", {})
        url = url_info.get("url")

        frames = self._load_and_process_video(url, item)

        pixel_stack = np.stack([np.array(f.convert("RGB")) for f in frames], axis=0)
        return pixel_stack

    def process_vision_info(self, messages: List[Dict[str, Any]]):
        """Preprocess messages into lists of text, images, and videos."""
        images = []
        videos = []

        for msg in messages:
            content_items = msg.get("content")
            if not isinstance(content_items, list):
                content_items = [content_items]

            for item in content_items:
                if item.get("type") == "image_url":
                    img = self._download_image(item)
                    images.append(img)
                elif item.get("type") == "video_url":
                    pixel_stack = self._download_video(item)
                    videos.append(pixel_stack)

        return images, videos

    def __call__(
        self,
        text: List[str] = None,
        images: List[Image.Image] = None,
        videos: List[List[Image.Image]] = None,
        **kwargs,
    ) -> Dict[str, Union[np.ndarray, List[np.ndarray], None]]:
        """
        Convert chat messages into model inputs.
        Returns a dict with input_ids, token_type_ids, position_ids, images, grid_thw, image_type_ids, labels.
        """
        outputs = {
            "input_ids": [],
            "token_type_ids": [],
            "position_ids": [],
            "images": [],
            "grid_thw": [],
            "image_type_ids": [],
            "cur_position": 0,
            "pic_cnt": 0,
            "video_cnt": 0,
        }
        if images is None:
            images = []
        if videos is None:
            videos = []
        if not isinstance(text, list):
            text = [text]

        texts = text[0]

        new_video_seg = True
        for text_with_image in texts.split(self.VID_START + "<|video@placeholder|>" + self.VID_END):
            new_text_seg = True
            if not new_video_seg:
                self._add_video(videos[outputs["video_cnt"]], outputs)
            for text in text_with_image.split(self.IMG_START + "<|image@placeholder|>" + self.IMG_END):
                if not new_text_seg:
                    self._add_image(images[outputs["pic_cnt"]], outputs)
                self._add_text(text, outputs)
                new_text_seg = False
            new_video_seg = False

        for key in ["cur_position", "pic_cnt", "video_cnt"]:
            outputs.pop(key, None)

        outputs = self._pack_outputs(outputs)
        for key in outputs.keys():
            if isinstance(outputs[key], np.ndarray):
                if key in ["images", "grid_thw"]:
                    outputs[key] = np.array(outputs[key])
                else:
                    outputs[key] = np.array([outputs[key]])

        return_tensors = kwargs.pop("return_tensors", None)

        return BatchFeature(data=outputs, tensor_type=return_tensors)

    def _add_special_token(self, token: Union[str, int], outputs: Dict) -> None:
        """add special token to outputs"""
        token_id = token if isinstance(token, int) else self.tokenizer.convert_tokens_to_ids(token)
        outputs["input_ids"].append(token_id)
        outputs["token_type_ids"].append(self.token_type_mapping[token])
        pos = outputs["cur_position"]
        outputs["position_ids"].append([pos] * 3)
        outputs["cur_position"] += 1

    def _add_text(self, text: str, outputs: Dict) -> None:
        """add text to outputs"""
        tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        outputs["input_ids"].extend(tokens)
        outputs["token_type_ids"].extend([IDS_TYPE_FLAG["text"]] * len(tokens))

        start = outputs["cur_position"]
        for i in range(len(tokens)):
            outputs["position_ids"].append([start + i] * 3)
        outputs["cur_position"] += len(tokens)

    def _add_image(self, img: Image.Image, outputs: Dict) -> None:
        """add image to outputs"""
        outputs["pic_cnt"] += 1
        self._add_special_token(self.IMG_START, outputs)

        patches_h, patches_w = self.image_processor.get_smarted_resize(
            img.height,
            img.width,
            min_pixels=self.image_min_pixels,
            max_pixels=self.image_max_pixels,
        )[1]
        num_tokens = (patches_h * patches_w) // (self.spatial_conv_size**2)

        outputs["input_ids"].extend([self.image_patch_id] * num_tokens)
        outputs["token_type_ids"].extend([IDS_TYPE_FLAG["image"]] * num_tokens)

        pos_ids = self._compute_3d_positions(1, patches_h, patches_w, outputs["cur_position"])
        outputs["position_ids"].extend(pos_ids)
        outputs["cur_position"] = np.max(pos_ids) + 1

        # Preprocess pixels
        ret = self.image_processor.preprocess(
            images=[img.convert("RGB")],
            do_normalize=False,
            do_rescale=False,
            predetermined_grid_thw=np.array([[patches_h, patches_w]]),
            do_convert_rgb=True,
            input_data_format=ChannelDimension.LAST,
        )
        outputs["images"].append(ret["pixel_values"])
        outputs["grid_thw"].append(ret["image_grid_thw"])
        outputs["image_type_ids"].append(0)

        self._add_special_token(self.IMG_END, outputs)

    def render_frame_timestamp(self, frame, timestamp, font_rate=0.1):
        return render_frame_timestamp(frame, timestamp, font_rate)

    def _add_video(self, pixel_stack, outputs: Dict) -> None:
        if not isinstance(pixel_stack, np.ndarray):
            pixel_stack = np.stack([np.array(frame.convert("RGB")) for frame in pixel_stack], axis=0)

        outputs["video_cnt"] += 1
        self._add_special_token(self.VID_START, outputs)

        patches_h, patches_w = self.image_processor.get_smarted_resize(
            pixel_stack.shape[1],
            pixel_stack.shape[2],
            min_pixels=self.video_min_pixels,
            max_pixels=self.video_max_pixels,
        )[1]
        num_frames = pixel_stack.shape[0]
        num_tokens = (num_frames * patches_h * patches_w) // (self.spatial_conv_size**2 * self.temporal_conv_size)

        ret = self.image_processor.preprocess(
            images=None,
            videos=pixel_stack,
            do_normalize=False,
            do_rescale=False,
            predetermined_grid_thw=np.array([[patches_h, patches_w]] * num_frames),
            do_convert_rgb=True,
            input_data_format=ChannelDimension.LAST,
        )
        outputs["images"].append(ret["pixel_values_videos"])
        outputs["grid_thw"].append(ret["video_grid_thw"])
        outputs["image_type_ids"].extend([1] * num_frames)

        outputs["input_ids"].extend([self.image_patch_id] * num_tokens)
        outputs["token_type_ids"].extend([IDS_TYPE_FLAG["video"]] * num_tokens)

        pos_ids = self._compute_3d_positions(num_frames, patches_h, patches_w, outputs["cur_position"])
        outputs["position_ids"].extend(pos_ids)
        outputs["cur_position"] = np.max(pos_ids) + 1

        self._add_special_token(self.VID_END, outputs)

    def _load_and_process_video(self, url: str, item: Dict) -> List[Image.Image]:
        reader, meta, path = read_video_decord(url, save_to_disk=False)

        video_frame_args = dict()
        video_frame_args["fps"] = item.get("fps", -1)
        video_frame_args["min_frames"] = item.get("min_frames", self.min_frames)
        video_frame_args["max_frames"] = item.get("max_frames", self.max_frames)
        video_frame_args["target_frames"] = item.get("target_frames", -1)
        video_frame_args["frames_sample"] = item.get("frames_sample", self.frames_sample)
        if video_frame_args["fps"] <= 0 and video_frame_args["target_frames"] <= 0:
            video_frame_args["fps"] = self.fps
            video_frame_args["target_frames"] = self.target_frames

        video_frame_args = self._set_video_frame_args(video_frame_args, meta)

        frames_data, timestamps = read_frames_decord(
            path,
            reader,
            meta,
            target_frames=video_frame_args["target_frames"],
            target_fps=video_frame_args["fps"],
            frames_sample=video_frame_args["frames_sample"],
        )

        frames: List[Image.Image] = []
        for img_array, ts in zip(frames_data, timestamps):
            frames.append(self.render_frame_timestamp(img_array, ts))
        # Ensure even number of frames for temporal conv
        if len(frames) % 2 != 0:
            frames.append(copy.deepcopy(frames[-1]))
        return frames

    def _set_video_frame_args(self, video_frame_args, video_meta):
        """
        Set the final frame extraction parameters based on known parameters and priorities
        """
        # Priority: video_target_frames > (video_min_frames, video_max_frames) > video_fps
        if video_frame_args["target_frames"] > 0:
            if video_frame_args["fps"] > 0:
                raise ValueError("fps must not be positive if target_frames is given")
            if (
                video_frame_args["min_frames"] > 0
                and video_frame_args["target_frames"] < video_frame_args["min_frames"]
            ):
                raise ValueError("target_frames must be larger than min_frames")
            if (
                video_frame_args["max_frames"] > 0
                and video_frame_args["target_frames"] > video_frame_args["max_frames"]
            ):
                raise ValueError("target_frames must be smaller than max_frames")
        else:
            if video_frame_args["fps"] <= 0:
                raise ValueError("Must provide either positive target_fps or positive target_frames.")
            frames_to_extract = int(video_meta["duration"] * video_frame_args["fps"])
            video_frame_args["target_frames"] = frames_to_extract
            video_frame_args["fps"] = -1

            if (
                video_frame_args["min_frames"] > 0
                and video_frame_args["max_frames"] > 0
                and video_frame_args["min_frames"] > video_frame_args["max_frames"]
            ):
                raise ValueError("min_frames must be smaller than max_frames")
            if video_frame_args["min_frames"] > 0 and frames_to_extract < video_frame_args["min_frames"]:
                video_frame_args["target_frames"] = video_frame_args["min_frames"]
            if video_frame_args["max_frames"] > 0 and frames_to_extract > video_frame_args["max_frames"]:
                video_frame_args["target_frames"] = video_frame_args["max_frames"]

        return video_frame_args

    def _compute_3d_positions(self, t: int, h: int, w: int, start_idx: int) -> List[List[int]]:
        # Downsample time if needed
        t_eff = t // self.temporal_conv_size if t != 1 else 1
        gh, gw = h // self.spatial_conv_size, w // self.spatial_conv_size
        time_idx = np.repeat(np.arange(t_eff), gh * gw)
        h_idx = np.tile(np.repeat(np.arange(gh), gw), t_eff)
        w_idx = np.tile(np.arange(gw), t_eff * gh)

        coords = list(zip(time_idx, h_idx, w_idx))
        return [[start_idx + ti, start_idx + hi, start_idx + wi] for ti, hi, wi in coords]

    def _pack_outputs(self, outs: Dict) -> Dict[str, Any]:
        # Stack or nullify image-related fields
        if not outs["images"]:
            outs["images"] = []
            outs["grid_thw"] = []
            outs["image_type_ids"] = []
        else:
            outs["images"] = np.vstack(outs["images"])
            outs["grid_thw"] = np.vstack(outs["grid_thw"])
            outs["image_type_ids"] = np.array(outs["image_type_ids"])

        # Convert lists to arrays
        outs["input_ids"] = np.array(outs["input_ids"], dtype=np.int64)
        outs["token_type_ids"] = np.array(outs["token_type_ids"], dtype=np.int64)
        outs["position_ids"] = np.array(outs["position_ids"], dtype=np.int64)
        return outs

    @property
    def model_input_names(self):
        """get model input names"""
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = ["images", "grid_thw", "image_type_ids", "token_type_ids"]
        return list(tokenizer_input_names) + list(image_processor_input_names)


__all__ = ["Ernie4_5_VLProcessor"]
