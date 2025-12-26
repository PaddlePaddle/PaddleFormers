# coding=utf-8
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
from __future__ import annotations

import unittest

import numpy as np
import paddle
from PIL import Image

from paddleformers.transformers import Qwen3VLVideoProcessor
from paddleformers.transformers.image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    get_image_size,
)
from paddleformers.transformers.qwen3_vl.video_processor import smart_resize

from ..test_video_processing_common import (
    VideoProcessingTestMixin,
    prepare_video_inputs,
)


class Qwen3VLVideoProcessingTester:
    """
    Helper class for testing Qwen3VLVideoProcessor.
    Defines configuration and expected behaviors.
    """

    def __init__(
        self,
        parent,
        batch_size=5,
        num_frames=8,
        num_channels=3,
        min_resolution=40,
        max_resolution=80,
        do_resize=True,
        size={"shortest_edge": 128 * 32 * 32, "longest_edge": 32 * 32 * 768},
        do_normalize=True,
        do_sample_frames=False,
        image_mean=OPENAI_CLIP_MEAN,
        image_std=OPENAI_CLIP_STD,
        do_convert_rgb=True,
        temporal_patch_size=2,
        patch_size=16,
        min_pixels=256 * 256,
        max_pixels=1280 * 1280,
        merge_size=2,
    ):
        size = size if size is not None else {"shortest_edge": 20}
        self.parent = parent
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.num_channels = num_channels
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size = size
        self.do_sample_frames = do_sample_frames
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_convert_rgb = do_convert_rgb
        self.temporal_patch_size = temporal_patch_size
        self.patch_size = patch_size
        self.min_pixels = size["shortest_edge"]
        self.max_pixels = size["longest_edge"]
        self.merge_size = merge_size

    def prepare_video_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_convert_rgb": self.do_convert_rgb,
            "temporal_patch_size": self.temporal_patch_size,
            "patch_size": self.patch_size,
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
            "merge_size": self.merge_size,
            "size": self.size,
            "do_sample_frames": self.do_sample_frames,
        }

    def expected_output_video_shape(self, videos, num_frames=None):
        num_frames = num_frames if num_frames is not None else self.num_frames
        grid_t = num_frames // self.temporal_patch_size
        hidden_dim = self.num_channels * self.temporal_patch_size * self.patch_size * self.patch_size
        seq_len = 0
        for video in videos:
            if isinstance(video[0], Image.Image):
                video = np.stack([np.array(frame) for frame in video])
            height, width = get_image_size(video)
            resized_height, resized_width = smart_resize(
                num_frames,
                height,
                width,
                self.temporal_patch_size,
                factor=self.patch_size * self.merge_size,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )
            grid_h, grid_w = resized_height // self.patch_size, resized_width // self.patch_size
            seq_len += grid_t * grid_h * grid_w
        return [seq_len, hidden_dim]

    def prepare_video_inputs(self, equal_resolution=False, return_tensors="pil"):
        videos = prepare_video_inputs(
            batch_size=self.batch_size,
            num_frames=self.num_frames,
            num_channels=self.num_channels,
            min_resolution=self.min_resolution,
            max_resolution=self.max_resolution,
            equal_resolution=equal_resolution,
            return_tensors=return_tensors,
        )
        return videos


class Qwen3VLVideoProcessingTest(VideoProcessingTestMixin, unittest.TestCase):
    fast_video_processing_class = Qwen3VLVideoProcessor

    def setUp(self):
        super().setUp()
        self.video_processor_tester = Qwen3VLVideoProcessingTester(self)

    @property
    def video_processor_dict(self):
        return self.video_processor_tester.prepare_video_processor_dict()

    def test_video_processor_properties(self):
        """
        Verifies that the processor instance has correct attributes set from config.
        """
        video_processing = self.fast_video_processing_class(**self.video_processor_dict)
        self.assertTrue(hasattr(video_processing, "do_resize"))
        self.assertTrue(hasattr(video_processing, "size"))
        self.assertTrue(hasattr(video_processing, "do_normalize"))
        self.assertTrue(hasattr(video_processing, "image_mean"))
        self.assertTrue(hasattr(video_processing, "image_std"))
        self.assertTrue(hasattr(video_processing, "do_convert_rgb"))

    def test_video_processor_from_dict_with_kwargs(self):
        """
        Tests initialization from dict with overrides.
        """
        for video_processing_class in self.video_processor_list:
            video_processor = video_processing_class(**self.video_processor_dict)
            self.assertEqual(video_processor.min_pixels, self.video_processor_tester.min_pixels)
            self.assertEqual(video_processor.max_pixels, self.video_processor_tester.max_pixels)

            video_processor = video_processing_class.from_dict(
                self.video_processor_dict, min_pixels=256 * 256, max_pixels=640 * 640
            )
            self.assertEqual(video_processor.min_pixels, 256 * 256)
            self.assertEqual(video_processor.max_pixels, 640 * 640)

    def test_call_paddle(self):
        """
        Tests processing Paddle tensor inputs.
        """
        for video_processing_class in self.video_processor_list:
            video_processing = video_processing_class(**self.video_processor_dict)
            video_inputs = self.video_processor_tester.prepare_video_inputs(
                equal_resolution=False, return_tensors="pd"
            )

            for video in video_inputs:
                self.assertIsInstance(video, paddle.Tensor)

            encoded_videos = video_processing(video_inputs[0], return_tensors="pd")[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape([video_inputs[0]])
            self.assertEqual(list(encoded_videos.shape), expected_output_video_shape)

            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape(video_inputs)
            encoded_videos = video_processing(video_inputs, return_tensors="pd")[self.input_name]
            self.assertEqual(
                list(encoded_videos.shape),
                expected_output_video_shape,
            )

    def test_call_numpy(self):
        """
        Tests processing Numpy array inputs.
        """
        for video_processing_class in self.video_processor_list:
            video_processing = video_processing_class(**self.video_processor_dict)
            video_inputs = self.video_processor_tester.prepare_video_inputs(
                equal_resolution=False, return_tensors="np"
            )
            for video in video_inputs:
                self.assertIsInstance(video, np.ndarray)

            encoded_videos = video_processing(video_inputs[0], return_tensors="pd")[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape([video_inputs[0]])
            self.assertEqual(list(encoded_videos.shape), expected_output_video_shape)

            encoded_videos = video_processing(video_inputs, return_tensors="pd")[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape(video_inputs)
            self.assertEqual(list(encoded_videos.shape), expected_output_video_shape)

    def test_nested_input(self):
        """
        Tests processing nested lists of inputs.
        """
        # NOTE: Temporarily skip CPU fallback cases. Remove this check after the issue is fixed.
        if not paddle.to_tensor([0]).place.is_gpu_place():
            self.skipTest("No GPU currently available/allocated")

        for video_processing_class in self.video_processor_list:
            video_processing = video_processing_class(**self.video_processor_dict)
            video_inputs = self.video_processor_tester.prepare_video_inputs(
                equal_resolution=False, return_tensors="np"
            )

            video_inputs_nested = [list(video) for video in video_inputs]
            encoded_videos = video_processing(video_inputs_nested[0], return_tensors="pd")[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape([video_inputs[0]])
            self.assertEqual(list(encoded_videos.shape), expected_output_video_shape)

    def test_call_sample_frames(self):
        """
        Tests frame sampling functionality.
        """
        # NOTE: Temporarily skip CPU fallback cases. Remove this check after the issue is fixed.
        if not paddle.to_tensor([0]).place.is_gpu_place():
            self.skipTest("No GPU currently available/allocated")

        for video_processing_class in self.video_processor_list:
            video_processing = video_processing_class(**self.video_processor_dict)

            self.video_processor_tester.num_frames = 8
            video_inputs = self.video_processor_tester.prepare_video_inputs(
                equal_resolution=False,
                return_tensors="pd",
            )

            # Case 1: do_sample_frames = False
            video_processing.do_sample_frames = False
            encoded_videos = video_processing(video_inputs[0], return_tensors="pd", fps=3)[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape([video_inputs[0]])
            self.assertListEqual(list(encoded_videos.shape), expected_output_video_shape)

            # Case 2: do_sample_frames = True
            video_processing.do_sample_frames = True
            encoded_videos = video_processing(video_inputs[0], return_tensors="pd", fps=4)[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape(
                [video_inputs[0]], num_frames=4
            )
            self.assertListEqual(list(encoded_videos.shape), expected_output_video_shape)

            # Case 3: FPS based sampling
            metadata = [[{"duration": 2.0, "total_num_frames": 8, "fps": 4}]]
            encoded_videos = video_processing(video_inputs[0], return_tensors="pd", fps=3, video_metadata=metadata)[
                self.input_name
            ]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape(
                [video_inputs[0]], num_frames=6
            )
            self.assertListEqual(list(encoded_videos.shape), expected_output_video_shape)

    @unittest.skip(
        "Qwen3VL VideoProcessor supports 3-channel RGB only. 4-channel inputs are not supported by the underlying smart_resize logic."
    )
    def test_call_numpy_4_channels(self):
        """
        Overriding the inherited test from VideoProcessingTestMixin to skip it.
        Qwen3VL is designed for RGB vision-language tasks and does not support RGBA or 4-channel tensors.
        """
        pass
