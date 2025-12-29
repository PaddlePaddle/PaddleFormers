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
"""Tests for Qwen2/2.5-VL vision processing functions."""
from __future__ import annotations

import math
import os
import unittest

import paddle
from PIL import Image

from paddleformers.transformers.qwen2_vl import vision_process


class TestQwenVisionProcessing(unittest.TestCase):
    """Test cases for Qwen vision processing functions."""

    def setUp(self):
        """Set up test fixtures."""
        # Create test image
        self.test_image = Image.new("RGB", (100, 100), color="red")

        # Create test image url
        self.test_image_url = "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example1.jpg"

        # Create test video frames
        self.test_frames = [Image.new("RGB", (64, 64), color=(i * 10, i * 20, i * 30)) for i in range(10)]

        # Create test video url
        self.test_video_url = "http://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_video/example_video.mp4"

        os.environ["MODEL_SEQ_LEN"] = "128000"

    def tearDown(self):
        """Clean up after tests."""
        if "MODEL_SEQ_LEN" in os.environ and os.environ["MODEL_SEQ_LEN"] == "128000":
            del os.environ["MODEL_SEQ_LEN"]

    def test_round_by_factor(self):
        """Test round_by_factor function."""
        self.assertEqual(vision_process.round_by_factor(15, 5), 15)
        self.assertEqual(vision_process.round_by_factor(13, 5), 15)
        self.assertEqual(vision_process.round_by_factor(12, 5), 10)
        self.assertEqual(vision_process.round_by_factor(0, 5), 0)

    def test_ceil_by_factor(self):
        """Test ceil_by_factor function."""
        self.assertEqual(vision_process.ceil_by_factor(15, 5), 15)
        self.assertEqual(vision_process.ceil_by_factor(13, 5), 15)
        self.assertEqual(vision_process.ceil_by_factor(12, 5), 15)
        self.assertEqual(vision_process.ceil_by_factor(0, 5), 0)

    def test_floor_by_factor(self):
        """Test floor_by_factor function."""
        self.assertEqual(vision_process.floor_by_factor(15, 5), 15)
        self.assertEqual(vision_process.floor_by_factor(13, 5), 10)
        self.assertEqual(vision_process.floor_by_factor(12, 5), 10)
        self.assertEqual(vision_process.floor_by_factor(0, 5), 0)

    def test_smart_resize_basic(self):
        """Test smart_resize function with basic inputs."""
        height, width = 100, 200
        factor = 14

        h_bar, w_bar = vision_process.smart_resize(height, width, factor)

        # Check divisibility by factor
        self.assertEqual(h_bar % factor, 0)
        self.assertEqual(w_bar % factor, 0)

        # Check aspect ratio preservation
        original_ratio = width / height
        resized_ratio = w_bar / h_bar
        self.assertAlmostEqual(original_ratio, resized_ratio, delta=0.5)

    def test_smart_resize_with_constraints(self):
        """Test smart_resize with min_pixels and max_pixels constraints."""
        height, width = 1000, 2000
        factor = 14
        min_pixels = 10000
        max_pixels = 50000

        h_bar, w_bar = vision_process.smart_resize(height, width, factor, min_pixels, max_pixels)

        # Check divisibility
        self.assertEqual(h_bar % factor, 0)
        self.assertEqual(w_bar % factor, 0)

        # Check pixel constraints
        total_pixels = h_bar * w_bar
        self.assertGreaterEqual(total_pixels, min_pixels)
        self.assertLessEqual(total_pixels, max_pixels)

    def test_smart_resize_aspect_ratio_validation(self):
        """Test smart_resize aspect ratio validation."""
        # Test valid aspect ratio
        height, width = 100, 2000  # Ratio 20:1
        factor = 14

        # This should work fine
        h_bar, w_bar = vision_process.smart_resize(height, width, factor)
        self.assertGreater(h_bar, 0)
        self.assertGreater(w_bar, 0)

        # Test invalid aspect ratio (should raise ValueError)
        height, width = 10, 3000  # Ratio 300:1 > MAX_RATIO(200)

        with self.assertRaises(ValueError):
            vision_process.smart_resize(height, width, factor)

    def test_to_rgb_conversion(self):
        """Test to_rgb function."""
        # Test RGBA conversion
        rgba_image = Image.new("RGBA", (10, 10), (255, 0, 0, 128))
        rgb_image = vision_process.to_rgb(rgba_image)
        self.assertEqual(rgb_image.mode, "RGB")

        # Test RGB passthrough
        rgb_input = Image.new("RGB", (10, 10), "blue")
        rgb_output = vision_process.to_rgb(rgb_input)
        self.assertEqual(rgb_output.mode, "RGB")

    def test_smart_nframes_basic(self):
        """Test smart_nframes function with basic inputs."""
        ele = {"nframes": 20}
        total_frames = 100
        video_fps = 30

        nframes = vision_process.smart_nframes(ele, total_frames, video_fps)
        self.assertEqual(nframes, 20)

    def test_smart_nframes_fps_calculation(self):
        """Test smart_nframes with fps calculation."""
        ele = {"fps": 10}
        total_frames = 60
        video_fps = 30

        nframes = vision_process.smart_nframes(ele, total_frames, video_fps)
        expected_nframes = 20  # (60/30)*10 = 20
        self.assertEqual(nframes, expected_nframes)

    def test_smart_nframes_invalid_inputs(self):
        """Test smart_nframes with invalid inputs."""
        # Test nframes below FRAME_FACTOR
        ele = {"nframes": 1}
        total_frames = 100
        video_fps = 30

        with self.assertRaises(ValueError):
            vision_process.smart_nframes(ele, total_frames, video_fps)

        # Test nframes above total_frames
        ele = {"nframes": 200}
        total_frames = 100

        with self.assertRaises(ValueError):
            vision_process.smart_nframes(ele, total_frames, video_fps)

    def test_fetch_image_from_url(self):
        """Test fetch_image function with HTTP URL."""
        ele = {"image_url": self.test_image_url}
        result = vision_process.fetch_image(ele)

        self.assertIsInstance(result, Image.Image)

    def test_fetch_image_from_pil_object(self):
        """Test fetch_image function with PIL Image object."""
        ele = {"image": self.test_image}
        result = vision_process.fetch_image(ele)

        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, (112, 112))

    def test_fetch_image_with_resized_dimensions(self):
        """Test fetch_image with pre-specified resized dimensions."""
        ele = {"image": self.test_image, "resized_height": 50, "resized_width": 50}
        result = vision_process.fetch_image(ele)

        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, (56, 56))

    def test_calculate_video_frame_range_full_video(self):
        """Test calculate_video_frame_range with full video."""
        ele = {}
        total_frames = 100
        video_fps = 30

        start_frame, end_frame, frame_count = vision_process.calculate_video_frame_range(ele, total_frames, video_fps)

        self.assertEqual(start_frame, 0)
        self.assertEqual(end_frame, 99)
        self.assertEqual(frame_count, 100)

    def test_calculate_video_frame_range_with_time_range(self):
        """Test calculate_video_frame_range with time range."""
        ele = {"video_start": 1.0, "video_end": 3.0}
        total_frames = 100
        video_fps = 30

        start_frame, end_frame, frame_count = vision_process.calculate_video_frame_range(ele, total_frames, video_fps)

        expected_start = math.ceil(1.0 * 30)  # 30
        expected_end = math.floor(3.0 * 30)  # 90
        self.assertEqual(start_frame, expected_start)
        self.assertEqual(end_frame, expected_end)
        self.assertEqual(frame_count, expected_end - expected_start + 1)

    def test_extract_vision_info_single_conversation(self):
        """Test extract_vision_info with single conversation."""
        conversations = [
            {"content": [{"type": "text", "value": "Hello"}, {"image": "path/to/image"}, {"video": "path/to/video"}]}
        ]

        vision_infos = vision_process.extract_vision_info(conversations)

        self.assertEqual(len(vision_infos), 2)
        self.assertIn("image", vision_infos[0])
        self.assertIn("video", vision_infos[1])

    def test_extract_vision_info_multiple_conversations(self):
        """Test extract_vision_info with multiple conversations."""
        conversations = [
            [{"content": [{"image": "path/to/image"}]}, {"content": [{"video": "path/to/video"}]}],
            [{"content": [{"image_url": "path/to/image_url"}]}],
        ]

        vision_infos = vision_process.extract_vision_info(conversations)

        self.assertEqual(len(vision_infos), 3)

    def test_fetch_video_with_decord(self):
        """Test fetch_video(default with frame list) function using decord backend."""
        # NOTE: Temporarily skip CPU fallback cases. Remove this check after the issue is fixed.
        if not paddle.to_tensor([0]).place.is_gpu_place():
            self.skipTest("No GPU currently available/allocated")

        ele = {"video": self.test_video_url}
        result = vision_process.fetch_video(ele, video_backend="decord")

        self.assertIsInstance(result, paddle.Tensor)

    def test_fetch_video_with_paddlecodec(self):
        """Test fetch_video(default with frame list) function using paddlecodec backend."""
        # NOTE: Temporarily skip CPU fallback cases. Remove this check after the issue is fixed.
        if not paddle.to_tensor([0]).place.is_gpu_place():
            self.skipTest("No GPU currently available/allocated")

        ele = {"video": self.test_video_url}
        result = vision_process.fetch_video(ele, video_backend="paddlecodec")

        import torchcodec

        if not getattr(torchcodec, "__is_paddle_compatible_library__", None):
            raise RuntimeError("Could not import 'torchcodec'. Please ensure it is installed.")

        self.assertIsInstance(result, paddle.Tensor)

    def test_fetch_video_with_frame_list(self):
        """Test fetch_video function with frame list."""
        # NOTE: Temporarily skip CPU fallback cases. Remove this check after the issue is fixed.
        if not paddle.to_tensor([0]).place.is_gpu_place():
            self.skipTest("No GPU currently available/allocated")

        ele = {"video": self.test_frames, "resized_height": 64, "resized_width": 64}
        result = vision_process.fetch_video(ele)

        self.assertIsInstance(result, paddle.Tensor)
        self.assertEqual(result.shape[0], 10)  # Number of frames

    def test_process_vision_info_images_only(self):
        """Test process_vision_info with images only."""
        conversations = [
            [
                {
                    "content": [
                        {"image": self.test_image},
                        {"image_url": self.test_image_url},
                    ]
                }
            ]
        ]

        result = vision_process.process_vision_info(conversations, return_video_kwargs=True)

        if len(result) == 3:
            image_inputs, video_inputs, video_kwargs = result
        elif len(result) == 2:
            image_inputs, video_inputs = result
            video_kwargs = None

        self.assertIsNotNone(image_inputs)
        self.assertIsNone(video_inputs)
        self.assertEqual(len(image_inputs), 2)
        if video_kwargs:
            self.assertIn("fps", video_kwargs)

    def test_process_vision_info_videos_only(self):
        """Test process_vision_info with videos only."""
        # NOTE: Temporarily skip CPU fallback cases. Remove this check after the issue is fixed.
        if not paddle.to_tensor([0]).place.is_gpu_place():
            self.skipTest("No GPU currently available/allocated")

        conversations = [[{"content": [{"video": self.test_frames}]}]]

        result = vision_process.process_vision_info(conversations)

        if len(result) == 3:
            image_inputs, video_inputs, _ = result
        elif len(result) == 2:
            image_inputs, video_inputs = result

        self.assertIsNone(image_inputs)
        self.assertIsNotNone(video_inputs)

    def test_process_vision_info_empty(self):
        """Test processing of empty conversation."""
        conversations = [[]]

        result = vision_process.process_vision_info(conversations)

        if len(result) == 3:
            image_inputs, video_inputs, video_kwargs = result
        elif len(result) == 2:
            image_inputs, video_inputs = result

        self.assertIsNone(image_inputs)
        self.assertIsNone(video_inputs)

    def test_process_vision_info_mixed_content(self):
        """Test process_vision_info with mixed image and video content."""
        # NOTE: Temporarily skip CPU fallback cases. Remove this check after the issue is fixed.
        if not paddle.to_tensor([0]).place.is_gpu_place():
            self.skipTest("No GPU currently available/allocated")

        conversations = [[{"content": [{"image": self.test_image}, {"video": self.test_frames}]}]]

        result = vision_process.process_vision_info(conversations)

        if len(result) == 3:
            image_inputs, video_inputs, _ = result
        elif len(result) == 2:
            image_inputs, video_inputs = result

        self.assertIsNotNone(image_inputs)
        self.assertIsNotNone(video_inputs)
        self.assertEqual(len(image_inputs), 1)
        self.assertEqual(len(video_inputs), 1)
