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
from __future__ import annotations

import unittest

from decord import VideoReader
from PIL.Image import Image

from paddleformers.transformers.ernie4_5_moe_vl.vision_process import (
    get_downloadable_image,
    read_frames_decord,
    read_video_decord,
    render_frame_timestamp,
)


class Ernie4_5_VLVisionProcessTest(unittest.TestCase):
    def test_get_downloadable_image(self):
        img_url = "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example1.jpg"

        image, exif_info = get_downloadable_image(img_url, need_exif_info=False)
        self.assertIsInstance(image, Image)
        self.assertEqual(exif_info, {})

        image, exif_info = get_downloadable_image(img_url, need_exif_info=True)
        self.assertEqual(exif_info, {})

    def test_video(self):
        video_url = "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_video/example_video.mp4"
        video_reader, video_meta, video_path = read_video_decord(video_url, save_to_disk=False)

        expect_video_meta = {"fps": 29.99418363739351, "duration": 8.768366666666667, "num_of_frame": 263}
        self.assertIsInstance(video_reader, VideoReader)
        self.assertEqual(video_meta, expect_video_meta)

        with self.assertRaises(ValueError):
            ret, time_stamps = read_frames_decord(video_path, video_reader, video_meta)

        with self.assertRaises(AssertionError):
            ret, time_stamps = read_frames_decord(video_path, video_reader, video_meta, target_frames=10, target_fps=1)

        ret, time_stamps = read_frames_decord(video_path, video_reader, video_meta, target_frames=10)

        self.assertEqual(len(ret), 10)
        self.assertEqual(len(time_stamps), 10)

        ret, time_stamps = read_frames_decord(video_path, video_reader, video_meta, target_fps=1)

        self.assertEqual(len(ret), 9)
        self.assertEqual(len(time_stamps), 9)

        new_frame = render_frame_timestamp(ret[0], time_stamps[0])
        self.assertIsInstance(new_frame, Image)
