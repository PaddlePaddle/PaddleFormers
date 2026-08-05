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

import unittest
from types import SimpleNamespace

from paddleformers.cli.train.sft.workflow import _apply_preprocess_args_to_processor


class TestSFTPreprocessArguments(unittest.TestCase):
    def test_multimodal_preprocess_args_are_applied(self):
        processor = SimpleNamespace(
            image_processor=SimpleNamespace(size={}, min_pixels=None, max_pixels=None),
            video_processor=SimpleNamespace(size={}),
        )
        preprocess_args = SimpleNamespace(
            max_pixels=4096,
            min_pixels=4096,
            video_max_pixels=4096,
            video_min_pixels=4096,
            video_fps=2,
            video_max_frames=2,
            video_target_frames=2,
        )

        _apply_preprocess_args_to_processor(processor, preprocess_args)

        self.assertEqual(processor.image_max_pixels, 4096)
        self.assertEqual(processor.image_min_pixels, 4096)
        self.assertEqual(processor.video_max_pixels, 4096)
        self.assertEqual(processor.video_min_pixels, 4096)
        self.assertEqual(processor.video_fps, 2)
        self.assertEqual(processor.video_maxlen, 2)
        self.assertEqual(processor.video_target_frames, 2)
        self.assertEqual(
            processor.image_processor.size,
            {"shortest_edge": 4096, "longest_edge": 4096},
        )
        self.assertEqual(
            processor.video_processor.size,
            {"shortest_edge": 8192, "longest_edge": 8192},
        )
        self.assertEqual(processor.image_processor.min_pixels, 4096)
        self.assertEqual(processor.image_processor.max_pixels, 4096)
        self.assertFalse(hasattr(processor, "accuracy_compatible_preprocessing"))

    def test_accuracy_compatible_preprocessing_contract_is_explicit(self):
        processor = SimpleNamespace(
            image_processor=SimpleNamespace(size={}, min_pixels=None, max_pixels=None, merge_size=2),
            video_processor=SimpleNamespace(size={}),
        )
        preprocess_args = SimpleNamespace(
            max_pixels=4096,
            min_pixels=4096,
            video_max_pixels=4096,
            video_min_pixels=4096,
            video_fps=2,
            video_max_frames=2,
            video_target_frames=2,
        )

        _apply_preprocess_args_to_processor(processor, preprocess_args, accuracy_compatible=True)

        self.assertTrue(processor.accuracy_compatible_preprocessing)
        self.assertTrue(processor.image_processor.accuracy_compatible_rescale_normalize)
        self.assertEqual(processor.video_frame_min_pixels, 16384)
        self.assertEqual(processor.video_frame_max_pixels, 16384)
        self.assertEqual(
            processor.video_processor.size,
            {"shortest_edge": 8192, "longest_edge": 8192},
        )


if __name__ == "__main__":
    unittest.main()
