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

from paddleformers.cli.train.ernie_pretrain.src.utils.training_utils import (
    reset_per_device_batch_size,
)


class TestResetPerDeviceBatchSize(unittest.TestCase):
    """Tests for reset_per_device_batch_size function."""

    def test_batch_per_device_less_than_per_device_train_batch_size(self):
        """Test when batch_per_device < per_device_train_batch_size."""
        # global_batch_size=16, world_size=8 => batch_per_device=2
        # per_device_train_batch_size=4 => 2 < 4
        per_device_bsz, grad_accum = reset_per_device_batch_size(16, 4, 8)
        self.assertEqual(per_device_bsz, 2)
        self.assertEqual(grad_accum, 1)

    def test_batch_per_device_equals_per_device_train_batch_size(self):
        """Test when batch_per_device == per_device_train_batch_size."""
        # global_batch_size=32, world_size=8 => batch_per_device=4
        # per_device_train_batch_size=4 => 4 == 4
        per_device_bsz, grad_accum = reset_per_device_batch_size(32, 4, 8)
        self.assertEqual(per_device_bsz, 4)
        self.assertEqual(grad_accum, 1)

    def test_batch_per_device_greater_than_per_device_train_batch_size(self):
        """Test when batch_per_device > per_device_train_batch_size."""
        # global_batch_size=64, world_size=8 => batch_per_device=8
        # per_device_train_batch_size=4 => 8 / 4 = 2
        per_device_bsz, grad_accum = reset_per_device_batch_size(64, 4, 8)
        self.assertEqual(per_device_bsz, 4)
        self.assertEqual(grad_accum, 2)

    def test_not_evenly_divided_raises(self):
        """Test that global_batch_size not evenly divided by world_size raises."""
        with self.assertRaises(AssertionError):
            reset_per_device_batch_size(17, 4, 8)

    def test_batch_per_device_not_divisible_by_per_device_raises(self):
        """Test that batch_per_device not divisible by per_device_train_batch_size raises."""
        # global_batch_size=40, world_size=8 => batch_per_device=5
        # per_device_train_batch_size=4 => 5 % 4 != 0
        with self.assertRaises(AssertionError):
            reset_per_device_batch_size(40, 4, 8)

    def test_single_device(self):
        """Test with single device (world_size=1)."""
        per_device_bsz, grad_accum = reset_per_device_batch_size(32, 8, 1)
        self.assertEqual(per_device_bsz, 8)
        self.assertEqual(grad_accum, 4)

    def test_large_gradient_accumulation(self):
        """Test with large gradient accumulation."""
        per_device_bsz, grad_accum = reset_per_device_batch_size(256, 4, 4)
        # batch_per_device = 256 / 4 = 64
        # gradient_accumulation_steps = 64 / 4 = 16
        self.assertEqual(per_device_bsz, 4)
        self.assertEqual(grad_accum, 16)


if __name__ == "__main__":
    unittest.main()
