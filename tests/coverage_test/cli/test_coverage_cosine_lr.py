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

from paddleformers.cli.train.ernie_pretrain.src.lr_schedulers.cosine_lr import (
    get_cosine_schedule_with_warmup,
)


class TestCosineScheduleWithWarmup(unittest.TestCase):
    """Tests for get_cosine_schedule_with_warmup function."""

    def test_returns_lambda_decay(self):
        """Test that the function returns a LambdaDecay scheduler."""
        from paddle.optimizer.lr import LambdaDecay

        scheduler = get_cosine_schedule_with_warmup(
            learning_rate=1.0,
            num_warmup_steps=100,
            num_training_steps=1000,
        )
        self.assertIsInstance(scheduler, LambdaDecay)

    def test_warmup_start(self):
        """Test learning rate at the start of warmup."""
        scheduler = get_cosine_schedule_with_warmup(
            learning_rate=1.0,
            num_warmup_steps=100,
            num_training_steps=1000,
        )
        lr_0 = scheduler.lr_lambda(0)
        self.assertAlmostEqual(lr_0, 0.0, places=5)

    def test_warmup_mid(self):
        """Test learning rate during warmup."""
        scheduler = get_cosine_schedule_with_warmup(
            learning_rate=1.0,
            num_warmup_steps=100,
            num_training_steps=1000,
        )
        lr_50 = scheduler.lr_lambda(50)
        self.assertAlmostEqual(lr_50, 0.5, places=5)

    def test_warmup_end(self):
        """Test learning rate at end of warmup."""
        scheduler = get_cosine_schedule_with_warmup(
            learning_rate=1.0,
            num_warmup_steps=100,
            num_training_steps=1000,
        )
        lr_100 = scheduler.lr_lambda(100)
        self.assertAlmostEqual(lr_100, 1.0, places=5)

    def test_cosine_decay(self):
        """Test cosine decay phase."""
        scheduler = get_cosine_schedule_with_warmup(
            learning_rate=1.0,
            num_warmup_steps=0,
            num_training_steps=1000,
        )
        # At step 0, should be at peak
        lr_0 = scheduler.lr_lambda(0)
        self.assertAlmostEqual(lr_0, 1.0, places=5)

        # At step 500, should be at midpoint of cosine
        lr_500 = scheduler.lr_lambda(500)
        self.assertAlmostEqual(lr_500, 0.5, places=2)

        # At step 1000, should be at minimum
        lr_1000 = scheduler.lr_lambda(1000)
        self.assertAlmostEqual(lr_1000, 0.0, places=5)

    def test_cosine_decay_with_min_lr(self):
        """Test cosine decay with non-zero min_lr."""
        scheduler = get_cosine_schedule_with_warmup(
            learning_rate=1.0,
            num_warmup_steps=0,
            num_training_steps=1000,
            min_lr=0.1,
        )
        # At end, lr should approach min_lr/learning_rate
        lr_1000 = scheduler.lr_lambda(1000)
        self.assertAlmostEqual(lr_1000, 0.1, places=2)

    def test_num_cycles_default(self):
        """Test default num_cycles (0.5 = half cosine)."""
        scheduler = get_cosine_schedule_with_warmup(
            learning_rate=1.0,
            num_warmup_steps=0,
            num_training_steps=1000,
        )
        # With num_cycles=0.5, one complete half cosine
        lr_0 = scheduler.lr_lambda(0)
        self.assertAlmostEqual(lr_0, 1.0, places=5)

    def test_different_num_cycles(self):
        """Test with different num_cycles."""
        scheduler = get_cosine_schedule_with_warmup(
            learning_rate=1.0,
            num_warmup_steps=0,
            num_training_steps=1000,
            num_cycles=1.0,
        )
        # With num_cycles=1.0, full cosine wave
        lr_0 = scheduler.lr_lambda(0)
        self.assertAlmostEqual(lr_0, 1.0, places=5)

    def test_zero_warmup_steps(self):
        """Test with zero warmup steps."""
        scheduler = get_cosine_schedule_with_warmup(
            learning_rate=1.0,
            num_warmup_steps=0,
            num_training_steps=1000,
        )
        # Step 0 should be at peak (no warmup)
        lr_0 = scheduler.lr_lambda(0)
        self.assertAlmostEqual(lr_0, 1.0, places=5)

    def test_monotonic_decrease_after_warmup(self):
        """Test that lr monotonically decreases after warmup (with default cycles)."""
        scheduler = get_cosine_schedule_with_warmup(
            learning_rate=1.0,
            num_warmup_steps=100,
            num_training_steps=1000,
        )
        prev_lr = scheduler.lr_lambda(100)
        for step in range(200, 1001, 100):
            curr_lr = scheduler.lr_lambda(step)
            self.assertLessEqual(curr_lr, prev_lr + 1e-6)  # Allow tiny numerical errors
            prev_lr = curr_lr


if __name__ == "__main__":
    unittest.main()
