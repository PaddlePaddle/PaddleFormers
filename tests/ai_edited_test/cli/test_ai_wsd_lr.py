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

from paddleformers.cli.train.ernie_pretrain.src.lr_schedulers.wsd_lr import (
    get_wsd_schedule_with_warmup,
)


class TestWsdScheduleWithWarmup(unittest.TestCase):
    """Tests for get_wsd_schedule_with_warmup function."""

    def test_warmup_phase(self):
        """Test learning rate during warmup phase."""
        scheduler = get_wsd_schedule_with_warmup(
            learning_rate=1.0,
            num_warmup_steps=100,
            num_training_steps=1000,
        )
        # At step 0, lr should be 0
        lr_0 = scheduler.lr_lambda(0)
        self.assertAlmostEqual(lr_0, 0.0, places=5)

        # At step 50 (mid warmup), lr should be 0.5
        lr_50 = scheduler.lr_lambda(50)
        self.assertAlmostEqual(lr_50, 0.5, places=5)

        # At step 100 (end of warmup), lr should be 1.0
        lr_100 = scheduler.lr_lambda(100)
        self.assertAlmostEqual(lr_100, 1.0, places=5)

    def test_steady_phase(self):
        """Test learning rate during steady phase."""
        scheduler = get_wsd_schedule_with_warmup(
            learning_rate=1.0,
            num_warmup_steps=100,
            num_training_steps=1000,
            num_steady_steps=900,
        )
        # During steady phase, lr should be 1.0
        lr_200 = scheduler.lr_lambda(200)
        self.assertAlmostEqual(lr_200, 1.0, places=5)

        lr_800 = scheduler.lr_lambda(800)
        self.assertAlmostEqual(lr_800, 1.0, places=5)

    def test_decay_phase_half_life(self):
        """Test learning rate during decay phase with half_life function."""
        scheduler = get_wsd_schedule_with_warmup(
            learning_rate=1.0,
            num_warmup_steps=100,
            num_training_steps=1000,
            num_steady_steps=900,
            decay_function="half_life",
            min_lr=0.0,
        )
        # At the start of decay phase
        lr_900 = scheduler.lr_lambda(900)
        self.assertAlmostEqual(lr_900, 1.0, places=2)

        # At the end of training
        lr_1000 = scheduler.lr_lambda(1000)
        self.assertAlmostEqual(lr_1000, 0.0, places=2)

    def test_decay_phase_1_sqrt(self):
        """Test learning rate during decay phase with 1-sqrt function."""
        scheduler = get_wsd_schedule_with_warmup(
            learning_rate=1.0,
            num_warmup_steps=100,
            num_training_steps=1000,
            num_steady_steps=900,
            decay_function="1-sqrt",
            min_lr=0.0,
        )
        lr_900 = scheduler.lr_lambda(900)
        self.assertAlmostEqual(lr_900, 1.0, places=2)

    def test_invalid_decay_function(self):
        """Test that invalid decay function raises ValueError."""
        with self.assertRaises(ValueError):
            scheduler = get_wsd_schedule_with_warmup(
                learning_rate=1.0,
                num_warmup_steps=100,
                num_training_steps=1000,
                decay_function="invalid",
            )
            scheduler.lr_lambda(950)

    def test_default_num_steady_steps(self):
        """Test default num_steady_steps is 0.9 * num_training_steps."""
        scheduler = get_wsd_schedule_with_warmup(
            learning_rate=1.0,
            num_warmup_steps=0,
            num_training_steps=1000,
        )
        # Default steady steps = 900
        lr_800 = scheduler.lr_lambda(800)
        self.assertAlmostEqual(lr_800, 1.0, places=5)

    def test_with_min_lr(self):
        """Test with non-zero min_lr."""
        scheduler = get_wsd_schedule_with_warmup(
            learning_rate=1.0,
            num_warmup_steps=0,
            num_training_steps=1000,
            num_steady_steps=900,
            min_lr=0.1,
            decay_function="1-sqrt",
        )
        # At end of training, lr should approach min_lr
        lr_1000 = scheduler.lr_lambda(1000)
        self.assertAlmostEqual(lr_1000, 0.1, places=2)

    def test_returns_lambda_decay(self):
        """Test that the function returns a LambdaDecay scheduler."""
        from paddle.optimizer.lr import LambdaDecay

        scheduler = get_wsd_schedule_with_warmup(
            learning_rate=1.0,
            num_warmup_steps=100,
            num_training_steps=1000,
        )
        self.assertIsInstance(scheduler, LambdaDecay)

    def test_warmup_with_zero_warmup_steps(self):
        """Test with zero warmup steps."""
        scheduler = get_wsd_schedule_with_warmup(
            learning_rate=1.0,
            num_warmup_steps=0,
            num_training_steps=1000,
            num_steady_steps=900,
        )
        # Step 0 should be in steady phase
        lr_0 = scheduler.lr_lambda(0)
        self.assertAlmostEqual(lr_0, 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
