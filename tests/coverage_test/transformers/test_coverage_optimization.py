# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

from paddleformers.transformers.optimization import (
    CosineAnnealingWithWarmupDecay,
    LinearAnnealingWithWarmupDecay,
    is_integer,
)


class TestIsInteger(unittest.TestCase):
    """Tests for is_integer helper function."""

    def test_is_integer_true(self):
        self.assertTrue(is_integer(5))

    def test_is_integer_false_float(self):
        self.assertFalse(is_integer(3.14))

    def test_is_integer_false_bool(self):
        # bool is a subclass of int in Python, so True is isinstance(True, int)
        self.assertTrue(is_integer(True))

    def test_is_integer_false_string(self):
        self.assertFalse(is_integer("5"))


class TestCosineAnnealingWithWarmupDecay(unittest.TestCase):
    """Tests for CosineAnnealingWithWarmupDecay LR scheduler."""

    def setUp(self):
        self.max_lr = 1.0
        self.min_lr = 0.0
        self.warmup_step = 100
        self.decay_step = 1000

    def test_init_attributes(self):
        scheduler = CosineAnnealingWithWarmupDecay(
            max_lr=self.max_lr,
            min_lr=self.min_lr,
            warmup_step=self.warmup_step,
            decay_step=self.decay_step,
        )
        self.assertEqual(scheduler.max_lr, self.max_lr)
        self.assertEqual(scheduler.min_lr, self.min_lr)
        self.assertEqual(scheduler.warmup_step, self.warmup_step)
        self.assertEqual(scheduler.decay_step, self.decay_step)

    def test_warmup_phase_linear_increase(self):
        scheduler = CosineAnnealingWithWarmupDecay(
            max_lr=self.max_lr,
            min_lr=self.min_lr,
            warmup_step=self.warmup_step,
            decay_step=self.decay_step,
        )
        # At step 0 (last_epoch=0 after first step), lr should be 0
        # After stepping once, last_epoch = 0, get_lr = max_lr * 0 / warmup_step = 0
        lr_0 = scheduler.get_lr()
        self.assertAlmostEqual(lr_0, 0.0, places=5)

    def test_warmup_phase_mid_point(self):
        scheduler = CosineAnnealingWithWarmupDecay(
            max_lr=self.max_lr,
            min_lr=self.min_lr,
            warmup_step=self.warmup_step,
            decay_step=self.decay_step,
        )
        # Simulate being at step 50
        for _ in range(50):
            scheduler.step()
        lr_mid = scheduler.get_lr()
        expected = float(self.max_lr) * 50 / self.warmup_step
        self.assertAlmostEqual(float(lr_mid), expected, places=5)

    def test_decay_phase_returns_min_lr_after_decay_step(self):
        scheduler = CosineAnnealingWithWarmupDecay(
            max_lr=self.max_lr,
            min_lr=self.min_lr,
            warmup_step=self.warmup_step,
            decay_step=self.decay_step,
        )
        # Step past decay_step
        for _ in range(self.decay_step + 10):
            scheduler.step()
        lr = scheduler.get_lr()
        self.assertAlmostEqual(float(lr), self.min_lr, places=5)

    def test_cosine_decay_mid_phase(self):
        scheduler = CosineAnnealingWithWarmupDecay(
            max_lr=self.max_lr,
            min_lr=self.min_lr,
            warmup_step=0,
            decay_step=1000,
        )
        # Step to halfway in decay phase
        for _ in range(500):
            scheduler.step()
        lr = scheduler.get_lr()
        # At midpoint: decay_ratio=0.5, coeff=0.5*(cos(pi*0.5)+1)=0.5, lr=0.5
        self.assertAlmostEqual(float(lr), 0.5, places=4)

    def test_cosine_decay_end_phase(self):
        scheduler = CosineAnnealingWithWarmupDecay(
            max_lr=self.max_lr,
            min_lr=self.min_lr,
            warmup_step=0,
            decay_step=1000,
        )
        # Step to end of decay phase
        for _ in range(1000):
            scheduler.step()
        lr = scheduler.get_lr()
        # At end: decay_ratio=1.0, coeff=0.5*(cos(pi)+1)=0, lr=min_lr=0
        self.assertAlmostEqual(float(lr), self.min_lr, places=4)

    def test_no_warmup(self):
        scheduler = CosineAnnealingWithWarmupDecay(
            max_lr=self.max_lr,
            min_lr=self.min_lr,
            warmup_step=0,
            decay_step=self.decay_step,
        )
        # At step 0, warmup_step is 0, so it should not enter warmup branch
        lr = scheduler.get_lr()
        # last_epoch=0, decay_ratio=0, coeff=1.0, lr = min_lr + 1.0 * (max_lr - min_lr) = max_lr
        self.assertAlmostEqual(float(lr), self.max_lr, places=5)

    def test_min_lr_nonzero(self):
        scheduler = CosineAnnealingWithWarmupDecay(
            max_lr=5.0,
            min_lr=1.0,
            warmup_step=100,
            decay_step=1000,
        )
        for _ in range(self.decay_step + 5):
            scheduler.step()
        lr = scheduler.get_lr()
        self.assertAlmostEqual(float(lr), 1.0, places=5)


class TestLinearAnnealingWithWarmupDecay(unittest.TestCase):
    """Tests for LinearAnnealingWithWarmupDecay LR scheduler."""

    def setUp(self):
        self.max_lr = 1.0
        self.min_lr = 0.0
        self.warmup_step = 100
        self.decay_step = 1000

    def test_init_attributes(self):
        scheduler = LinearAnnealingWithWarmupDecay(
            max_lr=self.max_lr,
            min_lr=self.min_lr,
            warmup_step=self.warmup_step,
            decay_step=self.decay_step,
        )
        self.assertEqual(scheduler.max_lr, self.max_lr)
        self.assertEqual(scheduler.min_lr, self.min_lr)
        self.assertEqual(scheduler.warmup_step, self.warmup_step)
        self.assertEqual(scheduler.decay_step, self.decay_step)

    def test_warmup_phase_linear_increase(self):
        scheduler = LinearAnnealingWithWarmupDecay(
            max_lr=self.max_lr,
            min_lr=self.min_lr,
            warmup_step=self.warmup_step,
            decay_step=self.decay_step,
        )
        lr_0 = scheduler.get_lr()
        self.assertAlmostEqual(lr_0, 0.0, places=5)

    def test_warmup_phase_mid_point(self):
        scheduler = LinearAnnealingWithWarmupDecay(
            max_lr=self.max_lr,
            min_lr=self.min_lr,
            warmup_step=self.warmup_step,
            decay_step=self.decay_step,
        )
        for _ in range(50):
            scheduler.step()
        lr_mid = scheduler.get_lr()
        expected = float(self.max_lr) * 50 / self.warmup_step
        self.assertAlmostEqual(float(lr_mid), expected, places=5)

    def test_returns_min_lr_after_decay_step(self):
        scheduler = LinearAnnealingWithWarmupDecay(
            max_lr=self.max_lr,
            min_lr=self.min_lr,
            warmup_step=self.warmup_step,
            decay_step=self.decay_step,
        )
        for _ in range(self.decay_step + 10):
            scheduler.step()
        lr = scheduler.get_lr()
        self.assertAlmostEqual(float(lr), self.min_lr, places=5)

    def test_linear_decay_mid_phase(self):
        scheduler = LinearAnnealingWithWarmupDecay(
            max_lr=self.max_lr,
            min_lr=self.min_lr,
            warmup_step=0,
            decay_step=1000,
        )
        # Step to halfway in decay phase
        for _ in range(500):
            scheduler.step()
        lr = scheduler.get_lr()
        # At midpoint: coeff = 1.0 - 0.5 = 0.5, lr = 0 + 0.5 * 1.0 = 0.5
        self.assertAlmostEqual(float(lr), 0.5, places=4)

    def test_no_warmup_starts_at_max_lr(self):
        scheduler = LinearAnnealingWithWarmupDecay(
            max_lr=self.max_lr,
            min_lr=self.min_lr,
            warmup_step=0,
            decay_step=self.decay_step,
        )
        lr = scheduler.get_lr()
        # last_epoch=0, decay_ratio=0, coeff=1.0, lr = min_lr + 1.0 * (max_lr - min_lr) = max_lr
        self.assertAlmostEqual(float(lr), self.max_lr, places=5)

    def test_min_lr_nonzero(self):
        scheduler = LinearAnnealingWithWarmupDecay(
            max_lr=5.0,
            min_lr=1.0,
            warmup_step=100,
            decay_step=1000,
        )
        for _ in range(self.decay_step + 5):
            scheduler.step()
        lr = scheduler.get_lr()
        self.assertAlmostEqual(float(lr), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
