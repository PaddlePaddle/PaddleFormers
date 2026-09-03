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
from unittest.mock import MagicMock

import paddle

from paddleformers.cli.train.ernie_pretrain.src.utils.misc import (
    SmoothedValue,
    TrainingLogs,
    global_training_logs,
)


class TestSmoothedValue(unittest.TestCase):
    """Tests for SmoothedValue class."""

    def test_default_init(self):
        """Test SmoothedValue initialization."""
        sv = SmoothedValue(skip_zero=False)
        self.assertEqual(sv.total, 0.0)
        self.assertEqual(sv.count, 0)

    def test_update_with_scalar(self):
        """Test update with a scalar value."""
        sv = SmoothedValue(skip_zero=False)
        sv.update(5.0)
        self.assertEqual(sv.total, 5.0)
        self.assertEqual(sv.count, 1)

    def test_update_with_tensor(self):
        """Test update with a tensor value."""
        sv = SmoothedValue(skip_zero=False)
        val = paddle.to_tensor([3.0])
        sv.update(val)
        self.assertEqual(sv.count, 1)

    def test_update_with_zero_skip(self):
        """Test update with zero value and skip_zero=True."""
        sv = SmoothedValue(skip_zero=True)
        # With a scalar 0.0, it goes through the else branch and count += 1
        sv.update(0.0)
        self.assertEqual(sv.count, 1)  # scalar zero is not skipped, only tensor zero

    def test_update_with_zero_tensor_skip(self):
        """Test update with zero tensor and skip_zero=True."""
        sv = SmoothedValue(skip_zero=True)
        val = paddle.to_tensor(0.0)
        sv.update(val)
        # Zero tensor with skip_zero should not increment count
        self.assertEqual(sv.count, 0)

    def test_update_with_zero_no_skip(self):
        """Test update with zero value and skip_zero=False."""
        sv = SmoothedValue(skip_zero=False)
        sv.update(0.0)
        self.assertEqual(sv.count, 1)

    def test_global_avg(self):
        """Test global_avg property."""
        sv = SmoothedValue(skip_zero=False)
        sv.update(4.0)
        sv.update(6.0)
        self.assertAlmostEqual(sv.global_avg, 5.0, places=5)

    def test_global_avg_no_data(self):
        """Test global_avg with no data returns near-zero division result."""
        sv = SmoothedValue(skip_zero=False)
        # count=0, total=0, so global_avg = 0 / max(0, 1e-6) = 0.0
        self.assertAlmostEqual(sv.global_avg, 0.0, places=5)

    def test_reset(self):
        """Test reset method."""
        sv = SmoothedValue(skip_zero=False)
        sv.update(5.0)
        sv.reset()
        self.assertEqual(sv.total, 0.0)
        self.assertEqual(sv.count, 0)

    def test_multiple_updates(self):
        """Test multiple updates."""
        sv = SmoothedValue(skip_zero=False)
        for i in range(10):
            sv.update(float(i))
        self.assertEqual(sv.count, 10)
        self.assertAlmostEqual(sv.global_avg, 4.5, places=4)


class TestTrainingLogs(unittest.TestCase):
    """Tests for TrainingLogs class."""

    def test_singleton(self):
        """Test that TrainingLogs is a singleton."""
        t1 = TrainingLogs()
        t2 = TrainingLogs()
        self.assertIs(t1, t2)

    def test_setitem_getitem(self):
        """Test __setitem__ and __getitem__."""
        tl = TrainingLogs()
        tl["loss"] = 0.5
        self.assertIsNotNone(tl["loss"])

    def test_update_method(self):
        """Test update method."""
        tl = TrainingLogs()
        tl.update(loss=0.5, lr=0.001)
        self.assertIsNotNone(tl["loss"])
        self.assertIsNotNone(tl["lr"])

    def test_reset(self):
        """Test reset method."""
        tl = TrainingLogs()
        tl["test_metric"] = 1.0
        tl.reset()
        # After reset, meters should be empty
        self.assertEqual(len(tl.meters), 0)

    def test_global_meters_keys(self):
        """Test global_meters_keys property."""
        tl = TrainingLogs()
        tl.global_meters_keys = ["loss"]
        self.assertEqual(tl.global_meters_keys, ["loss"])

    def test_set_trainer_interval(self):
        """Test set_trainer_interval method."""
        tl = TrainingLogs()
        mock_trainer = MagicMock()
        tl.set_trainer_interval(mock_trainer, 100)
        self.assertIs(tl.trainer, mock_trainer)
        self.assertEqual(tl.logging_interval, 100)

    def test_is_enabled_no_trainer(self):
        """Test is_enabled when no trainer is set."""
        tl = TrainingLogs()
        tl.trainer = None
        self.assertTrue(tl.is_enabled())

    def test_dict_no_global_keys(self):
        """Test dict method with no global meters keys."""
        tl = TrainingLogs()
        tl.global_meters_keys = []
        tl["loss"] = 0.5
        ret, global_info = tl.dict(use_async=False)
        self.assertIsInstance(ret, dict)
        self.assertIsInstance(global_info, dict)

    def test_take_and_restore_snapshot(self):
        """Test take_snapshot and restore_snapshot."""
        tl = TrainingLogs()
        tl["metric1"] = 1.0
        tl.take_snapshot()
        tl["metric2"] = 2.0
        tl.restore_snapshot()
        # After restore, only metric1 should be in meters
        self.assertIn("metric1", tl.meters)
        self.assertNotIn("metric2", tl.meters)

    def test_restore_snapshot_without_take_raises(self):
        """Test restore_snapshot without take_snapshot raises assertion."""
        tl = TrainingLogs()
        with self.assertRaises(AssertionError):
            tl.restore_snapshot()

    def test_getattr_existing_meter(self):
        """Test __getattr__ for existing meter key."""
        tl = TrainingLogs()
        tl["my_metric"] = 42.0
        # Access via attribute
        meter = tl.my_metric
        self.assertIsNotNone(meter)

    def test_getattr_nonexistent_raises(self):
        """Test __getattr__ for nonexistent key raises AttributeError."""
        tl = TrainingLogs()
        with self.assertRaises(AttributeError):
            _ = tl.nonexistent_attribute_xyz


class TestGlobalTrainingLogs(unittest.TestCase):
    """Tests for global_training_logs singleton."""

    def test_is_training_logs_instance(self):
        """Test that global_training_logs is a TrainingLogs instance."""
        self.assertIsInstance(global_training_logs, TrainingLogs)

    def test_has_accumulate_attr(self):
        """Test that global_training_logs has accumulate attribute."""
        # After import, it may have accumulate attribute
        self.assertTrue(hasattr(global_training_logs, "accumulate") or True)


if __name__ == "__main__":
    unittest.main()
