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

from paddleformers.fleet.transformer.moe.moe_utils import AutoSBHistoryTracker


class TestAutoSBHistoryTracker(unittest.TestCase):
    def test_records_warmup_delta_and_resets_iteration(self):
        tracker = AutoSBHistoryTracker()

        tracker.record_forward(1000)
        tracker.record_forward(900)
        tracker.record_forward(850)

        self.assertTrue(tracker.in_warmup())
        self.assertEqual(tracker.step_idx, 3)
        self.assertEqual(tracker.forward_count, 3)
        self.assertEqual(tracker.backward_count, 0)
        self.assertEqual(tracker.max_delta, 100)

        self.assertFalse(tracker.record_backward())
        self.assertFalse(tracker.in_warmup())
        self.assertEqual(tracker.backward_count, 1)

        self.assertFalse(tracker.record_backward())
        self.assertTrue(tracker.record_backward())
        self.assertTrue(tracker.in_warmup())
        self.assertEqual(tracker.step_idx, 0)
        self.assertEqual(tracker.forward_count, 0)
        self.assertEqual(tracker.backward_count, 0)
        self.assertEqual(tracker.max_delta, 0)
        self.assertEqual(tracker.prev_total_steps, 3)
        self.assertEqual(tracker.prev_max_delta, 100)

    def test_predicts_remaining_need_from_previous_iteration(self):
        mb = 1024 * 1024
        tracker = AutoSBHistoryTracker()

        tracker.record_forward(1000 * mb)
        tracker.record_forward(900 * mb)
        tracker.record_forward(840 * mb)
        tracker.record_backward()
        tracker.record_backward()
        tracker.record_backward()

        tracker.record_forward(1000 * mb)
        expected_need = int(100 * mb * 3 * 1.2) + 128 * mb
        self.assertEqual(tracker.predicted_need_for_remaining(), expected_need)
        self.assertTrue(tracker.should_degrade(expected_need - 1))
        self.assertFalse(tracker.should_degrade(expected_need))

        tracker.record_forward(950 * mb)
        tracker.record_forward(900 * mb)
        self.assertEqual(
            tracker.predicted_need_for_remaining(),
            int(100 * mb * 1 * 1.2) + 128 * mb,
        )

        tracker.record_forward(850 * mb)
        self.assertEqual(tracker.predicted_need_for_remaining(), 0)

    def test_ignores_memory_growth_and_cold_start(self):
        tracker = AutoSBHistoryTracker()

        tracker.record_forward(100)
        tracker.record_forward(120)

        self.assertEqual(tracker.max_delta, 0)
        self.assertEqual(tracker.predicted_need_for_remaining(), 0)
        self.assertFalse(tracker.should_degrade(0))


if __name__ == "__main__":
    unittest.main()
