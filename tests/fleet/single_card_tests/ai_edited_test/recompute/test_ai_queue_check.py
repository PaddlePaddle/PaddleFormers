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
import os
import sys

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)


import queue
import unittest

from paddleformers.fleet.refined_recompute.queue_check import (
    RefinedRcomputeQueue,
    global_rr_queue_log,
)


class TestRefinedRcomputeQueue(unittest.TestCase):
    """Tests for RefinedRcomputeQueue class."""

    def test_init_creates_defaultdict(self):
        """Test that init creates a defaultdict of queues."""
        rr = RefinedRcomputeQueue()
        self.assertEqual(len(rr.rr_queue), 0)

    def test_update_single_queue(self):
        """Test updating with a single queue."""
        rr = RefinedRcomputeQueue()
        q = queue.Queue()
        rr.update(q, "test_queue")

        self.assertEqual(len(rr.rr_queue), 1)
        # Verify the key format includes queue id
        keys = list(rr.rr_queue.keys())
        self.assertTrue(keys[0].startswith("test_queue_"))
        self.assertTrue(keys[0].endswith(f"_{id(q)}"))

    def test_update_multiple_queues(self):
        """Test updating with multiple different queues."""
        rr = RefinedRcomputeQueue()
        q1 = queue.Queue()
        q2 = queue.Queue()

        rr.update(q1, "queue_a")
        rr.update(q2, "queue_b")

        self.assertEqual(len(rr.rr_queue), 2)

    def test_update_default_queue_name(self):
        """Test updating with default queue name."""
        rr = RefinedRcomputeQueue()
        q = queue.Queue()
        rr.update(q)

        keys = list(rr.rr_queue.keys())
        self.assertTrue(keys[0].startswith("unknown_"))

    def test_update_duplicate_queue_name_raises(self):
        """Test that updating with the same queue raises ValueError."""
        rr = RefinedRcomputeQueue()
        q = queue.Queue()

        rr.update(q, "duplicate_name")
        with self.assertRaises(ValueError) as ctx:
            rr.update(q, "duplicate_name")
        self.assertIn("already exists", str(ctx.exception))

    def test_check_empty_queues_passes(self):
        """Test that check passes when all queues are empty."""
        rr = RefinedRcomputeQueue()
        q1 = queue.Queue()
        q2 = queue.Queue()

        rr.update(q1, "empty1")
        rr.update(q2, "empty2")

        # Should not raise
        rr.check()

    def test_check_nonempty_queue_raises(self):
        """Test that check raises ValueError when a queue is not empty."""
        rr = RefinedRcomputeQueue()
        q = queue.Queue()
        q.put("some_data")

        rr.update(q, "nonempty")
        with self.assertRaises(ValueError) as ctx:
            rr.check()
        self.assertIn("nonempty", str(ctx.exception))
        self.assertIn("not empty", str(ctx.exception))

    def test_check_multiple_nonempty_queues_raises(self):
        """Test that check raises listing all non-empty queues."""
        rr = RefinedRcomputeQueue()
        q1 = queue.Queue()
        q1.put("data1")
        q2 = queue.Queue()
        q2.put("data2")

        rr.update(q1, "q1")
        rr.update(q2, "q2")

        with self.assertRaises(ValueError) as ctx:
            rr.check()
        msg = str(ctx.exception)
        self.assertIn("q1", msg)
        self.assertIn("q2", msg)


class TestGlobalQueueLog(unittest.TestCase):
    """Tests for the global_rr_queue_log singleton."""

    def test_global_instance_exists(self):
        """Test that global_rr_queue_log is a RefinedRcomputeQueue instance."""
        self.assertIsInstance(global_rr_queue_log, RefinedRcomputeQueue)

    def test_global_all_export(self):
        """Test that global_rr_queue_log is in __all__."""
        from paddleformers.fleet.refined_recompute.queue_check import __all__

        self.assertIn("global_rr_queue_log", __all__)


if __name__ == "__main__":
    unittest.main()
