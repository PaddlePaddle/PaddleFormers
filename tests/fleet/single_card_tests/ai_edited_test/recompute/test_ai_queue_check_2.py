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


class TestRefinedRcomputeQueueInit(unittest.TestCase):
    """Tests for RefinedRcomputeQueue initialization."""

    def test_init_creates_defaultdict(self):
        """Test init creates a defaultdict of queues."""
        rq = RefinedRcomputeQueue()
        self.assertEqual(len(rq.rr_queue), 0)

    def test_init_empty(self):
        """Test that the queue starts empty."""
        rq = RefinedRcomputeQueue()
        self.assertEqual(len(rq.rr_queue), 0)


class TestRefinedRcomputeQueueUpdate(unittest.TestCase):
    """Tests for RefinedRcomputeQueue.update method."""

    def test_update_adds_queue(self):
        """Test update adds a queue to the tracker."""
        rq = RefinedRcomputeQueue()
        q = queue.Queue()
        rq.update(q, "test_queue")
        self.assertEqual(len(rq.rr_queue), 1)

    def test_update_queue_name_includes_id(self):
        """Test queue name includes id to differentiate instances."""
        rq = RefinedRcomputeQueue()
        q1 = queue.Queue()
        q2 = queue.Queue()
        rq.update(q1, "test_queue")
        rq.update(q2, "test_queue")
        # Different queue instances should have different names
        self.assertEqual(len(rq.rr_queue), 2)

    def test_update_duplicate_name_raises(self):
        """Test update raises for duplicate queue name (same id)."""
        rq = RefinedRcomputeQueue()
        q = queue.Queue()
        rq.update(q, "test_queue")
        with self.assertRaises(ValueError) as ctx:
            rq.update(q, "test_queue")
        self.assertIn("already exists", str(ctx.exception))

    def test_update_stores_queue_reference(self):
        """Test update stores the actual queue reference."""
        rq = RefinedRcomputeQueue()
        q = queue.Queue()
        rq.update(q, "test_queue")
        # Find the key
        key = next(iter(rq.rr_queue.keys()))
        self.assertTrue(rq.rr_queue[key] is q)


class TestRefinedRcomputeQueueCheck(unittest.TestCase):
    """Tests for RefinedRcomputeQueue.check method."""

    def test_check_empty_queues_passes(self):
        """Test check passes when all queues are empty."""
        rq = RefinedRcomputeQueue()
        q1 = queue.Queue()
        q2 = queue.Queue()
        rq.update(q1, "queue1")
        rq.update(q2, "queue2")
        # Should not raise
        rq.check()

    def test_check_non_empty_queue_raises(self):
        """Test check raises when a queue is not empty."""
        rq = RefinedRcomputeQueue()
        q = queue.Queue()
        q.put("item")
        rq.update(q, "non_empty_queue")
        with self.assertRaises(ValueError) as ctx:
            rq.check()
        self.assertIn("not empty", str(ctx.exception))

    def test_check_multiple_non_empty_queues(self):
        """Test check raises listing all non-empty queues."""
        rq = RefinedRcomputeQueue()
        q1 = queue.Queue()
        q1.put("item1")
        q2 = queue.Queue()
        q2.put("item2")
        rq.update(q1, "queue_a")
        rq.update(q2, "queue_b")
        with self.assertRaises(ValueError) as ctx:
            rq.check()
        msg = str(ctx.exception)
        # Both queue names should be in the error message
        self.assertIn("queue_a", msg)
        self.assertIn("queue_b", msg)

    def test_check_mixed_empty_and_non_empty(self):
        """Test check raises when some queues are non-empty."""
        rq = RefinedRcomputeQueue()
        q_empty = queue.Queue()
        q_full = queue.Queue()
        q_full.put("item")
        rq.update(q_empty, "empty_queue")
        rq.update(q_full, "full_queue")
        with self.assertRaises(ValueError) as ctx:
            rq.check()
        self.assertIn("full_queue", str(ctx.exception))
        self.assertNotIn("empty_queue", str(ctx.exception))


class TestGlobalRRQueueLog(unittest.TestCase):
    """Tests for global_rr_queue_log singleton."""

    def test_global_instance_exists(self):
        """Test global_rr_queue_log is a RefinedRcomputeQueue instance."""
        self.assertIsInstance(global_rr_queue_log, RefinedRcomputeQueue)

    def test_global_instance_is_shared(self):
        """Test global_rr_queue_log is the same instance across accesses."""
        from paddleformers.fleet.refined_recompute.queue_check import (
            global_rr_queue_log as g1,
        )
        from paddleformers.fleet.refined_recompute.queue_check import (
            global_rr_queue_log as g2,
        )

        self.assertIs(g1, g2)


class TestRefinedRcomputeQueueIntegration(unittest.TestCase):
    """Integration tests for RefinedRcomputeQueue."""

    def test_update_check_cycle(self):
        """Test full update -> check cycle."""
        rq = RefinedRcomputeQueue()
        q = queue.Queue()
        rq.update(q, "test")

        # Should pass check when empty
        rq.check()

        # Put something in
        q.put("data")
        with self.assertRaises(ValueError):
            rq.check()

        # Empty it again
        q.get()
        rq.check()

    def test_multiple_queues_independent(self):
        """Test multiple queues operate independently."""
        rq = RefinedRcomputeQueue()
        q1 = queue.Queue()
        q2 = queue.Queue()
        rq.update(q1, "q1")
        rq.update(q2, "q2")

        q1.put("item")
        # q2 is still empty but q1 is not
        with self.assertRaises(ValueError) as ctx:
            rq.check()
        # Only q1 should be mentioned
        self.assertNotIn("q2", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
