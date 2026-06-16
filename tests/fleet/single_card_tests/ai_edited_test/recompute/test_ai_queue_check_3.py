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
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

import queue
import unittest


class TestRefinedRcomputeQueue(unittest.TestCase):
    """Tests for RefinedRcomputeQueue in queue_check module."""

    def test_init(self):
        """Test RefinedRcomputeQueue initialization."""
        from paddleformers.fleet.refined_recompute.queue_check import (
            RefinedRcomputeQueue,
        )

        rrq = RefinedRcomputeQueue()
        self.assertIsInstance(rrq.rr_queue, dict)

    def test_update_adds_queue(self):
        """Test update adds a queue."""
        from paddleformers.fleet.refined_recompute.queue_check import (
            RefinedRcomputeQueue,
        )

        rrq = RefinedRcomputeQueue()
        q = queue.Queue()
        rrq.update(q, "test_queue")
        # Check that some key was added
        self.assertGreater(len(rrq.rr_queue), 0)

    def test_update_duplicate_raises(self):
        """Test update with duplicate queue raises ValueError."""
        from paddleformers.fleet.refined_recompute.queue_check import (
            RefinedRcomputeQueue,
        )

        rrq = RefinedRcomputeQueue()
        q = queue.Queue()
        rrq.update(q, "test_queue")
        with self.assertRaises(ValueError):
            rrq.update(q, "test_queue")

    def test_check_empty_queues(self):
        """Test check passes when all queues are empty."""
        from paddleformers.fleet.refined_recompute.queue_check import (
            RefinedRcomputeQueue,
        )

        rrq = RefinedRcomputeQueue()
        q = queue.Queue()
        rrq.update(q, "empty_queue")
        # Should not raise
        rrq.check()

    def test_check_non_empty_queues_raises(self):
        """Test check raises when queues are non-empty."""
        from paddleformers.fleet.refined_recompute.queue_check import (
            RefinedRcomputeQueue,
        )

        rrq = RefinedRcomputeQueue()
        q = queue.Queue()
        q.put("item")
        rrq.update(q, "non_empty_queue")
        with self.assertRaises(ValueError):
            rrq.check()

    def test_global_rr_queue_log_exists(self):
        """Test global_rr_queue_log is initialized."""
        from paddleformers.fleet.refined_recompute.queue_check import (
            global_rr_queue_log,
        )

        self.assertIsNotNone(global_rr_queue_log)

    def test_update_with_different_names(self):
        """Test update with different queue names."""
        from paddleformers.fleet.refined_recompute.queue_check import (
            RefinedRcomputeQueue,
        )

        rrq = RefinedRcomputeQueue()
        q1 = queue.Queue()
        q2 = queue.Queue()
        rrq.update(q1, "queue1")
        rrq.update(q2, "queue2")
        self.assertEqual(len(rrq.rr_queue), 2)


if __name__ == "__main__":
    unittest.main()
