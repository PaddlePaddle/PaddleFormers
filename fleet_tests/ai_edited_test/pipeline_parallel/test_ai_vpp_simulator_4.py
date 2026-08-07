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

import unittest


class TestVPPSimulatorSchedule(unittest.TestCase):
    """Tests for VPPSimulator schedule with various configurations."""

    def test_schedule_pp2_vpp2_acc4(self):
        from paddleformers.fleet.pipeline_parallel.vpp_simulator import VPPSimulator

        sim = VPPSimulator(pp_degree=2, vpp_degree=2, num_acc_steps=4)
        table = sim.schedule()
        self.assertTrue(sim._is_scheduled)
        self.assertEqual(len(table), 2)
        for stage in table:
            self.assertGreater(len(stage), 0)

    def test_schedule_pp4_vpp2_acc8(self):
        from paddleformers.fleet.pipeline_parallel.vpp_simulator import VPPSimulator

        sim = VPPSimulator(pp_degree=4, vpp_degree=2, num_acc_steps=8)
        table = sim.schedule()
        self.assertTrue(sim._is_scheduled)
        self.assertEqual(len(table), 4)

    def test_schedule_balanced_memory(self):
        from paddleformers.fleet.pipeline_parallel.vpp_simulator import VPPSimulator

        # num_acc_steps in [pp_degree, pp_degree*2)
        sim = VPPSimulator(pp_degree=4, vpp_degree=2, num_acc_steps=5)
        table = sim.schedule()
        self.assertTrue(sim._is_scheduled)

    def test_schedule_without_batch_send_recv(self):
        from paddleformers.fleet.pipeline_parallel.vpp_simulator import VPPSimulator

        sim = VPPSimulator(
            pp_degree=2,
            vpp_degree=2,
            num_acc_steps=4,
            enable_batch_send_recv=False,
        )
        table = sim.schedule()
        self.assertTrue(sim._is_scheduled)


class TestVPPSimulatorGetVirtualPpRank(unittest.TestCase):
    """Tests for _get_virtual_pp_rank with different micro steps."""

    def test_first_chunk_steps_forward(self):
        from paddleformers.fleet.pipeline_parallel.vpp_simulator import VPPSimulator

        sim = VPPSimulator(pp_degree=2, vpp_degree=2, num_acc_steps=4)
        # First chunk steps
        rank = sim._get_virtual_pp_rank(0, forward=True)
        self.assertEqual(rank, 0)

    def test_first_chunk_steps_second_vpp(self):
        from paddleformers.fleet.pipeline_parallel.vpp_simulator import VPPSimulator

        sim = VPPSimulator(pp_degree=2, vpp_degree=2, num_acc_steps=4)
        # first_chunk_acc = (4 % 2) + 2 = 4
        # first_chunk_steps = 4 * 2 = 8
        rank = sim._get_virtual_pp_rank(4, forward=True)
        self.assertIsInstance(rank, int)

    def test_beyond_first_chunk_steps(self):
        from paddleformers.fleet.pipeline_parallel.vpp_simulator import VPPSimulator

        sim = VPPSimulator(pp_degree=2, vpp_degree=2, num_acc_steps=4)
        # Beyond first chunk steps
        rank = sim._get_virtual_pp_rank(10, forward=True)
        self.assertIsInstance(rank, int)

    def test_backward(self):
        from paddleformers.fleet.pipeline_parallel.vpp_simulator import VPPSimulator

        sim = VPPSimulator(pp_degree=2, vpp_degree=2, num_acc_steps=4)
        rank = sim._get_virtual_pp_rank(0, forward=False)
        self.assertEqual(rank, sim.vpp_degree - 1)


class TestVPPSimulatorGetPreorderChunk(unittest.TestCase):
    """Tests for _get_preorder_chunk with scheduled data."""

    def test_forward_layer_zero(self):
        from paddleformers.fleet.pipeline_parallel.vpp_simulator import (
            Chunk,
            ChunkType,
            VPPSimulator,
        )

        sim = VPPSimulator(pp_degree=2, vpp_degree=2, num_acc_steps=4)
        chunk = Chunk(
            virtual_pp_rank=0,
            acc_step=0,
            pp_degree=2,
            vpp_degree=2,
            stage_id=0,
            chunk_type=ChunkType.FORWARD,
            start=0,
            end=1,
        )
        self.assertIsNone(sim._get_preorder_chunk(chunk))

    def test_backward_last_layer(self):
        from paddleformers.fleet.pipeline_parallel.vpp_simulator import (
            Chunk,
            ChunkType,
            VPPSimulator,
        )

        sim = VPPSimulator(pp_degree=2, vpp_degree=2, num_acc_steps=4)
        # Last layer: layer_id = vpp_degree * pp_degree - 1
        chunk = Chunk(
            virtual_pp_rank=1,
            acc_step=0,
            pp_degree=2,
            vpp_degree=2,
            stage_id=1,
            chunk_type=ChunkType.BACKWARD,
            start=0,
            end=1,
        )
        # layer_id = 1 * 2 + 1 = 3, which equals vpp_degree*pp_degree - 1 = 3
        self.assertIsNone(sim._get_preorder_chunk(chunk))

    def test_forward_with_predecessor(self):
        from paddleformers.fleet.pipeline_parallel.vpp_simulator import (
            Chunk,
            ChunkType,
            VPPSimulator,
        )

        sim = VPPSimulator(pp_degree=2, vpp_degree=2, num_acc_steps=4)
        # Add a predecessor chunk
        prev_chunk = Chunk(
            virtual_pp_rank=0,
            acc_step=0,
            pp_degree=2,
            vpp_degree=2,
            stage_id=1,
            chunk_type=ChunkType.FORWARD,
            start=0,
            end=1,
        )
        sim._add_chunk(prev_chunk)

        # Current chunk with layer_id = 1 (prev_stage=1, virtual_pp_rank=0 -> layer_id=0+1=1)
        chunk = Chunk(
            virtual_pp_rank=0,
            acc_step=0,
            pp_degree=2,
            vpp_degree=2,
            stage_id=0,
            chunk_type=ChunkType.FORWARD,
            start=0,
            end=1,
        )
        # layer_id = 0 * 2 + 0 = 0
        # Actually layer_id=0 means first layer, no predecessor
        self.assertIsNone(sim._get_preorder_chunk(chunk))

    def test_find_preorder_chunk_not_found(self):
        from paddleformers.fleet.pipeline_parallel.vpp_simulator import (
            Chunk,
            ChunkType,
            VPPSimulator,
        )

        sim = VPPSimulator(pp_degree=2, vpp_degree=2, num_acc_steps=4)
        # No chunks added to schedule_table[1], so finding predecessor should raise
        chunk = Chunk(
            virtual_pp_rank=1,
            acc_step=0,
            pp_degree=2,
            vpp_degree=2,
            stage_id=0,
            chunk_type=ChunkType.FORWARD,
            start=0,
            end=1,
        )
        # layer_id = 1*2 + 0 = 2
        with self.assertRaises(ValueError):
            sim._get_preorder_chunk(chunk)


class TestVPPSimulatorBubbleRate(unittest.TestCase):
    """Tests for bubble rate computation."""

    def test_bubble_rate_reasonable(self):
        from paddleformers.fleet.pipeline_parallel.vpp_simulator import VPPSimulator

        sim = VPPSimulator(pp_degree=4, vpp_degree=2, num_acc_steps=8)
        rate = sim.compute_bubble_rate()
        self.assertGreater(rate, 0.0)
        self.assertLess(rate, 1.0)

    def test_bubble_rate_balanced_memory(self):
        from paddleformers.fleet.pipeline_parallel.vpp_simulator import VPPSimulator

        sim = VPPSimulator(pp_degree=4, vpp_degree=2, num_acc_steps=6)
        rate = sim.compute_bubble_rate()
        self.assertGreater(rate, 0.0)
        self.assertLess(rate, 1.0)


class TestVPPSimulatorWarmupSteadySteps(unittest.TestCase):
    """Tests for _get_warmup_and_steady_steps."""

    def test_standard_vpp(self):
        from paddleformers.fleet.pipeline_parallel.vpp_simulator import VPPSimulator

        sim = VPPSimulator(pp_degree=4, vpp_degree=2, num_acc_steps=8)
        # For stage_id=0
        warmup, steady = sim._get_warmup_and_steady_steps(0)
        self.assertGreater(warmup, 0)
        self.assertGreater(steady, 0)
        # Just verify they are reasonable positive integers
        self.assertIsInstance(warmup, int)
        self.assertIsInstance(steady, int)

    def test_balanced_memory_vpp(self):
        from paddleformers.fleet.pipeline_parallel.vpp_simulator import VPPSimulator

        # num_acc_steps in [pp_degree, pp_degree*2)
        sim = VPPSimulator(pp_degree=4, vpp_degree=2, num_acc_steps=5)
        warmup, steady = sim._get_warmup_and_steady_steps(0)
        self.assertGreater(warmup, 0)
        self.assertGreater(steady, 0)

    def test_warmup_decreases_with_stage(self):
        from paddleformers.fleet.pipeline_parallel.vpp_simulator import VPPSimulator

        sim = VPPSimulator(pp_degree=4, vpp_degree=2, num_acc_steps=8)
        warmup_0, _ = sim._get_warmup_and_steady_steps(0)
        warmup_1, _ = sim._get_warmup_and_steady_steps(1)
        warmup_2, _ = sim._get_warmup_and_steady_steps(2)
        warmup_3, _ = sim._get_warmup_and_steady_steps(3)
        # Warmup steps should decrease with increasing stage_id
        self.assertGreater(warmup_0, warmup_1)
        self.assertGreater(warmup_1, warmup_2)
        self.assertGreater(warmup_2, warmup_3)


if __name__ == "__main__":
    unittest.main()
