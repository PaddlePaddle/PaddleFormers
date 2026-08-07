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
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)


import unittest
from unittest.mock import MagicMock


class TestChunkType(unittest.TestCase):
    """Tests for ChunkType enum."""

    def test_chunk_type_values(self):
        from paddleformers.fleet.pipeline_parallel.vpp_simulator import ChunkType

        self.assertEqual(ChunkType.FORWARD.value, "F")
        self.assertEqual(ChunkType.BACKWARD.value, "B")
        self.assertEqual(ChunkType.BUBBLE.value, "Z")


class TestChunk(unittest.TestCase):
    """Tests for Chunk class."""

    def test_chunk_init(self):
        from paddleformers.fleet.pipeline_parallel.vpp_simulator import Chunk, ChunkType

        chunk = Chunk(
            virtual_pp_rank=0,
            acc_step=1,
            pp_degree=2,
            vpp_degree=3,
            stage_id=0,
            chunk_type=ChunkType.FORWARD,
            start=0,
            end=1,
        )
        self.assertEqual(chunk.virtual_pp_rank, 0)
        self.assertEqual(chunk.acc_step, 1)
        self.assertEqual(chunk.stage_id, 0)
        self.assertEqual(chunk.chunk_type, ChunkType.FORWARD)
        self.assertEqual(chunk.start, 0)
        self.assertEqual(chunk.end, 1)
        self.assertEqual(chunk.layer_id, 0)
        self.assertEqual(chunk.barrier_step, -1)

    def test_chunk_layer_id_bubble(self):
        from paddleformers.fleet.pipeline_parallel.vpp_simulator import Chunk, ChunkType

        chunk = Chunk(
            virtual_pp_rank=0,
            acc_step=0,
            pp_degree=2,
            vpp_degree=3,
            stage_id=0,
            chunk_type=ChunkType.BUBBLE,
            start=0,
            end=1,
        )
        self.assertIsNone(chunk.layer_id)

    def test_chunk_str_forward(self):
        from paddleformers.fleet.pipeline_parallel.vpp_simulator import Chunk, ChunkType

        chunk = Chunk(
            virtual_pp_rank=1,
            acc_step=0,
            pp_degree=2,
            vpp_degree=3,
            stage_id=1,
            chunk_type=ChunkType.FORWARD,
            start=5,
            end=6,
        )
        s = str(chunk)
        self.assertIn("F", s)
        self.assertIn("1", s)

    def test_chunk_str_bubble(self):
        from paddleformers.fleet.pipeline_parallel.vpp_simulator import Chunk, ChunkType

        chunk = Chunk(
            virtual_pp_rank=0,
            acc_step=0,
            pp_degree=2,
            vpp_degree=3,
            stage_id=0,
            chunk_type=ChunkType.BUBBLE,
            start=0,
            end=2,
        )
        s = str(chunk)
        self.assertIn("Z", s)

    def test_chunk_repr(self):
        from paddleformers.fleet.pipeline_parallel.vpp_simulator import Chunk, ChunkType

        chunk = Chunk(
            virtual_pp_rank=0,
            acc_step=0,
            pp_degree=2,
            vpp_degree=3,
            stage_id=0,
            chunk_type=ChunkType.FORWARD,
            start=0,
            end=1,
        )
        self.assertEqual(str(chunk), repr(chunk))


class TestVPPSimulator(unittest.TestCase):
    """Tests for VPPSimulator class."""

    def test_init(self):
        from paddleformers.fleet.pipeline_parallel.vpp_simulator import VPPSimulator

        sim = VPPSimulator(pp_degree=2, vpp_degree=2, num_acc_steps=4)
        self.assertEqual(sim.pp_degree, 2)
        self.assertEqual(sim.vpp_degree, 2)
        self.assertEqual(sim.num_acc_steps, 4)
        self.assertEqual(len(sim.schedule_table), 2)
        self.assertFalse(sim._is_scheduled)

    def test_init_enable_batch_send_recv(self):
        from paddleformers.fleet.pipeline_parallel.vpp_simulator import VPPSimulator

        sim = VPPSimulator(
            pp_degree=2,
            vpp_degree=2,
            num_acc_steps=4,
            enable_batch_send_recv=False,
        )
        self.assertFalse(sim.enable_batch_send_recv)

    def test_get_consume_time_forward(self):
        from paddleformers.fleet.pipeline_parallel.vpp_simulator import (
            ChunkType,
            VPPSimulator,
        )

        sim = VPPSimulator(pp_degree=2, vpp_degree=2, num_acc_steps=4)
        self.assertEqual(sim._get_consume_time(0, 0, ChunkType.FORWARD), 1)

    def test_get_consume_time_backward(self):
        from paddleformers.fleet.pipeline_parallel.vpp_simulator import (
            ChunkType,
            VPPSimulator,
        )

        sim = VPPSimulator(pp_degree=2, vpp_degree=2, num_acc_steps=4)
        self.assertEqual(sim._get_consume_time(0, 0, ChunkType.BACKWARD), 2)

    def test_get_consume_time_bubble(self):
        from paddleformers.fleet.pipeline_parallel.vpp_simulator import (
            ChunkType,
            VPPSimulator,
        )

        sim = VPPSimulator(pp_degree=2, vpp_degree=2, num_acc_steps=4)
        self.assertEqual(sim._get_consume_time(0, 0, ChunkType.BUBBLE), 1)

    def test_get_warmup_and_steady_steps_standard(self):
        from paddleformers.fleet.pipeline_parallel.vpp_simulator import VPPSimulator

        sim = VPPSimulator(pp_degree=4, vpp_degree=2, num_acc_steps=8)
        warmup, steady = sim._get_warmup_and_steady_steps(0)
        self.assertGreater(warmup, 0)
        self.assertGreater(steady, 0)

    def test_get_warmup_and_steady_steps_balanced_memory(self):
        from paddleformers.fleet.pipeline_parallel.vpp_simulator import VPPSimulator

        # num_acc_steps in [pp_degree, pp_degree*2)
        sim = VPPSimulator(pp_degree=4, vpp_degree=2, num_acc_steps=5)
        warmup, steady = sim._get_warmup_and_steady_steps(0)
        self.assertGreater(warmup, 0)
        self.assertGreater(steady, 0)

    def test_get_virtual_pp_rank_forward_first_chunk(self):
        from paddleformers.fleet.pipeline_parallel.vpp_simulator import VPPSimulator

        sim = VPPSimulator(pp_degree=4, vpp_degree=2, num_acc_steps=4)
        rank = sim._get_virtual_pp_rank(0, forward=True)
        self.assertEqual(rank, 0)

    def test_get_virtual_pp_rank_backward(self):
        from paddleformers.fleet.pipeline_parallel.vpp_simulator import VPPSimulator

        sim = VPPSimulator(pp_degree=2, vpp_degree=2, num_acc_steps=4)
        rank = sim._get_virtual_pp_rank(0, forward=False)
        self.assertEqual(rank, sim.vpp_degree - 1)

    def test_schedule(self):
        from paddleformers.fleet.pipeline_parallel.vpp_simulator import VPPSimulator

        sim = VPPSimulator(pp_degree=2, vpp_degree=2, num_acc_steps=4)
        table = sim.schedule()
        self.assertTrue(sim._is_scheduled)
        self.assertEqual(len(table), 2)
        for stage in table:
            self.assertGreater(len(stage), 0)

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

    def test_compute_bubble_rate(self):
        from paddleformers.fleet.pipeline_parallel.vpp_simulator import VPPSimulator

        sim = VPPSimulator(pp_degree=2, vpp_degree=2, num_acc_steps=4)
        rate = sim.compute_bubble_rate()
        self.assertGreaterEqual(rate, 0.0)
        self.assertLessEqual(rate, 1.0)

    def test_compute_bubble_rate_unscheduled(self):
        from paddleformers.fleet.pipeline_parallel.vpp_simulator import VPPSimulator

        sim = VPPSimulator(pp_degree=2, vpp_degree=2, num_acc_steps=4)
        # schedule() is called internally if not scheduled
        rate = sim.compute_bubble_rate()
        self.assertTrue(sim._is_scheduled)
        self.assertGreaterEqual(rate, 0.0)

    def test_add_chunk(self):
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
        sim._add_chunk(chunk)
        self.assertEqual(len(sim.schedule_table[0]), 1)

    def test_barrier_two_chunk(self):
        from paddleformers.fleet.pipeline_parallel.vpp_simulator import (
            Chunk,
            ChunkType,
            VPPSimulator,
        )

        sim = VPPSimulator(pp_degree=2, vpp_degree=2, num_acc_steps=4)
        c1 = Chunk(
            virtual_pp_rank=0,
            acc_step=0,
            pp_degree=2,
            vpp_degree=2,
            stage_id=0,
            chunk_type=ChunkType.FORWARD,
            start=1,
            end=2,
        )
        c2 = Chunk(
            virtual_pp_rank=0,
            acc_step=0,
            pp_degree=2,
            vpp_degree=2,
            stage_id=1,
            chunk_type=ChunkType.FORWARD,
            start=5,
            end=6,
        )
        sim._barrier_two_chunk(c1, c2)
        self.assertEqual(c1.start, 5)

    def test_barrier_two_chunk_mismatch_type(self):
        from paddleformers.fleet.pipeline_parallel.vpp_simulator import (
            Chunk,
            ChunkType,
            VPPSimulator,
        )

        sim = VPPSimulator(pp_degree=2, vpp_degree=2, num_acc_steps=4)
        c1 = Chunk(
            virtual_pp_rank=0,
            acc_step=0,
            pp_degree=2,
            vpp_degree=2,
            stage_id=0,
            chunk_type=ChunkType.FORWARD,
            start=1,
            end=2,
        )
        c2 = Chunk(
            virtual_pp_rank=0,
            acc_step=0,
            pp_degree=2,
            vpp_degree=2,
            stage_id=1,
            chunk_type=ChunkType.BACKWARD,
            start=5,
            end=6,
        )
        with self.assertRaises(AssertionError):
            sim._barrier_two_chunk(c1, c2)

    def test_get_preorder_chunk_forward_layer_zero(self):
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
        result = sim._get_preorder_chunk(chunk)
        self.assertIsNone(result)

    def test_get_preorder_chunk_bubble(self):
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
            chunk_type=ChunkType.BUBBLE,
            start=0,
            end=1,
        )
        result = sim._get_preorder_chunk(chunk)
        self.assertIsNone(result)


class TestPPChunkRecorder(unittest.TestCase):
    """Tests for PPChunkRecorder class."""

    def test_init(self):
        from paddleformers.fleet.pipeline_parallel.vpp_simulator import PPChunkRecorder

        recorder = PPChunkRecorder(
            pp_degree=2,
            vpp_degree=2,
            num_acc_steps=4,
            num_hidden_layers=8,
            num_empty_layers_add_in_head=0,
            num_empty_layers_add_in_tail=0,
        )
        self.assertEqual(recorder.num_hidden_layers, 8)
        self.assertEqual(len(recorder.acc_stamp), 8)

    def test_step(self):
        from paddleformers.fleet.pipeline_parallel.vpp_simulator import PPChunkRecorder

        recorder = PPChunkRecorder(
            pp_degree=2,
            vpp_degree=2,
            num_acc_steps=4,
            num_hidden_layers=8,
            num_empty_layers_add_in_head=0,
            num_empty_layers_add_in_tail=0,
        )
        recorder.acc_stamp[0] = 5
        recorder.step()
        self.assertEqual(recorder.acc_stamp[0], 0)

    def test_record_chunk_forward_in_range(self):
        from paddleformers.fleet.pipeline_parallel.vpp_simulator import PPChunkRecorder

        recorder = PPChunkRecorder(
            pp_degree=2,
            vpp_degree=2,
            num_acc_steps=4,
            num_hidden_layers=8,
            num_empty_layers_add_in_head=2,
            num_empty_layers_add_in_tail=0,
        )
        result = recorder.record_chunk_forward(4)
        self.assertFalse(result)
        self.assertEqual(recorder.acc_stamp[2], 1)

    def test_record_chunk_forward_out_of_range_head(self):
        from paddleformers.fleet.pipeline_parallel.vpp_simulator import PPChunkRecorder

        recorder = PPChunkRecorder(
            pp_degree=2,
            vpp_degree=2,
            num_acc_steps=4,
            num_hidden_layers=8,
            num_empty_layers_add_in_head=2,
            num_empty_layers_add_in_tail=0,
        )
        result = recorder.record_chunk_forward(0)
        self.assertFalse(result)
        self.assertEqual(recorder.acc_stamp[0], 0)

    def test_record_chunk_forward_out_of_range_tail(self):
        from paddleformers.fleet.pipeline_parallel.vpp_simulator import PPChunkRecorder

        recorder = PPChunkRecorder(
            pp_degree=2,
            vpp_degree=2,
            num_acc_steps=4,
            num_hidden_layers=8,
            num_empty_layers_add_in_head=2,
            num_empty_layers_add_in_tail=2,
        )
        result = recorder.record_chunk_forward(9)
        self.assertFalse(result)
        self.assertEqual(recorder.acc_stamp[0], 0)


class TestGlobalRecorder(unittest.TestCase):
    """Tests for global recorder functions."""

    def test_set_and_get_global_recorder(self):
        from paddleformers.fleet.pipeline_parallel.vpp_simulator import (
            get_global_pp_recorder,
            set_global_pp_chunk_recorder,
        )

        mock_recorder = MagicMock()
        set_global_pp_chunk_recorder(mock_recorder)
        self.assertEqual(get_global_pp_recorder(), mock_recorder)


if __name__ == "__main__":
    unittest.main()
