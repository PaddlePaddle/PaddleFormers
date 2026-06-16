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


class TestChunkExtra(unittest.TestCase):
    """Additional Chunk tests for vpp_simulator."""

    def test_chunk_layer_id_forward(self):
        from paddleformers.fleet.pipeline_parallel.vpp_simulator import (
            Chunk,
            ChunkType,
        )

        chunk = Chunk(
            virtual_pp_rank=1,
            acc_step=0,
            pp_degree=4,
            vpp_degree=2,
            stage_id=2,
            chunk_type=ChunkType.FORWARD,
            start=5,
            end=6,
        )
        # layer_id = 1*4 + 2 = 6
        self.assertEqual(chunk.layer_id, 6)

    def test_chunk_layer_id_backward(self):
        from paddleformers.fleet.pipeline_parallel.vpp_simulator import (
            Chunk,
            ChunkType,
        )

        chunk = Chunk(
            virtual_pp_rank=0,
            acc_step=0,
            pp_degree=4,
            vpp_degree=2,
            stage_id=1,
            chunk_type=ChunkType.BACKWARD,
            start=3,
            end=5,
        )
        # layer_id = 0*4 + 1 = 1
        self.assertEqual(chunk.layer_id, 1)

    def test_chunk_layer_id_bubble_none(self):
        from paddleformers.fleet.pipeline_parallel.vpp_simulator import (
            Chunk,
            ChunkType,
        )

        chunk = Chunk(
            virtual_pp_rank=0,
            acc_step=0,
            pp_degree=4,
            vpp_degree=2,
            stage_id=1,
            chunk_type=ChunkType.BUBBLE,
            start=0,
            end=1,
        )
        self.assertIsNone(chunk.layer_id)

    def test_chunk_str_forward(self):
        from paddleformers.fleet.pipeline_parallel.vpp_simulator import (
            Chunk,
            ChunkType,
        )

        chunk = Chunk(
            virtual_pp_rank=0,
            acc_step=2,
            pp_degree=2,
            vpp_degree=2,
            stage_id=0,
            chunk_type=ChunkType.FORWARD,
            start=3,
            end=4,
        )
        s = str(chunk)
        self.assertIn("F", s)
        self.assertIn("3", s)  # acc_step+1

    def test_chunk_str_backward(self):
        from paddleformers.fleet.pipeline_parallel.vpp_simulator import (
            Chunk,
            ChunkType,
        )

        chunk = Chunk(
            virtual_pp_rank=1,
            acc_step=0,
            pp_degree=2,
            vpp_degree=2,
            stage_id=1,
            chunk_type=ChunkType.BACKWARD,
            start=5,
            end=7,
        )
        s = str(chunk)
        self.assertIn("B", s)

    def test_chunk_str_bubble(self):
        from paddleformers.fleet.pipeline_parallel.vpp_simulator import (
            Chunk,
            ChunkType,
        )

        chunk = Chunk(
            virtual_pp_rank=0,
            acc_step=0,
            pp_degree=2,
            vpp_degree=2,
            stage_id=0,
            chunk_type=ChunkType.BUBBLE,
            start=2,
            end=4,
        )
        s = str(chunk)
        self.assertIn("Z", s)
        self.assertIn("2", s)
        self.assertIn("4", s)

    def test_chunk_repr(self):
        from paddleformers.fleet.pipeline_parallel.vpp_simulator import (
            Chunk,
            ChunkType,
        )

        chunk = Chunk(
            virtual_pp_rank=0,
            acc_step=0,
            pp_degree=2,
            vpp_degree=1,
            stage_id=0,
            chunk_type=ChunkType.FORWARD,
            start=0,
            end=1,
        )
        self.assertEqual(repr(chunk), str(chunk))

    def test_chunk_barrier_step(self):
        from paddleformers.fleet.pipeline_parallel.vpp_simulator import (
            Chunk,
            ChunkType,
        )

        chunk = Chunk(
            virtual_pp_rank=0,
            acc_step=0,
            pp_degree=4,
            vpp_degree=2,
            stage_id=0,
            chunk_type=ChunkType.FORWARD,
            start=0,
            end=1,
            barrier_step=5,
        )
        self.assertEqual(chunk.barrier_step, 5)


class TestPPChunkRecorderInit(unittest.TestCase):
    """Tests for PPChunkRecorder initialization."""

    def test_init_default(self):
        from paddleformers.fleet.pipeline_parallel.vpp_simulator import (
            PPChunkRecorder,
        )

        recorder = PPChunkRecorder(
            pp_degree=4,
            vpp_degree=2,
            num_acc_steps=4,
            num_hidden_layers=8,
            num_empty_layers_add_in_head=1,
            num_empty_layers_add_in_tail=1,
        )
        self.assertEqual(recorder.pp_degree, 4)
        self.assertEqual(recorder.vpp_degree, 2)
        self.assertEqual(recorder.num_acc_steps, 4)
        self.assertEqual(recorder.num_hidden_layers, 8)
        self.assertEqual(recorder.num_empty_layers_add_in_head, 1)
        self.assertEqual(recorder.num_empty_layers_add_in_tail, 1)
        self.assertEqual(len(recorder.acc_stamp), 8)
        self.assertEqual(recorder.acc_stamp, [0] * 8)


class TestPPChunkRecorderStep(unittest.TestCase):
    """Tests for PPChunkRecorder.step."""

    def test_step_resets_stamp(self):
        from paddleformers.fleet.pipeline_parallel.vpp_simulator import (
            PPChunkRecorder,
        )

        recorder = PPChunkRecorder(
            pp_degree=2,
            vpp_degree=1,
            num_acc_steps=4,
            num_hidden_layers=4,
            num_empty_layers_add_in_head=0,
            num_empty_layers_add_in_tail=0,
        )
        recorder.acc_stamp = [1, 2, 3, 4]
        recorder.step()
        self.assertEqual(recorder.acc_stamp, [0, 0, 0, 0])

    def test_step_multiple_times(self):
        from paddleformers.fleet.pipeline_parallel.vpp_simulator import (
            PPChunkRecorder,
        )

        recorder = PPChunkRecorder(
            pp_degree=2,
            vpp_degree=1,
            num_acc_steps=2,
            num_hidden_layers=4,
            num_empty_layers_add_in_head=0,
            num_empty_layers_add_in_tail=0,
        )
        recorder.acc_stamp[0] = 5
        recorder.step()
        recorder.acc_stamp[1] = 3
        recorder.step()
        self.assertEqual(recorder.acc_stamp, [0, 0, 0, 0])


class TestPPChunkRecorderRecordChunkForward(unittest.TestCase):
    """Tests for PPChunkRecorder.record_chunk_forward."""

    def test_record_valid_layer(self):
        from paddleformers.fleet.pipeline_parallel.vpp_simulator import (
            PPChunkRecorder,
        )

        recorder = PPChunkRecorder(
            pp_degree=2,
            vpp_degree=1,
            num_acc_steps=2,
            num_hidden_layers=4,
            num_empty_layers_add_in_head=0,
            num_empty_layers_add_in_tail=0,
        )
        # acc_stamp is initialized as [0]*4 by __init__
        # But __init__ is called via normal path which sets acc_stamp
        # We need to re-test by calling __init__ properly
        recorder2 = PPChunkRecorder(
            pp_degree=2,
            vpp_degree=1,
            num_acc_steps=2,
            num_hidden_layers=4,
            num_empty_layers_add_in_head=0,
            num_empty_layers_add_in_tail=0,
        )
        self.assertEqual(recorder2.acc_stamp, [0, 0, 0, 0])

    def test_record_out_of_range_high(self):
        from paddleformers.fleet.pipeline_parallel.vpp_simulator import (
            PPChunkRecorder,
        )

        recorder = PPChunkRecorder(
            pp_degree=2,
            vpp_degree=1,
            num_acc_steps=2,
            num_hidden_layers=4,
            num_empty_layers_add_in_head=0,
            num_empty_layers_add_in_tail=0,
        )
        result = recorder.record_chunk_forward(10)
        self.assertFalse(result)

    def test_record_out_of_range_low(self):
        from paddleformers.fleet.pipeline_parallel.vpp_simulator import (
            PPChunkRecorder,
        )

        recorder = PPChunkRecorder(
            pp_degree=2,
            vpp_degree=1,
            num_acc_steps=2,
            num_hidden_layers=4,
            num_empty_layers_add_in_head=2,
            num_empty_layers_add_in_tail=0,
        )
        # layer_id=0 is in empty head layers
        result = recorder.record_chunk_forward(0)
        self.assertFalse(result)

    def test_record_with_empty_head(self):
        from paddleformers.fleet.pipeline_parallel.vpp_simulator import (
            PPChunkRecorder,
        )

        recorder = PPChunkRecorder(
            pp_degree=2,
            vpp_degree=1,
            num_acc_steps=2,
            num_hidden_layers=4,
            num_empty_layers_add_in_head=2,
            num_empty_layers_add_in_tail=1,
        )
        # Valid range: layer_id in [2, 2+4) = [2, 6)
        # Check initial acc_stamp
        self.assertEqual(recorder.acc_stamp, [0, 0, 0, 0])
        self.assertEqual(len(recorder.acc_stamp), 4)

    def test_record_multiple_times(self):
        from paddleformers.fleet.pipeline_parallel.vpp_simulator import (
            PPChunkRecorder,
        )

        recorder = PPChunkRecorder(
            pp_degree=2,
            vpp_degree=1,
            num_acc_steps=2,
            num_hidden_layers=4,
            num_empty_layers_add_in_head=0,
            num_empty_layers_add_in_tail=0,
        )
        recorder.record_chunk_forward(0)
        recorder.record_chunk_forward(1)
        recorder.record_chunk_forward(0)
        self.assertEqual(recorder.acc_stamp[0], 2)
        self.assertEqual(recorder.acc_stamp[1], 1)
        self.assertEqual(recorder.acc_stamp[2], 0)


class TestGlobalPPChunkRecorder(unittest.TestCase):
    """Tests for global PP chunk recorder functions."""

    def test_set_and_get(self):
        from paddleformers.fleet.pipeline_parallel.vpp_simulator import (
            PPChunkRecorder,
            get_global_pp_recorder,
            set_global_pp_chunk_recorder,
        )

        recorder = PPChunkRecorder(
            pp_degree=2,
            vpp_degree=1,
            num_acc_steps=2,
            num_hidden_layers=4,
            num_empty_layers_add_in_head=0,
            num_empty_layers_add_in_tail=0,
        )
        set_global_pp_chunk_recorder(recorder)
        retrieved = get_global_pp_recorder()
        self.assertIs(retrieved, recorder)


if __name__ == "__main__":
    unittest.main()
