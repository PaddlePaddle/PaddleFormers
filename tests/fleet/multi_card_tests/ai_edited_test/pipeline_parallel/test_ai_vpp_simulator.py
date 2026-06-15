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
import unittest
from unittest.mock import MagicMock

import numpy as np
import paddle
from paddle.distributed import fleet

# vpp_simulator.py imports matplotlib which may not be installed in CI
sys.modules["matplotlib"] = MagicMock()
sys.modules["matplotlib.pyplot"] = MagicMock()

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)

from paddleformers.fleet.pipeline_parallel.vpp_simulator import (
    PPChunkRecorder,
    VPPSimulator,
)
from paddleformers.fleet.training.initialize import initialize_fleet

PP_DEGREE = 4


def _init_pp():
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": 1,
        "pp_degree": PP_DEGREE,
        "sharding_degree": 1,
        "sep_degree": 1,
        "cp_degree": 1,
        "ep_degree": 1,
        "moe_sharding_degree": 1,
        "order": [
            "sharding",
            "moe_sharding",
            "pp",
            "sep",
            "cp",
            "dp",
            "ep",
            "mp",
        ],
    }
    initialize_fleet(strategy)


def setUpModule():
    """Initialize fleet once for all tests in this module (PP=4)."""
    _init_pp()
    np.random.seed(42)
    paddle.seed(42)


class TestVPPSimulatorBasicScheduling(unittest.TestCase):
    """Test VPPSimulator creates a schedule_table correctly."""

    def test_vpp_simulator_basic_scheduling(self):
        """Create VPPSimulator and verify schedule_table is populated."""
        pp_degree = 4
        vpp_degree = 2
        num_acc_steps = 4
        simulator = VPPSimulator(
            pp_degree=pp_degree,
            vpp_degree=vpp_degree,
            num_acc_steps=num_acc_steps,
        )
        # Trigger scheduling
        schedule_table = simulator.schedule()
        # schedule_table should have pp_degree entries
        self.assertEqual(len(schedule_table), pp_degree)
        # Each stage should have scheduled chunks
        for stage_chunks in schedule_table:
            self.assertGreater(len(stage_chunks), 0)

        # Verify internal state
        self.assertEqual(simulator.pp_degree, pp_degree)
        self.assertEqual(simulator.vpp_degree, vpp_degree)
        self.assertEqual(simulator.num_acc_steps, num_acc_steps)
        self.assertTrue(simulator._is_scheduled)


class TestVPPSimulatorBubbleRate(unittest.TestCase):
    """Test VPPSimulator.compute_bubble_rate returns a value between 0 and 1."""

    def test_vpp_simulator_bubble_rate(self):
        """compute_bubble_rate should return a float in [0, 1]."""
        simulator = VPPSimulator(
            pp_degree=4,
            vpp_degree=2,
            num_acc_steps=4,
        )
        bubble_rate = simulator.compute_bubble_rate()
        self.assertIsInstance(bubble_rate, float)
        self.assertGreaterEqual(bubble_rate, 0.0)
        self.assertLessEqual(bubble_rate, 1.0)

    def test_vpp_simulator_bubble_rate_different_configs(self):
        """Bubble rate should be in [0, 1] for various configurations."""
        for vpp_degree in [1, 2]:
            for num_acc_steps in [4, 8]:
                simulator = VPPSimulator(
                    pp_degree=4,
                    vpp_degree=vpp_degree,
                    num_acc_steps=num_acc_steps,
                )
                bubble_rate = simulator.compute_bubble_rate()
                self.assertGreaterEqual(bubble_rate, 0.0)
                self.assertLessEqual(bubble_rate, 1.0)


class TestVPPSimulatorWarmupSteps(unittest.TestCase):
    """Test VPPSimulator._get_warmup_and_steady_steps returns non-negative values."""

    def test_vpp_simulator_warmup_steps(self):
        """_get_warmup_and_steady_steps should return non-negative warmup_steps."""
        simulator = VPPSimulator(
            pp_degree=4,
            vpp_degree=2,
            num_acc_steps=4,
        )
        for stage_id in range(simulator.pp_degree):
            warmup_steps, steady_steps = simulator._get_warmup_and_steady_steps(stage_id)
            self.assertGreaterEqual(warmup_steps, 0)
            self.assertGreaterEqual(steady_steps, 0)


class TestPPChunkRecorder(unittest.TestCase):
    """Test PPChunkRecorder creation and basic usage."""

    def test_pp_chunk_recorder_creation(self):
        """PPChunkRecorder should be created with correct configuration."""
        recorder = PPChunkRecorder(
            pp_degree=4,
            vpp_degree=2,
            num_acc_steps=4,
            num_hidden_layers=8,
            num_empty_layers_add_in_head=0,
            num_empty_layers_add_in_tail=0,
        )
        self.assertEqual(recorder.pp_degree, 4)
        self.assertEqual(recorder.vpp_degree, 2)
        self.assertEqual(recorder.num_acc_steps, 4)
        self.assertEqual(recorder.num_hidden_layers, 8)
        self.assertEqual(recorder.num_empty_layers_add_in_head, 0)
        self.assertEqual(recorder.num_empty_layers_add_in_tail, 0)

    def test_pp_chunk_recorder_initial_acc_stamp(self):
        """Initial acc_stamp should be all zeros."""
        recorder = PPChunkRecorder(
            pp_degree=4,
            vpp_degree=2,
            num_acc_steps=4,
            num_hidden_layers=8,
            num_empty_layers_add_in_head=0,
            num_empty_layers_add_in_tail=0,
        )
        self.assertEqual(len(recorder.acc_stamp), 8)
        for stamp in recorder.acc_stamp:
            self.assertEqual(stamp, 0)

    def test_pp_chunk_recorder_record_chunk_forward(self):
        """record_chunk_forward should increment acc_stamp for valid layer."""
        recorder = PPChunkRecorder(
            pp_degree=4,
            vpp_degree=2,
            num_acc_steps=4,
            num_hidden_layers=8,
            num_empty_layers_add_in_head=0,
            num_empty_layers_add_in_tail=0,
        )
        # Record forward for layer 3 (valid: 0 <= 3 < 8)
        recorder.record_chunk_forward(3)
        # record_chunk_forward returns None for valid layers (not True/False)
        self.assertEqual(recorder.acc_stamp[3], 1)

    def test_pp_chunk_recorder_record_chunk_forward_out_of_range(self):
        """record_chunk_forward should return False for out-of-range layer."""
        recorder = PPChunkRecorder(
            pp_degree=4,
            vpp_degree=2,
            num_acc_steps=4,
            num_hidden_layers=8,
            num_empty_layers_add_in_head=0,
            num_empty_layers_add_in_tail=0,
        )
        # Layer 8 is out of range (0..7)
        result = recorder.record_chunk_forward(8)
        self.assertFalse(result)
        # acc_stamp should not change
        self.assertEqual(recorder.acc_stamp[7], 0)

    def test_pp_chunk_recorder_step_reset(self):
        """step() should reset all acc_stamp values to zero."""
        recorder = PPChunkRecorder(
            pp_degree=4,
            vpp_degree=2,
            num_acc_steps=4,
            num_hidden_layers=8,
            num_empty_layers_add_in_head=0,
            num_empty_layers_add_in_tail=0,
        )
        recorder.record_chunk_forward(0)
        recorder.record_chunk_forward(1)
        recorder.record_chunk_forward(2)
        self.assertEqual(recorder.acc_stamp[0], 1)
        self.assertEqual(recorder.acc_stamp[1], 1)
        self.assertEqual(recorder.acc_stamp[2], 1)
        recorder.step()
        for stamp in recorder.acc_stamp:
            self.assertEqual(stamp, 0)

    def test_pp_chunk_recorder_with_empty_layers(self):
        """PPChunkRecorder should handle empty layers in head and tail."""
        recorder = PPChunkRecorder(
            pp_degree=4,
            vpp_degree=2,
            num_acc_steps=4,
            num_hidden_layers=8,
            num_empty_layers_add_in_head=2,
            num_empty_layers_add_in_tail=1,
        )
        # Layer 0 and 1 are empty head layers, should return False
        self.assertFalse(recorder.record_chunk_forward(0))
        self.assertFalse(recorder.record_chunk_forward(1))
        # Layer 2..9 are real layers (0..7 after offset), valid range
        recorder.record_chunk_forward(2)
        recorder.record_chunk_forward(9)
        self.assertEqual(recorder.acc_stamp[0], 1)
        self.assertEqual(recorder.acc_stamp[7], 1)
        # Layer 10 is out of range (empty tail)
        self.assertFalse(recorder.record_chunk_forward(10))


class TestVPPSimulatorLayerNum(unittest.TestCase):
    """Test VPPSimulator computes layer_num correctly."""

    def test_vpp_simulator_layer_num(self):
        """layer_num should equal pp_degree * vpp_degree."""
        simulator = VPPSimulator(pp_degree=4, vpp_degree=2, num_acc_steps=4)
        self.assertEqual(simulator.layer_num, 8)

    def test_vpp_simulator_layer_num_vpp1(self):
        """With vpp_degree=1, layer_num should equal pp_degree."""
        simulator = VPPSimulator(pp_degree=4, vpp_degree=1, num_acc_steps=4)
        self.assertEqual(simulator.layer_num, 4)


if __name__ == "__main__":
    unittest.main()
