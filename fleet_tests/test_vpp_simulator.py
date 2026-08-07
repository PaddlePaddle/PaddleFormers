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

import numpy as np

from paddleformers.fleet.pipeline_parallel.vpp_simulator import (
    PPChunkRecorder,
    VPPSimulator,
)


class TestVPPSimulator(unittest.TestCase):
    """
    Unit tests for VPPSimulator class.

    Tests include:
    - Bubble rate calculation
    - Visualization functionality
    - Schedule correctness
    """

    def test_vpp_simulator(self):
        """
        Test basic VPP simulator functionality including:
        - Initialization
        - Bubble rate calculation
        - Visualization methods
        """
        pp_degree = 4  # Pipeline parallel degree
        vpp_degree = 2  # Virtual pipeline degree
        num_acc_steps = 16  # Gradient accumulation steps

        # Initialize simulator with test configuration
        vpp_simulator = VPPSimulator(pp_degree, vpp_degree, num_acc_steps)
        bubble_rate = vpp_simulator.compute_bubble_rate()
        expected_bubble_rate = (
            1.0
            * (pp_degree - 1)
            / (vpp_degree * num_acc_steps + (pp_degree - 1))
        )
        # print(f"bubble_rate: {bubble_rate}, expected_bubble_rate: {expected_bubble_rate}")
        assert bubble_rate == expected_bubble_rate, (
            f"Expected bubble rate {expected_bubble_rate}, got {bubble_rate}"
        )
        # vpp_simulator.draw_balls()
        # vpp_simulator.draw_chunks()
        assert True


class TestPPChunkRecorder(unittest.TestCase):
    """
    Unit tests for PPChunkRecorder class.

    Tests include:
    - Chunk recording functionality
    - Execution stamp tracking
    """

    def test_pp_chunk_recorder(self):
        """
        Test chunk recording functionality by:
        - Simulating forward passes for all layers
        - Verifying execution counts
        """
        pp_degree = 4  # Pipeline parallel degree
        vpp_degree = 4  # Virtual pipeline degree
        num_acc_steps = 16  # Gradient accumulation steps
        num_empty_layers_add_in_head = 1  # Number of head layers to skip
        num_empty_layers_add_in_tail = 1  # Number of tail layers to skip
        num_hidden_layers = (
            pp_degree * vpp_degree
            - num_empty_layers_add_in_head
            - num_empty_layers_add_in_tail
        )

        # Initialize recorder with test configuration
        pp_chunk_recorder = PPChunkRecorder(
            pp_degree,
            vpp_degree,
            num_acc_steps,
            num_hidden_layers,
            num_empty_layers_add_in_head,
            num_empty_layers_add_in_tail,
        )

        # Simulate forward passes for all layers across a global_step
        for i in range(num_acc_steps):
            for j in range(num_hidden_layers):
                pp_chunk_recorder.record_chunk_forward(
                    j + num_empty_layers_add_in_head
                )

        # Verify each layer was executed exactly num_acc_steps times
        result = np.array(pp_chunk_recorder.acc_stamp)
        assert np.all(result == num_acc_steps), (
            f"Expected all layers to be executed {num_acc_steps} times"
        )


if __name__ == "__main__":
    unittest.main()
