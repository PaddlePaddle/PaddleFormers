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


# Tests for src/paddleformers.fleet/recompute_utils.py

import unittest


class _MockConfig:
    """Mock config object for recompute tests."""

    def __init__(
        self,
        num_hidden_layers=8,
        num_empty_layers_add_in_head=0,
        num_empty_layers_add_in_tail=0,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
    ):
        self.num_hidden_layers = num_hidden_layers
        self.num_empty_layers_add_in_head = num_empty_layers_add_in_head
        self.num_empty_layers_add_in_tail = num_empty_layers_add_in_tail
        self.pipeline_model_parallel_size = pipeline_model_parallel_size
        self.virtual_pipeline_model_parallel_size = virtual_pipeline_model_parallel_size
        self.recompute_granularity = recompute_granularity
        self.recompute_method = recompute_method
        self.recompute_num_layers = recompute_num_layers


class TestNeedRecomputeInBlock(unittest.TestCase):
    """Tests for need_recompute_in_block."""

    def test_recompute_all_negative(self):
        """Test with negative recompute_num_layers recomputes all."""
        from paddleformers.fleet.recompute_utils import need_recompute_in_block

        config = _MockConfig(num_hidden_layers=8, pipeline_model_parallel_size=1)
        self.assertTrue(need_recompute_in_block(0, config, -1))
        self.assertTrue(need_recompute_in_block(7, config, -1))

    def test_recompute_block_basic(self):
        """Test basic block recompute."""
        from paddleformers.fleet.recompute_utils import need_recompute_in_block

        config = _MockConfig(num_hidden_layers=8, pipeline_model_parallel_size=1)
        # chunk_size=8, recompute first 4 layers
        self.assertTrue(need_recompute_in_block(0, config, 4))
        self.assertTrue(need_recompute_in_block(3, config, 4))
        self.assertFalse(need_recompute_in_block(4, config, 4))
        self.assertFalse(need_recompute_in_block(7, config, 4))

    def test_recompute_block_with_pp(self):
        """Test block recompute with pipeline parallelism."""
        from paddleformers.fleet.recompute_utils import need_recompute_in_block

        config = _MockConfig(num_hidden_layers=8, pipeline_model_parallel_size=2)
        # chunk_size=4, recompute first 2 layers per chunk
        self.assertTrue(need_recompute_in_block(0, config, 2))
        self.assertTrue(need_recompute_in_block(1, config, 2))
        self.assertFalse(need_recompute_in_block(2, config, 2))
        self.assertFalse(need_recompute_in_block(3, config, 2))
        self.assertTrue(need_recompute_in_block(4, config, 2))
        self.assertTrue(need_recompute_in_block(5, config, 2))
        self.assertFalse(need_recompute_in_block(6, config, 2))
        self.assertFalse(need_recompute_in_block(7, config, 2))

    def test_recompute_block_with_vpp(self):
        """Test block recompute with virtual pipeline parallelism."""
        from paddleformers.fleet.recompute_utils import need_recompute_in_block

        config = _MockConfig(
            num_hidden_layers=8,
            pipeline_model_parallel_size=2,
            virtual_pipeline_model_parallel_size=2,
        )
        # parallel_size=2*2=4, chunk_size=2
        self.assertTrue(need_recompute_in_block(0, config, 1))
        self.assertTrue(need_recompute_in_block(2, config, 1))
        self.assertTrue(need_recompute_in_block(4, config, 1))
        self.assertTrue(need_recompute_in_block(6, config, 1))

    def test_recompute_block_none_raises(self):
        """Test that None recompute_num_layers raises."""
        from paddleformers.fleet.recompute_utils import need_recompute_in_block

        config = _MockConfig()
        with self.assertRaises(AssertionError):
            need_recompute_in_block(0, config, None)

    def test_recompute_block_exceeds_chunk_raises(self):
        """Test that recompute_num_layers > chunk_size raises."""
        from paddleformers.fleet.recompute_utils import need_recompute_in_block

        config = _MockConfig(num_hidden_layers=8, pipeline_model_parallel_size=1)
        # chunk_size=8
        with self.assertRaises(AssertionError):
            need_recompute_in_block(0, config, 9)


class TestNeedRecomputeInFirstN(unittest.TestCase):
    """Tests for need_recompute_in_first_n."""

    def test_recompute_first_n_basic(self):
        """Test basic first_n recompute."""
        from paddleformers.fleet.recompute_utils import need_recompute_in_first_n

        config = _MockConfig(
            num_hidden_layers=8,
            recompute_method="first_n",
            recompute_num_layers=4,
        )
        self.assertTrue(need_recompute_in_first_n(0, config, 4))
        self.assertTrue(need_recompute_in_first_n(3, config, 4))
        self.assertFalse(need_recompute_in_first_n(4, config, 4))
        self.assertFalse(need_recompute_in_first_n(7, config, 4))

    def test_recompute_first_n_with_pp(self):
        """Test first_n recompute with pipeline parallelism."""
        from paddleformers.fleet.recompute_utils import need_recompute_in_first_n

        config = _MockConfig(
            num_hidden_layers=8,
            pipeline_model_parallel_size=2,
            recompute_method="first_n",
            recompute_num_layers=2,
        )
        # Each stage has 4 layers, recompute first 2 in each stage
        self.assertTrue(need_recompute_in_first_n(0, config, 2))
        self.assertTrue(need_recompute_in_first_n(1, config, 2))
        self.assertFalse(need_recompute_in_first_n(2, config, 2))
        self.assertFalse(need_recompute_in_first_n(3, config, 2))
        self.assertTrue(need_recompute_in_first_n(4, config, 2))
        self.assertTrue(need_recompute_in_first_n(5, config, 2))
        self.assertFalse(need_recompute_in_first_n(6, config, 2))
        self.assertFalse(need_recompute_in_first_n(7, config, 2))

    def test_recompute_first_n_with_vpp(self):
        """Test first_n recompute with VPP."""
        from paddleformers.fleet.recompute_utils import need_recompute_in_first_n

        config = _MockConfig(
            num_hidden_layers=8,
            pipeline_model_parallel_size=2,
            virtual_pipeline_model_parallel_size=2,
            recompute_method="first_n",
            recompute_num_layers=1,
        )
        # Should recompute layer 0 and 4 (first in each stage)
        self.assertTrue(need_recompute_in_first_n(0, config, 1))
        self.assertFalse(need_recompute_in_first_n(1, config, 1))
        self.assertFalse(need_recompute_in_first_n(2, config, 1))
        self.assertFalse(need_recompute_in_first_n(3, config, 1))
        self.assertTrue(need_recompute_in_first_n(4, config, 1))
        self.assertFalse(need_recompute_in_first_n(5, config, 1))
        self.assertFalse(need_recompute_in_first_n(6, config, 1))
        self.assertFalse(need_recompute_in_first_n(7, config, 1))

    def test_recompute_first_n_none_raises(self):
        """Test that None recompute_num_layers raises."""
        from paddleformers.fleet.recompute_utils import need_recompute_in_first_n

        config = _MockConfig()
        with self.assertRaises(AssertionError):
            need_recompute_in_first_n(0, config, None)


class TestNeedFullRecompute(unittest.TestCase):
    """Tests for need_full_recompute."""

    def test_full_recompute_uniform(self):
        """Test uniform recompute always returns True."""
        from paddleformers.fleet.recompute_utils import need_full_recompute

        config = _MockConfig(
            recompute_granularity="full",
            recompute_method="uniform",
            recompute_num_layers=1,
        )
        self.assertTrue(need_full_recompute(0, config))
        self.assertTrue(need_full_recompute(5, config))

    def test_full_recompute_not_full_granularity(self):
        """Test non-full granularity returns False."""
        from paddleformers.fleet.recompute_utils import need_full_recompute

        config = _MockConfig(
            recompute_granularity="selective",
            recompute_method="uniform",
            recompute_num_layers=1,
        )
        self.assertFalse(need_full_recompute(0, config))

    def test_full_recompute_first_n(self):
        """Test first_n recompute."""
        from paddleformers.fleet.recompute_utils import need_full_recompute

        config = _MockConfig(
            recompute_granularity="full",
            recompute_method="first_n",
            recompute_num_layers=4,
        )
        self.assertTrue(need_full_recompute(0, config))
        self.assertTrue(need_full_recompute(3, config))
        self.assertFalse(need_full_recompute(4, config))
        self.assertFalse(need_full_recompute(7, config))

    def test_full_recompute_block(self):
        """Test block recompute."""
        from paddleformers.fleet.recompute_utils import need_full_recompute

        config = _MockConfig(
            recompute_granularity="full",
            recompute_method="block",
            recompute_num_layers=4,
        )
        self.assertTrue(need_full_recompute(0, config))
        self.assertTrue(need_full_recompute(3, config))
        self.assertFalse(need_full_recompute(4, config))
        self.assertFalse(need_full_recompute(7, config))

    def test_full_recompute_uniform_non_one_raises(self):
        """Test uniform with recompute_num_layers != 1 raises."""
        from paddleformers.fleet.recompute_utils import need_full_recompute

        config = _MockConfig(
            recompute_granularity="full",
            recompute_method="uniform",
            recompute_num_layers=2,
        )
        with self.assertRaises(AssertionError):
            need_full_recompute(0, config)

    def test_full_recompute_empty_layers(self):
        """Test with empty layers added in head and tail."""
        from paddleformers.fleet.recompute_utils import need_full_recompute

        config = _MockConfig(
            num_hidden_layers=6,
            num_empty_layers_add_in_head=1,
            num_empty_layers_add_in_tail=2,
            pipeline_model_parallel_size=1,
            recompute_granularity="full",
            recompute_method="block",
            recompute_num_layers=2,
        )
        # total=1+6+2=9 layers
        self.assertTrue(need_full_recompute(0, config))
        self.assertTrue(need_full_recompute(1, config))


if __name__ == "__main__":
    unittest.main()
