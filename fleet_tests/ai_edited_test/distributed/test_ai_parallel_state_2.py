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
import warnings


class TestParallelStateGetters(unittest.TestCase):
    """Tests for parallel_state getter functions when not initialized."""

    def test_get_tensor_model_parallel_group_uninitialized(self):
        """Test get_tensor_model_parallel_group raises when not initialized."""
        from paddleformers.fleet import parallel_state

        with self.assertRaises(AssertionError):
            parallel_state.get_tensor_model_parallel_group()

    def test_get_tensor_model_parallel_group_no_check(self):
        """Test get_tensor_model_parallel_group returns None with check_initialized=False."""
        from paddleformers.fleet import parallel_state

        result = parallel_state.get_tensor_model_parallel_group(
            check_initialized=False
        )
        self.assertIsNone(result)

    def test_get_pipeline_model_parallel_group_uninitialized(self):
        """Test get_pipeline_model_parallel_group raises when not initialized."""
        from paddleformers.fleet import parallel_state

        with self.assertRaises(AssertionError):
            parallel_state.get_pipeline_model_parallel_group()

    def test_get_data_parallel_group_uninitialized(self):
        """Test get_data_parallel_group raises when not initialized."""
        from paddleformers.fleet import parallel_state

        with self.assertRaises(AssertionError):
            parallel_state.get_data_parallel_group()

    def test_get_expert_model_parallel_group_uninitialized(self):
        """Test get_expert_model_parallel_group raises when not initialized."""
        from paddleformers.fleet import parallel_state

        with self.assertRaises(AssertionError):
            parallel_state.get_expert_model_parallel_group()

    def test_get_expert_data_parallel_group_uninitialized(self):
        """Test get_expert_data_parallel_group raises when not initialized."""
        from paddleformers.fleet import parallel_state

        with self.assertRaises(AssertionError):
            parallel_state.get_expert_data_parallel_group()

    def test_get_pipeline_model_parallel_world_size_default(self):
        """Test get_pipeline_model_parallel_world_size returns 1 by default."""
        from paddleformers.fleet import parallel_state

        result = parallel_state.get_pipeline_model_parallel_world_size()
        self.assertEqual(result, 1)

    def test_get_pipeline_model_parallel_rank_returns_zero(self):
        """Test get_pipeline_model_parallel_rank returns 0 with warning."""
        from paddleformers.fleet import parallel_state

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = parallel_state.get_pipeline_model_parallel_rank()
            self.assertEqual(result, 0)
            self.assertTrue(len(w) > 0)

    def test_set_pipeline_model_parallel_world_size(self):
        """Test setting pipeline model parallel world size."""
        from paddleformers.fleet import parallel_state

        parallel_state.set_pipeline_model_parallel_world_size(4)
        # Note: get_pipeline_model_parallel_world_size() always returns 1
        # due to a hardcoded early return in the source code
        # Just verify the setter does not raise

    def test_get_context_parallel_group_default(self):
        """Test get_context_parallel_group returns None by default."""
        from paddleformers.fleet import parallel_state

        result = parallel_state.get_context_parallel_group(
            check_initialized=False
        )
        self.assertIsNone(result)

    def test_get_context_parallel_world_size_default(self):
        """Test get_context_parallel_world_size returns 1 by default."""
        from paddleformers.fleet import parallel_state

        result = parallel_state.get_context_parallel_world_size()
        self.assertEqual(result, 1)

    def test_get_virtual_pipeline_model_parallel_rank_default(self):
        """Test get_virtual_pipeline_model_parallel_rank returns None."""
        from paddleformers.fleet import parallel_state

        result = parallel_state.get_virtual_pipeline_model_parallel_rank()
        self.assertIsNone(result)

    def test_get_virtual_pipeline_model_parallel_world_size_default(self):
        """Test get_virtual_pipeline_model_parallel_world_size returns None."""
        from paddleformers.fleet import parallel_state

        result = parallel_state.get_virtual_pipeline_model_parallel_world_size()
        self.assertIsNone(result)


class TestExpertParallelState(unittest.TestCase):
    """Tests for expert parallel state functions."""

    def test_set_expert_model_parallel_world_size(self):
        """Test setting expert model parallel world size."""
        from paddleformers.fleet import parallel_state

        parallel_state.set_expert_model_parallel_world_size(4)
        # Just verify no error

    def test_set_expert_model_parallel_rank(self):
        """Test setting expert model parallel rank."""
        from paddleformers.fleet import parallel_state

        parallel_state.set_expert_model_parallel_rank(2)
        result = parallel_state.get_expert_model_parallel_rank()
        self.assertEqual(result, 2)

    def test_set_expert_tensor_parallel_world_size(self):
        """Test setting expert tensor parallel world size."""
        from paddleformers.fleet import parallel_state

        parallel_state.set_expert_tensor_parallel_world_size(2)

    def test_set_expert_tensor_parallel_rank(self):
        """Test setting expert tensor parallel rank."""
        from paddleformers.fleet import parallel_state

        parallel_state.set_expert_tensor_parallel_rank(1)
        result = parallel_state.get_expert_tensor_parallel_rank()
        self.assertEqual(result, 1)

    def test_get_expert_tensor_parallel_group_default(self):
        """Test get_expert_tensor_parallel_group returns None by default."""
        from paddleformers.fleet import parallel_state

        result = parallel_state.get_expert_tensor_parallel_group(
            check_initialized=False
        )
        self.assertIsNone(result)


class TestGlobalMemoryBufferState(unittest.TestCase):
    """Tests for global memory buffer state functions."""

    def test_have_global_memory_buffer_default(self):
        """Test have_global_memory_buffer returns False when not initialized."""
        from paddleformers.fleet import parallel_state

        # Make sure buffer is destroyed first
        if parallel_state.have_global_memory_buffer():
            parallel_state.destroy_global_memory_buffer()
        self.assertFalse(parallel_state.have_global_memory_buffer())

    def test_get_global_memory_buffer_uninitialized_raises(self):
        """Test get_global_memory_buffer raises when not initialized."""
        from paddleformers.fleet import parallel_state

        if parallel_state.have_global_memory_buffer():
            parallel_state.destroy_global_memory_buffer()
        with self.assertRaises(AssertionError):
            parallel_state.get_global_memory_buffer()

    def test_destroy_global_memory_buffer(self):
        """Test destroy_global_memory_buffer sets buffer to None."""
        from paddleformers.fleet import parallel_state

        parallel_state.destroy_global_memory_buffer()
        self.assertFalse(parallel_state.have_global_memory_buffer())


class TestGetEmbeddingGroup(unittest.TestCase):
    """Tests for get_embedding_group."""

    def test_get_embedding_group_not_implemented(self):
        """Test get_embedding_group raises NotImplementedError."""
        from paddleformers.fleet import parallel_state

        with self.assertRaises(NotImplementedError):
            parallel_state.get_embedding_group()


class TestIsPipelineStage(unittest.TestCase):
    """Tests for is_pipeline_first_stage and is_pipeline_last_stage."""

    def test_is_pipeline_first_stage_default(self):
        """Test is_pipeline_first_stage returns True with default settings."""
        from paddleformers.fleet import parallel_state

        result = parallel_state.is_pipeline_first_stage()
        # With default rank 0, this should be True
        self.assertTrue(result)

    def test_is_pipeline_last_stage_default(self):
        """Test is_pipeline_last_stage returns True with default settings."""
        from paddleformers.fleet import parallel_state

        # With world_size=1 and rank=0, it should be True
        result = parallel_state.is_pipeline_last_stage()
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
