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


# Tests for src/paddleformers.fleet/parallel_state.py
# Additional tests for expert parallel state functions, pipeline helpers,
# context parallel, memory buffer, and embedding group

import unittest
import warnings
from unittest import mock


class TestExpertModelParallelState(unittest.TestCase):
    """Tests for expert model parallel state functions."""

    def test_get_expert_model_parallel_rank_not_initialized(self):
        """Test get_expert_model_parallel_rank returns 0 when not initialized."""
        import paddleformers.fleet.parallel_state as ps
        from paddleformers.fleet.parallel_state import get_expert_model_parallel_rank

        original = ps._MPU_EXPERT_MODEL_PARALLEL_RANK
        ps._MPU_EXPERT_MODEL_PARALLEL_RANK = None

        with mock.patch("paddle.distributed.is_initialized", return_value=False):
            rank = get_expert_model_parallel_rank()
            self.assertEqual(rank, 0)

        ps._MPU_EXPERT_MODEL_PARALLEL_RANK = original

    def test_get_expert_model_parallel_rank_set(self):
        """Test get_expert_model_parallel_rank returns set value."""
        from paddleformers.fleet.parallel_state import (
            get_expert_model_parallel_rank,
            set_expert_model_parallel_rank,
        )

        set_expert_model_parallel_rank(5)
        self.assertEqual(get_expert_model_parallel_rank(), 5)

    def test_set_expert_model_parallel_world_size(self):
        """Test set_expert_model_parallel_world_size."""
        import paddleformers.fleet.parallel_state as ps
        from paddleformers.fleet.parallel_state import (
            set_expert_model_parallel_world_size,
        )

        set_expert_model_parallel_world_size(8)
        # The getter get_expert_model_parallel_world_size may not exist
        # in all versions, so we verify via the internal MPU variable.
        self.assertEqual(ps._MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE, 8)

    def test_get_expert_model_parallel_group_not_initialized(self):
        """Test get_expert_model_parallel_group raises when not initialized."""
        import paddleformers.fleet.parallel_state as ps
        from paddleformers.fleet.parallel_state import get_expert_model_parallel_group

        original = ps._EXPERT_MODEL_PARALLEL_GROUP
        ps._EXPERT_MODEL_PARALLEL_GROUP = None

        with self.assertRaises(AssertionError):
            get_expert_model_parallel_group(check_initialized=True)

        ps._EXPERT_MODEL_PARALLEL_GROUP = original

    def test_get_expert_data_parallel_group_not_initialized(self):
        """Test get_expert_data_parallel_group raises when not initialized."""
        import paddleformers.fleet.parallel_state as ps
        from paddleformers.fleet.parallel_state import get_expert_data_parallel_group

        original = ps._EXPERT_DATA_PARALLEL_GROUP
        ps._EXPERT_DATA_PARALLEL_GROUP = None

        with self.assertRaises(AssertionError):
            get_expert_data_parallel_group(check_initialized=True)

        ps._EXPERT_DATA_PARALLEL_GROUP = original


class TestExpertTensorParallelState(unittest.TestCase):
    """Tests for expert tensor parallel state functions."""

    def test_set_expert_tensor_parallel_world_size(self):
        """Test set_expert_tensor_parallel_world_size."""
        import paddleformers.fleet.parallel_state as ps
        from paddleformers.fleet.parallel_state import (
            set_expert_tensor_parallel_world_size,
        )

        set_expert_tensor_parallel_world_size(4)
        self.assertEqual(ps._MPU_EXPERT_TENSOR_PARALLEL_WORLD_SIZE, 4)

    def test_set_expert_tensor_parallel_rank(self):
        """Test set_expert_tensor_parallel_rank."""
        import paddleformers.fleet.parallel_state as ps
        from paddleformers.fleet.parallel_state import set_expert_tensor_parallel_rank

        set_expert_tensor_parallel_rank(3)
        self.assertEqual(ps._MPU_EXPERT_TENSOR_PARALLEL_RANK, 3)

    def test_get_expert_tensor_parallel_rank_not_initialized(self):
        """Test get_expert_tensor_parallel_rank returns 0 when not initialized."""
        import paddleformers.fleet.parallel_state as ps
        from paddleformers.fleet.parallel_state import get_expert_tensor_parallel_rank

        original = ps._MPU_EXPERT_TENSOR_PARALLEL_RANK
        original_group = ps._EXPERT_TENSOR_PARALLEL_GROUP
        ps._MPU_EXPERT_TENSOR_PARALLEL_RANK = None
        ps._EXPERT_TENSOR_PARALLEL_GROUP = None

        rank = get_expert_tensor_parallel_rank()
        self.assertEqual(rank, 0)

        ps._MPU_EXPERT_TENSOR_PARALLEL_RANK = original
        ps._EXPERT_TENSOR_PARALLEL_GROUP = original_group


class TestContextParallelState(unittest.TestCase):
    """Tests for context parallel state functions."""

    def test_get_context_parallel_group_none(self):
        """Test get_context_parallel_group returns None when not initialized."""
        import paddleformers.fleet.parallel_state as ps
        from paddleformers.fleet.parallel_state import get_context_parallel_group

        original = ps._CONTEXT_PARALLEL_GROUP
        ps._CONTEXT_PARALLEL_GROUP = None

        result = get_context_parallel_group(check_initialized=False)
        self.assertIsNone(result)

        ps._CONTEXT_PARALLEL_GROUP = original

    def test_get_context_parallel_group_raises(self):
        """Test get_context_parallel_group raises when check_initialized=True."""
        import paddleformers.fleet.parallel_state as ps
        from paddleformers.fleet.parallel_state import get_context_parallel_group

        original = ps._CONTEXT_PARALLEL_GROUP
        ps._CONTEXT_PARALLEL_GROUP = None

        with self.assertRaises(AssertionError):
            get_context_parallel_group(check_initialized=True)

        ps._CONTEXT_PARALLEL_GROUP = original

    def test_get_context_parallel_world_size_one_when_none(self):
        """Test get_context_parallel_world_size returns 1 when group is None."""
        import paddleformers.fleet.parallel_state as ps
        from paddleformers.fleet.parallel_state import get_context_parallel_world_size

        original = ps._CONTEXT_PARALLEL_GROUP
        ps._CONTEXT_PARALLEL_GROUP = None

        size = get_context_parallel_world_size()
        self.assertEqual(size, 1)

        ps._CONTEXT_PARALLEL_GROUP = original


class TestPipelineHelpers(unittest.TestCase):
    """Tests for pipeline parallel helper functions."""

    def test_is_pipeline_first_stage(self):
        """Test is_pipeline_first_stage returns True when rank is 0."""
        from paddleformers.fleet.parallel_state import is_pipeline_first_stage

        # get_pipeline_model_parallel_rank always returns 0
        result = is_pipeline_first_stage()
        self.assertTrue(result)

    def test_is_pipeline_last_stage_world_size_one(self):
        """Test is_pipeline_last_stage when world_size is 1."""
        from paddleformers.fleet.parallel_state import is_pipeline_last_stage

        # rank=0, world_size=1 => 0 == (1-1) => True
        result = is_pipeline_last_stage()
        self.assertTrue(result)

    def test_set_pipeline_model_parallel_world_size(self):
        """Test set_pipeline_model_parallel_world_size."""
        import paddleformers.fleet.parallel_state as ps
        from paddleformers.fleet.parallel_state import (
            set_pipeline_model_parallel_world_size,
        )

        set_pipeline_model_parallel_world_size(4)
        self.assertEqual(ps._PIPELINE_MODEL_PARALLEL_WORLD_SIZE, 4)

    def test_get_virtual_pipeline_model_parallel_rank_none(self):
        """Test get_virtual_pipeline_model_parallel_rank returns None."""
        from paddleformers.fleet.parallel_state import (
            get_virtual_pipeline_model_parallel_rank,
        )

        result = get_virtual_pipeline_model_parallel_rank()
        self.assertIsNone(result)

    def test_get_virtual_pipeline_model_parallel_world_size_none(self):
        """Test get_virtual_pipeline_model_parallel_world_size returns None."""
        from paddleformers.fleet.parallel_state import (
            get_virtual_pipeline_model_parallel_world_size,
        )

        result = get_virtual_pipeline_model_parallel_world_size()
        self.assertIsNone(result)

    def test_get_pipeline_model_parallel_rank_warns(self):
        """Test get_pipeline_model_parallel_rank issues warning."""
        from paddleformers.fleet.parallel_state import get_pipeline_model_parallel_rank

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            rank = get_pipeline_model_parallel_rank()
            self.assertEqual(rank, 0)
            self.assertTrue(len(w) > 0)
            self.assertIn("not implemented", str(w[0].message))


class TestGlobalMemoryBuffer(unittest.TestCase):
    """Tests for global memory buffer functions."""

    def test_get_global_memory_buffer_not_initialized_raises(self):
        """Test get_global_memory_buffer raises when not initialized."""
        import paddleformers.fleet.parallel_state as ps
        from paddleformers.fleet.parallel_state import get_global_memory_buffer

        original = ps._GLOBAL_MEMORY_BUFFER
        ps._GLOBAL_MEMORY_BUFFER = None

        with self.assertRaises(AssertionError):
            get_global_memory_buffer()

        ps._GLOBAL_MEMORY_BUFFER = original

    def test_have_global_memory_buffer_false(self):
        """Test have_global_memory_buffer returns False when not initialized."""
        import paddleformers.fleet.parallel_state as ps
        from paddleformers.fleet.parallel_state import have_global_memory_buffer

        original = ps._GLOBAL_MEMORY_BUFFER
        ps._GLOBAL_MEMORY_BUFFER = None

        self.assertFalse(have_global_memory_buffer())

        ps._GLOBAL_MEMORY_BUFFER = original

    def test_destroy_global_memory_buffer(self):
        """Test destroy_global_memory_buffer sets buffer to None."""
        import paddleformers.fleet.parallel_state as ps
        from paddleformers.fleet.parallel_state import destroy_global_memory_buffer

        ps._GLOBAL_MEMORY_BUFFER = mock.MagicMock()
        destroy_global_memory_buffer()
        self.assertIsNone(ps._GLOBAL_MEMORY_BUFFER)


class TestEmbeddingGroup(unittest.TestCase):
    """Tests for embedding group function."""

    def test_get_embedding_group_raises(self):
        """Test get_embedding_group raises NotImplementedError."""
        from paddleformers.fleet.parallel_state import get_embedding_group

        with self.assertRaises(NotImplementedError):
            get_embedding_group()


class TestDataParallelGroupWithCP(unittest.TestCase):
    """Tests for data parallel group with context parallel."""

    def test_get_data_parallel_group_with_cp_not_initialized(self):
        """Test get_data_parallel_group with with_context_parallel=True."""
        import paddleformers.fleet.parallel_state as ps
        from paddleformers.fleet.parallel_state import get_data_parallel_group

        original = ps._DATA_PARALLEL_GROUP_WITH_CP
        ps._DATA_PARALLEL_GROUP_WITH_CP = None

        with self.assertRaises(AssertionError):
            get_data_parallel_group(with_context_parallel=True, check_initialized=True)

        ps._DATA_PARALLEL_GROUP_WITH_CP = original

    def test_get_data_parallel_group_without_cp(self):
        """Test get_data_parallel_group with with_context_parallel=False."""
        import paddleformers.fleet.parallel_state as ps
        from paddleformers.fleet.parallel_state import get_data_parallel_group

        original = ps._DATA_PARALLEL_GROUP
        ps._DATA_PARALLEL_GROUP = None

        with self.assertRaises(AssertionError):
            get_data_parallel_group(check_initialized=True)

        ps._DATA_PARALLEL_GROUP = original


if __name__ == "__main__":
    unittest.main()
