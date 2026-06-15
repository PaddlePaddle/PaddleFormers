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


# Tests for src/paddleformers.fleet/parallel_state.py

import unittest
import warnings
from unittest import mock


class TestParallelState(unittest.TestCase):
    """Tests for parallel_state module functions."""

    def setUp(self):
        """Reset parallel state before each test."""
        import paddleformers.fleet.parallel_state as ps

        # Reset all global variables
        ps._TENSOR_MODEL_PARALLEL_GROUP = None
        ps._TENSOR_MODEL_PARALLEL_GLOBAL_RANKS = None
        ps._PIPELINE_MODEL_PARALLEL_GROUP = None
        ps._PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
        ps._DATA_PARALLEL_GROUP = None
        ps._EXPERT_MODEL_PARALLEL_GROUP = None
        ps._EXPERT_DATA_PARALLEL_GROUP = None
        ps._CONTEXT_PARALLEL_GROUP = None
        ps._DATA_PARALLEL_GROUP_WITH_CP = None
        ps._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
        ps._MPU_TENSOR_MODEL_PARALLEL_RANK = None
        ps._VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = None
        ps._VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
        ps._GLOBAL_MEMORY_BUFFER = None
        ps._EXPERT_TENSOR_PARALLEL_GROUP = None
        ps._MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE = None
        ps._MPU_EXPERT_MODEL_PARALLEL_RANK = None
        ps._MPU_EXPERT_TENSOR_PARALLEL_WORLD_SIZE = None
        ps._MPU_EXPERT_TENSOR_PARALLEL_RANK = None
        ps._EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP = None

    def tearDown(self):
        """Clean up after each test."""
        self.setUp()

    def _make_mock_group(self, nranks=4, rank=1, world_size=4):
        """Helper to create a mock group."""
        group = mock.MagicMock()
        group.nranks = nranks
        group.world_size = world_size
        group.size = mock.MagicMock(return_value=world_size)

        # rank needs to work both as attribute (some functions access group.rank)
        # and as callable (get_expert_tensor_and_model_parallel_group().rank())
        # Use a callable that returns the rank value and compares equal to it
        class _CallableInt(int):
            def __new__(cls, value):
                return super().__new__(cls, value)

            def __call__(self):
                return int(self)

            def __eq__(self, other):
                return int(self) == other

            def __hash__(self):
                return hash(int(self))

            def __repr__(self):
                return repr(int(self))

        group.rank = _CallableInt(rank)
        return group

    def test_initialize_model_parallel(self):
        """Test initialize_model_parallel sets all groups."""
        import paddleformers.fleet.parallel_state as ps

        with mock.patch("paddleformers.fleet.parallel_state.GlobalMemoryBuffer"):
            hcg = mock.MagicMock()
            hcg._mp_comm_group = self._make_mock_group(4, 0)
            hcg._mp_group = [0, 1, 2, 3]
            hcg._pp_comm_group = self._make_mock_group(2, 0)
            hcg._sharding_comm_group = self._make_mock_group(2, 0)
            hcg._ep_comm_group = self._make_mock_group(1, 0)
            hcg._moe_sharding_comm_group = self._make_mock_group(1, 0)
            hcg._cp_comm_group = None
            hcg._cp_sharding_comm_group = None

            ps.initialize_model_parallel(hcg)

            self.assertIsNotNone(ps._TENSOR_MODEL_PARALLEL_GROUP)
            self.assertIsNotNone(ps._PIPELINE_MODEL_PARALLEL_GROUP)
            self.assertIsNotNone(ps._DATA_PARALLEL_GROUP)
            self.assertEqual(ps._TENSOR_MODEL_PARALLEL_GLOBAL_RANKS, [0, 1, 2, 3])

    def test_initialize_model_parallel_with_virtual_pipeline(self):
        """Test initialize with virtual_pipeline_model_parallel_size."""
        import paddleformers.fleet.parallel_state as ps

        with mock.patch("paddleformers.fleet.parallel_state.GlobalMemoryBuffer"):
            hcg = mock.MagicMock()
            hcg._mp_comm_group = self._make_mock_group(4, 0)
            hcg._mp_group = [0, 1, 2, 3]
            hcg._pp_comm_group = self._make_mock_group(2, 0)
            hcg._sharding_comm_group = self._make_mock_group(2, 0)
            hcg._ep_comm_group = None
            hcg._moe_sharding_comm_group = None
            hcg._cp_comm_group = None
            hcg._cp_sharding_comm_group = None

            ps.initialize_model_parallel(hcg, virtual_pipeline_model_parallel_size=2)
            self.assertEqual(ps._VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE, 2)
            self.assertEqual(ps._VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK, 0)

    def test_initialize_model_parallel_virtual_pipeline_single_pp_raises(self):
        """Test that virtual pipeline with single pp stage raises."""
        import paddleformers.fleet.parallel_state as ps

        with mock.patch("paddleformers.fleet.parallel_state.GlobalMemoryBuffer"):
            hcg = mock.MagicMock()
            hcg._mp_comm_group = self._make_mock_group(4, 0)
            hcg._mp_group = [0, 1, 2, 3]
            hcg._pp_comm_group = self._make_mock_group(1, 0)
            hcg._sharding_comm_group = self._make_mock_group(4, 0)
            hcg._ep_comm_group = None
            hcg._moe_sharding_comm_group = None
            hcg._cp_comm_group = None
            hcg._cp_sharding_comm_group = None

            with self.assertRaises(RuntimeError):
                ps.initialize_model_parallel(hcg, virtual_pipeline_model_parallel_size=2)

    def test_get_tensor_model_parallel_group_not_initialized(self):
        """Test assertion when tensor model parallel group not initialized."""
        import paddleformers.fleet.parallel_state as ps

        with self.assertRaises(AssertionError):
            ps.get_tensor_model_parallel_group(check_initialized=True)

    def test_get_tensor_model_parallel_group_not_check(self):
        """Test get tensor model parallel group without check."""
        import paddleformers.fleet.parallel_state as ps

        result = ps.get_tensor_model_parallel_group(check_initialized=False)
        self.assertIsNone(result)

    def test_get_tensor_model_parallel_world_size_default(self):
        """Test default tensor model parallel world size."""
        import paddleformers.fleet.parallel_state as ps

        # When group is None, should return 1
        ps._TENSOR_MODEL_PARALLEL_GROUP = None
        size = ps.get_tensor_model_parallel_world_size()
        self.assertEqual(size, 1)

    def test_get_tensor_model_parallel_world_size_from_group(self):
        """Test tensor model parallel world size from group."""
        import paddleformers.fleet.parallel_state as ps

        ps._TENSOR_MODEL_PARALLEL_GROUP = self._make_mock_group(nranks=8)
        ps._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
        size = ps.get_tensor_model_parallel_world_size()
        self.assertEqual(size, 8)

    def test_get_tensor_model_parallel_world_size_override(self):
        """Test overridden tensor model parallel world size."""
        import paddleformers.fleet.parallel_state as ps

        ps._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = 16
        size = ps.get_tensor_model_parallel_world_size()
        self.assertEqual(size, 16)

    def test_get_tensor_model_parallel_rank_default(self):
        """Test default tensor model parallel rank."""
        import paddleformers.fleet.parallel_state as ps

        ps._TENSOR_MODEL_PARALLEL_GROUP = None
        ps._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
        ps._MPU_TENSOR_MODEL_PARALLEL_RANK = None
        rank = ps.get_tensor_model_parallel_rank()
        self.assertEqual(rank, 0)

    def test_get_tensor_model_parallel_rank_from_group(self):
        """Test tensor model parallel rank from group."""
        import paddleformers.fleet.parallel_state as ps

        ps._TENSOR_MODEL_PARALLEL_GROUP = self._make_mock_group(nranks=4, rank=2)
        ps._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
        ps._MPU_TENSOR_MODEL_PARALLEL_RANK = None
        rank = ps.get_tensor_model_parallel_rank()
        self.assertEqual(rank, 2)

    def test_get_pipeline_model_parallel_group_not_initialized(self):
        """Test assertion when pipeline model parallel group not initialized."""
        import paddleformers.fleet.parallel_state as ps

        with self.assertRaises(AssertionError):
            ps.get_pipeline_model_parallel_group(check_initialized=True)

    def test_get_pipeline_model_parallel_group_not_check(self):
        """Test get pipeline model parallel group without check."""
        import paddleformers.fleet.parallel_state as ps

        result = ps.get_pipeline_model_parallel_group(check_initialized=False)
        self.assertIsNone(result)

    def test_get_pipeline_model_parallel_world_size(self):
        """Test pipeline model parallel world size always returns 1."""
        import paddleformers.fleet.parallel_state as ps

        size = ps.get_pipeline_model_parallel_world_size()
        self.assertEqual(size, 1)

    def test_set_pipeline_model_parallel_world_size(self):
        """Test setting pipeline model parallel world size."""
        import paddleformers.fleet.parallel_state as ps

        ps.set_pipeline_model_parallel_world_size(4)
        self.assertEqual(ps._PIPELINE_MODEL_PARALLEL_WORLD_SIZE, 4)

    def test_get_pipeline_model_parallel_rank(self):
        """Test pipeline model parallel rank always returns 0 with warning."""
        import paddleformers.fleet.parallel_state as ps

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            rank = ps.get_pipeline_model_parallel_rank()
            self.assertEqual(rank, 0)
            self.assertTrue(len(w) > 0)

    def test_is_pipeline_first_stage(self):
        """Test is_pipeline_first_stage."""
        import paddleformers.fleet.parallel_state as ps

        self.assertTrue(ps.is_pipeline_first_stage())

    def test_is_pipeline_first_stage_virtual(self):
        """Test is_pipeline_first_stage with virtual pipeline."""
        import paddleformers.fleet.parallel_state as ps

        ps._VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = 2
        # Should raise without vp_stage
        with self.assertRaises(AssertionError):
            ps.is_pipeline_first_stage(ignore_virtual=False)

    def test_is_pipeline_first_stage_virtual_with_stage(self):
        """Test is_pipeline_first_stage with virtual pipeline and vp_stage=0."""
        import paddleformers.fleet.parallel_state as ps

        ps._VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = 2
        result = ps.is_pipeline_first_stage(ignore_virtual=False, vp_stage=0)
        self.assertTrue(result)

    def test_is_pipeline_first_stage_virtual_not_first(self):
        """Test is_pipeline_first_stage with virtual pipeline and vp_stage=1."""
        import paddleformers.fleet.parallel_state as ps

        ps._VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = 2
        result = ps.is_pipeline_first_stage(ignore_virtual=False, vp_stage=1)
        self.assertFalse(result)

    def test_is_pipeline_last_stage(self):
        """Test is_pipeline_last_stage."""
        import paddleformers.fleet.parallel_state as ps

        ps._VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
        self.assertTrue(ps.is_pipeline_last_stage())

    def test_is_pipeline_last_stage_virtual(self):
        """Test is_pipeline_last_stage with virtual pipeline."""
        import paddleformers.fleet.parallel_state as ps

        ps._VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = 2
        result = ps.is_pipeline_last_stage(ignore_virtual=False, vp_stage=1)
        self.assertTrue(result)

    def test_virtual_pipeline_parallel_rank(self):
        """Test get/set virtual pipeline parallel rank."""
        import paddleformers.fleet.parallel_state as ps

        self.assertIsNone(ps.get_virtual_pipeline_model_parallel_rank())
        ps.set_virtual_pipeline_model_parallel_rank(3)
        self.assertEqual(ps.get_virtual_pipeline_model_parallel_rank(), 3)

    def test_set_virtual_pipeline_parallel_rank_deprecation(self):
        """Test set_virtual_pipeline_model_parallel_rank deprecation warning."""
        import paddleformers.fleet.parallel_state as ps

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ps.set_virtual_pipeline_model_parallel_rank(1)
            self.assertTrue(any(issubclass(x.category, DeprecationWarning) for x in w))

    def test_get_data_parallel_group_with_cp(self):
        """Test data parallel group with context parallel."""
        import paddleformers.fleet.parallel_state as ps

        ps._DATA_PARALLEL_GROUP_WITH_CP = self._make_mock_group(2, 0)
        result = ps.get_data_parallel_group(with_context_parallel=True)
        self.assertIsNotNone(result)

    def test_get_data_parallel_group_no_cp(self):
        """Test data parallel group without context parallel."""
        import paddleformers.fleet.parallel_state as ps

        ps._DATA_PARALLEL_GROUP = self._make_mock_group(2, 0)
        result = ps.get_data_parallel_group(with_context_parallel=False)
        self.assertIsNotNone(result)

    def test_get_data_parallel_group_not_initialized(self):
        """Test data parallel group assertion when not initialized."""
        import paddleformers.fleet.parallel_state as ps

        with self.assertRaises(AssertionError):
            ps.get_data_parallel_group(check_initialized=True, with_context_parallel=False)

    def test_get_data_parallel_group_with_cp_not_initialized(self):
        """Test data parallel group with CP assertion when not initialized."""
        import paddleformers.fleet.parallel_state as ps

        with self.assertRaises(AssertionError):
            ps.get_data_parallel_group(check_initialized=True, with_context_parallel=True)

    def test_get_expert_model_parallel_group(self):
        """Test get expert model parallel group."""
        import paddleformers.fleet.parallel_state as ps

        ps._EXPERT_MODEL_PARALLEL_GROUP = self._make_mock_group(2, 0)
        result = ps.get_expert_model_parallel_group(check_initialized=False)
        self.assertIsNotNone(result)

    def test_get_expert_data_parallel_group(self):
        """Test get expert data parallel group."""
        import paddleformers.fleet.parallel_state as ps

        ps._EXPERT_DATA_PARALLEL_GROUP = self._make_mock_group(2, 0)
        result = ps.get_expert_data_parallel_group(check_initialized=False)
        self.assertIsNotNone(result)

    def test_get_context_parallel_group(self):
        """Test get context parallel group."""
        import paddleformers.fleet.parallel_state as ps

        ps._CONTEXT_PARALLEL_GROUP = self._make_mock_group(2, 0)
        result = ps.get_context_parallel_group(check_initialized=False)
        self.assertIsNotNone(result)

    def test_get_context_parallel_world_size_none(self):
        """Test context parallel world size when group is None."""
        import paddleformers.fleet.parallel_state as ps

        ps._CONTEXT_PARALLEL_GROUP = None
        size = ps.get_context_parallel_world_size()
        self.assertEqual(size, 1)

    def test_get_context_parallel_world_size(self):
        """Test context parallel world size."""
        import paddleformers.fleet.parallel_state as ps

        ps._CONTEXT_PARALLEL_GROUP = self._make_mock_group(4, 0, world_size=4)
        size = ps.get_context_parallel_world_size()
        self.assertEqual(size, 4)

    def test_set_expert_model_parallel_world_size(self):
        """Test set expert model parallel world size."""
        import paddleformers.fleet.parallel_state as ps

        ps.set_expert_model_parallel_world_size(8)
        self.assertEqual(ps._MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE, 8)

    def test_get_expert_model_parallel_rank_default(self):
        """Test expert model parallel rank default."""
        import paddleformers.fleet.parallel_state as ps

        with mock.patch("paddle.distributed.is_initialized", return_value=False):
            rank = ps.get_expert_model_parallel_rank()
            self.assertEqual(rank, 0)

    def test_get_expert_model_parallel_rank_from_state(self):
        """Test expert model parallel rank from state."""
        import paddleformers.fleet.parallel_state as ps

        ps._MPU_EXPERT_MODEL_PARALLEL_RANK = 3
        rank = ps.get_expert_model_parallel_rank()
        self.assertEqual(rank, 3)

    def test_set_expert_model_parallel_rank(self):
        """Test set expert model parallel rank."""
        import paddleformers.fleet.parallel_state as ps

        ps.set_expert_model_parallel_rank(5)
        self.assertEqual(ps._MPU_EXPERT_MODEL_PARALLEL_RANK, 5)

    def test_get_expert_tensor_parallel_group(self):
        """Test get expert tensor parallel group."""
        import paddleformers.fleet.parallel_state as ps

        ps._EXPERT_TENSOR_PARALLEL_GROUP = self._make_mock_group(2, 0)
        result = ps.get_expert_tensor_parallel_group(check_initialized=False)
        self.assertIsNotNone(result)

    def test_get_expert_tensor_parallel_world_size_default(self):
        """Test expert tensor parallel world size default."""
        import paddleformers.fleet.parallel_state as ps

        ps._EXPERT_TENSOR_PARALLEL_GROUP = None
        ps._MPU_EXPERT_TENSOR_PARALLEL_WORLD_SIZE = None
        ps._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = 4
        size = ps.get_expert_tensor_parallel_world_size()
        self.assertEqual(size, 4)

    def test_get_expert_tensor_parallel_world_size_from_state(self):
        """Test expert tensor parallel world size from state."""
        import paddleformers.fleet.parallel_state as ps

        ps._MPU_EXPERT_TENSOR_PARALLEL_WORLD_SIZE = 16
        size = ps.get_expert_tensor_parallel_world_size()
        self.assertEqual(size, 16)

    def test_get_expert_tensor_parallel_world_size_from_group(self):
        """Test expert tensor parallel world size from group."""
        import paddleformers.fleet.parallel_state as ps

        ps._EXPERT_TENSOR_PARALLEL_GROUP = self._make_mock_group(4, 0, world_size=4)
        ps._MPU_EXPERT_TENSOR_PARALLEL_WORLD_SIZE = None
        size = ps.get_expert_tensor_parallel_world_size()
        self.assertEqual(size, 4)

    def test_set_expert_tensor_parallel_world_size(self):
        """Test set expert tensor parallel world size."""
        import paddleformers.fleet.parallel_state as ps

        ps.set_expert_tensor_parallel_world_size(8)
        self.assertEqual(ps._MPU_EXPERT_TENSOR_PARALLEL_WORLD_SIZE, 8)

    def test_get_expert_tensor_parallel_rank_default(self):
        """Test expert tensor parallel rank default."""
        import paddleformers.fleet.parallel_state as ps

        ps._EXPERT_TENSOR_PARALLEL_GROUP = None
        ps._MPU_EXPERT_TENSOR_PARALLEL_RANK = None
        rank = ps.get_expert_tensor_parallel_rank()
        self.assertEqual(rank, 0)

    def test_get_expert_tensor_parallel_rank_from_state(self):
        """Test expert tensor parallel rank from state."""
        import paddleformers.fleet.parallel_state as ps

        ps._MPU_EXPERT_TENSOR_PARALLEL_RANK = 7
        rank = ps.get_expert_tensor_parallel_rank()
        self.assertEqual(rank, 7)

    def test_set_expert_tensor_parallel_rank(self):
        """Test set expert tensor parallel rank."""
        import paddleformers.fleet.parallel_state as ps

        ps.set_expert_tensor_parallel_rank(3)
        self.assertEqual(ps._MPU_EXPERT_TENSOR_PARALLEL_RANK, 3)

    def test_get_expert_tensor_and_model_parallel_group(self):
        """Test get expert tensor and model parallel group."""
        import paddleformers.fleet.parallel_state as ps

        ps._EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP = self._make_mock_group(8, 0)
        result = ps.get_expert_tensor_and_model_parallel_group(check_initialized=False)
        self.assertIsNotNone(result)

    def test_get_expert_tensor_and_model_parallel_world_size_no_dist(self):
        """Test expert tensor and model parallel world size without distributed."""
        import paddleformers.fleet.parallel_state as ps

        with mock.patch("paddle.distributed.is_available", return_value=False):
            size = ps.get_expert_tensor_and_model_parallel_world_size()
            self.assertEqual(size, 0)

    def test_get_expert_tensor_and_model_parallel_rank_no_dist(self):
        """Test expert tensor and model parallel rank without distributed."""
        import paddleformers.fleet.parallel_state as ps

        with mock.patch("paddle.distributed.is_available", return_value=False):
            rank = ps.get_expert_tensor_and_model_parallel_rank()
            self.assertEqual(rank, 0)

    def test_get_expert_tensor_and_model_parallel_world_size_with_dist(self):
        """Test expert tensor and model parallel world size with distributed."""
        import paddleformers.fleet.parallel_state as ps

        ps._EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP = self._make_mock_group(8, 0, world_size=8)
        with mock.patch("paddle.distributed.is_available", return_value=True):  # noqa: SIM117
            with mock.patch("paddle.distributed.is_initialized", return_value=True):
                size = ps.get_expert_tensor_and_model_parallel_world_size()
                self.assertEqual(size, 8)

    def test_get_expert_tensor_and_model_parallel_rank_with_dist(self):
        """Test expert tensor and model parallel rank with distributed."""
        import paddleformers.fleet.parallel_state as ps

        group = self._make_mock_group(8, 3, world_size=8)
        ps._EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP = group
        with mock.patch("paddle.distributed.is_available", return_value=True):  # noqa: SIM117
            with mock.patch("paddle.distributed.is_initialized", return_value=True):
                rank = ps.get_expert_tensor_and_model_parallel_rank()
                self.assertEqual(rank, 3)

    def test_get_embedding_group_raises(self):
        """Test get_embedding_group raises NotImplementedError."""
        import paddleformers.fleet.parallel_state as ps

        with self.assertRaises(NotImplementedError):
            ps.get_embedding_group()

    def test_global_memory_buffer(self):
        """Test global memory buffer lifecycle."""
        import paddleformers.fleet.parallel_state as ps

        with mock.patch("paddleformers.fleet.parallel_state.GlobalMemoryBuffer") as mock_gmb:
            ps._set_global_memory_buffer()
            self.assertTrue(ps.have_global_memory_buffer())
            buf = ps.get_global_memory_buffer()
            self.assertEqual(buf, mock_gmb.return_value)
            ps.destroy_global_memory_buffer()
            self.assertFalse(ps.have_global_memory_buffer())

    def test_global_memory_buffer_already_initialized_raises(self):
        """Test that double initialization raises."""
        import paddleformers.fleet.parallel_state as ps

        with mock.patch("paddleformers.fleet.parallel_state.GlobalMemoryBuffer"):
            ps._set_global_memory_buffer()
            with self.assertRaises(AssertionError):
                ps._set_global_memory_buffer()

    def test_get_global_memory_buffer_not_initialized_raises(self):
        """Test that get without init raises."""
        import paddleformers.fleet.parallel_state as ps

        with self.assertRaises(AssertionError):
            ps.get_global_memory_buffer()

    def test_have_global_memory_buffer_false(self):
        """Test have_global_memory_buffer returns False when not initialized."""
        import paddleformers.fleet.parallel_state as ps

        ps._GLOBAL_MEMORY_BUFFER = None
        self.assertFalse(ps.have_global_memory_buffer())


if __name__ == "__main__":
    unittest.main()
