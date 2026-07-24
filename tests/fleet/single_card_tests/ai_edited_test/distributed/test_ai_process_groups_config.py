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


# Tests for src/paddleformers.fleet/process_groups_config.py

import unittest
from unittest import mock


class TestProcessGroupCollection(unittest.TestCase):
    """Tests for ProcessGroupCollection."""

    def test_init_empty(self):
        """Test creating an empty ProcessGroupCollection."""
        from paddleformers.fleet.process_groups_config import (
            ProcessGroupCollection,
        )

        pgs = ProcessGroupCollection()
        # All fields should be un-initialized (since init=False)
        # but the object should exist
        self.assertIsNotNone(pgs)

    def test_init_with_kwargs(self):
        """Test creating ProcessGroupCollection with kwargs."""
        from paddleformers.fleet.process_groups_config import (
            ProcessGroupCollection,
        )

        mock_group = mock.MagicMock()
        pgs = ProcessGroupCollection(tp=mock_group, dp=mock_group)
        self.assertEqual(pgs.tp, mock_group)
        self.assertEqual(pgs.dp, mock_group)

    def test_init_with_unknown_attribute_raises(self):
        """Test that unknown attribute raises ValueError."""
        from paddleformers.fleet.process_groups_config import (
            ProcessGroupCollection,
        )

        with self.assertRaises(ValueError) as ctx:
            ProcessGroupCollection(unknown_attr=123)
        self.assertIn("Unknown attribute", str(ctx.exception))

    def test_use_mpu_process_groups_all(self):
        """Test use_mpu_process_groups with None (all groups)."""
        from paddleformers.fleet.process_groups_config import (
            ProcessGroupCollection,
        )

        with mock.patch(
            "paddleformers.fleet.process_groups_config.parallel_state"
        ) as mock_ps:
            mock_ps.get_tensor_model_parallel_group.return_value = (
                mock.MagicMock()
            )
            mock_ps.get_pipeline_model_parallel_group.return_value = (
                mock.MagicMock()
            )
            mock_ps.get_context_parallel_group.return_value = mock.MagicMock()
            mock_ps.get_expert_model_parallel_group.return_value = (
                mock.MagicMock()
            )
            mock_ps.get_data_parallel_group.return_value = mock.MagicMock()
            mock_ps.get_expert_data_parallel_group.return_value = (
                mock.MagicMock()
            )

            pgs = ProcessGroupCollection.use_mpu_process_groups(
                required_pgs=None
            )
            self.assertIsInstance(pgs, ProcessGroupCollection)

    def test_use_mpu_process_groups_subset(self):
        """Test use_mpu_process_groups with specific groups."""
        from paddleformers.fleet.process_groups_config import (
            ProcessGroupCollection,
        )

        mock_tp = mock.MagicMock()
        mock_dp = mock.MagicMock()
        with mock.patch(
            "paddleformers.fleet.process_groups_config.parallel_state"
        ) as mock_ps:
            mock_ps.get_tensor_model_parallel_group.return_value = mock_tp
            mock_ps.get_data_parallel_group.return_value = mock_dp
            pgs = ProcessGroupCollection.use_mpu_process_groups(
                required_pgs=["tp", "dp"]
            )
            self.assertEqual(pgs.tp, mock_tp)
            self.assertEqual(pgs.dp, mock_dp)

    def test_use_mpu_process_groups_invalid_raises(self):
        """Test use_mpu_process_groups with invalid groups raises."""
        from paddleformers.fleet.process_groups_config import (
            ProcessGroupCollection,
        )

        with self.assertRaises(ValueError) as ctx:
            ProcessGroupCollection.use_mpu_process_groups(
                required_pgs=["invalid_pg"]
            )
        self.assertIn("Invalid process groups", str(ctx.exception))

    def test_use_mpu_process_groups_with_cp(self):
        """Test use_mpu_process_groups with cp_dp group."""
        from paddleformers.fleet.process_groups_config import (
            ProcessGroupCollection,
        )

        mock_cp_dp = mock.MagicMock()
        with mock.patch(
            "paddleformers.fleet.process_groups_config.parallel_state"
        ) as mock_ps:
            mock_ps.get_data_parallel_group.return_value = mock_cp_dp
            pgs = ProcessGroupCollection.use_mpu_process_groups(
                required_pgs=["cp_dp"]
            )
            self.assertEqual(pgs.cp_dp, mock_cp_dp)


if __name__ == "__main__":
    unittest.main()
