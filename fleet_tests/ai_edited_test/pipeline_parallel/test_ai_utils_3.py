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
from unittest.mock import MagicMock, patch


class TestIsPpFirstStage(unittest.TestCase):
    """Tests for is_pp_first_stage in pipeline_parallel/utils.py."""

    def test_rank_zero_is_first(self):
        from paddleformers.fleet.pipeline_parallel.utils import is_pp_first_stage

        mock_group = MagicMock()
        with patch(
            "paddleformers.fleet.pipeline_parallel.utils.get_pg_rank", return_value=0
        ):
            self.assertTrue(is_pp_first_stage(mock_group))

    def test_rank_nonzero_not_first(self):
        from paddleformers.fleet.pipeline_parallel.utils import is_pp_first_stage

        mock_group = MagicMock()
        with patch(
            "paddleformers.fleet.pipeline_parallel.utils.get_pg_rank", return_value=1
        ):
            self.assertFalse(is_pp_first_stage(mock_group))


class TestIsPpLastStage(unittest.TestCase):
    """Tests for is_pp_last_stage in pipeline_parallel/utils.py."""

    def test_last_rank_is_last(self):
        from paddleformers.fleet.pipeline_parallel.utils import is_pp_last_stage

        mock_group = MagicMock()
        with (
            patch(
                "paddleformers.fleet.pipeline_parallel.utils.get_pg_rank",
                return_value=3,
            ),
            patch(
                "paddleformers.fleet.pipeline_parallel.utils.get_pg_size",
                return_value=4,
            ),
        ):
            self.assertTrue(is_pp_last_stage(mock_group))

    def test_not_last_rank(self):
        from paddleformers.fleet.pipeline_parallel.utils import is_pp_last_stage

        mock_group = MagicMock()
        with (
            patch(
                "paddleformers.fleet.pipeline_parallel.utils.get_pg_rank",
                return_value=1,
            ),
            patch(
                "paddleformers.fleet.pipeline_parallel.utils.get_pg_size",
                return_value=4,
            ),
        ):
            self.assertFalse(is_pp_last_stage(mock_group))


class TestIsVpFirstStage(unittest.TestCase):
    """Tests for is_vp_first_stage in pipeline_parallel/utils.py."""

    def test_vp_size_none(self):
        from paddleformers.fleet.pipeline_parallel.utils import is_vp_first_stage

        self.assertTrue(is_vp_first_stage(vp_stage=0, vp_size=None))

    def test_vp_size_one(self):
        from paddleformers.fleet.pipeline_parallel.utils import is_vp_first_stage

        self.assertTrue(is_vp_first_stage(vp_stage=0, vp_size=1))

    def test_vp_stage_zero(self):
        from paddleformers.fleet.pipeline_parallel.utils import is_vp_first_stage

        self.assertTrue(is_vp_first_stage(vp_stage=0, vp_size=4))

    def test_vp_stage_nonzero(self):
        from paddleformers.fleet.pipeline_parallel.utils import is_vp_first_stage

        self.assertFalse(is_vp_first_stage(vp_stage=2, vp_size=4))

    def test_vp_size_le_one_assertion(self):
        from paddleformers.fleet.pipeline_parallel.utils import is_vp_first_stage

        with self.assertRaises(AssertionError):
            is_vp_first_stage(vp_stage=3, vp_size=1)


class TestIsVpLastStage(unittest.TestCase):
    """Tests for is_vp_last_stage in pipeline_parallel/utils.py."""

    def test_vp_size_none(self):
        from paddleformers.fleet.pipeline_parallel.utils import is_vp_last_stage

        self.assertTrue(is_vp_last_stage(vp_stage=0, vp_size=None))

    def test_vp_size_one(self):
        from paddleformers.fleet.pipeline_parallel.utils import is_vp_last_stage

        self.assertTrue(is_vp_last_stage(vp_stage=0, vp_size=1))

    def test_vp_stage_last(self):
        from paddleformers.fleet.pipeline_parallel.utils import is_vp_last_stage

        self.assertTrue(is_vp_last_stage(vp_stage=3, vp_size=4))

    def test_vp_stage_not_last(self):
        from paddleformers.fleet.pipeline_parallel.utils import is_vp_last_stage

        self.assertFalse(is_vp_last_stage(vp_stage=1, vp_size=4))

    def test_vp_size_le_one_assertion(self):
        from paddleformers.fleet.pipeline_parallel.utils import is_vp_last_stage

        with self.assertRaises(AssertionError):
            is_vp_last_stage(vp_stage=5, vp_size=1)


if __name__ == "__main__":
    unittest.main()
