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
from unittest.mock import MagicMock, patch

import numpy as np
import paddle

from paddleformers.cli.train.ernie_pretrain.models.comm_utils import (
    all_gather,
    profile,
    reduce_scatter,
    scatter,
    subbatch,
)


class TestScatter(unittest.TestCase):
    """Tests for scatter function."""

    def test_parallelism_1_returns_clone(self):
        """Test that scatter with parallelism=1 returns a clone."""
        mock_group = MagicMock()
        mock_group.nranks = 1
        mock_group.rank = 0

        x = paddle.randn([4, 8])
        result = scatter(x, group=mock_group, axis=0)

        self.assertIsNot(result, x)
        np.testing.assert_allclose(result.numpy(), x.numpy(), rtol=1e-5)

    def test_scatter_with_parallelism_1_on_axis_0(self):
        """Test scatter with single rank on axis 0."""
        mock_group = MagicMock()
        mock_group.nranks = 1
        mock_group.rank = 0

        x = paddle.randn([6, 4])
        result = scatter(x, group=mock_group, axis=0)
        self.assertEqual(result.shape, [6, 4])

    def test_scatter_with_parallelism_1_on_axis_1(self):
        """Test scatter with single rank on axis 1."""
        mock_group = MagicMock()
        mock_group.nranks = 1
        mock_group.rank = 0

        x = paddle.randn([4, 8])
        result = scatter(x, group=mock_group, axis=1)
        self.assertEqual(result.shape, [4, 8])


class TestAllGather(unittest.TestCase):
    """Tests for all_gather function."""

    def test_parallelism_1_returns_clone(self):
        """Test that all_gather with parallelism=1 returns a clone."""
        mock_group = MagicMock()
        mock_group.nranks = 1

        x = paddle.randn([4, 8])
        result = all_gather(x, group=mock_group, axis=0)

        self.assertIsNot(result, x)
        np.testing.assert_allclose(result.numpy(), x.numpy(), rtol=1e-5)

    def test_all_gather_parallelism_1_axis_0(self):
        """Test all_gather with single rank on axis 0."""
        mock_group = MagicMock()
        mock_group.nranks = 1

        x = paddle.randn([4, 8])
        result = all_gather(x, group=mock_group, axis=0)
        self.assertEqual(result.shape, [4, 8])


class TestReduceScatter(unittest.TestCase):
    """Tests for reduce_scatter function."""

    def test_parallelism_1_returns_clone(self):
        """Test that reduce_scatter with parallelism=1 returns a clone."""
        mock_group = MagicMock()
        mock_group.nranks = 1

        x = paddle.randn([4, 8])
        result = reduce_scatter(x, group=mock_group)

        self.assertIsNot(result, x)
        np.testing.assert_allclose(result.numpy(), x.numpy(), rtol=1e-5)


class TestSubbatch(unittest.TestCase):
    """Tests for subbatch decorator."""

    def test_subbatch_small_input_passes_through(self):
        """Test that subbatch with small input passes through to function directly."""

        def simple_fn(x):
            return x * 2

        wrapped = subbatch(simple_fn, arg_idx=[0], axis=[0], bs=100, out_idx=0)
        x = paddle.randn([10, 4])
        result = wrapped(x)
        np.testing.assert_allclose(result.numpy(), (x * 2).numpy(), rtol=1e-5)

    def test_subbatch_large_input_splits(self):
        """Test that subbatch with large input splits correctly."""

        def simple_fn(x):
            return x * 3

        wrapped = subbatch(simple_fn, arg_idx=[0], axis=[0], bs=4, out_idx=0)
        x = paddle.randn([8, 4])
        result = wrapped(x)
        expected = x * 3
        np.testing.assert_allclose(result.numpy(), expected.numpy(), rtol=1e-5)

    def test_subbatch_mismatched_args_raises(self):
        """Test that mismatched arg_idx and axis lengths raise assertion when called."""

        def simple_fn(x):
            return x

        wrapped = subbatch(simple_fn, arg_idx=[0], axis=[0, 1], bs=4, out_idx=0)
        x = paddle.randn([8, 4])
        with self.assertRaises(AssertionError):
            wrapped(x)

    def test_subbatch_with_kwargs(self):
        """Test subbatch with keyword arguments."""

        def fn_with_kwargs(x, scale=1.0):
            return x * scale

        wrapped = subbatch(fn_with_kwargs, arg_idx=[0], axis=[0], bs=100, out_idx=0)
        x = paddle.randn([10, 4])
        result = wrapped(x, scale=2.0)
        expected = x * 2.0
        np.testing.assert_allclose(result.numpy(), expected.numpy(), rtol=1e-5)


class TestProfile(unittest.TestCase):
    """Tests for profile context manager."""

    @patch("paddleformers.cli.train.ernie_pretrain.models.comm_utils.get_timers")
    def test_profile_with_no_timers(self, mock_get_timers):
        """Test profile when get_timers returns None."""
        mock_get_timers.return_value = None
        with profile("test_op"):
            x = 1 + 1
        self.assertEqual(x, 2)

    @unittest.skip("get_timers module-level reference cannot be reliably patched in CI generator context")
    def test_profile_with_timers(self):
        """Test profile when get_timers returns a callable."""
        import paddleformers.cli.train.ernie_pretrain.models.comm_utils as comm_utils_mod

        mock_timer = MagicMock()
        original_get_timers = comm_utils_mod.get_timers
        try:
            comm_utils_mod.get_timers = lambda: (lambda name, use_event=True: mock_timer)

            with comm_utils_mod.profile("test_op"):
                pass
        finally:
            comm_utils_mod.get_timers = original_get_timers

        mock_timer.start.assert_called()
        mock_timer.stop.assert_called()


if __name__ == "__main__":
    unittest.main()
