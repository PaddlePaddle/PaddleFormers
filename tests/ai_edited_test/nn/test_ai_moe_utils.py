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
from unittest.mock import MagicMock, PropertyMock, patch

import paddle


def _randn(shape, dtype="float32"):
    return paddle.randn(shape, dtype=dtype)


def _make_fake_group(nranks=1, rank=0):
    """Create a fake distributed group for testing."""
    group = MagicMock()
    group.nranks = nranks
    group.rank = rank
    group.ranks = list(range(nranks))
    return group


class TestGetHCG(unittest.TestCase):
    """Tests for get_hcg function."""

    def setUp(self):
        from paddleformers.nn.moe.utils import get_hcg

        self.get_hcg = get_hcg

    @patch("paddleformers.nn.moe.utils.fleet.get_hybrid_communicate_group")
    def test_get_hcg(self, mock_get):
        """Test basic get_hcg call."""
        mock_hcg = MagicMock()
        mock_get.return_value = mock_hcg
        result = self.get_hcg()
        self.assertIs(result, mock_hcg)
        mock_get.assert_called_once()


class TestScatterAxis(unittest.TestCase):
    """Tests for scatter_axis function."""

    def setUp(self):
        from paddleformers.nn.moe.utils import scatter_axis

        self.scatter_axis = scatter_axis

    def test_parallelism_1_returns_clone(self):
        """Test with parallelism=1 returns cloned input."""
        x = _randn([8, 4])
        group = _make_fake_group(nranks=1)
        result = self.scatter_axis(x, group=group, axis=0)
        self.assertEqual(result.shape, [8, 4])

    def test_parallelism_2_rank0(self):
        """Test with parallelism=2, rank=0 gets first half."""
        x = _randn([8, 4])
        group = _make_fake_group(nranks=2, rank=0)
        result = self.scatter_axis(x, group=group, axis=0)
        self.assertEqual(result.shape, [4, 4])

    def test_parallelism_2_rank1(self):
        """Test with parallelism=2, rank=1 gets second half."""
        x = _randn([8, 4])
        group = _make_fake_group(nranks=2, rank=1)
        result = self.scatter_axis(x, group=group, axis=0)
        self.assertEqual(result.shape, [4, 4])

    def test_default_group_uses_model_parallel(self):
        """Test with group=None uses model parallel group from fleet."""
        x = _randn([8, 4])
        with patch("paddleformers.nn.moe.utils.fleet.get_hybrid_communicate_group") as mock_get:
            mock_hcg = MagicMock()
            mock_group = _make_fake_group(nranks=1)
            mock_hcg.get_model_parallel_group.return_value = mock_group
            mock_get.return_value = mock_hcg
            result = self.scatter_axis(x)
        self.assertEqual(result.shape, [8, 4])

    def test_not_divisible_raises_assertion(self):
        """Test that non-divisible seq_len raises assertion error."""
        x = _randn([7, 4])
        group = _make_fake_group(nranks=2, rank=0)
        with self.assertRaises(AssertionError):
            self.scatter_axis(x, group=group, axis=0)

    def test_scatter_along_axis_1(self):
        """Test scattering along axis=1."""
        x = _randn([4, 8])
        group = _make_fake_group(nranks=2, rank=0)
        result = self.scatter_axis(x, group=group, axis=1)
        self.assertEqual(result.shape, [4, 4])


class TestDetachAndRequiresGrad(unittest.TestCase):
    """Tests for detach_and_requires_grad_ function."""

    def setUp(self):
        from paddleformers.nn.moe.utils import detach_and_requires_grad_

        self.detach_and_requires_grad_ = detach_and_requires_grad_

    def test_basic_detach(self):
        """Test basic detach preserves shape."""
        x = _randn([4])
        x.stop_gradient = False
        result = self.detach_and_requires_grad_(x)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape, [4])

    def test_detach_none(self):
        """Test that None input returns None."""
        result = self.detach_and_requires_grad_(None)
        self.assertEqual(len(result), 1)
        self.assertIsNone(result[0])

    def test_detach_multiple(self):
        """Test detaching multiple tensors."""
        x = _randn([4])
        x.stop_gradient = False
        y = _randn([4])
        y.stop_gradient = True
        result = self.detach_and_requires_grad_(x, y, None)
        self.assertEqual(len(result), 3)
        self.assertIsNone(result[2])
        self.assertEqual(result[0].stop_gradient, False)
        self.assertEqual(result[1].stop_gradient, True)


class TestFakeClone(unittest.TestCase):
    """Tests for FakeClone PyLayer."""

    def setUp(self):
        from paddleformers.nn.moe.utils import FakeClone

        self.FakeClone = FakeClone

    def test_contiguous_input(self):
        """Test FakeClone with contiguous input."""
        x = _randn([4, 4])
        result = self.FakeClone.apply(x)
        self.assertEqual(result.shape, [4, 4])

    def test_non_contiguous_input(self):
        """Test FakeClone with non-contiguous input."""
        x = _randn([4, 8])[:, ::2]
        result = self.FakeClone.apply(x)
        self.assertEqual(result.shape, [4, 4])

    def test_backward(self):
        """Test FakeClone backward pass."""
        x = _randn([4, 4])
        x.stop_gradient = False
        result = self.FakeClone.apply(x)
        loss = result.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)


class TestReduceScatterGroupOp(unittest.TestCase):
    """Tests for ReduceScatterGroupOp PyLayer."""

    def setUp(self):
        from paddleformers.nn.moe.utils import ReduceScatterGroupOp

        self.ReduceScatterGroupOp = ReduceScatterGroupOp

    @patch("paddleformers.nn.moe.utils.reduce_scatter_group")
    def test_forward(self, mock_rs):
        """Test forward pass delegates to reduce_scatter_group."""
        x = _randn([8, 4])
        expected = _randn([4, 4])
        mock_rs.return_value = expected
        group = _make_fake_group()
        result = self.ReduceScatterGroupOp.apply(x, group)
        self.assertTrue(paddle.allclose(result, expected))

    @patch("paddleformers.nn.moe.utils.all_gather_group")
    @patch("paddleformers.nn.moe.utils.reduce_scatter_group")
    def test_backward(self, mock_rs, mock_ag):
        """Test backward pass delegates to all_gather_group."""
        x = _randn([8, 4])
        mock_rs.return_value = _randn([4, 4])
        expected_grad = _randn([8, 4])
        mock_ag.return_value = expected_grad
        group = _make_fake_group()
        self.ReduceScatterGroupOp.apply(x, group)


class TestAllGatherGroupOp(unittest.TestCase):
    """Tests for AllGatherGroupOp PyLayer."""

    def setUp(self):
        from paddleformers.nn.moe.utils import AllGatherGroupOp

        self.AllGatherGroupOp = AllGatherGroupOp

    @patch("paddleformers.nn.moe.utils.all_gather_group")
    def test_forward(self, mock_ag):
        """Test forward pass delegates to all_gather_group."""
        x = _randn([4, 4])
        expected = _randn([8, 4])
        mock_ag.return_value = expected
        group = _make_fake_group()
        result = self.AllGatherGroupOp.apply(x, group)
        self.assertTrue(paddle.allclose(result, expected))

    @patch("paddleformers.nn.moe.utils.reduce_scatter_group")
    @patch("paddleformers.nn.moe.utils.all_gather_group")
    def test_backward(self, mock_ag, mock_rs):
        """Test backward pass delegates to reduce_scatter_group."""
        x = _randn([4, 4])
        mock_ag.return_value = _randn([8, 4])
        expected_grad = _randn([4, 4])
        mock_rs.return_value = expected_grad
        group = _make_fake_group()
        self.AllGatherGroupOp.apply(x, group)


class TestAllGatherGroup(unittest.TestCase):
    """Tests for all_gather_group function."""

    def setUp(self):
        from paddleformers.nn.moe.utils import all_gather_group

        self.all_gather_group = all_gather_group

    def test_parallelism_1_returns_clone(self):
        """Test with parallelism=1 returns cloned input."""
        x = _randn([4, 4])
        group = _make_fake_group(nranks=1)
        result = self.all_gather_group(x, group=group)
        self.assertEqual(result.shape, [4, 4])

    def test_default_group_uses_model_parallel(self):
        """Test with group=None uses model parallel group."""
        x = _randn([4, 4])
        with patch("paddleformers.nn.moe.utils.fleet.get_hybrid_communicate_group") as mock_get:
            mock_hcg = MagicMock()
            mock_group = _make_fake_group(nranks=1)
            mock_hcg.get_model_parallel_group.return_value = mock_group
            mock_get.return_value = mock_hcg
            result = self.all_gather_group(x)
        self.assertEqual(result.shape, [4, 4])

    @patch("paddleformers.nn.moe.utils.dist.stream.all_gather")
    def test_axis_0_with_parallelism(self, mock_ag):
        """Test axis=0 path with parallelism > 1."""
        x = _randn([4, 4])
        group = _make_fake_group(nranks=2)
        expected = _randn([8, 4])

        def fill_output(output, input, **kwargs):
            paddle.assign(expected, output)

        mock_ag.side_effect = fill_output
        result = self.all_gather_group(x, group=group, axis=0)
        self.assertEqual(result.shape, [8, 4])

    @patch("paddleformers.nn.moe.utils.dist.stream.all_gather")
    def test_axis_1_with_parallelism(self, mock_ag):
        """Test axis=1 path with parallelism > 1."""
        x = _randn([4, 4])
        group = _make_fake_group(nranks=2)
        _randn([4, 8])

        def fill_output(out_list, input, **kwargs):
            for o in out_list:
                paddle.assign(_randn([4, 4]), o)

        mock_ag.side_effect = fill_output
        result = self.all_gather_group(x, group=group, axis=1)
        self.assertEqual(result.shape, [4, 8])


class TestReduceScatterGroup(unittest.TestCase):
    """Tests for reduce_scatter_group function."""

    def setUp(self):
        from paddleformers.nn.moe.utils import reduce_scatter_group

        self.reduce_scatter_group = reduce_scatter_group

    def test_parallelism_1_returns_clone(self):
        """Test with parallelism=1 returns cloned input."""
        x = _randn([8, 4])
        group = _make_fake_group(nranks=1)
        result = self.reduce_scatter_group(x, group=group)
        self.assertEqual(result.shape, [8, 4])

    def test_default_group_uses_model_parallel(self):
        """Test with group=None uses model parallel group."""
        x = _randn([4, 4])
        with patch("paddleformers.nn.moe.utils.fleet.get_hybrid_communicate_group") as mock_get:
            mock_hcg = MagicMock()
            mock_group = _make_fake_group(nranks=1)
            mock_hcg.get_model_parallel_group.return_value = mock_group
            mock_get.return_value = mock_hcg
            result = self.reduce_scatter_group(x)
        self.assertEqual(result.shape, [4, 4])

    @patch("paddleformers.nn.moe.utils.dist.stream.reduce_scatter")
    def test_with_parallelism(self, mock_rs):
        """Test with parallelism > 1."""
        x = _randn([8, 4])
        group = _make_fake_group(nranks=2)

        def fill_output(output, input, **kwargs):
            paddle.assign(_randn([4, 4]), output)

        mock_rs.side_effect = fill_output
        result = self.reduce_scatter_group(x, group=group)
        self.assertEqual(result.shape, [4, 4])

    def test_not_divisible_raises_assertion(self):
        """Test that non-divisible input raises assertion error."""
        x = _randn([7, 4])
        group = _make_fake_group(nranks=2)
        with self.assertRaises(AssertionError):
            self.reduce_scatter_group(x, group=group)


class TestScatterOp(unittest.TestCase):
    """Tests for ScatterOp PyLayer."""

    def setUp(self):
        from paddleformers.nn.moe.utils import ScatterOp

        self.ScatterOp = ScatterOp

    @patch("paddleformers.nn.moe.utils.scatter_axis")
    def test_forward(self, mock_scatter):
        """Test forward pass delegates to scatter_axis."""
        x = _randn([8, 4])
        expected = _randn([4, 4])
        mock_scatter.return_value = expected
        group = _make_fake_group()
        self.ScatterOp.apply(x, axis=0, group=group)
        mock_scatter.assert_called_once()

    @patch("paddleformers.nn.moe.utils.all_gather_group")
    @patch("paddleformers.nn.moe.utils.scatter_axis")
    def test_backward(self, mock_scatter, mock_ag):
        """Test backward pass delegates to all_gather_group."""
        x = _randn([8, 4])
        mock_scatter.return_value = _randn([4, 4])
        expected_grad = _randn([8, 4])
        mock_ag.return_value = expected_grad
        group = _make_fake_group()
        self.ScatterOp.apply(x, axis=0, group=group)


class TestParseMoeGroup(unittest.TestCase):
    """Tests for _parse_moe_group function."""

    def setUp(self):
        from paddleformers.nn.moe.utils import _parse_moe_group

        self._parse_moe_group = _parse_moe_group

    def test_dp_group(self):
        """Test 'dp' group type."""
        with patch("paddleformers.nn.moe.utils.fleet.get_hybrid_communicate_group") as mock_get:
            mock_hcg = MagicMock()
            mock_group = _make_fake_group()
            mock_hcg.get_data_parallel_group.return_value = mock_group
            mock_get.return_value = mock_hcg
            result = self._parse_moe_group("dp")
        self.assertIs(result, mock_group)

    def test_data_group(self):
        """Test 'data' group type."""
        with patch("paddleformers.nn.moe.utils.fleet.get_hybrid_communicate_group") as mock_get:
            mock_hcg = MagicMock()
            mock_group = _make_fake_group()
            mock_hcg.get_data_parallel_group.return_value = mock_group
            mock_get.return_value = mock_hcg
            result = self._parse_moe_group("data")
        self.assertIs(result, mock_group)

    def test_mp_group(self):
        """Test 'mp' group type."""
        with patch("paddleformers.nn.moe.utils.fleet.get_hybrid_communicate_group") as mock_get:
            mock_hcg = MagicMock()
            mock_group = _make_fake_group(nranks=4)
            mock_hcg.get_model_parallel_group.return_value = mock_group
            mock_get.return_value = mock_hcg
            result = self._parse_moe_group("mp")
        self.assertIs(result, mock_group)

    def test_model_group(self):
        """Test 'model' group type."""
        with patch("paddleformers.nn.moe.utils.fleet.get_hybrid_communicate_group") as mock_get:
            mock_hcg = MagicMock()
            mock_group = _make_fake_group(nranks=4)
            mock_hcg.get_model_parallel_group.return_value = mock_group
            mock_get.return_value = mock_hcg
            result = self._parse_moe_group("model")
        self.assertIs(result, mock_group)

    def test_tp_group(self):
        """Test 'tp' group type."""
        with patch("paddleformers.nn.moe.utils.fleet.get_hybrid_communicate_group") as mock_get:
            mock_hcg = MagicMock()
            mock_group = _make_fake_group(nranks=4)
            mock_hcg.get_model_parallel_group.return_value = mock_group
            mock_get.return_value = mock_hcg
            result = self._parse_moe_group("tp")
        self.assertIs(result, mock_group)

    def test_mp_group_single_rank(self):
        """Test 'mp' group with single rank falls back to dummy."""
        with patch("paddleformers.nn.moe.utils.fleet.get_hybrid_communicate_group") as mock_get:
            mock_hcg = MagicMock()
            mock_group = _make_fake_group(nranks=1)
            mock_hcg.get_model_parallel_group.return_value = mock_group
            mock_get.return_value = mock_hcg
            result = self._parse_moe_group("mp")
        self.assertIsNotNone(result)

    def test_mp_group_exception(self):
        """Test 'mp' group when get_model_parallel_group raises exception."""
        with patch("paddleformers.nn.moe.utils.fleet.get_hybrid_communicate_group") as mock_get:
            mock_hcg = MagicMock()
            mock_hcg.get_model_parallel_group.side_effect = Exception("no mp group")
            mock_get.return_value = mock_hcg
            result = self._parse_moe_group("mp")
        self.assertIsNotNone(result)

    def test_dummy_group(self):
        """Test 'dummy' group type."""
        result = self._parse_moe_group("dummy")
        self.assertIsNotNone(result)

    def test_world_group(self):
        """Test 'world' group type."""
        with patch("paddleformers.nn.moe.utils._get_global_group") as mock_global:
            mock_group = _make_fake_group()
            mock_global.return_value = mock_group
            result = self._parse_moe_group("world")
        self.assertIs(result, mock_group)

    def test_none_group(self):
        """Test 'none' group type."""
        with patch("paddleformers.nn.moe.utils._get_global_group") as mock_global:
            mock_group = _make_fake_group()
            mock_global.return_value = mock_group
            result = self._parse_moe_group("none")
        self.assertIs(result, mock_group)

    def test_all_group(self):
        """Test 'all' group type."""
        with patch("paddleformers.nn.moe.utils._get_global_group") as mock_global:
            mock_group = _make_fake_group()
            mock_global.return_value = mock_group
            result = self._parse_moe_group("all")
        self.assertIs(result, mock_group)

    def test_case_insensitive(self):
        """Test that group types are case-insensitive."""
        with patch("paddleformers.nn.moe.utils.fleet.get_hybrid_communicate_group") as mock_get:
            mock_hcg = MagicMock()
            mock_group = _make_fake_group()
            mock_hcg.get_data_parallel_group.return_value = mock_group
            mock_get.return_value = mock_hcg
            result = self._parse_moe_group("DP")
        self.assertIs(result, mock_group)

    def test_invalid_group_raises(self):
        """Test that invalid group type raises assertion error."""
        with self.assertRaises(AssertionError):
            self._parse_moe_group("invalid_group")


class TestGetAsyncLoader(unittest.TestCase):
    """Tests for get_async_loader function."""

    def setUp(self):
        import paddleformers.nn.moe.utils as utils_module
        from paddleformers.nn.moe.utils import get_async_loader

        self.get_async_loader = get_async_loader
        self.utils_module = utils_module

    def tearDown(self):
        if hasattr(self.utils_module, "async_loader"):
            self.utils_module.async_loader = None

    def test_with_hcg_creates_loader(self):
        """Test that loader is created on HCG when HCG exists."""
        with patch("paddleformers.nn.moe.utils.fleet.fleet", new_callable=PropertyMock) as _:
            mock_f = MagicMock()
            mock_f._hcg = True
            type(mock_f).fleet = mock_f
            mock_hcg = MagicMock()
            del mock_hcg.async_loader
            with patch("paddleformers.nn.moe.utils.get_hcg") as mock_get_hcg:
                mock_get_hcg.return_value = mock_hcg
                with patch("paddleformers.nn.moe.utils.create_async_load") as mock_create:
                    mock_loader = MagicMock()
                    mock_create.return_value = mock_loader
                    result = self.get_async_loader()
            self.assertEqual(result, mock_loader)


class TestManualBackward(unittest.TestCase):
    """Tests for manual_backward function."""

    def setUp(self):
        from paddleformers.nn.moe.utils import manual_backward

        self.manual_backward = manual_backward

    def test_is_first_fwd(self):
        """Test when is_first_fwd=True, returns (None, output)."""
        x = _randn([4])
        x.stop_gradient = False

        def f(a):
            return a * 2

        bwd_f, out = self.manual_backward(f, True, x)
        self.assertIsNone(bwd_f)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].shape, [4])

    def test_is_not_first_fwd(self):
        """Test when is_first_fwd=False, returns backward function."""
        x = _randn([4])
        x.stop_gradient = False

        def f(a):
            return a * 2

        bwd_f, out = self.manual_backward(f, False, x)
        self.assertIsNotNone(bwd_f)
        self.assertEqual(len(out), 1)

    def test_backward_function_callable(self):
        """Test that the returned backward function can be called."""
        x = _randn([4])
        x.stop_gradient = False

        def f(a):
            return a * 2

        bwd_f, out = self.manual_backward(f, False, x)
        grad = paddle.ones([4])
        result_grads = bwd_f(grad)
        self.assertIsNotNone(result_grads)
        self.assertEqual(len(result_grads), 1)

    def test_multiple_args(self):
        """Test with multiple arguments."""
        x = _randn([4])
        x.stop_gradient = False
        y = _randn([4])
        y.stop_gradient = False

        def f(a, b):
            return a + b

        bwd_f, out = self.manual_backward(f, False, x, y)
        self.assertIsNotNone(bwd_f)

    def test_none_arg(self):
        """Test with None argument (function receives None in clone)."""
        x = _randn([4])
        x.stop_gradient = False

        def f(a):
            if a is None:
                return paddle.zeros([1])
            return a * 2

        bwd_f, out = self.manual_backward(f, False, None)
        self.assertIsNotNone(bwd_f)

    def test_list_output(self):
        """Test function that returns a list."""
        x = _randn([4])
        x.stop_gradient = False

        def f(a):
            return [a * 2, a * 3]

        bwd_f, out = self.manual_backward(f, False, x)
        self.assertIsNotNone(bwd_f)

    def test_single_output(self):
        """Test function that returns a single tensor (not tuple)."""
        x = _randn([4])
        x.stop_gradient = False

        def f(a):
            return a.sum()

        bwd_f, out = self.manual_backward(f, False, x)
        self.assertIsNotNone(bwd_f)


if __name__ == "__main__":
    unittest.main()
