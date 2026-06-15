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

import functools
import unittest
import warnings
from unittest.mock import MagicMock

import paddle


class TestEnsureDivisibility(unittest.TestCase):
    """Tests for ensure_divisibility."""

    def test_divisible(self):
        """Test with divisible numbers."""
        from paddleformers.fleet.utils import ensure_divisibility

        # Should not raise
        ensure_divisibility(10, 5)

    def test_not_divisible_raises(self):
        """Test with non-divisible numbers raises."""
        from paddleformers.fleet.utils import ensure_divisibility

        with self.assertRaises(AssertionError):
            ensure_divisibility(10, 3)

    def test_zero_numerator(self):
        """Test with zero numerator."""
        from paddleformers.fleet.utils import ensure_divisibility

        ensure_divisibility(0, 5)  # 0 is divisible by anything


class TestDivide(unittest.TestCase):
    """Tests for divide function."""

    def test_divide_correct(self):
        """Test divide returns correct result."""
        from paddleformers.fleet.utils import divide

        self.assertEqual(divide(10, 5), 2)

    def test_divide_not_divisible_raises(self):
        """Test divide raises when not divisible."""
        from paddleformers.fleet.utils import divide

        with self.assertRaises(AssertionError):
            divide(10, 3)


class TestInitMethodNormal(unittest.TestCase):
    """Tests for init_method_normal."""

    def test_returns_callable(self):
        """Test init_method_normal returns a callable."""
        from paddleformers.fleet.utils import init_method_normal

        result = init_method_normal(0.02)
        self.assertTrue(callable(result))

    def test_is_functools_partial(self):
        """Test init_method_normal returns a partial."""
        from paddleformers.fleet.utils import init_method_normal

        result = init_method_normal(0.02)
        self.assertIsInstance(result, functools.partial)


class TestScaledInitMethodNormal(unittest.TestCase):
    """Tests for scaled_init_method_normal."""

    def test_returns_callable(self):
        """Test scaled_init_method_normal returns a callable."""
        from paddleformers.fleet.utils import scaled_init_method_normal

        result = scaled_init_method_normal(0.02, 32)
        self.assertTrue(callable(result))

    def test_is_functools_partial(self):
        """Test scaled_init_method_normal returns a partial."""
        from paddleformers.fleet.utils import scaled_init_method_normal

        result = scaled_init_method_normal(0.02, 32)
        self.assertIsInstance(result, functools.partial)


class TestPrepareInputTensorsForWgradCompute(unittest.TestCase):
    """Tests for prepare_input_tensors_for_wgrad_compute."""

    def test_2d_input(self):
        """Test with 2D input tensors."""
        from paddleformers.fleet.utils import prepare_input_tensors_for_wgrad_compute

        grad = paddle.randn([4, 8])
        inp = paddle.randn([4, 8])
        g, i = prepare_input_tensors_for_wgrad_compute(grad, inp)
        self.assertEqual(g.shape, [4, 8])
        self.assertEqual(i.shape, [4, 8])

    def test_3d_input_reshaped(self):
        """Test with 3D input tensors get reshaped to 2D."""
        from paddleformers.fleet.utils import prepare_input_tensors_for_wgrad_compute

        grad = paddle.randn([2, 4, 8])
        inp = paddle.randn([2, 4, 8])
        g, i = prepare_input_tensors_for_wgrad_compute(grad, inp)
        self.assertEqual(g.shape, [8, 8])
        self.assertEqual(i.shape, [8, 8])


class TestDeprecateInferenceParams(unittest.TestCase):
    """Tests for deprecate_inference_params."""

    def test_with_context_returns_context(self):
        """Test returns inference_context when both are provided."""
        from paddleformers.fleet.utils import deprecate_inference_params

        ctx = MagicMock()
        params = MagicMock()
        result = deprecate_inference_params(ctx, params)
        self.assertEqual(result, ctx)

    def test_no_context_with_params_warns(self):
        """Test warns when inference_context is None but inference_params is not."""
        from paddleformers.fleet.utils import deprecate_inference_params

        params = MagicMock()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = deprecate_inference_params(None, params)
            self.assertEqual(len(w), 1)
            self.assertIn("inference_params", str(w[0].message))

    def test_both_none_returns_none(self):
        """Test returns None when both are None."""
        from paddleformers.fleet.utils import deprecate_inference_params

        result = deprecate_inference_params(None, None)
        self.assertIsNone(result)


class TestGetPgSize(unittest.TestCase):
    """Tests for get_pg_size."""

    def test_group_none_returns_one(self):
        """Test returns 1 when group is None."""
        from paddleformers.fleet.utils import get_pg_size

        result = get_pg_size(None)
        self.assertEqual(result, 1)


class TestGetPgRank(unittest.TestCase):
    """Tests for get_pg_rank."""

    def test_group_none_returns_zero(self):
        """Test returns 0 when group is None."""
        from paddleformers.fleet.utils import get_pg_rank

        result = get_pg_rank(None)
        self.assertEqual(result, 0)


if __name__ == "__main__":
    unittest.main()
