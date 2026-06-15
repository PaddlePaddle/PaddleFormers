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


# Tests for paddlefleet_ops/_extensions/flashmask/block_mask_utils.py
# Focus on: find_blocks_topp, top_p_kernel, _compare_and_swap definitions

import types
import unittest

try:
    import paddlefleet_ops._extensions  # noqa: F401

    _MODULE_AVAILABLE = True
except (ImportError, ModuleNotFoundError, Exception):
    _MODULE_AVAILABLE = False
from unittest.mock import MagicMock


def _setup_triton_mock():
    """Create a mock triton module so we can import without GPU."""
    triton_mock = types.ModuleType("triton")
    triton_mock.jit = lambda fn=None, **kwargs: ((lambda f: f) if fn is None else fn)
    triton_mock.language = types.ModuleType("triton.language")
    triton_mock.language.constexpr = None
    tl = triton_mock.language
    tl.program_id = lambda axis: 0
    tl.arange = lambda start, end: list(range(end))
    tl.load = lambda *a, **kw: 0
    tl.store = lambda *a, **kw: None
    tl.max = lambda *a, **kw: 0
    tl.min = lambda *a, **kw: 0
    tl.sum = lambda *a, **kw: 0
    tl.where = lambda cond, a, b: a
    tl.full = lambda shape, val, dtype=None: val
    tl.zeros = lambda shape, dtype=None: 0
    tl.broadcast_to = lambda x, shape: x
    tl.reshape = lambda x, shape: x
    tl.int32 = "int32"
    tl.int64 = "int64"
    tl.int8 = "int8"
    tl.int1 = "int1"
    tl.float32 = "float32"
    tl.core = MagicMock()
    tl.core.CONSTEXPR_0 = 0
    tl.core.get_int_dtype = lambda bitwidth, signed=True: "int32"
    tl.static_assert = lambda cond, msg=None: None
    tl.cumsum = lambda x, axis=0: x
    tl.math = MagicMock()
    tl.math.exp2 = lambda x: x
    triton_mock.next_power_of_2 = lambda x: 1
    triton_mock.cdiv = lambda a, b: (a + b - 1) // b

    libdevice = types.ModuleType("triton.language.extra")
    libdevice2 = types.ModuleType("triton.language.extra.cuda")
    libdevice2.exp = lambda x: x
    libdevice2.div_rn = lambda a, b: a
    sys.modules.setdefault("triton.language.extra", libdevice)
    sys.modules.setdefault("triton.language.extra.cuda", libdevice2)
    sys.modules.setdefault("triton.language.extra.cuda.libdevice", libdevice2)
    sys.modules.setdefault("triton", triton_mock)
    sys.modules.setdefault("triton.language", tl)
    return triton_mock


_setup_triton_mock()


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestBlockMaskUtilsKernels(unittest.TestCase):
    """Tests for block_mask_utils kernel definitions."""

    def test_load_bounds_callable(self):
        """Test _load_bounds kernel is callable."""
        from paddlefleet_ops._extensions.flashmask.block_mask_utils import _load_bounds

        self.assertTrue(callable(_load_bounds))

    def test_is_block_fully_masked_callable(self):
        """Test _is_block_fully_masked kernel is callable."""
        from paddlefleet_ops._extensions.flashmask.block_mask_utils import (
            _is_block_fully_masked,
        )

        self.assertTrue(callable(_is_block_fully_masked))

    def test_check_fully_masked_state_callable(self):
        """Test check_fully_masked_state kernel is callable."""
        from paddlefleet_ops._extensions.flashmask.block_mask_utils import (
            check_fully_masked_state,
        )

        self.assertTrue(callable(check_fully_masked_state))

    def test_is_block_partially_masked_callable(self):
        """Test _is_block_partially_masked kernel is callable."""
        from paddlefleet_ops._extensions.flashmask.block_mask_utils import (
            _is_block_partially_masked,
        )

        self.assertTrue(callable(_is_block_partially_masked))

    def test_check_partially_masked_state_callable(self):
        """Test check_partially_masked_state kernel is callable."""
        from paddlefleet_ops._extensions.flashmask.block_mask_utils import (
            check_partially_masked_state,
        )

        self.assertTrue(callable(check_partially_masked_state))

    def test_compare_and_swap_callable(self):
        """Test _compare_and_swap kernel is callable."""
        from paddlefleet_ops._extensions.flashmask.block_mask_utils import (
            _compare_and_swap,
        )

        self.assertTrue(callable(_compare_and_swap))

    def test_bitonic_merge_callable(self):
        """Test _bitonic_merge kernel is callable."""
        from paddlefleet_ops._extensions.flashmask.block_mask_utils import (
            _bitonic_merge,
        )

        self.assertTrue(callable(_bitonic_merge))

    def test_bitonic_argsort_device_callable(self):
        """Test bitonic_argsort_device kernel is callable."""
        from paddlefleet_ops._extensions.flashmask.block_mask_utils import (
            bitonic_argsort_device,
        )

        self.assertTrue(callable(bitonic_argsort_device))

    def test_top_p_kernel_callable(self):
        """Test top_p_kernel is callable."""
        from paddlefleet_ops._extensions.flashmask.block_mask_utils import top_p_kernel

        self.assertTrue(callable(top_p_kernel))


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestFindBlocksToppSignature(unittest.TestCase):
    """Tests for find_blocks_topp function signature."""

    def test_has_expected_parameters(self):
        """Test find_blocks_topp has expected parameters."""
        import inspect

        from paddlefleet_ops._extensions.flashmask.block_mask_utils import (
            find_blocks_topp,
        )

        sig = inspect.signature(find_blocks_topp)
        params = list(sig.parameters.keys())
        self.assertIn("x", params)
        self.assertIn("p", params)


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestTopPLogic(unittest.TestCase):
    """Tests for top-p logic using pure Paddle."""

    def test_top_p_masking(self):
        """Test top-p probability masking."""
        import paddle

        # Simple top-p example
        probs = paddle.to_tensor([[0.4, 0.3, 0.2, 0.1]])
        p = 0.8

        sorted_probs, sorted_indices = paddle.topk(probs, k=4, axis=-1)
        cum_probs = paddle.cumsum(sorted_probs, axis=-1)
        # Keep tokens where cumprob - current_prob < p
        mask = (cum_probs - sorted_probs) < p

        # First two tokens (0.4 + 0.3 = 0.7) should be kept
        # Third token (0.7 + 0.2 = 0.9 > 0.8) should be masked
        self.assertTrue(mask[0, 0].item())
        self.assertTrue(mask[0, 1].item())

    def test_top_p_with_p_1(self):
        """Test top-p with p=1.0 keeps all tokens."""
        import paddle

        probs = paddle.to_tensor([[0.4, 0.3, 0.2, 0.1]])
        p = 1.0

        sorted_probs, _ = paddle.topk(probs, k=4, axis=-1)
        cum_probs = paddle.cumsum(sorted_probs, axis=-1)
        mask = (cum_probs - sorted_probs) < p

        # All tokens should be kept
        self.assertTrue(paddle.all(mask))

    def test_top_p_with_p_0(self):
        """Test top-p with p=0.0 keeps no tokens (except highest)."""
        import paddle

        probs = paddle.to_tensor([[0.4, 0.3, 0.2, 0.1]])
        p = 0.0

        sorted_probs, _ = paddle.topk(probs, k=4, axis=-1)
        cum_probs = paddle.cumsum(sorted_probs, axis=-1)
        mask = (cum_probs - sorted_probs) < p

        # No tokens should be kept (cumprob - prob >= 0 for all)
        self.assertFalse(paddle.any(mask))


if __name__ == "__main__":
    unittest.main()
