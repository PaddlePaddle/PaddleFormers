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


# Extra tests for paddleformers.fleet/refined_recompute/flash_attn.py
# Focus on: FlashMaskAttnCpAttention _first_fwd validation,
# FlashMaskAttnFunctor forward/backward structure

import unittest
from unittest.mock import MagicMock, patch

import paddle

try:
    from paddleformers.fleet.refined_recompute.flash_attn import (  # noqa: F401
        RefinedRcomputeFlashMaskCpAttention,
    )

    _MODULE_AVAILABLE = True
except (ImportError, ModuleNotFoundError, Exception):
    _MODULE_AVAILABLE = False


@unittest.skipUnless(
    _MODULE_AVAILABLE,
    "paddleformers.fleet.refined_recompute.flash_attn not available",
)
class TestFlashMaskCpAttentionQueryValidation(unittest.TestCase):
    """Tests for FlashMaskCpAttention query sequence length validation."""

    @unittest.skipUnless(
        _MODULE_AVAILABLE,
        "paddleformers.fleet.refined_recompute.flash_attn not available",
    )
    def test_odd_seq_len_asserts(self):
        """Test that odd query sequence length raises assertion."""
        from paddleformers.fleet.refined_recompute.flash_attn import (
            RefinedRcomputeFlashMaskCpAttention,
        )

        rr = RefinedRcomputeFlashMaskCpAttention()
        # _first_fwd requires _hcg (hybrid communication group) which
        # is only available in distributed/multi-card environments
        if not hasattr(rr, "_hcg"):
            self.skipTest("requires distributed environment with _hcg")
        with self.assertRaises(AssertionError):
            rr._first_fwd(
                paddle.randn([1, 7, 4, 16]),  # odd seq_len
                paddle.randn([1, 7, 4, 16]),
                paddle.randn([1, 7, 4, 16]),
                paddle.randint(0, 10, [1, 4, 7]),
            )


@unittest.skipUnless(
    _MODULE_AVAILABLE,
    "paddleformers.fleet.refined_recompute.flash_attn not available",
)
class TestFlashAttnFunctorForwardVersions(unittest.TestCase):
    """Tests for FlashAttnFunctor forward with different FA versions."""

    def test_forward_invalid_version_raises(self):
        """Test that invalid FA version raises ValueError in forward."""
        from paddleformers.fleet.refined_recompute.flash_attn import (
            FlashAttnFunctor,
        )

        ctx = MagicMock()
        q = paddle.randn([1, 4, 8], dtype=paddle.bfloat16)
        k = paddle.randn([1, 4, 8], dtype=paddle.bfloat16)
        v = paddle.randn([1, 4, 8], dtype=paddle.bfloat16)
        hold_tensors = {
            "result_attention": paddle.randn([1, 4, 8]),
            "softmax_lse": paddle.randn([1, 4]),
            "causal": True,
        }

        with (
            patch(
                "paddleformers.fleet.refined_recompute.flash_attn._get_fa_version",
                return_value=99,
            ),
            self.assertRaises(ValueError),
        ):
            FlashAttnFunctor.forward(ctx, q, k, v, hold_tensors)


@unittest.skipUnless(
    _MODULE_AVAILABLE,
    "paddleformers.fleet.refined_recompute.flash_attn not available",
)
class TestFlashAttnFunctorBackwardVersions(unittest.TestCase):
    """Tests for FlashAttnFunctor backward with different FA versions."""

    def test_backward_invalid_version_raises(self):
        """Test that invalid FA version raises ValueError in backward."""
        from paddleformers.fleet.refined_recompute.flash_attn import (
            FlashAttnFunctor,
        )

        ctx = MagicMock()
        ctx.fa_version = 99
        ctx.saved_tensor = MagicMock(
            return_value=[paddle.randn([1]) for _ in range(8)]
        )

        with self.assertRaises(ValueError):
            FlashAttnFunctor.backward(ctx, paddle.randn([1, 4, 8]))


@unittest.skipUnless(
    _MODULE_AVAILABLE,
    "paddleformers.fleet.refined_recompute.flash_attn not available",
)
class TestFlashMaskAttnFunctorForwardVersions(unittest.TestCase):
    """Tests for FlashMaskAttnFunctor forward with different FA versions."""

    def test_forward_invalid_version_raises(self):
        """Test that invalid FA version raises ValueError in forward."""
        from paddleformers.fleet.refined_recompute.flash_attn import (
            FlashMaskAttnFunctor,
        )

        ctx = MagicMock()
        q = paddle.randn([1, 4, 8], dtype=paddle.bfloat16)
        k = paddle.randn([1, 4, 8], dtype=paddle.bfloat16)
        v = paddle.randn([1, 4, 8], dtype=paddle.bfloat16)
        startend = paddle.randint(0, 10, [1, 4])
        hold_tensors = {
            "result_attention": paddle.randn([1, 4, 8]),
            "softmax_lse": paddle.randn([1, 4]),
            "causal": True,
        }

        with (
            patch(
                "paddleformers.fleet.refined_recompute.flash_attn._get_fa_version",
                return_value=99,
            ),
            self.assertRaises(ValueError),
        ):
            FlashMaskAttnFunctor.forward(ctx, q, k, v, startend, hold_tensors)


@unittest.skipUnless(
    _MODULE_AVAILABLE,
    "paddleformers.fleet.refined_recompute.flash_attn not available",
)
class TestFlashMaskAttnFunctorBackwardVersions(unittest.TestCase):
    """Tests for FlashMaskAttnFunctor backward with different FA versions."""

    def test_backward_invalid_version_raises(self):
        """Test that invalid FA version raises ValueError in backward."""
        from paddleformers.fleet.refined_recompute.flash_attn import (
            FlashMaskAttnFunctor,
        )

        ctx = MagicMock()
        ctx.fa_version = 99
        ctx.saved_tensor = MagicMock(
            return_value=[paddle.randn([1]) for _ in range(9)]
        )

        with self.assertRaises(ValueError):
            FlashMaskAttnFunctor.backward(ctx, paddle.randn([1, 4, 8]))


@unittest.skipUnless(
    _MODULE_AVAILABLE,
    "paddleformers.fleet.refined_recompute.flash_attn not available",
)
class TestFlashMaskCpAttentionForwardDispatch(unittest.TestCase):
    """Tests for FlashMaskCpAttention forward dispatching."""

    def test_forward_dispatches_to_first_fwd_when_no_grad(self):
        """Test that forward dispatches to _first_fwd when no grad."""
        from paddleformers.fleet.refined_recompute.flash_attn import (
            RefinedRcomputeFlashMaskCpAttention,
        )

        rr = RefinedRcomputeFlashMaskCpAttention()
        rr._first_fwd = MagicMock(return_value=paddle.randn([1, 4, 8]))

        with patch(
            "paddleformers.fleet.refined_recompute.flash_attn.framework._dygraph_tracer"
        ) as mock_tracer:
            mock_tracer.return_value._has_grad = False
            rr.forward(
                None,
                paddle.randn([1, 4, 8]),
                paddle.randn([1, 4, 8]),
                paddle.randn([1, 4, 8]),
                paddle.randint(0, 10, [1, 4]),
            )
            rr._first_fwd.assert_called_once()


if __name__ == "__main__":
    unittest.main()
