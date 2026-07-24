# Copyright (c) 2026 PaddleFaddle Authors. All Rights Reserved.
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

import paddle

# fmt: off
# Parameters dicts used to control which branch the signature check enters.
# The code does:  if "group" in sig_params  ->  elif "block_mask" in sig_params  ->  else
_SIG_HAS_GROUP      = {"block_mask": MagicMock(), "group": MagicMock()}
_SIG_HAS_BLOCK_MASK = {"block_mask": MagicMock()}
# fmt: on


# ---------------------------------------------------------------------------
# 1) cp_flashmask_allgatherkv_balance_backward  (context_parallel_utils)
# ---------------------------------------------------------------------------


class TestCpFlashmaskBackwardDispatch(unittest.TestCase):
    """Cover the "group" and "block_mask" branches in
    cp_flashmask_allgatherkv_balance_backward (fa_version==3).
    The "else" branch is naturally covered by CI."""

    def _call_backward(self, sig_params):
        from paddleformers.fleet.context_parallel_utils import (
            cp_flashmask_allgatherkv_balance_backward,
        )

        B, S, H, D = 1, 8, 2, 16
        q = paddle.randn([B, S, H, D])
        k = paddle.randn([B, S, H, D])
        v = paddle.randn([B, S, H, D])
        indices = paddle.zeros([B, 2, S], dtype="int64")
        out = paddle.randn([B, S, H, D])
        lse = paddle.randn([B, H, S])
        out_grad = paddle.randn([B, S, H, D])

        config = MagicMock(fa_version=3, deterministic_mode=False)
        group = MagicMock()
        dummy = paddle.randn([B, S, H, D])
        mock_v2_grad = MagicMock(return_value=(dummy, dummy, dummy))

        with (
            patch(
                "paddleformers.fleet.context_parallel_utils.inspect.signature"
            ) as mock_sig,
            patch(
                "paddleformers.fleet.context_parallel_utils.all_gather_balance",
                side_effect=lambda x, **kw: x,
            ),
            patch(
                "paddleformers.fleet.context_parallel_utils.reduce_scatter_any_axis_balance",
                side_effect=lambda x, **kw: x,
            ),
            patch(
                "paddle._C_ops.flashmask_attention_v2_grad",
                mock_v2_grad,
                create=True,
            ),
        ):
            mock_sig.return_value.parameters = sig_params
            cp_flashmask_allgatherkv_balance_backward(
                q,
                k,
                v,
                indices,
                out,
                lse,
                out_grad,
                None,
                group,
                False,
                config.fa_version,
                None,  # softmax_scale
            )

        return mock_v2_grad

    def test_group_branch(self):
        m = self._call_backward(_SIG_HAS_GROUP)
        args = m.call_args[0]
        self.assertEqual(len(args), 12)
        self.assertEqual(args[-2], 0)  # rank
        self.assertEqual(args[-1], 1)  # nranks

    def test_block_mask_branch(self):
        m = self._call_backward(_SIG_HAS_BLOCK_MASK)
        args = m.call_args[0]
        self.assertEqual(len(args), 10)
        self.assertIsNone(args[6])  # block_mask


# ---------------------------------------------------------------------------
# 2) FlashMaskAttnFunctor.backward  (flash_attn)
# ---------------------------------------------------------------------------


class TestFlashMaskAttnFunctorBackwardDispatch(unittest.TestCase):
    """Cover the "group" and "block_mask" branches in
    FlashMaskAttnFunctor.backward (fa_version==3).
    The "else" branch is naturally covered by CI."""

    def _call_backward(self, sig_params):
        from paddleformers.fleet.refined_recompute.flash_attn import (
            FlashMaskAttnFunctor,
        )

        B, S, H, D = 1, 8, 2, 16
        q = paddle.randn([B, S, H, D])
        k = paddle.randn([B, S, H, D])
        v = paddle.randn([B, S, H, D])
        indices = paddle.zeros([B, 2, S], dtype="int64")
        out = paddle.randn([B, S, H, D])
        lse = paddle.randn([B, H, S])
        grad = paddle.randn([B, S, H, D])
        dummy = paddle.randn([B, S, H, D])

        out._clear_dataptr = MagicMock()
        lse._clear_dataptr = MagicMock()

        mock_ctx = MagicMock()
        mock_ctx.fa_version = 3
        mock_ctx.sink_requires_grad = False
        mock_ctx.saved_tensor.return_value = (q, k, v, indices, out, lse, False)

        mock_v2_grad = MagicMock(return_value=(dummy, dummy, dummy))

        with (
            patch(
                "paddleformers.fleet.refined_recompute.flash_attn.inspect.signature"
            ) as mock_sig,
            patch(
                "paddleformers.fleet.refined_recompute.flash_attn._C_ops.flashmask_attention_v2_grad",
                mock_v2_grad,
                create=True,
            ),
        ):
            mock_sig.return_value.parameters = sig_params
            FlashMaskAttnFunctor.backward(mock_ctx, grad)

        return mock_v2_grad

    def test_group_branch(self):
        m = self._call_backward(_SIG_HAS_GROUP)
        args = m.call_args[0]
        self.assertEqual(len(args), 12)
        self.assertEqual(args[-2], 0)  # rank
        self.assertEqual(args[-1], 1)  # nranks

    def test_block_mask_branch(self):
        m = self._call_backward(_SIG_HAS_BLOCK_MASK)
        args = m.call_args[0]
        self.assertEqual(len(args), 10)
        self.assertIsNone(args[6])  # block_mask


# ---------------------------------------------------------------------------
# 3) RefinedRcomputeFlashMaskAttention._first_fwd  (flash_attn)
# ---------------------------------------------------------------------------


class TestRefinedRecomputeFirstFwdDispatch(unittest.TestCase):
    """Cover the "group" branch in RefinedRcomputeFlashMaskAttention._first_fwd
    (fa_version==3). The "block_mask" and "else" branches are already covered
    by test_ai_flash_attn.py."""

    @patch(
        "paddleformers.fleet.refined_recompute.flash_attn.framework._dygraph_tracer"
    )
    @patch(
        "paddleformers.fleet.refined_recompute.flash_attn._C_ops.flashmask_attention_v2",
        create=True,
    )
    @patch(
        "paddleformers.fleet.refined_recompute.flash_attn.get_fa_version",
        return_value=3,
    )
    @patch("paddleformers.fleet.refined_recompute.flash_attn.inspect.signature")
    def test_group_branch(self, mock_sig, mock_version, mock_v2, mock_tracer):
        mock_tracer_obj = MagicMock()
        mock_tracer_obj._has_grad = False
        mock_tracer.return_value = mock_tracer_obj

        mock_sig.return_value.parameters = _SIG_HAS_GROUP

        B, S, H, D = 1, 8, 2, 16
        q = paddle.randn([B, S, H, D], dtype=paddle.bfloat16)
        k = paddle.randn([B, S, H, D], dtype=paddle.bfloat16)
        v = paddle.randn([B, S, H, D], dtype=paddle.bfloat16)
        startend = paddle.zeros([B, 2, S], dtype="int64")

        mock_v2.return_value = (
            paddle.randn([B, S, H, D], dtype=paddle.bfloat16),
            paddle.randn([B, H, S], dtype=paddle.float32),
        )

        from paddleformers.fleet.refined_recompute.flash_attn import (
            RefinedRcomputeFlashMaskAttention,
        )

        obj = RefinedRcomputeFlashMaskAttention()
        obj.forward(q, k, v, startend, causal=False)

        args = mock_v2.call_args[0]
        self.assertEqual(len(args), 10)
        self.assertEqual(args[-2], 0)  # rank
        self.assertEqual(args[-1], 1)  # nranks


if __name__ == "__main__":
    unittest.main()
