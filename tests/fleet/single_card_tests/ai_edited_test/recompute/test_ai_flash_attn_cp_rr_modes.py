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
import unittest
from unittest.mock import MagicMock, patch

_REPO_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
        )
    )
)
sys.path.insert(0, _REPO_ROOT)

import paddle

from paddleformers.fleet.refined_recompute import flash_attn as fa


class FakeCtx:
    def save_for_backward(self, *tensors):
        self._saved = tensors

    def saved_tensor(self):
        return self._saved


class CpGroup:
    rank = 1
    nranks = 2
    world_size = 2


class TestFlashMaskAttnCpFunctor(unittest.TestCase):
    def test_backward_passes_mode_and_scale(self):
        q = paddle.randn([1, 4, 2, 4])
        k = paddle.randn([1, 4, 2, 4])
        v = paddle.randn([1, 4, 2, 4])
        out = paddle.randn([1, 4, 2, 4])
        lse = paddle.randn([1, 2, 4])
        startend = paddle.zeros([1, 1, 4, 2], dtype="int32")
        hold = {
            "mode": "contiguous_allgather",
            "result_attention": out,
            "softmax_lse": lse,
            "startend_row_indices": startend,
            "fa_version": 4,
            "group": CpGroup(),
            "causal": False,
            "softmax_scale": 0.25,
        }
        ctx = FakeCtx()

        self.assertIs(
            fa.FlashMaskAttnCpFunctor.forward(ctx, q, k, v, None, hold), out
        )
        with patch.object(
            fa,
            "cp_flashmask_allgatherkv_balance_backward",
            return_value=(q, k, v, None),
        ) as mock_backward:
            grads = fa.FlashMaskAttnCpFunctor.backward(
                ctx, paddle.ones_like(out)
            )

        self.assertIs(grads[0], q)
        self.assertIs(grads[1], k)
        self.assertIs(grads[2], v)
        self.assertEqual(mock_backward.call_args.args[-2], 0.25)
        self.assertEqual(
            mock_backward.call_args.args[-1], "contiguous_allgather"
        )

    def test_backward_returns_sink_grad_when_sink_requires_grad(self):
        q = paddle.randn([1, 4, 2, 4])
        k = paddle.randn([1, 4, 2, 4])
        v = paddle.randn([1, 4, 2, 4])
        out = paddle.randn([1, 4, 2, 4])
        sink = paddle.randn([2])
        sink.stop_gradient = False
        sink_grad = paddle.randn([2])
        hold = {
            "mode": "contiguous_allgather",
            "result_attention": out,
            "softmax_lse": paddle.randn([1, 2, 4]),
            "startend_row_indices": paddle.zeros([1, 1, 4, 2], dtype="int32"),
            "fa_version": 4,
            "group": CpGroup(),
            "causal": False,
        }
        ctx = FakeCtx()

        fa.FlashMaskAttnCpFunctor.forward(ctx, q, k, v, sink, hold)
        with patch.object(
            fa,
            "cp_flashmask_allgatherkv_balance_backward",
            return_value=(q, k, v, sink_grad),
        ):
            grads = fa.FlashMaskAttnCpFunctor.backward(
                ctx, paddle.ones_like(out)
            )

        self.assertIs(grads[0], q)
        self.assertIs(grads[1], k)
        self.assertIs(grads[2], v)
        self.assertIs(grads[3], sink_grad)


class TestFlashMaskAttnFunctor(unittest.TestCase):
    def test_v3_backward_passes_scalar_softmax_scale(self):
        q = paddle.randn([1, 4, 2, 4])
        k = paddle.randn([1, 4, 2, 4])
        v = paddle.randn([1, 4, 2, 4])
        out = paddle.randn([1, 4, 2, 4])
        lse = paddle.randn([1, 2, 4])
        startend = paddle.zeros([1, 1, 4, 2], dtype="int32")
        hold = {
            "result_attention": out,
            "softmax_lse": lse,
            "causal": False,
            "softmax_scale": None,
        }
        ctx = FakeCtx()

        def fake_flashmask_attention(query, key, value, block_mask=None):
            return None

        with patch.object(fa, "get_fa_version", return_value=3):
            self.assertIs(
                fa.FlashMaskAttnFunctor.forward(
                    ctx, q, k, v, startend, None, hold
                ),
                out,
            )
        with (
            patch.object(fa, "flashmask_attention", fake_flashmask_attention),
            patch.object(fa, "_C_ops") as mock_c_ops,
        ):
            mock_c_ops.flashmask_attention_v2_grad.return_value = (q, k, v)
            grads = fa.FlashMaskAttnFunctor.backward(ctx, paddle.ones_like(out))

        self.assertIs(grads[0], q)
        scale_arg = mock_c_ops.flashmask_attention_v2_grad.call_args.args[-2]
        self.assertIsInstance(scale_arg, float)
        self.assertNotIsInstance(scale_arg, tuple)


class TestFlashMaskSwaP2PFunctor(unittest.TestCase):
    def test_forward_returns_saved_output_and_backward_uses_saved_tensors(self):
        q = paddle.randn([1, 4, 2, 4])
        k = paddle.randn([1, 4, 2, 4])
        v = paddle.randn([1, 4, 2, 4])
        out = paddle.randn([1, 4, 2, 4])
        lse = paddle.randn([1, 2, 4])
        hold = {
            "result_attention": out,
            "softmax_lse": lse,
            "recv_key": paddle.randn([1, 2, 2, 4]),
            "recv_value": paddle.randn([1, 2, 2, 4]),
            "startend_row_indices": paddle.zeros([1, 1, 4, 2], dtype="int32"),
            "group": CpGroup(),
            "causal": False,
            "softmax_scale": 0.5,
            "window_size": 2,
        }
        ctx = FakeCtx()

        self.assertIs(
            fa.FlashMaskSwaP2PFunctor.forward(ctx, q, k, v, None, hold), out
        )
        with patch.object(
            fa,
            "cp_flashmask_swa_p2p_backward",
            return_value=(q, k, v, None),
        ) as mock_backward:
            grads = fa.FlashMaskSwaP2PFunctor.backward(
                ctx, paddle.ones_like(out)
            )

        self.assertIs(grads[0], q)
        self.assertIs(grads[1], k)
        self.assertIs(grads[2], v)
        mock_backward.assert_called_once()
        self.assertIs(mock_backward.call_args.args[0], q)
        self.assertIs(mock_backward.call_args.args[3], hold["recv_key"])
        self.assertEqual(mock_backward.call_args.args[-1], 2)

    def test_backward_returns_sink_grad_when_sink_requires_grad(self):
        q = paddle.randn([1, 4, 2, 4])
        k = paddle.randn([1, 4, 2, 4])
        v = paddle.randn([1, 4, 2, 4])
        out = paddle.randn([1, 4, 2, 4])
        sink = paddle.randn([2])
        sink.stop_gradient = False
        sink_grad = paddle.randn([2])
        hold = {
            "result_attention": out,
            "softmax_lse": paddle.randn([1, 2, 4]),
            "recv_key": paddle.randn([1, 2, 2, 4]),
            "recv_value": paddle.randn([1, 2, 2, 4]),
            "startend_row_indices": paddle.zeros([1, 1, 4, 2], dtype="int32"),
            "group": CpGroup(),
            "causal": False,
            "window_size": 2,
        }
        ctx = FakeCtx()

        fa.FlashMaskSwaP2PFunctor.forward(ctx, q, k, v, sink, hold)
        with patch.object(
            fa,
            "cp_flashmask_swa_p2p_backward",
            return_value=(q, k, v, sink_grad),
        ):
            grads = fa.FlashMaskSwaP2PFunctor.backward(
                ctx, paddle.ones_like(out)
            )

        self.assertIs(grads[0], q)
        self.assertIs(grads[1], k)
        self.assertIs(grads[2], v)
        self.assertIs(grads[3], sink_grad)


class TestUlyssesHelpers(unittest.TestCase):
    def test_slice_ulysses_mask_heads_broadcast_and_per_head(self):
        broadcast = paddle.zeros([1, 1, 4, 2], dtype="int32")
        self.assertIs(
            fa.slice_ulysses_mask_heads(broadcast, 4, CpGroup()), broadcast
        )

        per_head = paddle.arange(1 * 4 * 4 * 2, dtype="int32").reshape(
            [1, 4, 4, 2]
        )
        sliced = fa.slice_ulysses_mask_heads(per_head, 4, CpGroup())
        self.assertEqual(list(sliced.shape), [1, 2, 4, 2])
        self.assertTrue(
            bool(paddle.all(sliced == per_head[:, 2:4, :, :]).item())
        )

    def test_ulysses_local_first_forward_versions_save_rr_tensors(self):
        q = paddle.randn([1, 4, 2, 4])
        out = paddle.randn([1, 4, 2, 4])
        lse = paddle.randn([1, 2, 4])
        softmax = paddle.randn([1, 2, 4])
        seed = paddle.zeros([1], dtype="int64")
        startend = paddle.zeros([1, 1, 4, 2], dtype="int32")

        def fake_flashmask_attention(query, key, value, block_mask=None):
            return None

        cases = (
            (2, "flashmask_attention", (out, softmax, lse, seed)),
            (3, "flashmask_attention_v2", (out, lse)),
            (4, "_flash_attn_fwd", (out, lse)),
        )
        for version, target, return_value in cases:
            with self.subTest(version=version):
                with patch.object(fa, "get_fa_version", return_value=version):
                    if version == 4:
                        patcher = patch.object(
                            fa, target, return_value=return_value, create=True
                        )
                    else:
                        patcher = patch.object(fa, "_C_ops")
                    with patcher as mock_op:
                        if version != 4:
                            getattr(mock_op, target).return_value = return_value
                        with patch.object(
                            fa, "flashmask_attention", fake_flashmask_attention
                        ):
                            result, hold = fa.ulysses_local_flashmask_first_fwd(
                                q, q, q, startend, False, None
                            )

                self.assertIs(result, out)
                self.assertIs(hold["result_attention"], out)
                self.assertIs(hold["softmax_lse"], lse)
                self.assertFalse(hold["causal"])
                if version == 2:
                    self.assertIs(hold["seed_offset"], seed)
                if version == 3:
                    scale_arg = mock_op.flashmask_attention_v2.call_args.args[
                        -2
                    ]
                    self.assertIsInstance(scale_arg, float)
                if version == 4:
                    self.assertIs(
                        mock_op.call_args.kwargs["startend_row_indices"],
                        startend,
                    )
        with (
            patch.object(fa, "get_fa_version", return_value=2),
            self.assertRaises(NotImplementedError),
        ):
            fa.ulysses_local_flashmask_first_fwd(q, q, q, startend, False, 0.5)

        with (
            patch.object(fa, "get_fa_version", return_value=0),
            self.assertRaises(ValueError),
        ):
            fa.ulysses_local_flashmask_first_fwd(q, q, q, startend, False, None)

    def test_ulysses_local_v3_forward_signature_variants(self):
        q = paddle.randn([1, 4, 2, 4])
        out = paddle.randn([1, 4, 2, 4])
        lse = paddle.randn([1, 2, 4])
        startend = paddle.zeros([1, 1, 4, 2], dtype="int32")

        def flashmask_attention_with_group(query, key, value, group=None):
            return None

        def flashmask_attention_legacy(query, key, value):
            return None

        for fake_attention in (
            flashmask_attention_with_group,
            flashmask_attention_legacy,
        ):
            with self.subTest(signature=fake_attention.__name__):
                with (
                    patch.object(fa, "get_fa_version", return_value=3),
                    patch.object(fa, "flashmask_attention", fake_attention),
                    patch.object(fa, "_C_ops") as mock_c_ops,
                ):
                    mock_c_ops.flashmask_attention_v2.return_value = (out, lse)
                    result, hold = fa.ulysses_local_flashmask_first_fwd(
                        q, q, q, startend, False, None
                    )

                self.assertIs(result, out)
                self.assertEqual(hold["fa_version"], 3)


class TestRefinedRcomputeFlashMaskCpAttentionModes(unittest.TestCase):
    def setUp(self):
        self.q = paddle.randn([1, 4, 2, 4])
        self.k = paddle.randn([1, 4, 2, 4])
        self.v = paddle.randn([1, 4, 2, 4])
        self.startend = paddle.zeros([1, 1, 4, 2], dtype="int32")
        self.out = paddle.randn([1, 4, 2, 4])
        self.lse = paddle.randn([1, 2, 4])
        self.group = CpGroup()

    def _patch_group(self):
        hcg = MagicMock()
        hcg.get_context_parallel_group.return_value = self.group
        return patch.object(
            fa.fleet, "get_hybrid_communicate_group", return_value=hcg
        )

    def test_allgather_first_forward_stores_mode_for_backward(self):
        with (
            self._patch_group(),
            patch.object(
                fa,
                "cp_flashmask_allgatherkv_balance_forward",
                return_value=(self.out, self.lse, self.startend, 4),
            ) as mock_forward,
        ):
            attn = fa.RefinedRcomputeFlashMaskCpAttention()
            result = attn._first_fwd(
                self.q,
                self.k,
                self.v,
                self.startend,
                mode="contiguous_allgather",
            )

        self.assertIs(result, self.out)
        self.assertEqual(
            mock_forward.call_args.args[-1], "contiguous_allgather"
        )
        hold = attn._hold_tensors_queue.get_nowait()
        self.assertEqual(hold["mode"], "contiguous_allgather")
        self.assertEqual(hold["fa_version"], 4)

    def test_p2p_first_forward_requires_window_and_saves_recv_tensors(self):
        recv_k = paddle.randn([1, 2, 2, 4])
        recv_v = paddle.randn([1, 2, 2, 4])
        with (
            self._patch_group(),
            patch.object(fa, "is_flash_mask_available", return_value=True),
            patch.object(
                fa,
                "cp_flashmask_swa_p2p_forward",
                return_value=(
                    self.out,
                    self.lse,
                    recv_k,
                    recv_v,
                    self.startend,
                ),
            ) as mock_forward,
        ):
            attn = fa.RefinedRcomputeFlashMaskCpAttention()
            result = attn._first_fwd(
                self.q,
                self.k,
                self.v,
                self.startend,
                mode="contiguous_swap2p",
                window_size=64,
            )

        self.assertIs(result, self.out)
        self.assertEqual(mock_forward.call_args.args[-1], 64)
        hold = attn._hold_tensors_queue.get_nowait()
        self.assertEqual(hold["mode"], "contiguous_swap2p")
        self.assertIs(hold["recv_key"], recv_k)
        self.assertEqual(hold["window_size"], 64)

    def test_p2p_rejects_invalid_window(self):
        with (
            self._patch_group(),
            patch.object(fa, "is_flash_mask_available", return_value=True),
        ):
            attn = fa.RefinedRcomputeFlashMaskCpAttention()
            for window_size in (None, 0):
                with (
                    self.subTest(window_size=window_size),
                    self.assertRaises(ValueError),
                ):
                    attn._first_fwd(
                        self.q,
                        self.k,
                        self.v,
                        self.startend,
                        mode="contiguous_swap2p",
                        window_size=window_size,
                    )

    def test_ulysses_first_forward_allows_causal_and_odd_local_seq(self):
        attn = fa.RefinedRcomputeFlashMaskCpAttention()
        odd_q = self.q[:, :3]
        with (
            self._patch_group(),
            patch.object(
                attn, "_ulysses_first_fwd", return_value=self.out
            ) as mock_ulysses,
        ):
            result = attn._first_fwd(
                odd_q,
                self.k,
                self.v,
                self.startend,
                mode="contiguous_a2a",
                causal=True,
            )

        self.assertIs(result, self.out)
        mock_ulysses.assert_called_once()
        self.assertIs(mock_ulysses.call_args.args[0], odd_q)
        self.assertIs(mock_ulysses.call_args.args[4], self.group)
        self.assertTrue(mock_ulysses.call_args.args[5])

    def test_first_forward_rejects_invalid_mode(self):
        with self._patch_group(), self.assertRaises(ValueError):
            fa.RefinedRcomputeFlashMaskCpAttention()._first_fwd(
                self.q, self.k, self.v, self.startend, mode="bad_mode"
            )

    def test_ulysses_first_and_second_forward_use_surrogate(self):
        attn = fa.RefinedRcomputeFlashMaskCpAttention()

        def fake_alltoall(
            tensor, scatter_idx, gather_idx, batch_dim_idx, group
        ):
            return tensor

        with (
            patch.object(
                fa.UlyssesAlltoAll, "apply", side_effect=fake_alltoall
            ) as mock_alltoall,
            patch.object(
                fa,
                "ulysses_local_flashmask_first_fwd",
                return_value=(
                    self.out,
                    {
                        "result_attention": self.out,
                        "softmax_lse": self.lse,
                        "causal": False,
                        "fa_version": 4,
                    },
                ),
            ),
        ):
            result = attn._ulysses_first_fwd(
                self.q,
                self.k,
                self.v,
                self.startend,
                self.group,
                False,
                None,
                None,
            )
        self.assertIs(result, self.out)
        self.assertEqual(mock_alltoall.call_count, 4)
        self.assertEqual(
            mock_alltoall.call_args_list[0].kwargs["scatter_idx"], 2
        )
        self.assertEqual(
            mock_alltoall.call_args_list[-1].kwargs["scatter_idx"], 1
        )
        hold = attn._hold_tensors_queue.get_nowait()
        self.assertEqual(hold["mode"], "contiguous_a2a")
        self.assertIs(hold["result_attention"], self.out)
        self.assertIs(hold["local_query"], self.q)
        self.assertIs(hold["local_key"], self.k)
        self.assertIs(hold["local_value"], self.v)

        with patch.object(
            fa.FlashMaskUlyssesCpFunctor, "apply", return_value=self.out
        ) as mock_apply:
            attn._hold_tensors_queue.put(hold)
            self.assertIs(attn._second_fwd(self.q, self.k, self.v), self.out)
        mock_apply.assert_called_once()

    def _ulysses_hold(self, fa_version, **local_extra):
        local_hold = {
            "result_attention": self.out,
            "softmax_lse": self.lse,
            "causal": False,
            "fa_version": fa_version,
        }
        local_hold.update(local_extra)
        return {
            "group": self.group,
            "result_attention": self.out,
            "local_query": self.q,
            "local_key": self.k,
            "local_value": self.v,
            "startend_row_indices": self.startend,
            "local_hold_tensors": local_hold,
        }

    def test_ulysses_functor_backward_returns_original_layout_grads(self):
        ctx = FakeCtx()
        local_grad = paddle.ones_like(self.out)
        hold = self._ulysses_hold(4)

        self.assertIs(
            fa.FlashMaskUlyssesCpFunctor.forward(
                ctx, self.q, self.k, self.v, hold
            ),
            self.out,
        )
        flashmask_info = object()
        with (
            patch.object(fa, "_ulysses_fused_supported", return_value=False),
            patch.object(
                fa,
                "_ulysses_single_all_to_all",
                side_effect=(local_grad, self.q, self.k, self.v),
            ) as mock_a2a,
            patch.object(
                fa,
                "FlashMaskInfoPaddle",
                return_value=flashmask_info,
                create=True,
            ) as mock_info,
            patch.object(
                fa,
                "_flash_attn_bwd",
                return_value=(self.q, self.k, self.v, None),
                create=True,
            ) as mock_backward,
        ):
            grads = fa.FlashMaskUlyssesCpFunctor.backward(
                ctx, paddle.ones_like(self.out)
            )

        self.assertIs(grads[0], self.q)
        self.assertIs(grads[1], self.k)
        self.assertIs(grads[2], self.v)
        self.assertEqual(mock_a2a.call_args_list[0].args[1], 2)
        self.assertEqual(mock_a2a.call_args_list[0].args[2], 1)
        for call in mock_a2a.call_args_list[1:]:
            self.assertEqual(call.args[1], 1)
            self.assertEqual(call.args[2], 2)
        mock_info.assert_called_once_with(
            startend_row_indices=self.startend,
            is_causal=False,
        )
        self.assertIs(mock_backward.call_args.args[4], local_grad)
        self.assertIs(mock_backward.call_args.args[6], flashmask_info)

    def test_ulysses_functor_backward_uses_fused_a2a_when_supported(self):
        ctx = FakeCtx()
        local_grad = paddle.ones_like(self.out)
        hold = self._ulysses_hold(4)

        fa.FlashMaskUlyssesCpFunctor.forward(ctx, self.q, self.k, self.v, hold)
        with (
            patch.object(fa, "_ulysses_fused_supported", return_value=True),
            patch.object(
                fa,
                "_ulysses_single_all_to_all_fused",
                side_effect=(local_grad, self.q, self.k, self.v, self.q),
            ) as mock_fused,
            patch.object(fa, "_ulysses_single_all_to_all") as mock_ref,
            patch.object(
                fa, "FlashMaskInfoPaddle", return_value=object(), create=True
            ),
            patch.object(
                fa,
                "_flash_attn_bwd",
                return_value=(self.q, self.k, self.v, None),
                create=True,
            ),
        ):
            grads = fa.FlashMaskUlyssesCpFunctor.backward(
                ctx, paddle.ones_like(self.out)
            )
            self.assertIs(
                fa._ulysses_single_all_to_all_rr(self.q, 2, 1, 0, self.group),
                self.q,
            )

        self.assertIs(grads[0], self.q)
        self.assertIs(grads[1], self.k)
        self.assertIs(grads[2], self.v)
        mock_ref.assert_not_called()
        self.assertEqual(
            [call.args[1] for call in mock_fused.call_args_list],
            [2, 1, 1, 1, 2],
        )

    def test_ulysses_functor_backward_fa2(self):
        ctx = FakeCtx()
        seed = paddle.zeros([1], dtype="int64")
        local_grad = paddle.ones_like(self.out)
        hold = self._ulysses_hold(2, seed_offset=seed)

        fa.FlashMaskUlyssesCpFunctor.forward(ctx, self.q, self.k, self.v, hold)
        with (
            patch.object(fa, "_ulysses_fused_supported", return_value=False),
            patch.object(
                fa,
                "_ulysses_single_all_to_all",
                side_effect=(local_grad, self.q, self.k, self.v),
            ),
            patch.object(fa, "_C_ops") as mock_c_ops,
        ):
            mock_c_ops.flashmask_attention_grad.return_value = (
                self.q,
                self.k,
                self.v,
            )
            grads = fa.FlashMaskUlyssesCpFunctor.backward(
                ctx, paddle.ones_like(self.out)
            )

        self.assertIs(grads[0], self.q)
        self.assertIs(
            mock_c_ops.flashmask_attention_grad.call_args.args[6], seed
        )
        self.assertIs(
            mock_c_ops.flashmask_attention_grad.call_args.args[7], local_grad
        )

    def test_ulysses_functor_backward_fa3_signature_variants(self):
        local_grad = paddle.ones_like(self.out)

        def flashmask_attention_with_group(query, key, value, group=None):
            return None

        def flashmask_attention_with_block_mask(
            query, key, value, block_mask=None
        ):
            return None

        def flashmask_attention_legacy(query, key, value):
            return None

        cases = (
            (flashmask_attention_with_group, -5),
            (flashmask_attention_with_block_mask, -3),
            (flashmask_attention_legacy, -3),
        )
        for fake_attention, scale_arg_index in cases:
            with self.subTest(signature=fake_attention.__name__):
                ctx = FakeCtx()
                out = paddle.randn([1, 4, 2, 4])
                lse = paddle.randn([1, 2, 4])
                hold = self._ulysses_hold(
                    3, result_attention=out, softmax_lse=lse
                )
                fa.FlashMaskUlyssesCpFunctor.forward(
                    ctx, self.q, self.k, self.v, hold
                )
                with (
                    patch.object(fa, "flashmask_attention", fake_attention),
                    patch.object(
                        fa, "_ulysses_fused_supported", return_value=False
                    ),
                    patch.object(
                        fa,
                        "_ulysses_single_all_to_all",
                        side_effect=(local_grad, self.q, self.k, self.v),
                    ),
                    patch.object(fa, "_C_ops") as mock_c_ops,
                ):
                    mock_c_ops.flashmask_attention_v2_grad.return_value = (
                        self.q,
                        self.k,
                        self.v,
                    )
                    grads = fa.FlashMaskUlyssesCpFunctor.backward(
                        ctx, paddle.ones([1, 4, 2, 4])
                    )

                self.assertIs(grads[0], self.q)
                self.assertIs(
                    mock_c_ops.flashmask_attention_v2_grad.call_args.args[
                        scale_arg_index
                    ],
                    local_grad,
                )

    def test_ulysses_functor_backward_rejects_invalid_fa_version(self):
        ctx = FakeCtx()
        hold = self._ulysses_hold(0)
        fa.FlashMaskUlyssesCpFunctor.forward(ctx, self.q, self.k, self.v, hold)
        with (
            patch.object(fa, "_ulysses_fused_supported", return_value=False),
            patch.object(
                fa,
                "_ulysses_single_all_to_all",
                return_value=paddle.ones_like(self.out),
            ),
            self.assertRaises(ValueError),
        ):
            fa.FlashMaskUlyssesCpFunctor.backward(
                ctx, paddle.ones_like(self.out)
            )

    def test_second_forward_dispatches_each_mode(self):
        attn = fa.RefinedRcomputeFlashMaskCpAttention()
        for mode, target in (
            ("dualchunk_allgather", fa.FlashMaskAttnCpFunctor),
            ("contiguous_allgather", fa.FlashMaskAttnCpFunctor),
            ("contiguous_swap2p", fa.FlashMaskSwaP2PFunctor),
        ):
            attn._hold_tensors_queue.put({"mode": mode})
            with patch.object(
                target, "apply", return_value=self.out
            ) as mock_apply:
                self.assertIs(
                    attn._second_fwd(self.q, self.k, self.v), self.out
                )
                mock_apply.assert_called_once()

        attn._hold_tensors_queue.put({"mode": "contiguous_a2a"})
        with (
            patch.object(
                fa.FlashMaskUlyssesCpFunctor, "apply", return_value=self.out
            ) as mock_ulysses,
            patch.object(
                fa.UlyssesAlltoAll,
                "apply",
                side_effect=AssertionError(
                    "second forward should not all-to-all"
                ),
            ),
        ):
            self.assertIs(attn._second_fwd(self.q, self.k, self.v), self.out)
            mock_ulysses.assert_called_once()

        attn._hold_tensors_queue.put({"mode": "bad_mode"})
        with self.assertRaises(ValueError):
            attn._second_fwd(self.q, self.k, self.v)

    def test_validation_errors_are_preserved(self):
        attn = fa.RefinedRcomputeFlashMaskCpAttention()
        with self.assertRaises(NotImplementedError):
            attn._first_fwd(self.q, self.k, self.v, self.startend, causal=True)
        with self.assertRaises(NotImplementedError):
            attn._first_fwd(self.q, self.k, self.v, self.startend, dropout=0.1)
        with self._patch_group(), self.assertRaises(AssertionError):
            attn._first_fwd(self.q[:, :3], self.k, self.v, self.startend)
        for kwargs in ({"learnable_sink": self.q}, {"softmax_scale": 0.5}):
            with (
                self.subTest(kwargs=kwargs),
                self.assertRaises(NotImplementedError),
            ):
                attn._ulysses_first_fwd(
                    self.q,
                    self.k,
                    self.v,
                    self.startend,
                    self.group,
                    False,
                    kwargs.get("learnable_sink"),
                    kwargs.get("softmax_scale"),
                )


if __name__ == "__main__":
    unittest.main()
