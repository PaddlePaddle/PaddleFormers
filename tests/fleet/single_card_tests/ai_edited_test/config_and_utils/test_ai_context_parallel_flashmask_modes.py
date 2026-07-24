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

import importlib.util
import os
import sys
import types
import unittest
from unittest.mock import patch

import paddle

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
)
sys.path.insert(0, REPO_ROOT)

from paddleformers.fleet import context_parallel_utils as cp_utils


class FakeGroup:
    def __init__(self, rank=0, world_size=2):
        self.rank = rank
        self.world_size = world_size
        self.nranks = world_size
        self.ranks = list(range(world_size))


class FakeContext:
    def save_for_backward(self, *tensors):
        self.saved = tensors

    def saved_tensor(self):
        return self.saved


class FakeHcg:
    def __init__(self, group):
        self.group = group

    def get_context_parallel_group(self):
        return self.group

    def get_context_parallel_world_size(self):
        return self.group.world_size


class FakeTask:
    def wait(self):
        pass


class FakeFlashMaskInfoPaddle:
    def __init__(self, startend_row_indices, is_causal=False):
        self.startend_row_indices = startend_row_indices
        self.is_causal = is_causal


def fake_bshd_slice_contiguous_kv(
    key_grad_tensor, value_grad_tensor, start, length
):
    end = start + length
    return (
        key_grad_tensor[:, start:end, :, :].contiguous(),
        value_grad_tensor[:, start:end, :, :].contiguous(),
    )


def assert_tensor_equal(testcase, actual, expected):
    testcase.assertTrue(bool((actual == expected).all().item()))


class TestFlashMaskImportPath(unittest.TestCase):
    def test_sm100_imports_vendored_flashmask_utils(self):
        def fake_flash_attn_fwd(*args, **kwargs):
            return None

        def fake_flash_attn_bwd(*args, **kwargs):
            return None

        def fake_slice(*args, **kwargs):
            return None

        def fake_module(name, is_package=False):
            module = types.ModuleType(name)
            if is_package:
                module.__path__ = []
            return module

        fake_modules = {
            "paddlefleet_ops.flash_mask": fake_module(
                "paddlefleet_ops.flash_mask", is_package=True
            ),
            "paddlefleet_ops.flash_mask.cute": fake_module(
                "paddlefleet_ops.flash_mask.cute", is_package=True
            ),
            "paddlefleet_ops.flash_mask.cute.flashmask_utils": fake_module(
                "paddlefleet_ops.flash_mask.cute.flashmask_utils"
            ),
            "paddlefleet_ops.flash_mask.cute.interface": fake_module(
                "paddlefleet_ops.flash_mask.cute.interface"
            ),
            "paddlefleet_ops.flash_mask.utils": fake_module(
                "paddlefleet_ops.flash_mask.utils", is_package=True
            ),
        }
        fake_modules[
            "paddlefleet_ops.flash_mask.cute.flashmask_utils"
        ].FlashMaskInfoPaddle = FakeFlashMaskInfoPaddle
        fake_modules[
            "paddlefleet_ops.flash_mask.cute.interface"
        ]._flash_attn_fwd = fake_flash_attn_fwd
        fake_modules[
            "paddlefleet_ops.flash_mask.cute.interface"
        ]._flash_attn_bwd = fake_flash_attn_bwd
        fake_modules[
            "paddlefleet_ops.flash_mask.utils"
        ].bshd_slice_contiguous_kv = fake_slice

        module_path = os.path.join(
            REPO_ROOT, "paddleformers", "fleet", "context_parallel_utils.py"
        )
        spec = importlib.util.spec_from_file_location(
            "_test_context_parallel_utils_sm100", module_path
        )
        module = importlib.util.module_from_spec(spec)
        with (
            patch.dict(sys.modules, fake_modules),
            patch.object(paddle.cuda, "is_available", return_value=True),
            patch.object(
                paddle.cuda, "get_device_capability", return_value=(10, 0)
            ),
        ):
            spec.loader.exec_module(module)

        self.assertTrue(module._flash_mask_available)
        self.assertIs(module.FlashMaskInfoPaddle, FakeFlashMaskInfoPaddle)
        self.assertIs(module._flash_attn_fwd, fake_flash_attn_fwd)
        self.assertIs(module._flash_attn_bwd, fake_flash_attn_bwd)
        self.assertIs(module.bshd_slice_contiguous_kv, fake_slice)


class TestFlashMaskAllGatherModes(unittest.TestCase):
    def setUp(self):
        self.query = paddle.zeros([1, 4, 1, 1], dtype="float32")
        self.key = paddle.full([1, 4, 1, 1], 2.0, dtype="float32")
        self.value = paddle.full([1, 4, 1, 1], 3.0, dtype="float32")
        self.output = paddle.full([1, 4, 1, 1], 7.0, dtype="float32")
        self.lse = paddle.full([1, 1, 4], 0.5, dtype="float32")
        self.group = FakeGroup(rank=1, world_size=2)

    def _run_forward(self, mode, indices):
        captured = {}

        def fake_flashmask_attention(query, key, value, **kwargs):
            captured["key"] = key
            captured["value"] = value
            captured["indices"] = kwargs["startend_row_indices"]
            return self.output, self.lse

        patches = [
            patch.object(
                cp_utils.paddle.base.core, "nvprof_nvtx_push", lambda _: None
            ),
            patch.object(
                cp_utils.paddle.base.core, "nvprof_nvtx_pop", lambda: None
            ),
            patch.object(
                cp_utils.paddle.base.framework,
                "get_flags",
                lambda _: {"FLAGS_flash_attn_version": 2},
            ),
            patch.object(
                cp_utils.paddle,
                "get_flags",
                lambda _: {"FLAGS_cudnn_deterministic": False},
            ),
            patch.object(
                cp_utils, "flashmask_attention", fake_flashmask_attention
            ),
            patch.object(
                cp_utils, "all_gather_balance", lambda x, axis, group: x + 10
            ),
            patch.object(
                cp_utils, "all_gather_contiguous", lambda x, axis, group: x + 20
            ),
        ]
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
        ):
            output, lse, processed_indices, fa_version = (
                cp_utils.cp_flashmask_allgatherkv_balance_forward(
                    self.query,
                    self.key,
                    self.value,
                    indices,
                    None,
                    self.group,
                    False,
                    True,
                    # fa_version==2 does not support softmax_scale, pass None
                    None,
                    mode,
                )
            )

        assert_tensor_equal(self, output, self.output)
        assert_tensor_equal(self, lse, self.lse)
        self.assertEqual(fa_version, 2)
        assert_tensor_equal(self, processed_indices, captured["indices"])
        return captured, processed_indices

    def test_dualchunk_forward_preprocesses_indices_and_uses_balanced_layout(
        self,
    ):
        indices = paddle.to_tensor([0, 1, 2, 3, 4, 5, 6], dtype="int32")
        captured, processed_indices = self._run_forward(
            "dualchunk_allgather", indices
        )

        expected_indices = cp_utils.preprocess_index_dual_chunks(
            indices,
            chunk_id_first=1,
            chunk_id_second=2,
            seq_blocksize=2,
            max_seqlen_q=2,
        )
        assert_tensor_equal(self, processed_indices, expected_indices)
        assert_tensor_equal(self, captured["key"], self.key + 10)
        assert_tensor_equal(self, captured["value"], self.value + 10)

    def test_contiguous_forward_preprocesses_indices_and_uses_rank_order_layout(
        self,
    ):
        indices = paddle.to_tensor([0, 3, 4, 6, 8], dtype="int32")
        captured, processed_indices = self._run_forward(
            "contiguous_allgather", indices
        )

        expected_indices = cp_utils.preprocess_index(
            indices, chunk_id=1, seq_blocksize=4, max_seqlen_q=4
        )
        assert_tensor_equal(self, processed_indices, expected_indices)
        assert_tensor_equal(self, captured["key"], self.key + 20)
        assert_tensor_equal(self, captured["value"], self.value + 20)

    def test_forward_rejects_unknown_mode(self):
        with (
            patch.object(
                cp_utils.paddle.base.core, "nvprof_nvtx_push", lambda _: None
            ),
            self.assertRaises(ValueError),
        ):
            cp_utils.cp_flashmask_allgatherkv_balance_forward(
                self.query,
                self.key,
                self.value,
                paddle.to_tensor([0], dtype="int32"),
                None,
                self.group,
                False,
                True,
                None,
                "swa_p2p",
            )

    def _run_backward(self, mode):
        gathered_key_grad = paddle.full([1, 8, 1, 1], 5.0, dtype="float32")
        gathered_value_grad = paddle.full([1, 8, 1, 1], 6.0, dtype="float32")
        reduced_key_grad = paddle.full([1, 4, 1, 1], 7.0, dtype="float32")
        reduced_value_grad = paddle.full([1, 4, 1, 1], 8.0, dtype="float32")
        captured = {}

        def fake_flashmask_grad(query, key, value, *args):
            captured["key"] = key
            captured["value"] = value
            return self.query + 1, gathered_key_grad, gathered_value_grad

        def fake_reduce_balance(x, axis, group):
            captured.setdefault("reduced", []).append(x)
            return (
                reduced_key_grad
                if len(captured["reduced"]) == 1
                else reduced_value_grad
            )

        def fake_reduce_contiguous(x, axis, group):
            captured.setdefault("reduced", []).append(x)
            return (
                reduced_key_grad + 10
                if len(captured["reduced"]) == 1
                else reduced_value_grad + 10
            )

        patches = [
            patch.object(
                cp_utils.paddle.base.core, "nvprof_nvtx_push", lambda _: None
            ),
            patch.object(
                cp_utils.paddle.base.core, "nvprof_nvtx_pop", lambda: None
            ),
            patch.object(
                cp_utils, "all_gather_balance", lambda x, axis, group: x + 10
            ),
            patch.object(
                cp_utils, "all_gather_contiguous", lambda x, axis, group: x + 20
            ),
            patch.object(
                cp_utils, "reduce_scatter_any_axis_balance", fake_reduce_balance
            ),
            patch.object(
                cp_utils, "reduce_scatter_contiguous", fake_reduce_contiguous
            ),
            patch.object(
                cp_utils.paddle._C_ops,
                "flashmask_attention_grad",
                fake_flashmask_grad,
            ),
        ]
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
        ):
            query_grad, key_grad, value_grad, grad_sink = (
                cp_utils.cp_flashmask_allgatherkv_balance_backward(
                    self.query,
                    self.key,
                    self.value,
                    paddle.to_tensor([0, 1], dtype="int32"),
                    self.output,
                    self.lse,
                    self.output,
                    None,
                    self.group,
                    False,
                    2,
                    None,
                    mode,
                )
            )
        return captured, query_grad, key_grad, value_grad, grad_sink

    def test_dualchunk_backward_uses_balanced_gather_and_reduce_scatter(self):
        captured, query_grad, key_grad, value_grad, grad_sink = (
            self._run_backward("dualchunk_allgather")
        )

        assert_tensor_equal(self, captured["key"], self.key + 10)
        assert_tensor_equal(self, captured["value"], self.value + 10)
        assert_tensor_equal(self, query_grad, self.query + 1)
        assert_tensor_equal(self, key_grad, 7.0)
        assert_tensor_equal(self, value_grad, 8.0)
        self.assertIsNone(grad_sink)

    def test_contiguous_backward_uses_rank_order_gather_and_reduce_scatter(
        self,
    ):
        captured, query_grad, key_grad, value_grad, grad_sink = (
            self._run_backward("contiguous_allgather")
        )

        assert_tensor_equal(self, captured["key"], self.key + 20)
        assert_tensor_equal(self, captured["value"], self.value + 20)
        assert_tensor_equal(self, query_grad, self.query + 1)
        assert_tensor_equal(self, key_grad, 17.0)
        assert_tensor_equal(self, value_grad, 18.0)
        self.assertIsNone(grad_sink)

    def test_backward_rejects_unknown_mode(self):
        with (
            patch.object(
                cp_utils.paddle.base.core, "nvprof_nvtx_push", lambda _: None
            ),
            self.assertRaises(ValueError),
        ):
            cp_utils.cp_flashmask_allgatherkv_balance_backward(
                self.query,
                self.key,
                self.value,
                paddle.to_tensor([0], dtype="int32"),
                self.output,
                self.lse,
                self.output,
                None,
                self.group,
                False,
                2,
                None,
                "swa_p2p",
            )


class TestFlashMaskContextParallelMode(unittest.TestCase):
    def test_forward_passes_mode_to_impl_and_saves_it_for_backward(self):
        query = paddle.zeros([1, 4, 1, 1], dtype="float32")
        key = paddle.zeros([1, 4, 1, 1], dtype="float32")
        value = paddle.zeros([1, 4, 1, 1], dtype="float32")
        indices = paddle.to_tensor([0, 1], dtype="int32")
        output = paddle.full([1, 4, 1, 1], 9.0, dtype="float32")
        lse = paddle.full([1, 1, 4], 1.0, dtype="float32")
        ctx = FakeContext()
        captured = {}

        def fake_forward(*args):
            captured["mode"] = args[-1]
            return output, lse, indices + 1, 2

        with (
            patch.object(
                cp_utils.fleet,
                "get_hybrid_communicate_group",
                lambda: FakeHcg(FakeGroup(rank=0, world_size=2)),
            ),
            patch.object(
                cp_utils,
                "cp_flashmask_allgatherkv_balance_forward",
                fake_forward,
            ),
        ):
            result = cp_utils.FlashMaskContextParallel.forward(
                ctx,
                query,
                key,
                value,
                indices,
                mode="contiguous_allgather",
            )

        assert_tensor_equal(self, result, output)
        self.assertEqual(captured["mode"], "contiguous_allgather")
        self.assertEqual(ctx.mode, "contiguous_allgather")
        self.assertEqual(ctx.fa_version, 2)
        assert_tensor_equal(self, ctx.saved[-1], indices + 1)


class TestContextParallelOpsModes(unittest.TestCase):
    def setUp(self):
        self.group = FakeGroup(rank=0, world_size=2)
        self.tensor = paddle.zeros([2, 4], dtype="float32")

    def _patch_hcg(self):
        return patch.object(
            cp_utils.fleet,
            "get_hybrid_communicate_group",
            lambda: FakeHcg(self.group),
        )

    def test_contiguous_prefixed_modes_use_rank_order_scatter(self):
        for mode in ("contiguous_a2a", "contiguous_swap2p"):
            with (
                self.subTest(mode=mode),
                self._patch_hcg(),
                patch.object(cp_utils, "scatter_contiguous") as contiguous,
                patch.object(cp_utils, "scatter_balance") as balanced,
            ):
                contiguous.return_value = self.tensor + 1
                result = cp_utils.ContextParallelScatterOp.forward(
                    FakeContext(), self.tensor, -1, mode
                )

            assert_tensor_equal(self, result, self.tensor + 1)
            contiguous.assert_called_once()
            balanced.assert_not_called()

    def test_non_contiguous_mode_uses_balanced_scatter(self):
        with (
            self._patch_hcg(),
            patch.object(cp_utils, "scatter_contiguous") as contiguous,
            patch.object(cp_utils, "scatter_balance") as balanced,
        ):
            balanced.return_value = self.tensor + 2
            result = cp_utils.ContextParallelScatterOp.forward(
                FakeContext(), self.tensor, -1, "dualchunk_allgather"
            )

        assert_tensor_equal(self, result, self.tensor + 2)
        contiguous.assert_not_called()
        balanced.assert_called_once()

    def test_contiguous_prefixed_modes_use_rank_order_gather_and_allgather(
        self,
    ):
        for op_cls, contiguous_name, balanced_name in (
            (
                cp_utils.ContextParallelGatherOp,
                "all_gather_contiguous",
                "all_gather_balance",
            ),
            (
                cp_utils.ContextParallelAllGatherOp,
                "all_gather_contiguous",
                "all_gather_balance",
            ),
        ):
            with (
                self.subTest(op=op_cls.__name__),
                self._patch_hcg(),
                patch.object(cp_utils, contiguous_name) as contiguous,
                patch.object(cp_utils, balanced_name) as balanced,
            ):
                contiguous.return_value = self.tensor + 3
                result = op_cls.forward(
                    FakeContext(), self.tensor, -1, "contiguous_a2a"
                )

            assert_tensor_equal(self, result, self.tensor + 3)
            contiguous.assert_called_once()
            balanced.assert_not_called()

    def test_contiguous_prefixed_modes_use_rank_order_backward_ops(self):
        cases = (
            (
                cp_utils.ContextParallelScatterOp,
                "all_gather_contiguous",
                "all_gather_balance",
            ),
            (
                cp_utils.ContextParallelGatherOp,
                "scatter_contiguous",
                "scatter_balance",
            ),
            (
                cp_utils.ContextParallelAllGatherOp,
                "reduce_scatter_contiguous",
                "reduce_scatter_any_axis_balance",
            ),
        )
        for op_cls, contiguous_name, balanced_name in cases:
            ctx = FakeContext()
            ctx.mode = "contiguous_a2a"
            ctx.axis = -1
            ctx.group = self.group
            with (
                self.subTest(op=op_cls.__name__),
                patch.object(cp_utils, contiguous_name) as contiguous,
                patch.object(cp_utils, balanced_name) as balanced,
            ):
                contiguous.return_value = self.tensor + 4
                result = op_cls.backward(ctx, self.tensor)

            assert_tensor_equal(self, result, self.tensor + 4)
            contiguous.assert_called_once()
            balanced.assert_not_called()


class TestFlashMaskSwaP2PHelpers(unittest.TestCase):
    def test_scatter_kv_places_local_and_received_windows(self):
        group = FakeGroup(rank=1, world_size=2)
        key = paddle.arange(4, dtype="float32").reshape([1, 4, 1, 1]) + 10
        value = key + 100
        recv_key = paddle.arange(2, dtype="float32").reshape([1, 2, 1, 1]) + 8
        recv_value = recv_key + 100

        key_tensor, value_tensor = cp_utils._scatter_kv_to_global_tensor(
            key, value, recv_key, recv_value, group
        )

        assert_tensor_equal(self, key_tensor[:, 2:4, :, :], recv_key)
        assert_tensor_equal(self, value_tensor[:, 2:4, :, :], recv_value)
        assert_tensor_equal(self, key_tensor[:, 4:8, :, :], key)
        assert_tensor_equal(self, value_tensor[:, 4:8, :, :], value)

    def test_send_window_grad_back_returns_original_tensors_for_single_rank(
        self,
    ):
        group = FakeGroup(rank=0, world_size=1)
        key_grad_tensor = paddle.zeros([1, 4, 1, 1], dtype="float32")
        value_grad_tensor = paddle.ones([1, 4, 1, 1], dtype="float32")
        key = paddle.zeros([1, 4, 1, 1], dtype="float32")
        value = paddle.zeros([1, 4, 1, 1], dtype="float32")

        key_grad, value_grad = cp_utils._send_window_grad_back(
            key_grad_tensor, value_grad_tensor, key, value, group, 128
        )

        self.assertIs(key_grad, key_grad_tensor)
        self.assertIs(value_grad, value_grad_tensor)

    def test_send_window_grad_back_uses_bshd_slice_op_for_local_grad(self):
        group = FakeGroup(rank=1, world_size=2)
        key_grad_tensor = paddle.arange(8, dtype="float32").reshape(
            [1, 8, 1, 1]
        )
        value_grad_tensor = key_grad_tensor + 100
        key = paddle.zeros([1, 4, 1, 1], dtype="float32")
        value = paddle.zeros([1, 4, 1, 1], dtype="float32")
        sliced_key = paddle.full([1, 4, 1, 1], 7.0, dtype="float32")
        sliced_value = paddle.full([1, 4, 1, 1], 8.0, dtype="float32")
        captured = {}

        def fake_slice(key_grad_arg, value_grad_arg, start, length):
            captured["key_grad_tensor"] = key_grad_arg
            captured["value_grad_tensor"] = value_grad_arg
            captured["start"] = start
            captured["length"] = length
            return sliced_key, sliced_value

        with (
            patch.object(cp_utils.dist, "P2POp", lambda *args: object()),
            patch.object(
                cp_utils.dist, "batch_isend_irecv", lambda ops: [FakeTask()]
            ),
            patch.object(
                cp_utils, "bshd_slice_contiguous_kv", fake_slice, create=True
            ),
        ):
            key_grad, value_grad = cp_utils._send_window_grad_back(
                key_grad_tensor, value_grad_tensor, key, value, group, 2
            )

        self.assertIs(captured["key_grad_tensor"], key_grad_tensor)
        self.assertIs(captured["value_grad_tensor"], value_grad_tensor)
        self.assertEqual(captured["start"], 4)
        self.assertEqual(captured["length"], 4)
        self.assertIs(key_grad, sliced_key)
        self.assertIs(value_grad, sliced_value)

    def test_send_window_grad_back_accumulates_received_tail_grad(self):
        group = FakeGroup(rank=0, world_size=2)
        key_grad_tensor = paddle.arange(256, dtype="float32").reshape(
            [1, 256, 1, 1]
        )
        value_grad_tensor = key_grad_tensor + 100
        key = paddle.zeros([1, 128, 1, 1], dtype="float32")
        value = paddle.zeros([1, 128, 1, 1], dtype="float32")
        recv_grad_window = paddle.stack(
            [
                paddle.full([1, 128, 1, 1], 10.0, dtype="float32"),
                paddle.full([1, 128, 1, 1], 20.0, dtype="float32"),
            ],
            axis=0,
        )

        with (
            patch.object(
                cp_utils.paddle,
                "empty",
                lambda *args, **kwargs: recv_grad_window,
            ),
            patch.object(cp_utils.dist, "P2POp", lambda *args: object()),
            patch.object(
                cp_utils.dist, "batch_isend_irecv", lambda ops: [FakeTask()]
            ),
            patch.object(
                cp_utils,
                "bshd_slice_contiguous_kv",
                fake_bshd_slice_contiguous_kv,
                create=True,
            ),
        ):
            key_grad, value_grad = cp_utils._send_window_grad_back(
                key_grad_tensor, value_grad_tensor, key, value, group, 128
            )

        assert_tensor_equal(
            self, key_grad, key_grad_tensor[:, :128, :, :] + 10.0
        )
        assert_tensor_equal(
            self, value_grad, value_grad_tensor[:, :128, :, :] + 20.0
        )

    def test_send_window_grad_back_sends_previous_window_grad_to_owner(self):
        group = FakeGroup(rank=1, world_size=2)
        key_grad_tensor = paddle.arange(256, dtype="float32").reshape(
            [1, 256, 1, 1]
        )
        value_grad_tensor = key_grad_tensor + 100
        key = paddle.zeros([1, 128, 1, 1], dtype="float32")
        value = paddle.zeros([1, 128, 1, 1], dtype="float32")
        captured = {}

        def fake_p2p_op(op, tensor, peer, group_arg):
            captured["tensor"] = tensor
            captured["peer"] = peer
            return object()

        with (
            patch.object(cp_utils.dist, "P2POp", fake_p2p_op),
            patch.object(
                cp_utils.dist, "batch_isend_irecv", lambda ops: [FakeTask()]
            ),
            patch.object(
                cp_utils,
                "bshd_slice_contiguous_kv",
                fake_bshd_slice_contiguous_kv,
                create=True,
            ),
        ):
            key_grad, value_grad = cp_utils._send_window_grad_back(
                key_grad_tensor, value_grad_tensor, key, value, group, 128
            )

        expected_send = paddle.stack(
            [key_grad_tensor[:, :128, :, :], value_grad_tensor[:, :128, :, :]],
            axis=0,
        )
        assert_tensor_equal(self, captured["tensor"], expected_send)
        self.assertEqual(captured["peer"], 0)
        assert_tensor_equal(self, key_grad, key_grad_tensor[:, 128:, :, :])
        assert_tensor_equal(self, value_grad, value_grad_tensor[:, 128:, :, :])

    def test_exchange_prev_window_receives_from_previous_rank(self):
        group = FakeGroup(rank=1, world_size=2)
        key = paddle.arange(4, dtype="float32").reshape([1, 4, 1, 1])
        value = key + 10
        recv_window = paddle.stack(
            [key[:, :2, :, :] + 20, value[:, :2, :, :] + 20], axis=0
        )

        with (
            patch.object(
                cp_utils.paddle, "empty", lambda *args, **kwargs: recv_window
            ),
            patch.object(cp_utils.dist, "P2POp", lambda *args: object()),
            patch.object(
                cp_utils.dist, "batch_isend_irecv", lambda ops: [FakeTask()]
            ),
        ):
            recv_key, recv_value = cp_utils._exchange_prev_window(
                key, value, group, window_size=2
            )

        assert_tensor_equal(self, recv_key, recv_window[0])
        assert_tensor_equal(self, recv_value, recv_window[1])

    def test_exchange_prev_window_sends_tail_kv_to_next_rank(self):
        group = FakeGroup(rank=0, world_size=2)
        key = paddle.arange(4, dtype="float32").reshape([1, 4, 1, 1])
        value = key + 10
        captured = {}

        def fake_p2p_op(op, tensor, peer, group_arg):
            captured["tensor"] = tensor
            captured["peer"] = peer
            return object()

        with (
            patch.object(cp_utils.dist, "P2POp", fake_p2p_op),
            patch.object(
                cp_utils.dist, "batch_isend_irecv", lambda ops: [FakeTask()]
            ),
        ):
            cp_utils._exchange_prev_window(key, value, group, window_size=2)

        expected = paddle.stack(
            [key[:, -2:, :, :], value[:, -2:, :, :]], axis=0
        )
        assert_tensor_equal(self, captured["tensor"], expected)
        self.assertEqual(captured["peer"], 1)


class TestFlashMaskSwaP2PPath(unittest.TestCase):
    def setUp(self):
        self.query = paddle.zeros([1, 128, 1, 1], dtype="float32")
        self.key = paddle.arange(128, dtype="float32").reshape([1, 128, 1, 1])
        self.value = self.key + 10
        self.indices = paddle.to_tensor([0, 1, 128], dtype="int32")
        self.group = FakeGroup(rank=0, world_size=1)

    def test_p2p_forward_preprocesses_indices_and_calls_flash_attention(self):
        output = paddle.full([1, 128, 1, 1], 3.0, dtype="float32")
        lse = paddle.full([1, 1, 128], 4.0, dtype="float32")
        captured = {}

        def fake_flash_fwd(query, key, value, **kwargs):
            captured["query"] = query
            captured["key"] = key
            captured["value"] = value
            captured["indices"] = kwargs["startend_row_indices"]
            captured["learnable_sink"] = kwargs["learnable_sink"]
            captured["causal"] = kwargs["causal"]
            captured["softmax_scale"] = kwargs["softmax_scale"]
            return output, lse

        with (
            patch.object(
                cp_utils.paddle.base.core, "nvprof_nvtx_push", lambda _: None
            ),
            patch.object(
                cp_utils.paddle.base.core, "nvprof_nvtx_pop", lambda: None
            ),
            patch.object(
                cp_utils, "_flash_attn_fwd", fake_flash_fwd, create=True
            ),
        ):
            result, result_lse, recv_key, recv_value, processed = (
                cp_utils.cp_flashmask_swa_p2p_forward(
                    self.query,
                    self.key,
                    self.value,
                    self.indices,
                    None,
                    self.group,
                    causal=False,
                    is_training=True,
                    softmax_scale=0.5,
                    window_size=64,
                )
            )

        expected_indices = cp_utils.preprocess_index(
            self.indices, chunk_id=0, seq_blocksize=128, max_seqlen_q=128
        )
        assert_tensor_equal(self, result, output)
        assert_tensor_equal(self, result_lse, lse)
        assert_tensor_equal(self, processed, expected_indices)
        self.assertIs(captured["query"], self.query)
        self.assertIs(captured["key"], self.key)
        self.assertIs(captured["value"], self.value)
        assert_tensor_equal(self, captured["indices"], expected_indices)
        self.assertIsNone(captured["learnable_sink"])
        self.assertFalse(captured["causal"])
        self.assertEqual(captured["softmax_scale"], 0.5)
        self.assertEqual(list(recv_key.shape), [1, 64, 1, 1])
        self.assertEqual(list(recv_value.shape), [1, 64, 1, 1])

    def test_p2p_backward_calls_flash_bwd_and_returns_local_grads(self):
        output = paddle.full([1, 128, 1, 1], 3.0, dtype="float32")
        lse = paddle.full([1, 1, 128], 4.0, dtype="float32")
        output_grad = paddle.ones([1, 128, 1, 1], dtype="float32")
        recv_key = paddle.zeros([1, 128, 1, 1], dtype="float32")
        recv_value = paddle.zeros([1, 128, 1, 1], dtype="float32")
        query_grad = self.query + 1
        key_grad = self.key + 2
        value_grad = self.value + 3
        captured = {}

        def fake_flash_bwd(query, key, value, out, dout, lse_arg, **kwargs):
            captured["key"] = key
            captured["value"] = value
            captured["output"] = out
            captured["output_grad"] = dout
            captured["lse"] = lse_arg
            captured["flashmask_info"] = kwargs["flashmask_info"]
            captured["learnable_sink"] = kwargs["learnable_sink"]
            captured["causal"] = kwargs["causal"]
            captured["softmax_scale"] = kwargs["softmax_scale"]
            captured["deterministic"] = kwargs["deterministic"]
            captured["kv_postprocess_start"] = kwargs["kv_postprocess_start"]
            captured["kv_postprocess_end"] = kwargs["kv_postprocess_end"]
            return query_grad, key_grad, value_grad, None

        with (
            patch.object(
                cp_utils.paddle.base.core, "nvprof_nvtx_push", lambda _: None
            ),
            patch.object(
                cp_utils.paddle.base.core, "nvprof_nvtx_pop", lambda: None
            ),
            patch.object(
                cp_utils.paddle,
                "get_flags",
                lambda _: {"FLAGS_cudnn_deterministic": False},
            ),
            patch.object(
                cp_utils, "_flash_attn_bwd", fake_flash_bwd, create=True
            ),
            patch.object(
                cp_utils,
                "FlashMaskInfoPaddle",
                FakeFlashMaskInfoPaddle,
                create=True,
            ),
        ):
            qg, kg, vg, grad_sink = cp_utils.cp_flashmask_swa_p2p_backward(
                self.query,
                self.key,
                self.value,
                recv_key,
                recv_value,
                self.indices,
                output,
                lse,
                output_grad,
                None,
                self.group,
                causal=False,
                softmax_scale=0.5,
                window_size=128,
            )

        assert_tensor_equal(self, qg, query_grad)
        assert_tensor_equal(self, kg, key_grad)
        assert_tensor_equal(self, vg, value_grad)
        self.assertIs(captured["key"], self.key)
        self.assertIs(captured["value"], self.value)
        self.assertIs(captured["output"], output)
        self.assertIs(captured["output_grad"], output_grad)
        self.assertIs(captured["lse"], lse)
        self.assertIs(
            captured["flashmask_info"].startend_row_indices, self.indices
        )
        self.assertFalse(captured["flashmask_info"].is_causal)
        self.assertIsNone(captured["learnable_sink"])
        self.assertFalse(captured["causal"])
        self.assertEqual(captured["softmax_scale"], 0.5)
        self.assertFalse(captured["deterministic"])
        self.assertEqual(captured["kv_postprocess_start"], 0)
        self.assertEqual(captured["kv_postprocess_end"], 128)
        self.assertIsNone(grad_sink)

    def test_p2p_backward_sets_nonzero_rank_kv_postprocess_range(self):
        group = FakeGroup(rank=1, world_size=2)
        output = paddle.full([1, 128, 1, 1], 3.0, dtype="float32")
        lse = paddle.full([1, 1, 128], 4.0, dtype="float32")
        output_grad = paddle.ones([1, 128, 1, 1], dtype="float32")
        recv_key = paddle.zeros([1, 128, 1, 1], dtype="float32")
        recv_value = paddle.zeros([1, 128, 1, 1], dtype="float32")
        query_grad = self.query + 1
        key_grad_tensor = paddle.zeros([1, 256, 1, 1], dtype="float32")
        value_grad_tensor = paddle.ones([1, 256, 1, 1], dtype="float32")
        captured = {}

        def fake_flash_bwd(*args, **kwargs):
            captured["kv_postprocess_start"] = kwargs["kv_postprocess_start"]
            captured["kv_postprocess_end"] = kwargs["kv_postprocess_end"]
            captured["flashmask_info"] = kwargs["flashmask_info"]
            captured["causal"] = kwargs["causal"]
            return query_grad, key_grad_tensor, value_grad_tensor, None

        with (
            patch.object(
                cp_utils.paddle.base.core, "nvprof_nvtx_push", lambda _: None
            ),
            patch.object(
                cp_utils.paddle.base.core, "nvprof_nvtx_pop", lambda: None
            ),
            patch.object(
                cp_utils.paddle,
                "get_flags",
                lambda _: {"FLAGS_cudnn_deterministic": False},
            ),
            patch.object(
                cp_utils, "_flash_attn_bwd", fake_flash_bwd, create=True
            ),
            patch.object(
                cp_utils,
                "FlashMaskInfoPaddle",
                FakeFlashMaskInfoPaddle,
                create=True,
            ),
            patch.object(cp_utils.dist, "P2POp", lambda *args: object()),
            patch.object(
                cp_utils.dist, "batch_isend_irecv", lambda ops: [FakeTask()]
            ),
            patch.object(
                cp_utils,
                "bshd_slice_contiguous_kv",
                fake_bshd_slice_contiguous_kv,
                create=True,
            ),
        ):
            qg, kg, vg, grad_sink = cp_utils.cp_flashmask_swa_p2p_backward(
                self.query,
                self.key,
                self.value,
                recv_key,
                recv_value,
                self.indices,
                output,
                lse,
                output_grad,
                None,
                group,
                causal=True,
                softmax_scale=None,
                window_size=64,
            )

        assert_tensor_equal(self, qg, query_grad)
        assert_tensor_equal(self, kg, key_grad_tensor[:, 128:256, :, :])
        assert_tensor_equal(self, vg, value_grad_tensor[:, 128:256, :, :])
        self.assertEqual(captured["kv_postprocess_start"], 64)
        self.assertEqual(captured["kv_postprocess_end"], 256)
        self.assertTrue(captured["flashmask_info"].is_causal)
        self.assertTrue(captured["causal"])
        self.assertIsNone(grad_sink)

    def test_pylayer_forward_saves_tensors_and_backward_uses_saved_context(
        self,
    ):
        ctx = FakeContext()
        output = paddle.full([1, 4, 1, 1], 3.0, dtype="float32")
        lse = paddle.full([1, 1, 4], 4.0, dtype="float32")
        recv_key = paddle.zeros([1, 128, 1, 1], dtype="float32")
        recv_value = paddle.zeros([1, 128, 1, 1], dtype="float32")
        qg = self.query + 1
        kg = self.key + 2
        vg = self.value + 3

        captured = {}

        def fake_forward(*args):
            captured["forward_window_size"] = args[-1]
            return output, lse, recv_key, recv_value, self.indices + 1

        with patch.object(
            cp_utils,
            "cp_flashmask_swa_p2p_forward",
            fake_forward,
        ):
            result = cp_utils.FlashMaskSwaP2P.forward(
                ctx,
                self.query,
                self.key,
                self.value,
                self.indices,
                group=self.group,
                window_size=64,
            )

        assert_tensor_equal(self, result, output)
        self.assertIs(ctx.group, self.group)
        self.assertFalse(ctx.causal)
        self.assertEqual(ctx.window_size, 64)
        self.assertEqual(captured["forward_window_size"], 64)
        assert_tensor_equal(self, ctx.saved[-1], self.indices + 1)

        def fake_backward(*args):
            captured["backward_window_size"] = args[-1]
            return qg, kg, vg, None

        with patch.object(
            cp_utils,
            "cp_flashmask_swa_p2p_backward",
            fake_backward,
        ):
            backward_result = cp_utils.FlashMaskSwaP2P.backward(ctx, output)

        self.assertEqual(captured["backward_window_size"], 64)
        self.assertEqual(len(backward_result), 3)
        assert_tensor_equal(self, backward_result[0], qg)
        assert_tensor_equal(self, backward_result[1], kg)
        assert_tensor_equal(self, backward_result[2], vg)

    def test_flashmask_attention_cp_requires_flashmask_for_p2p_mode(self):
        with (
            patch.object(cp_utils, "_flash_mask_available", False),
            patch.object(
                cp_utils.fleet,
                "get_hybrid_communicate_group",
                lambda: FakeHcg(self.group),
            ),
            self.assertRaises(AssertionError),
        ):
            cp_utils.flashmask_attention_cp(
                self.query,
                self.key,
                self.value,
                self.indices,
                mode="contiguous_swap2p",
            )

    def test_flashmask_attention_cp_dispatches_p2p_mode_to_pylayer(self):
        captured = {}
        result = paddle.full([1, 4, 1, 1], 5.0, dtype="float32")
        learnable_sink = paddle.ones([1], dtype="float32")

        def fake_apply(*args):
            captured["args"] = args
            return result

        with (
            patch.object(cp_utils, "_flash_mask_available", True),
            patch.object(
                cp_utils.fleet,
                "get_hybrid_communicate_group",
                lambda: FakeHcg(self.group),
            ),
            patch.object(cp_utils.FlashMaskSwaP2P, "apply", fake_apply),
        ):
            out = cp_utils.flashmask_attention_cp(
                self.query,
                self.key,
                self.value,
                self.indices,
                causal=True,
                training=False,
                learnable_sink=learnable_sink,
                softmax_scale=0.5,
                mode="contiguous_swap2p",
                window_size=64,
            )

        assert_tensor_equal(self, out, result)
        self.assertIs(captured["args"][0], self.query)
        self.assertIs(captured["args"][1], self.key)
        self.assertIs(captured["args"][2], self.value)
        self.assertTrue(captured["args"][6])
        self.assertFalse(captured["args"][7])
        self.assertIs(captured["args"][8], learnable_sink)
        self.assertEqual(captured["args"][9], 0.5)
        self.assertIs(captured["args"][10], self.group)
        self.assertEqual(captured["args"][11], "contiguous_swap2p")
        self.assertEqual(captured["args"][12], 64)


if __name__ == "__main__":
    unittest.main()
