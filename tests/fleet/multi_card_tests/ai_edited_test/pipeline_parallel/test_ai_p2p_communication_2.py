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

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)

import paddle

from paddleformers.fleet.pipeline_parallel.pp_utils import p2p_communication
from paddleformers.fleet.pipeline_parallel.pp_utils.p2p_communication import (
    P2pHelper,
    P2PonCalcStream,
    SendRecvMeta,
    _batch_p2p_tuple_or_tensor,
    _recv_on_calc_stream,
    _send_on_calc_stream,
)
from paddleformers.fleet.pipeline_parallel.pp_utils.utils import paddle_2_number


class RecordingProcessGroup:
    def __init__(self):
        self.calls = []

    def send_on_calc_stream(self, tensor, rank):
        self.calls.append(("send", tensor, rank))
        return "send-task"

    def recv_on_calc_stream(self, tensor, rank):
        self.calls.append(("recv", tensor, rank))
        return "recv-task"

    def send_partial_on_calc_stream(self, tensor, rank, nranks, rank_id):
        self.calls.append(("send_partial", tensor, rank, nranks, rank_id))
        return "send-partial-task"

    def recv_partial_on_calc_stream(self, tensor, rank, nranks, rank_id):
        self.calls.append(("recv_partial", tensor, rank, nranks, rank_id))
        return "recv-partial-task"


class RecordingGroup:
    id = 7

    def __init__(self):
        self.process_group = RecordingProcessGroup()

    def is_member(self):
        return True

    def get_group_rank(self, rank):
        return rank + 100


class DummyHCG:
    def __init__(self):
        self.pipe_group = RecordingGroup()
        self.model_group = RecordingGroup()

    def get_pipe_parallel_group(self):
        return self.pipe_group

    def get_model_parallel_world_size(self):
        return 1

    def get_model_parallel_rank(self):
        return 0

    def get_model_parallel_group(self):
        return self.model_group

    def _get_p2p_prev_rank(self):
        return 2

    def _get_p2p_next_rank(self):
        return 3


class TestSendRecvMetaExtra(unittest.TestCase):
    def test_init_or_erase_meta_clears_all_fields(self):
        meta = SendRecvMeta()
        meta.send_shape_message = [1]
        meta.send_dtype_message = 1
        meta.send_key_message = "send"
        meta.recv_shape_message = [2]
        meta.recv_dtype_message = 2
        meta.recv_stop_gradient = True
        meta.recv_key_message = "recv"
        meta.has_send_meta = True
        meta.has_recv_meta = True

        meta.init_or_erase_meta()

        self.assertIsNone(meta.send_shape_message)
        self.assertIsNone(meta.send_dtype_message)
        self.assertIsNone(meta.send_key_message)
        self.assertIsNone(meta.recv_shape_message)
        self.assertIsNone(meta.recv_dtype_message)
        self.assertIsNone(meta.recv_stop_gradient)
        self.assertIsNone(meta.recv_key_message)
        self.assertFalse(meta.has_send_meta)
        self.assertFalse(meta.has_recv_meta)

    def test_obtain_send_message_filters_stop_gradient_and_keeps_keys(self):
        first = paddle.ones([2, 3], dtype="float32")
        first.stop_gradient = False
        first.key = "first"
        skipped = paddle.ones([1], dtype="float32")
        skipped.stop_gradient = True
        last = paddle.ones([4], dtype="int64")
        last.stop_gradient = False
        last.key = "last"

        shapes, dtypes, keys = SendRecvMeta()._obtain_send_message([first, skipped, last])

        self.assertEqual(shapes, ([2, 3], [4]))
        self.assertEqual(
            dtypes,
            (paddle_2_number(first.dtype), paddle_2_number(last.dtype)),
        )
        self.assertEqual(keys, ("first", "last"))

    def test_check_send_message_validates_dtype_and_key(self):
        meta = SendRecvMeta()
        tensor = paddle.ones([2], dtype="float32")
        tensor.key = "expected"
        meta.set_send_message(tensor)

        wrong_dtype = paddle.ones([2], dtype="float16")
        wrong_dtype.key = "expected"
        with self.assertRaises(AssertionError):
            meta.check_send_message(wrong_dtype)

        wrong_key = paddle.ones([2], dtype="float32")
        wrong_key.key = "actual"
        with self.assertRaises(AssertionError):
            meta.check_send_message(wrong_key)

    def test_obtain_send_message_rejects_non_tensor_member(self):
        with self.assertRaises(AssertionError):
            SendRecvMeta()._obtain_send_message((paddle.ones([1]), object()))


class TestP2PCommunicationOpsExtra(unittest.TestCase):
    def test_send_and_recv_on_calc_stream_select_partial_or_full_ops(self):
        group = RecordingGroup()
        partial_tensor = paddle.ones([4], dtype="float32")
        full_tensor = paddle.ones([3], dtype="float32")

        self.assertEqual(
            _send_on_calc_stream(partial_tensor, group, 5, nranks=2, rank_id=1),
            "send-partial-task",
        )
        self.assertEqual(
            _recv_on_calc_stream(partial_tensor, group, 6, nranks=2, rank_id=1),
            "recv-partial-task",
        )
        self.assertEqual(
            _send_on_calc_stream(full_tensor, group, 7, nranks=2, rank_id=1),
            "send-task",
        )
        self.assertEqual(
            _recv_on_calc_stream(full_tensor, group, 8, nranks=2, rank_id=1),
            "recv-task",
        )
        self.assertEqual(
            [call[0] for call in group.process_group.calls],
            ["send_partial", "recv_partial", "send", "recv"],
        )
        self.assertEqual(group.process_group.calls[0][2:], (105, 2, 1))
        self.assertEqual(group.process_group.calls[2][2], 107)

    def test_batch_p2p_tuple_or_tensor_builds_operations(self):
        group = RecordingGroup()
        first = paddle.ones([1], dtype="float32")
        second = paddle.ones([2], dtype="float32")

        single_ops = _batch_p2p_tuple_or_tensor(first, _send_on_calc_stream, 3, group, mp_degree=4, mp_rank=2)
        tuple_ops = _batch_p2p_tuple_or_tensor(
            (first, second),
            _recv_on_calc_stream,
            4,
            group,
            mp_degree=2,
            mp_rank=1,
        )

        self.assertEqual(len(single_ops), 1)
        self.assertIsInstance(single_ops[0], P2PonCalcStream)
        self.assertIs(single_ops[0].tensor, first)
        self.assertIs(single_ops[0].group, group)
        self.assertEqual(single_ops[0].peer, 3)
        self.assertEqual(single_ops[0].nranks, 4)
        self.assertEqual(single_ops[0].rank_id, 2)
        self.assertEqual([op.tensor for op in tuple_ops], [first, second])
        self.assertEqual(
            [op.op for op in tuple_ops],
            [_recv_on_calc_stream, _recv_on_calc_stream],
        )

    def test_invalid_p2p_op_is_rejected(self):
        with self.assertRaises(RuntimeError):
            P2PonCalcStream(lambda *args: None, paddle.ones([1]), 0, RecordingGroup())


class TestP2pHelperExtra(unittest.TestCase):
    def test_dynamic_shape_helper_initial_state_and_unsupported_methods(self):
        helper = P2pHelper(use_cache=False, dynamic_shape=True)
        self.assertFalse(helper._use_cache)
        self.assertTrue(helper._dynamic_shape)
        self.assertEqual(helper._send_recv_meta_list, [])
        self.assertEqual(helper._dynamic_cnt, 0)

        tensor = paddle.ones([1], dtype="float32")
        with self.assertRaises(AssertionError):
            helper.send_forward_recv_backward(tensor, pp_last_stage=True)
        with self.assertRaises(AssertionError):
            helper.send_backward_recv_forward(tensor, pp_first_stage=True)
        with self.assertRaises(AssertionError):
            helper.send_forward_backward_recv_forward_backward(tensor, tensor, recv_prev=False, recv_next=False)

    def test_clear_meta_cache_and_repr(self):
        helper = P2pHelper(use_cache=True)
        helper._send_recv_meta.send_shape_message = [2]
        helper._send_recv_meta.send_dtype_message = 1
        helper._send_recv_meta.has_send_meta = True

        helper.clear_meta_cache()

        self.assertIsNone(helper._send_recv_meta.send_shape_message)
        self.assertIsNone(helper._send_recv_meta.send_dtype_message)
        self.assertFalse(helper._send_recv_meta.has_send_meta)
        self.assertIn("using cache: True", repr(helper))

    def test_p2p_helper_asserts_meta_and_no_comm_returns_empty_triplet(self):
        old_hcg = p2p_communication._hcg
        p2p_communication._hcg = DummyHCG()
        try:
            with self.assertRaises(AssertionError):
                p2p_communication._p2p_helper(None, None, False, False, send_recv_meta=None)

            self.assertEqual(
                p2p_communication._p2p_helper(
                    None,
                    None,
                    False,
                    False,
                    send_recv_meta=SendRecvMeta(),
                ),
                (None, None, None),
            )
        finally:
            p2p_communication._hcg = old_hcg


if __name__ == "__main__":
    unittest.main()
