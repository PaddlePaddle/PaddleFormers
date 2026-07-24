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
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

import paddle

from paddleformers.fleet.pipeline_parallel.pp_utils import (
    four_directions_p2p_communication,
)
from paddleformers.fleet.pipeline_parallel.pp_utils.four_directions_p2p_communication import (
    P2pHelper,
    SendRecvMeta,
    _partial_recv_op,
    _partial_send_op,
    allgather_partial,
    recv_partial,
    send_partial,
)
from paddleformers.fleet.pipeline_parallel.pp_utils.utils import paddle_2_number


class RecordingProcessGroup:
    def __init__(self):
        self.calls = []

    def send_partial_on_calc_stream(self, tensor, rank, nranks, rank_id):
        self.calls.append(("send_partial_calc", tensor, rank, nranks, rank_id))
        return "send-calc-task"

    def send_partial(self, tensor, rank, nranks, rank_id):
        self.calls.append(("send_partial", tensor, rank, nranks, rank_id))
        return "send-task"

    def recv_partial_on_calc_stream(self, tensor, rank, nranks, rank_id):
        self.calls.append(("recv_partial_calc", tensor, rank, nranks, rank_id))
        return "recv-calc-task"

    def recv_partial(self, tensor, rank, nranks, rank_id):
        self.calls.append(("recv_partial", tensor, rank, nranks, rank_id))
        return "recv-task"

    def all_gather_partial_on_calc_stream(
        self, out_tensor, in_tensor, nranks, rank_id
    ):
        self.calls.append(
            ("allgather_calc", out_tensor, in_tensor, nranks, rank_id)
        )
        return "allgather-calc-task"

    def all_gather_partial(self, out_tensor, in_tensor, nranks, rank_id):
        self.calls.append(("allgather", out_tensor, in_tensor, nranks, rank_id))
        return "allgather-task"


class RecordingGroup:
    id = 5

    def __init__(self, is_member=True):
        self.process_group = RecordingProcessGroup()
        self._is_member = is_member

    def is_member(self):
        return self._is_member

    def get_group_rank(self, rank):
        return rank + 20


class DummyHCG:
    def __init__(self):
        self.send_next_group = RecordingGroup()
        self.send_prev_group = RecordingGroup()
        self.recv_next_group = RecordingGroup()
        self.recv_prev_group = RecordingGroup()
        self.model_group = RecordingGroup()

    def get_model_parallel_group(self):
        return self.model_group

    def get_model_parallel_world_size(self):
        return 1

    def get_model_parallel_rank(self):
        return 0

    def _get_p2p_prev_rank(self):
        return 2

    def _get_p2p_next_rank(self):
        return 3


class TestFourDirectionMetaExtra(unittest.TestCase):
    def test_set_send_message_filters_all_stop_gradient_tuple(self):
        meta = SendRecvMeta()
        first = paddle.ones([2], dtype="float32")
        second = paddle.ones([3], dtype="float32")
        first.stop_gradient = True
        second.stop_gradient = True

        meta.set_send_message((first, second))

        self.assertEqual(meta.send_shape_message, ())
        self.assertEqual(meta.send_dtype_message, ())

    def test_set_send_message_ignores_unsupported_list(self):
        meta = SendRecvMeta()
        meta.send_shape_message = [1]
        meta.send_dtype_message = paddle_2_number(paddle.float32)

        meta.set_send_message([paddle.ones([2], dtype="float32")])

        self.assertEqual(meta.send_shape_message, [1])
        self.assertEqual(
            meta.send_dtype_message, paddle_2_number(paddle.float32)
        )


class TestFourDirectionPartialExtra(unittest.TestCase):
    def test_disabled_partial_allows_zero_tensor_without_assertion(self):
        old_value = four_directions_p2p_communication._enable_partial_send_recv
        four_directions_p2p_communication._enable_partial_send_recv = False
        try:
            tensor = paddle.empty([0], dtype="float32")
            self.assertFalse(
                four_directions_p2p_communication._is_valid_send_recv_partial(
                    tensor, 2
                )
            )
            self.assertIs(allgather_partial(tensor, nranks=2), tensor)
        finally:
            four_directions_p2p_communication._enable_partial_send_recv = (
                old_value
            )

    def test_non_member_partial_send_recv_and_allgather_return_none(self):
        group = RecordingGroup(is_member=False)
        tensor = paddle.ones([4], dtype="float32")
        old_hcg = four_directions_p2p_communication._hcg
        four_directions_p2p_communication._hcg = DummyHCG()
        try:
            self.assertIsNone(
                send_partial(tensor, dst=1, nranks=2, group=group)
            )
            self.assertIsNone(
                recv_partial(tensor, src=0, nranks=2, group=group)
            )
            self.assertIsNone(allgather_partial(tensor, nranks=2, group=group))
        finally:
            four_directions_p2p_communication._hcg = old_hcg

    def test_partial_send_recv_ops_use_group_rank_and_stream_flag(self):
        tensor = paddle.ones([4], dtype="float32")
        group = RecordingGroup()

        self.assertEqual(
            _partial_send_op(tensor, group, True, group.id, 4, 2, 1),
            "send-calc-task",
        )
        self.assertEqual(
            _partial_send_op(tensor, group, False, group.id, 5, 2, 1),
            "send-task",
        )
        self.assertEqual(
            _partial_recv_op(tensor, group, True, group.id, 6, 2, 1),
            "recv-calc-task",
        )
        self.assertEqual(
            _partial_recv_op(tensor, group, False, group.id, 7, 2, 1),
            "recv-task",
        )
        self.assertEqual(
            [call[0] for call in group.process_group.calls],
            [
                "send_partial_calc",
                "send_partial",
                "recv_partial_calc",
                "recv_partial",
            ],
        )
        self.assertEqual(group.process_group.calls[0][2:], (24, 2, 1))
        self.assertEqual(group.process_group.calls[3][2:], (27, 2, 1))

    def test_xpu_group_start_end_noop_on_non_xpu(self):
        four_directions_p2p_communication._xpu_comm_group_started = False
        four_directions_p2p_communication._xpu_comm_group_start()
        four_directions_p2p_communication._xpu_comm_group_end()
        self.assertFalse(
            four_directions_p2p_communication._xpu_comm_group_started
        )


class TestFourDirectionP2pHelperExtra(unittest.TestCase):
    def test_p2p_helper_asserts_meta_and_no_comm_returns_empty_pair(self):
        old_hcg = four_directions_p2p_communication._hcg
        four_directions_p2p_communication._hcg = DummyHCG()
        try:
            with self.assertRaises(AssertionError):
                four_directions_p2p_communication._p2p_helper(
                    None, None, False, False, send_recv_meta=None
                )

            self.assertEqual(
                four_directions_p2p_communication._p2p_helper(
                    None,
                    None,
                    False,
                    False,
                    send_recv_meta=SendRecvMeta(),
                ),
                (None, None),
            )
        finally:
            four_directions_p2p_communication._hcg = old_hcg

    def test_helper_early_stage_methods_and_cache_flag(self):
        helper = P2pHelper(use_cache=False)
        self.assertFalse(helper._use_cache)
        self.assertIsNone(helper.recv_forward(pp_first_stage=True))
        self.assertIsNone(helper.recv_backward(pp_last_stage=True))
        self.assertIsNone(
            helper.send_forward_recv_backward(
                paddle.ones([1], dtype="float32"), pp_last_stage=True
            )
        )
        self.assertIsNone(
            helper.send_backward_recv_forward(
                paddle.ones([1], dtype="float32"), pp_first_stage=True
            )
        )


if __name__ == "__main__":
    unittest.main()
