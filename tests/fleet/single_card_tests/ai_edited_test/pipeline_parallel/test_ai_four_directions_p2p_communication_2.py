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

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import paddle

from paddleformers.fleet.pipeline_parallel.pp_utils import (
    four_directions_p2p_communication as fd,
)
from paddleformers.fleet.pipeline_parallel.pp_utils.utils import paddle_2_number


class Request:
    def __init__(self, calls, name):
        self.calls = calls
        self.name = name
        self.wait_called = False

    def wait(self):
        self.wait_called = True
        self.calls.append(("request_wait", self.name))


class ProcessGroup:
    def __init__(self, calls):
        self.calls = calls

    def send_partial_on_calc_stream(self, tensor, dst, nranks, rank_id):
        self.calls.append(("send_partial_on_calc_stream", dst, nranks, rank_id))
        return Request(self.calls, "send_partial_on_calc_stream")

    def send_partial(self, tensor, dst, nranks, rank_id):
        self.calls.append(("send_partial", dst, nranks, rank_id))
        return Request(self.calls, "send_partial")

    def recv_partial_on_calc_stream(self, tensor, src, nranks, rank_id):
        self.calls.append(("recv_partial_on_calc_stream", src, nranks, rank_id))
        return Request(self.calls, "recv_partial_on_calc_stream")

    def recv_partial(self, tensor, src, nranks, rank_id):
        self.calls.append(("recv_partial", src, nranks, rank_id))
        return Request(self.calls, "recv_partial")

    def all_gather_partial_on_calc_stream(self, out, tensor, nranks, rank_id):
        self.calls.append(("all_gather_partial_on_calc_stream", nranks, rank_id))
        return Request(self.calls, "all_gather_partial_on_calc_stream")

    def all_gather_partial(self, out, tensor, nranks, rank_id):
        self.calls.append(("all_gather_partial", nranks, rank_id))
        return Request(self.calls, "all_gather_partial")


class Group:
    id = 11
    rank = 0
    ranks = [0, 1]
    backend = "nccl"

    def __init__(self, calls, member=True):
        self.calls = calls
        self._member = member
        self.process_group = ProcessGroup(calls)

    def is_member(self):
        return self._member

    def get_group_rank(self, rank):
        self.calls.append(("get_group_rank", rank))
        return rank + 200


class HCG:
    def __init__(self, calls):
        self.calls = calls
        self.send_next_group = Group(calls)
        self.send_prev_group = Group(calls)
        self.recv_next_group = Group(calls)
        self.recv_prev_group = Group(calls)
        self.pipe_group = Group(calls)
        self.model_group = Group(calls)

    def get_p2p_groups(self):
        return (
            self.send_next_group,
            self.send_prev_group,
            self.recv_next_group,
            self.recv_prev_group,
        )

    def get_pipe_parallel_group(self):
        return self.pipe_group

    def get_model_parallel_group(self):
        return self.model_group

    def get_model_parallel_world_size(self):
        return 2

    def get_model_parallel_rank(self):
        return 1

    def _get_p2p_prev_rank(self):
        return 4

    def _get_p2p_next_rank(self):
        return 5


class TimerItem:
    def __init__(self, calls, name):
        self.calls = calls
        self.name = name

    def start(self):
        self.calls.append(("timer_start", self.name))

    def stop(self):
        self.calls.append(("timer_stop", self.name))


class Timers:
    def __init__(self, calls):
        self.calls = calls

    def __call__(self, name):
        return TimerItem(self.calls, name)


class FourDirectionsStateTest(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.hcg = HCG(self.calls)
        self.old_hcg = fd._hcg
        self.old_enable = fd._enable_partial_send_recv
        self.old_sync_send = fd._sync_send
        self.old_timers = fd._timers
        self.old_send = paddle.distributed.send
        self.old_recv = paddle.distributed.recv
        self.old_isend = paddle.distributed.isend
        self.old_irecv = paddle.distributed.irecv
        self.old_wait = paddle.distributed.wait
        self.old_send_partial = fd.send_partial
        self.old_recv_partial = fd.recv_partial
        self.old_allgather_partial = fd.allgather_partial
        self.old_in_dynamic_mode = fd.framework.in_dynamic_mode
        fd._hcg = self.hcg
        fd._enable_partial_send_recv = True
        fd.framework.in_dynamic_mode = lambda: True

    def tearDown(self):
        fd._hcg = self.old_hcg
        fd._enable_partial_send_recv = self.old_enable
        fd._sync_send = self.old_sync_send
        fd._timers = self.old_timers
        paddle.distributed.send = self.old_send
        paddle.distributed.recv = self.old_recv
        paddle.distributed.isend = self.old_isend
        paddle.distributed.irecv = self.old_irecv
        paddle.distributed.wait = self.old_wait
        fd.send_partial = self.old_send_partial
        fd.recv_partial = self.old_recv_partial
        fd.allgather_partial = self.old_allgather_partial
        fd.framework.in_dynamic_mode = self.old_in_dynamic_mode

    def _tensor(self, shape=(2, 2), dtype="float32"):
        tensor = paddle.ones(shape, dtype=dtype)
        tensor.stop_gradient = False
        return tensor

    def _fill(self, tensor, values):
        tensor.set_value(paddle.to_tensor(values, dtype=tensor.dtype))

    def _meta(self, tuple_recv=False, tuple_send=False):
        meta = fd.SendRecvMeta()
        if tuple_recv:
            meta.recv_shape_message = ([2, 2], [1, 4])
            meta.recv_dtype_message = (
                paddle_2_number(paddle.float32),
                paddle_2_number(paddle.float16),
            )
            meta.recv_stop_gradient = (False, True)
        else:
            meta.recv_shape_message = [2, 2]
            meta.recv_dtype_message = paddle_2_number(paddle.float32)
            meta.recv_stop_gradient = False
        if tuple_send:
            meta.send_shape_message = ([2, 2], [4, 1])
            meta.send_dtype_message = (
                paddle_2_number(paddle.float32),
                paddle_2_number(paddle.int64),
            )
        else:
            meta.send_shape_message = [2, 2]
            meta.send_dtype_message = paddle_2_number(paddle.float32)
        return meta


class TestFourDirectionsMetaAndPartial(FourDirectionsStateTest):
    def test_initialize_and_recv_meta_single_and_tuple(self):
        fd.initialize_p2p_groups(self.hcg, enable_partial_send_recv=False)
        self.assertIs(fd._hcg, self.hcg)
        self.assertFalse(fd._enable_partial_send_recv)
        fd._enable_partial_send_recv = True

        single_values = [
            [0],
            [2],
            [3, 4],
            [paddle_2_number(paddle.float32)],
            [1],
        ]

        def recv_single(tensor, src, group):
            self.calls.append(("recv_single", src, group is self.hcg.pipe_group))
            self._fill(tensor, single_values.pop(0))

        paddle.distributed.recv = recv_single
        meta = fd.SendRecvMeta()
        meta.recv_meta(self.hcg.pipe_group)
        self.assertEqual(meta.recv_shape_message, [3, 4])
        self.assertTrue(meta.recv_stop_gradient)

        tuple_values = [
            [1],
            [2],
            [1],
            [5],
            [paddle_2_number(paddle.int64)],
            [0],
            [2],
            [2, 3],
            [paddle_2_number(paddle.float16)],
            [1],
        ]

        def recv_tuple(tensor, src, group):
            self.calls.append(("recv_tuple", src, group is self.hcg.pipe_group))
            self._fill(tensor, tuple_values.pop(0))

        paddle.distributed.recv = recv_tuple
        meta = fd.SendRecvMeta()
        meta.recv_meta(self.hcg.pipe_group)
        self.assertEqual(meta.recv_shape_message, ([5], [2, 3]))
        self.assertEqual(meta.recv_stop_gradient, (False, True))

    def test_send_recv_partial_and_invalid_paths(self):
        tensor = self._tensor([4], "float32")
        self.assertEqual(
            fd.send_partial(
                tensor,
                dst=1,
                nranks=2,
                rank_id=1,
                group=self.hcg.send_next_group,
            ).name,
            "send_partial_on_calc_stream",
        )
        self.assertEqual(
            fd.recv_partial(
                tensor,
                src=0,
                nranks=2,
                rank_id=1,
                group=self.hcg.recv_prev_group,
            ).name,
            "recv_partial_on_calc_stream",
        )
        self.assertEqual(
            fd.send_partial(
                tensor,
                dst=0,
                nranks=2,
                rank_id=0,
                group=self.hcg.send_prev_group,
                use_calc_stream=False,
            ).name,
            "send_partial",
        )
        self.assertEqual(
            fd.recv_partial(
                tensor,
                src=1,
                nranks=2,
                rank_id=0,
                group=self.hcg.recv_next_group,
                use_calc_stream=False,
            ).name,
            "recv_partial",
        )
        self.assertEqual(
            fd.allgather_partial(tensor, nranks=2, rank_id=1, group=self.hcg.model_group).name,
            "all_gather_partial_on_calc_stream",
        )
        self.assertEqual(
            fd.allgather_partial(
                tensor,
                nranks=2,
                rank_id=1,
                group=self.hcg.model_group,
                use_calc_stream=False,
            ).name,
            "all_gather_partial",
        )

        fd._enable_partial_send_recv = False
        calls = []

        def isend(value, dst, group):
            calls.append(("isend", dst))
            return Request(self.calls, "isend")

        def recv(value, src, group):
            calls.append(("recv", src))
            return Request(self.calls, "recv")

        def irecv(value, src, group):
            calls.append(("irecv", src))
            return Request(self.calls, "irecv")

        paddle.distributed.isend = isend
        paddle.distributed.recv = recv
        paddle.distributed.irecv = irecv
        self.assertEqual(
            fd.send_partial(tensor, dst=1, group=self.hcg.send_next_group).name,
            "isend",
        )
        self.assertEqual(
            fd.recv_partial(tensor, src=0, group=self.hcg.recv_prev_group).name,
            "recv",
        )
        self.assertEqual(
            fd.recv_partial(
                tensor,
                src=0,
                group=self.hcg.recv_prev_group,
                use_calc_stream=False,
            ).name,
            "irecv",
        )
        self.assertIs(fd.allgather_partial(tensor, nranks=2), tensor)
        non_member = Group(self.calls, member=False)
        self.assertIsNone(fd.send_partial(tensor, group=non_member))
        self.assertIsNone(fd.recv_partial(tensor, group=non_member))
        fd._enable_partial_send_recv = True
        with self.assertRaises(AssertionError):
            fd._is_valid_send_recv_partial(paddle.empty([0]), 2)


class TestFourDirectionsHelperCore(FourDirectionsStateTest):
    def _install_lightweight_ops(self):
        def send_partial(tensor, dst=0, nranks=1, rank_id=0, group=None, use_calc_stream=True):
            self.calls.append(("send_partial", dst, nranks, rank_id, use_calc_stream))
            return Request(self.calls, f"send-{dst}")

        def recv_partial(tensor, src=0, nranks=1, rank_id=0, group=None, use_calc_stream=True):
            self.calls.append(("recv_partial", src, nranks, rank_id, use_calc_stream))
            return Request(self.calls, f"recv-{src}")

        def allgather_partial(tensor, nranks=1, rank_id=0, group=None, use_calc_stream=True):
            self.calls.append(("allgather_partial", nranks, rank_id, use_calc_stream))
            return Request(self.calls, "allgather")

        def wait(tensor, use_calc_stream=True):
            self.calls.append(("wait", tuple(tensor.shape), use_calc_stream))

        fd.send_partial = send_partial
        fd.recv_partial = recv_partial
        fd.allgather_partial = allgather_partial
        paddle.distributed.wait = wait

    def test_p2p_helper_async_and_sync_branches(self):
        self._install_lightweight_ops()
        fd._sync_send = False
        meta = self._meta(tuple_recv=True, tuple_send=True)
        recv_prev, recv_next = fd._p2p_helper(
            (self._tensor([2, 2]), self._tensor([4, 1])),
            (self._tensor([2, 2]), self._tensor([1, 4])),
            recv_prev=True,
            recv_next=True,
            sync_recv=False,
            send_recv_meta=meta,
        )
        self.assertEqual(len(recv_prev), 2)
        self.assertEqual(len(recv_next), 2)
        self.assertTrue(any(call[0] == "request_wait" for call in self.calls))
        self.assertEqual(
            len([call for call in self.calls if call[0] == "allgather_partial"]),
            4,
        )

        self.calls.clear()
        fd._sync_send = True
        meta = self._meta(tuple_recv=False, tuple_send=False)
        recv_prev, recv_next = fd._p2p_helper(
            self._tensor([2, 2]),
            self._tensor([2, 2]),
            recv_prev=True,
            recv_next=True,
            sync_recv=True,
            send_recv_meta=meta,
        )
        self.assertEqual(recv_prev.shape, [2, 2])
        self.assertEqual(recv_next.shape, [2, 2])
        self.assertTrue(any(call[0] == "wait" for call in self.calls))


class TestFourDirectionsPublicHelper(FourDirectionsStateTest):
    def setUp(self):
        super().setUp()
        fd._timers = Timers(self.calls)
        self.old_helper = fd._p2p_helper

        def helper(
            tensor_send_next,
            tensor_send_prev,
            recv_prev,
            recv_next,
            sync_recv=True,
            send_recv_meta=None,
        ):
            self.calls.append(
                (
                    "p2p_helper",
                    tensor_send_next is not None,
                    tensor_send_prev is not None,
                    recv_prev,
                    recv_next,
                    sync_recv,
                )
            )
            prev = paddle.ones([1], dtype="float32") if recv_prev else None
            nxt = paddle.ones([1], dtype="float32") if recv_next else None
            return prev, nxt

        fd._p2p_helper = helper
        paddle.distributed.send = lambda tensor, dst, group: self.calls.append(("send", dst))
        paddle.distributed.recv = lambda tensor, src, group: self._fill(tensor, [0])

    def tearDown(self):
        fd._p2p_helper = self.old_helper
        super().tearDown()

    def _prepared_helper(self):
        helper = fd.P2pHelper(use_cache=True)
        helper._send_recv_meta.recv_shape_message = [1]
        helper._send_recv_meta.recv_dtype_message = paddle_2_number(paddle.float32)
        helper._send_recv_meta.recv_stop_gradient = False
        helper._send_recv_meta.send_shape_message = [1]
        helper._send_recv_meta.send_dtype_message = paddle_2_number(paddle.float32)
        helper._send_recv_meta.has_recv_meta = True
        helper._send_recv_meta.has_send_meta = True
        return helper

    def test_public_methods_cover_stage_shortcuts_and_transfers(self):
        helper = self._prepared_helper()
        tensor = self._tensor([1])
        self.assertIsNone(helper.recv_forward(pp_first_stage=True))
        self.assertIsNotNone(helper.recv_forward(pp_first_stage=False, sync_recv=False))
        self.assertIsNone(helper.recv_backward(pp_last_stage=True))
        self.assertIsNotNone(helper.recv_backward(pp_last_stage=False))
        helper.send_forward(tensor, pp_last_stage=True)
        helper.send_forward(tensor, pp_last_stage=False)
        helper.send_backward(tensor, pp_first_stage=True)
        helper.send_backward(tensor, pp_first_stage=False)
        self.assertIsNone(helper.send_forward_recv_backward(tensor, pp_last_stage=True))
        self.assertIsNotNone(helper.send_forward_recv_backward(tensor, pp_last_stage=False))
        self.assertIsNone(helper.send_backward_recv_forward(tensor, pp_first_stage=True))
        self.assertIsNotNone(helper.send_backward_recv_forward(tensor, pp_first_stage=False))
        prev, nxt = helper.send_forward_backward_recv_forward_backward(tensor, tensor, recv_prev=True, recv_next=True)
        self.assertIsNotNone(prev)
        self.assertIsNotNone(nxt)
        self.assertIsNone(helper.send_forward_recv_forward(None, recv_prev=False))
        self.assertIsNotNone(helper.send_forward_recv_forward(tensor, recv_prev=True))
        self.assertIsNone(helper.send_backward_recv_backward(tensor, recv_next=False))
        self.assertIsNotNone(helper.send_backward_recv_backward(tensor, recv_next=True))
        self.assertTrue(any(call[0] == "timer_start" for call in self.calls))


if __name__ == "__main__":
    unittest.main()
