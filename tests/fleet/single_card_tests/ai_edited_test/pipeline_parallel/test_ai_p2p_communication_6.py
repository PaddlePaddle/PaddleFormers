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

from paddleformers.fleet.pipeline_parallel.pp_utils import p2p_communication as p2p
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

    def send_on_calc_stream(self, tensor, dst):
        self.calls.append(("send_on_calc_stream", dst))
        return Request(self.calls, "send_on_calc_stream")

    def recv_partial_on_calc_stream(self, tensor, src, nranks, rank_id):
        self.calls.append(("recv_partial_on_calc_stream", src, nranks, rank_id))
        return Request(self.calls, "recv_partial_on_calc_stream")

    def recv_on_calc_stream(self, tensor, src):
        self.calls.append(("recv_on_calc_stream", src))
        return Request(self.calls, "recv_on_calc_stream")

    def all_gather_partial_on_calc_stream(self, out, tensor, nranks, rank_id):
        self.calls.append(("all_gather_partial_on_calc_stream", nranks, rank_id))
        return Request(self.calls, "all_gather_partial_on_calc_stream")

    def all_gather_partial(self, out, tensor, nranks, rank_id):
        self.calls.append(("all_gather_partial", nranks, rank_id))
        return Request(self.calls, "all_gather_partial")


class Group:
    id = 7
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
        return rank + 100


class HCG:
    def __init__(self, calls):
        self.calls = calls
        self.stage_id = 0
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

    def get_stage_id(self):
        return self.stage_id

    def _get_p2p_prev_rank(self):
        return 2

    def _get_p2p_next_rank(self):
        return 3


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


class P2PStateTest(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.hcg = HCG(self.calls)
        self.old_hcg = p2p._hcg
        self.old_enable = p2p._enable_partial_send_recv
        self.old_sync_send = p2p._sync_send
        self.old_timers = p2p._timers
        self.old_send = paddle.distributed.send
        self.old_recv = paddle.distributed.recv
        self.old_isend = paddle.distributed.isend
        self.old_irecv = paddle.distributed.irecv
        self.old_wait = paddle.distributed.wait
        self.old_broadcast = paddle.distributed.broadcast
        self.old_timer_get_timers = p2p.timer.get_timers
        self.old_cuda_synchronize = paddle.device.cuda.synchronize
        self.old_batched_p2p_ops = p2p._batched_p2p_ops
        self.old_p2p_ops = p2p._p2p_ops
        self.old_batch_send_recv = p2p.batch_send_recv_on_calc_stream
        self.old_warn = p2p._warn_cur_rank_not_in_group
        self.old_coalescing_manager = p2p._coalescing_manager
        self.old_check_naninf = p2p.check_naninf
        self.old_get_rank = paddle.distributed.get_rank
        self.old_allgather_partial = p2p.allgather_partial
        p2p._hcg = self.hcg
        p2p._enable_partial_send_recv = True

    def tearDown(self):
        p2p._hcg = self.old_hcg
        p2p._enable_partial_send_recv = self.old_enable
        p2p._sync_send = self.old_sync_send
        p2p._timers = self.old_timers
        paddle.distributed.send = self.old_send
        paddle.distributed.recv = self.old_recv
        paddle.distributed.isend = self.old_isend
        paddle.distributed.irecv = self.old_irecv
        paddle.distributed.wait = self.old_wait
        paddle.distributed.broadcast = self.old_broadcast
        p2p.timer.get_timers = self.old_timer_get_timers
        paddle.device.cuda.synchronize = self.old_cuda_synchronize
        p2p._batched_p2p_ops = self.old_batched_p2p_ops
        p2p._p2p_ops = self.old_p2p_ops
        p2p.batch_send_recv_on_calc_stream = self.old_batch_send_recv
        p2p._warn_cur_rank_not_in_group = self.old_warn
        p2p._coalescing_manager = self.old_coalescing_manager
        p2p.check_naninf = self.old_check_naninf
        paddle.distributed.get_rank = self.old_get_rank
        p2p.allgather_partial = self.old_allgather_partial

    def _tensor(self, shape=(2, 2), dtype="float32"):
        tensor = paddle.ones(shape, dtype=dtype)
        tensor.stop_gradient = False
        return tensor

    def _fill(self, tensor, values):
        tensor.set_value(paddle.to_tensor(values, dtype=tensor.dtype))

    def _meta(self, tuple_recv=False, tuple_send=False):
        meta = p2p.SendRecvMeta()
        if tuple_recv:
            meta.recv_shape_message = ([2, 2], [1, 4])
            meta.recv_dtype_message = (
                paddle_2_number(paddle.float32),
                paddle_2_number(paddle.float16),
            )
            meta.recv_stop_gradient = (False, True)
            meta.recv_key_message = (None, None)
        else:
            meta.recv_shape_message = [2, 2]
            meta.recv_dtype_message = paddle_2_number(paddle.float32)
            meta.recv_stop_gradient = False
            meta.recv_key_message = None
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


class TestInitializeP2PGroupsNoMock(P2PStateTest):
    def test_initialize_sets_timer_and_partial_state(self):
        p2p.timer.get_timers = lambda: Timers(self.calls)

        p2p.initialize_p2p_groups(self.hcg, enable_partial_send_recv=False, enable_timer=True)

        self.assertIs(p2p._hcg, self.hcg)
        self.assertFalse(p2p._enable_partial_send_recv)
        self.assertIsInstance(p2p._timers, Timers)


class TestSendRecvMetaNoMock(P2PStateTest):
    def test_recv_meta_single_and_tuple_messages(self):
        single_payload = [0, 2, 3, 4, paddle_2_number(paddle.float32), 1, 0]
        single_values = [[len(single_payload)], single_payload]

        def recv_single(tensor, src, group):
            self.calls.append(("recv", src, group is self.hcg.pipe_group))
            self._fill(tensor, single_values.pop(0))

        paddle.distributed.recv = recv_single
        meta = p2p.SendRecvMeta()
        meta.recv_meta(self.hcg.pipe_group)
        self.assertEqual(meta.recv_shape_message, [3, 4])
        self.assertEqual(meta.recv_dtype_message, paddle_2_number(paddle.float32))
        self.assertTrue(meta.recv_stop_gradient)

        tuple_payload = [
            1,
            2,
            1,
            5,
            paddle_2_number(paddle.int64),
            0,
            0,
            2,
            2,
            3,
            paddle_2_number(paddle.float16),
            1,
            0,
        ]
        tuple_values = [[len(tuple_payload)], tuple_payload]

        def recv_tuple(tensor, src, group):
            self.calls.append(("recv_tuple", src, group is self.hcg.pipe_group))
            self._fill(tensor, tuple_values.pop(0))

        paddle.distributed.recv = recv_tuple
        meta = p2p.SendRecvMeta()
        meta.recv_meta(self.hcg.pipe_group)
        self.assertEqual(meta.recv_shape_message, ([5], [2, 3]))
        self.assertEqual(
            meta.recv_dtype_message,
            (paddle_2_number(paddle.int64), paddle_2_number(paddle.float16)),
        )
        self.assertEqual(meta.recv_stop_gradient, (False, True))

    def test_send_meta_tensor_tuple_and_set_message(self):
        sent = []

        def send(tensor, dst, group):
            sent.append((tensor.numpy().tolist(), dst, group is self.hcg.pipe_group))

        paddle.distributed.send = send
        meta = p2p.SendRecvMeta()
        tensor = self._tensor([2, 3], "float32")
        meta.set_send_message(tensor)
        self.assertEqual(meta.send_shape_message, [2, 3])
        meta.send_meta(tensor, self.hcg.pipe_group)

        first = self._tensor([2, 2], "float32")
        second = self._tensor([1, 4], "int64")
        second.stop_gradient = True
        meta.set_send_message((first, second))
        self.assertEqual(meta.send_shape_message, ([2, 2],))
        meta.send_meta((first, second), self.hcg.pipe_group)
        self.assertEqual(len(sent), 4)
        self.assertEqual(sent[0][1], 3)

    def test_recv_meta_broadcast_with_key_message(self):
        key_tensor, key_len = p2p.convert_object_to_tensor("activation_key")
        key_data = key_tensor.astype("int64").numpy().tolist()
        payload = [
            0,
            2,
            3,
            4,
            paddle_2_number(paddle.float32),
            0,
            key_len.item(),
            *key_data,
        ]
        values = [[len(payload)], payload]

        def broadcast(tensor, src, group):
            self.calls.append(("broadcast", src, group is self.hcg.pipe_group))
            self._fill(tensor, values.pop(0))

        paddle.distributed.broadcast = broadcast
        meta = p2p.SendRecvMeta()
        meta.recv_meta(self.hcg.pipe_group, broadcast=True)

        self.assertEqual(meta.recv_shape_message, [3, 4])
        self.assertFalse(meta.recv_stop_gradient)
        self.assertEqual(meta.recv_key_message, "activation_key")
        self.assertEqual(
            [call for call in self.calls if call[0] == "broadcast"],
            [("broadcast", 0, True), ("broadcast", 0, True)],
        )

    def test_send_meta_broadcast_reverse_list_and_invalid_type(self):
        broadcasts = []

        def broadcast(tensor, src, group):
            broadcasts.append((tensor.numpy().tolist(), src, group is self.hcg.pipe_group))

        paddle.distributed.broadcast = broadcast
        first = self._tensor([2, 2], "float32")
        first.key = "forward_hidden"
        second = self._tensor([1, 4], "float16")
        meta = p2p.SendRecvMeta()
        meta.send_meta([first, second], self.hcg.pipe_group, reverse=True, broadcast=True)

        self.assertEqual(len(broadcasts), 2)
        self.assertEqual(broadcasts[0][1], self.hcg.pipe_group.ranks[0])
        self.assertTrue(broadcasts[1][2])
        self.assertEqual(broadcasts[1][0][0], 1)
        self.assertEqual(broadcasts[1][0][1], 2)
        self.assertGreater(broadcasts[1][0][7], 0)

        with self.assertRaises(TypeError):
            meta.send_meta("not-a-tensor", self.hcg.pipe_group)

    def test_check_send_message_early_return_and_mismatches(self):
        meta = p2p.SendRecvMeta()
        self.assertIsNone(meta.check_send_message(self._tensor([2, 2])))

        baseline = self._tensor([2, 2], "float32")
        baseline.key = "expected"
        meta.set_send_message(baseline)

        shape_mismatch = self._tensor([3, 2], "float32")
        shape_mismatch.key = "expected"
        with self.assertRaisesRegex(AssertionError, "send_shape_message"):
            meta.check_send_message(shape_mismatch)

        dtype_mismatch = self._tensor([2, 2], "float16")
        dtype_mismatch.key = "expected"
        with self.assertRaisesRegex(AssertionError, "send_dtype_message"):
            meta.check_send_message(dtype_mismatch)

        key_mismatch = self._tensor([2, 2], "float32")
        key_mismatch.key = "actual"
        with self.assertRaisesRegex(AssertionError, "send_key_message"):
            meta.check_send_message(key_mismatch)


class TestPartialCommunicationNoMock(P2PStateTest):
    def test_calc_stream_send_recv_and_allgather_paths(self):
        tensor = self._tensor([4], "float32")
        send_req = p2p._send_on_calc_stream(tensor, self.hcg.send_next_group, dst=3, nranks=2, rank_id=1)
        recv_req = p2p._recv_on_calc_stream(tensor, self.hcg.recv_prev_group, src=2, nranks=2, rank_id=1)
        gather_req = p2p.allgather_partial(tensor, nranks=2, rank_id=1, group=self.hcg.model_group)
        self.assertEqual(send_req.name, "send_partial_on_calc_stream")
        self.assertEqual(recv_req.name, "recv_partial_on_calc_stream")
        self.assertEqual(gather_req.name, "all_gather_partial_on_calc_stream")

        p2p._enable_partial_send_recv = False
        self.assertFalse(p2p._is_valid_send_recv_partial(tensor, 2))
        send_req = p2p._send_on_calc_stream(tensor, self.hcg.send_next_group, dst=3, nranks=2, rank_id=1)
        recv_req = p2p._recv_on_calc_stream(tensor, self.hcg.recv_prev_group, src=2, nranks=2, rank_id=1)
        self.assertEqual(send_req.name, "send_on_calc_stream")
        self.assertEqual(recv_req.name, "recv_on_calc_stream")
        self.assertIs(p2p.allgather_partial(tensor, nranks=2), tensor)

        with self.assertRaises(AssertionError):
            p2p._send_on_calc_stream(tensor, None, dst=3)
        with self.assertRaises(AssertionError):
            p2p._recv_on_calc_stream(tensor, None, src=2)

        non_member = Group(self.calls, member=False)
        p2p._enable_partial_send_recv = True
        self.assertIsNone(p2p.allgather_partial(tensor, nranks=2, group=non_member))
        self.assertEqual(
            p2p._partial_allgather_op(
                tensor,
                self.hcg.model_group,
                False,
                self.hcg.model_group.id,
                2,
                1,
            ).name,
            "all_gather_partial",
        )

    def test_zero_sized_tensor_partial_assertion(self):
        with self.assertRaises(AssertionError):
            p2p._is_valid_send_recv_partial(paddle.empty([0]), 2)


class TestP2PHelperCoreNoMock(P2PStateTest):
    def _install_lightweight_ops(self):
        def batch_send_recv(ops):
            self.calls.append(("batch", [(op.op.__name__, op.peer) for op in ops]))

        def allgather_partial(tensor, nranks=1, rank_id=0, group=None, use_calc_stream=True):
            self.calls.append(("allgather_partial", nranks, rank_id, use_calc_stream))
            return Request(self.calls, "allgather")

        p2p.batch_send_recv_on_calc_stream = batch_send_recv
        p2p.allgather_partial = allgather_partial

    def test_p2p_helper_batched_tuple_and_single_paths(self):
        self._install_lightweight_ops()
        p2p._sync_send = False
        meta = self._meta(tuple_recv=True, tuple_send=True)
        send_prev = (self._tensor([2, 2]), self._tensor([1, 4]))
        send_next = (self._tensor([2, 2]), self._tensor([4, 1]))
        recv_prev, recv_next, reqs = p2p._p2p_helper(
            send_next,
            send_prev,
            recv_prev=True,
            recv_next=True,
            sync_recv=False,
            send_recv_meta=meta,
            batch_p2p_comm=True,
        )
        self.assertEqual(len(recv_prev), 2)
        self.assertEqual(len(recv_next), 2)
        self.assertIsNone(reqs)
        self.assertEqual(
            len([call for call in self.calls if call[0] == "allgather_partial"]),
            4,
        )

        self.calls.clear()
        meta = self._meta(tuple_recv=False, tuple_send=False)
        recv_prev, recv_next, reqs = p2p._p2p_helper(
            self._tensor([2, 2]),
            self._tensor([2, 2]),
            recv_prev=True,
            recv_next=True,
            sync_recv=False,
            send_recv_meta=meta,
            batch_p2p_comm=True,
        )
        self.assertEqual(recv_prev.shape, [2, 2])
        self.assertEqual(recv_next.shape, [2, 2])
        self.assertIsNone(reqs)
        self.assertEqual(
            len([call for call in self.calls if call[0] == "allgather_partial"]),
            2,
        )

    def test_batched_p2p_sync_send_tuple_and_single_paths(self):
        self._install_lightweight_ops()
        p2p._sync_send = True
        tensors = [self._tensor([2, 2]) for _ in range(4)]
        p2p._batched_p2p_ops(*tensors, self.hcg)
        self.assertEqual(
            self.calls[0][1],
            [
                ("_recv_on_calc_stream", 2),
                ("_send_on_calc_stream", 3),
                ("_recv_on_calc_stream", 3),
                ("_send_on_calc_stream", 2),
            ],
        )
        self.assertEqual(
            len([call for call in self.calls if call[0] == "allgather_partial"]),
            2,
        )

        self.calls.clear()
        tuple_tensors = tuple((self._tensor([2, 2]), self._tensor([1, 4])) for _ in range(4))
        p2p._batched_p2p_ops(*tuple_tensors, self.hcg)
        self.assertEqual(len(self.calls[0][1]), 8)
        self.assertEqual(
            len([call for call in self.calls if call[0] == "allgather_partial"]),
            4,
        )

    def test_batched_p2p_ops_device_synchronize_flag(self):
        self._install_lightweight_ops()
        paddle.device.cuda.synchronize = lambda: self.calls.append(("cuda_synchronize",))
        old_flag = os.environ.get("FLAGS_p2p_device_synchronize")
        os.environ["FLAGS_p2p_device_synchronize"] = "1"
        try:
            p2p._batched_p2p_ops(None, None, self._tensor([2, 2]), None, self.hcg)
        finally:
            if old_flag is None:
                os.environ.pop("FLAGS_p2p_device_synchronize", None)
            else:
                os.environ["FLAGS_p2p_device_synchronize"] = old_flag

        self.assertIn(("cuda_synchronize",), self.calls)

    def test_p2p_helper_sets_recv_prev_key_and_dynamic_recv_next_meta(self):
        self._install_lightweight_ops()
        meta = self._meta(tuple_recv=True, tuple_send=False)
        meta.recv_key_message = ("first_key", "second_key")
        recv_prev, _, _ = p2p._p2p_helper(
            None,
            None,
            recv_prev=True,
            recv_next=False,
            sync_recv=False,
            send_recv_meta=meta,
            batch_p2p_comm=True,
        )
        self.assertEqual(recv_prev[0].key, "first_key")
        self.assertEqual(recv_prev[1].key, "second_key")
        self.assertFalse(recv_prev[0].stop_gradient)
        self.assertTrue(recv_prev[1].stop_gradient)

        meta = self._meta(tuple_recv=True, tuple_send=False)
        meta.recv_shape_message = ([3, 1], [1, 2])
        meta.recv_dtype_message = (
            paddle_2_number(paddle.float16),
            paddle_2_number(paddle.int64),
        )
        _, recv_next, _ = p2p._p2p_helper(
            None,
            None,
            recv_prev=False,
            recv_next=True,
            sync_recv=False,
            send_recv_meta=meta,
            batch_p2p_comm=True,
            dynamic_shape=True,
        )
        self.assertEqual(recv_next[0].shape, [3, 1])
        self.assertEqual(recv_next[0].dtype, paddle.float16)
        self.assertEqual(recv_next[1].shape, [1, 2])
        self.assertEqual(recv_next[1].dtype, paddle.int64)

    def test_batch_send_recv_nan_check_and_coalescing_paths(self):
        tensor = self._tensor([2, 2])
        ops = [
            p2p.P2PonCalcStream(p2p._send_on_calc_stream, tensor, 3, self.hcg.pipe_group, 2, 1),
            p2p.P2PonCalcStream(p2p._recv_on_calc_stream, tensor, 2, self.hcg.pipe_group, 2, 1),
        ]

        class Manager:
            def __init__(self, calls, group, tasks):
                self.calls = calls
                self.group = group
                self.tasks = tasks

            def __enter__(self):
                self.calls.append(("enter_coalescing", self.group.backend))
                return self

            def __exit__(self, exc_type, exc, tb):
                self.calls.append(("exit_coalescing", exc_type is None))
                return False

        p2p._warn_cur_rank_not_in_group = lambda group: False
        p2p._coalescing_manager = lambda group, tasks: Manager(self.calls, group, tasks)
        os.environ["FLAGS_pp_check_naninf"] = "0"
        p2p.batch_send_recv_on_calc_stream(ops)
        self.assertTrue(any(call[0] == "enter_coalescing" for call in self.calls))
        self.assertTrue(any(call[0] == "send_partial_on_calc_stream" for call in self.calls))
        self.assertTrue(any(call[0] == "recv_partial_on_calc_stream" for call in self.calls))

        self.calls.clear()
        p2p._warn_cur_rank_not_in_group = lambda group: True
        p2p.batch_send_recv_on_calc_stream(ops)
        self.assertEqual(self.calls, [])

        self.calls.clear()
        p2p._warn_cur_rank_not_in_group = lambda group: False
        p2p.check_naninf = lambda value: "bad tensor"
        paddle.distributed.get_rank = lambda: 9
        os.environ["FLAGS_pp_check_naninf"] = "1"
        with self.assertRaises(ValueError):
            p2p.batch_send_recv_on_calc_stream(ops)
        os.environ["FLAGS_pp_check_naninf"] = "0"

    def test_p2p_ops_tuple_or_tensor_nan_check_raises_for_isend(self):
        p2p.check_naninf = lambda value: "bad tensor"
        paddle.distributed.get_rank = lambda: 9
        old_flag = os.environ.get("FLAGS_pp_check_naninf")
        os.environ["FLAGS_pp_check_naninf"] = "1"
        try:
            with self.assertRaisesRegex(ValueError, "rank 9"):
                p2p._p2p_ops_tuple_or_tensor(
                    self._tensor([2, 2]),
                    paddle.distributed.isend,
                    3,
                    self.hcg.pipe_group,
                )
        finally:
            if old_flag is None:
                os.environ.pop("FLAGS_pp_check_naninf", None)
            else:
                os.environ["FLAGS_pp_check_naninf"] = old_flag

    def test_unbatched_p2p_ops_stage_order_and_helper_waits(self):
        records = []

        def isend(tensor, rank, group):
            records.append(("send", rank))
            return Request(self.calls, "send")

        def irecv(tensor, rank, group):
            records.append(("recv", rank))
            return Request(self.calls, "recv")

        paddle.distributed.isend = isend
        paddle.distributed.irecv = irecv
        tensors = [self._tensor([2, 2]) for _ in range(4)]
        self.hcg.stage_id = 0
        self.assertEqual(len(p2p._p2p_ops(*tensors, self.hcg)), 4)
        self.assertEqual(records, [("send", 3), ("recv", 2), ("send", 2), ("recv", 3)])

        records.clear()
        self.hcg.stage_id = 1
        self.assertEqual(len(p2p._p2p_ops(*tensors, self.hcg)), 4)
        self.assertEqual(records, [("recv", 2), ("send", 3), ("recv", 3), ("send", 2)])

        meta = self._meta(tuple_recv=False, tuple_send=False)
        _, _, reqs = p2p._p2p_helper(
            None,
            None,
            recv_prev=False,
            recv_next=False,
            sync_recv=False,
            send_recv_meta=meta,
            batch_p2p_comm=False,
            wait_on_reqs=False,
        )
        self.assertEqual(reqs, [])

        self.calls.clear()
        _, _, reqs = p2p._p2p_helper(
            self._tensor([2, 2]),
            None,
            recv_prev=False,
            recv_next=False,
            sync_recv=False,
            send_recv_meta=meta,
            batch_p2p_comm=False,
            wait_on_reqs=True,
        )
        self.assertIsNone(reqs)
        self.assertIn(("request_wait", "send"), self.calls)


class TestP2pHelperPublicMethodsNoMock(P2PStateTest):
    def setUp(self):
        super().setUp()
        p2p._timers = Timers(self.calls)
        p2p._sync_send = False
        self.old_helper = p2p._p2p_helper

        def helper(
            tensor_send_next,
            tensor_send_prev,
            recv_prev,
            recv_next,
            sync_recv=True,
            send_recv_meta=None,
            batch_p2p_comm=True,
            wait_on_reqs=True,
            dynamic_shape=False,
        ):
            self.calls.append(
                (
                    "p2p_helper",
                    tensor_send_next is not None,
                    tensor_send_prev is not None,
                    recv_prev,
                    recv_next,
                    sync_recv,
                    batch_p2p_comm,
                    wait_on_reqs,
                    dynamic_shape,
                )
            )
            prev = paddle.ones([1], dtype="float32") if recv_prev else None
            nxt = paddle.ones([1], dtype="float32") if recv_next else None
            return prev, nxt, [Request(self.calls, "overlap")]

        p2p._p2p_helper = helper
        paddle.distributed.send = lambda tensor, dst, group: self.calls.append(("send", dst))
        self.recv_values = []

        def recv(tensor, src, group):
            values = self.recv_values.pop(0) if self.recv_values else [0]
            self._fill(tensor, values)

        paddle.distributed.recv = recv

    def tearDown(self):
        p2p._p2p_helper = self.old_helper
        super().tearDown()

    def _fill_static_meta(self, helper):
        helper._send_recv_meta.recv_shape_message = [1]
        helper._send_recv_meta.recv_dtype_message = paddle_2_number(paddle.float32)
        helper._send_recv_meta.recv_stop_gradient = False
        helper._send_recv_meta.recv_key_message = None
        helper._send_recv_meta.send_shape_message = [1]
        helper._send_recv_meta.send_dtype_message = paddle_2_number(paddle.float32)
        helper._send_recv_meta.send_key_message = None
        helper._send_recv_meta.has_recv_meta = True
        helper._send_recv_meta.has_send_meta = True

    def _prepared_helper(self):
        helper = p2p.P2pHelper(use_cache=True)
        self._fill_static_meta(helper)
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
        result, handles = helper.send_forward_recv_forward(tensor, recv_prev=True, overlap_p2p_comm=True)
        self.assertIsNotNone(result)
        self.assertEqual(handles[0].name, "overlap")
        result, handles = helper.send_backward_recv_backward(tensor, recv_next=True, overlap_p2p_comm=True)
        self.assertIsNotNone(result)
        self.assertEqual(handles[0].name, "overlap")
        self.assertIn("using cache", repr(helper))
        self.assertTrue(any(call[0] == "timer_start" for call in self.calls))

    def test_static_meta_cache_and_clear_paths(self):
        sent = []

        def send(tensor, dst, group):
            sent.append((tensor.numpy().tolist(), dst, group is self.hcg.pipe_group))

        paddle.distributed.send = send
        helper = p2p.P2pHelper(use_cache=True)
        tensor = self._tensor([2, 2])
        helper._send_meta(tensor)
        self.assertTrue(helper._send_recv_meta.has_send_meta)
        self.assertEqual(len(sent), 2)
        helper._send_meta(tensor)
        self.assertEqual(len(sent), 2)
        with self.assertRaisesRegex(AssertionError, "send_shape_message"):
            helper._send_meta(self._tensor([3, 2]))
        helper._send_meta(self._tensor([3, 2]), skip_check_meta=True)
        self.assertEqual(len(sent), 2)

        recv_payload = [0, 1, 2, paddle_2_number(paddle.float32), 1, 0]
        self.recv_values = [[len(recv_payload)], recv_payload]
        helper = p2p.P2pHelper(use_cache=True)
        helper._recv_meta()
        self.assertTrue(helper._send_recv_meta.has_recv_meta)
        self.assertEqual(helper._send_recv_meta.recv_shape_message, [2])
        recv_count = len([call for call in self.calls if call[0] == "recv"])
        helper._recv_meta()
        self.assertEqual(len([call for call in self.calls if call[0] == "recv"]), recv_count)
        self.assertFalse(helper.clear_meta_cache())
        self.assertIsNone(helper._send_recv_meta.recv_shape_message)
        self.assertFalse(helper._send_recv_meta.has_recv_meta)

    def test_dynamic_shape_meta_paths_and_assertions(self):
        helper = p2p.P2pHelper(use_cache=True, dynamic_shape=True)
        tensor = self._tensor([1])
        helper._send_meta(tensor)
        self.assertEqual(len(helper._send_recv_meta_list), 1)
        self.assertFalse(helper.clear_meta_cache())
        helper._dynamic_cnt = 0
        helper._send_meta(tensor)
        helper._dynamic_cnt = 0
        self.recv_values = [
            [6],
            [0, 1, 2, paddle_2_number(paddle.float32), 0, 0],
        ]
        helper._recv_meta()
        helper._send_recv_meta_list[0].has_recv_meta = False
        self.recv_values = [
            [6],
            [0, 1, 2, paddle_2_number(paddle.float32), 0, 0],
        ]
        helper._recv_meta(reverse=True)
        helper._send_recv_meta_list[0].has_recv_meta = True
        helper._recv_meta(reverse=True)

        self._fill_static_meta(helper)
        helper.recv_forward(pp_first_stage=False)
        self.assertEqual(helper._dynamic_cnt, 1)
        helper._dynamic_cnt = 0
        helper.recv_backward(pp_last_stage=False)
        self.assertEqual(helper._dynamic_cnt, 1)
        helper._dynamic_cnt = 0
        helper.send_forward(tensor, pp_last_stage=False)
        self.assertEqual(helper._dynamic_cnt, 1)
        helper._dynamic_cnt = 0
        helper.send_backward(tensor, pp_first_stage=False)
        self.assertEqual(helper._dynamic_cnt, 1)
        helper._dynamic_cnt = 0
        helper.send_forward_recv_forward(tensor, recv_prev=True)
        self.assertEqual(helper._dynamic_cnt, 1)
        helper._dynamic_cnt = 0
        helper.send_backward_recv_backward(tensor, recv_next=True)
        self.assertEqual(helper._dynamic_cnt, 1)

        with self.assertRaises(AssertionError):
            helper.send_forward_recv_backward(tensor, pp_last_stage=False)
        with self.assertRaises(AssertionError):
            helper.send_backward_recv_forward(tensor, pp_first_stage=False)
        with self.assertRaises(AssertionError):
            helper.send_forward_backward_recv_forward_backward(tensor, tensor, recv_prev=False, recv_next=False)


if __name__ == "__main__":
    unittest.main()
