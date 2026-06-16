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

from paddleformers.fleet.pipeline_parallel.pp_utils import p2p_communication
from paddleformers.fleet.pipeline_parallel.pp_utils.p2p_communication import (
    SendRecvMeta,
)
from paddleformers.fleet.pipeline_parallel.pp_utils.utils import paddle_2_number


class Task:
    def __init__(self, name):
        self.name = name
        self.wait_called = False

    def wait(self):
        self.wait_called = True


class RecordingGroup:
    id = 13
    backend = "custom"

    def __init__(self):
        self.calls = []

    def is_member(self):
        return True

    def get_group_rank(self, rank):
        return rank + 10


class DummyHCG:
    def __init__(self):
        self.pipe_group = RecordingGroup()
        self.model_group = RecordingGroup()
        self.stage_id = 0

    def get_pipe_parallel_group(self):
        return self.pipe_group

    def get_model_parallel_group(self):
        return self.model_group

    def get_model_parallel_world_size(self):
        return 1

    def get_model_parallel_rank(self):
        return 0

    def get_stage_id(self):
        return self.stage_id

    def _get_p2p_prev_rank(self):
        return 2

    def _get_p2p_next_rank(self):
        return 3


class TestP2PHelperAllocationAndOps(unittest.TestCase):
    def setUp(self):
        self.old_hcg = p2p_communication._hcg
        self.old_batched = p2p_communication._batched_p2p_ops
        self.old_ops = p2p_communication._p2p_ops
        self.old_batch_send_recv = (
            p2p_communication.batch_send_recv_on_calc_stream
        )
        self.old_allgather_partial = p2p_communication.allgather_partial
        self.old_sync_send = p2p_communication._sync_send
        p2p_communication._hcg = DummyHCG()

    def tearDown(self):
        p2p_communication._hcg = self.old_hcg
        p2p_communication._batched_p2p_ops = self.old_batched
        p2p_communication._p2p_ops = self.old_ops
        p2p_communication.batch_send_recv_on_calc_stream = (
            self.old_batch_send_recv
        )
        p2p_communication.allgather_partial = self.old_allgather_partial
        p2p_communication._sync_send = self.old_sync_send

    def _meta(self):
        meta = SendRecvMeta()
        meta.recv_shape_message = ([1, 2], [2, 1])
        meta.recv_dtype_message = (
            paddle_2_number(paddle.float32),
            paddle_2_number(paddle.int64),
        )
        meta.recv_stop_gradient = (True, False)
        meta.recv_key_message = ("prev", None)
        meta.send_shape_message = [3]
        meta.send_dtype_message = paddle_2_number(paddle.float16)
        meta.send_key_message = None
        return meta

    def test_p2p_helper_allocates_tuple_prev_and_next_with_keys(self):
        recorded = []

        def fake_batched(send_prev, recv_prev, send_next, recv_next, hcg):
            recorded.append((send_prev, recv_prev, send_next, recv_next, hcg))

        p2p_communication._batched_p2p_ops = fake_batched
        meta = self._meta()

        recv_prev, recv_next, reqs = p2p_communication._p2p_helper(
            tensor_send_next=None,
            tensor_send_prev=None,
            recv_prev=True,
            recv_next=True,
            send_recv_meta=meta,
            batch_p2p_comm=True,
        )

        self.assertEqual(reqs, None)
        self.assertEqual(
            [tensor.shape for tensor in recv_prev], [[1, 2], [2, 1]]
        )
        self.assertEqual(recv_prev[0].dtype, paddle.float32)
        self.assertEqual(recv_prev[1].dtype, paddle.int64)
        self.assertTrue(recv_prev[0].stop_gradient)
        self.assertFalse(recv_prev[1].stop_gradient)
        self.assertEqual(recv_prev[0].key, "prev")
        self.assertEqual(recv_next.shape, [3])
        self.assertEqual(recv_next.dtype, paddle.float16)
        self.assertIs(recorded[0][1], recv_prev)
        self.assertIs(recorded[0][3], recv_next)

    def test_dynamic_shape_recv_next_reuses_recv_metadata(self):
        recorded = []

        def fake_batched(send_prev, recv_prev, send_next, recv_next, hcg):
            recorded.append((send_prev, recv_prev, send_next, recv_next, hcg))

        p2p_communication._batched_p2p_ops = fake_batched
        meta = self._meta()

        recv_prev, recv_next, _ = p2p_communication._p2p_helper(
            tensor_send_next=None,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=True,
            send_recv_meta=meta,
            batch_p2p_comm=True,
            dynamic_shape=True,
        )

        self.assertIsNone(recv_prev)
        self.assertEqual(
            [tensor.shape for tensor in recv_next], [[1, 2], [2, 1]]
        )
        self.assertIs(recorded[0][3], recv_next)

    def test_unbatched_helper_waits_or_returns_wait_handles(self):
        tasks = [Task("first"), Task("second")]

        def fake_ops(send_prev, recv_prev, send_next, recv_next, hcg):
            return tasks

        p2p_communication._p2p_ops = fake_ops
        meta = self._meta()

        _, _, reqs = p2p_communication._p2p_helper(
            None,
            None,
            False,
            False,
            send_recv_meta=meta,
            batch_p2p_comm=False,
            wait_on_reqs=True,
        )
        self.assertIsNone(reqs)
        self.assertTrue(tasks[0].wait_called)
        self.assertTrue(tasks[1].wait_called)

        tasks = [Task("third")]
        _, _, reqs = p2p_communication._p2p_helper(
            None,
            None,
            False,
            False,
            send_recv_meta=meta,
            batch_p2p_comm=False,
            wait_on_reqs=False,
        )
        self.assertIs(reqs, tasks)
        self.assertFalse(tasks[0].wait_called)

    def test_p2p_ops_even_and_odd_stage_order(self):
        calls = []

        def send(tensor, rank, group):
            calls.append(("send", tensor, rank, group))
            return Task("send")

        def recv(tensor, rank, group):
            calls.append(("recv", tensor, rank, group))
            return Task("recv")

        old_isend = paddle.distributed.isend
        old_irecv = paddle.distributed.irecv
        paddle.distributed.isend = send
        paddle.distributed.irecv = recv
        hcg = p2p_communication._hcg
        send_prev = paddle.ones([1], dtype="float32")
        recv_prev = paddle.ones([1], dtype="float32")
        send_next = paddle.ones([1], dtype="float32")
        recv_next = paddle.ones([1], dtype="float32")
        try:
            hcg.stage_id = 0
            even_reqs = p2p_communication._p2p_ops(
                send_prev, recv_prev, send_next, recv_next, hcg
            )
            self.assertEqual(
                [(name, rank) for name, _, rank, _ in calls],
                [("send", 3), ("recv", 2), ("send", 2), ("recv", 3)],
            )
            self.assertEqual(len(even_reqs), 4)

            calls.clear()
            hcg.stage_id = 1
            odd_reqs = p2p_communication._p2p_ops(
                send_prev, recv_prev, send_next, recv_next, hcg
            )
            self.assertEqual(
                [(name, rank) for name, _, rank, _ in calls],
                [("recv", 2), ("send", 3), ("recv", 3), ("send", 2)],
            )
            self.assertEqual(len(odd_reqs), 4)
        finally:
            paddle.distributed.isend = old_isend
            paddle.distributed.irecv = old_irecv

    def test_batched_p2p_ops_builds_sync_and_async_order(self):
        recorded = []

        def fake_batch(ops):
            recorded.append(
                [(op.op.__name__, op.peer, op.tensor) for op in ops]
            )

        p2p_communication.batch_send_recv_on_calc_stream = fake_batch
        p2p_communication.allgather_partial = (
            lambda *args, **kwargs: recorded.append(("allgather", args[0]))
        )
        hcg = p2p_communication._hcg
        tensors = [paddle.ones([1], dtype="float32") for _ in range(4)]

        p2p_communication._sync_send = False
        p2p_communication._batched_p2p_ops(*tensors, hcg)
        self.assertEqual(
            [(name, peer) for name, peer, _ in recorded[0]],
            [
                ("_send_on_calc_stream", 2),
                ("_recv_on_calc_stream", 2),
                ("_send_on_calc_stream", 3),
                ("_recv_on_calc_stream", 3),
            ],
        )
        self.assertEqual(
            len([item for item in recorded if item[0] == "allgather"]), 2
        )

        recorded.clear()
        p2p_communication._sync_send = True
        p2p_communication._batched_p2p_ops(*tensors, hcg)
        self.assertEqual(
            [(name, peer) for name, peer, _ in recorded[0]],
            [
                ("_recv_on_calc_stream", 2),
                ("_send_on_calc_stream", 3),
                ("_recv_on_calc_stream", 3),
                ("_send_on_calc_stream", 2),
            ],
        )


if __name__ == "__main__":
    unittest.main()
