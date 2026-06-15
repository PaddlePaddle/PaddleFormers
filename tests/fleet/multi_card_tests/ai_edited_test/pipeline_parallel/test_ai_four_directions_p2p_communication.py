# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

os.environ["PADDLE_USE_FOUR_DIRECTIONS_P2P"] = "True"

import numpy as np
import paddle
from paddle.distributed import fleet

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)

from paddleformers.fleet.pipeline_parallel.pp_utils.four_directions_p2p_communication import (
    P2pHelper,
    SendRecvMeta,
    _is_valid_send_recv_partial,
    _p2p_helper,
    _xpu_comm_group_end,
    _xpu_comm_group_start,
    allgather_partial,
    initialize_p2p_groups,
    recv_partial,
    send_partial,
)
from paddleformers.fleet.pipeline_parallel.pp_utils.utils import paddle_2_number
from paddleformers.fleet.training.initialize import initialize_fleet

PP_DEGREE = 2


def _init_pp():
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": 1,
        "pp_degree": PP_DEGREE,
        "sharding_degree": 1,
        "sep_degree": 1,
        "cp_degree": 1,
        "ep_degree": 1,
        "moe_sharding_degree": 1,
        "order": [
            "sharding",
            "moe_sharding",
            "pp",
            "sep",
            "cp",
            "dp",
            "ep",
            "mp",
        ],
    }
    initialize_fleet(strategy)


def setUpModule():
    _init_pp()
    hcg = fleet.get_hybrid_communicate_group()
    initialize_p2p_groups(hcg, enable_partial_send_recv=True, enable_timer=False)
    np.random.seed(42)
    paddle.seed(42)


class TestXpuCommGroupFunctions(unittest.TestCase):
    """Test XPU communication group start/end functions."""

    def test_xpu_comm_group_start_end_non_xpu(self):
        """Test _xpu_comm_group_start/end on non-XPU environment."""
        _xpu_comm_group_start()
        _xpu_comm_group_end()
        _xpu_comm_group_end()

    def test_xpu_comm_group_multiple_calls(self):
        """Test multiple calls to _xpu_comm_group_end should not raise."""
        _xpu_comm_group_end()
        _xpu_comm_group_end()


class TestSendRecvMetaInit(unittest.TestCase):
    """Test SendRecvMeta initialization."""

    def test_send_recv_meta_init(self):
        """Verify SendRecvMeta initializes with None and False values."""
        meta = SendRecvMeta()
        self.assertIsNone(meta.send_shape_message)
        self.assertIsNone(meta.send_dtype_message)
        self.assertIsNone(meta.recv_shape_message)
        self.assertIsNone(meta.recv_dtype_message)
        self.assertIsNone(meta.recv_stop_gradient)
        self.assertFalse(meta.has_send_meta)
        self.assertFalse(meta.has_recv_meta)


class TestSendRecvMetaSetSendMessage(unittest.TestCase):
    """Test SendRecvMeta.set_send_message extracts shape/dtype from tensor."""

    def test_set_send_message_single_tensor(self):
        """Verify set_send_message correctly populates shape and dtype."""
        meta = SendRecvMeta()
        tensor = paddle.randn([3, 6, 9], dtype="float32")
        meta.set_send_message(tensor)
        self.assertEqual(meta.send_shape_message, [3, 6, 9])
        self.assertEqual(meta.send_dtype_message, paddle_2_number(paddle.float32))

    def test_set_send_message_tuple_tensor(self):
        """Verify set_send_message works with tuple input."""
        meta = SendRecvMeta()
        t1 = paddle.randn([2, 4], dtype="float32")
        t1.stop_gradient = False
        t2 = paddle.randn([2, 4], dtype="float16")
        t2.stop_gradient = False
        t3 = paddle.randn([2, 4], dtype="float32")
        t3.stop_gradient = True
        meta.set_send_message((t1, t2, t3))
        self.assertIsInstance(meta.send_shape_message, tuple)
        self.assertEqual(len(meta.send_shape_message), 2)
        self.assertIsInstance(meta.send_dtype_message, tuple)
        self.assertEqual(len(meta.send_dtype_message), 2)


class TestIsValidSendRecvPartial(unittest.TestCase):
    """Test _is_valid_send_recv_partial with various tensor shapes and mp_degree."""

    def test_is_valid_send_recv_partial_not_divisible(self):
        """Should return False when tensor numel is not divisible by mp_degree."""
        tensor = paddle.randn([3, 7], dtype="float32")
        self.assertFalse(_is_valid_send_recv_partial(tensor, 4))

    def test_is_valid_send_recv_partial_mp_degree_one(self):
        """Should return False when mp_degree is 1."""
        tensor = paddle.randn([4, 8], dtype="float32")
        self.assertFalse(_is_valid_send_recv_partial(tensor, 1))

    def test_is_valid_send_recv_partial_divisible(self):
        """Should return True when tensor numel is divisible by mp_degree > 1."""
        import paddleformers.fleet.pipeline_parallel.pp_utils.four_directions_p2p_communication as p2p_mod

        p2p_mod._enable_partial_send_recv = True
        tensor = paddle.randn([4, 8], dtype="float32")
        self.assertTrue(_is_valid_send_recv_partial(tensor, 2))

    def test_is_valid_send_recv_partial_zero_elements(self):
        """Should raise assertion error for zero element tensor."""
        import paddleformers.fleet.pipeline_parallel.pp_utils.four_directions_p2p_communication as p2p_mod

        p2p_mod._enable_partial_send_recv = True
        tensor = paddle.randn([0], dtype="float32")
        with self.assertRaises(AssertionError):
            _is_valid_send_recv_partial(tensor, 2)

    def test_is_valid_send_recv_partial_disabled(self):
        """Should return False when partial send/recv is disabled."""
        import paddleformers.fleet.pipeline_parallel.pp_utils.four_directions_p2p_communication as p2p_mod

        p2p_mod._enable_partial_send_recv = False
        tensor = paddle.randn([4, 8], dtype="float32")
        self.assertFalse(_is_valid_send_recv_partial(tensor, 2))
        p2p_mod._enable_partial_send_recv = True


class TestAllgatherPartial(unittest.TestCase):
    """Test allgather_partial function."""

    def test_allgather_partial_no_op(self):
        """allgather_partial returns tensor unchanged when not divisible by nranks."""
        tensor = paddle.randn([3, 7], dtype="float32")
        result = allgather_partial(tensor, nranks=4, rank_id=0)
        self.assertIs(result, tensor)

    def test_allgather_partial_mp_degree_one(self):
        """allgather_partial returns tensor unchanged when nranks=1."""
        tensor = paddle.randn([4, 8], dtype="float32")
        result = allgather_partial(tensor, nranks=1, rank_id=0)
        self.assertIs(result, tensor)


class TestP2pHelperInit(unittest.TestCase):
    """Test P2pHelper initialization."""

    def test_p2p_helper_init_default(self):
        """P2pHelper should initialize with default use_cache=True."""
        helper = P2pHelper()
        self.assertTrue(helper._use_cache)
        self.assertIsInstance(helper._send_recv_meta, SendRecvMeta)

    def test_p2p_helper_init_no_cache(self):
        """P2pHelper should respect use_cache=False parameter."""
        helper = P2pHelper(use_cache=False)
        self.assertFalse(helper._use_cache)


class TestInitializeP2PGroups(unittest.TestCase):
    """Test initialize_p2p_groups with real PP group."""

    def test_initialize_p2p_groups_basic(self):
        """initialize_p2p_groups should succeed with a valid HCG."""
        hcg = fleet.get_hybrid_communicate_group()
        initialize_p2p_groups(hcg, enable_partial_send_recv=True)


class TestP2pHelperStageMethods(unittest.TestCase):
    """Test P2pHelper methods that return early for certain stages."""

    def test_send_forward_last_stage(self):
        """send_forward should do nothing for last stage."""
        helper = P2pHelper()
        tensor = paddle.randn([2, 4], dtype="float32")
        helper.send_forward(tensor, pp_last_stage=True)

    def test_recv_forward_first_stage(self):
        """recv_forward should return None for first stage."""
        helper = P2pHelper()
        result = helper.recv_forward(pp_first_stage=True, sync_recv=True)
        self.assertIsNone(result)

    def test_recv_backward_last_stage(self):
        """recv_backward should return None for last stage."""
        helper = P2pHelper()
        result = helper.recv_backward(pp_last_stage=True, sync_recv=True)
        self.assertIsNone(result)

    def test_send_backward_first_stage(self):
        """send_backward should do nothing for first stage."""
        helper = P2pHelper()
        tensor = paddle.randn([2, 4], dtype="float32")
        helper.send_backward(tensor, pp_first_stage=True)

    def test_send_forward_recv_backward_last_stage(self):
        """send_forward_recv_backward should return None for last stage."""
        helper = P2pHelper()
        tensor = paddle.randn([2, 4], dtype="float32")
        result = helper.send_forward_recv_backward(tensor, pp_last_stage=True)
        self.assertIsNone(result)

    def test_send_backward_recv_forward_first_stage(self):
        """send_backward_recv_forward should return None for first stage."""
        helper = P2pHelper()
        tensor = paddle.randn([2, 4], dtype="float32")
        result = helper.send_backward_recv_forward(tensor, pp_first_stage=True)
        self.assertIsNone(result)


class TestSendRecvMetaCommunication(unittest.TestCase):
    """Test SendRecvMeta send_meta and recv_meta with coordinated communication."""

    def test_send_recv_meta_single_tensor(self):
        """Test send_meta and recv_meta with single tensor.

        Key: Rank 0 sends, Rank 1 receives - this avoids deadlock.
        """
        hcg = fleet.get_hybrid_communicate_group()
        pp_rank = hcg.get_stage_id()
        pp_group = hcg.get_pipe_parallel_group()

        meta = SendRecvMeta()
        tensor = paddle.randn([2, 4], dtype="float32")

        if pp_rank == 0:
            meta.send_meta(tensor, pp_group)
        else:
            meta.recv_meta(pp_group)
            self.assertEqual(meta.recv_shape_message, [2, 4])
            self.assertEqual(meta.recv_dtype_message, paddle_2_number(paddle.float32))

    def test_send_recv_meta_tuple_tensor(self):
        """Test send_meta and recv_meta with tuple of tensors."""
        hcg = fleet.get_hybrid_communicate_group()
        pp_rank = hcg.get_stage_id()
        pp_group = hcg.get_pipe_parallel_group()

        meta = SendRecvMeta()
        t1 = paddle.randn([2, 4], dtype="float32")
        t2 = paddle.randn([2, 4], dtype="float16")

        if pp_rank == 0:
            meta.send_meta((t1, t2), pp_group)
        else:
            meta.recv_meta(pp_group)
            self.assertIsInstance(meta.recv_shape_message, tuple)
            self.assertEqual(len(meta.recv_shape_message), 2)


class TestSendPartialRecvPartial(unittest.TestCase):
    """Test send_partial and recv_partial functions."""

    def test_send_partial_non_member_group(self):
        """send_partial should return early if group is not a member."""

        class NonMemberGroup:
            def is_member(self):
                return False

        mock_group = NonMemberGroup()
        tensor = paddle.randn([2, 4], dtype="float32")
        result = send_partial(tensor, dst=0, nranks=1, group=mock_group)
        self.assertIsNone(result)

    def test_recv_partial_non_member_group(self):
        """recv_partial should return early if group is not a member."""

        class NonMemberGroup:
            def is_member(self):
                return False

        mock_group = NonMemberGroup()
        tensor = paddle.randn([2, 4], dtype="float32")
        result = recv_partial(tensor, src=0, nranks=1, group=mock_group)
        self.assertIsNone(result)


class TestAllgatherPartialGroup(unittest.TestCase):
    """Test allgather_partial with group membership."""

    def test_allgather_partial_non_member_group(self):
        """allgather_partial should return early if group is not a member."""

        class NonMemberGroup:
            def is_member(self):
                return False

        mock_group = NonMemberGroup()
        tensor = paddle.randn([4, 8], dtype="float32")
        result = allgather_partial(tensor, nranks=2, rank_id=0, group=mock_group)
        self.assertIsNone(result)


class TestDistributedCommunication(unittest.TestCase):
    """Test distributed P2P communication with asymmetric operations.

    Key insight: One rank sends, the other receives - this avoids deadlocks.
    """

    def test_send_recv_asymmetric(self):
        """Test send_partial and recv_partial with asymmetric operations.

        Rank 0 sends, Rank 1 receives - avoids deadlock.
        """
        hcg = fleet.get_hybrid_communicate_group()
        pp_rank = hcg.get_stage_id()

        tensor = paddle.randn([2, 4], dtype="float32")

        if pp_rank == 0:
            # Rank 0 sends to rank 1
            send_partial(tensor, dst=1, nranks=1, use_calc_stream=False)
        else:
            # Rank 1 receives from rank 0
            recv_tensor = paddle.empty([2, 4], dtype="float32")
            recv_partial(recv_tensor, src=0, nranks=1, use_calc_stream=True)
            # Verify received data
            self.assertEqual(recv_tensor.shape, [2, 4])

        paddle.distributed.barrier()

    def test_send_recv_forward_pattern(self):
        """Test forward pass P2P pattern: Rank 0 sends forward, Rank 1 receives.

        This mimics the actual pipeline parallel forward pass.
        """
        hcg = fleet.get_hybrid_communicate_group()
        pp_rank = hcg.get_stage_id()

        tensor = paddle.randn([2, 4], dtype="float32")

        if pp_rank == 0:
            # Rank 0 (first stage) sends forward to rank 1
            # Using send_partial with dst=1 (next rank)
            send_partial(tensor, dst=1, nranks=1, use_calc_stream=False)
        else:
            # Rank 1 receives from rank 0
            recv_tensor = paddle.empty([2, 4], dtype="float32")
            recv_partial(recv_tensor, src=0, nranks=1, use_calc_stream=True)
            self.assertEqual(recv_tensor.shape, [2, 4])

        paddle.distributed.barrier()

    def test_send_recv_backward_pattern(self):
        """Test backward pass P2P pattern: Rank 1 sends backward, Rank 0 receives.

        This mimics the actual pipeline parallel backward pass.
        """
        hcg = fleet.get_hybrid_communicate_group()
        pp_rank = hcg.get_stage_id()

        tensor = paddle.randn([2, 4], dtype="float32")

        if pp_rank == 1:
            # Rank 1 (last stage) sends backward to rank 0
            # Using send_partial with dst=0 (prev rank)
            send_partial(tensor, dst=0, nranks=1, use_calc_stream=False)
        else:
            # Rank 0 receives from rank 1
            recv_tensor = paddle.empty([2, 4], dtype="float32")
            recv_partial(recv_tensor, src=1, nranks=1, use_calc_stream=True)
            self.assertEqual(recv_tensor.shape, [2, 4])

        paddle.distributed.barrier()


class TestP2pHelperWithCoordination(unittest.TestCase):
    """Test _p2p_helper function with properly coordinated communication."""

    def test_p2p_helper_send_next_only(self):
        """Test _p2p_helper with only send_next operation.

        Rank 0 sends to rank 1, rank 1 does nothing.
        """
        hcg = fleet.get_hybrid_communicate_group()
        pp_rank = hcg.get_stage_id()

        meta = SendRecvMeta()
        meta.send_shape_message = [2, 4]
        meta.send_dtype_message = paddle_2_number(paddle.float32)
        meta.recv_shape_message = [2, 4]
        meta.recv_dtype_message = paddle_2_number(paddle.float32)
        meta.recv_stop_gradient = False

        if pp_rank == 0:
            tensor = paddle.randn([2, 4], dtype="float32")
            recv_prev, recv_next = _p2p_helper(
                tensor_send_next=tensor,
                tensor_send_prev=None,
                recv_prev=False,
                recv_next=False,
                sync_recv=True,
                send_recv_meta=meta,
            )
            self.assertIsNone(recv_prev)
            self.assertIsNone(recv_next)
        else:
            # Rank 1 receives
            recv_prev, recv_next = _p2p_helper(
                tensor_send_next=None,
                tensor_send_prev=None,
                recv_prev=True,
                recv_next=False,
                sync_recv=True,
                send_recv_meta=meta,
            )
            self.assertIsNotNone(recv_prev)
            self.assertEqual(recv_prev.shape, [2, 4])
            self.assertIsNone(recv_next)

        paddle.distributed.barrier()

    def test_p2p_helper_send_prev_only(self):
        """Test _p2p_helper with only send_prev operation.

        Rank 1 sends to rank 0, rank 0 receives.
        """
        hcg = fleet.get_hybrid_communicate_group()
        pp_rank = hcg.get_stage_id()

        meta = SendRecvMeta()
        meta.send_shape_message = [2, 4]
        meta.send_dtype_message = paddle_2_number(paddle.float32)
        meta.recv_shape_message = [2, 4]
        meta.recv_dtype_message = paddle_2_number(paddle.float32)
        meta.recv_stop_gradient = False

        if pp_rank == 1:
            tensor = paddle.randn([2, 4], dtype="float32")
            recv_prev, recv_next = _p2p_helper(
                tensor_send_next=None,
                tensor_send_prev=tensor,
                recv_prev=False,
                recv_next=False,
                sync_recv=True,
                send_recv_meta=meta,
            )
            self.assertIsNone(recv_prev)
            self.assertIsNone(recv_next)
        else:
            # Rank 0 receives
            recv_prev, recv_next = _p2p_helper(
                tensor_send_next=None,
                tensor_send_prev=None,
                recv_prev=False,
                recv_next=True,
                sync_recv=True,
                send_recv_meta=meta,
            )
            self.assertIsNone(recv_prev)
            self.assertIsNotNone(recv_next)
            self.assertEqual(recv_next.shape, [2, 4])

        paddle.distributed.barrier()

    def test_p2p_helper_tuple_tensors(self):
        """Test _p2p_helper with tuple of tensors."""
        hcg = fleet.get_hybrid_communicate_group()
        pp_rank = hcg.get_stage_id()

        meta = SendRecvMeta()
        meta.send_shape_message = ([2, 4], [3, 5])
        meta.send_dtype_message = (
            paddle_2_number(paddle.float32),
            paddle_2_number(paddle.float32),
        )
        meta.recv_shape_message = ([2, 4], [3, 5])
        meta.recv_dtype_message = (
            paddle_2_number(paddle.float32),
            paddle_2_number(paddle.float32),
        )
        meta.recv_stop_gradient = (False, False)

        if pp_rank == 0:
            t1 = paddle.randn([2, 4], dtype="float32")
            t2 = paddle.randn([3, 5], dtype="float32")
            recv_prev, recv_next = _p2p_helper(
                tensor_send_next=(t1, t2),
                tensor_send_prev=None,
                recv_prev=False,
                recv_next=False,
                sync_recv=True,
                send_recv_meta=meta,
            )
        else:
            recv_prev, recv_next = _p2p_helper(
                tensor_send_next=None,
                tensor_send_prev=None,
                recv_prev=True,
                recv_next=False,
                sync_recv=True,
                send_recv_meta=meta,
            )
            self.assertIsNotNone(recv_prev)
            self.assertIsInstance(recv_prev, tuple)
            self.assertEqual(len(recv_prev), 2)

        paddle.distributed.barrier()

    def test_p2p_helper_sync_recv_false(self):
        """Test _p2p_helper with sync_recv=False (non-blocking receive)."""
        hcg = fleet.get_hybrid_communicate_group()
        pp_rank = hcg.get_stage_id()

        meta = SendRecvMeta()
        meta.send_shape_message = [2, 4]
        meta.send_dtype_message = paddle_2_number(paddle.float32)
        meta.recv_shape_message = [2, 4]
        meta.recv_dtype_message = paddle_2_number(paddle.float32)
        meta.recv_stop_gradient = False

        if pp_rank == 0:
            tensor = paddle.randn([2, 4], dtype="float32")
            recv_prev, recv_next = _p2p_helper(
                tensor_send_next=tensor,
                tensor_send_prev=None,
                recv_prev=False,
                recv_next=False,
                sync_recv=False,
                send_recv_meta=meta,
            )
        else:
            recv_prev, recv_next = _p2p_helper(
                tensor_send_next=None,
                tensor_send_prev=None,
                recv_prev=True,
                recv_next=False,
                sync_recv=False,
                send_recv_meta=meta,
            )
            self.assertIsNotNone(recv_prev)

        paddle.distributed.barrier()

    def test_p2p_helper_recv_tuple_with_stop_gradient(self):
        """Test _p2p_helper receiving tuple with stop_gradient info."""
        hcg = fleet.get_hybrid_communicate_group()
        pp_rank = hcg.get_stage_id()

        meta = SendRecvMeta()
        meta.send_shape_message = ([2, 4], [3, 5])
        meta.send_dtype_message = (
            paddle_2_number(paddle.float32),
            paddle_2_number(paddle.float16),
        )
        meta.recv_shape_message = ([2, 4], [3, 5])
        meta.recv_dtype_message = (
            paddle_2_number(paddle.float32),
            paddle_2_number(paddle.float16),
        )
        meta.recv_stop_gradient = (True, False)  # One with stop_gradient=True

        if pp_rank == 0:
            t1 = paddle.randn([2, 4], dtype="float32")
            t2 = paddle.randn([3, 5], dtype="float16")
            recv_prev, recv_next = _p2p_helper(
                tensor_send_next=(t1, t2),
                tensor_send_prev=None,
                recv_prev=False,
                recv_next=False,
                sync_recv=True,
                send_recv_meta=meta,
            )
        else:
            recv_prev, recv_next = _p2p_helper(
                tensor_send_next=None,
                tensor_send_prev=None,
                recv_prev=True,
                recv_next=False,
                sync_recv=True,
                send_recv_meta=meta,
            )
            self.assertIsNotNone(recv_prev)
            self.assertIsInstance(recv_prev, tuple)
            # Verify stop_gradient is correctly set
            self.assertTrue(recv_prev[0].stop_gradient)
            self.assertFalse(recv_prev[1].stop_gradient)

        paddle.distributed.barrier()

    def test_p2p_helper_recv_next_tuple_tensors(self):
        """Test _p2p_helper with recv_next=True and tuple send_shape_message."""
        hcg = fleet.get_hybrid_communicate_group()
        pp_rank = hcg.get_stage_id()

        meta = SendRecvMeta()
        meta.send_shape_message = ([2, 4], [3, 5])
        meta.send_dtype_message = (
            paddle_2_number(paddle.float32),
            paddle_2_number(paddle.float16),
        )
        meta.recv_shape_message = ([2, 4], [3, 5])
        meta.recv_dtype_message = (
            paddle_2_number(paddle.float32),
            paddle_2_number(paddle.float16),
        )
        meta.recv_stop_gradient = (False, False)

        if pp_rank == 1:
            t1 = paddle.randn([2, 4], dtype="float32")
            t2 = paddle.randn([3, 5], dtype="float16")
            recv_prev, recv_next = _p2p_helper(
                tensor_send_next=None,
                tensor_send_prev=(t1, t2),
                recv_prev=False,
                recv_next=False,
                sync_recv=True,
                send_recv_meta=meta,
            )
        else:
            # Rank 0 receives next from rank 1
            recv_prev, recv_next = _p2p_helper(
                tensor_send_next=None,
                tensor_send_prev=None,
                recv_prev=False,
                recv_next=True,
                sync_recv=True,
                send_recv_meta=meta,
            )
            self.assertIsNone(recv_prev)
            self.assertIsNotNone(recv_next)
            self.assertIsInstance(recv_next, tuple)
            self.assertEqual(len(recv_next), 2)

        paddle.distributed.barrier()


if __name__ == "__main__":
    unittest.main()
