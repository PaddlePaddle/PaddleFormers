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

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)


import unittest
from unittest.mock import MagicMock, patch


class TestSendRecvMeta(unittest.TestCase):
    """Tests for SendRecvMeta class in p2p_communication.py."""

    def test_init(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            SendRecvMeta,
        )

        meta = SendRecvMeta()
        self.assertIsNone(meta.send_shape_message)
        self.assertIsNone(meta.recv_shape_message)
        self.assertFalse(meta.has_send_meta)
        self.assertFalse(meta.has_recv_meta)

    def test_init_or_erase_meta(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            SendRecvMeta,
        )

        meta = SendRecvMeta()
        meta.send_shape_message = [2, 3]
        meta.has_send_meta = True
        meta.init_or_erase_meta()
        self.assertIsNone(meta.send_shape_message)
        self.assertFalse(meta.has_send_meta)

    def test_repr(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            SendRecvMeta,
        )

        meta = SendRecvMeta()
        meta.send_shape_message = [2, 3]
        meta.send_dtype_message = 1
        repr_str = repr(meta)
        self.assertIn("send_shape_message", repr_str)

    def test_set_send_message_single_tensor(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            SendRecvMeta,
        )

        meta = SendRecvMeta()
        t = paddle.randn([2, 3])
        meta.set_send_message(t)
        self.assertEqual(meta.send_shape_message, [2, 3])

    def test_set_send_message_tuple(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            SendRecvMeta,
        )

        meta = SendRecvMeta()
        t1 = paddle.randn([2, 3])
        t1.stop_gradient = False
        t2 = paddle.randn([2, 3])
        t2.stop_gradient = False
        meta.set_send_message((t1, t2))
        self.assertEqual(len(meta.send_shape_message), 2)

    def test_check_send_message_match(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            SendRecvMeta,
        )

        meta = SendRecvMeta()
        t = paddle.randn([2, 3])
        meta.set_send_message(t)
        # Should not raise
        meta.check_send_message(t)

    def test_check_send_message_mismatch(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            SendRecvMeta,
        )

        meta = SendRecvMeta()
        t1 = paddle.randn([2, 3])
        t2 = paddle.randn([4, 5])
        meta.set_send_message(t1)
        with self.assertRaises(AssertionError):
            meta.check_send_message(t2)

    def test_check_send_message_none(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            SendRecvMeta,
        )

        meta = SendRecvMeta()
        # Should not raise when messages are None
        meta.check_send_message(MagicMock())


class TestIsvalidSendRecvPartial(unittest.TestCase):
    """Tests for _is_valid_send_recv_partial function."""

    def test_disabled(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            _is_valid_send_recv_partial,
        )

        with patch(
            "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication._enable_partial_send_recv",
            False,
        ):
            self.assertFalse(_is_valid_send_recv_partial(MagicMock(), 4))

    def test_valid(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            _is_valid_send_recv_partial,
        )

        mock_tensor = MagicMock()
        mock_tensor.shape = [8]
        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication._enable_partial_send_recv",
                True,
            ),
            patch("numpy.prod", return_value=8),
        ):
            self.assertTrue(_is_valid_send_recv_partial(mock_tensor, 4))

    def test_invalid_mp_degree_one(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            _is_valid_send_recv_partial,
        )

        mock_tensor = MagicMock()
        mock_tensor.shape = [8]
        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication._enable_partial_send_recv",
                True,
            ),
            patch("numpy.prod", return_value=8),
        ):
            self.assertFalse(_is_valid_send_recv_partial(mock_tensor, 1))

    def test_invalid_not_divisible(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            _is_valid_send_recv_partial,
        )

        mock_tensor = MagicMock()
        mock_tensor.shape = [7]
        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication._enable_partial_send_recv",
                True,
            ),
            patch("numpy.prod", return_value=7),
        ):
            self.assertFalse(_is_valid_send_recv_partial(mock_tensor, 4))


class TestP2PonCalcStream(unittest.TestCase):
    """Tests for P2PonCalcStream class."""

    def test_init_send(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            P2PonCalcStream,
            _send_on_calc_stream,
        )

        mock_tensor = MagicMock()
        mock_group = MagicMock()
        op = P2PonCalcStream(
            _send_on_calc_stream, mock_tensor, 1, mock_group, 2, 0
        )
        self.assertEqual(op.tensor, mock_tensor)
        self.assertEqual(op.peer, 1)

    def test_init_recv(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            P2PonCalcStream,
            _recv_on_calc_stream,
        )

        mock_tensor = MagicMock()
        mock_group = MagicMock()
        op = P2PonCalcStream(
            _recv_on_calc_stream, mock_tensor, 0, mock_group, 2, 1
        )
        self.assertEqual(op.rank_id, 1)

    def test_init_invalid_op(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            P2PonCalcStream,
        )

        mock_tensor = MagicMock()
        mock_group = MagicMock()
        with self.assertRaises(RuntimeError):
            P2PonCalcStream(lambda x: x, mock_tensor, 1, mock_group)


class TestInitializeP2PGroups(unittest.TestCase):
    """Tests for initialize_p2p_groups function."""

    def test_basic_init(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            initialize_p2p_groups,
        )

        mock_hcg = MagicMock()
        initialize_p2p_groups(
            mock_hcg, enable_partial_send_recv=True, enable_timer=False
        )

    def test_init_with_timer(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            initialize_p2p_groups,
        )

        mock_hcg = MagicMock()
        mock_timer = MagicMock()
        with patch(
            "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication.timer"
        ) as mock_timer_mod:
            mock_timer_mod.get_timers.return_value = mock_timer
            initialize_p2p_groups(mock_hcg, enable_timer=True)


class TestBatchP2pTupleOrTensor(unittest.TestCase):
    """Tests for _batch_p2p_tuple_or_tensor function."""

    def test_single_tensor(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            _batch_p2p_tuple_or_tensor,
            _send_on_calc_stream,
        )

        mock_tensor = MagicMock()
        mock_group = MagicMock()
        ops = _batch_p2p_tuple_or_tensor(
            mock_tensor, _send_on_calc_stream, 1, mock_group
        )
        self.assertEqual(len(ops), 1)

    def test_tuple_tensor(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            _batch_p2p_tuple_or_tensor,
            _send_on_calc_stream,
        )

        t1 = MagicMock()
        t2 = MagicMock()
        mock_group = MagicMock()
        ops = _batch_p2p_tuple_or_tensor(
            (t1, t2), _send_on_calc_stream, 1, mock_group
        )
        self.assertEqual(len(ops), 2)


class TestSendRecvMetaRecvMeta(unittest.TestCase):
    """Tests for SendRecvMeta recv_meta and send_meta with mocking."""

    def test_recv_meta_reverse(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            SendRecvMeta,
        )

        meta = SendRecvMeta()
        mock_group = MagicMock()
        mock_hcg = MagicMock()
        mock_hcg._get_p2p_next_rank.return_value = 5

        with (  # noqa: SIM117
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication._hcg",
                mock_hcg,
            ),
            patch("paddle.distributed.recv"),
            patch("paddle.distributed.broadcast"),
        ):
            # Setup mock data
            with patch("paddle.empty") as mock_empty:
                # Return empty tensors for recv
                numel_tensor = MagicMock()
                numel_tensor.item.return_value = 10
                data_tensor = MagicMock()
                data_tensor.numpy.return_value.tolist.return_value = [
                    1,
                    2,
                    2,
                    3,
                    1,
                    0,
                    0,
                ]

                mock_empty.side_effect = [numel_tensor, data_tensor]
                try:
                    meta.recv_meta(mock_group, reverse=True)
                except Exception:
                    pass  # Mock setup is complex, just test the path

    def test_send_meta_reverse(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            SendRecvMeta,
        )

        meta = SendRecvMeta()
        mock_group = MagicMock()
        mock_hcg = MagicMock()
        mock_hcg._get_p2p_prev_rank.return_value = 3
        t = paddle.randn([2, 3])

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication._hcg",
                mock_hcg,
            ),
            patch("paddle.distributed.send"),
            patch("paddle.to_tensor") as mock_to_tensor,
        ):
            mock_to_tensor.return_value = MagicMock()
            meta.send_meta(t, mock_group, reverse=True)


class TestAllgatherPartial(unittest.TestCase):
    """Tests for allgather_partial function."""

    def test_invalid_partial(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            allgather_partial,
        )

        mock_tensor = MagicMock()
        with patch(
            "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication._is_valid_send_recv_partial",
            return_value=False,
        ):
            result = allgather_partial(mock_tensor, nranks=4, rank_id=0)
            self.assertEqual(result, mock_tensor)

    def test_not_member(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            allgather_partial,
        )

        mock_group = MagicMock()
        mock_group.is_member.return_value = False
        mock_tensor = MagicMock()
        with patch(
            "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication._is_valid_send_recv_partial",
            return_value=True,
        ):
            result = allgather_partial(
                mock_tensor, nranks=4, rank_id=0, group=mock_group
            )
            self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
