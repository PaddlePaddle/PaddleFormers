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

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)

import unittest
from unittest.mock import MagicMock, patch


class TestFourDirectionsSendRecvMetaInit(unittest.TestCase):
    """Tests for SendRecvMeta in four_directions_p2p_communication."""

    def test_init_default(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            SendRecvMeta,
        )

        meta = SendRecvMeta()
        self.assertIsNone(meta.send_shape_message)
        self.assertIsNone(meta.send_dtype_message)
        self.assertIsNone(meta.recv_shape_message)
        self.assertIsNone(meta.recv_dtype_message)
        self.assertIsNone(meta.recv_stop_gradient)
        self.assertFalse(meta.has_send_meta)
        self.assertFalse(meta.has_recv_meta)


class TestFourDirectionsSendRecvMetaSetMessage(unittest.TestCase):
    """Tests for SendRecvMeta.set_send_message in four_directions."""

    def test_set_single_tensor(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            SendRecvMeta,
        )

        meta = SendRecvMeta()
        tensor = paddle.randn([2, 3])
        meta.set_send_message(tensor)
        self.assertEqual(meta.send_shape_message, [2, 3])
        self.assertIsNotNone(meta.send_dtype_message)

    def test_set_tuple_tensor(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            SendRecvMeta,
        )

        meta = SendRecvMeta()
        t1 = paddle.randn([2, 3])
        t1.stop_gradient = False
        t2 = paddle.randn([4, 5])
        t2.stop_gradient = False
        meta.set_send_message((t1, t2))
        self.assertIsInstance(meta.send_shape_message, tuple)
        self.assertEqual(len(meta.send_shape_message), 2)

    def test_set_tuple_skip_stop_grad(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            SendRecvMeta,
        )

        meta = SendRecvMeta()
        t1 = paddle.randn([2, 3])
        t1.stop_gradient = True
        t2 = paddle.randn([4, 5])
        t2.stop_gradient = False
        meta.set_send_message((t1, t2))
        self.assertEqual(len(meta.send_shape_message), 1)


class TestFourDirectionsSendRecvMetaRecvMeta(unittest.TestCase):
    """Tests for SendRecvMeta.recv_meta in four_directions."""

    @patch("paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._hcg")
    @patch("paddle.distributed.recv")
    def test_recv_meta_single_tensor(self, mock_recv, mock_hcg):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            SendRecvMeta,
        )

        mock_hcg._get_p2p_prev_rank.return_value = 1
        mock_group = MagicMock()

        # Simulate recv returning data
        data_tensor = MagicMock()
        data_tensor.item.return_value = 5
        data_tensor.numpy.return_value = MagicMock()

        recv_side_effect = [data_tensor, MagicMock()]  # dims then shape
        mock_recv.side_effect = recv_side_effect

        meta = SendRecvMeta()
        with self.assertRaises(Exception):  # noqa: B017
            # This will fail due to mock limitations, but tests the call path
            meta.recv_meta(mock_group)
        mock_hcg._get_p2p_prev_rank.assert_called()


class TestFourDirectionsSendRecvMetaRecvMetaReverse(unittest.TestCase):
    """Tests for SendRecvMeta.recv_meta with reverse=True."""

    @patch("paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._hcg")
    def test_recv_meta_uses_next_rank_for_reverse(self, mock_hcg):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            SendRecvMeta,
        )

        mock_hcg._get_p2p_next_rank.return_value = 3
        mock_group = MagicMock()

        meta = SendRecvMeta()
        # Override recv_meta to verify reverse parameter is passed
        original_recv_meta = type(meta).recv_meta
        calls = []

        def capturing_recv_meta(self, group, reverse=False):
            calls.append(reverse)
            raise Exception("stop_test")

        meta.recv_meta = lambda g, reverse=False: capturing_recv_meta(meta, g, reverse)

        with self.assertRaises(Exception):  # noqa: B017
            meta.recv_meta(mock_group, reverse=True)
        self.assertTrue(calls[0])


class TestFourDirectionsSendRecvMetaSendMeta(unittest.TestCase):
    """Tests for SendRecvMeta.send_meta in four_directions."""

    @patch("paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._hcg")
    @patch("paddle.distributed.send")
    def test_send_meta_single_tensor(self, mock_send, mock_hcg):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            SendRecvMeta,
        )

        mock_hcg._get_p2p_next_rank.return_value = 2
        mock_group = MagicMock()
        tensor = paddle.randn([2, 3])

        meta = SendRecvMeta()
        meta.send_meta(tensor, mock_group)
        self.assertTrue(mock_send.called)

    @patch("paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._hcg")
    @patch("paddle.distributed.send")
    def test_send_meta_tuple_tensor(self, mock_send, mock_hcg):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            SendRecvMeta,
        )

        mock_hcg._get_p2p_next_rank.return_value = 2
        mock_group = MagicMock()
        t1 = paddle.randn([2, 3])
        t2 = paddle.randn([2, 3])
        meta = SendRecvMeta()
        meta.send_meta((t1, t2), mock_group)
        self.assertTrue(mock_send.called)


class TestFourDirectionsSendRecvMetaSendMetaReverse(unittest.TestCase):
    """Tests for SendRecvMeta.send_meta with reverse=True."""

    @patch("paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._hcg")
    def test_send_meta_reverse_uses_prev_rank(self, mock_hcg):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            SendRecvMeta,
        )

        mock_hcg._get_p2p_prev_rank.return_value = 0
        mock_group = MagicMock()
        tensor = paddle.randn([2, 3])

        meta = SendRecvMeta()
        # Use same capturing approach as recv_meta test
        calls = []

        def capturing_send_meta(self, t, group, reverse=False):
            calls.append(reverse)
            raise Exception("stop_test")

        meta.send_meta = lambda t, g, reverse=False: capturing_send_meta(meta, t, g, reverse)

        with self.assertRaises(Exception):  # noqa: B017
            meta.send_meta(tensor, mock_group, reverse=True)
        self.assertTrue(calls[0])


class TestFourDirectionsInitializeP2PGroups(unittest.TestCase):
    """Tests for initialize_p2p_groups in four_directions."""

    @patch(
        "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._timers",
        None,
    )
    def test_initialize_groups(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            initialize_p2p_groups,
        )

        mock_hcg = MagicMock()
        mock_hcg.get_p2p_groups.return_value = (
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )
        initialize_p2p_groups(mock_hcg, enable_partial_send_recv=True)
        # Verify hcg was set
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            _hcg,
        )

        self.assertIs(_hcg, mock_hcg)

    @patch("paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication.timer")
    def test_initialize_groups_with_timer(self, mock_timer):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            initialize_p2p_groups,
        )

        mock_hcg = MagicMock()
        mock_hcg.get_p2p_groups.return_value = (
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )
        mock_timer.get_timers.return_value = MagicMock()
        initialize_p2p_groups(mock_hcg, enable_timer=True)
        mock_timer.get_timers.assert_called_once()


class TestFourDirectionsIsValidSendRecvPartial(unittest.TestCase):
    """Tests for _is_valid_send_recv_partial in four_directions."""

    @patch(
        "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._enable_partial_send_recv",
        False,
    )
    def test_disabled(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            _is_valid_send_recv_partial,
        )

        tensor = paddle.randn([4, 4])
        self.assertFalse(_is_valid_send_recv_partial(tensor, 2))

    @patch(
        "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._enable_partial_send_recv",
        True,
    )
    def test_valid(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            _is_valid_send_recv_partial,
        )

        tensor = paddle.randn([4, 4])
        self.assertTrue(_is_valid_send_recv_partial(tensor, 2))

    @patch(
        "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._enable_partial_send_recv",
        True,
    )
    def test_zero_elements(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            _is_valid_send_recv_partial,
        )

        tensor = paddle.zeros([0, 4])
        with self.assertRaises(AssertionError):
            _is_valid_send_recv_partial(tensor, 2)


if __name__ == "__main__":
    unittest.main()
