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
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

import unittest
from unittest.mock import MagicMock, patch

import paddle


class TestP2pCommInitializeP2PGroups(unittest.TestCase):
    """Tests for initialize_p2p_groups in p2p_communication."""

    def test_initialize_groups(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            initialize_p2p_groups,
        )

        mock_hcg = MagicMock()
        initialize_p2p_groups(
            mock_hcg, enable_partial_send_recv=True, enable_timer=False
        )

    def test_initialize_groups_with_timer(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            initialize_p2p_groups,
        )

        mock_hcg = MagicMock()
        with patch(
            "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication.timer"
        ) as mock_timer_mod:
            mock_timer_mod.get_timers.return_value = MagicMock()
            initialize_p2p_groups(mock_hcg, enable_timer=True)
            mock_timer_mod.get_timers.assert_called_once()


class TestP2pCommSendRecvMetaKeyAttribute(unittest.TestCase):
    """Tests for SendRecvMeta with key attributes in p2p_communication."""

    def test_obtain_send_message_with_key(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            SendRecvMeta,
        )

        meta = SendRecvMeta()
        tensor = paddle.randn([2, 3])
        tensor.key = "test_key"
        shape, dtype, key = meta._obtain_send_message(tensor)
        self.assertEqual(key, "test_key")

    def test_obtain_send_message_tuple_with_key(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            SendRecvMeta,
        )

        meta = SendRecvMeta()
        t1 = paddle.randn([2, 3])
        t1.stop_gradient = False
        t1.key = "key1"
        t2 = paddle.randn([3, 4])
        t2.stop_gradient = False
        t2.key = "key2"
        shapes, dtypes, keys = meta._obtain_send_message((t1, t2))
        self.assertEqual(keys, ("key1", "key2"))

    def test_send_meta_with_key(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            SendRecvMeta,
        )

        meta = SendRecvMeta()
        mock_group = MagicMock()
        mock_hcg = MagicMock()
        mock_hcg._get_p2p_next_rank.return_value = 2

        tensor = paddle.randn([2, 3])
        tensor.key = "test_key"

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication._hcg",
                mock_hcg,
            ),
            patch("paddle.distributed.send"),
            patch("paddle.to_tensor") as mock_to_tensor,
        ):
            mock_to_tensor.return_value = MagicMock()
            meta.send_meta(tensor, mock_group)


class TestP2pCommSendOnCalcStream(unittest.TestCase):
    """Tests for _send_on_calc_stream."""

    def test_send_partial_on_calc_stream(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            _send_on_calc_stream,
        )

        mock_tensor = MagicMock()
        mock_group = MagicMock()
        mock_group.get_group_rank.return_value = 1

        with patch(
            "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication._is_valid_send_recv_partial",
            return_value=True,
        ):
            _send_on_calc_stream(
                mock_tensor, mock_group, dst=3, nranks=2, rank_id=0
            )
            mock_group.process_group.send_partial_on_calc_stream.assert_called_once()

    def test_send_full_on_calc_stream(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            _send_on_calc_stream,
        )

        mock_tensor = MagicMock()
        mock_group = MagicMock()
        mock_group.get_group_rank.return_value = 1

        with patch(
            "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication._is_valid_send_recv_partial",
            return_value=False,
        ):
            _send_on_calc_stream(
                mock_tensor, mock_group, dst=3, nranks=1, rank_id=0
            )
            mock_group.process_group.send_on_calc_stream.assert_called_once()

    def test_send_group_none_raises(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            _send_on_calc_stream,
        )

        mock_tensor = MagicMock()
        with self.assertRaises(AssertionError):
            _send_on_calc_stream(mock_tensor, None, dst=3, nranks=1, rank_id=0)


class TestP2pCommRecvOnCalcStream(unittest.TestCase):
    """Tests for _recv_on_calc_stream."""

    def test_recv_partial_on_calc_stream(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            _recv_on_calc_stream,
        )

        mock_tensor = MagicMock()
        mock_group = MagicMock()
        mock_group.get_group_rank.return_value = 0

        with patch(
            "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication._is_valid_send_recv_partial",
            return_value=True,
        ):
            _recv_on_calc_stream(
                mock_tensor, mock_group, src=2, nranks=2, rank_id=0
            )
            mock_group.process_group.recv_partial_on_calc_stream.assert_called_once()

    def test_recv_full_on_calc_stream(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            _recv_on_calc_stream,
        )

        mock_tensor = MagicMock()
        mock_group = MagicMock()
        mock_group.get_group_rank.return_value = 0

        with patch(
            "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication._is_valid_send_recv_partial",
            return_value=False,
        ):
            _recv_on_calc_stream(
                mock_tensor, mock_group, src=2, nranks=1, rank_id=0
            )
            mock_group.process_group.recv_on_calc_stream.assert_called_once()

    def test_recv_group_none_raises(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            _recv_on_calc_stream,
        )

        mock_tensor = MagicMock()
        with self.assertRaises(AssertionError):
            _recv_on_calc_stream(mock_tensor, None, src=2, nranks=1, rank_id=0)


class TestP2pCommAllgatherPartialValid(unittest.TestCase):
    """Tests for allgather_partial with valid partial in p2p_communication."""

    def test_allgather_partial_valid(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            allgather_partial,
        )

        mock_tensor = MagicMock()
        mock_group = MagicMock()
        mock_group.is_member.return_value = True

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication._is_valid_send_recv_partial",
                return_value=True,
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication._partial_allgather_op",
                return_value=MagicMock(),
            ),
        ):
            result = allgather_partial(
                mock_tensor,
                nranks=2,
                rank_id=0,
                group=mock_group,
                use_calc_stream=True,
            )


if __name__ == "__main__":
    unittest.main()
