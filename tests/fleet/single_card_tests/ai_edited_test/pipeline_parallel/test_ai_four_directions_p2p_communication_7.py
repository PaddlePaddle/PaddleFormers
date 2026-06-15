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


class TestFourDirsSendRecvMetaKeyMessage(unittest.TestCase):
    """Tests for SendRecvMeta with key messages in four_directions."""

    def test_set_send_message_with_key(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            SendRecvMeta,
        )

        meta = SendRecvMeta()
        tensor = paddle.randn([2, 3])
        tensor.key = "hidden_states"
        meta.set_send_message(tensor)
        self.assertEqual(meta.send_shape_message, [2, 3])
        # key is not stored in set_send_message for four_directions version

    def test_set_send_message_tuple_with_stop_gradient(self):
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
        # Only t2 should be included
        self.assertEqual(len(meta.send_shape_message), 1)
        self.assertEqual(list(meta.send_shape_message[0]), [4, 5])


class TestFourDirsPartialSendRecvWithGroup(unittest.TestCase):
    """Tests for send_partial and recv_partial with group parameter variations."""

    def test_send_partial_group_none(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            send_partial,
        )

        mock_hcg = MagicMock()
        mock_hcg._get_p2p_next_rank.return_value = 5
        mock_tensor = MagicMock()

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._hcg",
                mock_hcg,
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._is_valid_send_recv_partial",
                return_value=False,
            ),
            patch("paddle.distributed.isend") as mock_isend,
        ):
            send_partial(mock_tensor, dst=1, nranks=1, rank_id=0, group=None)
            mock_isend.assert_called_once()

    def test_recv_partial_group_none(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            recv_partial,
        )

        mock_hcg = MagicMock()
        mock_hcg._get_p2p_prev_rank.return_value = 3
        mock_tensor = MagicMock()

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._hcg",
                mock_hcg,
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._is_valid_send_recv_partial",
                return_value=False,
            ),
            patch("paddle.distributed.recv") as mock_recv,
        ):
            recv_partial(mock_tensor, src=0, nranks=1, rank_id=0, group=None)
            mock_recv.assert_called_once()

    def test_allgather_partial_group_none_valid(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            allgather_partial,
        )

        mock_tensor = MagicMock()
        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._is_valid_send_recv_partial",
                return_value=True,
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._partial_allgather_op",
                return_value=MagicMock(),
            ),
            patch("paddle.distributed.collective._get_default_group") as mock_default,
        ):
            mock_default.return_value = MagicMock()
            allgather_partial(
                mock_tensor,
                nranks=2,
                rank_id=0,
                group=None,
                use_calc_stream=True,
            )


class TestFourDirsXpuCommGroup(unittest.TestCase):
    """Tests for XPU comm group management."""

    def test_xpu_comm_group_start_not_xpu(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            _xpu_comm_group_start,
        )

        with patch("paddle.is_compiled_with_xpu", return_value=False):
            _xpu_comm_group_start()

    def test_xpu_comm_group_end_not_xpu(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            _xpu_comm_group_end,
        )

        with patch("paddle.is_compiled_with_xpu", return_value=False):
            _xpu_comm_group_end()

    def test_xpu_comm_group_end_not_started(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            _xpu_comm_group_end,
        )

        with (
            patch("paddle.is_compiled_with_xpu", return_value=True),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._xpu_comm_group_started",
                False,
            ),
        ):
            _xpu_comm_group_end()

    def test_xpu_comm_group_start_already_started(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            _xpu_comm_group_start,
        )

        with (
            patch("paddle.is_compiled_with_xpu", return_value=True),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._xpu_comm_group_started",
                True,
            ),
            self.assertRaises(AssertionError),
        ):
            _xpu_comm_group_start()


class TestFourDirsIsValidSendRecvPartial(unittest.TestCase):
    """Additional tests for _is_valid_send_recv_partial in four_directions."""

    def test_mp_degree_one(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            _is_valid_send_recv_partial,
        )

        mock_tensor = MagicMock()
        mock_tensor.shape = [8]

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._enable_partial_send_recv",
                True,
            ),
            patch("numpy.prod", return_value=8),
        ):
            self.assertFalse(_is_valid_send_recv_partial(mock_tensor, 1))

    def test_not_divisible(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            _is_valid_send_recv_partial,
        )

        mock_tensor = MagicMock()
        mock_tensor.shape = [7]

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._enable_partial_send_recv",
                True,
            ),
            patch("numpy.prod", return_value=7),
        ):
            self.assertFalse(_is_valid_send_recv_partial(mock_tensor, 4))


if __name__ == "__main__":
    unittest.main()
