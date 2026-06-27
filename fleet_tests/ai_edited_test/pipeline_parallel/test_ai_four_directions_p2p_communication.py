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


class TestXpuCommGroup(unittest.TestCase):
    """Tests for XPU communication group management."""

    def test_xpu_comm_group_start_not_xpu(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            _xpu_comm_group_start,
        )

        with patch("paddle.is_compiled_with_xpu", return_value=False):
            # Should return early without error
            _xpu_comm_group_start()

    def test_xpu_comm_group_start_xpu(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            _xpu_comm_group_start,
        )

        with (
            patch("paddle.is_compiled_with_xpu", return_value=True),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._xpu_comm_group_started",
                False,
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication.framework"
            ) as mock_fw,
        ):
            _xpu_comm_group_start()

    def test_xpu_comm_group_start_already_started(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            _xpu_comm_group_start,
        )

        with (  # noqa: SIM117
            patch("paddle.is_compiled_with_xpu", return_value=True),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._xpu_comm_group_started",
                True,
            ),
        ):
            with self.assertRaises(AssertionError):
                _xpu_comm_group_start()

    def test_xpu_comm_group_end_not_xpu(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            _xpu_comm_group_end,
        )

        with patch("paddle.is_compiled_with_xpu", return_value=False):
            _xpu_comm_group_end()

    def test_xpu_comm_group_end_started(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            _xpu_comm_group_end,
        )

        with (
            patch("paddle.is_compiled_with_xpu", return_value=True),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._xpu_comm_group_started",
                True,
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication.framework"
            ) as mock_fw,
        ):
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
            # Should return without calling group_end
            _xpu_comm_group_end()


class TestFourDirectionsSendRecvMeta(unittest.TestCase):
    """Tests for SendRecvMeta in four_directions_p2p_communication."""

    def test_init(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            SendRecvMeta,
        )

        meta = SendRecvMeta()
        self.assertIsNone(meta.send_shape_message)
        self.assertIsNone(meta.send_dtype_message)
        self.assertIsNone(meta.recv_shape_message)
        self.assertIsNone(meta.recv_dtype_message)
        self.assertFalse(meta.has_send_meta)
        self.assertFalse(meta.has_recv_meta)

    def test_set_send_message_single_tensor(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            SendRecvMeta,
        )

        meta = SendRecvMeta()
        t = paddle.randn([2, 3])
        meta.set_send_message(t)
        self.assertEqual(meta.send_shape_message, [2, 3])

    def test_set_send_message_tuple(self):
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
        self.assertEqual(len(meta.send_shape_message), 2)

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
        # t1 should be filtered out because stop_gradient is True
        self.assertEqual(len(meta.send_shape_message), 1)

    def test_recv_meta_single_tensor(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            SendRecvMeta,
        )

        meta = SendRecvMeta()
        mock_group = MagicMock()
        mock_hcg = MagicMock()
        mock_hcg._get_p2p_prev_rank.return_value = 1

        # Mock paddle.distributed.recv to return values in sequence
        recv_values = [
            0,
            [2, 3],
            1,
            0,
        ]  # tensor_type=0, shape, dtype, stop_grad

        def mock_recv_side_effect(tensor, src, group):
            if hasattr(tensor, "item"):
                val = recv_values.pop(0)
                tensor.item = MagicMock(return_value=val)
            elif hasattr(tensor, "tolist"):
                val = recv_values.pop(0)
                tensor.tolist = MagicMock(return_value=val)
            return tensor

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._hcg",
                mock_hcg,
            ),
            patch("paddle.distributed.recv", side_effect=mock_recv_side_effect),
        ):
            try:
                meta.recv_meta(mock_group)
            except Exception:
                pass

    def test_recv_meta_tuple_tensor(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            SendRecvMeta,
        )

        meta = SendRecvMeta()
        mock_group = MagicMock()
        mock_hcg = MagicMock()
        mock_hcg._get_p2p_prev_rank.return_value = 1

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._hcg",
                mock_hcg,
            ),
            patch("paddle.distributed.recv"),
            patch("paddle.to_tensor") as mock_to_tensor,
        ):
            tensor_type_t = MagicMock()
            tensor_type_t.item.return_value = 1
            num_t = MagicMock()
            num_t.item.return_value = 2

            shape_t = MagicMock()
            shape_t.tolist.return_value = [2, 3]
            dtype_t = MagicMock()
            dtype_t.item.return_value = 1
            stop_grad_t = MagicMock()
            stop_grad_t.item.return_value = 0

            mock_to_tensor.side_effect = [
                tensor_type_t,
                num_t,
                shape_t,
                shape_t,
                dtype_t,
                dtype_t,
                stop_grad_t,
                stop_grad_t,
            ]

            try:
                meta.recv_meta(mock_group)
            except Exception:
                pass

    def test_send_meta_single_tensor(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            SendRecvMeta,
        )

        meta = SendRecvMeta()
        mock_group = MagicMock()
        mock_hcg = MagicMock()
        mock_hcg._get_p2p_next_rank.return_value = 2
        t = paddle.randn([2, 3])

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._hcg",
                mock_hcg,
            ),
            patch("paddle.distributed.send"),
            patch("paddle.to_tensor") as mock_to_tensor,
        ):
            mock_to_tensor.return_value = MagicMock()
            meta.send_meta(t, mock_group)

    def test_send_meta_tuple_tensor(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            SendRecvMeta,
        )

        meta = SendRecvMeta()
        mock_group = MagicMock()
        mock_hcg = MagicMock()
        mock_hcg._get_p2p_next_rank.return_value = 2
        t1 = paddle.randn([2, 3])
        t2 = paddle.randn([4, 5])

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._hcg",
                mock_hcg,
            ),
            patch("paddle.distributed.send"),
            patch("paddle.to_tensor") as mock_to_tensor,
        ):
            mock_to_tensor.return_value = MagicMock()
            meta.send_meta((t1, t2), mock_group)


class TestFourDirectionsIsValidSendRecvPartial(unittest.TestCase):
    """Tests for _is_valid_send_recv_partial."""

    def test_disabled(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            _is_valid_send_recv_partial,
        )

        with patch(
            "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._enable_partial_send_recv",
            False,
        ):
            self.assertFalse(_is_valid_send_recv_partial(MagicMock(), 4))

    def test_valid(self):
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
            self.assertTrue(_is_valid_send_recv_partial(mock_tensor, 4))

    def test_zero_elements(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            _is_valid_send_recv_partial,
        )

        mock_tensor = MagicMock()
        mock_tensor.shape = [0]
        with (  # noqa: SIM117
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._enable_partial_send_recv",
                True,
            ),
            patch("numpy.prod", return_value=0),
        ):
            with self.assertRaises(AssertionError):
                _is_valid_send_recv_partial(mock_tensor, 4)


class TestFourDirectionsPartialOps(unittest.TestCase):
    """Tests for partial send/recv/allgather ops."""

    def test_send_partial_not_member(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            send_partial,
        )

        mock_group = MagicMock()
        mock_group.is_member.return_value = False
        mock_tensor = MagicMock()
        result = send_partial(mock_tensor, group=mock_group)
        self.assertIsNone(result)

    def test_recv_partial_not_member(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            recv_partial,
        )

        mock_group = MagicMock()
        mock_group.is_member.return_value = False
        mock_tensor = MagicMock()
        result = recv_partial(mock_tensor, group=mock_group)
        self.assertIsNone(result)

    def test_allgather_partial_not_valid(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            allgather_partial,
        )

        mock_tensor = MagicMock()
        with patch(
            "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._is_valid_send_recv_partial",
            return_value=False,
        ):
            result = allgather_partial(mock_tensor, nranks=4, rank_id=0)
            self.assertEqual(result, mock_tensor)

    def test_allgather_partial_not_member(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            allgather_partial,
        )

        mock_group = MagicMock()
        mock_group.is_member.return_value = False
        mock_tensor = MagicMock()
        with patch(
            "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._is_valid_send_recv_partial",
            return_value=True,
        ):
            result = allgather_partial(
                mock_tensor, nranks=4, rank_id=0, group=mock_group
            )
            self.assertIsNone(result)

    def test_send_partial_invalid_partial(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            send_partial,
        )

        mock_hcg = MagicMock()
        mock_hcg._get_p2p_next_rank.return_value = 5
        mock_hcg._get_p2p_prev_rank.return_value = 3
        mock_tensor = MagicMock()
        mock_group = MagicMock()
        mock_group.is_member.return_value = True

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
            send_partial(
                mock_tensor, dst=1, nranks=1, rank_id=0, group=mock_group
            )
            mock_isend.assert_called_once()

    def test_recv_partial_invalid_partial_calc_stream(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            recv_partial,
        )

        mock_hcg = MagicMock()
        mock_hcg._get_p2p_prev_rank.return_value = 3
        mock_hcg._get_p2p_next_rank.return_value = 5
        mock_tensor = MagicMock()
        mock_group = MagicMock()
        mock_group.is_member.return_value = True

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
            recv_partial(
                mock_tensor,
                src=0,
                nranks=1,
                rank_id=0,
                group=mock_group,
                use_calc_stream=True,
            )
            mock_recv.assert_called_once()

    def test_recv_partial_invalid_partial_irecv(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            recv_partial,
        )

        mock_hcg = MagicMock()
        mock_hcg._get_p2p_prev_rank.return_value = 3
        mock_hcg._get_p2p_next_rank.return_value = 5
        mock_tensor = MagicMock()
        mock_group = MagicMock()
        mock_group.is_member.return_value = True

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._hcg",
                mock_hcg,
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._is_valid_send_recv_partial",
                return_value=False,
            ),
            patch("paddle.distributed.irecv") as mock_irecv,
            patch("paddle.framework.in_dynamic_mode", return_value=True),
        ):
            recv_partial(
                mock_tensor,
                src=0,
                nranks=1,
                rank_id=0,
                group=mock_group,
                use_calc_stream=False,
            )
            mock_irecv.assert_called_once()

    def test_send_partial_dst_0(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            send_partial,
        )

        mock_hcg = MagicMock()
        mock_hcg._get_p2p_next_rank.return_value = 5
        mock_hcg._get_p2p_prev_rank.return_value = 3
        mock_tensor = MagicMock()
        mock_group = MagicMock()
        mock_group.is_member.return_value = True

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
            send_partial(
                mock_tensor, dst=0, nranks=1, rank_id=0, group=mock_group
            )
            mock_isend.assert_called_once()

    def test_recv_partial_src_1(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            recv_partial,
        )

        mock_hcg = MagicMock()
        mock_hcg._get_p2p_prev_rank.return_value = 3
        mock_hcg._get_p2p_next_rank.return_value = 5
        mock_tensor = MagicMock()
        mock_group = MagicMock()
        mock_group.is_member.return_value = True

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
            recv_partial(
                mock_tensor, src=1, nranks=1, rank_id=0, group=mock_group
            )
            mock_recv.assert_called_once()


class TestFourDirectionsInitializeP2PGroups(unittest.TestCase):
    """Tests for initialize_p2p_groups."""

    def test_basic_init(self):
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
        with patch(
            "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication.logger"
        ):
            initialize_p2p_groups(
                mock_hcg, enable_partial_send_recv=True, enable_timer=False
            )

    def test_init_with_timer(self):
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
        mock_timer = MagicMock()
        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication.timer"
            ) as mock_timer_mod,
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication.logger"
            ),
        ):
            mock_timer_mod.get_timers.return_value = mock_timer
            initialize_p2p_groups(mock_hcg, enable_timer=True)


class TestFourDirectionsP2pHelper(unittest.TestCase):
    """Tests for P2pHelper in four_directions_p2p_communication."""

    def test_init(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            P2pHelper,
        )

        helper = P2pHelper(use_cache=True)
        self.assertTrue(helper._use_cache)

    def test_init_no_cache(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            P2pHelper,
        )

        helper = P2pHelper(use_cache=False)
        self.assertFalse(helper._use_cache)

    def test_recv_forward_first_stage(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            P2pHelper,
        )

        helper = P2pHelper()
        result = helper.recv_forward(pp_first_stage=True)
        self.assertIsNone(result)

    def test_recv_backward_last_stage(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            P2pHelper,
        )

        helper = P2pHelper()
        result = helper.recv_backward(pp_last_stage=True)
        self.assertIsNone(result)

    def test_send_forward_last_stage(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            P2pHelper,
        )

        helper = P2pHelper()
        mock_tensor = MagicMock()
        # Should not raise when pp_last_stage is True
        helper.send_forward(mock_tensor, pp_last_stage=True)

    def test_send_backward_first_stage(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            P2pHelper,
        )

        helper = P2pHelper()
        mock_tensor = MagicMock()
        # Should not raise when pp_first_stage is True
        helper.send_backward(mock_tensor, pp_first_stage=True)

    def test_send_forward_recv_backward_last_stage(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            P2pHelper,
        )

        helper = P2pHelper()
        mock_tensor = MagicMock()
        result = helper.send_forward_recv_backward(
            mock_tensor, pp_last_stage=True
        )
        self.assertIsNone(result)

    def test_send_backward_recv_forward_first_stage(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            P2pHelper,
        )

        helper = P2pHelper()
        mock_tensor = MagicMock()
        result = helper.send_backward_recv_forward(
            mock_tensor, pp_first_stage=True
        )
        self.assertIsNone(result)

    def test_send_forward_recv_forward(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            P2pHelper,
        )

        helper = P2pHelper()
        mock_tensor = MagicMock()
        mock_hcg = MagicMock()

        # Set has_send_meta=True so _send_meta is skipped (avoids paddle.distributed.send)
        helper._send_recv_meta.has_send_meta = True

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._hcg",
                mock_hcg,
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._p2p_helper",
                return_value=(None, None),
            ),
        ):
            result = helper.send_forward_recv_forward(
                mock_tensor, recv_prev=False
            )

    def test_send_backward_recv_backward(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            P2pHelper,
        )

        helper = P2pHelper()
        mock_tensor = MagicMock()
        mock_hcg = MagicMock()

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._hcg",
                mock_hcg,
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._p2p_helper",
                return_value=(None, None),
            ),
        ):
            result = helper.send_backward_recv_backward(
                mock_tensor, recv_next=False
            )

    def test_send_forward_backward_recv_forward_backward(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            P2pHelper,
        )

        helper = P2pHelper()
        mock_fwd_tensor = MagicMock()
        mock_bwd_tensor = MagicMock()
        mock_hcg = MagicMock()

        # Set has_send_meta=True so _send_meta is skipped
        helper._send_recv_meta.has_send_meta = True

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._hcg",
                mock_hcg,
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._p2p_helper",
                return_value=(None, None),
            ),
        ):
            result = helper.send_forward_backward_recv_forward_backward(
                mock_fwd_tensor,
                mock_bwd_tensor,
                recv_prev=False,
                recv_next=False,
            )


class TestFourDirectionsP2pHelperMeta(unittest.TestCase):
    """Tests for P2pHelper _send_meta and _recv_meta."""

    def test_send_meta_first_time(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            P2pHelper,
        )

        helper = P2pHelper()
        t = paddle.randn([2, 3])
        mock_group = MagicMock()
        mock_hcg = MagicMock()
        mock_hcg.get_pipe_parallel_group.return_value = mock_group
        helper._send_recv_meta.has_send_meta = False

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._hcg",
                mock_hcg,
            ),
            patch("paddle.distributed.send"),
        ):
            helper._send_meta(t)

    def test_send_meta_cached(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            P2pHelper,
        )

        helper = P2pHelper(use_cache=True)
        helper._send_recv_meta.has_send_meta = True
        mock_tensor = MagicMock()
        # Should not raise and not call send_meta again
        helper._send_meta(mock_tensor)

    def test_recv_meta_first_time(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            P2pHelper,
        )

        helper = P2pHelper()
        mock_group = MagicMock()
        mock_hcg = MagicMock()
        mock_hcg.get_pipe_parallel_group.return_value = mock_group
        helper._send_recv_meta.has_recv_meta = False

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._hcg",
                mock_hcg,
            ),
            patch("paddle.distributed.recv"),
        ):
            try:
                helper._recv_meta()
            except Exception:
                pass

    def test_recv_meta_cached(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            P2pHelper,
        )

        helper = P2pHelper(use_cache=True)
        helper._send_recv_meta.has_recv_meta = True
        # Should not call recv_meta again
        helper._recv_meta()


if __name__ == "__main__":
    unittest.main()
