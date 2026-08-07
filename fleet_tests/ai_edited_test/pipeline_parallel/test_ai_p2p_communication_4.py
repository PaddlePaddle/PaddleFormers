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


class TestP2pCommBatchSendRecvOnCalcStream(unittest.TestCase):
    """Tests for batch_send_recv_on_calc_stream in p2p_communication."""

    def test_warn_not_in_group(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            P2PonCalcStream,
            _send_on_calc_stream,
            batch_send_recv_on_calc_stream,
        )

        mock_group = MagicMock()
        with patch(
            "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication._warn_cur_rank_not_in_group",
            return_value=True,
        ):
            op = P2PonCalcStream(
                _send_on_calc_stream, MagicMock(), 1, mock_group
            )
            # Should return early
            batch_send_recv_on_calc_stream([op])

    def test_batch_send_recv_ops(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            P2PonCalcStream,
            _send_on_calc_stream,
            batch_send_recv_on_calc_stream,
        )

        mock_group = MagicMock()
        mock_group.backend = "nccl"
        mock_tensor = MagicMock()

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication._warn_cur_rank_not_in_group",
                return_value=False,
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication._get_global_group",
                return_value=mock_group,
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication._is_valid_send_recv_partial",
                return_value=False,
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication._coalescing_manager"
            ) as mock_coalesce,
            patch.dict(os.environ, {"FLAGS_pp_check_naninf": "0"}),
        ):
            mock_coalesce.return_value.__enter__ = MagicMock()
            mock_coalesce.return_value.__exit__ = MagicMock(return_value=False)

            op = P2PonCalcStream(
                _send_on_calc_stream, mock_tensor, 1, mock_group
            )
            batch_send_recv_on_calc_stream([op])

    def test_batch_send_recv_with_nan_check(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            P2PonCalcStream,
            _send_on_calc_stream,
            batch_send_recv_on_calc_stream,
        )

        mock_group = MagicMock()
        mock_group.backend = "nccl"
        mock_tensor = MagicMock()

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication._warn_cur_rank_not_in_group",
                return_value=False,
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication._get_global_group",
                return_value=mock_group,
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication._is_valid_send_recv_partial",
                return_value=False,
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication._coalescing_manager"
            ) as mock_coalesce,
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication.check_naninf",
                return_value=None,
            ),
            patch.dict(os.environ, {"FLAGS_pp_check_naninf": "1"}),
        ):
            mock_coalesce.return_value.__enter__ = MagicMock()
            mock_coalesce.return_value.__exit__ = MagicMock(return_value=False)

            op = P2PonCalcStream(
                _send_on_calc_stream, mock_tensor, 1, mock_group
            )
            batch_send_recv_on_calc_stream([op])

    def test_batch_send_recv_nan_detected(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            P2PonCalcStream,
            _send_on_calc_stream,
            batch_send_recv_on_calc_stream,
        )

        mock_group = MagicMock()
        mock_group.backend = "nccl"
        mock_tensor = MagicMock()

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication._warn_cur_rank_not_in_group",
                return_value=False,
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication.check_naninf",
                return_value="NaN detected",
            ),
            patch.dict(os.environ, {"FLAGS_pp_check_naninf": "1"}),
        ):
            op = P2PonCalcStream(
                _send_on_calc_stream, mock_tensor, 1, mock_group
            )
            with self.assertRaises(ValueError):
                batch_send_recv_on_calc_stream([op])


class TestP2pCommBatchP2pOps(unittest.TestCase):
    """Tests for _batch_p2p_tuple_or_tensor in p2p_communication."""

    def test_tuple_tensors(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            P2PonCalcStream,
            _batch_p2p_tuple_or_tensor,
            _send_on_calc_stream,
        )

        t1 = MagicMock()
        t2 = MagicMock()
        mock_group = MagicMock()
        ops = _batch_p2p_tuple_or_tensor(
            (t1, t2),
            _send_on_calc_stream,
            1,
            mock_group,
            mp_degree=2,
            mp_rank=0,
        )
        self.assertEqual(len(ops), 2)
        self.assertIsInstance(ops[0], P2PonCalcStream)


class TestP2pCommP2pOps(unittest.TestCase):
    """Tests for _p2p_ops_tuple_or_tensor in p2p_communication."""

    def test_single_tensor_with_isend(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            _p2p_ops_tuple_or_tensor,
        )

        mock_tensor = MagicMock()
        mock_tensor.place = "cuda:0"
        mock_group = MagicMock()

        with (
            patch("paddle.device.get_device", return_value="cuda:0"),
            patch(
                "paddle.distributed.isend", return_value=MagicMock()
            ) as mock_isend,
        ):
            reqs = _p2p_ops_tuple_or_tensor(
                mock_tensor, paddle.distributed.isend, 2, mock_group
            )
            self.assertEqual(len(reqs), 1)
            mock_isend.assert_called_once()

    def test_tuple_tensor_with_irecv(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            _p2p_ops_tuple_or_tensor,
        )

        t1 = MagicMock()
        t1.place = "cuda:0"
        t2 = MagicMock()
        t2.place = "cuda:0"
        mock_group = MagicMock()

        with (
            patch("paddle.device.get_device", return_value="cuda:0"),
            patch(
                "paddle.distributed.irecv", return_value=MagicMock()
            ) as mock_irecv,
        ):
            reqs = _p2p_ops_tuple_or_tensor(
                (t1, t2), paddle.distributed.irecv, 0, mock_group
            )
            self.assertEqual(len(reqs), 2)
            self.assertEqual(mock_irecv.call_count, 2)

    def test_nan_check_raises(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            _p2p_ops_tuple_or_tensor,
        )

        mock_tensor = MagicMock()
        mock_tensor.place = "cuda:0"
        mock_group = MagicMock()

        with (
            patch("paddle.device.get_device", return_value="cuda:0"),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication.check_naninf",
                return_value="NaN found",
            ),
            patch.dict(os.environ, {"FLAGS_pp_check_naninf": "1"}),
            self.assertRaises(ValueError),
        ):
            _p2p_ops_tuple_or_tensor(
                mock_tensor, paddle.distributed.isend, 2, mock_group
            )


class TestP2pCommP2pHelper(unittest.TestCase):
    """Tests for P2pHelper in p2p_communication."""

    def test_send_forward_recv_forward(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            P2pHelper,
        )

        helper = P2pHelper()
        helper._send_recv_meta.has_send_meta = True
        helper._send_recv_meta.has_recv_meta = True

        mock_hcg = MagicMock()
        mock_tensor = MagicMock()

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication._hcg",
                mock_hcg,
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication._p2p_helper",
                return_value=(None, None, None),
            ),
        ):
            result = helper.send_forward_recv_forward(
                mock_tensor, recv_prev=False
            )

    def test_send_backward_recv_backward(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            P2pHelper,
        )

        helper = P2pHelper()
        mock_hcg = MagicMock()
        mock_tensor = MagicMock()

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication._hcg",
                mock_hcg,
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication._p2p_helper",
                return_value=(None, MagicMock(), None),
            ),
        ):
            result = helper.send_backward_recv_backward(
                mock_tensor, recv_next=False
            )

    def test_send_forward_backward_recv_forward_backward(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            P2pHelper,
        )

        helper = P2pHelper()
        helper._send_recv_meta.has_send_meta = True
        mock_hcg = MagicMock()
        mock_fwd = MagicMock()
        mock_bwd = MagicMock()

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication._hcg",
                mock_hcg,
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication._p2p_helper",
                return_value=(None, None, None),
            ),
        ):
            result = helper.send_forward_backward_recv_forward_backward(
                mock_fwd, mock_bwd, recv_prev=False, recv_next=False
            )

    def test_recv_forward_not_first_stage(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            P2pHelper,
        )

        helper = P2pHelper()
        helper._send_recv_meta.has_recv_meta = True
        mock_hcg = MagicMock()

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication._hcg",
                mock_hcg,
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication._p2p_helper",
                return_value=(MagicMock(), None, None),
            ),
        ):
            result = helper.recv_forward(pp_first_stage=False)
            self.assertIsNotNone(result)

    def test_recv_backward_not_last_stage(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            P2pHelper,
        )

        helper = P2pHelper()
        helper._send_recv_meta.has_recv_meta = True
        mock_hcg = MagicMock()

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication._hcg",
                mock_hcg,
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication._p2p_helper",
                return_value=(None, MagicMock(), None),
            ),
        ):
            result = helper.recv_backward(pp_last_stage=False)
            self.assertIsNotNone(result)


class TestP2pCommP2pHelperWithTimers(unittest.TestCase):
    """Tests for P2pHelper with timers in p2p_communication."""

    def test_recv_forward_timer(self):
        import paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication as mod
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            P2pHelper,
        )

        orig_timers = mod._timers
        try:
            mock_timer = MagicMock()
            mod._timers = mock_timer

            helper = P2pHelper()
            result = helper.recv_forward(pp_first_stage=True)
            mock_timer.assert_called_with("recv_forward")
            self.assertIsNone(result)
        finally:
            mod._timers = orig_timers

    def test_send_forward_recv_backward_timer(self):
        import paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication as mod
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            P2pHelper,
        )

        orig_timers = mod._timers
        try:
            mock_timer = MagicMock()
            mod._timers = mock_timer

            helper = P2pHelper()
            result = helper.send_forward_recv_backward(
                MagicMock(), pp_last_stage=True
            )
            mock_timer.assert_called_with("send_forward_recv_backward")
            self.assertIsNone(result)
        finally:
            mod._timers = orig_timers


if __name__ == "__main__":
    unittest.main()
