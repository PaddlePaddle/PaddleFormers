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


class TestFourDirsSyncSendPath(unittest.TestCase):
    """Tests for _p2p_helper with _sync_send=True in four_directions."""

    def test_sync_send_recv_prev(self):
        import paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication as mod
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            SendRecvMeta,
            _p2p_helper,
        )

        orig_sync = mod._sync_send
        try:
            mod._sync_send = True

            meta = SendRecvMeta()
            meta.recv_shape_message = [2, 3]
            meta.recv_dtype_message = 1
            meta.recv_stop_gradient = False
            meta.send_shape_message = [2, 3]
            meta.send_dtype_message = 1

            mock_hcg = MagicMock()
            mock_hcg.get_model_parallel_group.return_value = MagicMock()
            mock_hcg.get_model_parallel_world_size.return_value = 1
            mock_hcg.get_model_parallel_rank.return_value = 0
            mock_hcg.recv_prev_group = MagicMock()
            mock_hcg.recv_next_group = MagicMock()

            import paddle

            tensor = paddle.randn([2, 3])

            with (
                patch(
                    "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._hcg",
                    mock_hcg,
                ),
                patch(
                    "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication.send_partial"
                ),
                patch(
                    "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication.recv_partial",
                    return_value=None,
                ),
                patch(
                    "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication.allgather_partial"
                ),
                patch("paddle.distributed.wait"),
            ):
                recv_prev, recv_next = _p2p_helper(
                    tensor_send_next=None,
                    tensor_send_prev=None,
                    recv_prev=True,
                    recv_next=False,
                    sync_recv=True,
                    send_recv_meta=meta,
                )
                self.assertIsNotNone(recv_prev)
        finally:
            mod._sync_send = orig_sync

    def test_sync_send_send_next(self):
        import paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication as mod
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            SendRecvMeta,
            _p2p_helper,
        )

        orig_sync = mod._sync_send
        try:
            mod._sync_send = True

            meta = SendRecvMeta()
            meta.recv_shape_message = [2, 3]
            meta.recv_dtype_message = 1
            meta.recv_stop_gradient = False
            meta.send_shape_message = [2, 3]
            meta.send_dtype_message = 1

            mock_hcg = MagicMock()
            mock_hcg.get_model_parallel_group.return_value = MagicMock()
            mock_hcg.get_model_parallel_world_size.return_value = 1
            mock_hcg.get_model_parallel_rank.return_value = 0
            mock_hcg.send_next_group = MagicMock()

            import paddle

            tensor = paddle.randn([2, 3])

            with (
                patch(
                    "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._hcg",
                    mock_hcg,
                ),
                patch(
                    "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication.send_partial"
                ),
                patch(
                    "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication.recv_partial",
                    return_value=None,
                ),
                patch(
                    "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication.allgather_partial"
                ),
                patch("paddle.distributed.wait"),
            ):
                recv_prev, recv_next = _p2p_helper(
                    tensor_send_next=tensor,
                    tensor_send_prev=None,
                    recv_prev=False,
                    recv_next=False,
                    sync_recv=True,
                    send_recv_meta=meta,
                )
        finally:
            mod._sync_send = orig_sync


class TestFourDirsSendPartialDstSrc(unittest.TestCase):
    """Tests for send_partial and recv_partial with different dst/src in four_directions."""

    def test_send_partial_dst0(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            send_partial,
        )

        mock_hcg = MagicMock()
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
            # dst=0 -> uses _get_p2p_prev_rank
            mock_isend.assert_called_once()

    def test_recv_partial_src1(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            recv_partial,
        )

        mock_hcg = MagicMock()
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
                src=1,
                nranks=1,
                rank_id=0,
                group=mock_group,
                use_calc_stream=True,
            )
            # src=1 -> uses _get_p2p_next_rank
            mock_recv.assert_called_once()


class TestFourDirsPartialSendDynamic(unittest.TestCase):
    """Tests for _partial_send_op and _partial_recv_op in four_directions."""

    def test_partial_send_op_dynamic_mode(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            _partial_send_op,
        )

        mock_tensor = MagicMock()
        mock_group = MagicMock()
        mock_group.get_group_rank.return_value = 1

        with patch(
            "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication.framework.in_dynamic_mode",
            return_value=True,
        ):
            _partial_send_op(
                mock_tensor,
                mock_group,
                use_calc_stream=True,
                ring_id=0,
                dst=1,
                nranks=2,
                rank_id=0,
            )
            mock_group.process_group.send_partial_on_calc_stream.assert_called_once()

    def test_partial_recv_op(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            _partial_recv_op,
        )

        mock_tensor = MagicMock()
        mock_group = MagicMock()
        mock_group.get_group_rank.return_value = 0

        _partial_recv_op(
            mock_tensor,
            mock_group,
            use_calc_stream=True,
            ring_id=0,
            src=0,
            nranks=2,
            rank_id=0,
        )
        mock_group.process_group.recv_partial_on_calc_stream.assert_called_once()

    def test_partial_recv_op_not_calc_stream(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            _partial_recv_op,
        )

        mock_tensor = MagicMock()
        mock_group = MagicMock()
        mock_group.get_group_rank.return_value = 0

        _partial_recv_op(
            mock_tensor,
            mock_group,
            use_calc_stream=False,
            ring_id=0,
            src=0,
            nranks=2,
            rank_id=0,
        )
        mock_group.process_group.recv_partial.assert_called_once()


class TestFourDirsAllgatherPartialOp(unittest.TestCase):
    """Tests for _partial_allgather_op in four_directions."""

    def test_partial_allgather_calc_stream(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            _partial_allgather_op,
        )

        mock_tensor = MagicMock()
        mock_group = MagicMock()

        _partial_allgather_op(
            mock_tensor,
            mock_group,
            use_calc_stream=True,
            ring_id=0,
            nranks=2,
            rank_id=0,
        )
        mock_group.process_group.all_gather_partial_on_calc_stream.assert_called_once()

    def test_partial_allgather_not_calc_stream(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            _partial_allgather_op,
        )

        mock_tensor = MagicMock()
        mock_group = MagicMock()

        _partial_allgather_op(
            mock_tensor,
            mock_group,
            use_calc_stream=False,
            ring_id=0,
            nranks=2,
            rank_id=0,
        )
        mock_group.process_group.all_gather_partial.assert_called_once()


if __name__ == "__main__":
    unittest.main()
