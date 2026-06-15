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

import paddle


class TestFourDirsP2pHelperRecvPrevNext(unittest.TestCase):
    """Tests for _p2p_helper with recv_prev and recv_next in four_directions."""

    def test_recv_prev_single_tensor(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            SendRecvMeta,
            _p2p_helper,
        )

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

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._hcg",
                mock_hcg,
            ),
            patch("paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication.send_partial"),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication.recv_partial",
                return_value=None,
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication.allgather_partial"
            ),
            patch("paddle.distributed.wait"),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._xpu_comm_group_start"
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._xpu_comm_group_end"
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._sync_send",
                False,
            ),
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
            self.assertIsNone(recv_next)

    def test_recv_next_single_tensor(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            SendRecvMeta,
            _p2p_helper,
        )

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

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._hcg",
                mock_hcg,
            ),
            patch("paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication.send_partial"),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication.recv_partial",
                return_value=None,
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication.allgather_partial"
            ),
            patch("paddle.distributed.wait"),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._xpu_comm_group_start"
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._xpu_comm_group_end"
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._sync_send",
                False,
            ),
        ):
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

    def test_recv_prev_tuple(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            SendRecvMeta,
            _p2p_helper,
        )

        meta = SendRecvMeta()
        meta.recv_shape_message = ([2, 3], [4, 5])
        meta.recv_dtype_message = (1, 1)
        meta.recv_stop_gradient = (False, False)
        meta.send_shape_message = [2, 3]
        meta.send_dtype_message = 1

        mock_hcg = MagicMock()
        mock_hcg.get_model_parallel_group.return_value = MagicMock()
        mock_hcg.get_model_parallel_world_size.return_value = 1
        mock_hcg.get_model_parallel_rank.return_value = 0

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._hcg",
                mock_hcg,
            ),
            patch("paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication.send_partial"),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication.recv_partial",
                return_value=None,
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication.allgather_partial"
            ),
            patch("paddle.distributed.wait"),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._xpu_comm_group_start"
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._xpu_comm_group_end"
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._sync_send",
                False,
            ),
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
            self.assertIsInstance(recv_prev, tuple)
            self.assertEqual(len(recv_prev), 2)


class TestFourDirsP2pHelperSendTensor(unittest.TestCase):
    """Tests for _p2p_helper with sending tensors in four_directions."""

    def test_send_prev_single(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            SendRecvMeta,
            _p2p_helper,
        )

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

        tensor = paddle.randn([2, 3])

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._hcg",
                mock_hcg,
            ),
            patch("paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication.send_partial"),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication.recv_partial",
                return_value=None,
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication.allgather_partial"
            ),
            patch("paddle.distributed.wait"),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._xpu_comm_group_start"
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._xpu_comm_group_end"
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._sync_send",
                False,
            ),
        ):
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

    def test_send_next_single(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            SendRecvMeta,
            _p2p_helper,
        )

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

        tensor = paddle.randn([2, 3])

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._hcg",
                mock_hcg,
            ),
            patch("paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication.send_partial"),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication.recv_partial",
                return_value=None,
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication.allgather_partial"
            ),
            patch("paddle.distributed.wait"),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._xpu_comm_group_start"
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._xpu_comm_group_end"
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._sync_send",
                False,
            ),
        ):
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

    def test_send_tuple(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            SendRecvMeta,
            _p2p_helper,
        )

        meta = SendRecvMeta()
        meta.recv_shape_message = [2, 3]
        meta.recv_dtype_message = 1
        meta.recv_stop_gradient = False
        meta.send_shape_message = ([2, 3], [4, 5])
        meta.send_dtype_message = (1, 1)

        mock_hcg = MagicMock()
        mock_hcg.get_model_parallel_group.return_value = MagicMock()
        mock_hcg.get_model_parallel_world_size.return_value = 1
        mock_hcg.get_model_parallel_rank.return_value = 0

        t1 = paddle.randn([2, 3])
        t2 = paddle.randn([4, 5])

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._hcg",
                mock_hcg,
            ),
            patch("paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication.send_partial"),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication.recv_partial",
                return_value=None,
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication.allgather_partial"
            ),
            patch("paddle.distributed.wait"),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._xpu_comm_group_start"
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._xpu_comm_group_end"
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._sync_send",
                False,
            ),
        ):
            recv_prev, recv_next = _p2p_helper(
                tensor_send_next=(t1, t2),
                tensor_send_prev=None,
                recv_prev=False,
                recv_next=False,
                sync_recv=True,
                send_recv_meta=meta,
            )


class TestFourDirsSyncRecvFalse(unittest.TestCase):
    """Tests for _p2p_helper with sync_recv=False in four_directions."""

    def test_sync_recv_false(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            SendRecvMeta,
            _p2p_helper,
        )

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

        mock_task = MagicMock()

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._hcg",
                mock_hcg,
            ),
            patch("paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication.send_partial"),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication.recv_partial",
                return_value=mock_task,
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication.allgather_partial"
            ),
            patch("paddle.distributed.wait"),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._xpu_comm_group_start"
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._xpu_comm_group_end"
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._sync_send",
                False,
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication.framework.in_dynamic_mode",
                return_value=True,
            ),
        ):
            recv_prev, recv_next = _p2p_helper(
                tensor_send_next=None,
                tensor_send_prev=None,
                recv_prev=True,
                recv_next=False,
                sync_recv=False,
                send_recv_meta=meta,
            )
            self.assertIsNotNone(recv_prev)


if __name__ == "__main__":
    unittest.main()
