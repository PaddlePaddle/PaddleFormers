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


class TestP2pBatchedP2pOps(unittest.TestCase):
    """Tests for _batched_p2p_ops in p2p_communication."""

    def _make_mock_hcg(self):
        mock_hcg = MagicMock()
        mock_hcg.get_pipe_parallel_group.return_value = MagicMock()
        mock_hcg.get_model_parallel_world_size.return_value = 1
        mock_hcg.get_model_parallel_rank.return_value = 0
        mock_hcg.get_model_parallel_group.return_value = MagicMock()
        mock_hcg._get_p2p_next_rank.return_value = 3
        mock_hcg._get_p2p_prev_rank.return_value = 1
        return mock_hcg

    def test_all_none_tensors(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            _batched_p2p_ops,
        )

        mock_hcg = self._make_mock_hcg()
        # No tensors to send/recv should not raise
        with patch(
            "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication.allgather_partial"
        ):
            _batched_p2p_ops(None, None, None, None, mock_hcg)

    def test_send_prev_only(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            _batched_p2p_ops,
        )

        mock_hcg = self._make_mock_hcg()
        import paddle

        tensor = paddle.randn([2, 3])

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication.batch_send_recv_on_calc_stream"
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication.allgather_partial"
            ),
            patch.dict(os.environ, {"FLAGS_p2p_device_synchronize": "0"}),
        ):
            _batched_p2p_ops(tensor, None, None, None, mock_hcg)

    def test_recv_prev_only(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            _batched_p2p_ops,
        )

        mock_hcg = self._make_mock_hcg()
        import paddle

        tensor = paddle.randn([2, 3])

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication.batch_send_recv_on_calc_stream"
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication.allgather_partial"
            ),
            patch.dict(os.environ, {"FLAGS_p2p_device_synchronize": "0"}),
        ):
            _batched_p2p_ops(None, tensor, None, None, mock_hcg)

    def test_send_next_only(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            _batched_p2p_ops,
        )

        mock_hcg = self._make_mock_hcg()
        import paddle

        tensor = paddle.randn([2, 3])

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication.batch_send_recv_on_calc_stream"
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication.allgather_partial"
            ),
            patch.dict(os.environ, {"FLAGS_p2p_device_synchronize": "0"}),
        ):
            _batched_p2p_ops(None, None, tensor, None, mock_hcg)

    def test_recv_next_only(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            _batched_p2p_ops,
        )

        mock_hcg = self._make_mock_hcg()
        import paddle

        tensor = paddle.randn([2, 3])

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication.batch_send_recv_on_calc_stream"
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication.allgather_partial"
            ),
            patch.dict(os.environ, {"FLAGS_p2p_device_synchronize": "0"}),
        ):
            _batched_p2p_ops(None, None, None, tensor, mock_hcg)


class TestP2pP2pOps(unittest.TestCase):
    """Tests for _p2p_ops in p2p_communication."""

    def _make_mock_hcg(self, stage_id=0):
        mock_hcg = MagicMock()
        mock_hcg.get_pipe_parallel_group.return_value = MagicMock()
        mock_hcg.get_stage_id.return_value = stage_id
        mock_hcg._get_p2p_next_rank.return_value = 3
        mock_hcg._get_p2p_prev_rank.return_value = 1
        return mock_hcg

    def test_even_stage_send_recv(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            _p2p_ops,
        )

        mock_hcg = self._make_mock_hcg(stage_id=0)
        import paddle

        tensor = paddle.randn([2, 3])

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication._hcg",
                mock_hcg,
            ),
            patch("paddle.device.get_device", return_value="cuda:0"),
            patch("paddle.distributed.isend", return_value=MagicMock()),
            patch("paddle.distributed.irecv", return_value=MagicMock()),
            patch.dict(os.environ, {"FLAGS_pp_check_naninf": "0"}),
        ):
            reqs = _p2p_ops(tensor, None, tensor, None, mock_hcg)
            self.assertGreater(len(reqs), 0)

    def test_odd_stage_send_recv(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            _p2p_ops,
        )

        mock_hcg = self._make_mock_hcg(stage_id=1)
        import paddle

        tensor = paddle.randn([2, 3])

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication._hcg",
                mock_hcg,
            ),
            patch("paddle.device.get_device", return_value="cuda:0"),
            patch("paddle.distributed.isend", return_value=MagicMock()),
            patch("paddle.distributed.irecv", return_value=MagicMock()),
            patch.dict(os.environ, {"FLAGS_pp_check_naninf": "0"}),
        ):
            reqs = _p2p_ops(tensor, None, tensor, None, mock_hcg)
            self.assertGreater(len(reqs), 0)

    def test_all_none(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            _p2p_ops,
        )

        mock_hcg = self._make_mock_hcg()
        reqs = _p2p_ops(None, None, None, None, mock_hcg)
        self.assertEqual(len(reqs), 0)


class TestP2pHelper(unittest.TestCase):
    """Additional tests for _p2p_helper in p2p_communication."""

    def test_p2p_helper_recv_prev_single(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            SendRecvMeta,
            _p2p_helper,
        )

        meta = SendRecvMeta()
        meta.recv_shape_message = [2, 3]
        meta.recv_dtype_message = 1
        meta.recv_stop_gradient = False
        meta.recv_key_message = None
        meta.send_shape_message = [2, 3]
        meta.send_dtype_message = 1
        meta.send_key_message = None

        mock_hcg = MagicMock()
        mock_hcg.get_model_parallel_group.return_value = MagicMock()
        mock_hcg.get_model_parallel_world_size.return_value = 1
        mock_hcg.get_model_parallel_rank.return_value = 0

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication._hcg",
                mock_hcg,
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication._batched_p2p_ops",
                return_value=None,
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication.allgather_partial"
            ),
        ):
            recv_prev, recv_next, reqs = _p2p_helper(
                tensor_send_next=None,
                tensor_send_prev=None,
                recv_prev=True,
                recv_next=False,
                send_recv_meta=meta,
                batch_p2p_comm=True,
            )
            self.assertIsNotNone(recv_prev)
            self.assertIsNone(recv_next)

    def test_p2p_helper_recv_next_single(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            SendRecvMeta,
            _p2p_helper,
        )

        meta = SendRecvMeta()
        meta.recv_shape_message = [2, 3]
        meta.recv_dtype_message = 1
        meta.recv_stop_gradient = False
        meta.recv_key_message = None
        meta.send_shape_message = [2, 3]
        meta.send_dtype_message = 1
        meta.send_key_message = None

        mock_hcg = MagicMock()
        mock_hcg.get_model_parallel_group.return_value = MagicMock()
        mock_hcg.get_model_parallel_world_size.return_value = 1
        mock_hcg.get_model_parallel_rank.return_value = 0

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication._hcg",
                mock_hcg,
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication._batched_p2p_ops",
                return_value=None,
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication.allgather_partial"
            ),
        ):
            recv_prev, recv_next, reqs = _p2p_helper(
                tensor_send_next=None,
                tensor_send_prev=None,
                recv_prev=False,
                recv_next=True,
                send_recv_meta=meta,
                batch_p2p_comm=True,
            )
            self.assertIsNone(recv_prev)
            self.assertIsNotNone(recv_next)

    def test_p2p_helper_recv_prev_tuple(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            SendRecvMeta,
            _p2p_helper,
        )

        meta = SendRecvMeta()
        meta.recv_shape_message = ([2, 3], [4, 5])
        meta.recv_dtype_message = (1, 1)
        meta.recv_stop_gradient = (False, False)
        meta.recv_key_message = (None, None)
        meta.send_shape_message = [2, 3]
        meta.send_dtype_message = 1
        meta.send_key_message = None

        mock_hcg = MagicMock()
        mock_hcg.get_model_parallel_group.return_value = MagicMock()
        mock_hcg.get_model_parallel_world_size.return_value = 1
        mock_hcg.get_model_parallel_rank.return_value = 0

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication._hcg",
                mock_hcg,
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication._batched_p2p_ops",
                return_value=None,
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication.allgather_partial"
            ),
        ):
            recv_prev, recv_next, reqs = _p2p_helper(
                tensor_send_next=None,
                tensor_send_prev=None,
                recv_prev=True,
                recv_next=False,
                send_recv_meta=meta,
                batch_p2p_comm=True,
            )
            self.assertIsNotNone(recv_prev)
            self.assertIsInstance(recv_prev, tuple)

    def test_p2p_helper_no_recv(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            SendRecvMeta,
            _p2p_helper,
        )

        meta = SendRecvMeta()
        meta.recv_shape_message = [2, 3]
        meta.recv_dtype_message = 1
        meta.recv_stop_gradient = False
        meta.recv_key_message = None
        meta.send_shape_message = [2, 3]
        meta.send_dtype_message = 1
        meta.send_key_message = None

        mock_hcg = MagicMock()
        mock_hcg.get_model_parallel_group.return_value = MagicMock()
        mock_hcg.get_model_parallel_world_size.return_value = 1
        mock_hcg.get_model_parallel_rank.return_value = 0

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication._hcg",
                mock_hcg,
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication._batched_p2p_ops",
                return_value=None,
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication.allgather_partial"
            ),
        ):
            recv_prev, recv_next, reqs = _p2p_helper(
                tensor_send_next=None,
                tensor_send_prev=None,
                recv_prev=False,
                recv_next=False,
                send_recv_meta=meta,
                batch_p2p_comm=True,
            )
            self.assertIsNone(recv_prev)
            self.assertIsNone(recv_next)


if __name__ == "__main__":
    unittest.main()
