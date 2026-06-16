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


class TestP2pDynamicShapeP2pHelper(unittest.TestCase):
    """Tests for P2pHelper with dynamic_shape=True in p2p_communication."""

    def test_init_dynamic_shape(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            P2pHelper,
        )

        helper = P2pHelper(use_cache=True, dynamic_shape=True)
        self.assertTrue(helper._dynamic_shape)
        self.assertEqual(helper._dynamic_cnt, 0)
        self.assertIsInstance(helper._send_recv_meta_list, list)

    def test_send_meta_dynamic_shape_first_call(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            P2pHelper,
        )

        helper = P2pHelper(use_cache=True, dynamic_shape=True)
        mock_group = MagicMock()
        mock_hcg = MagicMock()
        mock_hcg.get_pipe_parallel_group.return_value = mock_group
        tensor = paddle.randn([2, 3])

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication._hcg",
                mock_hcg,
            ),
            patch("paddle.distributed.send"),
        ):
            helper._send_meta(tensor)
            self.assertEqual(len(helper._send_recv_meta_list), 1)
            self.assertEqual(helper._dynamic_cnt, 0)

    def test_send_meta_dynamic_shape_second_call(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            P2pHelper,
        )

        helper = P2pHelper(use_cache=True, dynamic_shape=True)
        mock_group = MagicMock()
        mock_hcg = MagicMock()
        mock_hcg.get_pipe_parallel_group.return_value = mock_group
        tensor = paddle.randn([2, 3])

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication._hcg",
                mock_hcg,
            ),
            patch("paddle.distributed.send"),
        ):
            # First call creates meta
            helper._send_meta(tensor)
            # Simulate incrementing dynamic_cnt
            helper._dynamic_cnt = 1
            # Second call with same shape should check
            helper._send_meta(tensor, skip_check_meta=False)

    def test_recv_meta_dynamic_shape_first_call(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            P2pHelper,
        )

        helper = P2pHelper(use_cache=True, dynamic_shape=True)
        mock_group = MagicMock()
        mock_hcg = MagicMock()
        mock_hcg.get_pipe_parallel_group.return_value = mock_group

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication._hcg",
                mock_hcg,
            ),
            patch("paddle.distributed.recv"),
        ):
            try:
                helper._recv_meta()
            except Exception:
                pass

    def test_clear_meta_cache(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            P2pHelper,
        )

        helper = P2pHelper(use_cache=True, dynamic_shape=True)
        helper._send_recv_meta.has_send_meta = True
        helper.clear_meta_cache()
        self.assertFalse(helper._send_recv_meta.has_send_meta)

    def test_recv_forward_dynamic_increases_cnt(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            P2pHelper,
        )

        helper = P2pHelper(use_cache=True, dynamic_shape=True)
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
            patch.object(helper, "_recv_meta"),
        ):
            helper.recv_forward(pp_first_stage=False)
            self.assertEqual(helper._dynamic_cnt, 1)

    def test_send_forward_dynamic_increases_cnt(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            P2pHelper,
        )

        helper = P2pHelper(use_cache=True, dynamic_shape=True)
        helper._send_recv_meta.has_send_meta = True

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
            patch.object(helper, "_send_meta"),
        ):
            helper.send_forward(mock_tensor, pp_last_stage=False)
            self.assertEqual(helper._dynamic_cnt, 1)

    def test_send_forward_recv_forward_dynamic_increases_cnt(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            P2pHelper,
        )

        helper = P2pHelper(use_cache=True, dynamic_shape=True)
        helper._send_recv_meta.has_send_meta = True
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
            patch.object(helper, "_send_meta"),
            patch.object(helper, "_recv_meta"),
        ):
            helper.send_forward_recv_forward(
                MagicMock(), recv_prev=True, overlap_p2p_comm=False
            )
            self.assertEqual(helper._dynamic_cnt, 1)

    def test_send_forward_recv_backward_dynamic_raises(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            P2pHelper,
        )

        helper = P2pHelper(use_cache=True, dynamic_shape=True)
        with self.assertRaises(AssertionError):
            helper.send_forward_recv_backward(MagicMock(), pp_last_stage=False)

    def test_send_backward_recv_forward_dynamic_raises(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            P2pHelper,
        )

        helper = P2pHelper(use_cache=True, dynamic_shape=True)
        with self.assertRaises(AssertionError):
            helper.send_backward_recv_forward(MagicMock(), pp_first_stage=False)

    def test_send_forward_backward_recv_dynamic_raises(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            P2pHelper,
        )

        helper = P2pHelper(use_cache=True, dynamic_shape=True)
        with self.assertRaises(AssertionError):
            helper.send_forward_backward_recv_forward_backward(
                MagicMock(), MagicMock(), recv_prev=False, recv_next=False
            )


class TestP2pHelperOverlapP2pComm(unittest.TestCase):
    """Tests for P2pHelper with overlap_p2p_comm."""

    def test_send_forward_recv_forward_overlap(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            P2pHelper,
        )

        helper = P2pHelper(use_cache=True, dynamic_shape=False)
        helper._send_recv_meta.has_send_meta = True
        helper._send_recv_meta.has_recv_meta = True

        mock_hcg = MagicMock()
        mock_tensor = MagicMock()
        mock_handle = MagicMock()

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication._hcg",
                mock_hcg,
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication._p2p_helper",
                return_value=(MagicMock(), None, mock_handle),
            ),
        ):
            result = helper.send_forward_recv_forward(
                mock_tensor, recv_prev=True, overlap_p2p_comm=True
            )
            # When overlap, returns tuple of (input_tensor, wait_handles)
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 2)

    def test_send_backward_recv_backward_overlap(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            P2pHelper,
        )

        helper = P2pHelper(use_cache=True, dynamic_shape=False)

        mock_hcg = MagicMock()
        mock_tensor = MagicMock()
        mock_handle = MagicMock()

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication._hcg",
                mock_hcg,
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication._p2p_helper",
                return_value=(None, MagicMock(), mock_handle),
            ),
        ):
            result = helper.send_backward_recv_backward(
                mock_tensor, recv_next=True, overlap_p2p_comm=True
            )
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
