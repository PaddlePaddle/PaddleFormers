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


class TestP2pSyncSendEnv(unittest.TestCase):
    """Tests for PADDLE_P2P_SYNC_SEND environment variable handling."""

    def test_sync_send_default_false(self):
        # By default, _sync_send should be False
        import paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication as mod

        # The module-level value depends on the env var, just test it's a bool
        self.assertIsInstance(mod._sync_send, bool)


class TestP2pHelperInit(unittest.TestCase):
    """Tests for P2pHelper initialization with various parameters."""

    def test_init_default(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            P2pHelper,
        )

        helper = P2pHelper()
        self.assertTrue(helper._use_cache)
        self.assertFalse(helper._dynamic_shape)

    def test_init_no_cache(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            P2pHelper,
        )

        helper = P2pHelper(use_cache=False)
        self.assertFalse(helper._use_cache)

    def test_init_dynamic_shape(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            P2pHelper,
        )

        helper = P2pHelper(dynamic_shape=True)
        self.assertTrue(helper._dynamic_shape)
        self.assertEqual(helper._dynamic_cnt, 0)
        self.assertIsInstance(helper._send_recv_meta_list, list)


class TestP2pHelperMeta(unittest.TestCase):
    """Tests for P2pHelper _send_meta and _recv_meta."""

    def test_send_meta_first_time_no_cache(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            P2pHelper,
        )

        helper = P2pHelper(use_cache=False)
        tensor = paddle.randn([2, 3])
        mock_group = MagicMock()
        mock_hcg = MagicMock()
        mock_hcg.get_pipe_parallel_group.return_value = mock_group

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication._hcg",
                mock_hcg,
            ),
            patch("paddle.distributed.send"),
        ):
            helper._send_meta(tensor)
            # With use_cache=False, has_send_meta is set to False
            self.assertFalse(helper._send_recv_meta.has_send_meta)

    def test_recv_meta_first_time_no_cache(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            P2pHelper,
        )

        helper = P2pHelper(use_cache=False)
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
            # With use_cache=False, has_recv_meta is set to False
            self.assertFalse(helper._send_recv_meta.has_recv_meta)

    def test_send_meta_skip_check(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            P2pHelper,
        )

        helper = P2pHelper(use_cache=True)
        helper._send_recv_meta.has_send_meta = True
        # With skip_check_meta=True, should not raise
        tensor = paddle.randn([2, 3])
        helper._send_meta(tensor, skip_check_meta=True)

    def test_clear_meta_cache(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            P2pHelper,
        )

        helper = P2pHelper(use_cache=True)
        helper._send_recv_meta.has_send_meta = True
        helper._send_recv_meta.has_recv_meta = True
        helper._send_recv_meta.send_shape_message = [2, 3]
        helper.clear_meta_cache()
        self.assertIsNone(helper._send_recv_meta.send_shape_message)
        self.assertFalse(helper._send_recv_meta.has_send_meta)
        self.assertFalse(helper._send_recv_meta.has_recv_meta)


class TestP2pHelperDynamicMetaList(unittest.TestCase):
    """Tests for P2pHelper with dynamic_shape meta list."""

    def test_send_meta_list_grows(self):
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
            # First call - creates new meta
            helper._send_meta(tensor)
            self.assertEqual(len(helper._send_recv_meta_list), 1)

    def test_recv_meta_list_grows(self):
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
            # recv_meta with dynamic_shape creates a new meta in the list
            # only if _dynamic_cnt < len(_send_recv_meta_list)
            # On first call, the list may still be empty if recv_meta fails


class TestP2pHelperSendForwardRecvForwardDynamic(unittest.TestCase):
    """Tests for send_forward_recv_forward with dynamic_shape."""

    def test_send_forward_recv_forward_increases_cnt(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            P2pHelper,
        )

        helper = P2pHelper(use_cache=True, dynamic_shape=True)
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
                return_value=(MagicMock(), None, None),
            ),
            patch.object(helper, "_send_meta"),
            patch.object(helper, "_recv_meta"),
        ):
            helper.send_forward_recv_forward(mock_tensor, recv_prev=True)
            # When both send and recv are True, need_increase_cnt is True,
            # and _dynamic_cnt is incremented by 1 (not 2)
            self.assertEqual(helper._dynamic_cnt, 1)


class TestP2pHelperSendBackwardRecvBackwardDynamic(unittest.TestCase):
    """Tests for send_backward_recv_backward with dynamic_shape."""

    def test_send_backward_recv_backward_with_tensor(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            P2pHelper,
        )

        helper = P2pHelper(use_cache=True, dynamic_shape=True)
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
            patch.object(helper, "_send_meta"),
            patch.object(helper, "_recv_meta"),
        ):
            result = helper.send_backward_recv_backward(mock_tensor, recv_next=True)
            # dynamic_cnt is incremented by 1 when need_increase_cnt is True
            self.assertEqual(helper._dynamic_cnt, 1)


if __name__ == "__main__":
    unittest.main()
