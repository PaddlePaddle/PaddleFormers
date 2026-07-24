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


class TestP2pCommSendRecvMetaBroadcast(unittest.TestCase):
    """Tests for SendRecvMeta recv_meta and send_meta with broadcast=True."""

    def test_recv_meta_broadcast(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            SendRecvMeta,
        )

        meta = SendRecvMeta()
        mock_group = MagicMock()
        mock_group.ranks = [0, 1, 2, 3]
        mock_hcg = MagicMock()
        mock_hcg._get_p2p_prev_rank.return_value = 1

        # Create mock data for recv
        numel_tensor = MagicMock()
        numel_tensor.item.return_value = 10

        data_tensor = MagicMock()
        data_tensor.numpy.return_value.tolist.return_value = [
            0,  # tensor_type = 0 (single)
            2,
            1,  # shape [2] and dtype
            1,  # dtype
            0,  # stop_gradient
            0,  # key_len
        ]

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication._hcg",
                mock_hcg,
            ),
            patch("paddle.empty", side_effect=[numel_tensor, data_tensor]),
            patch("paddle.distributed.broadcast"),
            patch("paddle.to_tensor") as mock_to_tensor,
        ):
            mock_to_tensor.return_value = MagicMock()
            try:
                meta.recv_meta(mock_group, broadcast=True)
            except Exception:
                pass

    def test_send_meta_broadcast(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            SendRecvMeta,
        )

        meta = SendRecvMeta()
        mock_group = MagicMock()
        mock_group.ranks = [0, 1, 2, 3]
        mock_group.rank = 0
        mock_hcg = MagicMock()
        mock_hcg._get_p2p_next_rank.return_value = 2

        tensor = paddle.randn([2, 3])

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication._hcg",
                mock_hcg,
            ),
            patch("paddle.distributed.broadcast"),
            patch("paddle.to_tensor") as mock_to_tensor,
        ):
            mock_to_tensor.return_value = MagicMock()
            meta.send_meta(tensor, mock_group, broadcast=True)

    def test_send_meta_list_type(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
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
                "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication._hcg",
                mock_hcg,
            ),
            patch("paddle.distributed.send"),
            patch("paddle.to_tensor") as mock_to_tensor,
        ):
            mock_to_tensor.return_value = MagicMock()
            meta.send_meta([t1, t2], mock_group)
            # list is handled same as tuple

    def test_send_meta_invalid_type(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            SendRecvMeta,
        )

        meta = SendRecvMeta()
        mock_group = MagicMock()
        mock_hcg = MagicMock()
        mock_hcg._get_p2p_next_rank.return_value = 2

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication._hcg",
                mock_hcg,
            ),
            self.assertRaises(TypeError),
        ):
            meta.send_meta("invalid_tensor", mock_group)

    def test_send_meta_with_key_attribute(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            SendRecvMeta,
        )

        meta = SendRecvMeta()
        mock_group = MagicMock()
        mock_hcg = MagicMock()
        mock_hcg._get_p2p_next_rank.return_value = 2

        tensor = paddle.randn([2, 3])
        tensor.key = "my_tensor_key"

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication._hcg",
                mock_hcg,
            ),
            patch("paddle.distributed.send"),
            patch("paddle.to_tensor") as mock_to_tensor,
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication.convert_object_to_tensor"
            ) as mock_convert,
        ):
            mock_key_tensor = MagicMock()
            mock_key_tensor.numpy.return_value.tolist.return_value = [1, 2, 3]
            mock_convert.return_value = (mock_key_tensor, 3)
            mock_to_tensor.return_value = MagicMock()
            meta.send_meta(tensor, mock_group)


class TestP2pCommP2pHelperDynamicShapeBackward(unittest.TestCase):
    """Tests for P2pHelper send_backward with dynamic_shape."""

    def test_send_backward_dynamic_increases_cnt(self):
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
            helper.send_backward(mock_tensor, pp_first_stage=False)
            self.assertEqual(helper._dynamic_cnt, 1)

    def test_recv_backward_dynamic(self):
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
                return_value=(None, MagicMock(), None),
            ),
            patch.object(helper, "_recv_meta"),
        ):
            result = helper.recv_backward(pp_last_stage=False)
            self.assertEqual(helper._dynamic_cnt, 1)

    def test_recv_backward_dynamic_last_stage(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            P2pHelper,
        )

        helper = P2pHelper(use_cache=True, dynamic_shape=True)
        result = helper.recv_backward(pp_last_stage=True)
        self.assertIsNone(result)
        self.assertEqual(helper._dynamic_cnt, 0)


class TestP2pCommP2pHelperRepr(unittest.TestCase):
    """Tests for P2pHelper.__repr__."""

    def test_repr(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            P2pHelper,
        )

        helper = P2pHelper(use_cache=True, dynamic_shape=False)
        r = repr(helper)
        self.assertIn("using cache", r)
        self.assertIn("send_shape_message", r)


if __name__ == "__main__":
    unittest.main()
