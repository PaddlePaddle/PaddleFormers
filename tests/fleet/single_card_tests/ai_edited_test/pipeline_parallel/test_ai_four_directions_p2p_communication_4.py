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


class TestFourDirectionsP2pHelperSyncRecv(unittest.TestCase):
    """Tests for P2pHelper recv_forward/recv_backward with sync_recv in four_directions."""

    def test_recv_forward_first_stage(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            P2pHelper,
        )

        helper = P2pHelper()
        result = helper.recv_forward(pp_first_stage=True, sync_recv=True)
        self.assertIsNone(result)

    def test_recv_backward_last_stage(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            P2pHelper,
        )

        helper = P2pHelper()
        result = helper.recv_backward(pp_last_stage=True, sync_recv=True)
        self.assertIsNone(result)

    def test_recv_forward_not_first_stage(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            P2pHelper,
        )

        helper = P2pHelper()
        helper._send_recv_meta.has_recv_meta = True

        mock_hcg = MagicMock()
        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._hcg",
                mock_hcg,
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._p2p_helper",
                return_value=(MagicMock(), None),
            ),
        ):
            result = helper.recv_forward(pp_first_stage=False, sync_recv=True)
            self.assertIsNotNone(result)


class TestFourDirectionsP2pHelperSendMethods(unittest.TestCase):
    """Tests for P2pHelper send methods in four_directions."""

    def test_send_forward_not_last_stage(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            P2pHelper,
        )

        helper = P2pHelper()
        helper._send_recv_meta.has_send_meta = True

        mock_hcg = MagicMock()
        mock_tensor = MagicMock()
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
            helper.send_forward(mock_tensor, pp_last_stage=False)

    def test_send_backward_not_first_stage(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            P2pHelper,
        )

        helper = P2pHelper()

        mock_hcg = MagicMock()
        mock_tensor = MagicMock()
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
            helper.send_backward(mock_tensor, pp_first_stage=False)


class TestFourDirectionsP2pHelperCombined(unittest.TestCase):
    """Tests for combined send/recv methods in four_directions."""

    def test_send_forward_recv_backward_not_last(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            P2pHelper,
        )

        helper = P2pHelper()
        mock_hcg = MagicMock()
        mock_tensor = MagicMock()
        mock_grad = MagicMock()

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._hcg",
                mock_hcg,
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._p2p_helper",
                return_value=(None, mock_grad),
            ),
        ):
            result = helper.send_forward_recv_backward(mock_tensor, pp_last_stage=False)
            self.assertEqual(result, mock_grad)

    def test_send_backward_recv_forward_not_first(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            P2pHelper,
        )

        helper = P2pHelper()
        mock_hcg = MagicMock()
        mock_tensor = MagicMock()
        mock_input = MagicMock()

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._hcg",
                mock_hcg,
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._p2p_helper",
                return_value=(mock_input, None),
            ),
        ):
            result = helper.send_backward_recv_forward(mock_tensor, pp_first_stage=False)
            self.assertEqual(result, mock_input)


class TestFourDirectionsP2pHelperTimers(unittest.TestCase):
    """Tests for P2pHelper with timers enabled in four_directions."""

    def test_recv_forward_with_timers(self):
        import paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication as mod
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
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

    def test_send_forward_with_timers(self):
        import paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication as mod
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            P2pHelper,
        )

        orig_timers = mod._timers
        try:
            mock_timer = MagicMock()
            mod._timers = mock_timer

            helper = P2pHelper()
            mock_tensor = MagicMock()
            helper.send_forward(mock_tensor, pp_last_stage=True)
            mock_timer.assert_called_with("send_forward")
        finally:
            mod._timers = orig_timers


class TestFourDirectionsPartialSendRecv(unittest.TestCase):
    """Tests for partial send/recv with valid partial in four_directions."""

    def test_send_partial_valid_partial(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            send_partial,
        )

        mock_hcg = MagicMock()
        mock_hcg._get_p2p_next_rank.return_value = 5

        mock_tensor = MagicMock()
        mock_tensor.shape = [4, 4]
        mock_group = MagicMock()
        mock_group.is_member.return_value = True

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._hcg",
                mock_hcg,
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._is_valid_send_recv_partial",
                return_value=True,
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication.framework"
            ) as mock_fw,
        ):
            mock_fw.in_dynamic_mode.return_value = True
            mock_group.process_group.send_partial_on_calc_stream.return_value = None
            send_partial(
                mock_tensor,
                dst=1,
                nranks=2,
                rank_id=0,
                group=mock_group,
                use_calc_stream=True,
            )

    def test_recv_partial_valid_partial(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication import (
            recv_partial,
        )

        mock_hcg = MagicMock()
        mock_hcg._get_p2p_prev_rank.return_value = 3

        mock_tensor = MagicMock()
        mock_tensor.shape = [4, 4]
        mock_group = MagicMock()
        mock_group.is_member.return_value = True

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._hcg",
                mock_hcg,
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication._is_valid_send_recv_partial",
                return_value=True,
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.four_directions_p2p_communication.framework"
            ) as mock_fw,
        ):
            mock_fw.in_dynamic_mode.return_value = True
            mock_group.process_group.recv_partial_on_calc_stream.return_value = None
            recv_partial(
                mock_tensor,
                src=0,
                nranks=2,
                rank_id=0,
                group=mock_group,
                use_calc_stream=True,
            )


if __name__ == "__main__":
    unittest.main()
