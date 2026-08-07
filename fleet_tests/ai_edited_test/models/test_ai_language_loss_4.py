# Copyright (c) 2026 PaddleFleet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless distributed on applicable law or agreed to in writing, software
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

from paddle.distributed.fleet.meta_parallel import ScheduleNode

from paddleformers.fleet.models.common.language_loss.language_loss import (
    LanguageLoss,
    MainLanguageLoss,
    MTPLanguageLoss,
)


class TestLanguageLossInit(unittest.TestCase):
    """Tests for LanguageLoss init."""

    def test_language_loss_is_class(self):
        """LanguageLoss should be importable."""
        self.assertTrue(LanguageLoss is not None)

    def test_language_loss_has_forward(self):
        """LanguageLoss should have a forward method."""
        self.assertTrue(hasattr(LanguageLoss, "forward"))


class TestLanguageLossBuildScheduleNode(unittest.TestCase):
    """Tests for LanguageLoss.build_schedule_node."""

    def test_main_build_schedule_node(self):
        """MainLanguageLoss.build_schedule_node should return a ScheduleNode."""
        with patch.object(
            MainLanguageLoss, "__init__", lambda self, *a, **kw: None
        ):
            loss = MainLanguageLoss.__new__(MainLanguageLoss)
            loss.forward = MagicMock()
            node = loss.build_schedule_node()
            self.assertIsInstance(node, ScheduleNode)

    def test_mtp_build_schedule_node(self):
        """MTPLanguageLoss.build_schedule_node should return a ScheduleNode."""
        with patch.object(
            MTPLanguageLoss, "__init__", lambda self, *a, **kw: None
        ):
            loss = MTPLanguageLoss.__new__(MTPLanguageLoss)
            loss.forward = MagicMock()
            node = loss.build_schedule_node()
            self.assertIsInstance(node, ScheduleNode)


class TestSubbatchBasic(unittest.TestCase):
    """Tests for subbatch function basic behavior."""

    def test_subbatch_is_callable(self):
        """subbatch should be callable."""
        from paddleformers.fleet.models.common.language_loss.language_loss import (
            subbatch,
        )

        self.assertTrue(callable(subbatch))

    def test_subbatch_has_expected_params(self):
        """subbatch should have expected parameter names."""
        import inspect

        from paddleformers.fleet.models.common.language_loss.language_loss import (
            subbatch,
        )

        sig = inspect.signature(subbatch)
        params = list(sig.parameters.keys())
        self.assertIn("f", params)
        self.assertIn("arg_idx", params)
        self.assertIn("axis", params)
        self.assertIn("bs", params)


class TestMtpLossTracker(unittest.TestCase):
    """Tests for mtp_loss_tracker attribute."""

    def test_mtp_loss_tracker_exists(self):
        """MTPLanguageLoss should have mtp_loss_tracker attribute."""
        self.assertTrue(hasattr(MTPLanguageLoss, "mtp_loss_tracker"))


if __name__ == "__main__":
    unittest.main()
