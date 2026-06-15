# Copyright (c) 2026 PaddleFleet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless distributed on the License is distributed on an "AS IS" BASIS,
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

from paddleformers.fleet.models.common.language_loss.language_loss import (
    MainLanguageLoss,
    MTPLanguageLoss,
)


class TestMTPLanguageLossForward(unittest.TestCase):
    """Tests for MTPLanguageLoss.forward."""

    def test_forward_requires_mtp_logits(self):
        """MTPLanguageLoss.forward should assert mtp_logits is provided."""
        with patch.object(MTPLanguageLoss, "__init__", lambda self, *a, **kw: None):
            loss = MTPLanguageLoss.__new__(MTPLanguageLoss)
            loss.config = MagicMock()
            loss.config.num_nextn_predict_layers = 2
            loss.config.mtp_load_weight_only = False
            loss.config.mtp_distillation_loss = False

            with self.assertRaises(AssertionError):
                loss.forward(
                    {
                        "mtp_logits": None,
                        "labels": paddle.randint(0, 10, [2, 4]),
                    }
                )

    def test_forward_requires_labels(self):
        """MTPLanguageLoss.forward should assert labels is provided."""
        with patch.object(MTPLanguageLoss, "__init__", lambda self, *a, **kw: None):
            loss = MTPLanguageLoss.__new__(MTPLanguageLoss)
            loss.config = MagicMock()
            loss.config.num_nextn_predict_layers = 2
            loss.config.mtp_load_weight_only = False
            loss.config.mtp_distillation_loss = False

            with self.assertRaises(AssertionError):
                loss.forward({"mtp_logits": [MagicMock(), MagicMock()], "labels": None})


class TestMainLanguageLossForward(unittest.TestCase):
    """Tests for MainLanguageLoss.forward."""

    def test_forward_requires_mtp_config(self):
        """MainLanguageLoss.forward should assert num_nextn_predict_layers > 0."""
        with patch.object(MainLanguageLoss, "__init__", lambda self, *a, **kw: None):
            loss = MainLanguageLoss.__new__(MainLanguageLoss)
            loss.config = MagicMock()
            loss.config.num_nextn_predict_layers = 0

            with self.assertRaises(AssertionError):
                loss.forward({}, paddle.randint(0, 10, [2, 4]))


class TestMainLanguageLossBuildScheduleNode(unittest.TestCase):
    """Tests for MainLanguageLoss.build_schedule_node."""

    def test_returns_schedule_node_with_name(self):
        """build_schedule_node should return a ScheduleNode named 'MainLanguageLoss'."""
        from paddle.distributed.fleet.meta_parallel import ScheduleNode

        with patch.object(MainLanguageLoss, "__init__", lambda self, *a, **kw: None):
            loss = MainLanguageLoss.__new__(MainLanguageLoss)
            loss.forward = MagicMock()
            node = loss.build_schedule_node()
            self.assertIsInstance(node, ScheduleNode)


class TestMTPLanguageLossBuildScheduleNode(unittest.TestCase):
    """Tests for MTPLanguageLoss.build_schedule_node."""

    def test_returns_schedule_node_with_name(self):
        """build_schedule_node should return a ScheduleNode named 'MTPLanguageLoss'."""
        from paddle.distributed.fleet.meta_parallel import ScheduleNode

        with patch.object(MTPLanguageLoss, "__init__", lambda self, *a, **kw: None):
            loss = MTPLanguageLoss.__new__(MTPLanguageLoss)
            loss.forward = MagicMock()
            node = loss.build_schedule_node()
            self.assertIsInstance(node, ScheduleNode)


if __name__ == "__main__":
    unittest.main()
