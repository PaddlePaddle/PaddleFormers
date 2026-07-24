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

import paddle

from paddleformers.fleet.models.common.language_loss.language_loss import (
    subbatch,
)


class TestSubbatchWithKwargs(unittest.TestCase):
    """Tests for subbatch function with keyword arguments."""

    def test_subbatch_has_same_arg_idx_param(self):
        """subbatch should accept same_arg_idx parameter."""

        import inspect

        sig = inspect.signature(subbatch)
        self.assertIn("same_arg_idx", sig.parameters)

    def test_subbatch_has_use_recompute_param(self):
        """subbatch should accept use_recompute parameter."""

        import inspect

        sig = inspect.signature(subbatch)
        self.assertIn("use_recompute", sig.parameters)


class TestLanguageLossForwardImpl(unittest.TestCase):
    """Tests for LanguageLoss forward_impl basic behavior."""

    def test_language_loss_has_forward_impl(self):
        """LanguageLoss should have forward_impl method."""
        from paddleformers.fleet.models.common.language_loss.language_loss import (
            LanguageLoss,
        )

        self.assertTrue(hasattr(LanguageLoss, "forward_impl"))


class TestMainLanguageLossForward(unittest.TestCase):
    """Tests for MainLanguageLoss forward method."""

    def test_forward_requires_mtp_config(self):
        """MainLanguageLoss.forward should assert num_nextn_predict_layers > 0."""
        from paddleformers.fleet.models.common.language_loss.language_loss import (
            MainLanguageLoss,
        )

        with patch.object(
            MainLanguageLoss, "__init__", lambda self, *a, **kw: None
        ):
            loss = MainLanguageLoss.__new__(MainLanguageLoss)
            loss.config = MagicMock()
            loss.config.num_nextn_predict_layers = 0

            with self.assertRaises(AssertionError):
                loss.forward({}, paddle.randint(0, 10, [2, 4]))


class TestMTPLanguageLossForward(unittest.TestCase):
    """Tests for MTPLanguageLoss forward method."""

    def test_forward_requires_mtp_logits(self):
        """MTPLanguageLoss.forward should assert mtp_logits is provided."""
        from paddleformers.fleet.models.common.language_loss.language_loss import (
            MTPLanguageLoss,
        )

        with patch.object(
            MTPLanguageLoss, "__init__", lambda self, *a, **kw: None
        ):
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


if __name__ == "__main__":
    unittest.main()
