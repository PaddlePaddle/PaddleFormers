# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import gc
import unittest
from unittest.mock import MagicMock

from paddleformers.cli.train.ernie_pretrain.src.callbacks.gc_callback import GCCallback


class TestGCCallback(unittest.TestCase):
    """Tests for GCCallback class."""

    def setUp(self):
        """Set up test fixtures."""
        self.callback = GCCallback()

    def test_on_train_begin_with_gc_interval_gt_zero(self):
        """Test on_train_begin disables gc when gc_interval > 0."""
        # Save current gc state
        gc_was_enabled = gc.isenabled()

        args = MagicMock()
        args.gc_interval = 10
        state = MagicMock()
        control = MagicMock()

        self.callback.on_train_begin(args, state, control)

        # gc should be disabled
        self.assertFalse(gc.isenabled())

        # Restore gc state
        if gc_was_enabled:
            gc.enable()

    def test_on_train_begin_with_gc_interval_zero(self):
        """Test on_train_begin does not disable gc when gc_interval == 0."""
        gc.enable()  # Make sure gc is enabled

        args = MagicMock()
        args.gc_interval = 0
        state = MagicMock()
        control = MagicMock()

        self.callback.on_train_begin(args, state, control)

        # gc should still be enabled
        self.assertTrue(gc.isenabled())

    def test_on_step_end_with_gc_interval_gt_zero_and_matching_step(self):
        """Test on_step_end triggers gc.collect when step matches interval."""
        gc.enable()  # Re-enable first
        args = MagicMock()
        args.gc_interval = 10

        state = MagicMock()
        state.global_step = 10

        control = MagicMock()

        # Should not raise
        self.callback.on_step_end(args, state, control)

    def test_on_step_end_with_gc_interval_zero(self):
        """Test on_step_end does not trigger gc when gc_interval == 0."""
        args = MagicMock()
        args.gc_interval = 0

        state = MagicMock()
        state.global_step = 10

        control = MagicMock()

        # Should not raise and should not call gc.collect
        self.callback.on_step_end(args, state, control)

    def test_on_step_end_non_matching_step(self):
        """Test on_step_end does not trigger gc on non-matching steps."""
        args = MagicMock()
        args.gc_interval = 10

        state = MagicMock()
        state.global_step = 5

        control = MagicMock()

        # Should not raise
        self.callback.on_step_end(args, state, control)

    def test_inherits_from_trainer_callback(self):
        """Test that GCCallback has TrainerCallback methods."""
        self.assertTrue(hasattr(self.callback, "on_train_begin"))
        self.assertTrue(hasattr(self.callback, "on_step_end"))


if __name__ == "__main__":
    unittest.main()
