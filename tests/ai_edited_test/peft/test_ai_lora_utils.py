# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

import unittest
from contextlib import nullcontext
from unittest.mock import MagicMock, patch


class TestRngCtx(unittest.TestCase):
    """Tests for paddleformers.peft.lora.utils.rng_ctx function."""

    def test_rng_ctx_with_mp_and_dynamic_mode(self):
        """Test rng_ctx returns rng_state when is_mp=True and in_dynamic_mode=True."""
        from paddleformers.peft.lora.utils import rng_ctx

        mock_tracker = MagicMock()
        mock_rng_state = MagicMock()
        mock_tracker.rng_state.return_value = mock_rng_state

        with patch("paddleformers.peft.lora.utils.get_rng_state_tracker", return_value=mock_tracker):
            ctx = rng_ctx(is_mp=True, in_dynamic_mode=True)
            # Should return the rng_state context manager, not nullcontext
            self.assertEqual(ctx, mock_rng_state)
            mock_tracker.rng_state.assert_called_once()

    def test_rng_ctx_without_mp(self):
        """Test rng_ctx returns nullcontext when is_mp=False."""
        from paddleformers.peft.lora.utils import rng_ctx

        with patch("paddleformers.peft.lora.utils.get_rng_state_tracker") as mock_get_tracker:
            ctx = rng_ctx(is_mp=False, in_dynamic_mode=True)
            self.assertIsInstance(ctx, nullcontext)
            mock_get_tracker.assert_not_called()

    def test_rng_ctx_without_dynamic_mode(self):
        """Test rng_ctx returns nullcontext when in_dynamic_mode=False."""
        from paddleformers.peft.lora.utils import rng_ctx

        with patch("paddleformers.peft.lora.utils.get_rng_state_tracker") as mock_get_tracker:
            ctx = rng_ctx(is_mp=True, in_dynamic_mode=False)
            self.assertIsInstance(ctx, nullcontext)
            mock_get_tracker.assert_not_called()

    def test_rng_ctx_both_false(self):
        """Test rng_ctx returns nullcontext when both flags are False."""
        from paddleformers.peft.lora.utils import rng_ctx

        with patch("paddleformers.peft.lora.utils.get_rng_state_tracker") as mock_get_tracker:
            ctx = rng_ctx(is_mp=False, in_dynamic_mode=False)
            self.assertIsInstance(ctx, nullcontext)
            mock_get_tracker.assert_not_called()

    def test_rng_ctx_with_mp_dynamic_true_context_manager(self):
        """Test rng_ctx context manager works correctly when both flags are True."""
        from paddleformers.peft.lora.utils import rng_ctx

        mock_tracker = MagicMock()
        mock_rng_state = MagicMock()
        mock_tracker.rng_state.return_value = mock_rng_state
        mock_rng_state.__enter__ = MagicMock(return_value=None)
        mock_rng_state.__exit__ = MagicMock(return_value=False)

        with patch("paddleformers.peft.lora.utils.get_rng_state_tracker", return_value=mock_tracker):
            ctx = rng_ctx(is_mp=True, in_dynamic_mode=True)
            with ctx:
                pass
            mock_rng_state.__enter__.assert_called_once()
            mock_rng_state.__exit__.assert_called_once()

    def test_rng_ctx_nullcontext_usage(self):
        """Test rng_ctx nullcontext can be used as a context manager."""
        from paddleformers.peft.lora.utils import rng_ctx

        with patch("paddleformers.peft.lora.utils.get_rng_state_tracker"):
            ctx = rng_ctx(is_mp=False, in_dynamic_mode=False)
            with ctx:
                # Should not raise
                pass


if __name__ == "__main__":
    unittest.main()
