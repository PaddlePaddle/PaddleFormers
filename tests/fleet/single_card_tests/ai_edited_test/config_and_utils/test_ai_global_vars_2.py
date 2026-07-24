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


# Tests for src/paddleformers.fleet/training/global_vars.py
# Additional tests for get_args, set_args, destroy_global_vars,
# get_timers, _set_timers, set_global_variables, unset_global_variables,
# _ensure_var_is_initialized, _ensure_var_is_not_initialized

import unittest
from unittest import mock


class TestGetArgs(unittest.TestCase):
    """Tests for get_args function."""

    def setUp(self):
        """Reset global state before each test."""
        import paddleformers.fleet.training.global_vars as gv

        gv._GLOBAL_ARGS = None
        gv._GLOBAL_TIMERS = None

    def tearDown(self):
        """Clean up global state after each test."""
        import paddleformers.fleet.training.global_vars as gv

        gv._GLOBAL_ARGS = None
        gv._GLOBAL_TIMERS = None

    def test_get_args_not_initialized_raises(self):
        """Test get_args raises when args is not initialized."""
        from paddleformers.fleet.training.global_vars import get_args

        with self.assertRaises(AssertionError) as ctx:
            get_args()
        self.assertIn("not initialized", str(ctx.exception))

    def test_get_args_returns_set_value(self):
        """Test get_args returns the value set by set_args."""
        from paddleformers.fleet.training.global_vars import get_args, set_args

        mock_args = mock.MagicMock()
        set_args(mock_args)
        result = get_args()
        self.assertIs(result, mock_args)


class TestSetArgs(unittest.TestCase):
    """Tests for set_args function."""

    def setUp(self):
        import paddleformers.fleet.training.global_vars as gv

        gv._GLOBAL_ARGS = None

    def tearDown(self):
        import paddleformers.fleet.training.global_vars as gv

        gv._GLOBAL_ARGS = None

    def test_set_args_basic(self):
        """Test set_args sets the global args."""
        import paddleformers.fleet.training.global_vars as gv
        from paddleformers.fleet.training.global_vars import set_args

        mock_args = mock.MagicMock()
        set_args(mock_args)
        self.assertIs(gv._GLOBAL_ARGS, mock_args)

    def test_set_args_overwrite(self):
        """Test set_args can overwrite previous value."""
        import paddleformers.fleet.training.global_vars as gv
        from paddleformers.fleet.training.global_vars import set_args

        set_args(mock.MagicMock(name="first"))
        mock_args2 = mock.MagicMock(name="second")
        set_args(mock_args2)
        self.assertIs(gv._GLOBAL_ARGS, mock_args2)


class TestDestroyGlobalVars(unittest.TestCase):
    """Tests for destroy_global_vars function."""

    def setUp(self):
        import paddleformers.fleet.training.global_vars as gv

        gv._GLOBAL_ARGS = None
        gv._GLOBAL_TIMERS = None

    def tearDown(self):
        import paddleformers.fleet.training.global_vars as gv

        gv._GLOBAL_ARGS = None
        gv._GLOBAL_TIMERS = None

    def test_destroy_global_vars_clears_args(self):
        """Test destroy_global_vars sets _GLOBAL_ARGS to None."""
        import paddleformers.fleet.training.global_vars as gv
        from paddleformers.fleet.training.global_vars import destroy_global_vars

        gv._GLOBAL_ARGS = mock.MagicMock()
        gv._GLOBAL_TIMERS = mock.MagicMock()
        destroy_global_vars()
        self.assertIsNone(gv._GLOBAL_ARGS)
        self.assertIsNone(gv._GLOBAL_TIMERS)

    def test_destroy_when_none(self):
        """Test destroy_global_vars when already None."""
        from paddleformers.fleet.training.global_vars import destroy_global_vars

        destroy_global_vars()  # Should not raise

    def test_get_args_after_destroy_raises(self):
        """Test get_args raises after destroy."""
        from paddleformers.fleet.training.global_vars import (
            destroy_global_vars,
            get_args,
            set_global_variables,
        )

        set_global_variables(mock.MagicMock())
        destroy_global_vars()
        with self.assertRaises(AssertionError):
            get_args()


class TestSetGlobalVariables(unittest.TestCase):
    """Tests for set_global_variables function."""

    def setUp(self):
        import paddleformers.fleet.training.global_vars as gv

        gv._GLOBAL_ARGS = None
        gv._GLOBAL_TIMERS = None

    def tearDown(self):
        import paddleformers.fleet.training.global_vars as gv

        gv._GLOBAL_ARGS = None
        gv._GLOBAL_TIMERS = None

    def test_set_global_variables_none_raises(self):
        """Test set_global_variables raises when args is None."""
        from paddleformers.fleet.training.global_vars import (
            set_global_variables,
        )

        with self.assertRaises(AssertionError):
            set_global_variables(None)

    def test_set_global_variables_already_initialized_raises(self):
        """Test set_global_variables raises when already initialized."""
        from paddleformers.fleet.training.global_vars import (
            set_global_variables,
        )

        set_global_variables(mock.MagicMock())
        with self.assertRaises(AssertionError) as ctx:
            set_global_variables(mock.MagicMock())
        self.assertIn("already initialized", str(ctx.exception))

    def test_set_global_variables_sets_args_and_timers(self):
        """Test set_global_variables sets both args and timers."""
        import paddleformers.fleet.training.global_vars as gv
        from paddleformers.fleet.training.global_vars import (
            get_args,
            get_timers,
            set_global_variables,
        )

        mock_args = mock.MagicMock()
        set_global_variables(mock_args)
        self.assertIsNotNone(gv._GLOBAL_ARGS)
        self.assertIsNotNone(gv._GLOBAL_TIMERS)
        self.assertIs(get_args(), mock_args)
        self.assertIsInstance(get_timers(), type(gv._GLOBAL_TIMERS))


class TestUnsetGlobalVariables(unittest.TestCase):
    """Tests for unset_global_variables function."""

    def setUp(self):
        import paddleformers.fleet.training.global_vars as gv

        gv._GLOBAL_ARGS = None
        gv._GLOBAL_TIMERS = None

    def tearDown(self):
        import paddleformers.fleet.training.global_vars as gv

        gv._GLOBAL_ARGS = None
        gv._GLOBAL_TIMERS = None

    def test_unset_clears_both(self):
        """Test unset_global_variables clears both args and timers."""
        import paddleformers.fleet.training.global_vars as gv
        from paddleformers.fleet.training.global_vars import (
            unset_global_variables,
        )

        gv._GLOBAL_ARGS = mock.MagicMock()
        gv._GLOBAL_TIMERS = mock.MagicMock()
        unset_global_variables()
        self.assertIsNone(gv._GLOBAL_ARGS)
        self.assertIsNone(gv._GLOBAL_TIMERS)

    def test_unset_when_none(self):
        """Test unset when already None."""
        from paddleformers.fleet.training.global_vars import (
            unset_global_variables,
        )

        unset_global_variables()  # Should not raise


class TestEnsureVarHelpers(unittest.TestCase):
    """Tests for _ensure_var_is_initialized and _ensure_var_is_not_initialized."""

    def test_ensure_initialized_passes_when_not_none(self):
        """Test _ensure_var_is_initialized passes when var is not None."""
        from paddleformers.fleet.training.global_vars import (
            _ensure_var_is_initialized,
        )

        _ensure_var_is_initialized("some_value", "test")

    def test_ensure_initialized_raises_when_none(self):
        """Test _ensure_var_is_initialized raises when var is None."""
        from paddleformers.fleet.training.global_vars import (
            _ensure_var_is_initialized,
        )

        with self.assertRaises(AssertionError):
            _ensure_var_is_initialized(None, "test_var")

    def test_ensure_not_initialized_passes_when_none(self):
        """Test _ensure_var_is_not_initialized passes when var is None."""
        from paddleformers.fleet.training.global_vars import (
            _ensure_var_is_not_initialized,
        )

        _ensure_var_is_not_initialized(None, "test")

    def test_ensure_not_initialized_raises_when_not_none(self):
        """Test _ensure_var_is_not_initialized raises when var is not None."""
        from paddleformers.fleet.training.global_vars import (
            _ensure_var_is_not_initialized,
        )

        with self.assertRaises(AssertionError) as ctx:
            _ensure_var_is_not_initialized("value", "test_var")
        self.assertIn("already initialized", str(ctx.exception))


class TestGetTimers(unittest.TestCase):
    """Tests for get_timers function."""

    def setUp(self):
        import paddleformers.fleet.training.global_vars as gv

        gv._GLOBAL_ARGS = None
        gv._GLOBAL_TIMERS = None

    def tearDown(self):
        import paddleformers.fleet.training.global_vars as gv

        gv._GLOBAL_ARGS = None
        gv._GLOBAL_TIMERS = None

    def test_get_timers_not_initialized_raises(self):
        """Test get_timers raises when timers not initialized."""
        from paddleformers.fleet.training.global_vars import get_timers

        with self.assertRaises(AssertionError):
            get_timers()

    def test_get_timers_returns_timers_instance(self):
        """Test get_timers returns a Timers instance after initialization."""
        from paddleformers.fleet.training.global_vars import (
            get_timers,
            set_global_variables,
        )

        set_global_variables(mock.MagicMock())
        timers = get_timers()
        self.assertIsNotNone(timers)


class TestSetTimers(unittest.TestCase):
    """Tests for _set_timers function."""

    def setUp(self):
        import paddleformers.fleet.training.global_vars as gv

        gv._GLOBAL_ARGS = None
        gv._GLOBAL_TIMERS = None

    def tearDown(self):
        import paddleformers.fleet.training.global_vars as gv

        gv._GLOBAL_ARGS = None
        gv._GLOBAL_TIMERS = None

    def test_set_timers_already_initialized_raises(self):
        """Test _set_timers raises when already initialized."""
        from paddleformers.fleet.training.global_vars import _set_timers

        _set_timers()
        with self.assertRaises(AssertionError):
            _set_timers()

    def test_set_timers_creates_timers(self):
        """Test _set_timers creates a new Timers instance."""
        import paddleformers.fleet.training.global_vars as gv
        from paddleformers.fleet.training.global_vars import _set_timers

        _set_timers()
        self.assertIsNotNone(gv._GLOBAL_TIMERS)


if __name__ == "__main__":
    unittest.main()
