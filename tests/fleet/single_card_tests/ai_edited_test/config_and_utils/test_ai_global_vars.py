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
import os
import sys

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)


import unittest


class TestGlobalVars(unittest.TestCase):
    """Tests for global_vars in paddleformers.fleet.training.global_vars."""

    def setUp(self):
        """Clean up global state before each test."""
        import paddleformers.fleet.training.global_vars as gv

        gv._GLOBAL_ARGS = None
        gv._GLOBAL_TIMERS = None

    def tearDown(self):
        """Clean up global state after each test."""
        import paddleformers.fleet.training.global_vars as gv

        gv._GLOBAL_ARGS = None
        gv._GLOBAL_TIMERS = None

    def test_get_args_not_initialized_raises(self):
        from paddleformers.fleet.training.global_vars import get_args

        with self.assertRaises(AssertionError):
            get_args()

    def test_set_args_and_get_args(self):
        from paddleformers.fleet.training.global_vars import get_args, set_args

        mock_args = {"key": "value"}
        set_args(mock_args)
        result = get_args()
        self.assertEqual(result, mock_args)

    def test_get_timers_not_initialized_raises(self):
        from paddleformers.fleet.training.global_vars import get_timers

        with self.assertRaises(AssertionError):
            get_timers()

    def test_set_timers_and_get_timers(self):
        from paddleformers.fleet.training.global_vars import _set_timers, get_timers

        _set_timers()
        timers = get_timers()
        self.assertIsNotNone(timers)

    def test_set_timers_already_initialized_raises(self):
        from paddleformers.fleet.training.global_vars import _set_timers

        _set_timers()
        with self.assertRaises(AssertionError):
            _set_timers()

    def test_ensure_var_is_initialized_none_raises(self):
        from paddleformers.fleet.training.global_vars import _ensure_var_is_initialized

        with self.assertRaises(AssertionError) as ctx:
            _ensure_var_is_initialized(None, "test_var")
        self.assertIn("test_var", str(ctx.exception))

    def test_ensure_var_is_initialized_not_none_passes(self):
        from paddleformers.fleet.training.global_vars import _ensure_var_is_initialized

        # Should not raise
        _ensure_var_is_initialized("some_value", "test_var")

    def test_ensure_var_is_not_initialized_none_passes(self):
        from paddleformers.fleet.training.global_vars import (
            _ensure_var_is_not_initialized,
        )

        # Should not raise
        _ensure_var_is_not_initialized(None, "test_var")

    def test_ensure_var_is_not_initialized_not_none_raises(self):
        from paddleformers.fleet.training.global_vars import (
            _ensure_var_is_not_initialized,
        )

        with self.assertRaises(AssertionError) as ctx:
            _ensure_var_is_not_initialized("some_value", "test_var")
        self.assertIn("test_var", str(ctx.exception))

    def test_destroy_global_vars(self):
        from paddleformers.fleet.training.global_vars import (
            _set_timers,
            destroy_global_vars,
            get_args,
            get_timers,
            set_args,
        )

        set_args({"key": "value"})
        _set_timers()
        destroy_global_vars()

        with self.assertRaises(AssertionError):
            get_args()
        with self.assertRaises(AssertionError):
            get_timers()

    def test_set_global_variables(self):
        from paddleformers.fleet.training.global_vars import (
            get_args,
            get_timers,
            set_global_variables,
        )

        mock_args = {"data": "test"}
        set_global_variables(mock_args)
        self.assertEqual(get_args(), mock_args)
        self.assertIsNotNone(get_timers())

    def test_set_global_variables_none_raises(self):
        from paddleformers.fleet.training.global_vars import set_global_variables

        with self.assertRaises(AssertionError):
            set_global_variables(None)

    def test_set_global_variables_already_initialized_raises(self):
        from paddleformers.fleet.training.global_vars import (
            set_args,
            set_global_variables,
        )

        set_args({"existing": True})
        with self.assertRaises(AssertionError):
            set_global_variables({"new": True})

    def test_unset_global_variables(self):
        from paddleformers.fleet.training.global_vars import (
            get_args,
            get_timers,
            set_global_variables,
            unset_global_variables,
        )

        set_global_variables({"data": "test"})
        unset_global_variables()

        with self.assertRaises(AssertionError):
            get_args()
        with self.assertRaises(AssertionError):
            get_timers()

    def test_unset_global_variables_when_not_set(self):
        from paddleformers.fleet.training.global_vars import unset_global_variables

        # Should not raise even when nothing is set
        unset_global_variables()

    def test_set_global_variables_creates_timers(self):
        from paddleformers.fleet.training.global_vars import (
            get_timers,
            set_global_variables,
        )

        set_global_variables({"test": True})
        timers = get_timers()
        # Verify it is a Timers instance
        from paddleformers.fleet.timers import Timers

        self.assertIsInstance(timers, Timers)
