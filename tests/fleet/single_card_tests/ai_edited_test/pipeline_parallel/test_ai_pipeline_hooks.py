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
from unittest.mock import MagicMock


class TestPipelineHooks(unittest.TestCase):
    """Unit tests for pipeline_hooks.py"""

    def test_init(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_hooks import PipelineHook

        hook = PipelineHook()
        self.assertEqual(hook._current_id, 0)
        self.assertEqual(hook._hooks_capacity, 0)

    def test_reset_current_id(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_hooks import PipelineHook

        hook = PipelineHook()
        hook._current_id = 5
        hook.reset_current_id()
        self.assertEqual(hook._current_id, 0)

    def test_set_hooks_capacity(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_hooks import PipelineHook

        hook = PipelineHook()
        hook.set_hooks_capacity(10)
        self.assertEqual(hook._hooks_capacity, 10)

    def test_register_hook_valid(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_hooks import PipelineHook

        hook = PipelineHook()
        hook.set_hooks_capacity(5)
        mock_fn = MagicMock()
        hook.register_hook(0, mock_fn)
        self.assertEqual(len(hook.hooks[0]), 1)

    def test_register_hook_out_of_range(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_hooks import PipelineHook

        hook = PipelineHook()
        hook.set_hooks_capacity(3)
        mock_fn = MagicMock()
        with self.assertRaises(AssertionError):
            hook.register_hook(5, mock_fn)

    def test_register_hook_multiple(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_hooks import PipelineHook

        hook = PipelineHook()
        hook.set_hooks_capacity(5)
        fn1 = MagicMock()
        fn2 = MagicMock()
        hook.register_hook(2, fn1)
        hook.register_hook(2, fn2)
        self.assertEqual(len(hook.hooks[2]), 2)

    def test_run_hook_valid(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_hooks import PipelineHook

        hook = PipelineHook()
        hook.set_hooks_capacity(5)
        fn1 = MagicMock()
        fn2 = MagicMock()
        hook.register_hook(0, fn1)
        hook.register_hook(0, fn2)
        hook.run_hook()
        fn1.assert_called_once_with(0)
        fn2.assert_called_once_with(0)
        self.assertEqual(hook._current_id, 1)

    def test_run_hook_out_of_range(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_hooks import PipelineHook

        hook = PipelineHook()
        hook.set_hooks_capacity(1)
        hook.run_hook()
        self.assertEqual(hook._current_id, 1)
        with self.assertRaises(AssertionError):
            hook.run_hook()

    def test_run_hook_no_hooks_registered(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_hooks import PipelineHook

        hook = PipelineHook()
        hook.set_hooks_capacity(5)
        # No hooks registered for current_id, but should not raise
        hook.run_hook()
        self.assertEqual(hook._current_id, 1)

    def test_run_hook_sequential(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_hooks import PipelineHook

        hook = PipelineHook()
        hook.set_hooks_capacity(5)
        fn0 = MagicMock()
        fn1 = MagicMock()
        hook.register_hook(0, fn0)
        hook.register_hook(1, fn1)
        hook.run_hook()
        self.assertEqual(hook._current_id, 1)
        fn0.assert_called_once()
        fn1.assert_not_called()
        hook.run_hook()
        self.assertEqual(hook._current_id, 2)
        fn1.assert_called_once()

    def test_current_id_property(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_hooks import PipelineHook

        hook = PipelineHook()
        self.assertEqual(hook.current_id, 0)
        hook._current_id = 3
        self.assertEqual(hook.current_id, 3)

    def test_hooks_capacity_property(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_hooks import PipelineHook

        hook = PipelineHook()
        hook.set_hooks_capacity(10)
        self.assertEqual(hook.hooks_capacity, 10)


if __name__ == "__main__":
    unittest.main()
