# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

import unittest
from unittest.mock import patch


class TestToolsUtils(unittest.TestCase):
    """Tests for utils/tools.py"""

    def test_compare_version_equal(self):
        from paddleformers.utils.tools import compare_version

        result = compare_version("2.2.0", "2.2.0")
        self.assertEqual(result, 0)

    def test_compare_version_greater(self):
        from paddleformers.utils.tools import compare_version

        result = compare_version("2.2.1", "2.2.0")
        self.assertEqual(result, 1)

    def test_compare_version_less(self):
        from paddleformers.utils.tools import compare_version

        result = compare_version("2.1.0", "2.2.0")
        self.assertEqual(result, -1)

    def test_compare_version_rc(self):
        from paddleformers.utils.tools import compare_version

        result = compare_version("2.2.0-rc0", "2.2.0")
        self.assertEqual(result, -1)

    def test_compare_version_rc_greater(self):
        from paddleformers.utils.tools import compare_version

        result = compare_version("2.3.0-rc0", "2.2.0")
        self.assertEqual(result, 1)

    def test_get_bool_ids_greater_than_simple(self):
        pass

        from paddleformers.utils.tools import get_bool_ids_greater_than

        probs = [0.1, 0.6, 0.8, 0.3]
        result = get_bool_ids_greater_than(probs, limit=0.5)
        self.assertEqual(result, [1, 2])

    def test_get_bool_ids_greater_than_with_prob(self):
        pass

        from paddleformers.utils.tools import get_bool_ids_greater_than

        probs = [0.1, 0.6, 0.8, 0.3]
        result = get_bool_ids_greater_than(probs, limit=0.5, return_prob=True)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][0], 1)
        self.assertEqual(result[1][0], 2)

    def test_get_bool_ids_greater_than_2d(self):
        from paddleformers.utils.tools import get_bool_ids_greater_than

        probs = [[0.1, 0.6], [0.8, 0.3]]
        result = get_bool_ids_greater_than(probs, limit=0.5)
        self.assertEqual(len(result), 2)

    def test_get_span_simple(self):
        from paddleformers.utils.tools import get_span

        start_ids = [0, 2]
        end_ids = [1, 3]
        result = get_span(start_ids, end_ids)
        self.assertIsInstance(result, set)

    def test_get_span_with_prob(self):
        from paddleformers.utils.tools import get_span

        start_ids = [(0, 0.9), (2, 0.8)]
        end_ids = [(1, 0.7), (3, 0.6)]
        result = get_span(start_ids, end_ids, with_prob=True)
        self.assertIsInstance(result, set)

    def test_dispatch_to(self):
        from paddleformers.utils.tools import dispatch_to

        def dispatch_fn(*args, **kwargs):
            return "dispatched"

        def original_fn(*args, **kwargs):
            return "original"

        decorator = dispatch_to(dispatch_fn)
        wrapped = decorator(original_fn)
        self.assertEqual(wrapped("test"), "dispatched")

    def test_dispatch_to_cond_false(self):
        from paddleformers.utils.tools import dispatch_to

        def dispatch_fn(*args, **kwargs):
            return "dispatched"

        def original_fn(*args, **kwargs):
            return "original"

        decorator = dispatch_to(dispatch_fn, cond=lambda *a, **k: False)
        wrapped = decorator(original_fn)
        self.assertEqual(wrapped("test"), "original")

    def test_device_guard_cpu(self):
        from paddleformers.utils.tools import device_guard

        with patch("paddle.device.get_device", return_value="gpu:0"):
            guard = device_guard("cpu")
            with guard:
                pass

    def test_get_env_device_mock_cpu(self):
        with patch("paddle.is_compiled_with_cuda", return_value=False):
            with patch("paddle.is_compiled_with_rocm", return_value=False):
                with patch("paddle.is_compiled_with_xpu", return_value=False):
                    with patch("paddle.device.get_all_custom_device_type", return_value=[]):
                        from paddleformers.utils.tools import get_env_device

                        result = get_env_device()
                        self.assertEqual(result, "cpu")
