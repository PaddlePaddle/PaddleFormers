# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

import unittest
from unittest.mock import patch


class TestPerfUtils(unittest.TestCase):
    """Tests for utils/perf_utils.py"""

    def test_register_profile_hook_layer(self):
        import paddle.nn as nn

        from paddleformers.utils.perf_utils import register_profile_hook

        model = nn.Linear(10, 5)
        register_profile_hook(model, debug=None)
        # Just ensure no exception

    def test_register_profile_hook_list(self):
        import paddle.nn as nn

        from paddleformers.utils.perf_utils import register_profile_hook

        models = [nn.Linear(10, 5), nn.Linear(5, 3)]
        register_profile_hook(models)
        # Just ensure no exception

    def test_register_profile_hook_debug_memory(self):
        import paddle.nn as nn

        from paddleformers.utils.perf_utils import register_profile_hook

        model = nn.Linear(10, 5)
        register_profile_hook(model, debug="memory")
        # Just ensure no exception

    def test_add_record_event(self):
        from paddleformers.utils.perf_utils import add_record_event

        with add_record_event("test_event"):
            pass
        # Context manager should complete without error

    @patch("paddleformers.utils.perf_utils._PROFILER_ENABLED", True)
    def test_add_record_event_profiler_enabled(self):
        from paddleformers.utils.perf_utils import add_record_event

        with patch("paddle.base.core.nvprof_nvtx_push"), patch("paddle.base.core.nvprof_nvtx_pop"):
            with add_record_event("test_event"):
                pass

    def test_time_cost_average(self):
        from paddleformers.utils.tools import TimeCostAverage

        tca = TimeCostAverage()
        self.assertEqual(tca.get_average(), 0)

    def test_time_cost_average_record(self):
        from paddleformers.utils.tools import TimeCostAverage

        tca = TimeCostAverage()
        tca.record(1.0)
        tca.record(3.0)
        self.assertEqual(tca.get_average(), 2.0)

    def test_time_cost_average_reset(self):
        from paddleformers.utils.tools import TimeCostAverage

        tca = TimeCostAverage()
        tca.record(2.0)
        tca.record(4.0)
        tca.reset()
        self.assertEqual(tca.get_average(), 0)
        self.assertEqual(tca.cnt, 0)
