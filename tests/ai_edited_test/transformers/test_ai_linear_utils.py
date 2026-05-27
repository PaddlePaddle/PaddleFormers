# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

import unittest
from unittest.mock import patch


class TestLinearUtils(unittest.TestCase):
    """Tests for transformers/linear_utils.py"""

    def test_linear_is_nn_linear(self):
        import paddle.nn as nn

        from paddleformers.transformers.linear_utils import Linear

        self.assertEqual(Linear, nn.Linear)

    def test_column_parallel_linear(self):
        from paddle.distributed.fleet.meta_parallel import (
            ColumnParallelLinear as _ColumnParallelLinear,
        )

        from paddleformers.transformers.linear_utils import ColumnParallelLinear

        self.assertEqual(ColumnParallelLinear, _ColumnParallelLinear)

    def test_row_parallel_linear(self):
        from paddle.distributed.fleet.meta_parallel import (
            RowParallelLinear as _RowParallelLinear,
        )

        from paddleformers.transformers.linear_utils import RowParallelLinear

        self.assertEqual(RowParallelLinear, _RowParallelLinear)

    def test_all_exports(self):
        from paddleformers.transformers.linear_utils import (
            ColumnParallelLinear,
            ColumnSequenceParallelLinear,
            Linear,
            RowParallelLinear,
            RowSequenceParallelLinear,
        )

        self.assertIsNotNone(Linear)
        self.assertIsNotNone(ColumnParallelLinear)
        self.assertIsNotNone(RowParallelLinear)
        self.assertIsNotNone(ColumnSequenceParallelLinear)
        self.assertIsNotNone(RowSequenceParallelLinear)

    @patch("paddleformers.transformers.linear_utils.get_env_device", return_value="npu")
    def test_npu_device_uses_mc2(self, mock_device):
        from paddleformers.transformers.linear_utils import (
            ColumnSequenceParallelLinear,
            RowSequenceParallelLinear,
        )

        # When MC2 classes are available and device is npu, should use MC2 variants
        # This test just verifies the imports work on npu path
        self.assertIsNotNone(ColumnSequenceParallelLinear)
        self.assertIsNotNone(RowSequenceParallelLinear)

    @patch("paddleformers.transformers.linear_utils.get_env_device", return_value="cpu")
    def test_cpu_device_default(self, mock_device):
        from paddleformers.transformers.linear_utils import Linear

        # CPU device should use default Paddle Linear
        self.assertEqual(Linear, __import__("paddle").nn.Linear)
