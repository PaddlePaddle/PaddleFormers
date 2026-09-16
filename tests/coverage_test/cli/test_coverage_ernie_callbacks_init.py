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

import unittest

from paddleformers.cli.train.ernie_pretrain.src.callbacks import (
    FP8QuantWeightCallback,
    GCCallback,
    GlobalRNGCallback,
    LoggingCallback,
    MoECorrectionBiasAdjustCallback,
    MoeLoggingCallback,
    OrthogonalCallback,
    SPGradSyncCallback,
    TensorBoardCallback,
)


class TestCallbacksInit(unittest.TestCase):
    """Tests for the callbacks __init__ module."""

    def test_gc_callback_importable(self):
        """Test GCCallback is importable."""
        self.assertIsNotNone(GCCallback)

    def test_logging_callback_importable(self):
        """Test LoggingCallback is importable."""
        self.assertIsNotNone(LoggingCallback)

    def test_tensorboard_callback_importable(self):
        """Test TensorBoardCallback is importable."""
        self.assertIsNotNone(TensorBoardCallback)

    def test_global_rng_callback_importable(self):
        """Test GlobalRNGCallback is importable."""
        self.assertIsNotNone(GlobalRNGCallback)

    def test_moe_logging_callback_importable(self):
        """Test MoeLoggingCallback is importable."""
        self.assertIsNotNone(MoeLoggingCallback)

    def test_sp_grad_sync_callback_importable(self):
        """Test SPGradSyncCallback is importable."""
        self.assertIsNotNone(SPGradSyncCallback)

    def test_moe_correction_bias_adjust_callback_importable(self):
        """Test MoECorrectionBiasAdjustCallback is importable."""
        self.assertIsNotNone(MoECorrectionBiasAdjustCallback)

    def test_fp8_quant_weight_callback_importable(self):
        """Test FP8QuantWeightCallback is importable."""
        self.assertIsNotNone(FP8QuantWeightCallback)

    def test_orthogonal_callback_importable(self):
        """Test OrthogonalCallback is importable."""
        self.assertIsNotNone(OrthogonalCallback)

    def test_all_exports(self):
        """Test all __all__ exports are importable."""
        from paddleformers.cli.train.ernie_pretrain.src.callbacks import __all__

        expected = [
            "TensorBoardCallback",
            "LoggingCallback",
            "GCCallback",
            "GlobalRNGCallback",
            "MoeLoggingCallback",
            "SPGradSyncCallback",
            "MoECorrectionBiasAdjustCallback",
            "FP8QuantWeightCallback",
            "OrthogonalCallback",
        ]
        for name in expected:
            self.assertIn(name, __all__)

    def test_gc_callback_instantiation(self):
        """Test GCCallback can be instantiated."""
        callback = GCCallback()
        self.assertIsNotNone(callback)

    def test_logging_callback_instantiation(self):
        """Test LoggingCallback can be instantiated."""
        callback = LoggingCallback()
        self.assertIsNotNone(callback)


if __name__ == "__main__":
    unittest.main()
