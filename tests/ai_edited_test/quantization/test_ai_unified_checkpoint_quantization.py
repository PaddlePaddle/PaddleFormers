# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

import unittest
from unittest.mock import MagicMock, patch


class TestUnifiedCheckpointQuantization(unittest.TestCase):
    """Tests for quantization/unified_checkpoint_quantization.py"""

    def test_dequant_unified_optimizer_o0(self):
        from paddleformers.quantization.unified_checkpoint_quantization import (
            dequant_unified_optimizer,
        )

        state_dict = {"test": MagicMock()}
        scale_dict = {}
        result = dequant_unified_optimizer(state_dict, "O0", scale_dict)
        self.assertEqual(result, state_dict)

    def test_dequant_unified_optimizer_o1_moment1(self):
        import paddle

        from paddleformers.quantization.unified_checkpoint_quantization import (
            dequant_unified_optimizer,
        )

        with patch("paddle.distributed.get_world_size", return_value=1):
            with patch("paddleformers.quantization.unified_checkpoint_quantization.qdq_weight") as mock_qdq:
                mock_qdq.return_value = (paddle.randn([4, 8], dtype="float32"), paddle.randn([8], dtype="float32"))

                state_dict = {"layer/moment1": paddle.randint(-10, 10, [4, 8]).astype("int8")}
                scale_dict = {"layer/moment1.scale": paddle.randn([8], dtype="float32")}
                result = dequant_unified_optimizer(state_dict, "O1", scale_dict)
                self.assertIn("layer/moment1", result)

    def test_dequant_unified_optimizer_o1_moment2(self):
        import paddle

        from paddleformers.quantization.unified_checkpoint_quantization import (
            dequant_unified_optimizer,
        )

        with patch("paddle.distributed.get_world_size", return_value=1):
            with patch("paddleformers.quantization.unified_checkpoint_quantization.asymmetry_qdq_weight") as mock_aqdq:
                mock_aqdq.return_value = (
                    paddle.randn([4, 8], dtype="float32"),
                    paddle.randn([8], dtype="float32"),
                )

                state_dict = {"layer/moment2": paddle.randint(-10, 10, [4, 8]).astype("int8")}
                scale_dict = {
                    "layer/moment2.min_scale": paddle.randn([8], dtype="float32"),
                    "layer/moment2.max_scale": paddle.randn([8], dtype="float32"),
                }
                result = dequant_unified_optimizer(state_dict, "O1", scale_dict, use_pd=True)
                self.assertIn("layer/moment2", result)

    def test_quant_unified_optimizer_o0(self):
        from paddleformers.quantization.unified_checkpoint_quantization import (
            quant_unified_optimizer,
        )

        state_dict = {"test": MagicMock()}
        result = quant_unified_optimizer(state_dict, "model_weight", "O0")
        self.assertEqual(result, state_dict)

    def test_quant_unified_optimizer_o1_optimizer_weight(self):
        import paddle

        from paddleformers.quantization.unified_checkpoint_quantization import (
            quant_unified_optimizer,
        )

        with patch("paddleformers.quantization.unified_checkpoint_quantization.cal_ratio") as mock_ratio:
            with patch("paddleformers.quantization.unified_checkpoint_quantization.qdq_weight") as mock_qdq:
                with patch(
                    "paddleformers.quantization.unified_checkpoint_quantization.asymmetry_qdq_weight"
                ) as mock_aqdq:
                    mock_ratio.return_value = paddle.randn([4, 8], dtype="float32")
                    mock_qdq.return_value = (
                        paddle.randint(-10, 10, [4, 8]).astype("int8"),
                        paddle.randn([8], dtype="float32"),
                    )
                    mock_aqdq.return_value = (
                        paddle.randint(0, 255, [4, 8]).astype("uint8"),
                        paddle.randn([8], dtype="float32"),
                        paddle.randn([8], dtype="float32"),
                    )

                    state_dict = {
                        "layer/moment1": paddle.randn([4, 8], dtype="float32"),
                        "layer/moment2": paddle.randn([4, 8], dtype="float32"),
                    }
                    result = quant_unified_optimizer(state_dict, "optimizer_weight", "O1")
                    self.assertIsInstance(result, dict)

    def test_quant_unified_optimizer_o1_model_weight(self):
        from paddleformers.quantization.unified_checkpoint_quantization import (
            quant_unified_optimizer,
        )

        state_dict = {"weight": MagicMock()}
        result = quant_unified_optimizer(state_dict, "model_weight", "O1")
        self.assertEqual(result, state_dict)
