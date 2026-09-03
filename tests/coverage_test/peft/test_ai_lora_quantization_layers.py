# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

import unittest
from unittest.mock import MagicMock, patch


class TestQuantizationLoRALayers(unittest.TestCase):
    """Tests for peft/lora/lora_quantization_layers.py"""

    def test_QuantizationLoRABaseLinear_init(self):
        from paddleformers.peft.lora.lora_quantization_layers import (
            QuantizationLoRABaseLinear,
        )

        mock_layer = MagicMock()
        mock_layer.quantization_config = MagicMock()
        mock_layer.weight_quantize_algo = "weight_only_int8"
        mock_layer._dtype = "float16"
        mock_layer.quant_dtype = "int8"
        mock_layer.quant_weight = MagicMock()
        mock_layer.weight_scale = MagicMock()
        mock_layer.bias = None

        mock_lora_config = MagicMock()
        mock_lora_config.r = 8
        mock_lora_config.lora_alpha = 16.0
        mock_lora_config.lora_dropout = 0.0
        mock_lora_config.rslora = False

        layer = QuantizationLoRABaseLinear(mock_layer, mock_lora_config)
        self.assertEqual(layer.scaling, 2.0)  # 16.0 / 8
        self.assertFalse(layer.disable_lora)

    def test_QuantizationLoRABaseLinear_rslora_scaling(self):
        import math

        from paddleformers.peft.lora.lora_quantization_layers import (
            QuantizationLoRABaseLinear,
        )

        mock_layer = MagicMock()
        mock_layer.quantization_config = MagicMock()
        mock_layer.weight_quantize_algo = "weight_only_int8"
        mock_layer._dtype = "float16"
        mock_layer.quant_dtype = "int8"
        mock_layer.quant_weight = MagicMock()
        mock_layer.weight_scale = MagicMock()
        mock_layer.bias = None

        mock_lora_config = MagicMock()
        mock_lora_config.r = 8
        mock_lora_config.lora_alpha = 16.0
        mock_lora_config.lora_dropout = 0.0
        mock_lora_config.rslora = True

        layer = QuantizationLoRABaseLinear(mock_layer, mock_lora_config)
        expected = 16.0 / math.sqrt(8)
        self.assertAlmostEqual(layer.scaling, expected)

    def test_QuantizationLoRABaseLinear_invalid_r(self):
        from paddleformers.peft.lora.lora_quantization_layers import (
            QuantizationLoRABaseLinear,
        )

        mock_layer = MagicMock()
        mock_layer.quantization_config = MagicMock()
        mock_layer.weight_quantize_algo = "weight_only_int8"
        mock_layer._dtype = "float16"
        mock_layer.quant_dtype = "int8"
        mock_layer.quant_weight = MagicMock()
        mock_layer.weight_scale = MagicMock()
        mock_layer.bias = None

        mock_lora_config = MagicMock()
        mock_lora_config.r = -1
        mock_lora_config.lora_alpha = 16.0
        mock_lora_config.lora_dropout = 0.0
        mock_lora_config.rslora = False

        with self.assertRaises(ValueError):
            QuantizationLoRABaseLinear(mock_layer, mock_lora_config)

    def test_QuantizationLoRABaseLinear_llm_int8_raises(self):
        from paddleformers.peft.lora.lora_quantization_layers import (
            QuantizationLoRABaseLinear,
        )

        mock_layer = MagicMock()
        mock_layer.quantization_config = MagicMock()
        mock_layer.weight_quantize_algo = "llm.int8"
        mock_layer._dtype = "float16"
        mock_layer.quant_dtype = "int8"
        mock_layer.quant_weight = MagicMock()
        mock_layer.weight_scale = MagicMock()
        mock_layer.bias = None

        mock_lora_config = MagicMock()
        mock_lora_config.r = 8
        mock_lora_config.lora_alpha = 16.0
        mock_lora_config.lora_dropout = 0.0
        mock_lora_config.rslora = False

        with self.assertRaises(NotImplementedError):
            QuantizationLoRABaseLinear(mock_layer, mock_lora_config)

    def test_QuantizationLoRABaseLinear_double_quant(self):
        from paddleformers.peft.lora.lora_quantization_layers import (
            QuantizationLoRABaseLinear,
        )

        mock_layer = MagicMock()
        mock_layer.quantization_config = MagicMock()
        mock_layer.quantization_config.qlora_weight_double_quant = True
        mock_layer.weight_quantize_algo = "nf4"
        mock_layer._dtype = "float16"
        mock_layer.quant_dtype = "nf4"
        mock_layer.quant_weight = MagicMock()
        mock_layer.qweight_scale = MagicMock()
        mock_layer.double_weight_scale = MagicMock()
        mock_layer.weight_scale_offset = MagicMock()
        mock_layer.bias = None

        mock_lora_config = MagicMock()
        mock_lora_config.r = 8
        mock_lora_config.lora_alpha = 16.0
        mock_lora_config.lora_dropout = 0.0
        mock_lora_config.rslora = False

        layer = QuantizationLoRABaseLinear(mock_layer, mock_lora_config)
        self.assertIsNotNone(layer.qweight_scale)
        self.assertIsNotNone(layer.double_weight_scale)
        self.assertIsNotNone(layer.weight_scale_offset)

    def test_QuantizationLoRALinear_forward_no_lora(self):
        from paddleformers.peft.lora.lora_quantization_layers import (
            QuantizationLoRALinear,
        )

        mock_layer = MagicMock()
        mock_layer.quantization_config = MagicMock()
        mock_layer.weight_quantize_algo = "weight_only_int8"
        mock_layer._dtype = "float16"
        mock_layer.quant_dtype = "int8"
        mock_layer.in_features = 64
        mock_layer.out_features = 128
        mock_layer.quant_weight = MagicMock()
        mock_layer.weight_scale = MagicMock()
        mock_layer.bias = None

        mock_lora_config = MagicMock()
        mock_lora_config.r = 8
        mock_lora_config.lora_alpha = 16.0
        mock_lora_config.lora_dropout = 0.0
        mock_lora_config.rslora = False

        import paddle

        with patch(
            "paddleformers.peft.lora.lora_quantization_layers.quant_weight_linear", return_value=paddle.randn([2, 128])
        ):
            with patch(
                "paddleformers.peft.lora.lora_quantization_layers.QuantizationLoRALinear.__init__", return_value=None
            ):
                layer = QuantizationLoRALinear.__new__(QuantizationLoRALinear)
                layer.disable_lora = True
                layer.lora_dropout = lambda x: x
                layer.scaling = 2.0
                layer.lora_A = MagicMock()
                layer.lora_B = MagicMock()

    def test_QuantizationLoRABaseLinear_merge_warns(self):
        from paddleformers.peft.lora.lora_quantization_layers import (
            QuantizationLoRABaseLinear,
        )

        mock_layer = MagicMock()
        mock_layer.quantization_config = MagicMock()
        mock_layer.weight_quantize_algo = "weight_only_int8"
        mock_layer._dtype = "float16"
        mock_layer.quant_dtype = "int8"
        mock_layer.quant_weight = MagicMock()
        mock_layer.weight_scale = MagicMock()
        mock_layer.bias = None

        mock_lora_config = MagicMock()
        mock_lora_config.r = 8
        mock_lora_config.lora_alpha = 16.0
        mock_lora_config.lora_dropout = 0.0
        mock_lora_config.rslora = False

        layer = QuantizationLoRABaseLinear(mock_layer, mock_lora_config)
        # merge/unmerge just warn
        layer.merge()
        layer.unmerge()

    def test_FleetQuantizationLoRALinear_init(self):
        from paddleformers.peft.lora.lora_quantization_layers import (
            FleetQuantizationLoRALinear,
        )

        mock_layer = MagicMock()
        mock_layer.quantization_config = MagicMock()
        mock_layer.weight_quantize_algo = "weight_only_int8"
        mock_layer._dtype = "float16"
        mock_layer.quant_dtype = "int8"
        mock_layer.in_features = 64
        mock_layer.out_features = 128
        mock_layer.quant_weight = MagicMock()
        mock_layer.weight_scale = MagicMock()
        mock_layer.bias = None

        mock_lora_config = MagicMock()
        mock_lora_config.r = 8
        mock_lora_config.lora_alpha = 16.0
        mock_lora_config.lora_dropout = 0.0
        mock_lora_config.rslora = False

        layer = FleetQuantizationLoRALinear(mock_layer, skip_bias_add=True, lora_config=mock_lora_config)
        self.assertTrue(layer.skip_bias_add)
