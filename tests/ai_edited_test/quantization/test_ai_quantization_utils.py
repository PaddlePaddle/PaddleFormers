# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

import unittest
from unittest.mock import MagicMock, patch


class TestQuantizationUtils(unittest.TestCase):
    """Tests for quantization/quantization_utils.py"""

    def test_parse_weight_quantize_algo_string(self):
        from paddleformers.quantization.quantization_utils import (
            parse_weight_quantize_algo,
        )

        mock_config = MagicMock()
        mock_config.ignore_modules = None
        mock_config.weight_quantize_algo = "weight_only_int8"

        result = parse_weight_quantize_algo(mock_config, "layer.0")
        self.assertEqual(result, "weight_only_int8")

    def test_parse_weight_quantize_algo_ignored(self):
        from paddleformers.quantization.quantization_utils import (
            parse_weight_quantize_algo,
        )

        mock_config = MagicMock()
        mock_config.ignore_modules = ["layer\\.0"]
        mock_config.weight_quantize_algo = "weight_only_int8"

        result = parse_weight_quantize_algo(mock_config, "layer.0")
        self.assertIsNone(result)

    def test_parse_weight_quantize_algo_dict_match(self):
        from paddleformers.quantization.quantization_utils import (
            parse_weight_quantize_algo,
        )

        mock_config = MagicMock()
        mock_config.ignore_modules = None
        mock_config.weight_quantize_algo = {
            "weight_only_int8": ["layer\\.0"],
            "weight_only_int4": ["layer\\.1"],
        }

        result = parse_weight_quantize_algo(mock_config, "layer.0")
        self.assertEqual(result, "weight_only_int8")

        result = parse_weight_quantize_algo(mock_config, "layer.1")
        self.assertEqual(result, "weight_only_int4")

    def test_parse_weight_quantize_algo_dict_no_match(self):
        from paddleformers.quantization.quantization_utils import (
            parse_weight_quantize_algo,
        )

        mock_config = MagicMock()
        mock_config.ignore_modules = None
        mock_config.weight_quantize_algo = {
            "weight_only_int8": ["layer\\.0"],
        }

        result = parse_weight_quantize_algo(mock_config, "layer.99")
        self.assertIsNone(result)

    def test_convert_to_quantize_state_dict_already_quantized(self):
        from paddleformers.quantization.quantization_utils import (
            convert_to_weight_quantize_state_dict,
        )

        state_dict = {
            "layer.0.quant_weight": MagicMock(),
            "layer.0.weight_scale": MagicMock(),
        }

        mock_config = MagicMock()

        result = convert_to_weight_quantize_state_dict(
            state_dict, "layer.0", mock_config, "float16", "weight_only_int8"
        )
        # Should return early since quant_weight already exists
        self.assertIn("layer.0.quant_weight", result)

    def test_convert_to_quantize_state_dict_no_weight(self):
        from paddleformers.quantization.quantization_utils import (
            convert_to_weight_quantize_state_dict,
        )

        state_dict = {}
        mock_config = MagicMock()

        result = convert_to_weight_quantize_state_dict(
            state_dict, "layer.0", mock_config, "float16", "weight_only_int8"
        )
        self.assertEqual(result, state_dict)

    def test_convert_to_qlora_state_dict_import_error(self):
        from paddleformers.quantization.quantization_utils import (
            convert_to_qlora_state_dict,
        )

        state_dict = {"layer.0.weight": MagicMock()}
        mock_config = MagicMock()

        with patch("paddleformers.quantization.quantization_utils.qlora_weight_quantize", None):
            with self.assertRaises(ImportError):
                convert_to_qlora_state_dict(state_dict, "layer.0", mock_config, "float16", "nf4")

    def test_update_loaded_state_dict_keys_no_change(self):
        from paddleformers.quantization.quantization_utils import (
            update_loaded_state_dict_keys,
        )

        state_dict = ["layer.0.quant_weight", "layer.0.weight_scale"]
        mock_config = MagicMock()
        mock_config.qlora_weight_double_quant = False

        result = update_loaded_state_dict_keys(state_dict, ["layer.0"], mock_config)
        # No change since quant_weight and weight_scale already present
        self.assertIn("layer.0.quant_weight", result)

    def test_update_loaded_state_dict_keys_replace(self):
        from paddleformers.quantization.quantization_utils import (
            update_loaded_state_dict_keys,
        )

        state_dict = ["layer.0.weight"]
        mock_config = MagicMock()
        mock_config.qlora_weight_double_quant = False

        result = update_loaded_state_dict_keys(state_dict, ["layer.0"], mock_config)
        self.assertNotIn("layer.0.weight", result)
        self.assertIn("layer.0.quant_weight", result)
        self.assertIn("layer.0.weight_scale", result)

    def test_LINEAR_CLASSES(self):
        import paddle.nn as nn

        from paddleformers.quantization.quantization_utils import LINEAR_CLASSES

        self.assertIn(nn.Linear, LINEAR_CLASSES)

    def test_convert_to_quantize_state_dict_unsupported_algo(self):
        from paddleformers.quantization.quantization_utils import (
            convert_to_quantize_state_dict,
        )

        state_dict = {"layer.0.weight": MagicMock()}
        mock_config = MagicMock()

        with patch(
            "paddleformers.quantization.quantization_utils.parse_weight_quantize_algo", return_value="unsupported_algo"
        ):
            with self.assertRaises(NotImplementedError):
                convert_to_quantize_state_dict(state_dict, ["layer.0"], mock_config, "float16")
