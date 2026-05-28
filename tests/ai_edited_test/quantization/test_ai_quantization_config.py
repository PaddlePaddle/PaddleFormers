# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

import json
import os
import tempfile
import unittest
from unittest.mock import patch


class TestQuantizationConfigInit(unittest.TestCase):
    """Tests for QuantizationConfig initialization."""

    def test_default_init(self):
        """Test QuantizationConfig default values."""
        from paddleformers.quantization.quantization_config import QuantizationConfig

        config = QuantizationConfig()
        self.assertIsNone(config.weight_quantize_algo)
        self.assertIsNone(config.quant_type)
        self.assertFalse(config.shift)
        self.assertFalse(config.smooth)
        self.assertFalse(config.shift_smooth_all_linears)
        self.assertEqual(config.quant_round_type, 0)
        self.assertAlmostEqual(config.llm_int8_threshold, 6.0)
        self.assertFalse(config.qlora_weight_double_quant)
        self.assertEqual(config.qlora_weight_blocksize, 64)
        self.assertEqual(config.qlora_weight_double_quant_block_size, 256)
        self.assertEqual(config.weight_quant_method, "abs_max_channel_wise")
        self.assertEqual(config.act_quant_method, "abs_max")
        self.assertIsNone(config.activation_scheme)
        self.assertIsNone(config.fmt)
        self.assertIsNone(config.quant_method)
        self.assertIsNone(config.weight_block_size)
        self.assertIsNone(config.dtype)
        self.assertIsNone(config.ignore_modules)
        self.assertEqual(config.group_size, -1)
        self.assertFalse(config.apply_hadamard)
        self.assertEqual(config.hadamard_block_size, 32)
        self.assertFalse(config.quant_input_grad)
        self.assertFalse(config.quant_weight_grad)
        self.assertEqual(config.apply_online_actscale_step, 200)
        self.assertAlmostEqual(config.actscale_moving_rate, 0.01)
        self.assertEqual(config.fp8_format_type, "hybrid")
        self.assertAlmostEqual(config.scale_epsilon, 1e-8)

    def test_init_with_string_algo(self):
        """Test QuantizationConfig with string weight_quantize_algo."""
        from paddleformers.quantization.quantization_config import QuantizationConfig

        config = QuantizationConfig(weight_quantize_algo="weight_only_int8")
        self.assertEqual(config.weight_quantize_algo, "weight_only_int8")

    def test_init_with_dict_algo(self):
        """Test QuantizationConfig with dict weight_quantize_algo."""
        from paddleformers.quantization.quantization_config import QuantizationConfig

        algo = {"weight_only_int8": [".*mlp.*"]}
        config = QuantizationConfig(weight_quantize_algo=algo)
        self.assertEqual(config.weight_quantize_algo, algo)

    def test_init_with_unsupported_string_algo_raises(self):
        """Test QuantizationConfig with unsupported string algo raises ValueError."""
        from paddleformers.quantization.quantization_config import QuantizationConfig

        with self.assertRaises(ValueError) as ctx:
            QuantizationConfig(weight_quantize_algo="unsupported_algo")
        self.assertIn("not in supported list", str(ctx.exception))

    def test_init_with_unsupported_dict_algo_raises(self):
        """Test QuantizationConfig with unsupported dict algo raises ValueError."""
        from paddleformers.quantization.quantization_config import QuantizationConfig

        with self.assertRaises(ValueError) as ctx:
            QuantizationConfig(weight_quantize_algo={"bad_algo": [".*mlp.*"]})
        self.assertIn("not in supported list", str(ctx.exception))

    def test_init_with_quant_type(self):
        """Test QuantizationConfig with valid quant_type."""
        from paddleformers.quantization.quantization_config import QuantizationConfig

        for qt in ["weight_only_int8", "weight_only_int4", "a8w8", "a8w8c8", "a8w8_fp8", "a8w8c8_fp8"]:
            config = QuantizationConfig(quant_type=qt)
            self.assertEqual(config.quant_type, qt)

    def test_init_with_unsupported_quant_type_raises(self):
        """Test QuantizationConfig with unsupported quant_type raises ValueError."""
        from paddleformers.quantization.quantization_config import QuantizationConfig

        with self.assertRaises(ValueError) as ctx:
            QuantizationConfig(quant_type="bad_quant_type")
        self.assertIn("not in supported list", str(ctx.exception))

    def test_init_with_all_supported_string_algos(self):
        """Test all supported string weight_quantize_algo values."""
        from paddleformers.quantization.quantization_config import QuantizationConfig

        non_fp8 = [
            "weight_only_int8",
            "weight_only_int4",
            "llm.int8",
            "a8w8",
            "nf4",
            "fp4",
            "a8w8linear",
            "a8w4linear",
        ]
        for algo in non_fp8:
            config = QuantizationConfig(weight_quantize_algo=algo)
            self.assertEqual(config.weight_quantize_algo, algo)

        # fp8linear requires Hopper GPU architecture
        with patch("paddleformers.quantization.quantization_config._get_arch_info", return_value=89):
            config = QuantizationConfig(weight_quantize_algo="fp8linear")
            self.assertEqual(config.weight_quantize_algo, "fp8linear")

    def test_init_shift_smooth(self):
        """Test QuantizationConfig with shift and smooth enabled."""
        from paddleformers.quantization.quantization_config import QuantizationConfig

        config = QuantizationConfig(shift=True, smooth=True, shift_smooth_all_linears=True)
        self.assertTrue(config.shift)
        self.assertTrue(config.smooth)
        self.assertTrue(config.shift_smooth_all_linears)

    def test_init_custom_params(self):
        """Test QuantizationConfig with various custom parameters."""
        from paddleformers.quantization.quantization_config import QuantizationConfig

        config = QuantizationConfig(
            quant_round_type=1,
            llm_int8_threshold=8.0,
            qlora_weight_double_quant=True,
            qlora_weight_blocksize=128,
            group_size=32,
            apply_hadamard=True,
            hadamard_block_size=64,
            quant_input_grad=True,
            quant_weight_grad=True,
        )
        self.assertEqual(config.quant_round_type, 1)
        self.assertAlmostEqual(config.llm_int8_threshold, 8.0)
        self.assertTrue(config.qlora_weight_double_quant)
        self.assertEqual(config.qlora_weight_blocksize, 128)
        self.assertEqual(config.group_size, 32)
        self.assertTrue(config.apply_hadamard)
        self.assertEqual(config.hadamard_block_size, 64)
        self.assertTrue(config.quant_input_grad)
        self.assertTrue(config.quant_weight_grad)

    def test_act_quant_method_mapping(self):
        """Test act_quant_method is mapped through quant_inference_mapping."""
        from paddleformers.quantization.quantization_config import QuantizationConfig

        config = QuantizationConfig(act_quant_method="avg")
        self.assertEqual(config.act_quant_method, "abs_max")

    def test_init_fp8_format_type(self):
        """Test fp8_format_type parameter."""
        from paddleformers.quantization.quantization_config import QuantizationConfig

        config = QuantizationConfig(fp8_format_type="e4m3")
        self.assertEqual(config.fp8_format_type, "e4m3")

    def test_init_extra_kwargs_ignored(self):
        """Test extra kwargs are silently ignored."""
        from paddleformers.quantization.quantization_config import QuantizationConfig

        config = QuantizationConfig(unknown_param="value")
        self.assertFalse(hasattr(config, "unknown_param"))

    def test_fp8linear_on_non_hopper_raises(self):
        """Test fp8linear on non-Hopper GPU raises RuntimeError."""
        from paddleformers.quantization.quantization_config import QuantizationConfig

        with patch("paddleformers.quantization.quantization_config._get_arch_info", return_value=80):
            with self.assertRaises(RuntimeError) as ctx:
                QuantizationConfig(weight_quantize_algo="fp8linear")
            self.assertIn("Hopper", str(ctx.exception))

    def test_fp8linear_on_hopper_ok(self):
        """Test fp8linear on Hopper GPU (arch 89) works fine."""
        from paddleformers.quantization.quantization_config import QuantizationConfig

        with patch("paddleformers.quantization.quantization_config._get_arch_info", return_value=89):
            config = QuantizationConfig(weight_quantize_algo="fp8linear")
            self.assertEqual(config.weight_quantize_algo, "fp8linear")

    def test_fp8linear_dict_on_non_hopper_raises(self):
        """Test fp8linear in dict on non-Hopper GPU raises RuntimeError."""
        from paddleformers.quantization.quantization_config import QuantizationConfig

        with patch("paddleformers.quantization.quantization_config._get_arch_info", return_value=80):
            with self.assertRaises(RuntimeError):
                QuantizationConfig(weight_quantize_algo={"fp8linear": [".*mlp.*"]})

    def test_get_arch_info_hopper_90_passes_fp8linear(self):
        """Test fp8linear passes when _get_arch_info returns 90 (Hopper)."""
        from paddleformers.quantization.quantization_config import QuantizationConfig

        with patch("paddleformers.quantization.quantization_config._get_arch_info", return_value=90):
            config = QuantizationConfig(weight_quantize_algo="fp8linear")
            self.assertEqual(config.weight_quantize_algo, "fp8linear")

    def test_init_with_dense_and_moe_quant_type(self):
        """Test dense_quant_type and moe_quant_type parameters."""
        from paddleformers.quantization.quantization_config import QuantizationConfig

        config = QuantizationConfig(
            dense_quant_type="wint4",
            moe_quant_type="wint8",
        )
        self.assertEqual(config.dense_quant_type, "wint4")
        self.assertEqual(config.moe_quant_type, "wint8")

    def test_init_with_quantization_and_linear_list(self):
        """Test quantization and quantization_linear_list parameters."""
        from paddleformers.quantization.quantization_config import QuantizationConfig

        config = QuantizationConfig(
            quantization="a8w8",
            quantization_linear_list=[".*mlp.*", ".*self_attn.*"],
        )
        self.assertEqual(config.quantization, "a8w8")
        self.assertEqual(config.quantization_linear_list, [".*mlp.*", ".*self_attn.*"])


class TestQuantizationConfigMethods(unittest.TestCase):
    """Tests for QuantizationConfig methods."""

    def test_fp8_format_property_hybrid(self):
        """Test fp8_format property with 'hybrid' type."""
        from paddleformers.quantization.quantization_config import QuantizationConfig

        config = QuantizationConfig(fp8_format_type="hybrid")
        fmt = config.fp8_format
        self.assertEqual(fmt["weight"], "float8_e4m3fn")
        self.assertEqual(fmt["activation"], "float8_e4m3fn")
        self.assertEqual(fmt["grad_output"], "float8_e5m2")

    def test_fp8_format_property_e4m3(self):
        """Test fp8_format property with 'e4m3' type."""
        from paddleformers.quantization.quantization_config import QuantizationConfig

        config = QuantizationConfig(fp8_format_type="e4m3")
        fmt = config.fp8_format
        self.assertEqual(fmt["weight"], "float8_e4m3fn")
        self.assertEqual(fmt["activation"], "float8_e4m3fn")
        self.assertEqual(fmt["grad_output"], "float8_e4m3fn")

    def test_is_weight_quantize_none(self):
        """Test is_weight_quantize returns False when weight_quantize_algo is None."""
        from paddleformers.quantization.quantization_config import QuantizationConfig

        config = QuantizationConfig()
        self.assertFalse(config.is_weight_quantize())

    def test_is_weight_quantize_dict(self):
        """Test is_weight_quantize returns True for dict weight_quantize_algo."""
        from paddleformers.quantization.quantization_config import QuantizationConfig

        config = QuantizationConfig(weight_quantize_algo={"weight_only_int8": [".*mlp.*"]})
        self.assertTrue(config.is_weight_quantize())

    def test_is_weight_quantize_string(self):
        """Test is_weight_quantize returns True for supported string algos."""
        from paddleformers.quantization.quantization_config import QuantizationConfig

        non_fp8 = [
            "weight_only_int8",
            "weight_only_int4",
            "llm.int8",
            "nf4",
            "fp4",
            "a8w8",
            "a8w8linear",
            "a8w4linear",
        ]
        for algo in non_fp8:
            config = QuantizationConfig(weight_quantize_algo=algo)
            self.assertTrue(config.is_weight_quantize())

        # fp8linear requires Hopper GPU
        with patch("paddleformers.quantization.quantization_config._get_arch_info", return_value=89):
            config = QuantizationConfig(weight_quantize_algo="fp8linear")
            self.assertTrue(config.is_weight_quantize())

    def test_is_weight_quantize_none_after_string_tests(self):
        """Test is_weight_quantize returns False when weight_quantize_algo is None."""
        from paddleformers.quantization.quantization_config import QuantizationConfig

        config = QuantizationConfig()
        self.assertFalse(config.is_weight_quantize())

    def test_is_support_merge_tensor_parallel_true(self):
        """Test is_support_merge_tensor_parallel returns True for non-listed algos."""
        from paddleformers.quantization.quantization_config import QuantizationConfig

        non_fp8 = ["nf4", "fp4", "a8w8linear", "a8w4linear"]
        for algo in non_fp8:
            config = QuantizationConfig(weight_quantize_algo=algo)
            self.assertTrue(config.is_support_merge_tensor_parallel())

        # fp8linear requires Hopper GPU
        with patch("paddleformers.quantization.quantization_config._get_arch_info", return_value=89):
            config = QuantizationConfig(weight_quantize_algo="fp8linear")
            self.assertTrue(config.is_support_merge_tensor_parallel())

    def test_is_support_merge_tensor_parallel_false(self):
        """Test is_support_merge_tensor_parallel returns False for listed algos."""
        from paddleformers.quantization.quantization_config import QuantizationConfig

        for algo in ["weight_only_int8", "weight_only_int4", "llm.int8", "a8w8"]:
            config = QuantizationConfig(weight_quantize_algo=algo)
            self.assertFalse(config.is_support_merge_tensor_parallel())

    def test_is_support_merge_tensor_parallel_none(self):
        """Test is_support_merge_tensor_parallel with None returns True (else branch)."""
        from paddleformers.quantization.quantization_config import QuantizationConfig

        config = QuantizationConfig()
        self.assertTrue(config.is_support_merge_tensor_parallel())


class TestQuantizationConfigFromDict(unittest.TestCase):
    """Tests for QuantizationConfig.from_dict classmethod."""

    def test_from_dict_basic(self):
        """Test from_dict creates config from dictionary."""
        from paddleformers.quantization.quantization_config import QuantizationConfig

        config_dict = {"weight_quantize_algo": "weight_only_int8", "quant_type": "weight_only_int8"}
        config = QuantizationConfig.from_dict(config_dict)
        self.assertEqual(config.weight_quantize_algo, "weight_only_int8")
        self.assertEqual(config.quant_type, "weight_only_int8")

    def test_from_dict_with_extra_kwargs(self):
        """Test from_dict with extra kwargs and return_unused_kwargs=False."""
        from paddleformers.quantization.quantization_config import QuantizationConfig

        config_dict = {"weight_quantize_algo": "nf4"}
        config = QuantizationConfig.from_dict(config_dict, shift=True, unknown_param="ignored")
        self.assertEqual(config.weight_quantize_algo, "nf4")
        self.assertTrue(config.shift)

    def test_from_dict_return_unused_kwargs(self):
        """Test from_dict with return_unused_kwargs=True returns tuple."""
        from paddleformers.quantization.quantization_config import QuantizationConfig

        config_dict = {"weight_quantize_algo": "a8w8"}
        config, unused = QuantizationConfig.from_dict(config_dict, return_unused_kwargs=True, unknown_param="value")
        self.assertEqual(config.weight_quantize_algo, "a8w8")
        self.assertEqual(unused, {"unknown_param": "value"})

    def test_from_dict_return_unused_kwargs_empty(self):
        """Test from_dict with return_unused_kwargs=True and no unused kwargs."""
        from paddleformers.quantization.quantization_config import QuantizationConfig

        config_dict = {"weight_quantize_algo": "llm.int8", "shift": True}
        config, unused = QuantizationConfig.from_dict(config_dict, return_unused_kwargs=True)
        self.assertEqual(config.weight_quantize_algo, "llm.int8")
        self.assertEqual(unused, {})


class TestQuantizationConfigSerialization(unittest.TestCase):
    """Tests for QuantizationConfig serialization methods."""

    def test_to_dict(self):
        """Test to_dict returns a copy of __dict__."""
        from paddleformers.quantization.quantization_config import QuantizationConfig

        config = QuantizationConfig(weight_quantize_algo="nf4", group_size=32)
        d = config.to_dict()
        self.assertEqual(d["weight_quantize_algo"], "nf4")
        self.assertEqual(d["group_size"], 32)
        # Verify it's a copy
        d["weight_quantize_algo"] = "modified"
        self.assertEqual(config.weight_quantize_algo, "nf4")

    def test_to_json_file(self):
        """Test to_json_file writes valid JSON to file."""
        from paddleformers.quantization.quantization_config import QuantizationConfig

        config = QuantizationConfig(weight_quantize_algo="a8w8", shift=True)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json_path = f.name
        try:
            config.to_json_file(json_path)
            with open(json_path, "r") as f:
                data = json.load(f)
            self.assertEqual(data["weight_quantize_algo"], "a8w8")
            self.assertTrue(data["shift"])
        finally:
            os.unlink(json_path)

    def test_to_json_string_with_diff(self):
        """Test to_json_string with use_diff=True only includes changed values."""
        from paddleformers.quantization.quantization_config import QuantizationConfig

        config = QuantizationConfig(weight_quantize_algo="nf4")
        json_str = config.to_json_string(use_diff=True)
        data = json.loads(json_str)
        self.assertIn("weight_quantize_algo", data)
        # shift should not be in diff since it's False by default
        self.assertNotIn("shift", data)

    def test_to_json_string_without_diff(self):
        """Test to_json_string with use_diff=False includes all values."""
        from paddleformers.quantization.quantization_config import QuantizationConfig

        config = QuantizationConfig()
        json_str = config.to_json_string(use_diff=False)
        data = json.loads(json_str)
        self.assertIn("shift", data)
        self.assertIn("smooth", data)
        self.assertIn("weight_quantize_algo", data)

    def test_to_diff_dict(self):
        """Test to_diff_dict only contains values different from defaults."""
        from paddleformers.quantization.quantization_config import QuantizationConfig

        config = QuantizationConfig(weight_quantize_algo="weight_only_int4", group_size=64)
        diff = config.to_diff_dict()
        self.assertIn("weight_quantize_algo", diff)
        self.assertIn("group_size", diff)
        self.assertNotIn("shift", diff)  # default is False

    def test_to_diff_dict_all_defaults(self):
        """Test to_diff_dict with all defaults returns empty dict."""
        from paddleformers.quantization.quantization_config import QuantizationConfig

        config = QuantizationConfig()
        diff = config.to_diff_dict()
        # Most values should be absent since they match defaults
        # Note: some values like act_quant_method get mapped, so they may differ
        self.assertNotIn("shift", diff)
        self.assertNotIn("smooth", diff)

    def test_repr(self):
        """Test __repr__ returns a string containing class name."""
        from paddleformers.quantization.quantization_config import QuantizationConfig

        config = QuantizationConfig(weight_quantize_algo="a8w8")
        repr_str = repr(config)
        self.assertIn("QuantizationConfig", repr_str)


if __name__ == "__main__":
    unittest.main()
