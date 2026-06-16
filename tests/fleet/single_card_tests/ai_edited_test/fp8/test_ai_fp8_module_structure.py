# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

import importlib.util
import unittest


def _load_fp8_module(mod_name, filename):
    """Load fp8 submodule directly from source to avoid deep_gemm import."""
    src_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "..",
        "..",
        "src",
        "paddleformers.fleet",
        "fp8",
        filename,
    )
    if not os.path.exists(src_path):
        return None
    spec = importlib.util.spec_from_file_location(mod_name, src_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestFP8ModuleStructure(unittest.TestCase):
    """Tests for fp8 module structure using direct source loading."""

    def test_fp8_module_quantization_source_exists(self):
        """Test fp8 quantization source file exists and loads."""
        mod = _load_fp8_module("fp8_quantization", "quantization.py")
        self.assertIsNotNone(mod)

    def test_fp8_has_quantization(self):
        """Test fp8 has quantization submodule with get_quant_func."""
        mod = _load_fp8_module("fp8_quantization", "quantization.py")
        self.assertTrue(hasattr(mod, "get_quant_func"))

    def test_fp8_has_utils(self):
        """Test fp8 has utils submodule with is_fp8_tensor."""
        mod = _load_fp8_module("fp8_utils", "utils.py")
        self.assertTrue(hasattr(mod, "is_fp8_tensor"))

    def test_fp8_has_linear(self):
        """Test fp8 has linear source file."""
        src_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            "..",
            "..",
            "src",
            "paddleformers.fleet",
            "fp8",
            "linear.py",
        )
        if not os.path.exists(src_path):
            self.skipTest("fp8/linear.py source not found")
        # The linear.py module imports deep_gem which requires Hopper GPU,
        # so we just verify the source file exists rather than loading it
        self.assertTrue(os.path.exists(src_path))


class TestFP8QuantizationModule(unittest.TestCase):
    """Tests for fp8 quantization module."""

    def test_get_quant_func_callable(self):
        """Test get_quant_func is callable."""
        mod = _load_fp8_module("fp8_quantization", "quantization.py")
        self.assertTrue(callable(mod.get_quant_func))

    def test_get_quant_func_module_has_expected_names(self):
        """Test module exports expected names."""
        mod = _load_fp8_module("fp8_quantization", "quantization.py")
        self.assertTrue(hasattr(mod, "get_quant_func"))


class TestFP8UtilsModule(unittest.TestCase):
    """Tests for fp8 utils module."""

    def test_is_fp8_tensor_callable(self):
        """Test is_fp8_tensor is callable."""
        mod = _load_fp8_module("fp8_utils", "utils.py")
        self.assertTrue(callable(mod.is_fp8_tensor))


if __name__ == "__main__":
    unittest.main()
