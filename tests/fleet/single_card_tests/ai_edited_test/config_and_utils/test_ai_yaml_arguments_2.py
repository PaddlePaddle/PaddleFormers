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
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)


# Extra tests for paddleformers.fleet/training/yaml_arguments.py
# Focus on edge cases and additional scenarios

import tempfile
import unittest

try:
    from omegaconf import OmegaConf

    HAVE_OMEGACONF = True
except ImportError:
    HAVE_OMEGACONF = False


@unittest.skipUnless(HAVE_OMEGACONF, "omegaconf not available in CI")
class TestFlattenConfigsEdgeCases(unittest.TestCase):
    """Edge case tests for _flatten_configs."""

    def test_single_key_nested(self):
        """Test _flatten_configs with single key deeply nested."""
        from paddleformers.fleet.training.yaml_arguments import _flatten_configs

        cfg = OmegaConf.create({"a": {"b": {"c": {"d": 1}}}})
        result = _flatten_configs(cfg)
        self.assertEqual(result.d, 1)
        self.assertEqual(len(result), 1)

    def test_mixed_nested_and_flat(self):
        """Test _flatten_configs with mixed nested and flat keys."""
        from paddleformers.fleet.training.yaml_arguments import _flatten_configs

        cfg = OmegaConf.create(
            {
                "flat_key": "flat_value",
                "nested": {"inner_key": "inner_value"},
            }
        )
        result = _flatten_configs(cfg)
        self.assertEqual(result.flat_key, "flat_value")
        self.assertEqual(result.inner_key, "inner_value")

    def test_numeric_values(self):
        """Test _flatten_configs preserves numeric types."""
        from paddleformers.fleet.training.yaml_arguments import _flatten_configs

        cfg = OmegaConf.create({"model": {"layers": 12, "dim": 768.5, "dropout": 0.1}})
        result = _flatten_configs(cfg)
        self.assertEqual(result.layers, 12)
        self.assertAlmostEqual(result.dim, 768.5)
        self.assertAlmostEqual(result.dropout, 0.1)

    def test_string_values(self):
        """Test _flatten_configs preserves string values."""
        from paddleformers.fleet.training.yaml_arguments import _flatten_configs

        cfg = OmegaConf.create({"model": {"name": "gpt2", "type": "decoder"}})
        result = _flatten_configs(cfg)
        self.assertEqual(result.name, "gpt2")
        self.assertEqual(result.type, "decoder")

    def test_boolean_values(self):
        """Test _flatten_configs preserves boolean values."""
        from paddleformers.fleet.training.yaml_arguments import _flatten_configs

        cfg = OmegaConf.create({"training": {"use_fp16": True, "use_bf16": False}})
        result = _flatten_configs(cfg)
        self.assertTrue(result.use_fp16)
        self.assertFalse(result.use_bf16)


@unittest.skipUnless(HAVE_OMEGACONF, "omegaconf not available in CI")
class TestLoadYamlEdgeCases(unittest.TestCase):
    """Edge case tests for load_yaml."""

    def test_load_yaml_with_boolean(self):
        """Test loading YAML with boolean values."""
        from paddleformers.fleet.training.yaml_arguments import load_yaml

        yaml_content = """
training:
  use_cuda: true
  debug: false
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            result = load_yaml(f.name)
            self.assertTrue(result.use_cuda)
            self.assertFalse(result.debug)
            os.unlink(f.name)

    def test_load_yaml_with_float(self):
        """Test loading YAML with float values."""
        from paddleformers.fleet.training.yaml_arguments import load_yaml

        yaml_content = """
training:
  learning_rate: 0.0001
  weight_decay: 1e-4
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            result = load_yaml(f.name)
            self.assertAlmostEqual(result.learning_rate, 0.0001)
            self.assertAlmostEqual(result.weight_decay, 1e-4)
            os.unlink(f.name)

    def test_load_yaml_with_int(self):
        """Test loading YAML with integer values."""
        from paddleformers.fleet.training.yaml_arguments import load_yaml

        yaml_content = """
model:
  num_layers: 24
  hidden_size: 1024
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            result = load_yaml(f.name)
            self.assertEqual(result.num_layers, 24)
            self.assertEqual(result.hidden_size, 1024)
            os.unlink(f.name)

    def test_load_yaml_deep_nesting(self):
        """Test loading YAML with deep nesting."""
        from paddleformers.fleet.training.yaml_arguments import load_yaml

        yaml_content = """
level1:
  level2:
    level3:
      value: deep
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            result = load_yaml(f.name)
            self.assertEqual(result.value, "deep")
            os.unlink(f.name)

    def test_load_yaml_multiple_sections(self):
        """Test loading YAML with multiple sections all flattened."""
        from paddleformers.fleet.training.yaml_arguments import load_yaml

        yaml_content = """
model:
  hidden_size: 768
training:
  batch_size: 32
optimizer:
  lr: 0.001
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            result = load_yaml(f.name)
            self.assertEqual(result.hidden_size, 768)
            self.assertEqual(result.batch_size, 32)
            self.assertAlmostEqual(result.lr, 0.001)
            os.unlink(f.name)

    def test_load_yaml_string_path(self):
        """Test load_yaml accepts string path."""
        from paddleformers.fleet.training.yaml_arguments import load_yaml

        yaml_content = "key: value\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            result = load_yaml(f.name)
            self.assertEqual(result.key, "value")
            os.unlink(f.name)


if __name__ == "__main__":
    unittest.main()
