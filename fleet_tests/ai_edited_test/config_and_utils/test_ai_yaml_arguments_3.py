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


# Extra tests for paddlefleet/training/yaml_arguments.py
# Focus on: error handling and more complex YAML structures

import tempfile
import unittest

try:
    from omegaconf import OmegaConf

    HAVE_OMEGACONF = True
except ImportError:
    HAVE_OMEGACONF = False


@unittest.skipUnless(HAVE_OMEGACONF, "omegaconf not available in CI")
class TestFlattenConfigsWithNone(unittest.TestCase):
    """Tests for _flatten_configs with None values."""

    def test_none_values(self):
        """Test _flatten_configs with None values in nested dict."""
        from paddleformers.fleet.training.yaml_arguments import _flatten_configs

        cfg = OmegaConf.create({"model": {"name": None, "size": 10}})
        result = _flatten_configs(cfg)
        self.assertIsNone(result.name)
        self.assertEqual(result.size, 10)

    def test_empty_nested_dict(self):
        """Test _flatten_configs with empty nested dict."""
        from paddleformers.fleet.training.yaml_arguments import _flatten_configs

        cfg = OmegaConf.create({"model": {}})
        result = _flatten_configs(cfg)
        # Empty nested dict should produce no keys
        self.assertEqual(len(result), 0)


@unittest.skipUnless(HAVE_OMEGACONF, "omegaconf not available in CI")
class TestLoadYamlSpecialCases(unittest.TestCase):
    """Special case tests for load_yaml."""

    def test_load_yaml_with_null(self):
        """Test loading YAML with null values."""
        from paddleformers.fleet.training.yaml_arguments import load_yaml

        yaml_content = """
model:
  name: null
  size: 10
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()
            result = load_yaml(f.name)
            self.assertIsNone(result.name)
            self.assertEqual(result.size, 10)
            os.unlink(f.name)

    def test_load_yaml_with_quoted_strings(self):
        """Test loading YAML with quoted strings."""
        from paddleformers.fleet.training.yaml_arguments import load_yaml

        yaml_content = """
model:
  name: "my-model-name"
  path: '/path/to/model'
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()
            result = load_yaml(f.name)
            self.assertEqual(result.name, "my-model-name")
            self.assertEqual(result.path, "/path/to/model")
            os.unlink(f.name)

    def test_load_yaml_with_multiline_string(self):
        """Test loading YAML with multiline string values."""
        from paddleformers.fleet.training.yaml_arguments import load_yaml

        yaml_content = """
model:
  description: |
    This is a long
    description that spans
    multiple lines.
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()
            result = load_yaml(f.name)
            self.assertIn("long", result.description)
            self.assertIn("spans", result.description)
            os.unlink(f.name)

    def test_load_yaml_with_anchors(self):
        """Test loading YAML with anchors and aliases."""
        from paddleformers.fleet.training.yaml_arguments import load_yaml

        yaml_content = """
defaults: &defaults
  batch_size: 32
  learning_rate: 0.001

training:
  <<: *defaults
  epochs: 10
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()
            result = load_yaml(f.name)
            self.assertEqual(result.batch_size, 32)
            self.assertAlmostEqual(result.learning_rate, 0.001)
            self.assertEqual(result.epochs, 10)
            os.unlink(f.name)

    def test_load_yaml_scientific_notation(self):
        """Test loading YAML with scientific notation."""
        from paddleformers.fleet.training.yaml_arguments import load_yaml

        yaml_content = """
training:
  lr: 1.0e-4
  weight_decay: 5.0e-2
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()
            result = load_yaml(f.name)
            self.assertAlmostEqual(result.lr, 1.0e-4)
            self.assertAlmostEqual(result.weight_decay, 5.0e-2)
            os.unlink(f.name)


if __name__ == "__main__":
    unittest.main()
