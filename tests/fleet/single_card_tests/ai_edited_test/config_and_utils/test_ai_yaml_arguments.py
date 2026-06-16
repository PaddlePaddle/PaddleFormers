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
import tempfile

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)


# Tests for src/paddleformers.fleet/training/yaml_arguments.py
# Test _flatten_configs, load_yaml

import unittest

try:
    from omegaconf import DictConfig, OmegaConf

    HAVE_OMEGACONF = True
except ImportError:
    HAVE_OMEGACONF = False


@unittest.skipUnless(HAVE_OMEGACONF, "omegaconf not available in CI")
class TestFlattenConfigs(unittest.TestCase):
    """Tests for _flatten_configs function."""

    def test_flat_dict(self):
        """Test _flatten_configs with a flat dictionary."""
        from paddleformers.fleet.training.yaml_arguments import _flatten_configs

        cfg = OmegaConf.create({"a": 1, "b": 2})
        result = _flatten_configs(cfg)
        self.assertEqual(result.a, 1)
        self.assertEqual(result.b, 2)

    def test_nested_dict(self):
        """Test _flatten_configs with nested dictionaries."""
        from paddleformers.fleet.training.yaml_arguments import _flatten_configs

        cfg = OmegaConf.create(
            {
                "outer": {"inner": 42},
                "top": "hello",
            }
        )
        result = _flatten_configs(cfg)
        self.assertEqual(result.inner, 42)
        self.assertEqual(result.top, "hello")

    def test_deeply_nested_dict(self):
        """Test _flatten_configs with deeply nested structures."""
        from paddleformers.fleet.training.yaml_arguments import _flatten_configs

        cfg = OmegaConf.create(
            {
                "level1": {"level2": {"level3": {"value": 99}}},
                "simple": 1,
            }
        )
        result = _flatten_configs(cfg)
        self.assertEqual(result.value, 99)
        self.assertEqual(result.simple, 1)

    def test_multiple_nested_keys(self):
        """Test _flatten_configs with multiple nested keys."""
        from paddleformers.fleet.training.yaml_arguments import _flatten_configs

        cfg = OmegaConf.create(
            {
                "model": {"hidden_size": 1024, "num_layers": 24},
                "training": {"lr": 0.001, "batch_size": 32},
            }
        )
        result = _flatten_configs(cfg)
        self.assertEqual(result.hidden_size, 1024)
        self.assertEqual(result.num_layers, 24)
        self.assertEqual(result.lr, 0.001)
        self.assertEqual(result.batch_size, 32)

    def test_empty_dict(self):
        """Test _flatten_configs with empty dictionary."""
        from paddleformers.fleet.training.yaml_arguments import _flatten_configs

        cfg = OmegaConf.create({})
        result = _flatten_configs(cfg)
        self.assertEqual(len(result), 0)

    def test_single_level_dict(self):
        """Test _flatten_configs with a single level dict."""
        from paddleformers.fleet.training.yaml_arguments import _flatten_configs

        cfg = OmegaConf.create({"key1": "val1", "key2": 3.14})
        result = _flatten_configs(cfg)
        self.assertEqual(result.key1, "val1")
        self.assertAlmostEqual(result.key2, 3.14)

    def test_result_is_dictconfig(self):
        """Test that result is a DictConfig."""
        from paddleformers.fleet.training.yaml_arguments import _flatten_configs

        cfg = OmegaConf.create({"a": {"b": 1}})
        result = _flatten_configs(cfg)
        self.assertIsInstance(result, DictConfig)

    def test_overlapping_keys_from_different_parents(self):
        """Test behavior when different parents have same key name."""
        from paddleformers.fleet.training.yaml_arguments import _flatten_configs

        cfg = OmegaConf.create(
            {
                "parent1": {"key": "value1"},
                "parent2": {"key": "value2"},
            }
        )
        result = _flatten_configs(cfg)
        # Later one wins (parent2)
        self.assertEqual(result.key, "value2")

    def test_mixed_types(self):
        """Test _flatten_configs with mixed value types."""
        from paddleformers.fleet.training.yaml_arguments import _flatten_configs

        cfg = OmegaConf.create(
            {
                "int_val": 42,
                "float_val": 3.14,
                "str_val": "hello",
                "bool_val": True,
                "nested": {"none_val": None},
            }
        )
        result = _flatten_configs(cfg)
        self.assertEqual(result.int_val, 42)
        self.assertAlmostEqual(result.float_val, 3.14)
        self.assertEqual(result.str_val, "hello")
        self.assertTrue(result.bool_val)
        self.assertIsNone(result.none_val)


@unittest.skipUnless(HAVE_OMEGACONF, "omegaconf not available in CI")
class TestLoadYaml(unittest.TestCase):
    """Tests for load_yaml function."""

    def test_load_simple_yaml(self):
        """Test loading a simple YAML file."""
        from paddleformers.fleet.training.yaml_arguments import load_yaml

        yaml_content = """
a: 1
b: 2
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()
            result = load_yaml(f.name)
            self.assertEqual(result.a, 1)
            self.assertEqual(result.b, 2)
            os.unlink(f.name)

    def test_load_nested_yaml(self):
        """Test loading nested YAML file."""
        from paddleformers.fleet.training.yaml_arguments import load_yaml

        yaml_content = """
model:
  hidden_size: 1024
  num_layers: 24
training:
  lr: 0.001
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()
            result = load_yaml(f.name)
            self.assertEqual(result.hidden_size, 1024)
            self.assertEqual(result.num_layers, 24)
            self.assertAlmostEqual(result.lr, 0.001)
            os.unlink(f.name)

    def test_load_empty_yaml(self):
        """Test loading an empty YAML file."""
        from paddleformers.fleet.training.yaml_arguments import load_yaml

        yaml_content = ""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()
            result = load_yaml(f.name)
            self.assertEqual(len(result), 0)
            os.unlink(f.name)

    def test_load_yaml_with_list(self):
        """Test loading YAML with list values."""
        from paddleformers.fleet.training.yaml_arguments import load_yaml

        yaml_content = """
items:
  - 1
  - 2
  - 3
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()
            result = load_yaml(f.name)
            # Lists should be skipped (not DictConfig), so items should not appear
            os.unlink(f.name)

    def test_file_not_found(self):
        """Test loading a non-existent YAML file raises error."""
        from paddleformers.fleet.training.yaml_arguments import load_yaml

        with self.assertRaises(FileNotFoundError):
            load_yaml("/nonexistent/path/config.yaml")


@unittest.skipUnless(HAVE_OMEGACONF, "omegaconf not available in CI")
class TestYamlArgumentsModule(unittest.TestCase):
    """Tests for module structure."""

    def test_module_exports(self):
        """Test that expected functions are exported."""
        import paddleformers.fleet.training.yaml_arguments as ya

        self.assertTrue(hasattr(ya, "_flatten_configs"))
        self.assertTrue(hasattr(ya, "load_yaml"))
        self.assertTrue(callable(ya._flatten_configs))
        self.assertTrue(callable(ya.load_yaml))


if __name__ == "__main__":
    unittest.main()
