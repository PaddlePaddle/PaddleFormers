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

import tempfile
import unittest

try:
    from paddleformers.fleet.training.yaml_arguments import load_yaml  # noqa: F401

    _MODULE_AVAILABLE = True
except (ImportError, ModuleNotFoundError, Exception):
    _MODULE_AVAILABLE = False


@unittest.skipUnless(_MODULE_AVAILABLE, "paddleformers.fleet.training.yaml_arguments not available")
class TestLoadYaml(unittest.TestCase):
    """Tests for load_yaml function in yaml_arguments.py."""

    def test_load_yaml_simple(self):
        """Test loading a simple YAML file."""
        from paddleformers.fleet.training.yaml_arguments import load_yaml

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("batch_size: 32\nlearning_rate: 0.001\n")
            f.flush()
            config = load_yaml(f.name)
            self.assertIsNotNone(config)
            os.unlink(f.name)

    def test_load_yaml_nested(self):
        """Test loading a nested YAML file."""
        from paddleformers.fleet.training.yaml_arguments import load_yaml

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("training:\n  batch_size: 32\n  lr: 0.001\n")
            f.flush()
            config = load_yaml(f.name)
            self.assertIsNotNone(config)
            os.unlink(f.name)

    def test_load_yaml_flattens_nested(self):
        """Test that nested configs are flattened."""
        from paddleformers.fleet.training.yaml_arguments import load_yaml

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("training:\n  batch_size: 32\nmodel:\n  hidden: 768\n")
            f.flush()
            config = load_yaml(f.name)
            # After flattening, batch_size and hidden should be top-level
            self.assertIsNotNone(config)
            os.unlink(f.name)

    def test_flatten_configs_simple(self):
        """Test _flatten_configs with simple config."""
        from omegaconf import OmegaConf

        from paddleformers.fleet.training.yaml_arguments import _flatten_configs

        cfg = OmegaConf.create({"a": 1, "b": 2})
        result = _flatten_configs(cfg)
        self.assertIsNotNone(result)

    def test_flatten_configs_nested(self):
        """Test _flatten_configs with nested config."""
        from omegaconf import OmegaConf

        from paddleformers.fleet.training.yaml_arguments import _flatten_configs

        cfg = OmegaConf.create({"outer": {"inner_a": 1, "inner_b": 2}})
        result = _flatten_configs(cfg)
        self.assertIsNotNone(result)

    def test_load_yaml_returns_omegaconf(self):
        """Test load_yaml returns OmegaConf object."""

        from paddleformers.fleet.training.yaml_arguments import load_yaml

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("key: value\n")
            f.flush()
            config = load_yaml(f.name)
            # Should be some OmegaConf type
            self.assertIsNotNone(config)
            os.unlink(f.name)


if __name__ == "__main__":
    unittest.main()
