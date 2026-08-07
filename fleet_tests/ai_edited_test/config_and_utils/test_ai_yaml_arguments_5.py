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

import tempfile
import unittest

try:
    from paddleformers.fleet.training.yaml_arguments import load_yaml  # noqa: F401

    _MODULE_AVAILABLE = True
except (ImportError, ModuleNotFoundError, Exception):
    _MODULE_AVAILABLE = False


@unittest.skipUnless(
    _MODULE_AVAILABLE, "paddleformers.fleet.training.yaml_arguments not available"
)
class TestYamlArgumentsEdgeCases(unittest.TestCase):
    """Edge case tests for yaml_arguments module."""

    def test_load_yaml_empty_file(self):
        """Test loading an empty YAML file."""
        from paddleformers.fleet.training.yaml_arguments import load_yaml

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("")
            f.flush()
            config = load_yaml(f.name)
            self.assertIsNotNone(config)
            os.unlink(f.name)

    def test_load_yaml_single_value(self):
        """Test loading a YAML file with a single value."""
        from paddleformers.fleet.training.yaml_arguments import load_yaml

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("key: value\n")
            f.flush()
            config = load_yaml(f.name)
            self.assertIsNotNone(config)
            os.unlink(f.name)

    def test_load_yaml_deep_nesting(self):
        """Test loading a YAML file with deeply nested structure."""
        from paddleformers.fleet.training.yaml_arguments import load_yaml

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("a:\n  b:\n    c:\n      d: 42\n")
            f.flush()
            config = load_yaml(f.name)
            self.assertIsNotNone(config)
            os.unlink(f.name)

    def test_load_yaml_multiple_keys(self):
        """Test loading a YAML file with multiple top-level keys."""
        from paddleformers.fleet.training.yaml_arguments import load_yaml

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("key1: val1\nkey2: 42\nkey3: true\n")
            f.flush()
            config = load_yaml(f.name)
            self.assertIsNotNone(config)
            os.unlink(f.name)

    def test_load_yaml_list_values(self):
        """Test loading a YAML file with list values."""
        from paddleformers.fleet.training.yaml_arguments import load_yaml

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("items:\n  - a\n  - b\n  - c\n")
            f.flush()
            config = load_yaml(f.name)
            self.assertIsNotNone(config)
            os.unlink(f.name)

    def test_load_yaml_numeric_values(self):
        """Test loading a YAML file with numeric values."""
        from paddleformers.fleet.training.yaml_arguments import load_yaml

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("int_val: 42\nfloat_val: 3.14\n")
            f.flush()
            config = load_yaml(f.name)
            self.assertIsNotNone(config)
            os.unlink(f.name)


if __name__ == "__main__":
    unittest.main()
