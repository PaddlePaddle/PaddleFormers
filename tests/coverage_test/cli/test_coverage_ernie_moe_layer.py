# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

from paddleformers.cli.train.ernie_pretrain.models.moe.moe_layer import (
    GateOutput,
    MOELayer,
    MoEStatics,
)


class TestMOELayer(unittest.TestCase):
    """Tests for MOELayer class."""

    def test_import_succeeds(self):
        """Test that MOELayer can be imported."""
        self.assertTrue(MOELayer is not None)

    def test_class_exists(self):
        """Test that MOELayer class exists and is callable."""
        self.assertTrue(callable(MOELayer))


class TestGateOutput(unittest.TestCase):
    """Tests for GateOutput namedtuple."""

    def test_gate_output_exists(self):
        """Test that GateOutput exists."""
        self.assertIsNotNone(GateOutput)


class TestMoEStatics(unittest.TestCase):
    """Tests for MoEStatics class."""

    def test_moe_statics_exists(self):
        """Test that MoEStatics exists."""
        self.assertIsNotNone(MoEStatics)


if __name__ == "__main__":
    unittest.main()
