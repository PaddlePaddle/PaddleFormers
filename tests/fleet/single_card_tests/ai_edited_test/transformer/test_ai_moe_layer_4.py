# Copyright (c) 2026 PaddleFleet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless distributed on applicable law or agreed to in writing, software
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

import unittest

from paddleformers.fleet.transformer.moe.moe_layer import MoELayer


class TestMoELayerMethodExistence(unittest.TestCase):
    """Tests for MoELayer method existence."""

    def test_has_forward(self):
        """MoELayer should have a forward method."""
        self.assertTrue(hasattr(MoELayer, "forward"))

    def test_has_use_fp8(self):
        """MoELayer should have use_fp8 method."""
        self.assertTrue(hasattr(MoELayer, "use_fp8"))

    def test_has_fp8_quant_weight(self):
        """MoELayer should have fp8_quant_weight method."""
        self.assertTrue(hasattr(MoELayer, "fp8_quant_weight"))

    def test_has_backward(self):
        """MoELayer should have backward method."""
        self.assertTrue(hasattr(MoELayer, "backward"))

    def test_has_aux_loss_compute(self):
        """MoELayer should have aux_loss_compute method."""
        self.assertTrue(hasattr(MoELayer, "aux_loss_compute"))


class TestMoELayerFP8Methods(unittest.TestCase):
    """Tests for MoELayer FP8 methods."""

    def test_use_fp8_is_callable(self):
        """use_fp8 should be callable."""
        self.assertTrue(callable(MoELayer.use_fp8))

    def test_fp8_quant_weight_is_callable(self):
        """fp8_quant_weight should be callable."""
        self.assertTrue(callable(MoELayer.fp8_quant_weight))


if __name__ == "__main__":
    unittest.main()
