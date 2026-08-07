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

import unittest

import paddle

try:
    from paddleformers.fleet.triton_ops.sigmoid_gate_fusion import (
        SigmoidGateFusionTriton,  # noqa: F401
    )

    _MODULE_AVAILABLE = True
except (ImportError, ModuleNotFoundError, Exception):
    _MODULE_AVAILABLE = False


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestSigmoidGateFusionStructure(unittest.TestCase):
    """Tests for sigmoid gate fusion module structure."""

    def test_sigmoid_gate_fusion_import(self):
        """Test SigmoidGateFusionTriton can be imported."""
        try:
            from paddleformers.fleet.triton_ops.sigmoid_gate_fusion import (
                SigmoidGateFusionTriton,
            )

            self.assertIsNotNone(SigmoidGateFusionTriton)
        except ImportError:
            self.skipTest("paddlefleet_ops not installed")

    def test_sigmoid_gate_is_pylayer(self):
        """Test SigmoidGateFusionTriton is a PyLayer subclass."""
        try:
            from paddleformers.fleet.triton_ops.sigmoid_gate_fusion import (
                SigmoidGateFusionTriton,
            )

            self.assertTrue(
                issubclass(SigmoidGateFusionTriton, paddle.autograd.PyLayer)
            )
        except ImportError:
            self.skipTest("paddlefleet_ops not installed")

    def test_fwd_kernel_exists(self):
        """Test fused_sigmoid_gate_fwd_kernel exists."""
        try:
            from paddleformers.fleet.triton_ops.sigmoid_gate_fusion import (
                fused_sigmoid_gate_fwd_kernel,
            )

            self.assertIsNotNone(fused_sigmoid_gate_fwd_kernel)
        except ImportError:
            self.skipTest("paddlefleet_ops not installed")

    def test_bwd_kernel_exists(self):
        """Test fused_sigmoid_gate_bwd_kernel exists."""
        try:
            from paddleformers.fleet.triton_ops.sigmoid_gate_fusion import (
                fused_sigmoid_gate_bwd_kernel,
            )

            self.assertIsNotNone(fused_sigmoid_gate_bwd_kernel)
        except ImportError:
            self.skipTest("paddlefleet_ops not installed")

    def test_sigmoid_precise_exists(self):
        """Test _sigmoid_precise helper exists."""
        try:
            from paddleformers.fleet.triton_ops.sigmoid_gate_fusion import (
                _sigmoid_precise,
            )

            self.assertIsNotNone(_sigmoid_precise)
        except ImportError:
            self.skipTest("paddlefleet_ops not installed")


if __name__ == "__main__":
    unittest.main()
