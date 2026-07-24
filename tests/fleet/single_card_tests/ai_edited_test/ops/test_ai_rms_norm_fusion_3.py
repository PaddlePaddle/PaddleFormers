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
    from paddleformers.fleet.triton_ops.rms_norm_fusion import (  # noqa: F401
        RMSNormFusionTriton,
    )

    _MODULE_AVAILABLE = True
except (ImportError, ModuleNotFoundError, Exception):
    _MODULE_AVAILABLE = False


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestRMSNormFusionStructure(unittest.TestCase):
    """Tests for RMS norm fusion module structure."""

    def test_rms_norm_fusion_import(self):
        """Test RMSNormFusionTriton can be imported."""
        try:
            from paddleformers.fleet.triton_ops.rms_norm_fusion import (
                RMSNormFusionTriton,
            )

            self.assertIsNotNone(RMSNormFusionTriton)
        except ImportError:
            self.skipTest("paddlefleet_ops not installed")

    def test_rms_norm_is_pylayer(self):
        """Test RMSNormFusionTriton is a PyLayer subclass."""
        try:
            from paddleformers.fleet.triton_ops.rms_norm_fusion import (
                RMSNormFusionTriton,
            )

            self.assertTrue(
                issubclass(RMSNormFusionTriton, paddle.autograd.PyLayer)
            )
        except ImportError:
            self.skipTest("paddlefleet_ops not installed")

    def test_rms_norm_has_forward(self):
        """Test RMSNormFusionTriton has forward static method."""
        try:
            from paddleformers.fleet.triton_ops.rms_norm_fusion import (
                RMSNormFusionTriton,
            )

            self.assertTrue(hasattr(RMSNormFusionTriton, "forward"))
        except ImportError:
            self.skipTest("paddlefleet_ops not installed")

    def test_rms_norm_has_backward(self):
        """Test RMSNormFusionTriton has backward static method."""
        try:
            from paddleformers.fleet.triton_ops.rms_norm_fusion import (
                RMSNormFusionTriton,
            )

            self.assertTrue(hasattr(RMSNormFusionTriton, "backward"))
        except ImportError:
            self.skipTest("paddlefleet_ops not installed")

    def test_fwd_kernel_exists(self):
        """Test rms_norm_fwd_kernel is defined."""
        try:
            from paddleformers.fleet.triton_ops.rms_norm_fusion import (
                rms_norm_fwd_kernel,
            )

            self.assertIsNotNone(rms_norm_fwd_kernel)
        except ImportError:
            self.skipTest("paddlefleet_ops not installed")

    def test_bwd_dx_kernel_exists(self):
        """Test rms_norm_bwd_dx_kernel is defined."""
        try:
            from paddleformers.fleet.triton_ops.rms_norm_fusion import (
                rms_norm_bwd_dx_kernel,
            )

            self.assertIsNotNone(rms_norm_bwd_dx_kernel)
        except ImportError:
            self.skipTest("paddlefleet_ops not installed")

    def test_bwd_dw_partial_kernel_exists(self):
        """Test rms_norm_bwd_dw_partial_kernel is defined."""
        try:
            from paddleformers.fleet.triton_ops.rms_norm_fusion import (
                rms_norm_bwd_dw_partial_kernel,
            )

            self.assertIsNotNone(rms_norm_bwd_dw_partial_kernel)
        except ImportError:
            self.skipTest("paddlefleet_ops not installed")

    def test_bwd_dw_final_kernel_exists(self):
        """Test rms_norm_bwd_dw_final_kernel is defined."""
        try:
            from paddleformers.fleet.triton_ops.rms_norm_fusion import (
                rms_norm_bwd_dw_final_kernel,
            )

            self.assertIsNotNone(rms_norm_bwd_dw_final_kernel)
        except ImportError:
            self.skipTest("paddlefleet_ops not installed")


if __name__ == "__main__":
    unittest.main()
