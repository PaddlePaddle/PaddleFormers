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

import unittest

import paddle

try:
    from paddleformers.fleet.triton_ops.moe_topk_fusion import (  # noqa: F401
        MoETopkFusion,
    )

    _MODULE_AVAILABLE = True
except (ImportError, ModuleNotFoundError, Exception):
    _MODULE_AVAILABLE = False


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestMoETopkFusionStructure(unittest.TestCase):
    """Tests for MoE TopK fusion module structure."""

    def test_moe_topk_fusion_import(self):
        """Test MoETopkFusion class can be imported."""
        try:
            from paddleformers.fleet.triton_ops.moe_topk_fusion import MoETopkFusion

            self.assertIsNotNone(MoETopkFusion)
        except ImportError:
            self.skipTest("paddlefleet_ops not installed")

    def test_moe_topk_fusion_is_pylayer(self):
        """Test MoETopkFusion is a PyLayer subclass."""
        try:
            from paddleformers.fleet.triton_ops.moe_topk_fusion import MoETopkFusion

            self.assertTrue(issubclass(MoETopkFusion, paddle.autograd.PyLayer))
        except ImportError:
            self.skipTest("paddlefleet_ops not installed")

    def test_moe_topk_fusion_has_forward(self):
        """Test MoETopkFusion has forward static method."""
        try:
            from paddleformers.fleet.triton_ops.moe_topk_fusion import MoETopkFusion

            self.assertTrue(hasattr(MoETopkFusion, "forward"))
            self.assertTrue(callable(MoETopkFusion.forward))
        except ImportError:
            self.skipTest("paddlefleet_ops not installed")

    def test_moe_topk_fusion_has_backward(self):
        """Test MoETopkFusion has backward static method."""
        try:
            from paddleformers.fleet.triton_ops.moe_topk_fusion import MoETopkFusion

            self.assertTrue(hasattr(MoETopkFusion, "backward"))
            self.assertTrue(callable(MoETopkFusion.backward))
        except ImportError:
            self.skipTest("paddlefleet_ops not installed")

    def test_routing_map_fusion_forward_exists(self):
        """Test routing_map_fusion_forward function exists."""
        try:
            from paddleformers.fleet.triton_ops.moe_topk_fusion import (
                routing_map_fusion_forward,
            )

            self.assertTrue(callable(routing_map_fusion_forward))
        except ImportError:
            self.skipTest("paddlefleet_ops not installed")

    def test_routing_map_forward_signature(self):
        """Test routing_map_fusion_forward has expected parameters."""
        try:
            import inspect

            from paddleformers.fleet.triton_ops.moe_topk_fusion import (
                routing_map_fusion_forward,
            )

            sig = inspect.signature(routing_map_fusion_forward)
            params = list(sig.parameters.keys())
            self.assertIn("gate_probs", params)
            self.assertIn("topk_indices", params)
        except ImportError:
            self.skipTest("paddlefleet_ops not installed")

    def test_fwd_kernel_exists(self):
        """Test _fwd_kernel is defined in module."""
        try:
            from paddleformers.fleet.triton_ops.moe_topk_fusion import _fwd_kernel

            self.assertIsNotNone(_fwd_kernel)
        except ImportError:
            self.skipTest("paddlefleet_ops not installed")

    def test_bwd_kernel_exists(self):
        """Test _bwd_kernel is defined in module."""
        try:
            from paddleformers.fleet.triton_ops.moe_topk_fusion import _bwd_kernel

            self.assertIsNotNone(_bwd_kernel)
        except ImportError:
            self.skipTest("paddlefleet_ops not installed")


if __name__ == "__main__":
    unittest.main()
