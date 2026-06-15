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
from unittest.mock import MagicMock

from paddleformers.fleet.models.gpt.moe_layer_specs import (
    get_moe_layer_spec_for_backend,
)
from paddleformers.fleet.models.kimi_k25.sd2_tpool_merge import (
    KimiK25VisionPatchMergerSpec,
)


class TestGetMoeLayerSpecForBackend(unittest.TestCase):
    """Test get_moe_layer_spec_for_backend function."""

    def test_returns_layer_spec(self):
        """Test that get_moe_layer_spec_for_backend returns a LayerSpec."""
        from paddle.distributed.fleet.meta_parallel import LayerSpec

        mock_backend = MagicMock()
        mock_backend.column_parallel_linear.return_value = MagicMock()
        mock_backend.row_parallel_linear.return_value = MagicMock()
        mock_backend.hidden_act.return_value = MagicMock()

        result = get_moe_layer_spec_for_backend(
            backend=mock_backend,
            num_experts=8,
            moe_expert_fusion=False,
        )
        self.assertIsInstance(result, LayerSpec)

    def test_calls_backend_methods(self):
        """Test that backend methods are called."""
        mock_backend = MagicMock()
        mock_backend.column_parallel_linear.return_value = MagicMock()
        mock_backend.row_parallel_linear.return_value = MagicMock()
        mock_backend.hidden_act.return_value = MagicMock()

        get_moe_layer_spec_for_backend(
            backend=mock_backend,
            num_experts=8,
        )
        mock_backend.column_parallel_linear.assert_called_once()
        mock_backend.row_parallel_linear.assert_called_once()
        mock_backend.hidden_act.assert_called_once()

    def test_none_num_experts_raises(self):
        """Test that None num_experts raises AssertionError."""
        mock_backend = MagicMock()
        with self.assertRaises(AssertionError):
            get_moe_layer_spec_for_backend(
                backend=mock_backend,
                num_experts=None,
            )

    def test_layer_is_moe_layer(self):
        """Test that the LayerSpec uses MoELayer."""
        from paddleformers.fleet.transformer.moe.moe_layer import MoELayer

        mock_backend = MagicMock()
        mock_backend.column_parallel_linear.return_value = MagicMock()
        mock_backend.row_parallel_linear.return_value = MagicMock()
        mock_backend.hidden_act.return_value = MagicMock()

        result = get_moe_layer_spec_for_backend(
            backend=mock_backend,
            num_experts=8,
        )
        self.assertEqual(result.layer, MoELayer)

    def test_extra_kwargs_has_sublayers(self):
        """Test that the LayerSpec has sublayers in extra_kwargs."""
        mock_backend = MagicMock()
        mock_backend.column_parallel_linear.return_value = MagicMock()
        mock_backend.row_parallel_linear.return_value = MagicMock()
        mock_backend.hidden_act.return_value = MagicMock()

        result = get_moe_layer_spec_for_backend(
            backend=mock_backend,
            num_experts=8,
        )
        self.assertIn("sublayers", result.extra_kwargs)


class TestKimiK25VisionPatchMergerSpec(unittest.TestCase):
    """Test KimiK25VisionPatchMergerSpec dataclass."""

    def test_default_norm_is_identity(self):
        """Test default norm is IdentityOp."""
        from paddleformers.fleet.transformer.identity_op import IdentityOp

        spec = KimiK25VisionPatchMergerSpec()
        self.assertEqual(spec.norm, IdentityOp)

    def test_custom_norm(self):
        """Test custom norm field."""
        mock_norm = MagicMock()
        spec = KimiK25VisionPatchMergerSpec(norm=mock_norm)
        self.assertEqual(spec.norm, mock_norm)


if __name__ == "__main__":
    unittest.main()
