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
from unittest.mock import patch

import paddle

try:
    from paddle.incubate.nn.functional.fused_layer_norm import (  # noqa: F401
        fused_layer_norm,
    )

    HAVE_FUSED_LAYER_NORM = True
except ImportError:
    HAVE_FUSED_LAYER_NORM = False


def _make_layernorm_config(**overrides):
    """Helper to create a TransformerConfig for FusedLayerNorm testing."""
    from paddleformers.fleet.transformer.transformer_config import (
        TransformerConfig,
    )

    # Separate fields that are not part of TransformerConfig dataclass
    extra_attrs = {}
    for key in ("persist_layer_norm",):
        if key in overrides:
            extra_attrs[key] = overrides.pop(key)

    defaults = {
        "hidden_size": 128,
        "num_attention_heads": 2,
        "intermediate_size": 256,
        "normalization": "LayerNorm",
        "layernorm_zero_centered_gamma": False,
        "sequence_parallel": False,
    }
    defaults.update(overrides)
    config = TransformerConfig(**defaults)
    # Set extra attributes that FusedLayerNorm reads from config
    config.persist_layer_norm = extra_attrs.get("persist_layer_norm", False)
    return config


@unittest.skipIf(not HAVE_FUSED_LAYER_NORM, "fused_layer_norm not available")
class TestFusedLayerNorm(unittest.TestCase):
    """Tests for FusedLayerNorm."""

    def test_init_with_default_hidden_size(self):
        from paddleformers.fleet.fusions.fused_layer_norm import FusedLayerNorm

        config = _make_layernorm_config(hidden_size=128)
        layer = FusedLayerNorm(config, hidden_size=128)
        self.assertEqual(layer.hidden_size, (128,))
        self.assertFalse(layer.zero_centered_gamma)
        self.assertEqual(layer.weight.shape, [128])
        self.assertEqual(layer.bias.shape, [128])

    def test_init_with_persist_ln_unsupported_size(self):
        from paddleformers.fleet.fusions.fused_layer_norm import FusedLayerNorm

        config = _make_layernorm_config(
            hidden_size=100,
            persist_layer_norm=True,
        )
        # 100 is not in the persist_ln_hidden_sizes list, should fallback
        layer = FusedLayerNorm(config, hidden_size=100)
        self.assertFalse(layer.persist_layer_norm)

    def test_init_with_persist_ln_supported_size(self):
        from paddleformers.fleet.fusions.fused_layer_norm import FusedLayerNorm

        # 1024 is in the persist_ln_hidden_sizes list
        config = _make_layernorm_config(
            hidden_size=1024,
            persist_layer_norm=True,
        )
        with patch(
            "paddleformers.fleet.fusions.fused_layer_norm.HAVE_PERSIST_LAYER_NORM",
            True,
        ):
            layer = FusedLayerNorm(config, hidden_size=1024)
            # persist_layer_norm should be True only if HAVE_PERSIST_LAYER_NORM is True
            self.assertTrue(layer.persist_layer_norm)

    def test_reset_parameters_zero_centered_gamma(self):
        from paddleformers.fleet.fusions.fused_layer_norm import FusedLayerNorm

        config = _make_layernorm_config(
            hidden_size=64,
            layernorm_zero_centered_gamma=True,
        )
        layer = FusedLayerNorm(config, hidden_size=64)
        import numpy as np

        np.testing.assert_allclose(layer.weight.numpy(), 0.0, atol=1e-6)
        np.testing.assert_allclose(layer.bias.numpy(), 0.0, atol=1e-6)

    def test_reset_parameters_standard(self):
        from paddleformers.fleet.fusions.fused_layer_norm import FusedLayerNorm

        config = _make_layernorm_config(
            hidden_size=64,
            layernorm_zero_centered_gamma=False,
        )
        layer = FusedLayerNorm(config, hidden_size=64)
        import numpy as np

        np.testing.assert_allclose(layer.weight.numpy(), 1.0, atol=1e-6)
        np.testing.assert_allclose(layer.bias.numpy(), 0.0, atol=1e-6)

    def test_forward_basic(self):
        from paddleformers.fleet.fusions.fused_layer_norm import FusedLayerNorm

        config = _make_layernorm_config(hidden_size=64)
        layer = FusedLayerNorm(config, hidden_size=64)
        layer.eval()
        x = paddle.randn([2, 8, 64], dtype=paddle.float32)
        out = layer(x)
        self.assertEqual(out.shape, [2, 8, 64])
        # Output should be finite
        self.assertTrue(paddle.all(paddle.isfinite(out)).item())

    def test_forward_2d_input(self):
        from paddleformers.fleet.fusions.fused_layer_norm import FusedLayerNorm

        config = _make_layernorm_config(hidden_size=32)
        layer = FusedLayerNorm(config, hidden_size=32)
        layer.eval()
        x = paddle.randn([4, 32], dtype=paddle.float32)
        out = layer(x)
        self.assertEqual(out.shape, [4, 32])

    def test_forward_zero_centered_gamma(self):
        from paddleformers.fleet.fusions.fused_layer_norm import FusedLayerNorm

        config = _make_layernorm_config(
            hidden_size=64,
            layernorm_zero_centered_gamma=True,
        )
        layer = FusedLayerNorm(config, hidden_size=64)
        layer.eval()
        x = paddle.randn([2, 8, 64], dtype=paddle.float32)
        out = layer(x)
        self.assertEqual(out.shape, [2, 8, 64])
        self.assertTrue(paddle.all(paddle.isfinite(out)).item())

    def test_sequence_parallel_flag(self):
        from paddleformers.fleet.fusions.fused_layer_norm import FusedLayerNorm

        config = _make_layernorm_config(
            hidden_size=64,
            sequence_parallel=False,
            tensor_model_parallel_size=1,
        )
        layer = FusedLayerNorm(config, hidden_size=64)
        self.assertFalse(layer.sequence_parallel)


class TestFusedLayerNormErrors(unittest.TestCase):
    """Tests for FusedLayerNorm error cases."""

    def test_assertion_wrong_normalization(self):
        from paddleformers.fleet.transformer.transformer_config import (
            TransformerConfig,
        )

        config = TransformerConfig(
            hidden_size=64,
            normalization="RMSNorm",
        )
        with self.assertRaises(AssertionError):
            from paddleformers.fleet.fusions.fused_layer_norm import (
                FusedLayerNorm,
            )

            FusedLayerNorm(config, hidden_size=64)

    @unittest.skipIf(
        not HAVE_FUSED_LAYER_NORM, "fused_layer_norm not available"
    )
    def test_value_error_no_fused_ln(self):
        """Test that ValueError is raised when fused_ln is not available."""
        from paddleformers.fleet.transformer.transformer_config import (
            TransformerConfig,
        )

        config = TransformerConfig(
            hidden_size=64,
            normalization="LayerNorm",
            layernorm_zero_centered_gamma=False,
        )
        config.persist_layer_norm = False
        with (
            patch(
                "paddleformers.fleet.fusions.fused_layer_norm.HAVE_FUSED_LAYER_NORM",
                False,
            ),
            self.assertRaises(ValueError),
        ):
            from paddleformers.fleet.fusions.fused_layer_norm import (
                FusedLayerNorm,
            )

            FusedLayerNorm(config, hidden_size=64)


if __name__ == "__main__":
    unittest.main()
