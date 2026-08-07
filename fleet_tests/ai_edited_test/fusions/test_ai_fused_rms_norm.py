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

import numpy as np
import paddle

try:
    from paddle.incubate.nn.functional.fused_rms_norm import (
        fused_rms_norm,  # noqa: F401
    )

    HAVE_FUSED_RMS_NORM = True
except ImportError:
    HAVE_FUSED_RMS_NORM = False


def _make_rmsnorm_config(**overrides):
    """Helper to create a TransformerConfig for FusedRmsNorm testing."""
    from paddleformers.fleet.transformer.transformer_config import TransformerConfig

    defaults = {
        "hidden_size": 128,
        "num_attention_heads": 2,
        "intermediate_size": 256,
        "normalization": "RMSNorm",
        "layernorm_zero_centered_gamma": False,
        "sequence_parallel": False,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


@unittest.skipIf(not HAVE_FUSED_RMS_NORM, "fused_rms_norm not available")
class TestFusedRmsNorm(unittest.TestCase):
    """Tests for FusedRmsNorm."""

    def test_init_with_default_hidden_size(self):
        from paddleformers.fleet.fusions.fused_rms_norm import FusedRmsNorm

        config = _make_rmsnorm_config(hidden_size=128)
        layer = FusedRmsNorm(config, hidden_size=128)
        self.assertEqual(layer.hidden_size, (128,))
        self.assertFalse(layer.zero_centered_gamma)
        self.assertEqual(layer.weight.shape, [128])
        self.assertEqual(layer.bias.shape, [128])

    def test_init_with_tuple_hidden_size(self):
        from paddleformers.fleet.fusions.fused_rms_norm import FusedRmsNorm

        config = _make_rmsnorm_config(hidden_size=128)
        layer = FusedRmsNorm(config, hidden_size=(128,))
        self.assertEqual(layer.hidden_size, (128,))

    def test_reset_parameters_zero_centered_gamma(self):
        from paddleformers.fleet.fusions.fused_rms_norm import FusedRmsNorm

        config = _make_rmsnorm_config(
            hidden_size=64,
            layernorm_zero_centered_gamma=True,
        )
        layer = FusedRmsNorm(config, hidden_size=64)
        np.testing.assert_allclose(layer.weight.numpy(), 0.0, atol=1e-6)
        np.testing.assert_allclose(layer.bias.numpy(), 0.0, atol=1e-6)

    def test_reset_parameters_standard(self):
        from paddleformers.fleet.fusions.fused_rms_norm import FusedRmsNorm

        config = _make_rmsnorm_config(
            hidden_size=64,
            layernorm_zero_centered_gamma=False,
        )
        layer = FusedRmsNorm(config, hidden_size=64)
        np.testing.assert_allclose(layer.weight.numpy(), 1.0, atol=1e-6)
        np.testing.assert_allclose(layer.bias.numpy(), 0.0, atol=1e-6)

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda(),
        "fused_rms_norm forward requires CUDA (uses bfloat16)",
    )
    def test_sequence_parallel_flag(self):
        from paddleformers.fleet.fusions.fused_rms_norm import FusedRmsNorm

        config = _make_rmsnorm_config(
            hidden_size=64,
            sequence_parallel=False,
            tensor_model_parallel_size=1,
        )
        layer = FusedRmsNorm(config, hidden_size=64)
        self.assertFalse(layer.sequence_parallel)


class TestFusedRmsNormErrors(unittest.TestCase):
    """Tests for FusedRmsNorm error cases."""

    def test_assertion_wrong_normalization(self):
        from paddleformers.fleet.transformer.transformer_config import TransformerConfig

        config = TransformerConfig(
            hidden_size=64,
            normalization="LayerNorm",
        )
        with self.assertRaises(AssertionError):
            from paddleformers.fleet.fusions.fused_rms_norm import FusedRmsNorm

            FusedRmsNorm(config, hidden_size=64)


if __name__ == "__main__":
    unittest.main()
