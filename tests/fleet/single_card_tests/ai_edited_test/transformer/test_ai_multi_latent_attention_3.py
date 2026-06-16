# Copyright (c) 2026 PaddlePleet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless distributed on the License is distributed on an "AS IS" BASIS,
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
from unittest.mock import MagicMock

import paddle

from paddleformers.fleet.transformer.multi_latent_attention import (
    MLASelfAttentionSublayersSpec,
    _ec_compatible_rope_apply,
)


class TestEcCompatibleRopeApply(unittest.TestCase):
    """Tests for _ec_compatible_rope_apply."""

    @unittest.skipIf(not paddle.is_compiled_with_cuda(), "Requires CUDA")
    def test_output_shapes_match_input(self):
        """Output shapes should match input shapes for q_pe and k_pe."""
        paddle.disable_static()
        B, S, H, D = 1, 8, 4, 16
        q_pe = paddle.randn([B, S, H, D])
        k_pe = paddle.randn([B, S, 1, D])
        q_out, k_out = _ec_compatible_rope_apply(q_pe, k_pe, S)
        self.assertEqual(q_out.shape, q_pe.shape)
        self.assertEqual(k_out.shape, k_pe.shape)

    @unittest.skipIf(not paddle.is_compiled_with_cuda(), "Requires CUDA")
    def test_output_dtype_matches_input(self):
        """Output dtype should match input dtype."""
        B, S, H, D = 1, 4, 2, 16
        q_pe = paddle.randn([B, S, H, D], dtype="float32")
        k_pe = paddle.randn([B, S, 1, D], dtype="float32")
        q_out, k_out = _ec_compatible_rope_apply(q_pe, k_pe, S)
        self.assertEqual(q_out.dtype, q_pe.dtype)
        self.assertEqual(k_out.dtype, k_pe.dtype)


class TestMLASelfAttentionSublayersSpecDefaults(unittest.TestCase):
    """Tests for MLASelfAttentionSublayersSpec default values."""

    def test_all_fields_default_to_none(self):
        """All fields of MLASelfAttentionSublayersSpec should default to None."""
        spec = MLASelfAttentionSublayersSpec()
        self.assertIsNone(spec.q_a_layernorm)
        self.assertIsNone(spec.kv_a_layernorm)
        self.assertIsNone(spec.q_proj)
        self.assertIsNone(spec.q_a_proj)
        self.assertIsNone(spec.q_b_proj)
        self.assertIsNone(spec.kv_a_proj_with_mqa)
        self.assertIsNone(spec.kv_b_proj)
        self.assertIsNone(spec.core_attention)
        self.assertIsNone(spec.o_proj)
        self.assertIsNone(spec.gate_proj)

    def test_can_set_fields(self):
        """Fields should be settable."""
        mock_spec = MagicMock()
        spec = MLASelfAttentionSublayersSpec(
            q_a_layernorm=mock_spec,
            kv_a_layernorm=mock_spec,
        )
        self.assertEqual(spec.q_a_layernorm, mock_spec)
        self.assertEqual(spec.kv_a_layernorm, mock_spec)


class TestFP8OverlapProj(unittest.TestCase):
    """Tests for FP8OverlapProj."""

    @unittest.skipIf(not paddle.is_compiled_with_cuda(), "Requires CUDA")
    def test_forward_output_shape(self):
        """FP8OverlapProj.forward should produce output matching F.linear."""
        from paddleformers.fleet.transformer.multi_latent_attention import (
            FP8OverlapProj,
        )

        paddle.disable_static()
        x = paddle.randn([2, 4, 8])
        weight = paddle.randn([8, 16])
        result = FP8OverlapProj.apply(x, weight)
        self.assertEqual(result.shape, [2, 4, 16])

    @unittest.skipIf(not paddle.is_compiled_with_cuda(), "Requires CUDA")
    def test_forward_output_matches_linear(self):
        """FP8OverlapProj.forward should match paddle.nn.functional.linear."""
        from paddleformers.fleet.transformer.multi_latent_attention import (
            FP8OverlapProj,
        )

        paddle.disable_static()
        x = paddle.randn([2, 4, 8])
        weight = paddle.randn([8, 16])
        result = FP8OverlapProj.apply(x, weight)
        expected = paddle.nn.functional.linear(x, weight)
        self.assertTrue(paddle.allclose(result, expected, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
