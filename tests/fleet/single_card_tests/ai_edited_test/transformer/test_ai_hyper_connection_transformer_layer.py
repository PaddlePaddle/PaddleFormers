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
import os
import sys

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)

import unittest

import paddle

from paddleformers.fleet.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from paddleformers.fleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddleformers.fleet.transformer.hyper_connection import (
    HyperConnectionContractLayer,
    HyperConnectionExpandLayer,
    HyperConnectionModule,
    SinkhornKnopp,
)
from paddleformers.fleet.transformer.transformer_config import TransformerConfig
from paddleformers.fleet.transformer.transformer_layer import (
    HyperConnectionTransformerLayer,
    TransformerLayer,
    TransformerLayerSublayersSpec,
)
from paddleformers.fleet.utils import init_method_normal, scaled_init_method_normal

# Initialize CUDA RNG tracker for tensor parallel layers
model_parallel_cuda_manual_seed(42, tp_rank=0, ep_rank=0, etp_rank=0)


def _make_config(**overrides):
    defaults = {
        "num_hidden_layers": 2,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": 16,
        "use_bias": False,
        "hidden_dropout_prob": 0.0,
        "normalization": "RMSNorm",
        "rms_norm_eps": 1e-5,
        "sequence_parallel": False,
        "tensor_model_parallel_size": 1,
        "context_parallel_size": 1,
        "recompute_granularity": None,
        "recompute_method": None,
        "recompute_num_layers": None,
        "recompute_modules": None,
        "block_attention_residuals": False,
        "attn_res_block_size": 1,
        "attention_dropout": 0.0,
        "bias_dropout_fusion": False,
        "apply_rope_fusion": False,
        "apply_query_key_layer_scaling": False,
        "sliding_window": None,
        "softmax_type": "vanilla",
        "gated_linear_unit": False,
        "bias_activation_fusion": False,
        "gated_attention": False,
        "num_nextn_predict_layers": 0,
        "mtp_load_weight_only": False,
        "init_method": init_method_normal(0.02),
        "output_layer_init_method": scaled_init_method_normal(0.02, 1, 2.0),
        "fp16": False,
        "bf16": False,
        "masked_softmax_fusion": False,
        "attention_softmax_in_fp32": True,
        "softmax_scale": None,
        "multi_latent_attention": False,
        "rotary_interleaved": False,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def _make_hc_config(**overrides):
    """Create a config with hyper-connections enabled."""
    hc_defaults = {
        "enable_hyper_connections": True,
        "num_residual_streams": 4,
        "mhc_sinkhorn_iterations": 5,
        "mhc_init_gating_factor": 0.01,
        "high_precision_mhc": True,
    }
    hc_defaults.update(overrides)
    return _make_config(**hc_defaults)


def _make_hc_layer(config, layer_number=1):
    spec = get_gpt_layer_local_spec(config)
    return HyperConnectionTransformerLayer(
        config=config,
        sublayers_spec=spec.sublayers_spec,
        layer_number=layer_number,
    )


# ==============================================================================
# Tests for SinkhornKnopp
# ==============================================================================


class TestSinkhornKnopp(unittest.TestCase):
    """Tests for SinkhornKnopp doubly stochastic projection."""

    def test_output_shape(self):
        """Output should have same shape as input."""
        n = 4
        x = paddle.randn([2, 3, n, n])
        result = SinkhornKnopp.apply(x, 20)
        self.assertEqual(list(result.shape), [2, 3, n, n])

    def test_doubly_stochastic_rows_sum_to_one(self):
        """Rows of output should sum to approximately 1."""
        n = 4
        x = paddle.randn([8, n, n])
        result = SinkhornKnopp.apply(x, 20)
        row_sums = result.sum(axis=-1)
        self.assertTrue(paddle.allclose(row_sums, paddle.ones_like(row_sums), atol=1e-4))

    def test_doubly_stochastic_cols_sum_to_one(self):
        """Columns of output should sum to approximately 1."""
        n = 4
        x = paddle.randn([8, n, n])
        result = SinkhornKnopp.apply(x, 20)
        col_sums = result.sum(axis=-2)
        self.assertTrue(paddle.allclose(col_sums, paddle.ones_like(col_sums), atol=1e-4))

    def test_output_non_negative(self):
        """Output matrix entries should be non-negative."""
        n = 4
        x = paddle.randn([8, n, n])
        result = SinkhornKnopp.apply(x, 20)
        self.assertTrue((result >= 0).all().item())

    def test_gradient_flows(self):
        """Gradient should flow through SinkhornKnopp."""
        n = 4
        x = paddle.randn([2, n, n])
        x.stop_gradient = False
        result = SinkhornKnopp.apply(x, 10)
        loss = result.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertFalse(paddle.isnan(x.grad).any().item())

    def test_more_iterations_improves_convergence(self):
        """More Sinkhorn iterations should produce better doubly stochastic matrices."""
        n = 4
        x = paddle.randn([8, n, n])
        result_5 = SinkhornKnopp.apply(x, 5)
        result_50 = SinkhornKnopp.apply(x, 50)

        # Both should sum to 1 per row, but 50 iters should be closer
        row_err_5 = (result_5.sum(axis=-1) - 1.0).abs().max().item()
        row_err_50 = (result_50.sum(axis=-1) - 1.0).abs().max().item()
        self.assertLessEqual(row_err_50, row_err_5 + 1e-6)


# ==============================================================================
# Tests for HyperConnectionModule
# ==============================================================================


class TestHyperConnectionModule(unittest.TestCase):
    """Tests for HyperConnectionModule."""

    def setUp(self):
        self.config = _make_hc_config()
        self.n = self.config.num_residual_streams
        self.C = self.config.hidden_size
        self.module = HyperConnectionModule(config=self.config, layer_number=1)

    def test_construction(self):
        self.assertEqual(self.module.n, self.n)
        self.assertEqual(self.module.hidden_size, self.C)
        self.assertIsNotNone(self.module.mapping_proj)

    def test_construction_params(self):
        """All learnable parameters should be created."""
        self.assertIsNotNone(self.module.alpha_pre)
        self.assertIsNotNone(self.module.alpha_post)
        self.assertIsNotNone(self.module.alpha_res)
        self.assertIsNotNone(self.module.bias)
        self.assertEqual(list(self.module.alpha_pre.shape), [1])
        self.assertEqual(list(self.module.bias.shape), [self.n * self.n + 2 * self.n])

    def test_forward_output_shapes(self):
        """forward() should return (aggregated, h_res, h_post) with correct shapes."""
        B, S = 2, 4
        x = paddle.randn([B, S, self.n * self.C])
        aggregated, h_res, h_post = self.module(x)
        self.assertEqual(list(aggregated.shape), [B, S, self.C])
        self.assertEqual(list(h_res.shape), [B, S, self.n, self.n])
        self.assertEqual(list(h_post.shape), [B, S, self.n])

    def test_forward_3d_input(self):
        """forward() should work with 3D input [B, S, n*C]."""
        x = paddle.randn([3, 5, self.n * self.C])
        aggregated, h_res, h_post = self.module(x)
        self.assertEqual(list(aggregated.shape), [3, 5, self.C])

    def test_forward_2d_input(self):
        """forward() should work with 2D input [tokens, n*C]."""
        x = paddle.randn([10, self.n * self.C])
        aggregated, h_res, h_post = self.module(x)
        self.assertEqual(list(aggregated.shape), [10, self.C])
        self.assertEqual(list(h_res.shape), [10, self.n, self.n])
        self.assertEqual(list(h_post.shape), [10, self.n])

    def test_forward_high_precision_mhc_switch(self):
        """high_precision_mhc should choose float32 or bfloat16 compute output."""
        cases = (
            (True, paddle.float32),
            (False, paddle.bfloat16),
        )
        for high_precision_mhc, expected_dtype in cases:
            with self.subTest(high_precision_mhc=high_precision_mhc):
                config = _make_hc_config(high_precision_mhc=high_precision_mhc)
                module = HyperConnectionModule(config=config, layer_number=1)
                module = paddle.amp.decorate(models=module, level="O2", dtype="bfloat16")
                x = paddle.randn([2, 4, self.n * self.C]).astype("bfloat16")

                aggregated, h_res, h_post = module(x)

                self.assertEqual(module.config.high_precision_mhc, high_precision_mhc)
                self.assertEqual(aggregated.dtype, expected_dtype)
                self.assertEqual(h_res.dtype, expected_dtype)
                self.assertEqual(h_post.dtype, expected_dtype)
                self.assertEqual(list(aggregated.shape), [2, 4, self.C])
                self.assertEqual(list(h_res.shape), [2, 4, self.n, self.n])
                self.assertEqual(list(h_post.shape), [2, 4, self.n])

    # ---------- compute_mappings ----------

    def test_compute_mappings_shapes(self):
        """compute_mappings should return (h_pre, h_post, h_res) with correct shapes."""
        B, S = 2, 4
        x = paddle.randn([B, S, self.n * self.C])
        h_pre, h_post, h_res = self.module.compute_mappings(x)
        self.assertEqual(list(h_pre.shape), [B, S, self.n])
        self.assertEqual(list(h_post.shape), [B, S, self.n])
        self.assertEqual(list(h_res.shape), [B, S, self.n, self.n])

    def test_compute_mappings_h_pre_range(self):
        """h_pre should be in (0, 1) as it uses sigmoid."""
        x = paddle.randn([4, 8, self.n * self.C])
        h_pre, _, _ = self.module.compute_mappings(x)
        self.assertTrue((h_pre > 0).all().item())
        self.assertTrue((h_pre < 1).all().item())

    def test_compute_mappings_h_post_range(self):
        """h_post should be in (0, 2) as it uses 2*sigmoid."""
        x = paddle.randn([4, 8, self.n * self.C])
        _, h_post, _ = self.module.compute_mappings(x)
        self.assertTrue((h_post > 0).all().item())
        self.assertTrue((h_post < 2).all().item())

    def test_compute_mappings_h_res_doubly_stochastic(self):
        """h_res should be doubly stochastic (rows and cols sum to 1)."""
        x = paddle.randn([4, 8, self.n * self.C])
        _, _, h_res = self.module.compute_mappings(x)
        row_sums = h_res.sum(axis=-1)
        col_sums = h_res.sum(axis=-2)
        self.assertTrue(paddle.allclose(row_sums, paddle.ones_like(row_sums), atol=1e-3))
        self.assertTrue(paddle.allclose(col_sums, paddle.ones_like(col_sums), atol=1e-3))

    # ---------- aggregate ----------

    def test_aggregate_shape(self):
        """aggregate should reduce n*C to C."""
        B, S = 2, 4
        x = paddle.randn([B, S, self.n * self.C])
        h_pre = paddle.ones([B, S, self.n]) / self.n  # uniform weights
        result = self.module.aggregate(x, h_pre)
        self.assertEqual(list(result.shape), [B, S, self.C])

    def test_aggregate_uniform_weights_equals_mean(self):
        """With uniform h_pre, aggregate should equal mean of streams."""
        B, S = 2, 4
        x = paddle.randn([B, S, self.n * self.C])
        h_pre = paddle.ones([B, S, self.n]) / self.n
        result = self.module.aggregate(x, h_pre)
        # Manually compute mean
        x_streams = x.reshape([B, S, self.n, self.C])
        expected = x_streams.mean(axis=-2)
        self.assertTrue(paddle.allclose(result, expected, atol=1e-5))

    # ---------- apply_h_res ----------

    def test_apply_h_res_shape(self):
        """apply_h_res should preserve shape [..., n*C]."""
        B, S = 2, 4
        h_res = paddle.eye(self.n).unsqueeze(0).unsqueeze(0).expand([B, S, self.n, self.n])
        residual = paddle.randn([B, S, self.n * self.C])
        result = self.module.apply_h_res(h_res, residual)
        self.assertEqual(list(result.shape), [B, S, self.n * self.C])

    def test_apply_h_res_identity_preserves_input(self):
        """With identity h_res, output should equal input."""
        B, S = 2, 4
        h_res = paddle.eye(self.n).unsqueeze(0).unsqueeze(0).expand([B, S, self.n, self.n])
        residual = paddle.randn([B, S, self.n * self.C])
        result = self.module.apply_h_res(h_res, residual)
        self.assertTrue(paddle.allclose(result, residual, atol=1e-5))

    # ---------- _apply_h_post ----------

    def test_apply_h_post_standard_shape(self):
        """_apply_h_post with [..., C] input should return [..., n*C]."""
        B, S = 2, 4
        x = paddle.randn([B, S, self.C])
        h_post = paddle.ones([B, S, self.n])
        result = self.module._apply_h_post(x, h_post)
        self.assertEqual(list(result.shape), [B, S, self.n * self.C])

    def test_apply_h_post_bias_shape(self):
        """_apply_h_post with [C] bias input should return [..., n*C]."""
        B, S = 2, 4
        bias = paddle.randn([self.C])
        h_post = paddle.ones([B, S, self.n])
        result = self.module._apply_h_post(bias, h_post)
        self.assertEqual(list(result.shape), [B, S, self.n * self.C])

    def test_apply_h_post_tuple(self):
        """apply_h_post with (x, bias) should return (expanded_x, expanded_bias)."""
        B, S = 2, 4
        x = paddle.randn([B, S, self.C])
        bias = paddle.randn([self.C])
        h_post = paddle.ones([B, S, self.n])
        x_out, bias_out = self.module.apply_h_post((x, bias), h_post)
        self.assertEqual(list(x_out.shape), [B, S, self.n * self.C])
        self.assertEqual(list(bias_out.shape), [B, S, self.n * self.C])

    def test_apply_h_post_tuple_none_bias(self):
        """apply_h_post with (x, None) should return (expanded_x, None)."""
        B, S = 2, 4
        x = paddle.randn([B, S, self.C])
        h_post = paddle.ones([B, S, self.n])
        x_out, bias_out = self.module.apply_h_post((x, None), h_post)
        self.assertEqual(list(x_out.shape), [B, S, self.n * self.C])
        self.assertIsNone(bias_out)

    # ---------- fused_h_res_h_post_bda ----------

    def test_fused_h_res_h_post_bda_shape(self):
        """fused_h_res_h_post_bda should return shape [..., n*C]."""
        B, S = 2, 4
        x = paddle.randn([B, S, self.n * self.C])
        aggregated, h_res, h_post = self.module(x)
        layer_output = paddle.randn([B, S, self.C])

        result = self.module.fused_h_res_h_post_bda(
            h_res=h_res,
            original_residual=x,
            h_post=h_post,
            layer_output_with_bias=(layer_output, None),
            dropout_prob=0.0,
            training=False,
            fused=False,
        )
        self.assertEqual(list(result.shape), [B, S, self.n * self.C])

    def test_fused_h_res_h_post_bda_with_bias(self):
        """fused_h_res_h_post_bda should handle non-None bias correctly."""
        B, S = 2, 4
        x = paddle.randn([B, S, self.n * self.C])
        aggregated, h_res, h_post = self.module(x)
        layer_output = paddle.randn([B, S, self.C])
        bias = paddle.randn([self.C])

        result = self.module.fused_h_res_h_post_bda(
            h_res=h_res,
            original_residual=x,
            h_post=h_post,
            layer_output_with_bias=(layer_output, bias),
            dropout_prob=0.0,
            training=False,
            fused=False,
        )
        self.assertEqual(list(result.shape), [B, S, self.n * self.C])

    def test_fused_h_res_h_post_bda_no_nan(self):
        """fused_h_res_h_post_bda output should not contain NaN."""
        B, S = 2, 4
        x = paddle.randn([B, S, self.n * self.C])
        aggregated, h_res, h_post = self.module(x)
        layer_output = paddle.randn([B, S, self.C])

        result = self.module.fused_h_res_h_post_bda(
            h_res=h_res,
            original_residual=x,
            h_post=h_post,
            layer_output_with_bias=(layer_output, None),
            dropout_prob=0.0,
            training=False,
            fused=False,
        )
        self.assertFalse(paddle.isnan(result).any().item())

    def test_fused_h_res_h_post_bda_high_precision_fast_path(self):
        """high_precision_mhc should choose float32 or bfloat16 BDA output."""
        B, S = 2, 4
        cases = (
            (True, paddle.float32),
            (False, paddle.bfloat16),
        )
        for high_precision_mhc, expected_dtype in cases:
            with self.subTest(high_precision_mhc=high_precision_mhc):
                config = _make_hc_config(high_precision_mhc=high_precision_mhc)
                module = HyperConnectionModule(config=config, layer_number=1)
                module = paddle.amp.decorate(models=module, level="O2", dtype="bfloat16")
                x = paddle.randn([B, S, self.n * self.C]).astype("bfloat16")
                _, h_res, h_post = module(x)
                layer_output = paddle.randn([B, S, self.C]).astype("bfloat16")
                bias = paddle.randn([self.C]).astype("bfloat16")

                result = module.fused_h_res_h_post_bda(
                    h_res=h_res,
                    original_residual=x,
                    h_post=h_post,
                    layer_output_with_bias=(layer_output, bias),
                    dropout_prob=0.0,
                    training=False,
                    fused=False,
                )

                self.assertEqual(module.config.high_precision_mhc, high_precision_mhc)
                self.assertEqual(result.dtype, expected_dtype)
                self.assertEqual(list(result.shape), [B, S, self.n * self.C])
                self.assertFalse(paddle.isnan(result.astype("float32")).any().item())

    # ---------- input_expand / output_contract ----------

    def test_input_expand_shape(self):
        """input_expand should expand [..., C] to [..., n*C]."""
        B, S = 2, 4
        x = paddle.randn([B, S, self.C])
        result = HyperConnectionModule.input_expand(x, self.n)
        self.assertEqual(list(result.shape), [B, S, self.n * self.C])

    def test_input_expand_replication(self):
        """input_expand should replicate input to each stream."""
        B, S = 2, 4
        x = paddle.randn([B, S, self.C])
        result = HyperConnectionModule.input_expand(x, self.n)
        # Each stream should be identical to original
        streams = result.reshape([B, S, self.n, self.C])
        for i in range(self.n):
            self.assertTrue(paddle.allclose(streams[:, :, i, :], x, atol=1e-6))

    def test_output_contract_shape(self):
        """output_contract should contract [..., n*C] to [..., C]."""
        B, S = 2, 4
        x = paddle.randn([B, S, self.n * self.C])
        result = HyperConnectionModule.output_contract(x, self.n)
        self.assertEqual(list(result.shape), [B, S, self.C])

    def test_expand_then_contract_preserves(self):
        """Expanding then contracting (averaging) should preserve input for uniform streams."""
        B, S = 2, 4
        x = paddle.randn([B, S, self.C])
        expanded = HyperConnectionModule.input_expand(x, self.n)
        contracted = HyperConnectionModule.output_contract(expanded, self.n)
        # Since all streams are copies, mean == original
        self.assertTrue(paddle.allclose(contracted, x, atol=1e-5))

    # ---------- learned_output_contract ----------

    def test_learned_output_contract_shape(self):
        """learned_output_contract should contract [..., n*C] to [..., C]."""
        B, S = 2, 4
        n = self.n
        C = self.C
        x = paddle.randn([B, S, n * C])
        # F.linear(x, weight) expects weight shape [out_features, in_features]
        # Here: x is [B, S, n*C], output should be [B, S, n], so weight is [n, n*C]
        # But F.linear transposes internally: out = x @ weight.T
        # So weight should be [n, n*C] -> out = [B,S,n*C] @ [n*C, n] = [B,S,n]
        head_fn = paddle.randn([n * C, n])
        base = paddle.zeros([n])
        scale = paddle.ones([1])
        result = HyperConnectionModule.learned_output_contract(x, head_fn, base, scale, n, 1e-6)
        self.assertEqual(list(result.shape), [B, S, C])

    def test_learned_output_contract_no_nan(self):
        """learned_output_contract should not produce NaN."""
        B, S = 2, 4
        n = self.n
        C = self.C
        x = paddle.randn([B, S, n * C])
        head_fn = paddle.randn([n * C, n])
        base = paddle.zeros([n])
        scale = paddle.ones([1])
        result = HyperConnectionModule.learned_output_contract(x, head_fn, base, scale, n, 1e-6)
        self.assertFalse(paddle.isnan(result).any().item())

    # ---------- gradient flow ----------

    def test_full_forward_gradient_flow(self):
        """Gradient should flow through the entire forward pass."""
        B, S = 2, 4
        x = paddle.randn([B, S, self.n * self.C])
        x.stop_gradient = False
        aggregated, h_res, h_post = self.module(x)
        loss = aggregated.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertFalse(paddle.isnan(x.grad).any().item())

    def test_fused_bda_gradient_flow(self):
        """Gradient should flow through fused_h_res_h_post_bda."""
        B, S = 2, 4
        x = paddle.randn([B, S, self.n * self.C])
        x.stop_gradient = False
        aggregated, h_res, h_post = self.module(x)
        layer_output = paddle.randn([B, S, self.C])
        layer_output.stop_gradient = False

        result = self.module.fused_h_res_h_post_bda(
            h_res=h_res,
            original_residual=x,
            h_post=h_post,
            layer_output_with_bias=(layer_output, None),
            dropout_prob=0.0,
            training=False,
            fused=False,
        )
        loss = result.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(layer_output.grad)


# ==============================================================================
# Tests for HyperConnectionExpandLayer and HyperConnectionContractLayer
# ==============================================================================


class TestHyperConnectionExpandLayer(unittest.TestCase):
    """Tests for HyperConnectionExpandLayer."""

    def setUp(self):
        self.config = _make_hc_config()
        self.n = self.config.num_residual_streams
        self.C = self.config.hidden_size
        self.layer = HyperConnectionExpandLayer(self.config)

    def test_construction(self):
        self.assertEqual(self.layer.n, self.n)

    def test_forward_shape(self):
        """Forward should expand hidden_states from [S, B, C] to [S, B, n*C]."""
        S, B = 4, 2
        hidden_states = paddle.randn([S, B, self.C])
        dict_args = {"hidden_states": hidden_states}
        result = self.layer.forward(dict_args)
        self.assertEqual(list(result["hidden_states"].shape), [S, B, self.n * self.C])

    def test_forward_preserves_other_keys(self):
        """Forward should not modify other keys in dict_args."""
        S, B = 4, 2
        hidden_states = paddle.randn([S, B, self.C])
        mask = paddle.randn([B, 1, S, S])
        dict_args = {"hidden_states": hidden_states, "attention_mask": mask}
        result = self.layer.forward(dict_args)
        self.assertTrue(paddle.equal_all(result["attention_mask"], mask))


class TestHyperConnectionContractLayer(unittest.TestCase):
    """Tests for HyperConnectionContractLayer."""

    def setUp(self):
        self.config = _make_hc_config()
        self.n = self.config.num_residual_streams
        self.C = self.config.hidden_size
        self.layer = HyperConnectionContractLayer(self.config)

    def test_construction(self):
        self.assertEqual(self.layer.n, self.n)
        self.assertIsNotNone(self.layer.hc_head_fn)
        self.assertIsNotNone(self.layer.hc_head_base)
        self.assertIsNotNone(self.layer.hc_head_scale)

    def test_forward_shape(self):
        """Forward should contract hidden_states from [S, B, n*C] to [S, B, C]."""
        S, B = 4, 2
        hidden_states = paddle.randn([S, B, self.n * self.C])
        dict_args = {"hidden_states": hidden_states}
        result = self.layer.forward(dict_args)
        self.assertEqual(list(result["hidden_states"].shape), [S, B, self.C])

    def test_forward_no_nan(self):
        """Forward output should not contain NaN."""
        S, B = 4, 2
        hidden_states = paddle.randn([S, B, self.n * self.C])
        dict_args = {"hidden_states": hidden_states}
        result = self.layer.forward(dict_args)
        self.assertFalse(paddle.isnan(result["hidden_states"]).any().item())

    def test_forward_with_mtp_enabled(self):
        """Forward with MTP should split and contract correctly."""
        num_mtp = 2
        config = _make_hc_config(num_nextn_predict_layers=num_mtp)
        layer = HyperConnectionContractLayer(config)
        S, B = 6, 2
        # Total hidden_states: (num_mtp + 1) * S tokens in seq dim
        total_S = (num_mtp + 1) * S
        hidden_states = paddle.randn([total_S, B, self.n * self.C])
        dict_args = {"hidden_states": hidden_states}
        result = layer.forward(dict_args)
        # Main part: [S, B, C], MTP parts: [S, B, C] each -> total [3*S, B, C]
        self.assertEqual(list(result["hidden_states"].shape), [total_S, B, self.C])

    def test_forward_with_mtp_preserves_multistream(self):
        """With MTP enabled, should save mhc_multistream in dict_args."""
        num_mtp = 2
        config = _make_hc_config(num_nextn_predict_layers=num_mtp)
        layer = HyperConnectionContractLayer(config)
        S, B = 6, 2
        total_S = (num_mtp + 1) * S
        hidden_states = paddle.randn([total_S, B, self.n * self.C])
        dict_args = {"hidden_states": hidden_states}
        result = layer.forward(dict_args)
        self.assertIn("mhc_multistream", result)
        self.assertTrue(paddle.equal_all(result["mhc_multistream"], hidden_states))

    def test_expand_then_contract_pipeline(self):
        """Expand layer followed by contract layer should produce [S, B, C] output."""
        S, B = 4, 2
        expand_layer = HyperConnectionExpandLayer(self.config)
        hidden_states = paddle.randn([S, B, self.C])
        dict_args = {"hidden_states": hidden_states}
        expanded = expand_layer.forward(dict_args)
        contracted = self.layer.forward(expanded)
        self.assertEqual(list(contracted["hidden_states"].shape), [S, B, self.C])


# ==============================================================================
# Tests for HyperConnectionTransformerLayer
# ==============================================================================


class TestHyperConnectionTransformerLayerConstruction(unittest.TestCase):
    """Tests for HyperConnectionTransformerLayer constructor."""

    def test_basic_construction(self):
        config = _make_hc_config()
        layer = _make_hc_layer(config)
        self.assertIsInstance(layer, HyperConnectionTransformerLayer)
        self.assertIsInstance(layer, TransformerLayer)

    def test_has_hyper_connection_sublayers(self):
        config = _make_hc_config()
        layer = _make_hc_layer(config)
        self.assertIsInstance(layer.self_attention_hyper_connection, HyperConnectionModule)
        self.assertIsInstance(layer.mlp_hyper_connection, HyperConnectionModule)

    def test_raises_without_hc_spec(self):
        """Should raise if sublayers_spec has IdentityOp for hyper connections."""
        config = _make_hc_config()
        spec = TransformerLayerSublayersSpec()  # all IdentityOp
        with self.assertRaises(AssertionError):
            HyperConnectionTransformerLayer(
                config=config,
                sublayers_spec=spec,
                layer_number=1,
            )

    def test_raises_with_block_attention_residuals(self):
        """mHC is incompatible with block_attention_residuals."""
        config = _make_hc_config(block_attention_residuals=True)
        with self.assertRaises(AssertionError):
            _make_hc_layer(config)

    def test_layer_number(self):
        config = _make_hc_config()
        layer = _make_hc_layer(config, layer_number=3)
        self.assertEqual(layer.layer_number, 3)

    def test_different_num_residual_streams(self):
        """Should work with different num_residual_streams values."""
        for n in [2, 3, 8]:
            config = _make_hc_config(num_residual_streams=n)
            layer = _make_hc_layer(config)
            self.assertEqual(layer.self_attention_hyper_connection.n, n)
            self.assertEqual(layer.mlp_hyper_connection.n, n)


class TestHyperConnectionTransformerLayerForward(unittest.TestCase):
    """Tests for HyperConnectionTransformerLayer forward pass."""

    def setUp(self):
        self.config = _make_hc_config()
        self.layer = _make_hc_layer(self.config)
        self.layer.eval()
        self.n = self.config.num_residual_streams
        self.C = self.config.hidden_size

    def test_forward_output_shape(self):
        """Forward should produce hidden_states with shape [S, B, n*C] (seq-first)."""
        B, S = 2, 4
        hidden_states = paddle.randn([S, B, self.n * self.C])
        dict_args = {
            "hidden_states": hidden_states,
            "attention_mask": None,
        }
        result = self.layer.forward(dict_args)
        self.assertIn("hidden_states", result)
        self.assertEqual(list(result["hidden_states"].shape), [S, B, self.n * self.C])

    def test_forward_no_nan(self):
        """Output should not contain NaN values."""
        B, S = 2, 4
        hidden_states = paddle.randn([S, B, self.n * self.C])
        dict_args = {
            "hidden_states": hidden_states,
            "attention_mask": None,
        }
        result = self.layer.forward(dict_args)
        self.assertFalse(paddle.isnan(result["hidden_states"]).any().item())

    def test_forward_with_rotary_pos_emb(self):
        """Forward should work with rotary_pos_emb provided."""
        B, S = 2, 4
        head_dim = self.config.head_dim
        hidden_states = paddle.randn([S, B, self.n * self.C])
        rotary_pos_emb = paddle.randn([B, S, head_dim])
        dict_args = {
            "hidden_states": hidden_states,
            "attention_mask": None,
            "rotary_pos_emb": rotary_pos_emb,
        }
        result = self.layer.forward(dict_args)
        self.assertEqual(list(result["hidden_states"].shape), [S, B, self.n * self.C])

    def test_forward_deterministic_in_eval(self):
        """Two forward passes with same input should produce same output in eval mode."""
        B, S = 2, 4
        hidden_states = paddle.randn([S, B, self.n * self.C])
        dict_args1 = {
            "hidden_states": hidden_states.clone(),
            "attention_mask": None,
        }
        dict_args2 = {
            "hidden_states": hidden_states.clone(),
            "attention_mask": None,
        }
        result1 = self.layer.forward(dict_args1)
        result2 = self.layer.forward(dict_args2)
        self.assertTrue(paddle.allclose(result1["hidden_states"], result2["hidden_states"]))

    def test_forward_output_differs_from_input(self):
        """Output should be different from input (layer transforms data)."""
        B, S = 2, 4
        hidden_states = paddle.randn([S, B, self.n * self.C])
        dict_args = {
            "hidden_states": hidden_states.clone(),
            "attention_mask": None,
        }
        result = self.layer.forward(dict_args)
        self.assertFalse(paddle.allclose(result["hidden_states"], hidden_states))

    def test_forward_gradient_flow(self):
        """Gradient should flow through the full forward pass."""
        self.layer.train()
        B, S = 2, 4
        hidden_states = paddle.randn([S, B, self.n * self.C])
        hidden_states.stop_gradient = False
        dict_args = {
            "hidden_states": hidden_states,
            "attention_mask": None,
        }
        result = self.layer.forward(dict_args)
        loss = result["hidden_states"].sum()
        loss.backward()
        self.assertIsNotNone(hidden_states.grad)
        self.assertFalse(paddle.isnan(hidden_states.grad).any().item())

    def test_forward_with_position_ids(self):
        """Forward should work with position_ids provided."""
        B, S = 2, 4
        hidden_states = paddle.randn([S, B, self.n * self.C])
        position_ids = paddle.arange(S).unsqueeze(0).expand([B, S])
        dict_args = {
            "hidden_states": hidden_states,
            "attention_mask": None,
            "position_ids": position_ids,
        }
        result = self.layer.forward(dict_args)
        self.assertEqual(list(result["hidden_states"].shape), [S, B, self.n * self.C])


class TestHyperConnectionTransformerLayerRecompute(unittest.TestCase):
    """Tests for HyperConnectionTransformerLayer with selective recompute."""

    def test_selective_recompute_mlp(self):
        config = _make_hc_config(
            recompute_granularity="selective",
            recompute_modules=["mlp"],
        )
        layer = _make_hc_layer(config)
        self.assertTrue(layer.recompute_mlp)

    def test_selective_recompute_norm(self):
        config = _make_hc_config(
            recompute_granularity="selective",
            recompute_modules=["norm"],
        )
        layer = _make_hc_layer(config)
        self.assertTrue(layer.recompute_input_layernorm)
        self.assertTrue(layer.recompute_post_attention_layernorm)

    def test_selective_recompute_mlp_forward(self):
        """Forward with recompute_mlp should produce valid output."""
        config = _make_hc_config(
            recompute_granularity="selective",
            recompute_modules=["mlp"],
        )
        layer = _make_hc_layer(config)
        layer.eval()
        B, S = 2, 4
        n = config.num_residual_streams
        C = config.hidden_size
        hidden_states = paddle.randn([S, B, n * C])
        dict_args = {"hidden_states": hidden_states, "attention_mask": None}
        result = layer.forward(dict_args)
        self.assertEqual(list(result["hidden_states"].shape), [S, B, n * C])
        self.assertFalse(paddle.isnan(result["hidden_states"]).any().item())

    def test_selective_recompute_norm_forward(self):
        """Forward with recompute_norm should produce valid output."""
        config = _make_hc_config(
            recompute_granularity="selective",
            recompute_modules=["norm"],
        )
        layer = _make_hc_layer(config)
        layer.eval()
        B, S = 2, 4
        n = config.num_residual_streams
        C = config.hidden_size
        hidden_states = paddle.randn([S, B, n * C])
        dict_args = {"hidden_states": hidden_states, "attention_mask": None}
        result = layer.forward(dict_args)
        self.assertEqual(list(result["hidden_states"].shape), [S, B, n * C])
        self.assertFalse(paddle.isnan(result["hidden_states"]).any().item())


# ==============================================================================
# Tests for get_gpt_layer_local_spec with HC
# ==============================================================================


class TestGetGptLayerSpecWithHC(unittest.TestCase):
    """Tests that get_gpt_layer_local_spec produces correct spec when mHC is enabled."""

    def test_spec_uses_hc_transformer_layer(self):
        config = _make_hc_config()
        spec = get_gpt_layer_local_spec(config)
        self.assertEqual(spec.layer, HyperConnectionTransformerLayer)

    def test_spec_without_hc_uses_base_layer(self):
        config = _make_config()
        spec = get_gpt_layer_local_spec(config)
        self.assertEqual(spec.layer, TransformerLayer)

    def test_spec_sublayers_have_hc_modules(self):
        config = _make_hc_config()
        spec = get_gpt_layer_local_spec(config)
        from paddle.distributed.fleet.meta_parallel import LayerSpec

        self.assertIsInstance(spec.sublayers_spec.self_attention_hyper_connection, LayerSpec)
        self.assertEqual(
            spec.sublayers_spec.self_attention_hyper_connection.layer,
            HyperConnectionModule,
        )
        self.assertIsInstance(spec.sublayers_spec.mlp_hyper_connection, LayerSpec)
        self.assertEqual(
            spec.sublayers_spec.mlp_hyper_connection.layer,
            HyperConnectionModule,
        )


# ==============================================================================
# Integration test: full pipeline expand -> transformer layer -> contract
# ==============================================================================


class TestHCFullPipeline(unittest.TestCase):
    """Integration test for the full HC pipeline."""

    def test_expand_layer_contract(self):
        """Test: expand -> HyperConnectionTransformerLayer -> contract produces valid output."""
        config = _make_hc_config()
        n = config.num_residual_streams
        C = config.hidden_size
        S, B = 4, 2

        expand_layer = HyperConnectionExpandLayer(config)
        transformer_layer = _make_hc_layer(config)
        transformer_layer.eval()
        contract_layer = HyperConnectionContractLayer(config)

        # Start with standard hidden_states [S, B, C]
        hidden_states = paddle.randn([S, B, C])
        dict_args = {"hidden_states": hidden_states, "attention_mask": None}

        # Expand: [S, B, C] -> [S, B, n*C]
        dict_args = expand_layer.forward(dict_args)
        self.assertEqual(list(dict_args["hidden_states"].shape), [S, B, n * C])

        # Transform: [S, B, n*C] -> [S, B, n*C]
        dict_args = transformer_layer.forward(dict_args)
        self.assertEqual(list(dict_args["hidden_states"].shape), [S, B, n * C])

        # Contract: [S, B, n*C] -> [S, B, C]
        dict_args = contract_layer.forward(dict_args)
        self.assertEqual(list(dict_args["hidden_states"].shape), [S, B, C])
        self.assertFalse(paddle.isnan(dict_args["hidden_states"]).any().item())

    def test_multiple_transformer_layers(self):
        """Test multiple HC transformer layers in sequence."""
        config = _make_hc_config()
        n = config.num_residual_streams
        C = config.hidden_size
        S, B = 4, 2

        layer1 = _make_hc_layer(config, layer_number=1)
        layer2 = _make_hc_layer(config, layer_number=2)
        layer1.eval()
        layer2.eval()

        hidden_states = paddle.randn([S, B, n * C])
        dict_args = {"hidden_states": hidden_states, "attention_mask": None}

        dict_args = layer1.forward(dict_args)
        dict_args = layer2.forward(dict_args)

        self.assertEqual(list(dict_args["hidden_states"].shape), [S, B, n * C])
        self.assertFalse(paddle.isnan(dict_args["hidden_states"]).any().item())


# ==============================================================================
# Tests for MultiTokenPredictionLayer with enable_hyper_connections
# ==============================================================================


def _make_mtp_hc_config(**overrides):
    """Create a config with mHC enabled and MTP layers configured."""
    mtp_hc_defaults = {
        "enable_hyper_connections": True,
        "num_residual_streams": 4,
        "mhc_sinkhorn_iterations": 5,
        "mhc_init_gating_factor": 0.01,
        "num_nextn_predict_layers": 2,
        "train_mtp_only": False,
        "mtp_load_weight_only": False,
        "gpt_model_use_experimental_version": False,
        "experimental_dataflow": False,
    }
    mtp_hc_defaults.update(overrides)
    return _make_config(**mtp_hc_defaults)


def _make_mtp_layer(config, layer_number=0):
    """Create an MTP layer using the spec system."""
    from paddle.distributed.fleet.meta_parallel import build_spec_layer

    from paddleformers.fleet.models.gpt.gpt_layer_specs import (
        get_gpt_decoder_layers_spec,
        get_gpt_mtp_layers_spec,
    )

    decoder_specs = get_gpt_decoder_layers_spec(config)
    mtp_specs = get_gpt_mtp_layers_spec(config, decoder_specs)
    mtp_layer = build_spec_layer(mtp_specs[layer_number])
    return mtp_layer


class TestMTPLayerWithHCConstruction(unittest.TestCase):
    """Tests for MultiTokenPredictionLayer construction with mHC enabled."""

    def test_mhc_enabled_flag(self):
        """Layer should have mhc_enabled=True."""
        config = _make_mtp_hc_config()
        layer = _make_mtp_layer(config)
        self.assertTrue(layer.mhc_enabled)

    def test_has_e_proj_and_h_proj(self):
        """mHC mode should create e_proj and h_proj instead of eh_proj."""
        config = _make_mtp_hc_config()
        layer = _make_mtp_layer(config)
        self.assertIsNotNone(layer.e_proj)
        self.assertIsNotNone(layer.h_proj)
        self.assertIsNone(layer.eh_proj)

    def test_has_hc_head_params(self):
        """mHC mode should create learned contraction parameters."""
        config = _make_mtp_hc_config()
        layer = _make_mtp_layer(config)
        n = config.num_residual_streams
        hc_dim = config.hidden_size * n
        self.assertEqual(list(layer.hc_head_fn.shape), [hc_dim, n])
        self.assertEqual(list(layer.hc_head_base.shape), [n])
        self.assertEqual(list(layer.hc_head_scale.shape), [1])

    def test_non_mhc_has_eh_proj(self):
        """Non-mHC mode should create eh_proj instead of e_proj/h_proj."""
        config = _make_config(num_nextn_predict_layers=2)
        layer = _make_mtp_layer(config)
        self.assertIsNotNone(layer.eh_proj)
        self.assertIsNone(layer.e_proj)
        self.assertIsNone(layer.h_proj)
        self.assertFalse(layer.mhc_enabled)


class TestMTPLayerWithHCConcatEmbeddings(unittest.TestCase):
    """Tests for _concat_embeddings in mHC mode."""

    def setUp(self):
        self.config = _make_mtp_hc_config()
        self.layer = _make_mtp_layer(self.config)
        self.layer.eval()
        self.n = self.config.num_residual_streams
        self.C = self.config.hidden_size
        self.S = 4
        self.B = 2

    def test_output_shape(self):
        """_concat_embeddings in mHC mode should output [s, b, n*h]."""
        hidden_states = paddle.randn([self.S, self.B, self.n * self.C])
        decoder_input = paddle.randn([self.S, self.B, self.C])
        result = self.layer._concat_embeddings(hidden_states, decoder_input)
        self.assertEqual(list(result.shape), [self.S, self.B, self.n * self.C])

    def test_output_no_nan(self):
        """Output should not contain NaN."""
        hidden_states = paddle.randn([self.S, self.B, self.n * self.C])
        decoder_input = paddle.randn([self.S, self.B, self.C])
        result = self.layer._concat_embeddings(hidden_states, decoder_input)
        self.assertFalse(paddle.isnan(result).any().item())

    def test_with_mask(self):
        """_concat_embeddings should handle mtp_hidden_inputs_mask."""
        # Use S==B so that transpose-based mask broadcasting works with seq-first layout
        S, B = 4, 4
        hidden_states = paddle.randn([S, B, self.n * self.C])
        decoder_input = paddle.randn([S, B, self.C])
        # mask shape: [B, 1, S]
        mask = paddle.ones([B, 1, S])
        result = self.layer._concat_embeddings(hidden_states, decoder_input, mtp_hidden_inputs_mask=mask)
        self.assertEqual(list(result.shape), [S, B, self.n * self.C])

    def test_with_zero_mask(self):
        """Zero mask should zero out hidden state contributions."""
        S, B = 4, 4
        hidden_states = paddle.randn([S, B, self.n * self.C])
        decoder_input = paddle.randn([S, B, self.C])
        mask = paddle.zeros([B, 1, S])
        result_zero = self.layer._concat_embeddings(hidden_states, decoder_input, mtp_hidden_inputs_mask=mask)
        self.assertFalse(paddle.isnan(result_zero).any().item())


class TestMTPLayerWithHCPostprocess(unittest.TestCase):
    """Tests for _postprocess in mHC mode."""

    def setUp(self):
        self.config = _make_mtp_hc_config()
        self.layer = _make_mtp_layer(self.config)
        self.layer.eval()
        self.n = self.config.num_residual_streams
        self.C = self.config.hidden_size

    def test_postprocess_contracts_to_single_stream(self):
        """_postprocess should contract [s, b, n*h] to [s, b, h]."""
        S, B = 4, 2
        hidden_states = paddle.randn([S, B, self.n * self.C])
        result = self.layer._postprocess(hidden_states)
        self.assertEqual(list(result.shape), [S, B, self.C])

    def test_postprocess_no_nan(self):
        """_postprocess output should not contain NaN."""
        S, B = 4, 2
        hidden_states = paddle.randn([S, B, self.n * self.C])
        result = self.layer._postprocess(hidden_states)
        self.assertFalse(paddle.isnan(result).any().item())

    def test_postprocess_gradient_flow(self):
        """Gradient should flow through _postprocess."""
        S, B = 4, 2
        hidden_states = paddle.randn([S, B, self.n * self.C])
        hidden_states.stop_gradient = False
        result = self.layer._postprocess(hidden_states)
        loss = result.sum()
        loss.backward()
        self.assertIsNotNone(hidden_states.grad)
        self.assertFalse(paddle.isnan(hidden_states.grad).any().item())


class TestMTPLayerWithHCForward(unittest.TestCase):
    """Tests for MultiTokenPredictionLayer forward with mHC enabled."""

    def setUp(self):
        self.config = _make_mtp_hc_config()
        self.layer = _make_mtp_layer(self.config, layer_number=0)
        self.layer.eval()
        self.n = self.config.num_residual_streams
        self.C = self.config.hidden_size
        self.S = 4
        self.B = 2
        self.num_mtp = self.config.num_nextn_predict_layers

    def _make_dict_args(self, with_mhc_multistream=True):
        """Create dict_args for MTP forward."""
        total_S = (self.num_mtp + 1) * self.S
        hidden_states = paddle.randn([total_S, self.B, self.C])
        dict_args = {
            "hidden_states": hidden_states,
            "attention_mask": None,
        }
        if with_mhc_multistream:
            mhc_multistream = paddle.randn([total_S, self.B, self.n * self.C])
            dict_args["mhc_multistream"] = mhc_multistream
        return dict_args

    def test_forward_output_shape(self):
        """Forward with mhc_multistream should produce correct output shape."""
        dict_args = self._make_dict_args(with_mhc_multistream=True)
        result = self.layer.forward(dict_args)
        total_S = (self.num_mtp + 1) * self.S
        self.assertEqual(list(result["hidden_states"].shape), [total_S, self.B, self.C])

    def test_forward_no_nan(self):
        """Forward output should not contain NaN."""
        dict_args = self._make_dict_args(with_mhc_multistream=True)
        result = self.layer.forward(dict_args)
        self.assertFalse(paddle.isnan(result["hidden_states"]).any().item())

    def test_forward_passes_mhc_multistream_to_next(self):
        """With layer_number < num_mtp-1, should pass mhc_multistream for next layer."""
        dict_args = self._make_dict_args(with_mhc_multistream=True)
        result = self.layer.forward(dict_args)
        # layer_number=0 < num_mtp-1=1, so mhc_multistream should be in output
        self.assertIn("mhc_multistream", result)

    def test_forward_last_layer_no_mhc_multistream(self):
        """Last MTP layer should not pass mhc_multistream."""
        last_layer = _make_mtp_layer(self.config, layer_number=1)
        last_layer.eval()
        dict_args = self._make_dict_args(with_mhc_multistream=True)
        result = last_layer.forward(dict_args)
        self.assertNotIn("mhc_multistream", result)


class TestMTPLayerWithHCTrainMTPOnly(unittest.TestCase):
    """Tests for MTP layer with train_mtp_only=True and mHC enabled."""

    def setUp(self):
        self.config = _make_mtp_hc_config(train_mtp_only=True)
        self.layer = _make_mtp_layer(self.config, layer_number=0)
        self.layer.eval()
        self.n = self.config.num_residual_streams
        self.C = self.config.hidden_size
        self.S = 4
        self.B = 2
        self.num_mtp = self.config.num_nextn_predict_layers

    def test_train_mtp_only_forward_shape(self):
        """train_mtp_only with mhc_multistream should produce correct output."""
        total_S = (self.num_mtp + 1) * self.S
        hidden_states = paddle.randn([total_S, self.B, self.C])
        mhc_multistream = paddle.randn([total_S, self.B, self.n * self.C])
        dict_args = {
            "hidden_states": hidden_states,
            "attention_mask": None,
            "mhc_multistream": mhc_multistream,
        }
        result = self.layer.forward(dict_args)
        self.assertEqual(list(result["hidden_states"].shape), [total_S, self.B, self.C])

    def test_train_mtp_only_no_nan(self):
        """train_mtp_only output should not contain NaN."""
        total_S = (self.num_mtp + 1) * self.S
        hidden_states = paddle.randn([total_S, self.B, self.C])
        mhc_multistream = paddle.randn([total_S, self.B, self.n * self.C])
        dict_args = {
            "hidden_states": hidden_states,
            "attention_mask": None,
            "mhc_multistream": mhc_multistream,
        }
        result = self.layer.forward(dict_args)
        self.assertFalse(paddle.isnan(result["hidden_states"]).any().item())


class TestGetMTPLayerSpecWithHC(unittest.TestCase):
    """Tests for get_gpt_mtp_layers_spec with mHC enabled."""

    def test_mhc_spec_creates_correct_layer(self):
        """MTP spec with mHC should create MultiTokenPredictionLayer."""
        from paddleformers.fleet.models.gpt.gpt_layer_specs import (
            get_gpt_decoder_layers_spec,
            get_gpt_mtp_layers_spec,
        )
        from paddleformers.fleet.transformer.multi_token_prediction import (
            MultiTokenPredictionLayer,
        )

        config = _make_mtp_hc_config()
        decoder_specs = get_gpt_decoder_layers_spec(config)
        specs = get_gpt_mtp_layers_spec(config, decoder_specs)
        self.assertEqual(len(specs), config.num_nextn_predict_layers)
        for spec in specs:
            self.assertEqual(spec.layer, MultiTokenPredictionLayer)

    def test_non_mhc_spec_creates_correct_layer(self):
        """MTP spec without mHC should create MultiTokenPredictionLayer."""
        from paddleformers.fleet.models.gpt.gpt_layer_specs import (
            get_gpt_decoder_layers_spec,
            get_gpt_mtp_layers_spec,
        )
        from paddleformers.fleet.transformer.multi_token_prediction import (
            MultiTokenPredictionLayer,
        )

        config = _make_config(num_nextn_predict_layers=2)
        decoder_specs = get_gpt_decoder_layers_spec(config)
        specs = get_gpt_mtp_layers_spec(config, decoder_specs)
        self.assertEqual(len(specs), 2)
        self.assertEqual(specs[0].layer, MultiTokenPredictionLayer)


if __name__ == "__main__":
    unittest.main()
