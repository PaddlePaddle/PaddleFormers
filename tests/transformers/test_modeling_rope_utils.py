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

import math
import unittest

import paddle

from paddleformers.transformers import LlamaRotaryEmbedding
from paddleformers.transformers.configuration_utils import PretrainedConfig
from paddleformers.transformers.modeling_rope_utils import (
    ROPE_INIT_FUNCTIONS,
    _compute_dynamic_ntk_parameters,
    _compute_linear_scaling_rope_parameters,
    _compute_llama3_parameters,
    _compute_longrope_parameters,
    _compute_yarn_parameters,
    dynamic_rope_update,
    rope_config_validation,
    standardize_rope_params,
)


class FakePretrainedConfig(PretrainedConfig):
    """A minimal fake config that mimics PretrainedConfig behavior."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class LlamaRotaryEmbeddingForwardWithLayerType(LlamaRotaryEmbedding):
    @dynamic_rope_update
    def forward(self, x, position_ids, layer_type=None):
        with paddle.amp.auto_cast(enable=False):
            inv_freq_expanded = self.inv_freq[None, :, None].float().expand([position_ids.shape[0], -1, 1])

            position_ids_expanded = position_ids[:, None, :].float()

            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose([0, 2, 1])

            emb = paddle.concat((freqs, freqs), axis=-1)

            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

            return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class RoPEUtilsTest(unittest.TestCase):
    def test_standardize_rope_params_without_rope_parameters(self):
        config = FakePretrainedConfig(
            rope_theta=10000.0,
            hidden_size=256,
            num_attention_heads=4,
        )
        standardize_rope_params(config)
        self.assertIn("rope_parameters", config.__dict__)
        self.assertEqual(config.rope_parameters["rope_theta"], 10000.0)
        self.assertEqual(config.rope_parameters["rope_type"], "default")

    def test_standardize_rope_params_backward_compatibility(self):
        config = FakePretrainedConfig(
            rope_theta=10000.0,
            hidden_size=256,
            num_attention_heads=4,
            rope_parameters={
                "type": "default",
            },
        )
        standardize_rope_params(config)
        self.assertIn("rope_parameters", config.__dict__)
        self.assertEqual(config.rope_parameters["rope_theta"], 10000.0)
        self.assertEqual(config.rope_parameters["rope_type"], "default")

    def test_standardize_rope_params(self):
        config = FakePretrainedConfig(
            rope_theta=10000.0,
            hidden_size=256,
            num_attention_heads=4,
            rope_parameters={
                "rope_type": "default",
            },
        )
        standardize_rope_params(config)
        self.assertIn("rope_parameters", config.__dict__)
        self.assertEqual(config.rope_parameters["rope_theta"], 10000.0)
        self.assertEqual(config.rope_parameters["rope_type"], "default")

    def test_standardize_rope_params_with_dict_per_layer_without_rope_parameters(self):
        config = FakePretrainedConfig(
            layer_types=["full_attention", "sliding_attention"],
            rope_theta={"full_attention": 10000.0, "sliding_attention": 15000.0},
            hidden_size=256,
            num_attention_heads=4,
        )
        standardize_rope_params(config)
        self.assertIn("rope_parameters", config.__dict__)
        self.assertEqual(config.rope_parameters["full_attention"]["rope_theta"], 10000.0)
        self.assertEqual(config.rope_parameters["sliding_attention"]["rope_theta"], 15000.0)

    def test_standardize_rope_params_with_dict_per_layer_not_in_new_format(self):
        config = FakePretrainedConfig(
            layer_types=["full_attention", "sliding_attention"],
            rope_theta={"full_attention": 10000.0, "sliding_attention": 15000.0},
            hidden_size=256,
            num_attention_heads=4,
            rope_parameters={
                "type": "default",
            },
        )
        standardize_rope_params(config)
        self.assertIn("rope_parameters", config.__dict__)
        self.assertEqual(config.rope_parameters["full_attention"]["rope_theta"], 10000.0)
        self.assertEqual(config.rope_parameters["full_attention"]["rope_type"], "default")
        self.assertEqual(config.rope_parameters["sliding_attention"]["rope_theta"], 15000.0)
        self.assertEqual(config.rope_parameters["sliding_attention"]["rope_type"], "default")

    def test_standardize_rope_params_with_dict_per_layer_in_new_format(self):
        config = FakePretrainedConfig(
            layer_types=["full_attention", "sliding_attention"],
            rope_theta={"full_attention": 10000.0, "sliding_attention": 15000.0},
            hidden_size=256,
            num_attention_heads=4,
            rope_parameters={"full_attention": {"rope_type": "default"}, "sliding_attention": {"rope_type": "linear"}},
        )
        standardize_rope_params(config)
        self.assertIn("rope_parameters", config.__dict__)
        self.assertEqual(config.rope_parameters["full_attention"]["rope_theta"], 10000.0)
        self.assertEqual(config.rope_parameters["full_attention"]["rope_type"], "default")
        self.assertEqual(config.rope_parameters["sliding_attention"]["rope_theta"], 15000.0)
        self.assertEqual(config.rope_parameters["sliding_attention"]["rope_type"], "linear")

    def test_compute_linear_scaling_rope_parameters(self):
        config = FakePretrainedConfig(
            rope_theta=10000.0,
            hidden_size=256,
            num_attention_heads=4,
            max_position_embeddings=2048,
            partial_rotary_factor=0.8,
            rope_parameters={"rope_type": "linear", "factor": 2.0, "rope_theta": 10000.0},
        )
        inv_freq, attn_factor = _compute_linear_scaling_rope_parameters(config)
        self.assertIsInstance(inv_freq, paddle.Tensor)
        expected_dim = int((config.hidden_size // config.num_attention_heads) * config.partial_rotary_factor + 1) // 2
        self.assertEqual(inv_freq.shape, [expected_dim])
        self.assertEqual(attn_factor, 1.0)
        self.assertTrue((inv_freq > 0).all())

    def test_compute_dynamic_ntk_parameters(self):
        # test with seq_len > max_position_embeddings -> trigger NTK scaling
        config = FakePretrainedConfig(
            rope_theta=10000.0,
            hidden_size=256,
            num_attention_heads=4,
            max_position_embeddings=2048,
            partial_rotary_factor=1.0,
            rope_parameters={"rope_type": "dynamic", "factor": 2.0, "rope_theta": 10000.0},
        )
        inv_freq, attn_factor = _compute_dynamic_ntk_parameters(config, seq_len=4096)
        self.assertIsInstance(inv_freq, paddle.Tensor)
        expected_dim = int((config.hidden_size // config.num_attention_heads) * config.partial_rotary_factor + 1) // 2
        self.assertEqual(inv_freq.shape, [expected_dim])
        self.assertTrue((inv_freq > 0).all())
        self.assertEqual(attn_factor, 1.0)

        # test with seq_len <= max_position_embeddings
        inv_freq_no_scale, _ = _compute_dynamic_ntk_parameters(config, seq_len=1024)
        base_no_scale = config.rope_theta
        dim = config.hidden_size // config.num_attention_heads
        expected_inv_freq_no_scale = 1.0 / (base_no_scale ** (paddle.arange(0, dim, 2, dtype=paddle.float32) / dim))
        self.assertTrue(paddle.allclose(inv_freq_no_scale, expected_inv_freq_no_scale, atol=1e-6))

        # test with seq_len None
        inv_freq, attn_factor = _compute_dynamic_ntk_parameters(config)
        self.assertIsInstance(inv_freq, paddle.Tensor)
        self.assertEqual(inv_freq.shape, [expected_dim])
        self.assertTrue((inv_freq > 0).all())
        self.assertEqual(attn_factor, 1.0)

        # test with seq_len paddle.Tensor
        inv_freq, attn_factor = _compute_dynamic_ntk_parameters(config, seq_len=paddle.to_tensor(1024))
        self.assertIsInstance(inv_freq, paddle.Tensor)
        self.assertEqual(inv_freq.shape, [expected_dim])
        self.assertTrue((inv_freq > 0).all())
        self.assertEqual(attn_factor, 1.0)

    def test_compute_yarn_parameters(self):
        config = FakePretrainedConfig(
            rope_theta=10000.0,
            hidden_size=256,
            num_attention_heads=4,
            max_position_embeddings=2048,
            partial_rotary_factor=0.6,
            rope_parameters={
                "rope_type": "yarn",
                "factor": 2.0,
                "rope_theta": 10000.0,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "mscale": 1.0,
                "mscale_all_dim": 1.0,
                "original_max_position_embeddings": 2048,
            },
        )
        inv_freq, attn_factor = _compute_yarn_parameters(config)
        self.assertIsInstance(inv_freq, paddle.Tensor)
        expected_dim = int((config.hidden_size // config.num_attention_heads) * config.partial_rotary_factor + 1) // 2
        self.assertEqual(inv_freq.shape, [expected_dim])
        self.assertTrue((inv_freq > 0).all())
        self.assertAlmostEqual(attn_factor, 1.0, places=6)

    def test_compute_yarn_parameters_without_mscale(self):
        config = FakePretrainedConfig(
            rope_theta=10000.0,
            hidden_size=256,
            num_attention_heads=4,
            max_position_embeddings=2048,
            partial_rotary_factor=0.6,
            rope_parameters={
                "rope_type": "yarn",
                "factor": 2.0,
                "rope_theta": 10000.0,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
            },
        )
        inv_freq, attn_factor = _compute_yarn_parameters(config)
        self.assertIsInstance(inv_freq, paddle.Tensor)
        expected_dim = int((config.hidden_size // config.num_attention_heads) * config.partial_rotary_factor + 1) // 2
        self.assertEqual(inv_freq.shape, [expected_dim])
        self.assertTrue((inv_freq > 0).all())
        expected_attention_factor = 0.1 * 1 * math.log(2.0) + 1.0
        self.assertAlmostEqual(attn_factor, expected_attention_factor, places=6)

    def test_compute_yarn_parameters_truncate_false(self):
        config = FakePretrainedConfig(
            rope_theta=10000.0,
            hidden_size=256,
            num_attention_heads=4,
            max_position_embeddings=2048,
            partial_rotary_factor=0.6,
            rope_parameters={
                "rope_type": "yarn",
                "factor": 2.0,
                "rope_theta": 10000.0,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "truncate": False,
            },
        )
        inv_freq, attn_factor = _compute_yarn_parameters(config)
        self.assertIsInstance(inv_freq, paddle.Tensor)
        expected_dim = int((config.hidden_size // config.num_attention_heads) * config.partial_rotary_factor + 1) // 2
        self.assertEqual(inv_freq.shape, [expected_dim])
        self.assertTrue((inv_freq > 0).all())

    def test_compute_longrope_parameters(self):
        dim_half = 32
        config = FakePretrainedConfig(
            rope_theta=10000.0,
            hidden_size=256,
            num_attention_heads=4,
            max_position_embeddings=4096,
            original_max_position_embeddings=2048,
            partial_rotary_factor=1.0,
            rope_parameters={
                "rope_type": "longrope",
                "factor": 2.0,
                "rope_theta": 10000.0,
                "short_factor": [1.0] * dim_half,
                "long_factor": [2.0] * dim_half,
                "original_max_position_embeddings": 2048,
            },
        )
        # test with seq_len >original_max_position_embeddings -> use long_factor
        inv_freq_long, attn_factor = _compute_longrope_parameters(config, seq_len=3000)
        self.assertIsInstance(inv_freq_long, paddle.Tensor)
        expected_dim = int((config.hidden_size // config.num_attention_heads) * config.partial_rotary_factor + 1) // 2
        self.assertEqual(inv_freq_long.shape, [expected_dim])
        self.assertTrue((inv_freq_long > 0).all())

        factor = config.max_position_embeddings / config.original_max_position_embeddings  # 4096 / 2048 = 2.0
        expected_attn_factor = math.sqrt(1 + math.log(factor) / math.log(config.original_max_position_embeddings))
        self.assertAlmostEqual(attn_factor, expected_attn_factor, places=6)

        # test with seq_len <= original_max_position_embeddings -> use short_factor
        inv_freq_short, attn_factor_short = _compute_longrope_parameters(config, seq_len=1000)
        self.assertEqual(inv_freq_short.shape, [expected_dim])
        self.assertTrue((inv_freq_short > 0).all())
        self.assertAlmostEqual(attn_factor_short, expected_attn_factor, places=6)

        self.assertTrue((inv_freq_long < inv_freq_short).all())

    def test_compute_longrope_parameters_without_original_max_position_embeddings(self):
        dim_half = 32
        config = FakePretrainedConfig(
            rope_theta=10000.0,
            hidden_size=256,
            num_attention_heads=4,
            max_position_embeddings=4096,
            partial_rotary_factor=1.0,
            rope_parameters={
                "rope_type": "longrope",
                "factor": 1.0,
                "rope_theta": 10000.0,
                "short_factor": [1.0] * dim_half,
                "long_factor": [2.0] * dim_half,
            },
        )
        # test with seq_len >original_max_position_embeddings -> use long_factor
        inv_freq_long, attn_factor = _compute_longrope_parameters(config, seq_len=5000)
        self.assertIsInstance(inv_freq_long, paddle.Tensor)
        expected_dim = int((config.hidden_size // config.num_attention_heads) * config.partial_rotary_factor + 1) // 2
        self.assertEqual(inv_freq_long.shape, [expected_dim])
        self.assertTrue((inv_freq_long > 0).all())

        expected_attn_factor = 1.0
        self.assertAlmostEqual(attn_factor, expected_attn_factor, places=6)

        # test with seq_len <= original_max_position_embeddings -> use short_factor
        inv_freq_short, attn_factor_short = _compute_longrope_parameters(config, seq_len=1000)
        self.assertEqual(inv_freq_short.shape, [expected_dim])
        self.assertTrue((inv_freq_short > 0).all())
        self.assertAlmostEqual(attn_factor_short, expected_attn_factor, places=6)

        self.assertTrue((inv_freq_long < inv_freq_short).all())

    def test_compute_llama3_parameters(self):
        config = FakePretrainedConfig(
            rope_theta=500000.0,
            hidden_size=256,
            num_attention_heads=4,
            partial_rotary_factor=1.0,
            rope_parameters={
                "rope_type": "llama3",
                "factor": 8.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192,
                "rope_theta": 500000.0,
            },
        )
        inv_freq, attn_factor = _compute_llama3_parameters(config)
        self.assertIsInstance(inv_freq, paddle.Tensor)
        expected_dim = int((config.hidden_size // config.num_attention_heads) * config.partial_rotary_factor + 1) // 2
        self.assertEqual(inv_freq.shape, [expected_dim])
        self.assertTrue((inv_freq > 0).all())
        self.assertEqual(attn_factor, 1.0)

        # High-frequency (first dim) should be unchanged
        base = config.rope_theta
        dim = int(config.hidden_size // config.num_attention_heads * config.partial_rotary_factor)
        expected_inv_freq_0 = 1.0 / (base ** (0 / dim))
        self.assertAlmostEqual(inv_freq[0].item(), expected_inv_freq_0, places=6)

        # Low-frequency (last dim): should be divided by factor=8
        freq_idx = dim - 2
        expected_inv_freq_last = 1.0 / (500000.0 ** (freq_idx / 64))
        wavelen = 2 * math.pi / expected_inv_freq_last
        self.assertGreater(wavelen, 8192)  # confirm it's low-freq
        expected_scaled = expected_inv_freq_last / 8.0
        self.assertAlmostEqual(inv_freq[-1].item(), expected_scaled, places=6)

    def test_rope_init_functions_coverage(self):
        expected_types = {"linear", "dynamic", "yarn", "longrope", "llama3"}
        self.assertEqual(set(ROPE_INIT_FUNCTIONS.keys()), expected_types)

    def test_rope_config_validation_dict_per_layer(self):
        config = FakePretrainedConfig(
            layer_types=["full_attention", "sliding_attention"],
            rope_parameters={
                "full_attention": {"rope_type": "default", "rope_theta": 10000.0},
                "sliding_attention": {"rope_type": "linear", "rope_theta": 15000.0, "factor": 1.0},
            },
        )
        rope_config_validation(config)

    def test_rope_config_validation_missing_validation_func_mapping(self):
        config = FakePretrainedConfig(rope_parameters={"rope_type": "defaulttt", "rope_theta": 10000.0})
        with self.assertLogs(logger="PaddleFormers", level="WARNING") as cm:
            rope_config_validation(config)
        self.assertIn("Missing validation function mapping in `ROPE_VALIDATION_FUNCTIONS`", cm.output[0])

    def test_rope_config_validation_default(self):
        config = FakePretrainedConfig(
            rope_parameters={
                "type": "default",
                "rope_theta": 10000.0,
                "ignore_key": "ignore_value",
                "unused_key": "unused_value",
            }
        )
        with self.assertLogs(logger="PaddleFormers", level="WARNING") as cm:
            rope_config_validation(config, ignore_keys={"ignore_key"})
        self.assertIn("Unrecognized keys in `rope_parameters`", cm.output[0])

    def test_rope_config_validation_linear_scaling_missing_key(self):
        config = FakePretrainedConfig(rope_parameters={"rope_type": "linear", "rope_theta": 10000.0})
        with self.assertRaises(KeyError):
            rope_config_validation(config)

    def test_rope_config_validation_linear_scaling_invalid_factor(self):
        config = FakePretrainedConfig(rope_parameters={"rope_type": "linear", "rope_theta": 10000.0, "factor": 0.5})
        with self.assertLogs(logger="PaddleFormers", level="WARNING") as cm:
            rope_config_validation(config)
        self.assertIn("factor field must be a float >= 1", cm.output[0])

    def test_rope_config_validation_dynamic_scsaling_invalid_params(self):
        config = FakePretrainedConfig(
            max_position_embeddings=16384,
            rope_parameters={
                "rope_type": "dynamic",
                "factor": 0.5,
                "rope_theta": 500000.0,
                "original_max_position_embeddings": 8192,
            },
        )
        with self.assertLogs(logger="PaddleFormers", level="WARNING") as cm:
            rope_config_validation(config)
        self.assertIn("factor field must be a float >= 1", cm.output[0])

    def test_rope_config_validation_yarn_invalid_params(self):
        config = FakePretrainedConfig(
            max_position_embeddings=16384,
            rope_parameters={
                "rope_type": "yarn",
                "attention_factor": -1.0,
                "factor": 0.5,
                "rope_theta": 500000.0,
                "beta_fast": 1,
                "beta_slow": 2,
                "original_max_position_embeddings": 8192,
            },
        )
        with self.assertLogs(logger="PaddleFormers", level="WARNING") as cm:
            rope_config_validation(config)
        messages = [record.getMessage() for record in cm.records]

        self.assertTrue(any("factor field must be a float >= 1" in msg for msg in messages))
        self.assertTrue(any("attention_factor field must be a float greater than 0" in msg for msg in messages))
        self.assertTrue(any("beta_fast field must be a float" in msg for msg in messages))
        self.assertTrue(any("beta_slow field must be a float" in msg for msg in messages))
        self.assertTrue(
            any("`rope_parameters`'s beta_fast field must be greater than beta_slow" in msg for msg in messages)
        )
        self.assertTrue(
            any("please correct the 'max_position_embeddings' fields in the model config" in msg for msg in messages)
        )

    def test_rope_config_validation_llama3_invalid_params(self):
        config = FakePretrainedConfig(
            max_position_embeddings=16384,
            rope_parameters={
                "rope_type": "llama3",
                "factor": 0.5,
                "low_freq_factor": 2,
                "high_freq_factor": 1,
                "original_max_position_embeddings": 16384.0,
                "rope_theta": 500000.0,
            },
        )
        with self.assertLogs(logger="PaddleFormers", level="WARNING") as cm:
            rope_config_validation(config)
        messages = [record.getMessage() for record in cm.records]

        self.assertTrue(any("factor field must be a float >= 1" in msg for msg in messages))
        self.assertTrue(any("low_freq_factor field must be a float" in msg for msg in messages))
        self.assertTrue(any("high_freq_factor field must be a float" in msg for msg in messages))
        self.assertTrue(any("high_freq_factor field must be greater than low_freq_factor" in msg for msg in messages))
        self.assertTrue(any("original_max_position_embeddings field must be an integer" in msg for msg in messages))
        self.assertTrue(
            any(
                "original_max_position_embeddings field must be less than max_position_embeddings" in msg
                for msg in messages
            )
        )

    def test_rope_config_validation_longrope_invalid_params(self):
        config = FakePretrainedConfig(
            hidden_size=256,
            num_attention_heads=4,
            max_position_embeddings=4096,
            original_max_position_embeddings=2048,
            partial_rotary_factor=1.0,
            rope_parameters={
                "rope_type": "longrope",
                "rope_theta": 10000.0,
                "short_factor": [1.0] * 10,  # wrong length
                "long_factor": [2.0] * 10,
                "factor": 2.0,
            },
        )
        with self.assertLogs(logger="PaddleFormers", level="WARNING") as cm:
            rope_config_validation(config)
        dim = int(config.hidden_size // config.num_attention_heads * config.partial_rotary_factor)
        messages = [record.getMessage() for record in cm.records]
        self.assertTrue(any(f"long_factor field must have length {dim // 2}" in msg for msg in messages))
        self.assertTrue(any(f"short_factor field must have length {dim // 2}" in msg for msg in messages))
        self.assertTrue(
            any("This model has set a `original_max_position_embeddings` field" in msg for msg in messages)
        )

    def test_rope_config_validation_longrope_invalid_short_factor_type(self):
        config = FakePretrainedConfig(
            hidden_size=256,
            num_attention_heads=4,
            max_position_embeddings=4096,
            partial_rotary_factor=1.0,
            rope_parameters={
                "rope_type": "longrope",
                "rope_theta": 10000.0,
                "short_factor": ["test"] * 10,
                "long_factor": ["test"] * 10,
            },
        )
        with self.assertLogs(logger="PaddleFormers", level="WARNING") as cm:
            rope_config_validation(config)
        messages = [record.getMessage() for record in cm.records]
        self.assertTrue(any("short_factor field must be a list of numbers" in msg for msg in messages))
        self.assertTrue(any("long_factor field must be a list of numbers" in msg for msg in messages))
        self.assertTrue(any("Missing required keys in `rope_parameters`: 'factor'" in msg for msg in messages))

    def test_rope_config_validation_longrope_invalid_factor(self):
        config = FakePretrainedConfig(
            hidden_size=256,
            num_attention_heads=4,
            max_position_embeddings=4096,
            partial_rotary_factor=1.0,
            rope_parameters={
                "rope_type": "longrope",
                "rope_theta": 10000.0,
                "short_factor": [1.0] * 32,
                "long_factor": [2.0] * 32,
                "factor": 0.5,
                "attention_factor": -1.0,
            },
        )
        with self.assertLogs(logger="PaddleFormers", level="WARNING") as cm:
            rope_config_validation(config)
        messages = [record.getMessage() for record in cm.records]
        self.assertTrue(any("rope_parameters`'s factor field must be a float >= 1" in msg for msg in messages))
        self.assertTrue(
            any("`rope_parameters`'s attention_factor field must be a float greater than 0" in msg for msg in messages)
        )

    def test_default_rope_numerically(self):
        # fmt: off
        EXPECTED_INV_FREQ = paddle.to_tensor(
            [
                1.0000e+00, 8.6596e-01, 7.4989e-01, 6.4938e-01, 5.6234e-01, 4.8697e-01,
                4.2170e-01, 3.6517e-01, 3.1623e-01, 2.7384e-01, 2.3714e-01, 2.0535e-01,
                1.7783e-01, 1.5399e-01, 1.3335e-01, 1.1548e-01, 1.0000e-01, 8.6596e-02,
                7.4989e-02, 6.4938e-02, 5.6234e-02, 4.8697e-02, 4.2170e-02, 3.6517e-02,
                3.1623e-02, 2.7384e-02, 2.3714e-02, 2.0535e-02, 1.7783e-02, 1.5399e-02,
                1.3335e-02, 1.1548e-02, 1.0000e-02, 8.6596e-03, 7.4989e-03, 6.4938e-03,
                5.6234e-03, 4.8697e-03, 4.2170e-03, 3.6517e-03, 3.1623e-03, 2.7384e-03,
                2.3714e-03, 2.0535e-03, 1.7783e-03, 1.5399e-03, 1.3335e-03, 1.1548e-03,
                1.0000e-03, 8.6596e-04, 7.4989e-04, 6.4938e-04, 5.6234e-04, 4.8697e-04,
                4.2170e-04, 3.6517e-04, 3.1623e-04, 2.7384e-04, 2.3714e-04, 2.0535e-04,
                1.7783e-04, 1.5399e-04, 1.3335e-04, 1.1548e-04
            ]
        )
        # fmt: on

        config = FakePretrainedConfig(
            hidden_size=4096,
            num_attention_heads=32,
            rope_theta=10000.0,
            max_position_embeddings=64,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        )
        standardize_rope_params(config)

        rope_fn = LlamaRotaryEmbedding.compute_default_rope_parameters
        inv_freq, attention_scale = rope_fn(config=config)

        self.assertEqual(attention_scale, 1.0)  # attention scale is always 1 for default RoPE
        self.assertTrue(paddle.allclose(inv_freq, EXPECTED_INV_FREQ, rtol=1e-4, atol=1e-6))

    def test_linear_rope_numerically(self):
        config = FakePretrainedConfig(
            hidden_size=4096,
            num_attention_heads=32,
            rope_theta=10000.0,
            max_position_embeddings=64,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        )
        default_rope_fn = LlamaRotaryEmbedding.compute_default_rope_parameters
        default_inv_freq, _ = default_rope_fn(config=config)

        rope_fn = ROPE_INIT_FUNCTIONS["linear"]
        for factor in (2.0, 10.0, 20.0):
            config.rope_parameters = {"rope_type": "linear", "rope_theta": 10000.0, "factor": factor}
            inv_freq, attention_scale = rope_fn(config=config)
            self.assertEqual(attention_scale, 1.0)  # attention scale is always 1 for linear RoPE
            self.assertTrue(paddle.allclose(inv_freq, default_inv_freq / factor, rtol=1e-4, atol=1e-6))

    def test_dynamic_rope_numerically(self):
        # fmt: off
        EXPECTED_INV_FREQ = paddle.to_tensor(
            [
                1.0000e+00, 8.0931e-01, 6.5498e-01, 5.3008e-01, 4.2900e-01, 3.4720e-01,
                2.8099e-01, 2.2741e-01, 1.8404e-01, 1.4895e-01, 1.2055e-01, 9.7558e-02,
                7.8955e-02, 6.3899e-02, 5.1714e-02, 4.1853e-02, 3.3872e-02, 2.7413e-02,
                2.2185e-02, 1.7955e-02, 1.4531e-02, 1.1760e-02, 9.5176e-03, 7.7027e-03,
                6.2339e-03, 5.0451e-03, 4.0831e-03, 3.3045e-03, 2.6744e-03, 2.1644e-03,
                1.7517e-03, 1.4176e-03, 1.1473e-03, 9.2852e-04, 7.5146e-04, 6.0817e-04,
                4.9220e-04, 3.9834e-04, 3.2238e-04, 2.6091e-04, 2.1115e-04, 1.7089e-04,
                1.3830e-04, 1.1193e-04, 9.0585e-05, 7.3312e-05, 5.9332e-05, 4.8018e-05,
                3.8861e-05, 3.1451e-05, 2.5453e-05, 2.0600e-05, 1.6672e-05, 1.3492e-05,
                1.0920e-05, 8.8374e-06, 7.1522e-06, 5.7883e-06, 4.6845e-06, 3.7912e-06,
                3.0683e-06, 2.4832e-06, 2.0097e-06, 1.6265e-06
            ]
        )
        # fmt: on

        config = FakePretrainedConfig(
            hidden_size=4096,
            num_attention_heads=32,
            rope_theta=10000.0,
            max_position_embeddings=2048,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        )
        default_rope_fn = LlamaRotaryEmbedding.compute_default_rope_parameters
        default_inv_freq, _ = default_rope_fn(config=config)

        rope_fn = ROPE_INIT_FUNCTIONS["dynamic"]
        for factor in (2.0, 10.0, 20.0):
            config.rope_parameters = {"rope_type": "dynamic", "rope_theta": 10000.0, "factor": factor}
            inv_freq, attention_scale = rope_fn(config=config)
            self.assertEqual(attention_scale, 1.0)  # attention scale is always 1 for dynamic RoPE
            self.assertTrue(paddle.allclose(inv_freq, default_inv_freq, rtol=1e-4, atol=1e-6))

            inv_freq, _ = rope_fn(config=config, seq_len=1)
            self.assertTrue(paddle.allclose(inv_freq, default_inv_freq, rtol=1e-4, atol=1e-6))

            inv_freq, _ = rope_fn(config=config, seq_len=paddle.to_tensor(1, dtype=paddle.int64))
            self.assertTrue(paddle.allclose(inv_freq, default_inv_freq, rtol=1e-4, atol=1e-6))

        factor = 10.0
        config.rope_parameters = {"rope_type": "dynamic", "rope_theta": 10000.0, "factor": factor}
        inv_freq, _ = rope_fn(config=config, seq_len=16384)
        with self.assertRaises(AssertionError):  # It is NOT a linear factor
            self.assertTrue(paddle.allclose(inv_freq, default_inv_freq / factor, rtol=1e-4, atol=1e-6))
        self.assertTrue(paddle.allclose(inv_freq, EXPECTED_INV_FREQ, rtol=1e-4, atol=1e-6))

    def test_yarn_rope_numerically(self):
        # fmt: off
        EXPECTED_INV_FREQ = paddle.to_tensor(
            [
                1.0000e+00, 8.6596e-01, 7.4989e-01, 6.4938e-01, 5.6234e-01, 4.8697e-01,
                4.2170e-01, 3.6517e-01, 3.1623e-01, 2.7384e-01, 2.3714e-01, 2.0535e-01,
                1.7783e-01, 1.5399e-01, 1.3335e-01, 1.1548e-01, 1.0000e-01, 8.3479e-02,
                6.9590e-02, 5.7925e-02, 4.8136e-02, 3.9931e-02, 3.3061e-02, 2.7315e-02,
                2.2515e-02, 1.8512e-02, 1.5177e-02, 1.2403e-02, 1.0101e-02, 8.1924e-03,
                6.6143e-03, 5.3120e-03, 4.2400e-03, 3.3599e-03, 2.6396e-03, 2.0520e-03,
                1.5746e-03, 1.1882e-03, 8.7713e-04, 6.2810e-04, 4.3007e-04, 2.7384e-04,
                2.3714e-04, 2.0535e-04, 1.7783e-04, 1.5399e-04, 1.3335e-04, 1.1548e-04,
                1.0000e-04, 8.6596e-05, 7.4989e-05, 6.4938e-05, 5.6234e-05, 4.8697e-05,
                4.2170e-05, 3.6517e-05, 3.1623e-05, 2.7384e-05, 2.3714e-05, 2.0535e-05,
                1.7783e-05, 1.5399e-05, 1.3335e-05, 1.1548e-05
            ]
        )
        # fmt: on

        config = FakePretrainedConfig(
            hidden_size=4096,
            num_attention_heads=32,
            rope_theta=10000.0,
            max_position_embeddings=2048,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        )

        rope_fn = LlamaRotaryEmbedding.compute_default_rope_parameters
        default_inv_freq, _ = rope_fn(config=config)

        rope_fn = ROPE_INIT_FUNCTIONS["yarn"]
        for factor in (2.0, 10.0, 20.0):
            config.rope_parameters = {"rope_type": "yarn", "rope_theta": 10000.0, "factor": factor}
            _, attention_scale = rope_fn(config=config)
            self.assertEqual(attention_scale, 0.1 * math.log(factor) + 1.0)

            config.rope_parameters = {
                "rope_type": "yarn",
                "rope_theta": 10000.0,
                "factor": factor,
                "attention_factor": 0.5,
            }
            _, attention_scale = rope_fn(config=config, seq_len=1)
            self.assertEqual(attention_scale, 0.5)

        factor = 10.0
        margin = 1e-8
        config.rope_parameters = {
            "rope_type": "yarn",
            "rope_theta": 10000.0,
            "factor": factor,
            "beta_fast": 32,
            "beta_slow": 1,
        }
        inv_freq, _ = rope_fn(config=config)
        is_bounded_by_factor = [
            ((default_inv_freq[idx] / factor) - margin) <= yarn_inv_freq_value <= (default_inv_freq[idx] + margin)
            for idx, yarn_inv_freq_value in enumerate(inv_freq)
        ]
        self.assertTrue(all(is_bounded_by_factor))

        config.rope_parameters = {
            "rope_type": "yarn",
            "rope_theta": 10000.0,
            "factor": factor,
            "beta_fast": 1000,
            "beta_slow": 1,
        }
        inv_freq, _ = rope_fn(config=config)
        is_interpolating = [
            yarn_inv_freq_value < (default_inv_freq[idx] + margin) for idx, yarn_inv_freq_value in enumerate(inv_freq)
        ]
        self.assertFalse(is_interpolating[0])
        self.assertTrue(all(is_interpolating[1:]))
        self.assertTrue(paddle.allclose(inv_freq[-20:], default_inv_freq[-20:] / factor, rtol=1e-4, atol=1e-6))

        config.rope_parameters = {
            "rope_type": "yarn",
            "rope_theta": 10000.0,
            "factor": factor,
            "beta_fast": 32,
            "beta_slow": 1,
        }
        inv_freq, _ = rope_fn(config=config)
        self.assertTrue(paddle.allclose(inv_freq, EXPECTED_INV_FREQ, rtol=1e-4, atol=1e-6))

    def test_longrope_rope_numerically(self):
        config = FakePretrainedConfig(
            hidden_size=4096,
            num_attention_heads=32,
            rope_theta=10000.0,
            max_position_embeddings=2048,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        )

        # longrope applies scaling on EACH inv frequency, `short_factor` or `long_factor`, depending on the seq_len
        dim = config.hidden_size // config.num_attention_heads
        short_factor = [2.0] * (dim // 2)  # scaling applied when seq_len <= max_position_embeddings
        long_factor = (
            paddle.ones(dim // 2).cumsum(0).tolist()
        )  # scaling applied when seq_len > max_position_embeddings

        rope_fn = LlamaRotaryEmbedding.compute_default_rope_parameters
        default_inv_freq, _ = rope_fn(config=config)

        rope_fn = ROPE_INIT_FUNCTIONS["longrope"]
        max_position_embeddings = config.max_position_embeddings
        for factor in (2.0, 10.0, 20.0):
            config.rope_parameters = {
                "rope_type": "longrope",
                "rope_theta": 10000.0,
                "factor": factor,
                "short_factor": short_factor,
                "long_factor": long_factor,
            }
            _, attention_scale = rope_fn(config=config)
            self.assertEqual(attention_scale, math.sqrt(1 + math.log(factor) / math.log(max_position_embeddings)))

            config.rope_parameters = {
                "rope_type": "longrope",
                "rope_theta": 10000.0,
                "factor": factor,
                "short_factor": short_factor,
                "long_factor": long_factor,
                "attention_factor": 0.5,
            }
            _, attention_scale = rope_fn(config=config, seq_len=1)
            self.assertEqual(attention_scale, 0.5)

            config.rope_parameters = {
                "rope_type": "longrope",
                "rope_theta": 10000.0,
                "factor": factor,
                "short_factor": short_factor,
                "long_factor": long_factor,
            }
            self.assertEqual(config.rope_parameters.get("attention_factor"), None)
            # Verify that "TypeError: '<' not supported between instances of 'NoneType' and 'int'" is not raised.
            rope_config_validation(config)

        config.rope_parameters = {
            "rope_type": "longrope",
            "rope_theta": 10000.0,
            "factor": 1.0,
            "short_factor": short_factor,
            "long_factor": long_factor,
        }
        inv_freq, _ = rope_fn(config=config, seq_len=0)
        self.assertTrue(
            paddle.allclose(inv_freq, default_inv_freq / paddle.to_tensor(short_factor), rtol=1e-4, atol=1e-6)
        )

        inv_freq, _ = rope_fn(config=config, seq_len=config.max_position_embeddings + 1)
        self.assertTrue(
            paddle.allclose(inv_freq, default_inv_freq / paddle.to_tensor(long_factor), rtol=1e-4, atol=1e-6)
        )

    def test_llama3_rope_numerically(self):
        # fmt: off
        EXPECTED_INV_FREQ = paddle.to_tensor(
            [
                1.0000e+00, 8.6596e-01, 7.4989e-01, 6.4938e-01, 5.6234e-01, 4.8697e-01,
                4.2170e-01, 3.6517e-01, 3.1623e-01, 2.7384e-01, 2.3714e-01, 2.0535e-01,
                1.7783e-01, 1.5399e-01, 1.3335e-01, 1.1548e-01, 1.0000e-01, 8.6596e-02,
                7.4989e-02, 6.4938e-02, 5.6234e-02, 4.8697e-02, 4.2170e-02, 3.6517e-02,
                3.1623e-02, 2.7384e-02, 2.3714e-02, 2.0535e-02, 1.7783e-02, 1.5399e-02,
                1.3335e-02, 1.0730e-02, 7.7785e-03, 5.6009e-03, 3.9991e-03, 2.8248e-03,
                1.9675e-03, 1.3449e-03, 8.9549e-04, 5.7363e-04, 3.4539e-04, 2.7384e-04,
                2.3714e-04, 2.0535e-04, 1.7783e-04, 1.5399e-04, 1.3335e-04, 1.1548e-04,
                1.0000e-04, 8.6596e-05, 7.4989e-05, 6.4938e-05, 5.6234e-05, 4.8697e-05,
                4.2170e-05, 3.6517e-05, 3.1623e-05, 2.7384e-05, 2.3714e-05, 2.0535e-05,
                1.7783e-05, 1.5399e-05, 1.3335e-05, 1.1548e-05
            ]
        )
        # fmt: on

        config = FakePretrainedConfig(
            hidden_size=4096,
            num_attention_heads=32,
            rope_theta=10000.0,
            max_position_embeddings=2048,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        )

        rope_fn = LlamaRotaryEmbedding.compute_default_rope_parameters
        default_inv_freq, _ = rope_fn(config=config)

        rope_fn = ROPE_INIT_FUNCTIONS["llama3"]
        for factor in (2.0, 10.0, 20.0):
            config.rope_parameters = {
                "rope_type": "llama3",
                "rope_theta": 10000.0,
                "factor": factor,
                "original_max_position_embeddings": 2048,
                "low_freq_factor": 1,
                "high_freq_factor": 4,
            }
            _, attention_scale = rope_fn(config=config)
            self.assertEqual(attention_scale, 1.0)

        factor = 10.0
        config.rope_parameters = {
            "rope_type": "llama3",
            "rope_theta": 10000.0,
            "factor": factor,
            "original_max_position_embeddings": 2048,
            "low_freq_factor": 1,
            "high_freq_factor": 4,
        }
        inv_freq, _ = rope_fn(config=config)
        is_bounded_by_factor = [
            (default_inv_freq[idx] / factor) <= llama3_inv_freq_value <= default_inv_freq[idx]
            for idx, llama3_inv_freq_value in enumerate(inv_freq)
        ]
        self.assertTrue(all(is_bounded_by_factor))

        config.rope_parameters = config.rope_parameters = {
            "rope_type": "llama3",
            "rope_theta": 10000.0,
            "factor": factor,
            "original_max_position_embeddings": 2048,
            "low_freq_factor": 1,
            "high_freq_factor": 1000,
        }
        inv_freq, _ = rope_fn(config=config)
        is_scaled = [yarn_inv_freq_value < default_inv_freq[idx] for idx, yarn_inv_freq_value in enumerate(inv_freq)]
        self.assertTrue(all(is_scaled))

        config.rope_parameters = {
            "rope_type": "llama3",
            "rope_theta": 10000.0,
            "factor": factor,
            "original_max_position_embeddings": 2048,
            "low_freq_factor": 1,
            "high_freq_factor": 4,
        }
        inv_freq, _ = rope_fn(config=config)
        self.assertTrue(paddle.allclose(inv_freq, EXPECTED_INV_FREQ, rtol=1e-4, atol=1e-6))

    def test_dynamic_rope_grows_cache(self):
        config = FakePretrainedConfig(
            hidden_size=256,
            num_attention_heads=4,
            rope_theta=10000.0,
            max_position_embeddings=64,
            rope_parameters={"rope_type": "dynamic", "factor": 2.0},
        )
        standardize_rope_params(config)
        model = LlamaRotaryEmbedding(config)
        model.max_seq_len_cached = 80
        model.original_max_seq_len = 64

        # short sequence length < max_position_embeddings
        x = paddle.randn([1, 10, 64])  # (batch, seq_len, head_dim)
        pos_ids_short = paddle.arange(10).unsqueeze(0)  # [1, 10]

        cos1, sin1 = model(x, pos_ids_short)

        # long sequence length > max_position_embeddings
        pos_ids_long = paddle.arange(100).unsqueeze(0)  # [1, 100]
        cos2, sin2 = model(x, pos_ids_long)

        self.assertFalse(paddle.allclose(cos1, cos2[:1, :10, :]), msg="Dynamic RoPE should change output for long seq")

    def test_longrope_switches_freq(self):
        config = FakePretrainedConfig(
            hidden_size=256,
            num_attention_heads=4,
            rope_theta=10000.0,
            max_position_embeddings=256,
            original_max_position_embeddings=64,
            rope_parameters={"rope_type": "longrope", "long_factor": [1.0] * 32, "short_factor": [2.0] * 32},
        )
        standardize_rope_params(config)
        model = LlamaRotaryEmbedding(config)

        x = paddle.randn([1, 10, 64])
        pos_ids_short = paddle.arange(10).unsqueeze(0)
        pos_ids_long = paddle.arange(100).unsqueeze(0)

        cos1, _ = model(x, pos_ids_short)
        original_inv_freq_copy = model.original_inv_freq.clone()

        cos2, _ = model(x, pos_ids_long)

        self.assertTrue(hasattr(model, "long_inv_freq"))
        self.assertFalse(paddle.allclose(model.inv_freq, original_inv_freq_copy))

        cos3, _ = model(x, pos_ids_short)
        self.assertTrue(paddle.allclose(model.inv_freq, original_inv_freq_copy.to(model.inv_freq.place)))

    def test_rope_with_layer_type(self):
        config = FakePretrainedConfig(
            hidden_size=256,
            num_attention_heads=4,
            rope_theta=10000.0,
            max_position_embeddings=64,
            rope_parameters={
                "full_attention": {"rope_theta": 10000.0, "factor": 2.0},
                "sliding_attention": {"rope_theta": 15000.0, "long_factor": [2.0] * 32, "short_factor": [2.0] * 32},
            },
        )
        standardize_rope_params(config)
        model = LlamaRotaryEmbeddingForwardWithLayerType(config)
        model.max_seq_len_cached = 32
        model.original_max_seq_len = 64
        model.rope_type = {"full_attention": "dynamic", "sliding_attention": "longrope"}
        model.full_attention_original_inv_freq = model.inv_freq.clone()
        model.sliding_attention_original_inv_freq = model.inv_freq.clone()

        x = paddle.randn([1, 50, 64])
        pos_ids = paddle.arange(50).unsqueeze(0)

        cos, sin = model(x, pos_ids, layer_type="full_attention")

        self.assertTrue(hasattr(model, "full_attention_inv_freq"))
        self.assertGreater(getattr(model, "full_attention_max_seq_len_cached", 0), 32)

        cos, sin = model(x, pos_ids, layer_type="sliding_attention")
        self.assertTrue(hasattr(model, "sliding_attention_inv_freq"))


if __name__ == "__main__":
    unittest.main()
