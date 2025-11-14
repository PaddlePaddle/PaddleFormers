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

from paddleformers.transformers.configuration_utils import PretrainedConfig
from paddleformers.transformers.modeling_rope_utils import (
    ROPE_INIT_FUNCTIONS,
    _compute_dynamic_ntk_parameters,
    _compute_linear_scaling_rope_parameters,
    _compute_llama3_parameters,
    _compute_longrope_parameters,
    _compute_yarn_parameters,
    rope_config_validation,
    standardize_rope_params,
)


class FakePretrainedConfig(PretrainedConfig):
    """A minimal fake config that mimics PretrainedConfig behavior."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


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

    def test_rope_config_validation_default(self):
        config = FakePretrainedConfig(rope_parameters={"rope_type": "default", "rope_theta": 10000.0})
        rope_config_validation(config)

    def test_rope_config_validation_linear_scaling_missing_key(self):
        config = FakePretrainedConfig(rope_parameters={"rope_type": "linear", "rope_theta": 10000.0})
        with self.assertRaises(KeyError):
            rope_config_validation(config)

    def test_rope_config_validation_linear_scaling_invalid_factor(self):
        config = FakePretrainedConfig(rope_parameters={"rope_type": "linear", "rope_theta": 10000.0, "factor": 0.5})
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
                "original_max_position_embeddings": 16384,
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


if __name__ == "__main__":
    unittest.main()
