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
from paddle import nn

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


class FakeRotaryEmbedding(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        base = config.rope_theta
        partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        rope_parameters = self.config.rope_parameters
        self.rope_type = rope_parameters.get("rope_type", rope_parameters.get("type", "default"))
        dim = int(head_dim * partial_rotary_factor)

        inv_freq = 1.0 / (base ** (paddle.arange(0, dim, 2, dtype=paddle.int64).astype(dtype=paddle.float32) / dim))
        self.attention_scaling = 1.0
        self.register_buffer("inv_freq", inv_freq, persistable=False)
        self.original_inv_freq = self.inv_freq

    @dynamic_rope_update
    def forward(self, x, position_ids, layer_type=None):
        inv_freq = getattr(self, f"{layer_type}_inv_freq", self.inv_freq)
        attention_scaling = getattr(self, f"{layer_type}_attention_scaling", 1.0)

        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.place)
        position_ids_expanded = position_ids[:, None, :].float()

        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = paddle.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * attention_scaling
        sin = emb.sin() * attention_scaling

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

    def test_dynamic_rope_grows_cache(self):
        config = FakePretrainedConfig(
            hidden_size=256,
            num_attention_heads=4,
            rope_theta=10000.0,
            max_position_embeddings=64,
            rope_parameters={"rope_type": "dynamic", "factor": 2.0},
        )
        standardize_rope_params(config)
        model = FakeRotaryEmbedding(config)
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
            rope_parameters={"rope_type": "longrope", "long_factor": [2.0] * 32, "short_factor": [2.0] * 32},
        )
        standardize_rope_params(config)
        model = FakeRotaryEmbedding(config)

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
        model = FakeRotaryEmbedding(config)
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
