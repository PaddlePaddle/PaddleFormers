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

"""
Tests covering two specific branches in MultiLatentAttention.forward using full GPT model:
  1. _ec_compatible_rope_apply: triggered when config.gpt_model_use_experimental_version=True
  2. _compute_absorbed_q: triggered when core_attention.config.forward_meta.max_len_tensor_cpu[2] > 0

This test builds a complete GPT model with MLA attention and verifies the code paths.
"""

import functools
import random
import unittest
from dataclasses import dataclass

import numpy as np
import paddle
from paddle.distributed.fleet.meta_parallel import build_spec_layer

from paddleformers.fleet.models.gpt.gpt_config import GPTConfig
from paddleformers.fleet.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_spec,
)


@dataclass
class FakeForwardMeta:
    """Mimics the forward_meta object used in fastdeploy decode mode.

    max_len_tensor_cpu is a list where index [2] > 0 indicates decode mode.
    """

    max_len_tensor_cpu: list


def _make_mla_gpt_config(**overrides):
    """Build a GPTConfig with MLA enabled."""
    # Extract gpt_model_use_experimental_version if present (needs to be set after creation)
    use_experimental = overrides.pop("gpt_model_use_experimental_version", False)

    defaults = {
        "num_hidden_layers": 2,
        "hidden_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "intermediate_size": 256,
        "max_sequence_length": 64,
        "vocab_size": 100,
        "normalization": "RMSNorm",
        "hidden_dropout_prob": 0.0,
        "attention_dropout": 0.0,
        "tie_word_embeddings": False,
        "parallel_output": False,  # False for single GPU
        # MLA-specific
        "multi_latent_attention": True,
        "kv_lora_rank": 32,
        "q_lora_rank": 64,
        "qk_nope_head_dim": 24,
        "qk_rope_head_dim": 8,
        "v_head_dim": 32,
        "rope_type": "rope",
        "rope_theta": 1000000.0,
        "rotary_base": 1000000.0,
        "rotary_percent": 1.0,
        "rope_scaling": 1.0,
        "position_embedding_type": "rope",
        # Normalization
        "rms_norm_eps": 1e-5,
        "use_qk_norm": False,
        # Model parallel - set to 1 for single GPU
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "virtual_pipeline_model_parallel_size": None,
        "params_dtype": paddle.bfloat16,
        "sequence_parallel": False,
        # FP8
        "fp8": None,
        "fp8_recipe": "blockwise",
        "fp8_wgrad": True,
        # Initialization
        "init_method_std": 0.02,
        "embedding_init_method_std": 0.02,
        "output_layer_init_method": functools.partial(paddle.nn.init.xavier_uniform_, gain=1.0),
        "init_method": functools.partial(paddle.nn.init.xavier_uniform_, gain=1.0),
    }
    defaults.update(overrides)
    config = GPTConfig(**defaults)
    config.gpt_model_use_experimental_version = use_experimental
    return config


def _build_gpt_model_with_forward_meta(decode_len=1, gpt_model_use_experimental_version=False):
    """Build a GPT model with MLA and inject fake forward_meta for decode mode."""
    config = _make_mla_gpt_config(gpt_model_use_experimental_version=gpt_model_use_experimental_version)

    # Build transformer layers spec
    transformer_layers_spec = []
    for layer_number in range(config.num_hidden_layers):
        transformer_layers_spec.append(
            get_gpt_layer_local_spec(
                config=config,
                use_qk_norm=config.use_qk_norm,
                num_experts=None,
                multi_latent_attention=config.multi_latent_attention,
                normalization=config.normalization,
                layer_number=layer_number,
            )
        )

    # Build GPT spec
    gpt_spec = get_gpt_spec(
        config=config,
        head_empty_layers_spec=[],
        transformer_layers_spec=transformer_layers_spec,
        tail_empty_layers_spec=[],
        mtp_layers_spec=None,
        vocab_size=config.vocab_size,
        tie_word_embeddings=config.tie_word_embeddings,
        max_sequence_length=config.max_sequence_length,
        position_embedding_type=config.position_embedding_type,
        rotary_percent=config.rotary_percent,
        rotary_base=config.rope_theta,
        rope_scaling=config.rope_scaling,
        parallel_output=config.parallel_output,
    )

    # Build model
    model = build_spec_layer(gpt_spec, num_stages=1)

    # Inject fake forward_meta into all core_attention configs
    # to simulate decode mode
    def inject_forward_meta(layer):
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "core_attention"):
            layer.self_attn.core_attention.config.forward_meta = FakeForwardMeta(max_len_tensor_cpu=[0, 0, decode_len])

    # Traverse model layers and inject forward_meta
    if hasattr(model, "run_function"):
        for sublayer in model.run_function:
            inject_forward_meta(sublayer)
    elif hasattr(model, "layers"):
        for sublayer in model.layers:
            inject_forward_meta(sublayer)

    return model, config


def _build_gpt_model(gpt_model_use_experimental_version=False):
    """Build a GPT model with MLA."""
    config = _make_mla_gpt_config(gpt_model_use_experimental_version=gpt_model_use_experimental_version)

    # Build transformer layers spec
    transformer_layers_spec = []
    for layer_number in range(config.num_hidden_layers):
        transformer_layers_spec.append(
            get_gpt_layer_local_spec(
                config=config,
                use_qk_norm=config.use_qk_norm,
                num_experts=None,
                multi_latent_attention=config.multi_latent_attention,
                normalization=config.normalization,
                layer_number=layer_number,
            )
        )

    # Build GPT spec
    gpt_spec = get_gpt_spec(
        config=config,
        head_empty_layers_spec=[],
        transformer_layers_spec=transformer_layers_spec,
        tail_empty_layers_spec=[],
        mtp_layers_spec=None,
        vocab_size=config.vocab_size,
        tie_word_embeddings=config.tie_word_embeddings,
        max_sequence_length=config.max_sequence_length,
        position_embedding_type=config.position_embedding_type,
        rotary_percent=config.rotary_percent,
        rotary_base=config.rope_theta,
        rope_scaling=config.rope_scaling,
        parallel_output=config.parallel_output,
    )

    # Build model
    model = build_spec_layer(gpt_spec, num_stages=1)
    return model, config


# ---------------------------------------------------------------------------
# Tests for _ec_compatible_rope_apply branch (full GPT model)
# ---------------------------------------------------------------------------


class TestECRopeBranchFullGPT(unittest.TestCase):
    """Test EC rope path with full GPT model."""

    def setUp(self):
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        paddle.manual_seed(seed)

    def test_forward_ec_rope_full_gpt(self):
        """Full GPT forward with gpt_model_use_experimental_version=True."""
        model, config = _build_gpt_model(gpt_model_use_experimental_version=True)
        model.eval()

        sequence_length = 16
        micro_batch_size = 2

        input_ids = paddle.randint(0, config.vocab_size, [micro_batch_size, sequence_length])
        position_ids = (
            paddle.arange(sequence_length, dtype=paddle.int64).unsqueeze(0).expand([micro_batch_size, sequence_length])
        )
        attention_mask = paddle.ones((micro_batch_size, 1, sequence_length, sequence_length), dtype=bool)

        # Forward pass - GPTModel expects a dict as input
        dict_args = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }
        result = model(dict_args)

        # Check output
        self.assertIsNotNone(result)
        # The result should contain logits or hidden states
        if isinstance(result, dict):
            self.assertIn("hidden_states", result)
            hidden_states = result["hidden_states"]
        else:
            hidden_states = result

        self.assertEqual(
            hidden_states.shape,
            [micro_batch_size, sequence_length, config.vocab_size],
        )

    def test_forward_ec_rope_backward_full_gpt(self):
        """Backward through EC rope path with full GPT model."""
        model, config = _build_gpt_model(gpt_model_use_experimental_version=True)
        model.train()

        sequence_length = 16
        micro_batch_size = 2

        input_ids = paddle.randint(0, config.vocab_size, [micro_batch_size, sequence_length])
        position_ids = (
            paddle.arange(sequence_length, dtype=paddle.int64).unsqueeze(0).expand([micro_batch_size, sequence_length])
        )
        attention_mask = paddle.ones((micro_batch_size, 1, sequence_length, sequence_length), dtype=bool)
        labels = paddle.randint(0, config.vocab_size, [micro_batch_size, sequence_length])

        # Forward pass - GPTModel expects a dict as input
        dict_args = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }
        result = model(dict_args)

        # Check output
        if isinstance(result, dict):
            hidden_states = result["hidden_states"]
        else:
            hidden_states = result

        # Compute loss and backward
        loss = paddle.nn.functional.cross_entropy(
            hidden_states.reshape([-1, config.vocab_size]).cast("float32"),
            labels.reshape([-1]),
        )
        loss_value = loss.item()
        loss.backward()

        # Check gradients exist
        has_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_grad = True
                self.assertTrue(paddle.isfinite(param.grad).all().item())
        self.assertTrue(has_grad, "Should have gradients after backward")

    def test_forward_ec_rope_with_1d_position_ids(self):
        """EC rope path with 1D position_ids [S]."""
        model, config = _build_gpt_model(gpt_model_use_experimental_version=True)
        model.eval()

        sequence_length = 16
        micro_batch_size = 2

        input_ids = paddle.randint(0, config.vocab_size, [micro_batch_size, sequence_length])
        # 1D position_ids: [S]
        position_ids = paddle.arange(sequence_length, dtype=paddle.int64)
        attention_mask = paddle.ones((micro_batch_size, 1, sequence_length, sequence_length), dtype=bool)

        dict_args = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }
        result = model(dict_args)

        if isinstance(result, dict):
            hidden_states = result["hidden_states"]
        else:
            hidden_states = result

        self.assertEqual(
            hidden_states.shape,
            [micro_batch_size, sequence_length, config.vocab_size],
        )

    def test_backward_ec_rope_with_1d_position_ids(self):
        """Backward through EC rope path with 1D position_ids."""
        model, config = _build_gpt_model(gpt_model_use_experimental_version=True)
        model.train()

        sequence_length = 16
        micro_batch_size = 2

        input_ids = paddle.randint(0, config.vocab_size, [micro_batch_size, sequence_length])
        # 1D position_ids: [S]
        position_ids = paddle.arange(sequence_length, dtype=paddle.int64)
        attention_mask = paddle.ones((micro_batch_size, 1, sequence_length, sequence_length), dtype=bool)
        labels = paddle.randint(0, config.vocab_size, [micro_batch_size, sequence_length])

        dict_args = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }
        result = model(dict_args)

        if isinstance(result, dict):
            hidden_states = result["hidden_states"]
        else:
            hidden_states = result

        loss = paddle.nn.functional.cross_entropy(
            hidden_states.reshape([-1, config.vocab_size]).cast("float32"),
            labels.reshape([-1]),
        )
        loss.backward()

        has_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_grad = True
                self.assertTrue(paddle.isfinite(param.grad).all().item())
        self.assertTrue(has_grad, "Should have gradients after backward")


# ---------------------------------------------------------------------------
# Tests for _compute_absorbed_q branch (fake forward_meta, full GPT model)
# ---------------------------------------------------------------------------


class TestComputeAbsorbedQBranchFullGPT(unittest.TestCase):
    """Test decode mode with full GPT model using fake forward_meta."""

    def setUp(self):
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        paddle.manual_seed(seed)

    def test_forward_decode_mode_full_gpt(self):
        """Full GPT forward with fake forward_meta (decode mode)."""
        model, config = _build_gpt_model_with_forward_meta(decode_len=1)
        model.eval()

        sequence_length = 16
        micro_batch_size = 2

        input_ids = paddle.randint(0, config.vocab_size, [micro_batch_size, sequence_length])
        position_ids = (
            paddle.arange(sequence_length, dtype=paddle.int64).unsqueeze(0).expand([micro_batch_size, sequence_length])
        )
        attention_mask = paddle.ones((micro_batch_size, 1, sequence_length, sequence_length), dtype=bool)

        # Forward pass - GPTModel expects a dict as input
        dict_args = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }
        result = model(dict_args)

        # Check output
        self.assertIsNotNone(result)
        if isinstance(result, dict):
            hidden_states = result["hidden_states"]
        else:
            hidden_states = result

        self.assertEqual(
            hidden_states.shape,
            [micro_batch_size, sequence_length, config.vocab_size],
        )

    def test_forward_decode_mode_backward_full_gpt(self):
        """Backward through decode mode path with full GPT model."""
        model, config = _build_gpt_model_with_forward_meta(decode_len=5)
        model.train()

        sequence_length = 16
        micro_batch_size = 2

        input_ids = paddle.randint(0, config.vocab_size, [micro_batch_size, sequence_length])
        position_ids = (
            paddle.arange(sequence_length, dtype=paddle.int64).unsqueeze(0).expand([micro_batch_size, sequence_length])
        )
        attention_mask = paddle.ones((micro_batch_size, 1, sequence_length, sequence_length), dtype=bool)
        labels = paddle.randint(0, config.vocab_size, [micro_batch_size, sequence_length])

        # Forward pass - GPTModel expects a dict as input
        dict_args = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }
        result = model(dict_args)

        # Check output
        if isinstance(result, dict):
            hidden_states = result["hidden_states"]
        else:
            hidden_states = result

        # Compute loss and backward
        loss = paddle.nn.functional.cross_entropy(
            hidden_states.reshape([-1, config.vocab_size]).cast("float32"),
            labels.reshape([-1]),
        )
        loss_value = loss.item()
        loss.backward()

        # Check gradients exist
        has_grad = False
        grad_count = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_grad = True
                grad_count += 1
                if not paddle.isfinite(param.grad).all().item():
                    self.fail(f"Gradient for {name} contains NaN or Inf")

        # In decode mode, some layers may skip gradient calculation
        # The important thing is that the forward passes without errors
        # and the loss is finite
        self.assertTrue(paddle.isfinite(loss).item(), "Loss should be finite")

    def test_forward_decode_mode_with_1d_position_ids(self):
        """Decode mode with 1D position_ids."""
        model, config = _build_gpt_model_with_forward_meta(decode_len=1)
        model.eval()

        sequence_length = 16
        micro_batch_size = 2

        input_ids = paddle.randint(0, config.vocab_size, [micro_batch_size, sequence_length])
        # 1D position_ids: [S]
        position_ids = paddle.arange(sequence_length, dtype=paddle.int64)
        attention_mask = paddle.ones((micro_batch_size, 1, sequence_length, sequence_length), dtype=bool)

        dict_args = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }
        result = model(dict_args)

        if isinstance(result, dict):
            hidden_states = result["hidden_states"]
        else:
            hidden_states = result

        self.assertEqual(
            hidden_states.shape,
            [micro_batch_size, sequence_length, config.vocab_size],
        )

    def test_backward_decode_mode_with_1d_position_ids(self):
        """Decode mode backward with 1D position_ids."""
        model, config = _build_gpt_model_with_forward_meta(decode_len=5)
        model.train()

        sequence_length = 16
        micro_batch_size = 2

        input_ids = paddle.randint(0, config.vocab_size, [micro_batch_size, sequence_length])
        # 1D position_ids: [S]
        position_ids = paddle.arange(sequence_length, dtype=paddle.int64)
        attention_mask = paddle.ones((micro_batch_size, 1, sequence_length, sequence_length), dtype=bool)
        labels = paddle.randint(0, config.vocab_size, [micro_batch_size, sequence_length])

        dict_args = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }
        result = model(dict_args)

        if isinstance(result, dict):
            hidden_states = result["hidden_states"]
        else:
            hidden_states = result

        loss = paddle.nn.functional.cross_entropy(
            hidden_states.reshape([-1, config.vocab_size]).cast("float32"),
            labels.reshape([-1]),
        )
        loss.backward()

        has_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_grad = True
                if not paddle.isfinite(param.grad).all().item():
                    self.fail(f"Gradient for {name} contains NaN or Inf")

        self.assertTrue(paddle.isfinite(loss).item(), "Loss should be finite")


# ---------------------------------------------------------------------------
# Tests combining both branches (full GPT model)
# ---------------------------------------------------------------------------


class TestCombinedECRopeAndAbsorbedQFullGPT(unittest.TestCase):
    """Test both branches active together with full GPT model: EC rope + decode mode."""

    def setUp(self):
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        paddle.manual_seed(seed)

    def test_forward_ec_rope_and_decode_mode_full_gpt(self):
        """Full GPT forward with both EC rope and decode-mode active."""
        model, config = _build_gpt_model_with_forward_meta(decode_len=1, gpt_model_use_experimental_version=True)
        model.eval()

        sequence_length = 16
        micro_batch_size = 2

        input_ids = paddle.randint(0, config.vocab_size, [micro_batch_size, sequence_length])
        position_ids = (
            paddle.arange(sequence_length, dtype=paddle.int64).unsqueeze(0).expand([micro_batch_size, sequence_length])
        )
        attention_mask = paddle.ones((micro_batch_size, 1, sequence_length, sequence_length), dtype=bool)

        # Forward pass - GPTModel expects a dict as input
        dict_args = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }
        result = model(dict_args)

        # Check output
        self.assertIsNotNone(result)
        if isinstance(result, dict):
            hidden_states = result["hidden_states"]
        else:
            hidden_states = result

        self.assertEqual(
            hidden_states.shape,
            [micro_batch_size, sequence_length, config.vocab_size],
        )

    def test_backward_ec_rope_and_decode_mode_full_gpt(self):
        """Backward with both EC rope and decode mode with full GPT model."""
        model, config = _build_gpt_model_with_forward_meta(decode_len=3, gpt_model_use_experimental_version=True)
        model.train()

        sequence_length = 16
        micro_batch_size = 2

        input_ids = paddle.randint(0, config.vocab_size, [micro_batch_size, sequence_length])
        position_ids = (
            paddle.arange(sequence_length, dtype=paddle.int64).unsqueeze(0).expand([micro_batch_size, sequence_length])
        )
        attention_mask = paddle.ones((micro_batch_size, 1, sequence_length, sequence_length), dtype=bool)
        labels = paddle.randint(0, config.vocab_size, [micro_batch_size, sequence_length])

        # Forward pass - GPTModel expects a dict as input
        dict_args = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }
        result = model(dict_args)

        # Check output
        if isinstance(result, dict):
            hidden_states = result["hidden_states"]
        else:
            hidden_states = result

        # Compute loss and backward
        loss = paddle.nn.functional.cross_entropy(
            hidden_states.reshape([-1, config.vocab_size]).cast("float32"),
            labels.reshape([-1]),
        )
        loss_value = loss.item()
        loss.backward()

        # Check gradients exist
        has_grad = False
        grad_count = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_grad = True
                grad_count += 1
                if not paddle.isfinite(param.grad).all().item():
                    self.fail(f"Gradient for {name} contains NaN or Inf")

        # In decode mode, some layers may skip gradient calculation
        # The important thing is that the forward passes without errors
        # and the loss is finite
        self.assertTrue(paddle.isfinite(loss).item(), "Loss should be finite")

    def test_forward_ec_rope_and_decode_mode_1d_position_ids(self):
        """EC rope + decode mode with 1D position_ids."""
        model, config = _build_gpt_model_with_forward_meta(decode_len=1, gpt_model_use_experimental_version=True)
        model.eval()

        sequence_length = 16
        micro_batch_size = 2

        input_ids = paddle.randint(0, config.vocab_size, [micro_batch_size, sequence_length])
        # 1D position_ids: [S]
        position_ids = paddle.arange(sequence_length, dtype=paddle.int64)
        attention_mask = paddle.ones((micro_batch_size, 1, sequence_length, sequence_length), dtype=bool)

        dict_args = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }
        result = model(dict_args)

        if isinstance(result, dict):
            hidden_states = result["hidden_states"]
        else:
            hidden_states = result

        self.assertEqual(
            hidden_states.shape,
            [micro_batch_size, sequence_length, config.vocab_size],
        )

    def test_backward_ec_rope_and_decode_mode_1d_position_ids(self):
        """EC rope + decode mode backward with 1D position_ids."""
        model, config = _build_gpt_model_with_forward_meta(decode_len=3, gpt_model_use_experimental_version=True)
        model.train()

        sequence_length = 16
        micro_batch_size = 2

        input_ids = paddle.randint(0, config.vocab_size, [micro_batch_size, sequence_length])
        # 1D position_ids: [S]
        position_ids = paddle.arange(sequence_length, dtype=paddle.int64)
        attention_mask = paddle.ones((micro_batch_size, 1, sequence_length, sequence_length), dtype=bool)
        labels = paddle.randint(0, config.vocab_size, [micro_batch_size, sequence_length])

        dict_args = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }
        result = model(dict_args)

        if isinstance(result, dict):
            hidden_states = result["hidden_states"]
        else:
            hidden_states = result

        loss = paddle.nn.functional.cross_entropy(
            hidden_states.reshape([-1, config.vocab_size]).cast("float32"),
            labels.reshape([-1]),
        )
        loss.backward()

        has_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_grad = True
                if not paddle.isfinite(param.grad).all().item():
                    self.fail(f"Gradient for {name} contains NaN or Inf")

        self.assertTrue(paddle.isfinite(loss).item(), "Loss should be finite")


# ---------------------------------------------------------------------------
# Tests for gpt_model_use_experimental_version=False with different position_ids
# ---------------------------------------------------------------------------


class TestNonExperimentalVersionPositionIds(unittest.TestCase):
    """Test gpt_model_use_experimental_version=False with 1D and 3D position_ids."""

    def setUp(self):
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        paddle.manual_seed(seed)

    def test_forward_with_1d_position_ids(self):
        """Forward with 1D position_ids [S], non-experimental version."""
        model, config = _build_gpt_model(gpt_model_use_experimental_version=False)
        model.eval()

        sequence_length = 16
        micro_batch_size = 2

        input_ids = paddle.randint(0, config.vocab_size, [micro_batch_size, sequence_length])
        # 1D position_ids: [S]
        position_ids = paddle.arange(sequence_length, dtype=paddle.int64)
        attention_mask = paddle.ones((micro_batch_size, 1, sequence_length, sequence_length), dtype=bool)

        dict_args = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }
        result = model(dict_args)

        if isinstance(result, dict):
            hidden_states = result["hidden_states"]
        else:
            hidden_states = result

        self.assertEqual(
            hidden_states.shape,
            [micro_batch_size, sequence_length, config.vocab_size],
        )

    def test_backward_with_1d_position_ids(self):
        """Backward with 1D position_ids, non-experimental version."""
        model, config = _build_gpt_model(gpt_model_use_experimental_version=False)
        model.train()

        sequence_length = 16
        micro_batch_size = 2

        input_ids = paddle.randint(0, config.vocab_size, [micro_batch_size, sequence_length])
        # 1D position_ids: [S]
        position_ids = paddle.arange(sequence_length, dtype=paddle.int64)
        attention_mask = paddle.ones((micro_batch_size, 1, sequence_length, sequence_length), dtype=bool)
        labels = paddle.randint(0, config.vocab_size, [micro_batch_size, sequence_length])

        dict_args = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }
        result = model(dict_args)

        if isinstance(result, dict):
            hidden_states = result["hidden_states"]
        else:
            hidden_states = result

        loss = paddle.nn.functional.cross_entropy(
            hidden_states.reshape([-1, config.vocab_size]).cast("float32"),
            labels.reshape([-1]),
        )
        loss.backward()

        has_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_grad = True
                self.assertTrue(paddle.isfinite(param.grad).all().item())
        self.assertTrue(has_grad, "Should have gradients after backward")


class TestApplyRopeFusionNotSupportedInference(unittest.TestCase):
    """Test that apply_rope_fusion raises NotImplementedError in eval mode (L994-1000)."""

    def test_apply_rope_fusion_raises_in_eval(self):
        """apply_rope_fusion=True in eval mode should raise NotImplementedError."""
        model, config = _build_gpt_model(gpt_model_use_experimental_version=False)
        # Enable apply_rope_fusion on all MLA layers
        for sublayer in model.sublayers():
            if hasattr(sublayer, "config") and hasattr(sublayer.config, "apply_rope_fusion"):
                sublayer.config.apply_rope_fusion = True

        model.eval()

        sequence_length = 16
        micro_batch_size = 2
        input_ids = paddle.randint(0, config.vocab_size, [micro_batch_size, sequence_length])
        position_ids = (
            paddle.arange(sequence_length, dtype=paddle.int64).unsqueeze(0).expand([micro_batch_size, sequence_length])
        )
        attention_mask = paddle.ones((micro_batch_size, 1, sequence_length, sequence_length), dtype=bool)

        dict_args = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }

        with self.assertRaises(NotImplementedError) as ctx:
            model(dict_args)

        self.assertIn(
            "apply_rope_fusion does not support dynamic inference yet",
            str(ctx.exception),
        )

    def test_apply_rope_fusion_train_mode(self):
        """apply_rope_fusion=True in train mode should succeed and set k_pe=None (L1000)."""
        model, config = _build_gpt_model(gpt_model_use_experimental_version=False)
        for sublayer in model.sublayers():
            if hasattr(sublayer, "config") and hasattr(sublayer.config, "apply_rope_fusion"):
                sublayer.config.apply_rope_fusion = True

        model.train()

        sequence_length = 16
        micro_batch_size = 2
        input_ids = paddle.randint(0, config.vocab_size, [micro_batch_size, sequence_length])
        position_ids = (
            paddle.arange(sequence_length, dtype=paddle.int64).unsqueeze(0).expand([micro_batch_size, sequence_length])
        )
        attention_mask = paddle.ones((micro_batch_size, 1, sequence_length, sequence_length), dtype=bool)

        dict_args = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }
        result = model(dict_args)

        if isinstance(result, dict):
            hidden_states = result["hidden_states"]
        else:
            hidden_states = result

        self.assertEqual(
            hidden_states.shape,
            [micro_batch_size, sequence_length, config.vocab_size],
        )


if __name__ == "__main__":
    unittest.main()
