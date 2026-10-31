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
from __future__ import annotations

import itertools
import tempfile
import unittest
from unittest.mock import patch

import paddle
from parameterized import parameterized

from paddleformers.transformers import (
    StableLmConfig,
    StableLmForCausalLM,
    StableLmForCausalLMPipe,
    StableLmModel,
    StableLmTokenizer,
)
from paddleformers.transformers.stablelm.modeling import (
    StableLmDecoderLayer,
    StableLmLayerNormPerHead,
    StableLmRotaryEmbedding,
)
from tests.testing_utils import (
    gpu_device_initializer,
    require_gpu,
    require_package,
    slow,
)
from tests.transformers.test_configuration_common import ConfigTester
from tests.transformers.test_generation_utils import GenerationTesterMixin
from tests.transformers.test_modeling_common import (
    ModelTesterMixin,
    ids_tensor,
    random_attention_mask,
)


class StableLmLayerNormPerHeadTest(unittest.TestCase):
    @parameterized.expand([(1,), (2,), (4,)])
    def test_per_head_normalization(self, num_heads):
        layer = StableLmLayerNormPerHead(8, num_heads)
        self.assertIsNone(layer.bias)
        hidden_states = paddle.randn([2, num_heads, 3, 8])
        weights = paddle.arange(1, num_heads + 1, dtype="float32").unsqueeze(-1).tile([1, 8])
        layer.weight.set_value(weights.flatten())
        centered = hidden_states - hidden_states.mean(axis=-1, keepdim=True)
        expected = centered * paddle.rsqrt(centered.square().mean(axis=-1, keepdim=True) + 1e-5)
        expected = expected * weights.reshape([1, num_heads, 1, 8])
        paddle.testing.assert_close(layer(hidden_states), expected, rtol=1e-5, atol=1e-6)


class StableLmModelTester:
    def __init__(
        self,
        parent,
        vocab_size=99,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        is_training=True,
        use_cache=False,
        bos_token_id=1,
        eos_token_id=98,
        pad_token_id=0,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        batch_size: int = 2,
        seq_length: int = 7,
        use_input_mask: bool = True,
        return_dict: bool = False,
    ):
        self.parent = parent
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.is_training = is_training
        self.use_cache = use_cache
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.use_input_mask = use_input_mask
        self.return_dict = return_dict

    def get_config(self, **kwargs):
        values = dict(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            intermediate_size=self.intermediate_size,
            layer_norm_eps=self.layer_norm_eps,
            hidden_dropout=self.hidden_dropout,
            attention_dropout=self.attention_dropout,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            initializer_range=self.initializer_range,
            use_cache=self.use_cache,
            return_dict=self.return_dict,
            _attn_implementation="eager",
        )
        values.update(kwargs)
        return StableLmConfig(**values)

    def prepare_config_and_inputs(self):
        config = self.get_config()
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size, dtype=paddle.int64)
        attention_mask = random_attention_mask([self.batch_size, self.seq_length]) if self.use_input_mask else None
        return config, input_ids, attention_mask

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, attention_mask = self.prepare_config_and_inputs()
        return config, {"input_ids": input_ids, "attention_mask": attention_mask}

    def create_and_check_model(self, config, input_ids, attention_mask):
        model = StableLmModel(config)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, return_dict=True)
        self.parent.assertEqual(result.last_hidden_state.shape, [self.batch_size, self.seq_length, self.hidden_size])

    def create_and_check_lm_head_model(self, config, input_ids, attention_mask):
        model = StableLmForCausalLM(config)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, return_dict=True)
        self.parent.assertEqual(result.logits.shape, [self.batch_size, self.seq_length, self.vocab_size])


class StableLmModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    base_model_class = StableLmModel
    return_dict = False
    use_labels = False
    use_test_model_name_list = False
    has_attentions = False
    all_model_classes = (StableLmModel, StableLmForCausalLM)
    all_generative_model_classes = {StableLmForCausalLM: (StableLmModel, "stablelm")}

    @gpu_device_initializer(log_prefix="StableLmModelTest")
    def setUp(self):
        super().setUp()
        self.model_tester = StableLmModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=StableLmConfig,
            vocab_size=99,
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
        self.config = self.model_tester.get_config()

    def test_qk_layernorm_tp_checkpoint_mapping(self):
        self.config.qk_layernorm = True
        self.config.tensor_model_parallel_size = 2
        statements = StableLmForCausalLM._gen_aoa_config(self.config)["aoa_statements"]
        q_norm = [line for line in statements if "q_layernorm" in line]
        self.assertEqual(len(q_norm), 1)
        self.assertIn(f"norms.{self.config.num_attention_heads - 1}.weight", q_norm[0])
        self.assertIn("q_layernorm.weight, axis=0", q_norm[0])
        self.assertFalse(any("layernorm.norms." in line and ".bias" in line for line in statements))

    @parameterized.expand([("linear",), ("dynamic",)])
    def test_legacy_rope_scaling(self, rope_type):
        scaling = {"type": rope_type, "factor": 2.0}
        config = StableLmConfig(rope_scaling=scaling)
        self.assertEqual(config.rope_parameters["rope_type"], rope_type)
        self.assertEqual(config.rope_parameters["factor"], 2.0)
        self.assertEqual(scaling, {"type": rope_type, "factor": 2.0})
        with tempfile.TemporaryDirectory() as tmpdir:
            config.save_pretrained(tmpdir)
            loaded = StableLmConfig.from_pretrained(tmpdir)
        self.assertEqual(loaded.rope_parameters, config.rope_parameters)

    def test_rope_parameters_take_precedence(self):
        config = StableLmConfig(
            rope_parameters={"rope_type": "linear", "factor": 3.0},
            rope_scaling={"type": "linear", "factor": 2.0},
        )
        self.assertEqual(config.rope_parameters["factor"], 3.0)

    @parameterized.expand([(-10000.0,), (float("nan"),), (float("inf"),), ("invalid",)])
    def test_invalid_effective_rope_theta(self, rope_theta):
        with self.assertRaisesRegex(ValueError, "rope_theta"):
            self.model_tester.get_config(rope_parameters={"rope_theta": rope_theta})

    @parameterized.expand([(False,), (True,)])
    def test_residual_formula(self, parallel):
        config = StableLmConfig(
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            use_parallel_residual=parallel,
            qk_layernorm=True,
        )
        layer = StableLmDecoderLayer(config, 0)
        layer.eval()
        hidden_states = paddle.randn([2, 4, 64])
        position_ids = paddle.arange(4).unsqueeze(0)
        with paddle.no_grad():
            position_embeddings = StableLmRotaryEmbedding(config)(hidden_states, position_ids)
            normalized = layer.input_layernorm(hidden_states)
            attention_output, _ = layer.self_attn(normalized, position_embeddings=position_embeddings)
            if parallel:
                expected = hidden_states + attention_output + layer.mlp(normalized)
            else:
                residual = hidden_states + attention_output
                expected = residual + layer.mlp(layer.post_attention_layernorm(residual))
            actual = layer(hidden_states, position_embeddings=position_embeddings)
        paddle.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_loss_mask(self):
        config = StableLmConfig(
            vocab_size=32,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
        model = StableLmForCausalLM(config)
        model.eval()
        input_ids = paddle.to_tensor([[1, 2, 3, 4]], dtype="int64")
        labels = paddle.to_tensor([[2, 3, 4, 5]], dtype="int64")
        loss_mask = paddle.to_tensor([[1.0, 0.0, 0.5, 0.0]])
        with paddle.no_grad():
            outputs = model(input_ids, labels=labels, loss_mask=loss_mask, return_dict=True)
            token_loss = paddle.nn.functional.cross_entropy(outputs.logits, labels, reduction="none")
            expected = (token_loss * loss_mask).sum() / loss_mask.sum()
            changed_labels = paddle.to_tensor([[2, 8, 4, 9]], dtype="int64")
            changed = model(input_ids, labels=changed_labels, loss_mask=loss_mask, return_dict=True)
        paddle.testing.assert_close(outputs.loss, expected, rtol=1e-5, atol=1e-6)
        paddle.testing.assert_close(changed.loss, outputs.loss, rtol=0.0, atol=0.0)

    def test_config(self):
        config = StableLmConfig()
        self.assertEqual(config.model_type, "stablelm")
        self.assertEqual(config.hidden_size, 2560)
        self.assertEqual(config.num_hidden_layers, 32)
        self.assertEqual(config.num_attention_heads, 32)
        self.assertEqual(config.vocab_size, 50304)
        self.assertIsInstance(config.rope_parameters, dict)
        self.assertEqual(config.rope_parameters.get("partial_rotary_factor"), 0.25)

    def test_model_creation(self):
        model = StableLmModel(self.config)
        self.assertIsInstance(model, StableLmModel)
        self.assertEqual(len(model.layers), self.config.num_hidden_layers)

    def test_model(self):
        self.model_tester.create_and_check_model(*self.model_tester.prepare_config_and_inputs())

    def test_stablelm_lm_head_model(self):
        self.model_tester.create_and_check_lm_head_model(*self.model_tester.prepare_config_and_inputs())

    @parameterized.expand(list(itertools.product([False, True], repeat=3)))
    def test_optional_branches_roundtrip(self, use_qkv_bias, qk_layernorm, use_parallel_residual):
        config = StableLmConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            use_qkv_bias=use_qkv_bias,
            qk_layernorm=qk_layernorm,
            use_parallel_residual=use_parallel_residual,
            hidden_dropout=0.0,
            attention_dropout=0.0,
        )
        model = StableLmForCausalLM(config)
        model.eval()
        with paddle.no_grad():
            for name, param in model.named_parameters():
                if "q_layernorm.weight" in name or "k_layernorm.weight" in name or name.endswith(".bias"):
                    param.set_value(paddle.randn(param.shape) * 0.1)
        input_ids = paddle.to_tensor([[1, 2, 3, 4]], dtype="int64")
        with paddle.no_grad():
            expected = model(input_ids, return_dict=True).logits
        self.assertTrue(paddle.isfinite(expected).all().item())
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            loaded = StableLmForCausalLM.from_pretrained(
                tmpdir, dtype="float32", load_checkpoint_format="flex_checkpoint"
            )
            loaded.eval()
            self.assertEqual(set(model.state_dict()), set(loaded.state_dict()))
            for name, value in model.state_dict().items():
                paddle.testing.assert_close(value, loaded.state_dict()[name], rtol=0.0, atol=0.0)
            with paddle.no_grad():
                actual = loaded(input_ids, return_dict=True).logits
            paddle.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_tiny_model_init_and_delete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = StableLmConfig(
                hidden_size=128,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=4,
                intermediate_size=512,
                vocab_size=1024,
            )
            model = StableLmForCausalLM(config)
            model.save_pretrained(tmpdir)

            loaded = StableLmForCausalLM.from_pretrained(
                tmpdir,
                dtype="float32",
                load_checkpoint_format="flex_checkpoint",
            )
            loaded.eval()

            input_ids = paddle.randint(0, 1024, [1, 4], dtype="int64")
            with paddle.no_grad():
                outputs = loaded(input_ids, return_dict=True)
            self.assertEqual(outputs.logits.shape, [1, 4, 1024])

    @parameterized.expand(
        [
            ("hidden_size", 63),
            ("num_key_value_heads", 3),
            ("partial_rotary_factor", 0.1),
            ("attention_dropout", 1.0),
            ("num_attention_heads", 0),
            ("layer_norm_eps", 0),
            ("head_dim", 32),
        ]
    )
    def test_invalid_config(self, key, value):
        with self.assertRaises(ValueError):
            self.model_tester.get_config(qk_layernorm=True, **{key: value})

    def test_generation_config_and_defaults(self):
        model = StableLmForCausalLM(
            self.model_tester.get_config(
                qk_layernorm=True, use_cache=True, output_hidden_states=True, return_dict=True
            )
        )
        model.eval()
        self.assertTrue(model.can_generate())
        self.assertIsNotNone(model.generation_config)
        with paddle.no_grad():
            outputs = model(paddle.to_tensor([[1, 2, 3]]))
        self.assertIsNotNone(outputs.past_key_values)
        self.assertEqual(len(outputs.hidden_states), 3)
        self.assertTrue(issubclass(StableLmForCausalLMPipe, paddle.distributed.fleet.meta_parallel.PipelineLayer))

    @parameterized.expand([(None,), (16,)])
    def test_head_dim_consistency(self, head_dim):
        config = self.model_tester.get_config(head_dim=head_dim)
        self.assertEqual(config.head_dim, config.hidden_size // config.num_attention_heads)
        with tempfile.TemporaryDirectory() as tmpdir:
            config.save_pretrained(tmpdir)
            config = StableLmConfig.from_pretrained(tmpdir)
        model = StableLmModel(config).eval()
        attention = model.layers[0].self_attn
        self.assertEqual(attention.head_dim, config.head_dim)
        self.assertEqual(model.rotary_emb.inv_freq.shape[0] * 2, attention.rotary_ndims)
        with paddle.no_grad():
            outputs = model(paddle.to_tensor([[1, 2, 3]]), return_dict=True)
        self.assertTrue(paddle.isfinite(outputs.last_hidden_state).all().item())

    def test_auto_data_sharding_config(self):
        from paddleformers.trainer.trainer_utils import ShardingOption

        model = StableLmForCausalLM(self.model_tester.get_config())
        config = model._generate_auto_dist_config(
            {
                "data_sharding_parallel": True,
                "sharding": [ShardingOption.FULL_SHARD],
                "sharding_mesh_dim": "dp",
            }
        )
        self.assertEqual(config["dp_config"], {"sharding_level": 3, "sharding_mesh_dim": "dp"})
        self.assertIsNone(config["mp_config"])
        self.assertIsNone(config["pp_config"])

    def test_output_attentions_config_and_override(self):
        input_ids = paddle.to_tensor([[1, 2, 3]])
        for model_class in self.all_model_classes:
            model = model_class(self.model_tester.get_config(output_attentions=True)).eval()
            with self.assertRaisesRegex(ValueError, "attention weights"):
                model(input_ids, return_dict=True)
            with paddle.no_grad():
                outputs = model(input_ids, output_attentions=False, return_dict=True)
            self.assertIsNone(outputs.attentions)
            with self.assertRaisesRegex(ValueError, "attention weights"):
                model(input_ids, output_attentions=True, return_dict=True)

    @parameterized.expand([(None,), ([],)])
    def test_auto_entrypoint_and_small_generation(self, architectures):
        import re

        from paddleformers.cli.utils.llm_utils import get_lora_target_modules
        from paddleformers.transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_config(
            self.model_tester.get_config(qk_layernorm=True, use_cache=True, architectures=architectures)
        )
        self.assertIsInstance(model, StableLmForCausalLM)
        targets = get_lora_target_modules(model)
        names = [name for name, _ in model.named_sublayers()]
        for pattern in targets:
            self.assertTrue(any(re.fullmatch(pattern, name) for name in names), pattern)
        model.eval()
        with paddle.no_grad():
            tokens = model.generate(
                paddle.to_tensor([[1, 2, 3]]), max_new_tokens=2, do_sample=False, eos_token_id=None, pad_token_id=0
            )[0]
        self.assertEqual(list(tokens.shape), [1, 2])

    @parameterized.expand([(False,), (True,)])
    def test_recompute_loss_gradients_and_reexecution(self, freeze_embedding):
        reference = StableLmForCausalLM(
            self.model_tester.get_config(qk_layernorm=True, hidden_dropout=0.2, attention_dropout=0.1)
        )
        config = self.model_tester.get_config(
            qk_layernorm=True,
            hidden_dropout=0.2,
            attention_dropout=0.1,
            recompute_granularity="full",
            recompute_method="uniform",
            recompute_num_layers=1,
        )
        actual = StableLmForCausalLM(config)
        actual.set_state_dict(reference.state_dict())
        for model in (reference, actual):
            model.train()
            model.model.embed_tokens.weight.stop_gradient = freeze_embedding
        x = paddle.to_tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
        paddle.seed(123)
        expected = reference(x, labels=x, use_cache=False, return_dict=True).loss
        expected.backward()
        layer = actual.model.layers[0]
        with patch.object(layer, "forward", wraps=layer.forward) as forward:
            paddle.seed(123)
            observed = actual(x, labels=x, use_cache=True, return_dict=True).loss
            observed.backward()
            self.assertGreaterEqual(forward.call_count, 2)
        paddle.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-6)
        for (name, param), (_, other) in zip(reference.named_parameters(), actual.named_parameters()):
            if not param.stop_gradient:
                self.assertIsNotNone(other.grad, name)
                self.assertTrue(paddle.isfinite(other.grad).all().item(), name)
                paddle.testing.assert_close(other.grad, param.grad, rtol=1e-5, atol=1e-6)
        before = actual.model.layers[0].self_attn.q_proj.weight.clone()
        paddle.optimizer.SGD(learning_rate=0.01, parameters=actual.parameters()).step()
        self.assertFalse(paddle.equal_all(before, actual.model.layers[0].self_attn.q_proj.weight).item())

    @parameterized.expand([("eager", "float32"), ("sdpa", "bfloat16")])
    def test_cache_multiple_steps_and_no_write(self, backend, dtype):
        model = StableLmForCausalLM(self.model_tester.get_config(qk_layernorm=True, _attn_implementation=backend)).to(
            dtype=dtype
        )
        model.eval()
        x = paddle.to_tensor([[1, 2, 3, 4, 5, 6]])
        with paddle.no_grad():
            expected = model(x, use_cache=False, return_dict=True).logits
            cache = model(x[:, :2], use_cache=True, return_dict=True).past_key_values
            cached = model(x[:, 2:4], past_key_values=cache, use_cache=True, return_dict=True)
            self.assertEqual(cache.get_seq_length(), 4)
            paddle.testing.assert_close(
                cached.logits.float(),
                expected[:, 2:4].float(),
                rtol=1e-3 if dtype == "bfloat16" else 1e-5,
                atol=1e-3 if dtype == "bfloat16" else 1e-6,
            )
            for end in (5, 6):
                cached = model(x[:, end - 1 : end], past_key_values=cache, use_cache=True, return_dict=True)
                self.assertEqual(cache.get_seq_length(), end)
                paddle.testing.assert_close(
                    cached.logits.float(),
                    expected[:, end - 1 : end].float(),
                    rtol=1e-3 if dtype == "bfloat16" else 1e-5,
                    atol=1e-3 if dtype == "bfloat16" else 1e-6,
                )
            keys = cache.layers[0].keys.clone()
            no_write = model(x[:, :1], past_key_values=cache, use_cache=False, return_dict=True)
            self.assertIsNone(no_write.past_key_values)
            self.assertEqual(cache.get_seq_length(), 6)
            paddle.testing.assert_close(cache.layers[0].keys.float(), keys.float(), rtol=0.0, atol=0.0)

    def test_inputs_embeds_padding_and_tied_checkpoint(self):
        model = StableLmForCausalLM(
            self.model_tester.get_config(qk_layernorm=True, tie_word_embeddings=True, pad_token_id=0)
        )
        model.eval()
        self.assertIs(model.model.embed_tokens.weight, model.lm_head.weight)
        self.assertEqual(model.model.embed_tokens.weight[0].abs().sum().item(), 0.0)
        x = paddle.to_tensor([[0, 0, 1, 2], [1, 2, 3, 4]])
        mask = paddle.to_tensor([[0, 0, 1, 1], [1, 1, 1, 1]])
        positions = paddle.clip(mask.cumsum(-1) - 1, min=0)
        with paddle.no_grad():
            expected = model(x, attention_mask=mask, position_ids=positions, return_dict=True).logits
            actual = model(
                inputs_embeds=model.get_input_embeddings()(x),
                attention_mask=mask,
                position_ids=positions,
                return_dict=True,
            ).logits
            single = model(x[:1, 2:], return_dict=True).logits
        paddle.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        paddle.testing.assert_close(expected[:1, 2:], single, rtol=1e-5, atol=1e-6)
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            loaded = StableLmForCausalLM.from_pretrained(
                tmpdir, dtype="float32", load_checkpoint_format="flex_checkpoint"
            )
            self.assertIs(loaded.model.embed_tokens.weight, loaded.lm_head.weight)
            loaded.eval()
            with paddle.no_grad():
                actual = loaded(x, attention_mask=mask, position_ids=positions, return_dict=True).logits
            paddle.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_dynamic_rope_growth_and_reset(self):
        config = self.model_tester.get_config(
            qk_layernorm=True, max_position_embeddings=8, rope_parameters={"rope_type": "dynamic", "factor": 2.0}
        )
        rope = StableLmRotaryEmbedding(config)
        original = rope.inv_freq.clone()
        x = paddle.zeros([1, 1, config.hidden_size])
        rope(x, paddle.to_tensor([[16]]))
        self.assertEqual(int(rope.max_seq_len_cached), 17)
        self.assertFalse(paddle.equal_all(rope.inv_freq, original).item())
        rope(x, paddle.to_tensor([[2]]))
        self.assertEqual(int(rope.max_seq_len_cached), 8)
        paddle.testing.assert_close(rope.inv_freq, original, rtol=0.0, atol=0.0)

    @parameterized.expand(
        [
            (layout, recompute, dtype)
            for layout in ("none", "right", "left", "empty", "prefix_3d", "prefix_4d", "broadcast_4d")
            for recompute in (False, True)
            for dtype in ("float32", "float16", "bfloat16")
        ]
    )
    def test_pipeline_stages_match_regular_model(self, mask_layout, with_recompute, dtype):
        from paddleformers.nn.pp_model import LayerNormPipe, LMHeadPipe
        from paddleformers.transformers.stablelm.modeling import (
            StableLmCriterionLayerPipe,
            StableLmDecoderLayerPipe,
            StableLmEmbeddingPipe,
        )

        config = self.model_tester.get_config(
            qk_layernorm=True,
        )
        if with_recompute:
            config.recompute_granularity = "full"
            config.recompute_method = "uniform"
            config.recompute_num_layers = 1
        model = StableLmForCausalLM(config)
        model.train()
        embed = StableLmEmbeddingPipe(config, rotary_emb_cls=StableLmRotaryEmbedding)
        embed.embed_tokens.set_state_dict(model.model.embed_tokens.state_dict())
        layers = [StableLmDecoderLayerPipe(config, i) for i in range(2)]
        for index, layer in enumerate(layers):
            layer.set_state_dict(model.model.layers[index].state_dict())
        norm = LayerNormPipe(config)
        norm.set_state_dict(model.model.norm.state_dict())
        head = LMHeadPipe(config)
        head.set_state_dict(model.lm_head.state_dict())
        criterion = StableLmCriterionLayerPipe(config)
        for component in (model, embed, *layers, norm, head):
            component.to(dtype=dtype)
        x = paddle.to_tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
        masks = {
            "right": [[1, 1, 1, 1], [1, 1, 1, 0]],
            "left": [[0, 0, 1, 1], [1, 1, 1, 1]],
            "empty": [[0, 0, 0, 0], [1, 1, 1, 1]],
        }
        mask = paddle.to_tensor(masks.get(mask_layout, masks["right"]))
        with_mask = mask_layout != "none"
        loss_mask = mask.astype("float32")
        if mask_layout in ("prefix_3d", "prefix_4d", "broadcast_4d"):
            loss_mask = paddle.ones_like(x, dtype="float32")
            prefix_mask = [[[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]]]
            mask = paddle.to_tensor(prefix_mask, dtype="bool").tile([2, 1, 1])
            if mask_layout == "prefix_4d":
                mask = mask.unsqueeze(1)
            elif mask_layout == "broadcast_4d":
                mask = paddle.ones([2, 1, 1, 4], dtype="bool")
        expected = model(
            x, attention_mask=mask if with_mask else None, labels=x, loss_mask=loss_mask, return_dict=True
        )
        states = embed((x, mask) if with_mask else x)
        for layer in layers:
            states = layer(states)
        logits = head(norm(states))
        loss = criterion(logits, (x, loss_mask))
        self.assertEqual(logits.dtype, expected.logits.dtype)
        paddle.testing.assert_close(logits.float(), expected.logits.float(), rtol=1e-5, atol=1e-6)
        paddle.testing.assert_close(loss, expected.loss, rtol=1e-5, atol=1e-6)
        loss.backward()
        expected.loss.backward()
        paddle.testing.assert_close(
            layers[0].self_attn.q_layernorm.weight.grad.float(),
            model.model.layers[0].self_attn.q_layernorm.weight.grad.float(),
            rtol=1e-5,
            atol=1e-6,
        )

    @require_gpu(min_gpus=2)
    def test_pipeline_parallel_training_and_checkpoint(self):
        from paddle.distributed import fleet

        from paddleformers.transformers import AutoModelForCausalLMPipe

        world_size = paddle.distributed.get_world_size()
        if world_size not in (2, 4):
            self.skipTest("Requires a distributed launch with two or four GPU ranks")
        paddle.set_device(f"gpu:{paddle.distributed.ParallelEnv().dev_id}")
        strategy = fleet.DistributedStrategy()
        tp_size = world_size // 2
        strategy.hybrid_configs = dict(dp_degree=1, mp_degree=tp_size, pp_degree=2, sharding_degree=1)
        strategy.pipeline_configs = dict(micro_batch_size=1, accumulate_steps=1)
        fleet.init(is_collective=True, strategy=strategy)
        hcg = fleet.get_hybrid_communicate_group()
        paddle.seed(123)
        config = self.model_tester.get_config(
            vocab_size=32,
            eos_token_id=31,
            qk_layernorm=True,
            use_cache=False,
            tensor_parallel_output=False,
            recompute_granularity="full",
            recompute_method="uniform",
            recompute_num_layers=1,
        )
        reference = StableLmForCausalLM(config)
        pipe = AutoModelForCausalLMPipe.from_config(StableLmConfig.from_dict(config.to_dict()))
        self.assertIsInstance(pipe, StableLmForCausalLMPipe)
        original = reference.state_dict()
        actions = pipe._get_tensor_parallel_mappings(pipe.config)
        local = {}
        for name in pipe.state_dict():
            key = name.removeprefix("model.")
            local[name] = actions[key](original[name]) if tp_size > 1 and key in actions else original[name]
        pipe.set_state_dict(local)
        x = paddle.to_tensor([[1, 2, 3, 4]])
        labels = paddle.to_tensor([[2, 3, 4, 5]])
        loss_mask = paddle.to_tensor([[1.0, 1.0, 0.0, 1.0]])
        expected = reference(x, labels=labels, loss_mask=loss_mask, return_dict=True)
        engine = fleet.distributed_model(pipe)
        engine.eval()
        with paddle.no_grad():
            outputs = engine.eval_batch([x, labels], compute_loss=False)
        if hcg.get_stage_id() == 1:
            paddle.testing.assert_close(outputs[0], expected.logits, rtol=1e-5, atol=1e-6)
        expected.loss.backward()
        engine.train()
        optimizer = fleet.distributed_optimizer(paddle.optimizer.SGD(learning_rate=0.01, parameters=pipe.parameters()))
        before = [parameter.clone() for parameter in pipe.parameters()]
        loss = engine.train_batch([x, (labels, loss_mask)], optimizer=optimizer)
        paddle.testing.assert_close(loss, expected.loss.detach(), rtol=1e-5, atol=1e-6)
        self.assertTrue(
            any(not paddle.equal_all(old, parameter).item() for old, parameter in zip(before, pipe.parameters()))
        )
        paddle.optimizer.SGD(learning_rate=0.01, parameters=reference.parameters()).step()
        updated = reference.state_dict()
        for name, parameter in pipe.state_dict().items():
            key = name.removeprefix("model.")
            expected_parameter = actions[key](updated[name]) if tp_size > 1 and key in actions else updated[name]
            paddle.testing.assert_close(parameter, expected_parameter, rtol=1e-5, atol=1e-6)
        with tempfile.TemporaryDirectory() as directory:
            directories = []
            paddle.distributed.all_gather_object(directories, directory)
            directory = directories[0]
            pipe.save_pretrained(directory, save_checkpoint_format="flex_checkpoint")
            paddle.distributed.barrier()
            loaded = AutoModelForCausalLMPipe.from_pretrained(
                directory, dtype="float32", load_checkpoint_format="flex_checkpoint"
            )
            for name, value in pipe.state_dict().items():
                paddle.testing.assert_close(loaded.state_dict()[name], value, rtol=0.0, atol=0.0)
            paddle.distributed.barrier()

    def test_lora_recompute_training(self):
        from paddleformers.cli.utils.llm_utils import get_lora_target_modules
        from paddleformers.peft import LoRAConfig, LoRAModel

        config = self.model_tester.get_config(
            qk_layernorm=True, recompute_granularity="full", recompute_method="uniform", recompute_num_layers=1
        )
        base = StableLmForCausalLM(config)
        model = LoRAModel(base, LoRAConfig(target_modules=get_lora_target_modules(base), r=2, lora_alpha=4))
        model.train()
        self.assertTrue(base.model.embed_tokens.weight.stop_gradient)
        x = paddle.to_tensor([[1, 2, 3, 4]])
        loss = model(x, labels=x, return_dict=True).loss
        loss.backward()
        trainable = [(name, parameter) for name, parameter in model.named_parameters() if not parameter.stop_gradient]
        self.assertTrue(trainable)
        for name, parameter in trainable:
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(paddle.isfinite(parameter.grad).all().item(), name)
        self.assertTrue(any(parameter.grad.abs().sum().item() > 0 for _, parameter in trainable))

    @parameterized.expand([("int64",), ("float32",), ("bool",)])
    def test_attention_mask_bool_cast(self, dtype):
        model = StableLmForCausalLM(self.model_tester.get_config()).eval()
        x = paddle.to_tensor([[1, 2, 3]])
        mask = paddle.to_tensor([[1, 1, 0]], dtype=dtype)
        with paddle.no_grad():
            actual = model(x, attention_mask=mask, return_dict=True).logits
            expected = model(x, attention_mask=mask.astype("bool"), return_dict=True).logits
            reference = model.model(x, return_dict=True)
        paddle.testing.assert_close(actual, expected)
        with patch.object(model.model, "forward", return_value=reference) as forward:
            model(x, attention_mask=mask)
        self.assertEqual(forward.call_args.kwargs["attention_mask"].dtype, paddle.bool)

    @parameterized.expand([(False,), (True,)])
    def test_attention_row_indices_compatibility(self, with_new_name):
        model = StableLmForCausalLM(self.model_tester.get_config()).eval()
        x = paddle.to_tensor([[1, 2, 3]])
        old_indices = paddle.zeros([1, 1, 3, 1], dtype="int32")
        new_indices = paddle.ones_like(old_indices)
        with paddle.no_grad():
            reference = model.model(x, return_dict=True)
        kwargs = {"attn_mask_start_row_indices": old_indices}
        if with_new_name:
            kwargs["attn_mask_startend_row_indices"] = new_indices
        with patch.object(model.model, "forward", return_value=reference) as forward, patch(
            "paddleformers.transformers.stablelm.modeling.logger.warning"
        ) as warning:
            model(x, attention_mask=paddle.ones_like(x), **kwargs)
        self.assertIsNone(forward.call_args.kwargs["attention_mask"])
        self.assertIs(
            forward.call_args.kwargs["attn_mask_startend_row_indices"],
            new_indices if with_new_name else old_indices,
        )
        warning.assert_called_once_with(
            "You have provided both attn_mask_startend_row_indices and attention_mask. "
            "The attn_mask_startend_row_indices will be used."
        )

    def test_flashmask_preserves_dense_mask_and_dropout(self):
        model = StableLmForCausalLM(
            self.model_tester.get_config(qk_layernorm=True, _attn_implementation="flashmask", attention_dropout=0.1)
        )
        model.train()
        x = paddle.to_tensor([[1, 2, 3]])
        attention = model.model.layers[0].self_attn
        from paddleformers.nn.attention.interface import ALL_ATTENTION_FUNCTIONS

        with patch.dict(ALL_ATTENTION_FUNCTIONS._global_mapping, sdpa=ALL_ATTENTION_FUNCTIONS["eager"]):
            result = model(x, labels=x, return_dict=True)
            result.loss.backward()
        self.assertIsNotNone(attention.q_proj.weight.grad)


class StableLmIntegrationTest(unittest.TestCase):
    @gpu_device_initializer(log_prefix="StableLmIntegrationTest")
    @require_package("transformers", "torch")
    @slow
    def test_generation_hello_world(self):
        tokenizer = StableLmTokenizer.from_pretrained(
            "stabilityai/stable-code-3b", trust_remote_code=False, download_hub="modelscope"
        )

        model = StableLmForCausalLM.from_pretrained(
            "stabilityai/stable-code-3b",
            load_via_cpu=True,
            download_hub="modelscope",
        )
        model.eval()

        prompt = "def hello_world():"
        inputs = tokenizer(prompt, return_tensors="np")
        input_ids = paddle.to_tensor(inputs["input_ids"])

        with paddle.no_grad():
            gen = model.generate(
                input_ids,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated_tokens = gen[0][0].tolist()
        decoded = tokenizer.decode(generated_tokens)
        print("create code:" + decoded)
        self.assertIn("hello", decoded.lower())
        self.assertGreater(len(generated_tokens), 3)


class StableLmCompatibilityTest(unittest.TestCase):
    @gpu_device_initializer(log_prefix="StableLmCompatibilityTest")
    @require_package("transformers", "torch")
    @slow
    def test_diff_generated_tokens(self):
        """Compare first 10 generated tokens between Paddle and HF."""
        import torch
        from transformers import AutoModelForCausalLM

        tokenizer = StableLmTokenizer.from_pretrained(
            "stabilityai/stable-code-3b", trust_remote_code=False, download_hub="modelscope"
        )

        prompt = "def fib(n):"
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids_np = inputs["input_ids"].cpu().numpy()

        hf_model = AutoModelForCausalLM.from_pretrained(
            "stabilityai/stable-code-3b",
            dtype=torch.float32,
            device_map="auto",
        )
        hf_model.eval()
        hf_inputs = {k: v.to(hf_model.device) for k, v in inputs.items()}
        with torch.no_grad():
            hf_gen = hf_model.generate(
                **hf_inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        hf_new_tokens = hf_gen[0, inputs["input_ids"].shape[1] :].tolist()

        pd_model = StableLmForCausalLM.from_pretrained(
            "stabilityai/stable-code-3b",
            dtype="float32",
            load_via_cpu=True,
            download_hub="modelscope",
        )
        pd_model.eval()
        pd_input_ids = paddle.to_tensor(input_ids_np)
        with paddle.no_grad():
            pd_gen = pd_model.generate(
                pd_input_ids,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        pd_new_tokens = pd_gen[0][0].tolist()

        self.assertEqual(
            hf_new_tokens,
            pd_new_tokens,
            f"Generated tokens differ: HF={hf_new_tokens}, PD={pd_new_tokens}",
        )

    @gpu_device_initializer(log_prefix="StableLmCompatibilityTest")
    @require_package("transformers", "torch")
    @slow
    def test_diff_first_token_logits(self):
        """Compare first token logits between Paddle and HF, diff should be within 1e-2."""
        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM

        tokenizer = StableLmTokenizer.from_pretrained(
            "stabilityai/stable-code-3b", trust_remote_code=False, download_hub="modelscope"
        )

        prompt = "def sort_array(arr):"
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids_np = inputs["input_ids"].cpu().numpy()

        hf_model = AutoModelForCausalLM.from_pretrained(
            "stabilityai/stable-code-3b",
            dtype=torch.float32,
            device_map="auto",
        )
        hf_model.eval()
        hf_inputs = {k: v.to(hf_model.device) for k, v in inputs.items()}
        with torch.no_grad():
            hf_outputs = hf_model(**hf_inputs)
            hf_logits = hf_outputs.logits[0, -1].cpu().float().numpy()

        pd_model = StableLmForCausalLM.from_pretrained(
            "stabilityai/stable-code-3b",
            dtype="float32",
            load_via_cpu=True,
            download_hub="modelscope",
        )
        pd_model.eval()
        pd_input_ids = paddle.to_tensor(input_ids_np)
        with paddle.no_grad():
            pd_outputs = pd_model(pd_input_ids, return_dict=True)
            pd_logits = pd_outputs.logits[0, -1].cpu().astype("float32").numpy()

        abs_diff = np.abs(hf_logits - pd_logits)
        mean_diff = np.mean(abs_diff)

        self.assertLess(
            mean_diff,
            1e-2,
            f"Logits mean diff {mean_diff:.8f} exceeds threshold 1e-2",
        )

        hf_top10 = np.argsort(hf_logits)[-10:][::-1]
        pd_top10 = np.argsort(pd_logits)[-10:][::-1]
        self.assertEqual(
            hf_top10.tolist(),
            pd_top10.tolist(),
            "Top 10 tokens differ between HF and Paddle",
        )


if __name__ == "__main__":
    unittest.main()
