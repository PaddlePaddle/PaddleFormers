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

import tempfile
import unittest
from unittest.mock import patch

import paddle
from parameterized import parameterized

from paddleformers.transformers import (
    AutoTokenizer,
    Ministral3Config,
    Ministral3ForCausalLM,
    Ministral3ForCausalLMPipe,
    Ministral3Model,
)
from paddleformers.transformers.ministral3.modeling import (
    Ministral3RotaryEmbedding,
    get_llama_4_attn_scale,
)
from tests.testing_utils import (
    gpu_device_initializer,
    require_gpu,
    require_package,
    slow,
)
from tests.transformers.test_generation_utils import GenerationTesterMixin
from tests.transformers.test_modeling_common import (
    ModelTesterMixin,
    ids_tensor,
    random_attention_mask,
)

_MODEL_HF_ID = "mistralai/Ministral-3-3B-Instruct-2512"


def _small_rope_parameters(original_max_position_embeddings=16):
    return {
        "rope_type": "yarn",
        "rope_theta": 10000.0,
        "factor": 8.0,
        "original_max_position_embeddings": original_max_position_embeddings,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
        "llama_4_scaling_beta": 0.1,
    }


class Ministral3ModelTester:
    def __init__(
        self,
        parent,
        vocab_size=99,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        head_dim=16,
        max_position_embeddings=128,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        attention_dropout=0.0,
        is_training=True,
        use_cache=False,
        bos_token_id=1,
        eos_token_id=98,
        pad_token_id=0,
        batch_size=2,
        seq_length=7,
        use_input_mask=True,
        return_dict=False,
    ):
        self.parent = parent
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.attention_dropout = attention_dropout
        self.is_training = is_training
        self.use_cache = use_cache
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
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
            head_dim=self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            rms_norm_eps=self.rms_norm_eps,
            attention_dropout=self.attention_dropout,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            initializer_range=self.initializer_range,
            use_cache=self.use_cache,
            return_dict=self.return_dict,
            rope_parameters=_small_rope_parameters(),
            _attn_implementation="eager",
        )
        values.update(kwargs)
        return Ministral3Config(**values)

    def prepare_config_and_inputs(self):
        config = self.get_config()
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size, dtype=paddle.int64)
        attention_mask = random_attention_mask([self.batch_size, self.seq_length]) if self.use_input_mask else None
        return config, input_ids, attention_mask

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, attention_mask = self.prepare_config_and_inputs()
        return config, {"input_ids": input_ids, "attention_mask": attention_mask}

    def create_and_check_model(self, config, input_ids, attention_mask):
        model = Ministral3Model(config)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, return_dict=True)
        self.parent.assertEqual(result.last_hidden_state.shape, [self.batch_size, self.seq_length, self.hidden_size])

    def create_and_check_lm_head_model(self, config, input_ids, attention_mask):
        model = Ministral3ForCausalLM(config)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, return_dict=True)
        self.parent.assertEqual(result.logits.shape, [self.batch_size, self.seq_length, self.vocab_size])


class Ministral3ModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    base_model_class = Ministral3Model
    return_dict = False
    use_labels = False
    use_test_model_name_list = False
    has_attentions = True
    all_model_classes = (Ministral3Model, Ministral3ForCausalLM)
    all_generative_model_classes = {Ministral3ForCausalLM: (Ministral3Model, "ministral3")}

    @gpu_device_initializer(log_prefix="Ministral3ModelTest")
    def setUp(self):
        super().setUp()
        self.model_tester = Ministral3ModelTester(self)
        self.config = self.model_tester.get_config()

    def test_config(self):
        config = Ministral3Config()
        self.assertEqual(config.model_type, "ministral3")
        self.assertEqual(config.hidden_size, 4096)
        self.assertEqual(config.num_hidden_layers, 34)
        self.assertEqual(config.num_key_value_heads, 8)
        self.assertEqual(config.head_dim, 128)
        self.assertFalse(config.tie_word_embeddings)
        self.assertEqual(config.pad_token_id, 11)
        self.assertIsInstance(config.rope_parameters, dict)
        self.assertEqual(config.rope_parameters["rope_type"], "yarn")
        self.assertEqual(config.rope_parameters["llama_4_scaling_beta"], 0.1)

        fallback = Ministral3Config(num_attention_heads=8, num_key_value_heads=None, head_dim=None, hidden_size=512)
        self.assertEqual(fallback.num_key_value_heads, 8)
        self.assertEqual(fallback.head_dim, 64)

    def test_config_rope_scaling_alias_is_not_mutated(self):
        rope_scaling = {"type": "linear", "rope_theta": 10000.0, "factor": 2.0}
        config = self.model_tester.get_config(rope_parameters=None, rope_scaling=rope_scaling)
        self.assertEqual(config.rope_parameters["rope_type"], "linear")
        self.assertNotIn("rope_type", rope_scaling)

    def test_model(self):
        self.model_tester.create_and_check_model(*self.model_tester.prepare_config_and_inputs())

    def test_ministral3_lm_head_model(self):
        self.model_tester.create_and_check_lm_head_model(*self.model_tester.prepare_config_and_inputs())

    def test_gqa_head_split(self):
        config = self.model_tester.get_config()
        model = Ministral3ForCausalLM(config)
        attention = model.model.layers[0].self_attn
        self.assertEqual(attention.num_key_value_groups, 2)
        self.assertEqual(attention.head_dim, config.head_dim)

    def test_llama4_attention_scale_uses_position_ids(self):
        """Packed sequences must reset the Llama-4 q scaling at every document start."""
        model = Ministral3ForCausalLM(self.model_tester.get_config())
        model.eval()
        input_ids = paddle.to_tensor([[1, 2, 3, 4, 5, 6]], dtype="int64")
        position_ids = paddle.to_tensor([[0, 1, 2, 0, 1, 2]], dtype="int64")
        row_indices = paddle.to_tensor([[[[3], [3], [3], [6], [6], [6]]]], dtype="int32")
        with paddle.no_grad():
            packed = model(
                input_ids=input_ids,
                position_ids=position_ids,
                attn_mask_startend_row_indices=row_indices,
                return_dict=True,
            ).logits
            first = model(input_ids=input_ids[:, :3], return_dict=True).logits
            second = model(input_ids=input_ids[:, 3:], return_dict=True).logits
        paddle.testing.assert_close(packed, paddle.concat([first, second], axis=1), rtol=1e-5, atol=1e-6)

    def test_llama4_attention_scale_is_applied(self):
        config = self.model_tester.get_config(max_position_embeddings=256)
        config.rope_parameters["original_max_position_embeddings"] = 8
        config.rope_parameters["factor"] = 32.0
        model = Ministral3ForCausalLM(config)
        model.eval()
        attention = model.model.layers[0].self_attn
        position_ids = paddle.to_tensor([[0, 7, 8, 100]], dtype="int64")
        scale = attention.llama_4_scaling_beta
        self.assertIsNotNone(scale)
        computed = get_llama_4_attn_scale(position_ids, attention.llama_4_scaling_beta, 8)
        self.assertEqual(list(computed.shape), [1, 1, 4, 1])
        self.assertEqual(float(computed[0, 0, 0, 0]), 1.0)
        self.assertGreater(float(computed[0, 0, 3, 0]), 1.0)

    def test_padding_does_not_leak(self):
        model = Ministral3ForCausalLM(self.model_tester.get_config())
        model.eval()
        input_ids = paddle.to_tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype="int64")
        attention_mask = paddle.to_tensor([[1, 1, 1, 1, 1, 0, 0, 0]], dtype="bool")
        padded_ids = input_ids.clone()
        padded_ids[0, 5:] = paddle.to_tensor([97, 66, 55], dtype="int64")
        with paddle.no_grad():
            clean = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True).logits
            padded = model(input_ids=padded_ids, attention_mask=attention_mask, return_dict=True).logits
        paddle.testing.assert_close(clean[:, :5], padded[:, :5], rtol=0.0, atol=0.0)
        self.assertTrue(paddle.isfinite(padded).all().item())

    def test_sparse_mask_matches_dense(self):
        model = Ministral3ForCausalLM(self.model_tester.get_config())
        model.eval()
        input_ids = paddle.to_tensor([[1, 2, 3, 4, 5, 6]], dtype="int64")
        attention_mask = paddle.to_tensor([[1, 1, 1, 1, 0, 0]], dtype="bool")
        row_indices = paddle.to_tensor([[[6, 6, 6, 6, 0, 0]]], dtype="int32")
        with paddle.no_grad():
            dense = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True, return_dict=True)
            for indices in (row_indices, row_indices.unsqueeze(-1)):
                sparse = model(
                    input_ids=input_ids,
                    attn_mask_startend_row_indices=indices,
                    output_attentions=True,
                    return_dict=True,
                )
                paddle.testing.assert_close(sparse.logits, dense.logits, rtol=1e-5, atol=1e-6)
                for weights in sparse.attentions:
                    self.assertEqual(paddle.triu(weights, diagonal=1).abs().max().item(), 0.0)
                    self.assertEqual(weights[:, :, :, 4:].abs().max().item(), 0.0)

    @parameterized.expand([("eager", "float32"), ("sdpa", "bfloat16")])
    def test_cache_multi_step_and_append(self, backend, dtype):
        config = self.model_tester.get_config(_attn_implementation=backend, use_cache=True)
        model = Ministral3ForCausalLM(config).to(dtype=dtype)
        model.eval()
        x = paddle.to_tensor([[1, 2, 3, 4, 5, 6]], dtype="int64")
        rtol, atol = (1e-3, 1e-3) if dtype == "bfloat16" else (1e-5, 1e-6)
        with paddle.no_grad():
            expected = model(x, use_cache=False, return_dict=True).logits
            cache = model(x[:, :2], use_cache=True, return_dict=True).past_key_values
            cached = model(x[:, 2:4], past_key_values=cache, use_cache=True, return_dict=True)
            self.assertEqual(cache.get_seq_length(), 4)
            paddle.testing.assert_close(cached.logits.float(), expected[:, 2:4].float(), rtol=rtol, atol=atol)
            for end in (5, 6):
                cached = model(x[:, end - 1 : end], past_key_values=cache, use_cache=True, return_dict=True)
                self.assertEqual(cache.get_seq_length(), end)
                paddle.testing.assert_close(
                    cached.logits.float(), expected[:, end - 1 : end].float(), rtol=rtol, atol=atol
                )

    def test_output_hidden_states_and_attentions(self):
        model = Ministral3ForCausalLM(self.model_tester.get_config())
        model.eval()
        input_ids = paddle.randint(0, 99, [1, 6])
        with paddle.no_grad():
            outputs = model(input_ids, output_hidden_states=True, output_attentions=True, return_dict=True)
        self.assertEqual(len(outputs.hidden_states), self.config.num_hidden_layers + 1)
        self.assertEqual(len(outputs.attentions), self.config.num_hidden_layers)
        self.assertTrue(paddle.isfinite(outputs.logits).all().item())

    def test_output_attentions_requires_eager(self):
        model = Ministral3ForCausalLM(self.model_tester.get_config(_attn_implementation="sdpa"))
        with self.assertRaises(ValueError):
            model(paddle.to_tensor([[1, 2, 3]]), output_attentions=True)

    def test_apply_rope_fusion_is_rejected(self):
        with self.assertRaises(ValueError):
            Ministral3ForCausalLM(self.model_tester.get_config(apply_rope_fusion=True))

    def test_save_load_roundtrip(self):
        config = self.model_tester.get_config(vocab_size=128, tie_word_embeddings=False)
        model = Ministral3ForCausalLM(config)
        model.eval()
        input_ids = paddle.to_tensor([[1, 2, 3, 4]], dtype="int64")
        with paddle.no_grad():
            expected = model(input_ids, return_dict=True).logits
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            loaded = Ministral3ForCausalLM.from_pretrained(
                tmpdir, dtype="float32", load_checkpoint_format="flex_checkpoint"
            )
        loaded.eval()
        self.assertIsNot(loaded.lm_head.weight, loaded.model.embed_tokens.weight)
        self.assertEqual(set(model.state_dict()), set(loaded.state_dict()))
        for name, value in model.state_dict().items():
            paddle.testing.assert_close(value, loaded.state_dict()[name], rtol=0.0, atol=0.0)
        with paddle.no_grad():
            actual = loaded(input_ids, return_dict=True).logits
        paddle.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_tied_embedding_roundtrip(self):
        config = self.model_tester.get_config(vocab_size=128, tie_word_embeddings=True)
        model = Ministral3ForCausalLM(config)
        self.assertIs(model.lm_head.weight, model.model.embed_tokens.weight)
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            loaded = Ministral3ForCausalLM.from_pretrained(
                tmpdir, dtype="float32", load_checkpoint_format="flex_checkpoint"
            )
        self.assertIs(loaded.lm_head.weight, loaded.model.embed_tokens.weight)

    def test_generation_smoke(self):
        model = Ministral3ForCausalLM(self.model_tester.get_config(use_cache=True))
        model.eval()
        with paddle.no_grad():
            generated = model.generate(
                paddle.to_tensor([[1, 2, 3]]),
                min_new_tokens=2,
                max_new_tokens=2,
                decode_strategy="greedy_search",
                eos_token_id=None,
                pad_token_id=0,
            )
        self.assertEqual(list(generated[0].shape), [1, 2])

    @parameterized.expand([(None,), ([],)])
    def test_auto_entrypoint_and_lora_targets(self, architectures):
        import re

        from paddleformers.cli.utils.llm_utils import get_lora_target_modules
        from paddleformers.transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_config(self.model_tester.get_config(architectures=architectures))
        self.assertIsInstance(model, Ministral3ForCausalLM)
        targets = get_lora_target_modules(model)
        names = [name for name, _ in model.named_sublayers()]
        for pattern in targets:
            self.assertTrue(any(re.fullmatch(pattern, name) for name in names), pattern)

    @parameterized.expand([("selective", "uniform", 1), ("full", "block", 1), ("full", "uniform", 2)])
    def test_recompute_fallback_training(self, granularity, method, num_layers):
        reference = Ministral3ForCausalLM(self.model_tester.get_config())
        actual = Ministral3ForCausalLM(
            self.model_tester.get_config(
                recompute_granularity=granularity,
                recompute_method=method,
                recompute_num_layers=num_layers,
            )
        )
        actual.set_state_dict(reference.state_dict())
        for model in (reference, actual):
            model.train()
        x = paddle.to_tensor([[1, 2, 3, 4]], dtype="int64")
        paddle.seed(123)
        expected = reference(x, labels=x, use_cache=False, return_dict=True).loss
        with patch("paddleformers.transformers.ministral3.modeling.recompute") as recompute:
            paddle.seed(123)
            observed = actual(x, labels=x, use_cache=False, return_dict=True).loss
            observed.backward()
            recompute.assert_not_called()
        expected.backward()
        paddle.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-6)
        for (name, param), (_, other) in zip(reference.named_parameters(), actual.named_parameters()):
            if not param.stop_gradient:
                self.assertIsNotNone(other.grad, name)
                self.assertTrue(paddle.isfinite(other.grad).all().item(), name)
                paddle.testing.assert_close(other.grad, param.grad, rtol=1e-5, atol=1e-6)

    @parameterized.expand([(False,), (True,)])
    def test_recompute_loss_gradients_and_reexecution(self, freeze_embedding):
        # Paddle's fused_rms_norm backward rejects a frozen input in this environment; the
        # unfused path is used for the frozen-embedding case so this test still exercises
        # recompute with an input that does not require gradients.
        extra = {"fuse_rms_norm": False} if freeze_embedding else {}
        reference = Ministral3ForCausalLM(self.model_tester.get_config(attention_dropout=0.1, **extra))
        actual = Ministral3ForCausalLM(
            self.model_tester.get_config(
                attention_dropout=0.1,
                recompute_granularity="full",
                recompute_method="uniform",
                recompute_num_layers=1,
                **extra,
            )
        )
        actual.set_state_dict(reference.state_dict())
        for model in (reference, actual):
            model.train()
            model.model.embed_tokens.weight.stop_gradient = freeze_embedding
        x = paddle.to_tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype="int64")
        paddle.seed(123)
        expected = reference(x, labels=x, use_cache=False, return_dict=True).loss
        expected.backward()
        layer = actual.model.layers[0]
        hook_calls = []
        pre_hook = layer.register_forward_pre_hook(lambda *_: hook_calls.append("pre"))
        post_hook = layer.register_forward_post_hook(lambda *_: hook_calls.append("post"))
        with patch.object(layer.self_attn, "forward", wraps=layer.self_attn.forward) as forward:
            paddle.seed(123)
            observed = actual(x, labels=x, use_cache=True, return_dict=True).loss
            self.assertEqual(hook_calls, ["pre", "post"])
            observed.backward()
            self.assertGreaterEqual(forward.call_count, 2)
        pre_hook.remove()
        post_hook.remove()
        paddle.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-6)
        for (name, param), (_, other) in zip(reference.named_parameters(), actual.named_parameters()):
            if not param.stop_gradient:
                self.assertIsNotNone(other.grad, name)
                self.assertTrue(paddle.isfinite(other.grad).all().item(), name)
                paddle.testing.assert_close(other.grad, param.grad, rtol=1e-5, atol=1e-6)

    def test_recompute_rejects_incompatible_inputs(self):
        model = Ministral3ForCausalLM(
            self.model_tester.get_config(
                recompute_granularity="full",
                recompute_method="uniform",
                recompute_num_layers=1,
            )
        )
        model.train()
        from paddleformers.transformers.cache_utils import DynamicCache

        x = paddle.to_tensor([[1, 2, 3]], dtype="int64")
        with self.assertRaises(ValueError):
            model(x, past_key_values=DynamicCache(), return_dict=True)

    def test_pipeline_stages_match_regular_model(self):
        from paddleformers.nn.pp_model import LMHeadPipe, RMSNormPipe
        from paddleformers.transformers.ministral3.modeling import (
            Ministral3CriterionLayerPipe,
            Ministral3DecoderLayerPipe,
            Ministral3EmbeddingPipe,
        )

        config = self.model_tester.get_config()
        model = Ministral3ForCausalLM(config)
        model.train()
        embed = Ministral3EmbeddingPipe(config, rotary_emb_cls=Ministral3RotaryEmbedding)
        embed.embed_tokens.set_state_dict(model.model.embed_tokens.state_dict())
        layers = [Ministral3DecoderLayerPipe(config, i) for i in range(config.num_hidden_layers)]
        for index, layer in enumerate(layers):
            layer.set_state_dict(model.model.layers[index].state_dict())
        norm = RMSNormPipe(config)
        norm.set_state_dict(model.model.norm.state_dict())
        head = LMHeadPipe(config)
        head.set_state_dict(model.lm_head.state_dict())
        criterion = Ministral3CriterionLayerPipe(config)
        x = paddle.to_tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype="int64")
        mask = paddle.to_tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype="bool")
        loss_mask = mask.astype("float32")
        expected = model(x, attention_mask=mask, labels=x, loss_mask=loss_mask, return_dict=True)
        states = embed((x, mask))
        for layer in layers:
            states = layer(states)
        logits = head(norm(states))
        loss = criterion(logits, (x, loss_mask))
        paddle.testing.assert_close(logits.float(), expected.logits.float(), rtol=1e-5, atol=1e-6)
        paddle.testing.assert_close(loss, expected.loss, rtol=1e-5, atol=1e-6)
        loss.backward()
        expected.loss.backward()
        paddle.testing.assert_close(
            layers[0].self_attn.q_proj.weight.grad.float(),
            model.model.layers[0].self_attn.q_proj.weight.grad.float(),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_pipeline_pipe_class_is_pipeline_layer(self):
        self.assertTrue(issubclass(Ministral3ForCausalLMPipe, paddle.distributed.fleet.meta_parallel.PipelineLayer))

    @require_gpu(min_gpus=2)
    def test_pipeline_parallel_from_config(self):
        world_size = paddle.distributed.get_world_size()
        if world_size not in (2, 4):
            self.skipTest("Requires a distributed launch with two or four GPU ranks")
        from paddle.distributed import fleet

        from paddleformers.transformers import AutoModelForCausalLMPipe

        paddle.set_device(f"gpu:{paddle.distributed.ParallelEnv().dev_id}")
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = dict(dp_degree=1, mp_degree=world_size // 2, pp_degree=2, sharding_degree=1)
        strategy.pipeline_configs = dict(micro_batch_size=1, accumulate_steps=1)
        fleet.init(is_collective=True, strategy=strategy)
        paddle.seed(123)
        config = self.model_tester.get_config(
            recompute_granularity="full",
            recompute_method="uniform",
            recompute_num_layers=1,
            tensor_parallel_output=False,
        )
        pipe = AutoModelForCausalLMPipe.from_config(Ministral3Config.from_dict(config.to_dict()), dtype="float32")
        self.assertIsInstance(pipe, Ministral3ForCausalLMPipe)
        x = paddle.to_tensor([[1, 2, 3, 4]], dtype="int64")
        labels = paddle.to_tensor([[2, 3, 4, 5]], dtype="int64")
        engine = fleet.distributed_model(pipe)
        engine.train()
        optimizer = fleet.distributed_optimizer(paddle.optimizer.SGD(learning_rate=0.01, parameters=pipe.parameters()))
        loss = engine.train_batch([x, (labels, paddle.ones_like(labels, dtype="float32"))], optimizer=optimizer)
        self.assertTrue(paddle.isfinite(loss).all().item())

    @slow
    @require_gpu(min_gpus=2)
    def test_pipeline_parallel_generate(self):
        # Reference PP greedy-decoding entry; runs only under a multi-rank launch.
        world_size = paddle.distributed.get_world_size()
        if world_size not in (2, 4):
            self.skipTest("Requires a distributed launch with two or four GPU ranks")
        from paddle.distributed import fleet

        from paddleformers.transformers import AutoModelForCausalLMPipe

        paddle.set_device(f"gpu:{paddle.distributed.ParallelEnv().dev_id}")
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = dict(dp_degree=1, mp_degree=world_size // 2, pp_degree=2, sharding_degree=1)
        strategy.pipeline_configs = dict(micro_batch_size=1, accumulate_steps=1)
        fleet.init(is_collective=True, strategy=strategy)
        paddle.seed(123)
        config = self.model_tester.get_config(vocab_size=32, eos_token_id=31)
        pipe = AutoModelForCausalLMPipe.from_config(Ministral3Config.from_dict(config.to_dict()), dtype="float32")
        self.assertIsInstance(pipe, Ministral3ForCausalLMPipe)
        input_ids = paddle.to_tensor([[1, 2, 3]], dtype="int64")
        generated = pipe.generate(input_ids, max_new_tokens=2, eos_token_id=None)
        self.assertEqual(list(generated.shape), [1, 5])


class Ministral3IntegrationTest(unittest.TestCase):
    @gpu_device_initializer(log_prefix="Ministral3IntegrationTest")
    @require_package("transformers", "torch")
    @slow
    def test_generation(self):
        tokenizer = AutoTokenizer.from_pretrained(_MODEL_HF_ID, trust_remote_code=False, download_hub="modelscope")
        model = Ministral3ForCausalLM.from_pretrained(
            _MODEL_HF_ID,
            load_via_cpu=True,
            download_hub="modelscope",
            load_checkpoint_format="flex_checkpoint",
        )
        model.eval()

        inputs = tokenizer("Hello, how are you today?", return_tensors="np")
        input_ids = paddle.to_tensor(inputs["input_ids"])
        with paddle.no_grad():
            generated = model.generate(
                input_ids,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        tokens = generated[0][0].tolist()
        decoded = tokenizer.decode(tokens)
        self.assertGreater(len(tokens), 3)
        self.assertGreater(len(decoded.strip()), 0)


class Ministral3CompatibilityTest(unittest.TestCase):
    @gpu_device_initializer(log_prefix="Ministral3CompatibilityTest")
    @require_package("transformers", "torch")
    @slow
    def test_diff_first_token_logits(self):
        import numpy as np
        import torch
        from transformers import Ministral3ForCausalLM as TorchMinistral3ForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(_MODEL_HF_ID, trust_remote_code=False, download_hub="modelscope")
        inputs = tokenizer("The capital of France is", return_tensors="pt")
        input_ids_np = inputs["input_ids"].cpu().numpy()

        hf_model = TorchMinistral3ForCausalLM.from_pretrained(
            _MODEL_HF_ID, dtype=torch.float32, device_map="auto", attn_implementation="eager"
        )
        hf_model.eval()
        with torch.no_grad():
            hf_logits = hf_model(input_ids=inputs["input_ids"].to(hf_model.device)).logits[0, -1].float().cpu().numpy()

        pd_model = Ministral3ForCausalLM.from_pretrained(
            _MODEL_HF_ID,
            dtype="float32",
            load_via_cpu=True,
            download_hub="modelscope",
            load_checkpoint_format="flex_checkpoint",
        )
        pd_model.eval()
        with paddle.no_grad():
            pd_logits = (
                pd_model(paddle.to_tensor(input_ids_np), return_dict=True).logits[0, -1].astype("float32").numpy()
            )

        mean_diff = float(np.mean(np.abs(hf_logits - pd_logits)))
        self.assertLess(mean_diff, 1e-2, f"Logits mean diff {mean_diff:.8f} exceeds threshold 1e-2")
        hf_top10 = np.argsort(hf_logits)[-10:][::-1].tolist()
        pd_top10 = np.argsort(pd_logits)[-10:][::-1].tolist()
        self.assertEqual(hf_top10, pd_top10, "Top 10 tokens differ between HF and Paddle")


if __name__ == "__main__":
    unittest.main()
