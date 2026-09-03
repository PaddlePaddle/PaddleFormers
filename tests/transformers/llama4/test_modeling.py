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

import numpy as np
import paddle

from paddleformers.cli.utils import get_lora_target_modules
from paddleformers.transformers import (
    AutoModel,
    Llama4ForCausalLM,
    Llama4TextConfig,
    Llama4TextModel,
)
from paddleformers.transformers.llama4.modeling import Llama4TextExperts
from tests.testing_utils import gpu_device_initializer, require_package
from tests.transformers.test_configuration_common import ConfigTester
from tests.transformers.test_modeling_common import (
    ModelTesterMixin,
    ids_tensor,
    random_attention_mask,
)


class Llama4ModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        seq_length=8,
        use_input_mask=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        intermediate_size=16,
        intermediate_size_mlp=37,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_local_experts=4,
        num_experts_per_tok=1,
        max_position_embeddings=128,
        attention_chunk_size=16,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        is_training=False,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.intermediate_size_mlp = intermediate_size_mlp
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_local_experts = num_local_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.max_position_embeddings = max_position_embeddings
        self.attention_chunk_size = attention_chunk_size
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.is_training = is_training

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size, dtype=paddle.int64)
        input_mask = random_attention_mask([self.batch_size, self.seq_length]) if self.use_input_mask else None
        token_labels = (
            ids_tensor([self.batch_size, self.seq_length], self.vocab_size, dtype=paddle.int64)
            if self.use_labels
            else None
        )
        config = self.get_config()
        return config, input_ids, input_mask, token_labels

    def get_config(self):
        return Llama4TextConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            intermediate_size_mlp=self.intermediate_size_mlp,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            num_local_experts=self.num_local_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            max_position_embeddings=self.max_position_embeddings,
            attention_chunk_size=self.attention_chunk_size,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            tie_word_embeddings=False,
            attention_dropout=0.0,
            router_jitter_noise=0.0,
            no_rope_layer_interval=4,
            _attn_implementation="eager",
        )

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, input_mask, _ = self.prepare_config_and_inputs()
        return config, {"input_ids": input_ids, "attention_mask": input_mask}

    def create_and_check_model(self, config, input_ids, input_mask, token_labels):
        model = Llama4TextModel(config)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.hidden_size])

    def create_and_check_for_causal_lm(self, config, input_ids, input_mask, token_labels):
        model = Llama4ForCausalLM(config)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=token_labels, return_dict=True)
        self.parent.assertEqual(result.logits.shape, [self.batch_size, self.seq_length, self.vocab_size])
        self.parent.assertIsInstance(result.loss.item(), float)

    def create_and_check_lm_head_model(self, config, input_ids, input_mask, token_labels):
        model = Llama4ForCausalLM(config)
        model.eval()
        result = model(input_ids, labels=input_ids, return_dict=self.parent.return_dict)
        if self.parent.return_dict:
            self.parent.assertEqual(result.logits.shape, [self.batch_size, self.seq_length, self.vocab_size])
        else:
            self.parent.assertIsInstance(result[0].item(), float)
            self.parent.assertEqual(result[1].shape, [self.batch_size, self.seq_length, self.vocab_size])

    def check_model_position_ids(self, config, input_ids, input_mask, token_labels):
        model = Llama4ForCausalLM(config)
        model.eval()
        batch_size, seq_len = input_ids.shape
        position_ids = paddle.arange(seq_len, dtype=paddle.int64).expand((batch_size, seq_len))

        result_no_position_id = model(input_ids, labels=token_labels, return_dict=True)
        result_position_id = model(input_ids, position_ids=position_ids, labels=token_labels, return_dict=True)
        self.parent.assertTrue(paddle.allclose(result_position_id.logits, result_no_position_id.logits))

    def create_and_check_model_attention_mask(self, config, input_ids):
        model = Llama4TextModel(config)
        model.eval()
        attention_mask = random_attention_mask([self.batch_size, self.seq_length])
        result_with_mask = model(input_ids, attention_mask=attention_mask)[0]
        result_without_mask = model(input_ids, attention_mask=None)[0]
        self.parent.assertEqual(result_with_mask.shape, result_without_mask.shape)


class Llama4ModelTest(ModelTesterMixin, unittest.TestCase):
    base_model_class = Llama4TextModel
    return_dict = False
    use_labels = False
    use_test_model_name_list = False

    all_model_classes = (Llama4TextModel, Llama4ForCausalLM)
    all_generative_model_classes = {Llama4ForCausalLM: {Llama4TextModel, "llama4_text"}}

    @gpu_device_initializer(log_prefix="Llama4ModelTest")
    def setUp(self):
        super().setUp()
        self.model_tester = Llama4ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Llama4TextConfig, hidden_size=37)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_attention_mask(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        self.model_tester.create_and_check_model_attention_mask(config, input_dict["input_ids"])

    def test_model_position_ids(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_model_position_ids(*config_and_inputs)

    def test_model_lm_head_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_lm_head_model(*config_and_inputs)

    def test_model_causal_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_causal_lm(*config_and_inputs)

    def test_causal_lm_uses_config_output_hidden_states(self):
        config, input_ids, input_mask, _ = self.model_tester.prepare_config_and_inputs()
        config.output_hidden_states = True
        model = Llama4ForCausalLM(config)
        model.eval()

        with paddle.no_grad():
            result = model(input_ids, attention_mask=input_mask, return_dict=True)

        self.assertEqual(len(result.hidden_states), config.num_hidden_layers + 1)

    def test_expert_parameters_are_trainable_from_initialization(self):
        config = self.model_tester.get_config()
        experts = Llama4TextExperts(config)

        self.assertGreater(paddle.count_nonzero(experts.gate_up_proj).item(), 0)
        self.assertGreater(paddle.count_nonzero(experts.down_proj).item(), 0)

        hidden_states = paddle.randn([config.num_local_experts * 2, config.hidden_size])
        experts(hidden_states).sum().backward()
        self.assertGreater(paddle.count_nonzero(experts.gate_up_proj.grad).item(), 0)

    def test_auto_model_mapping(self):
        config = self.model_tester.get_config()
        model = AutoModel.from_config(config)
        self.assertIsInstance(model, Llama4TextModel)

        config_dict = config.to_dict()
        config_dict["architectures"] = ["Llama4ForCausalLM"]
        inferred_model_class = AutoModel._get_model_class_from_config(None, None, config_dict)
        self.assertIs(inferred_model_class, Llama4TextModel)

    def test_lora_target_modules(self):
        model = Llama4ForCausalLM(self.model_tester.get_config())
        target_modules = get_lora_target_modules(model)
        self.assertIn(".*q_proj.*", target_modules)
        self.assertIn(".*gate_proj.*", target_modules)
        self.assertIn(".*down_proj.*", target_modules)

    def test_chunked_mask_uses_local_cache_and_left_padding(self):
        config = self.model_tester.get_config()
        config.attention_chunk_size = 4
        model = Llama4TextModel(config)

        cached_mask = model._create_chunked_causal_mask(1, 1, 6, paddle.float32, 4)
        self.assertEqual((cached_mask == 0).reshape([-1]).tolist(), [False, True, True, True])

        attention_mask = paddle.to_tensor([[0, 0, 1, 1, 1, 1]], dtype="int64")
        padded_mask = model._create_chunked_causal_mask(1, 6, 0, paddle.float32, 4, attention_mask=attention_mask)
        self.assertEqual(
            (padded_mask == 0).squeeze(0).squeeze(0).tolist(),
            [
                [False, False, False, False, False, False],
                [False, False, False, False, False, False],
                [False, False, True, False, False, False],
                [False, False, True, True, False, False],
                [False, False, True, True, True, False],
                [False, False, True, True, True, True],
            ],
        )

    def test_chunked_attention_keeps_padding_mask_with_row_indices(self):
        config = self.model_tester.get_config()
        config.num_hidden_layers = 1
        config.moe_layers = [0]
        config.no_rope_layers = [1]
        config.layer_types = ["chunked_attention"]
        config.attention_chunk_size = 4
        model = Llama4ForCausalLM(config)
        model.eval()

        input_ids = ids_tensor([1, 6], config.vocab_size, dtype=paddle.int64)
        attention_mask = paddle.to_tensor([[0, 0, 1, 1, 1, 1]], dtype="int64")
        row_indices = paddle.full([1, 1, 6, 1], 6, dtype="int32")

        with paddle.no_grad():
            expected = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            ).logits
            actual = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                attn_mask_startend_row_indices=row_indices,
                return_dict=True,
            ).logits

        self.assertTrue(paddle.allclose(actual, expected, atol=1e-6, rtol=1e-6))

    def test_chunked_cached_decode_matches_full_forward(self):
        config = self.model_tester.get_config()
        config.num_hidden_layers = 1
        config.moe_layers = [0]
        config.no_rope_layers = [1]
        config.layer_types = ["chunked_attention"]
        config.attention_chunk_size = 4
        model = Llama4TextModel(config)
        model.eval()

        input_ids = ids_tensor([1, 7], config.vocab_size, dtype=paddle.int64)
        attention_mask = paddle.ones([1, 7], dtype="int64")
        with paddle.no_grad():
            full_output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            ).last_hidden_state[:, -1]
            prefill_output = model(
                input_ids=input_ids[:, :-1],
                attention_mask=attention_mask[:, :-1],
                use_cache=True,
                return_dict=True,
            )
            cached_output = model(
                input_ids=input_ids[:, -1:],
                attention_mask=attention_mask,
                past_key_values=prefill_output.past_key_values,
                use_cache=True,
                return_dict=True,
            ).last_hidden_state[:, -1]

        self.assertTrue(paddle.allclose(cached_output, full_output, atol=1e-5, rtol=1e-5))

    @unittest.skip("Llama4 text model uses grouped KV cache; common cache format test is not applicable yet.")
    def test_past_key_values_format(self):
        pass


class Llama4CompatibilityTest(unittest.TestCase):
    @gpu_device_initializer(log_prefix="Llama4CompatibilityTest")
    def setUp(self):
        pass

    @classmethod
    @require_package("transformers", "torch")
    def setUpClass(cls):
        try:
            import torch  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("PyTorch is not available in the Paddle test environment.")

        from transformers.utils import is_torch_available

        if not is_torch_available():
            raise unittest.SkipTest("Transformers reports PyTorch as unavailable in this environment.")

        from transformers import Llama4ForCausalLM, Llama4TextConfig

        cls.torch_model_dir = tempfile.TemporaryDirectory()
        cls.torch_model_path = cls.torch_model_dir.name
        config = Llama4TextConfig(
            vocab_size=99,
            hidden_size=32,
            intermediate_size=16,
            intermediate_size_mlp=37,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            num_local_experts=4,
            num_experts_per_tok=1,
            max_position_embeddings=128,
            attention_chunk_size=16,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            no_rope_layer_interval=4,
            attn_implementation="eager",
        )
        model = Llama4ForCausalLM(config)
        model.save_pretrained(cls.torch_model_path)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "torch_model_dir"):
            cls.torch_model_dir.cleanup()

    @require_package("transformers", "torch")
    def test_llama4_converter_from_local_dir(self):
        import torch
        from transformers import Llama4ForCausalLM as TorchLlama4ForCausalLM

        input_ids = np.random.randint(3, 90, [1, 8])

        torch_model = TorchLlama4ForCausalLM.from_pretrained(
            self.torch_model_path,
            torch_dtype=torch.float32,
            attn_implementation="eager",
        )
        torch_model.eval()
        with torch.no_grad():
            torch_logits = torch_model(torch.tensor(input_ids), return_dict=False)[0]

        paddle_model = Llama4ForCausalLM.from_pretrained(
            self.torch_model_path,
            dtype="float32",
            load_checkpoint_format="flex_checkpoint",
        )
        paddle_model.eval()
        with paddle.no_grad():
            paddle_logits = paddle_model(paddle.to_tensor(input_ids), return_dict=False)[0]

        self.assertTrue(
            np.allclose(
                paddle_logits.detach().cpu().reshape([-1])[:9].astype("float32").numpy(),
                torch_logits.detach().cpu().reshape([-1])[:9].float().numpy(),
                atol=1e-2,
                rtol=1e-2,
            )
        )
