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

from paddleformers.transformers import (
    AutoModelForCausalLM,
    Olmo2Config,
    Olmo2ForCausalLM,
    Olmo2Model,
)
from tests.testing_utils import gpu_device_initializer, require_package
from tests.transformers.test_configuration_common import ConfigTester
from tests.transformers.test_modeling_common import ModelTesterMixin, ids_tensor


class Olmo2ModelTester:
    def __init__(
        self,
        parent,
        vocab_size=128,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        rms_norm_eps=1e-5,
        hidden_act="silu",
        attention_bias=False,
        attention_dropout=0.0,
        tie_word_embeddings=False,
        pad_token_id=1,
        bos_token_id=None,
        eos_token_id=2,
        batch_size=2,
        seq_length=8,
        is_training=True,
        use_labels=False,
        return_dict=False,
    ):
        self.parent = parent
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.hidden_act = hidden_act
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.tie_word_embeddings = tie_word_embeddings
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_labels = use_labels
        self.return_dict = return_dict

    def get_config(self):
        return Olmo2Config(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            max_position_embeddings=self.max_position_embeddings,
            rms_norm_eps=self.rms_norm_eps,
            hidden_act=self.hidden_act,
            attention_bias=self.attention_bias,
            attention_dropout=self.attention_dropout,
            tie_word_embeddings=self.tie_word_embeddings,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            rope_theta=10000.0,
            use_cache=False,
            _attn_implementation="eager",
        )

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size, dtype=paddle.int64)
        token_labels = ids_tensor([self.batch_size, self.seq_length], self.vocab_size, dtype=paddle.int64)
        config = self.get_config()
        return config, input_ids, token_labels

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, _ = self.prepare_config_and_inputs()
        return config, {"input_ids": input_ids}

    def create_and_check_model(self, config, input_ids, token_labels):
        model = Olmo2Model(config)
        model.eval()
        result = model(input_ids)
        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.hidden_size])

    def create_and_check_for_causal_lm(self, config, input_ids, token_labels):
        model = Olmo2ForCausalLM(config)
        model.eval()
        result = model(input_ids, labels=token_labels, return_dict=True)
        self.parent.assertEqual(result.logits.shape, [self.batch_size, self.seq_length, self.vocab_size])
        self.parent.assertIsNotNone(result.loss)

    def create_and_check_lm_head_model(self, config, input_ids, token_labels):
        model = Olmo2ForCausalLM(config)
        model.eval()
        result = model(
            input_ids,
            labels=input_ids if self.parent.use_labels else None,
            return_dict=self.parent.return_dict,
            use_cache=False,
        )
        if self.parent.use_labels:
            self.parent.assertIsInstance(result[0].item(), float)
            self.parent.assertEqual(result[1].shape, [self.batch_size, self.seq_length, self.vocab_size])
        else:
            self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.vocab_size])

    def check_model_position_ids(self, config, input_ids, token_labels):
        model = Olmo2ForCausalLM(config)
        model.eval()

        result_no_position_id = model(input_ids, return_dict=self.parent.return_dict, use_cache=False)
        batch_size, seq_len = input_ids.shape
        position_ids = paddle.arange(seq_len, dtype=paddle.int64).expand((batch_size, seq_len))
        result_position_id = model(
            input_ids,
            position_ids=position_ids,
            return_dict=self.parent.return_dict,
            use_cache=False,
        )
        self.parent.assertTrue((result_position_id[0] == result_no_position_id[0]).all())


class Olmo2ModelTest(ModelTesterMixin, unittest.TestCase):
    base_model_class = Olmo2Model
    return_dict = False
    use_labels = False
    use_test_model_name_list = False

    all_model_classes = (Olmo2Model, Olmo2ForCausalLM)

    @gpu_device_initializer(log_prefix="Olmo2ModelTest")
    def setUp(self):
        super().setUp()
        self.model_tester = Olmo2ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Olmo2Config, vocab_size=256, hidden_size=24)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_causal_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_causal_lm(*config_and_inputs)

    def test_auto_model_for_causal_lm_from_config(self):
        config = self.model_tester.get_config()
        model = AutoModelForCausalLM.from_config(config)
        self.assertIsInstance(model, Olmo2ForCausalLM)

    def test_model_lm_head_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_lm_head_model(*config_and_inputs)

    def test_model_position_ids(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_model_position_ids(*config_and_inputs)


class Olmo2CompatibilityTest(unittest.TestCase):
    @gpu_device_initializer(log_prefix="Olmo2CompatibilityTest")
    def setUp(self):
        pass

    @classmethod
    @require_package("transformers", "torch")
    def setUpClass(cls) -> None:
        import torch
        from transformers import Olmo2Config as HFOlmo2Config
        from transformers import Olmo2ForCausalLM as HFOlmo2ForCausalLM

        cls.torch_model_path = tempfile.TemporaryDirectory().name
        torch.manual_seed(0)
        config = HFOlmo2Config(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            rms_norm_eps=1e-5,
            attention_bias=False,
            attention_dropout=0.0,
            tie_word_embeddings=False,
            rope_theta=10000.0,
            use_cache=False,
            _attn_implementation="eager",
        )
        model = HFOlmo2ForCausalLM(config)
        model.save_pretrained(cls.torch_model_path)

    @require_package("transformers", "torch")
    def test_Olmo2_converter_from_local_dir(self):
        import torch
        from transformers import Olmo2ForCausalLM as HFOlmo2ForCausalLM

        input_ids = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int64)

        torch_model = HFOlmo2ForCausalLM.from_pretrained(
            self.torch_model_path, torch_dtype=torch.float32, attn_implementation="eager"
        )
        torch_model.eval()
        with torch.no_grad():
            torch_logit = torch_model(torch.tensor(input_ids), return_dict=False, use_cache=False)[0]

        paddle_model = Olmo2ForCausalLM.from_pretrained(
            self.torch_model_path,
            dtype="float32",
            load_checkpoint_format="flex_checkpoint",
        )
        paddle_model.eval()
        paddle_model.config._attn_implementation = "eager"
        with paddle.no_grad():
            paddle_logit = paddle_model(paddle.to_tensor(input_ids), use_cache=False)[0]

        self.assertTrue(
            np.allclose(
                paddle_logit.detach().cpu().reshape([-1])[:16].astype("float32").numpy(),
                torch_logit.detach().cpu().reshape([-1])[:16].float().numpy(),
                atol=1e-4,
                rtol=1e-4,
            )
        )


if __name__ == "__main__":
    unittest.main()
