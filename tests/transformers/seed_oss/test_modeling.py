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
    SeedOssConfig,
    SeedOssForCausalLM,
    SeedOssForQuestionAnswering,
    SeedOssForSequenceClassification,
    SeedOssForTokenClassification,
    SeedOssModel,
)
from tests.testing_utils import gpu_device_initializer, require_package
from tests.transformers.test_configuration_common import ConfigTester
from tests.transformers.test_modeling_common import (
    ModelTesterMixin,
    ModelTesterPretrainedMixin,
    ids_tensor,
    random_attention_mask,
)


class SeedOssModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        seq_length=5,
        is_training=True,
        use_input_mask=True,
        use_labels=True,
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        attention_bias=True,
        attention_out_bias=False,
        attention_dropout=0.0,
        residual_dropout=0.0,
        mlp_bias=False,
        tie_word_embeddings=False,
        type_sequence_label_size=2,
        num_labels=3,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        return_dict=False,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.attention_out_bias = attention_out_bias
        self.attention_dropout = attention_dropout
        self.residual_dropout = residual_dropout
        self.mlp_bias = mlp_bias
        self.tie_word_embeddings = tie_word_embeddings
        self.type_sequence_label_size = type_sequence_label_size
        self.num_labels = num_labels
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.return_dict = return_dict

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size, dtype=paddle.int64)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        sequence_labels = None
        token_labels = None
        start_positions = None
        end_positions = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size, dtype=paddle.int64)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.vocab_size, dtype=paddle.int64)
            start_positions = ids_tensor([self.batch_size], self.seq_length, dtype=paddle.int64)
            end_positions = ids_tensor([self.batch_size], self.seq_length, dtype=paddle.int64)

        config = self.get_config()
        return config, input_ids, input_mask, sequence_labels, token_labels, start_positions, end_positions

    def get_config(self):
        return SeedOssConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            rms_norm_eps=self.rms_norm_eps,
            attention_bias=self.attention_bias,
            attention_out_bias=self.attention_out_bias,
            attention_dropout=self.attention_dropout,
            residual_dropout=self.residual_dropout,
            mlp_bias=self.mlp_bias,
            tie_word_embeddings=self.tie_word_embeddings,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            use_cache=False,
            _attn_implementation="eager",
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, input_mask, *_ = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict

    def create_and_check_model(
        self, config, input_ids, input_mask, sequence_labels, token_labels, start_positions, end_positions
    ):
        model = SeedOssModel(config)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.hidden_size])

    def create_and_check_model_attention_mask(self, config, input_ids):
        model = SeedOssModel(config)
        model.eval()
        attn_mask_2d = random_attention_mask([self.batch_size, self.seq_length])
        result_2d = model(input_ids, attention_mask=attn_mask_2d)[0]
        batch, seq_length = input_ids.shape
        causal_mask = paddle.tril(paddle.ones((batch, seq_length, seq_length), dtype=attn_mask_2d.dtype))
        attn_mask_3d = causal_mask & attn_mask_2d.unsqueeze(-1)
        result_3d = model(input_ids, attention_mask=attn_mask_3d)[0]
        attn_mask_4d = attn_mask_3d.unsqueeze(1)
        result_4d = model(input_ids, attention_mask=attn_mask_4d)[0]
        result_no_attention_mask = model(input_ids, attention_mask=None)[0]
        self.parent.assertTrue((result_2d[attn_mask_2d] == result_3d[attn_mask_2d]).all())
        self.parent.assertTrue((result_2d[attn_mask_2d] == result_4d[attn_mask_2d]).all())
        self.parent.assertTrue((result_2d[attn_mask_2d] == result_no_attention_mask[attn_mask_2d]).all())

    def create_and_check_for_causal_lm(
        self, config, input_ids, input_mask, sequence_labels, token_labels, start_positions, end_positions
    ):
        model = SeedOssForCausalLM(config)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=token_labels, return_dict=True)
        self.parent.assertEqual(result.logits.shape, [self.batch_size, self.seq_length, self.vocab_size])
        self.parent.assertIsNotNone(result.loss)

    def create_and_check_lm_head_model(self, config, input_ids, input_mask, *args):
        model = SeedOssForCausalLM(config)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            labels=input_ids if self.parent.use_labels else None,
            return_dict=self.parent.return_dict,
            use_cache=False,
        )
        if self.parent.use_labels:
            self.parent.assertIsInstance(result[0].item(), float)
            self.parent.assertEqual(result[1].shape, [self.batch_size, self.seq_length, self.vocab_size])
        else:
            self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.vocab_size])

    def check_model_position_ids(self, config, input_ids, input_mask, *args):
        model = SeedOssForCausalLM(config)
        model.eval()

        result_no_position_id = model(
            input_ids,
            attention_mask=input_mask,
            labels=input_ids if self.parent.use_labels else None,
            return_dict=self.parent.return_dict,
            use_cache=False,
        )
        batch_size, seq_len = input_ids.shape
        position_ids = paddle.arange(seq_len, dtype=paddle.int64).expand((batch_size, seq_len))
        result_position_id = model(
            input_ids,
            attention_mask=input_mask,
            position_ids=position_ids,
            labels=input_ids if self.parent.use_labels else None,
            return_dict=self.parent.return_dict,
            use_cache=False,
        )
        if self.parent.use_labels:
            self.parent.assertTrue((result_position_id[1] == result_no_position_id[1]).all())
        else:
            self.parent.assertTrue((result_position_id[0] == result_no_position_id[0]).all())


class SeedOssModelTest(ModelTesterMixin, unittest.TestCase):
    base_model_class = SeedOssModel
    return_dict = False
    use_labels = False
    use_test_model_name_list = False

    all_model_classes = (SeedOssModel, SeedOssForCausalLM)
    pipeline_model_mapping = {
        "feature-extraction": SeedOssModel,
        "text-classification": SeedOssForSequenceClassification,
        "token-classification": SeedOssForTokenClassification,
        "text-generation": SeedOssForCausalLM,
        "zero-shot": SeedOssForSequenceClassification,
    }

    @gpu_device_initializer(log_prefix="SeedOssModelTest")
    def setUp(self):
        super().setUp()
        self.model_tester = SeedOssModelTester(self)
        self.config_tester = ConfigTester(self, config_class=SeedOssConfig, vocab_size=256, hidden_size=24)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_config_default_rope_parameters(self):
        config = SeedOssConfig()
        self.assertEqual(config.model_type, "seed_oss")
        self.assertEqual(config.rope_parameters["rope_type"], "default")
        self.assertEqual(config.rope_parameters["rope_theta"], 10000000.0)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_attention_mask(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        self.model_tester.create_and_check_model_attention_mask(config, input_dict["input_ids"])

    def test_model_position_ids(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_model_position_ids(*config_and_inputs)

    def test_auto_model_for_causal_lm_from_config(self):
        config = self.model_tester.get_config()
        model = AutoModelForCausalLM.from_config(config)
        self.assertIsInstance(model, SeedOssForCausalLM)

    def test_model_causal_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_causal_lm(*config_and_inputs)

    def test_model_lm_head_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_lm_head_model(*config_and_inputs)

    def test_seed_oss_sequence_classification_model(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = self.model_tester.num_labels
        input_ids = input_dict["input_ids"]
        attention_mask = paddle.not_equal(input_ids, paddle.ones_like(input_ids))
        sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)
        model = SeedOssForSequenceClassification(config)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels, return_dict=True)
        self.assertEqual(result.logits.shape, [self.model_tester.batch_size, self.model_tester.num_labels])
        self.assertIsNotNone(result.loss)

    def test_seed_oss_sequence_classification_model_for_multi_label(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = self.model_tester.num_labels
        input_ids = input_dict["input_ids"]
        attention_mask = paddle.not_equal(input_ids, paddle.ones_like(input_ids))
        sequence_labels = ids_tensor(
            [self.model_tester.batch_size, config.num_labels], self.model_tester.type_sequence_label_size
        ).to(paddle.float32)
        model = SeedOssForSequenceClassification(config)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels, return_dict=True)
        self.assertEqual(result.logits.shape, [self.model_tester.batch_size, self.model_tester.num_labels])
        self.assertIsNotNone(result.loss)

    def test_seed_oss_token_classification_model(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = self.model_tester.num_labels
        input_ids = input_dict["input_ids"]
        attention_mask = paddle.not_equal(input_ids, paddle.ones_like(input_ids))
        token_labels = ids_tensor([self.model_tester.batch_size, self.model_tester.seq_length], config.num_labels)
        model = SeedOssForTokenClassification(config)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=token_labels, return_dict=True)
        self.assertEqual(
            result.logits.shape,
            [self.model_tester.batch_size, self.model_tester.seq_length, self.model_tester.num_labels],
        )
        self.assertIsNotNone(result.loss)

    def test_seed_oss_question_answering_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        config, input_ids, input_mask, _, _, start_positions, end_positions = config_and_inputs
        model = SeedOssForQuestionAnswering(config)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            start_positions=start_positions,
            end_positions=end_positions,
            return_dict=True,
        )
        self.assertEqual(result.start_logits.shape, [self.model_tester.batch_size, self.model_tester.seq_length])
        self.assertEqual(result.end_logits.shape, [self.model_tester.batch_size, self.model_tester.seq_length])
        self.assertIsNotNone(result.loss)

    @unittest.skip("SeedOss uses GQA on all models so the KV cache is a non standard format")
    def test_past_key_values_format(self):
        pass


class SeedOssIntegrationTest(ModelTesterPretrainedMixin, unittest.TestCase):
    base_model_class = SeedOssModel


class SeedOssCompatibilityTest(unittest.TestCase):
    @gpu_device_initializer(log_prefix="SeedOssCompatibilityTest")
    def setUp(self):
        pass

    @classmethod
    @require_package("transformers", "torch")
    def setUpClass(cls) -> None:
        try:
            from transformers import SeedOssConfig, SeedOssForCausalLM
        except ImportError as exc:
            raise unittest.SkipTest("transformers does not provide SeedOss classes") from exc

        cls.torch_model_path = tempfile.TemporaryDirectory().name
        config = SeedOssConfig(
            vocab_size=128,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            attention_dropout=0.0,
            residual_dropout=0.0,
            use_cache=False,
            _attn_implementation="eager",
        )
        model = SeedOssForCausalLM(config)
        model.save_pretrained(cls.torch_model_path)

    @require_package("transformers", "torch")
    def test_seed_oss_converter(self):
        import torch

        try:
            from transformers import SeedOssForCausalLM as HFSeedOssForCausalLM
        except ImportError as exc:
            raise unittest.SkipTest("transformers does not provide SeedOss classes") from exc

        input_ids = np.random.randint(0, 128, [1, 8])

        torch_model = HFSeedOssForCausalLM.from_pretrained(
            self.torch_model_path,
            torch_dtype=torch.float32,
            attn_implementation="eager",
        )
        torch_model.eval()
        with torch.no_grad():
            torch_logit = torch_model(torch.tensor(input_ids), return_dict=False, use_cache=False)[0]

        paddle_model = SeedOssForCausalLM.from_pretrained(
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

    @require_package("transformers", "torch")
    def test_seed_oss_converter_from_local_dir(self):
        import torch

        try:
            from transformers import SeedOssForCausalLM as HFSeedOssForCausalLM
        except ImportError as exc:
            raise unittest.SkipTest("transformers does not provide SeedOss classes") from exc

        with tempfile.TemporaryDirectory() as tempdir:
            input_ids = np.random.randint(0, 128, [1, 8])

            torch_model = HFSeedOssForCausalLM.from_pretrained(
                self.torch_model_path,
                torch_dtype=torch.float32,
                attn_implementation="eager",
            )
            torch_model.eval()
            torch_model.save_pretrained(tempdir)
            with torch.no_grad():
                torch_logit = torch_model(torch.tensor(input_ids), return_dict=False, use_cache=False)[0]

            paddle_model = SeedOssForCausalLM.from_pretrained(
                tempdir,
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
