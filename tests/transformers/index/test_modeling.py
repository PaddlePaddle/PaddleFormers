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

import tempfile
import unittest
from unittest import mock

import paddle

from paddleformers.transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    IndexConfig,
    IndexForCausalLM,
    IndexModel,
)
from paddleformers.transformers.index.modeling import NormHead


class IndexModelTest(unittest.TestCase):
    def setUp(self):
        self.config = IndexConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=32,
            pad_token_id=0,
            eos_token_id=2,
        )
        self.input_ids = paddle.to_tensor([[1, 4, 5, 2], [1, 6, 0, 0]], dtype="int64")
        self.mask = paddle.to_tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype="int64")

    def test_config_and_auto_routing(self):
        config = IndexConfig(norm_head=1, rope_scaling={"type": "linear", "factor": 2.0})
        self.assertEqual(config.model_type, "index")
        self.assertEqual(config.max_length, 4096)
        self.assertTrue(config.norm_head)
        with tempfile.TemporaryDirectory() as directory:
            IndexForCausalLM(self.config).save_pretrained(directory)
            self.assertIsInstance(AutoConfig.from_pretrained(directory), IndexConfig)
            self.assertIsInstance(AutoModel.from_pretrained(directory), IndexModel)
            self.assertIsInstance(AutoModelForCausalLM.from_pretrained(directory), IndexForCausalLM)

    def test_forward_loss_mask_and_ignore_index(self):
        model = IndexForCausalLM(self.config)
        result = model(self.input_ids, labels=self.input_ids, return_dict=True)
        self.assertEqual(result.logits.shape, [2, 4, 32])
        self.assertTrue(paddle.isfinite(result.loss))
        loss_mask = paddle.to_tensor([[0, 1, 0, 0], [0, 0, 0, 0]], dtype="int64")
        masked = model(self.input_ids, labels=self.input_ids, loss_mask=loss_mask, return_dict=True)
        expected = paddle.nn.functional.cross_entropy(
            masked.logits[0, 0, :].unsqueeze(0), self.input_ids[0, 1].unsqueeze(0)
        )
        self.assertTrue(paddle.allclose(masked.loss, expected))
        empty = model(
            self.input_ids,
            labels=self.input_ids,
            loss_mask=paddle.zeros_like(self.input_ids),
            return_dict=True,
        )
        self.assertTrue(paddle.isfinite(empty.loss))
        self.assertEqual(empty.loss.item(), 0.0)
        labels = paddle.full_like(self.input_ids, -100)
        self.assertTrue(paddle.isfinite(model(self.input_ids, labels=labels, return_dict=True).loss))

    def test_padding_and_causal_masks_match(self):
        model = IndexModel(self.config)
        model.eval()
        result_2d = model(self.input_ids, attention_mask=self.mask)[0]
        causal = paddle.tril(paddle.ones([2, 4, 4], dtype="int64")) * self.mask.unsqueeze(1)
        result_4d = model(self.input_ids, attention_mask=causal.unsqueeze(1))[0]
        self.assertTrue(paddle.allclose(result_2d[self.mask.astype("bool")], result_4d[self.mask.astype("bool")]))

    def test_cache_return_dict_false_and_incremental(self):
        model = IndexForCausalLM(self.config)
        model.eval()
        base_outputs = model.model(self.input_ids[:, :3], use_cache=True, return_dict=False)
        self.assertIsNotNone(base_outputs[1])
        full = model(self.input_ids[:, :3], use_cache=True, return_dict=False)
        cached = model(self.input_ids[:, 3:], past_key_values=full[1], use_cache=True, return_dict=False)
        direct = model(self.input_ids, use_cache=False, return_dict=False)
        self.assertTrue(paddle.allclose(cached[0], direct[0][:, -1:], atol=1e-5))
        self.assertIsNotNone(cached[1])

    def test_aoa_lm_head_layout(self):
        normal = IndexForCausalLM._gen_aoa_config(self.config)["aoa_statements"]
        normal_inverse = IndexForCausalLM._gen_inv_aoa_config(self.config)["aoa_statements"]
        self.assertIn("lm_head.weight -> lm_head.weight", normal)
        self.assertIn("lm_head.weight -> lm_head.weight", normal_inverse)
        norm_config = IndexConfig(norm_head=True)
        norm = IndexForCausalLM._gen_aoa_config(norm_config)["aoa_statements"]
        norm_inverse = IndexForCausalLM._gen_inv_aoa_config(norm_config)["aoa_statements"]
        self.assertIn("lm_head.weight -> lm_head.weight", norm)
        self.assertIn("lm_head.weight -> lm_head.weight", norm_inverse)

    def test_norm_head_and_embeddings(self):
        head = NormHead(4, 8)
        hidden = paddle.randn([2, 3, 4])
        head.train()
        self.assertEqual(head(hidden).shape, [2, 3, 8])
        head.eval()
        head(hidden)
        self.assertFalse(head.first_flag)
        model = IndexForCausalLM(
            IndexConfig(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=4,
                norm_head=True,
            )
        )
        self.assertIs(model.get_output_embeddings(), model.lm_head)
        self.assertIs(model.get_input_embeddings(), model.model.embed_tokens)

        tied_config = IndexConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            tie_word_embeddings=True,
        )
        tied_model = IndexForCausalLM(tied_config)
        self.assertIs(tied_model.lm_head.weight, tied_model.model.embed_tokens.weight)
        self.assertEqual(tied_model(self.input_ids, return_dict=True).logits.shape, [2, 4, 32])

    def test_attn_mask_start_row_indices_alias(self):
        model = IndexForCausalLM(self.config)
        indices = paddle.zeros([2, 1, 4, 1], dtype="int32")
        with mock.patch.object(model.model, "forward", wraps=model.model.forward) as model_forward:
            model(self.input_ids, attn_mask_start_row_indices=indices)
        self.assertIs(model_forward.call_args.kwargs["attn_mask_startend_row_indices"], indices)

    def test_save_load_and_generation(self):
        model = IndexForCausalLM(self.config)
        model.eval()
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            loaded = IndexForCausalLM.from_pretrained(directory)
            self.assertTrue(paddle.allclose(model(self.input_ids)[0], loaded(self.input_ids)[0]))
        generated = model.generate(self.input_ids[:1, :2], max_new_tokens=2, use_cache=True)
        self.assertEqual(generated[0].shape[0], 1)
        self.assertGreaterEqual(generated[0].shape[1], 1)
        self.assertLessEqual(generated[0].shape[1], 2)


if __name__ == "__main__":
    unittest.main()
