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
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import paddle

from paddleformers.datasets.collate import mm_collate_fn
from paddleformers.datasets.SFTDataset import Sequence
from paddleformers.nn.attention.interface import ALL_ATTENTION_FUNCTIONS
from paddleformers.nn.mlp import MLP
from paddleformers.transformers import Mistral3Config, Mistral3ForConditionalGeneration
from paddleformers.transformers.pixtral.configuration import PixtralVisionConfig
from paddleformers.transformers.pixtral.modeling import (
    PixtralAttention,
    PixtralAttentionLayer,
    apply_rotary_pos_emb,
)

from .test_modeling import Mistral3ModelTester


class Mistral3ReviewTest(unittest.TestCase):
    def test_pixtral_uses_common_mlp_without_changing_math_or_keys(self):
        for activation in ("gelu", "silu"):
            config = PixtralVisionConfig(
                hidden_size=16, intermediate_size=23, num_attention_heads=4, hidden_act=activation, fuse_rms_norm=False
            )
            mlp = PixtralAttentionLayer(config).feed_forward
            self.assertIsInstance(mlp, MLP)
            self.assertEqual(set(mlp.state_dict()), {"gate_proj.weight", "up_proj.weight", "down_proj.weight"})
            x = paddle.randn([1, 3, 16])
            gate = paddle.matmul(x, mlp.gate_proj.weight)
            activated = getattr(paddle.nn.functional, activation)(gate)
            expected = paddle.matmul(activated * paddle.matmul(x, mlp.up_proj.weight), mlp.down_proj.weight)
            np.testing.assert_allclose(mlp(x).numpy(), expected.numpy(), atol=1e-6, rtol=1e-6)

    def test_tied_head_is_shared_and_checkpoint_mapping_complete(self):
        config = Mistral3ModelTester(self).get_config()
        config.tie_word_embeddings = True
        config.text_config.tie_word_embeddings = True
        model = Mistral3ForConditionalGeneration(config).eval()
        self.assertIs(model.lm_head.weight, model.get_input_embeddings().weight)
        statements = model._gen_aoa_config(config)["aoa_statements"]
        self.assertIn("language_model.model.embed_tokens.weight -> lm_head.weight", statements)
        inverse = model._gen_inv_aoa_config(config)["aoa_statements"]
        self.assertEqual(sum(s.endswith(" -> language_model.model.embed_tokens.weight") for s in inverse), 1)
        ids = paddle.to_tensor([[2, 4, 5]], dtype="int64")
        with paddle.no_grad():
            hidden = model.model(input_ids=ids, return_dict=True).last_hidden_state
            expected = paddle.matmul(hidden, model.get_input_embeddings().weight, transpose_y=True)
            np.testing.assert_allclose(model(input_ids=ids).logits.numpy(), expected.numpy(), atol=1e-6)
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            reloaded = Mistral3ForConditionalGeneration.from_pretrained(directory).eval()
            self.assertIs(reloaded.lm_head.weight, reloaded.get_input_embeddings().weight)
            np.testing.assert_allclose(reloaded(input_ids=ids).logits.numpy(), expected.numpy(), atol=1e-6)

    def test_text_config_dispatch_round_trip(self):
        config = Mistral3ModelTester(self).get_config()
        values = config.to_dict()
        values["text_config"]["model_type"] = "qwen2"
        actual = Mistral3Config(**values)
        self.assertEqual(type(actual.text_config).__name__, "Qwen2Config")
        self.assertEqual(actual.text_config.model_type, "qwen2")
        with tempfile.TemporaryDirectory() as directory:
            actual.save_pretrained(directory)
            loaded = Mistral3Config.from_pretrained(directory)
            self.assertEqual(loaded.text_config.model_type, "qwen2")
        model = Mistral3ForConditionalGeneration(actual)
        self.assertEqual(type(model.model.language_model).__name__, "Qwen2Model")
        statements = model._gen_aoa_config(actual)["aoa_statements"]
        self.assertTrue(any("model.language_model.layers.$LAYER_ID.self_attn.qkv_proj.bias" in s for s in statements))
        logits = model(input_ids=paddle.to_tensor([[2, 4, 5]], dtype="int64")).logits
        self.assertEqual(list(logits.shape), [1, 3, actual.text_config.vocab_size])
        values["text_config"]["model_type"] = "not_a_registered_decoder"
        with self.assertRaises(KeyError):
            Mistral3Config(**values)

    def test_pixtral_attention_matches_original_and_dispatches(self):
        config = PixtralVisionConfig(hidden_size=16, num_attention_heads=4, _attn_implementation="eager")
        layer = PixtralAttention(config).eval()
        x = paddle.randn([1, 4, 16])
        cos, sin = paddle.cos(paddle.randn([4, 4])), paddle.sin(paddle.randn([4, 4]))
        mask = paddle.zeros([1, 1, 4, 4])
        mask[:, :, :2, 2:] = -1e4
        mask[:, :, 2:, :2] = -1e4
        q, k, v = [
            projection(x).reshape([1, 4, 4, 4]).transpose([0, 2, 1, 3])
            for projection in (layer.q_proj, layer.k_proj, layer.v_proj)
        ]
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=0)
        weights = paddle.nn.functional.softmax(
            paddle.matmul(q, k.transpose([0, 1, 3, 2])) * layer.scaling + mask, axis=-1, dtype="float32"
        )
        expected = layer.o_proj(paddle.matmul(weights, v).transpose([0, 2, 1, 3]).reshape([1, 4, 16]))
        actual, _ = layer(x, attention_mask=mask, position_embeddings=(cos, sin))
        np.testing.assert_allclose(actual.numpy(), expected.numpy(), atol=1e-6, rtol=1e-6)
        backend = ALL_ATTENTION_FUNCTIONS["eager"]
        config._attn_implementation = "sdpa"
        # Verify dispatch and noncausal mask contract independently of CUDA.
        with patch.object(type(ALL_ATTENTION_FUNCTIONS), "__getitem__", return_value=backend) as dispatch:
            layer(x, attention_mask=mask, position_embeddings=(cos, sin))
            dispatch.assert_called_once_with("sdpa")
        self.assertFalse(layer.is_causal)

    def test_image_sizes_collected_once_for_packed_and_padded_batches(self):
        dimensions = [(24, 24), (24, 48), (48, 24)]
        samples = [
            Sequence(
                token_ids=[2, 4],
                position_ids=[0, 1],
                labels=[2, 4],
                num_examples=1,
                mm_inputs={
                    "image_sizes": paddle.to_tensor([[height, width]], dtype="int64"),
                    "pixel_values": paddle.full([1, 3, height, width], index + 1, dtype="float32"),
                },
            )
            for index, (height, width) in enumerate(dimensions)
        ]
        training = SimpleNamespace(
            num_nextn_predict_layers=0,
            context_parallel_size=1,
            tensor_model_parallel_size=1,
            sequence_parallel=False,
            fp8=False,
        )
        model_args = SimpleNamespace(use_attn_mask_startend_row_indices=False, use_global_causal_attn=False)
        for packed in (False, True):
            result = mm_collate_fn(
                [samples[:2], samples[2:]] if packed else [[sample] for sample in samples],
                template=None,
                processor=None,
                tokenizer=SimpleNamespace(pad_token_id=0),
                training_args=training,
                model_args=model_args,
                max_seq_len=4,
                padding_free=packed,
                model=None,
            )
            np.testing.assert_array_equal(result["image_sizes"].numpy(), dimensions)
            expected = np.zeros([3, 3, 48, 48], dtype="float32")
            for index, (height, width) in enumerate(dimensions):
                expected[index, :, :height, :width] = index + 1
            np.testing.assert_array_equal(result["pixel_values"].numpy(), expected)
