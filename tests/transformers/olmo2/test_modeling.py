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

"""OLMo2 functional, HF compatibility, QK-norm and TP mapping regression tests.

Covers the load path exercised when sharding a non-TP checkpoint onto
``tensor_model_parallel_size > 1``. The lm_head is vocab-parallel: its weight
has shape ``[vocab_size / tp, hidden_size]`` with ``split_axis = 0``, so the
split action must slice the vocabulary dimension (``is_column=False``), not the
hidden dimension. This test guards against that regression.
"""

import tempfile
import unittest

import numpy as np
import paddle

from paddleformers.transformers import (
    AutoModelForCausalLM,
    Olmo2Config,
    Olmo2ForCausalLM,
    Olmo2Model,
    Olmo3Config,
)
from paddleformers.transformers.olmo2.modeling import Olmo2Attention
from paddleformers.transformers.olmo3.modeling import Olmo3Attention
from tests.testing_utils import gpu_device_initializer, require_package
from tests.transformers.test_configuration_common import ConfigTester
from tests.transformers.test_modeling_common import ModelTesterMixin, ids_tensor


def _make_tp_config(tp_size, tensor_parallel_rank):
    config = Olmo2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
        tie_word_embeddings=False,
        dtype="float32",
    )
    config.tensor_model_parallel_size = tp_size
    config.tensor_parallel_rank = tensor_parallel_rank
    config.tensor_parallel_output = True
    config.sequence_parallel = False
    return config


def _get_split_actions(config):
    return Olmo2ForCausalLM._get_tensor_parallel_mappings(config, is_split=True)


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

    def test_head_dim_config_roundtrip_and_forward(self):
        for head_dim in (None, 4):
            with self.subTest(head_dim=head_dim):
                config = self.model_tester.get_config()
                config = Olmo2Config.from_dict({**config.to_dict(), "head_dim": head_dim})
                expected = 8 if head_dim is None else head_dim
                self.assertEqual(config.head_dim, expected)
                restored = Olmo2Config.from_dict(config.to_dict())
                self.assertEqual(restored.head_dim, expected)
                attention = Olmo2Attention(restored, layer_idx=0)
                self.assertEqual(list(attention.q_proj.weight.shape), [32, 4 * expected])
                self.assertEqual(list(attention.k_proj.weight.shape), [32, 2 * expected])
                self.model_tester.create_and_check_model(
                    restored,
                    ids_tensor([self.model_tester.batch_size, self.model_tester.seq_length], config.vocab_size),
                    None,
                )

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

        cls.torch_model_dir = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.torch_model_dir.cleanup)
        cls.torch_model_path = cls.torch_model_dir.name
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


class OlmoQKNormTest(unittest.TestCase):
    def test_tensor_parallel_attention_rejected(self):
        for config_cls, attention_cls in ((Olmo2Config, Olmo2Attention), (Olmo3Config, Olmo3Attention)):
            for kv_heads in (4, 2):
                for sequence_parallel in (False, True):
                    with self.subTest(model=config_cls.__name__, kv_heads=kv_heads, sp=sequence_parallel):
                        config = config_cls(**_make_tp_config(2, 0).to_dict())
                        config.num_key_value_heads = kv_heads
                        config.sequence_parallel = sequence_parallel
                        with self.assertRaisesRegex(NotImplementedError, "set tensor_model_parallel_size=1"):
                            attention_cls(config, layer_idx=0)

    def test_single_rank_norm_spans_full_projection(self):
        for config_cls, attention_cls in ((Olmo2Config, Olmo2Attention), (Olmo3Config, Olmo3Attention)):
            for kv_heads in (4, 2):
                with self.subTest(model=config_cls.__name__, kv_heads=kv_heads):
                    config = config_cls(**_make_tp_config(1, 0).to_dict())
                    config.num_key_value_heads = kv_heads
                    config.fuse_rms_norm = False
                    attention = attention_cls(config, layer_idx=0)
                    for norm, heads in ((attention.q_norm, 4), (attention.k_norm, kv_heads)):
                        width = heads * attention.head_dim
                        values = np.arange(1, 2 * width + 1, dtype=np.float32).reshape(1, 2, width)
                        weights = np.linspace(0.5, 1.5, width, dtype=np.float32)
                        norm.weight.set_value(weights)
                        expected = values / np.sqrt(np.mean(values**2, axis=-1, keepdims=True) + config.rms_norm_eps)
                        expected *= weights
                        np.testing.assert_allclose(norm(paddle.to_tensor(values)).numpy(), expected, rtol=1e-6, atol=1e-6)

    def test_norm_weights_are_not_split(self):
        for config_cls, attention_cls in ((Olmo2Config, Olmo2Attention), (Olmo3Config, Olmo3Attention)):
            config = config_cls(**_make_tp_config(2, 0).to_dict())
            actions = _get_split_actions(config)
            self.assertNotIn("model.layers.0.self_attn.q_norm.weight", actions)
            self.assertNotIn("model.layers.0.self_attn.k_norm.weight", actions)


class Olmo2LmHeadTPSplitTest(unittest.TestCase):
    """Verify lm_head.weight is split along the vocab dimension for TP."""

    def test_lm_head_weight_split_along_vocab(self):
        tp_size = 2
        vocab_size = 64
        hidden_size = 32

        full_weight = np.arange(vocab_size * hidden_size, dtype=np.float32).reshape(vocab_size, hidden_size)

        shards = []
        for rank in range(tp_size):
            config = _make_tp_config(tp_size, rank)
            actions = _get_split_actions(config)
            self.assertIn("lm_head.weight", actions)
            action = actions["lm_head.weight"]

            shard = action(full_weight)
            # vocab-parallel shard must keep hidden_size intact and halve vocab.
            self.assertEqual(
                list(shard.shape),
                [vocab_size // tp_size, hidden_size],
                f"rank {rank}: expected vocab-parallel shard "
                f"[{vocab_size // tp_size}, {hidden_size}], "
                f"got {list(shard.shape)}",
            )
            shards.append(np.asarray(shard))

        merged = np.concatenate(shards, axis=0)
        np.testing.assert_allclose(merged, full_weight)

    def test_lm_head_shard_matches_param_shape(self):
        """Shards produced by the split action must match LMHead's param shape."""
        from unittest.mock import MagicMock

        from paddleformers.nn.lm_head import LMHead

        tp_size = 2
        vocab_size = 64
        hidden_size = 32

        full_weight = np.arange(vocab_size * hidden_size, dtype=np.float32).reshape(vocab_size, hidden_size)

        for rank in range(tp_size):
            tp_config = _make_tp_config(tp_size, rank)
            actions = _get_split_actions(tp_config)
            shard = np.asarray(actions["lm_head.weight"](full_weight))

            head_config = MagicMock()
            head_config.vocab_size = vocab_size
            head_config.hidden_size = hidden_size
            head_config.tensor_model_parallel_size = tp_size
            head_config.lm_head_bias = False
            head_config.use_fused_head_and_loss_fn = False
            head_config.sequence_parallel = False
            head_config.max_sequence_length = 128
            head_config.tensor_parallel_output = False
            head_config.get = lambda key, default=None: default
            head = LMHead(head_config)

            self.assertEqual(
                list(head.weight.shape),
                list(shard.shape),
                f"rank {rank}: LMHead param {list(head.weight.shape)} "
                f"does not match split shard "
                f"{list(shard.shape)}",
            )


if __name__ == "__main__":
    unittest.main()
