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

"""Unit tests for OLMo3 model."""

from __future__ import annotations

import tempfile
import unittest

import numpy as np
import paddle

from paddleformers.transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Olmo3Config,
    Olmo3ForCausalLM,
    Olmo3ForCausalLMPipe,
    Olmo3Model,
)
from paddleformers.utils import device_guard
from tests.testing_utils import require_gpu, require_package, slow
from tests.transformers.test_modeling_common import (
    ModelTesterMixin,
    ids_tensor,
    random_attention_mask,
)


# Base test class, subsequent tests will call this base class
class Olmo3ModelTester:
    def __init__(
        self,
        parent,
        vocab_size=50304,
        hidden_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        intermediate_size=1024,
        hidden_act="silu",
        max_position_embeddings=512,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=1,
        bos_token_id=None,
        eos_token_id=50279,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        sliding_window=256,
        layer_types=None,
        is_training=True,
        dtype="bfloat16",
        batch_size=2,
        seq_length=10,
        use_input_mask=False,
        use_labels=False,
        return_dict=False,
    ):
        self.parent = parent
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.sliding_window = sliding_window
        self.layer_types = layer_types
        self.is_training = is_training
        self.dtype = dtype
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.return_dict = return_dict

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size, dtype=paddle.int64)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        config = self.get_config()
        return config, input_ids, input_mask

    def get_config(self) -> Olmo3Config:
        return Olmo3Config(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
            rms_norm_eps=self.rms_norm_eps,
            use_cache=self.use_cache,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            tie_word_embeddings=self.tie_word_embeddings,
            rope_theta=self.rope_theta,
            attention_bias=self.attention_bias,
            attention_dropout=self.attention_dropout,
            sliding_window=self.sliding_window,
            layer_types=self.layer_types,
            dtype=self.dtype,
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, input_mask = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict

    def create_and_check_model(self, config, input_ids, input_mask):
        model = Olmo3Model(config)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.hidden_size])

    def create_and_check_lm_head_model(self, config, input_ids, input_mask):
        model = Olmo3ForCausalLM(config)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, return_dict=self.return_dict)
        if self.return_dict:
            self.parent.assertEqual(result.logits.shape, [self.batch_size, self.seq_length, self.vocab_size])
        else:
            self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.vocab_size])


# Basic config test class, tests whether the basic configuration is correct
class Olmo3ModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (Olmo3Model, Olmo3ForCausalLM)

    def setUp(self):
        self.model_tester = Olmo3ModelTester(self)
        self.config_tester = None

    @require_gpu(min_gpus=1)
    @require_package("paddlefleet")
    def test_pipeline_checkpoint_roundtrip(self):
        from paddle.distributed import fleet
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            dict_to_tuple_helper,
            tuple_to_dict_helper,
        )

        if getattr(fleet.fleet, "_hcg", None) is None:
            strategy = fleet.DistributedStrategy()
            strategy.hybrid_configs = dict(dp_degree=1, mp_degree=1, pp_degree=1, sharding_degree=1)
            fleet.init(is_collective=True, strategy=strategy)
        with device_guard("gpu"):
            for tied in (False, True):
                with self.subTest(tie_word_embeddings=tied):
                    config = Olmo3Config(
                        vocab_size=32,
                        hidden_size=16,
                        intermediate_size=24,
                        num_hidden_layers=2,
                        num_attention_heads=4,
                        num_key_value_heads=2,
                        sliding_window=3,
                        layer_types=["sliding_attention", "full_attention"],
                        tie_word_embeddings=tied,
                        fuse_rms_norm=False,
                        use_cache=False,
                        dtype="float32",
                        max_sequence_length=4,
                        _attn_implementation="eager",
                    )
                    model = Olmo3ForCausalLM(config)
                    input_ids = paddle.to_tensor([[2, 3, 4, 5]], dtype="int64")
                    labels = paddle.to_tensor([[3, 4, 5, 6]], dtype="int64")
                    expected = model(input_ids, labels=labels, return_dict=True)
                    with tempfile.TemporaryDirectory() as checkpoint:
                        model.save_pretrained(checkpoint, safe_serialization=True)
                        pipe = Olmo3ForCausalLMPipe.from_pretrained(checkpoint, config=config, dtype="float32")
                        actual = pipe.run_function[0](dict(input_ids=input_ids))
                        actual, structured = tuple_to_dict_helper(dict_to_tuple_helper(actual))
                        self.assertTrue(structured)
                        for stage in pipe.run_function[1:]:
                            actual = stage(actual)
                        np.testing.assert_allclose(actual.numpy(), expected.logits.numpy(), atol=1e-5, rtol=1e-5)
                        loss = pipe._loss_fn[0](actual, labels).reshape([])
                        np.testing.assert_allclose(loss.numpy(), expected.loss.numpy(), atol=1e-5, rtol=1e-5)
                        loss.backward()
                        self.assertTrue(all(p.grad is not None for p in pipe.parameters()))
                        pipe.save_pretrained(checkpoint, safe_serialization=True)
                        restored = Olmo3ForCausalLM.from_pretrained(checkpoint, config=config, dtype="float32")
                        for name, value in model.state_dict().items():
                            np.testing.assert_allclose(
                                restored.state_dict()[name].numpy(), value.numpy(), atol=1e-5, rtol=1e-5
                            )
                        result = restored(input_ids, labels=labels, return_dict=True)
                        np.testing.assert_allclose(
                            result.logits.numpy(), expected.logits.numpy(), atol=1e-5, rtol=1e-5
                        )

    @require_gpu(min_gpus=2)
    @require_package("paddlefleet")
    def test_pipeline_parallel_training(self):
        from paddle.distributed import fleet

        if paddle.distributed.get_world_size() != 2:
            self.skipTest("Launch this test with two distributed workers for PP=2.")
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = dict(dp_degree=1, mp_degree=1, pp_degree=2, sharding_degree=1)
        strategy.pipeline_configs = dict(micro_batch_size=1, accumulate_steps=1)
        fleet.init(is_collective=True, strategy=strategy)
        with device_guard("gpu"):
            for tied in (False, True):
                config = Olmo3Config(
                    vocab_size=32,
                    hidden_size=16,
                    intermediate_size=24,
                    num_hidden_layers=2,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                    sliding_window=3,
                    layer_types=["sliding_attention", "full_attention"],
                    tie_word_embeddings=tied,
                    fuse_rms_norm=False,
                    use_cache=False,
                    dtype="float32",
                    max_sequence_length=4,
                    _attn_implementation="eager",
                )
                paddle.seed(42)
                model = Olmo3ForCausalLM(config)
                input_ids = paddle.to_tensor([[2, 3, 4, 5]], dtype="int64")
                labels = paddle.to_tensor([[3, 4, 5, 6]], dtype="int64")
                expected = model(input_ids, labels=labels, return_dict=True).loss
                with tempfile.TemporaryDirectory() as checkpoint:
                    directories = []
                    paddle.distributed.all_gather_object(directories, checkpoint)
                    if paddle.distributed.get_rank() == 0:
                        model.save_pretrained(directories[0], safe_serialization=True)
                    paddle.distributed.barrier()
                    config.pipeline_model_parallel_size = 2
                    pipe = Olmo3ForCausalLMPipe.from_pretrained(directories[0], config=config, dtype="float32")
                    engine = fleet.distributed_model(pipe)
                    optimizer = fleet.distributed_optimizer(
                        paddle.optimizer.SGD(learning_rate=0.01, parameters=pipe.parameters())
                    )
                    before = [parameter.clone() for parameter in pipe.parameters()]
                    data = [dict(input_ids=input_ids), labels]
                    actual = engine.eval_batch(data, compute_loss=True).reshape([])
                    np.testing.assert_allclose(actual.numpy(), expected.numpy(), atol=1e-5, rtol=1e-5)
                    loss = engine.train_batch(data, optimizer).reshape([])
                    np.testing.assert_allclose(loss.numpy(), expected.numpy(), atol=1e-5, rtol=1e-5)
                    self.assertTrue(all(paddle.isfinite(parameter).all().item() for parameter in pipe.parameters()))
                    self.assertTrue(
                        any(not paddle.equal_all(old, new).item() for old, new in zip(before, pipe.parameters()))
                    )
                    paddle.distributed.barrier()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

        tester = Olmo3ModelTester(
            self,
            vocab_size=128,
            hidden_size=32,
            intermediate_size=48,
            num_attention_heads=4,
            num_key_value_heads=2,
            eos_token_id=2,
            dtype="float32",
        )
        for head_dim in (None, 4):
            with self.subTest(head_dim=head_dim):
                config = tester.get_config()
                config = Olmo3Config.from_dict({**config.to_dict(), "head_dim": head_dim})
                config._attn_implementation = "eager"
                tester.create_and_check_model(
                    config,
                    ids_tensor([tester.batch_size, tester.seq_length], config.vocab_size),
                    None,
                )

    def test_lm_head_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_lm_head_model(*config_and_inputs)

        for architectures in (None, [], ["Olmo3ForCausalLM"]):
            with self.subTest(architectures=architectures):
                config = Olmo3Config(
                    vocab_size=128,
                    hidden_size=32,
                    intermediate_size=48,
                    num_hidden_layers=2,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                    architectures=architectures,
                    fuse_rms_norm=False,
                    _attn_implementation="eager",
                )
                model = AutoModelForCausalLM.from_config(config, dtype="float32")
                self.assertIsInstance(model, Olmo3ForCausalLM)
                model.eval()
                input_ids = ids_tensor([2, 6], config.vocab_size)
                with paddle.no_grad():
                    result = model(input_ids, return_dict=True)
                self.assertEqual(result.logits.shape, [2, 6, config.vocab_size])
                self.assertTrue(paddle.isfinite(result.logits).all().item())

    def test_training_with_recompute(self):
        for recompute in (False, True):
            with self.subTest(recompute=recompute):
                config = Olmo3Config(
                    vocab_size=32,
                    hidden_size=16,
                    intermediate_size=24,
                    num_hidden_layers=2,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                    layer_types=["sliding_attention", "full_attention"],
                    sliding_window=3,
                    use_cache=False,
                    fuse_rms_norm=False,
                    _attn_implementation="eager",
                )
                if recompute:
                    config.recompute_granularity = "full"
                    config.recompute_method = "uniform"
                    config.recompute_num_layers = 1
                model = Olmo3ForCausalLM(config)
                model.train()
                input_ids = ids_tensor([2, 6], config.vocab_size)
                loss = model(input_ids, labels=input_ids, return_dict=True).loss
                self.assertTrue(paddle.isfinite(loss).all().item())
                loss.backward()
                for parameter in model.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(paddle.isfinite(parameter.grad).all().item())

    def test_generation_cache_chunks_and_read_only_cache(self):
        config = Olmo3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=24,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=2,
            layer_types=["sliding_attention", "full_attention"],
            sliding_window=3,
            use_cache=True,
            fuse_rms_norm=False,
            _attn_implementation="eager",
        )
        model = Olmo3ForCausalLM(config)
        model.eval()
        self.assertTrue(model.can_generate())
        self.assertIsNotNone(model.generation_config)
        input_ids = paddle.to_tensor([[2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13]], dtype="int64")
        with paddle.no_grad():
            expected = model(input_ids, use_cache=False, return_dict=True)
            self.assertIsNone(expected.past_key_values)
            prefill = model(input_ids[:, :3], return_dict=True)
            cache = prefill.past_key_values
            self.assertEqual(cache.get_seq_length(), 3)
            snapshot = [(layer.keys.clone(), layer.values.clone()) for layer in cache.layers]
            read_only = model(input_ids[:, 3:5], past_key_values=cache, use_cache=False, return_dict=True)
            self.assertIsNone(read_only.past_key_values)
            self.assertEqual(cache.get_seq_length(), 3)
            for layer, (keys, values) in zip(cache.layers, snapshot):
                np.testing.assert_array_equal(layer.keys.numpy(), keys.numpy())
                np.testing.assert_array_equal(layer.values.numpy(), values.numpy())
            np.testing.assert_allclose(read_only.logits.numpy(), expected.logits[:, 3:5].numpy(), atol=1e-5, rtol=1e-5)
            prepared = model.prepare_inputs_for_generation(input_ids[:, :5], past_key_values=cache)
            self.assertEqual(prepared["input_ids"].shape[1], 2)
            chunk = model(**prepared)
            self.assertEqual(cache.get_seq_length(), 5)
            np.testing.assert_allclose(chunk.logits.numpy(), expected.logits[:, 3:5].numpy(), atol=1e-5, rtol=1e-5)
            prepared = model.prepare_inputs_for_generation(input_ids, past_key_values=cache)
            self.assertEqual(prepared["input_ids"].shape[1], 1)
            decoded = model(**prepared)
            self.assertEqual(cache.get_seq_length(), 6)
            np.testing.assert_allclose(decoded.logits.numpy(), expected.logits[:, 5:].numpy(), atol=1e-5, rtol=1e-5)

    def test_padding_mask_and_cached_chunks(self):
        config = Olmo3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=24,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=2,
            layer_types=["sliding_attention", "full_attention"],
            sliding_window=3,
            fuse_rms_norm=False,
            _attn_implementation="eager",
        )
        model = Olmo3ForCausalLM(config)
        model.eval()
        for left_padding in (False, True):
            with self.subTest(left_padding=left_padding), paddle.no_grad():
                tokens = [[2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 1, 1]]
                mask = [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]]
                if left_padding:
                    tokens[1] = [1, 1, 8, 9, 10, 11]
                    mask[1] = [0, 0, 1, 1, 1, 1]
                input_ids = paddle.to_tensor(tokens, dtype="int64")
                attention_mask = paddle.to_tensor(mask, dtype="bool")
                position_ids = paddle.clip(attention_mask.astype("int64").cumsum(-1) - 1, min=0)
                expected = model(
                    input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    return_dict=True,
                ).logits
                self.assertTrue(paddle.isfinite(expected).all().item())
                for index, length in enumerate((6, 4)):
                    valid = input_ids[index][attention_mask[index]].unsqueeze(0)
                    single = model(valid, use_cache=False, return_dict=True).logits[0]
                    np.testing.assert_allclose(
                        expected[index][attention_mask[index]].numpy(), single.numpy(), atol=1e-5, rtol=1e-5
                    )
                prefill = model(
                    input_ids[:, :4],
                    attention_mask=attention_mask[:, :4],
                    position_ids=position_ids[:, :4],
                    use_cache=True,
                    return_dict=True,
                )
                chunk = model(
                    input_ids[:, 4:],
                    attention_mask=attention_mask,
                    position_ids=position_ids[:, 4:],
                    past_key_values=prefill.past_key_values,
                    use_cache=True,
                    return_dict=True,
                ).logits
                valid = attention_mask[:, 4:]
                np.testing.assert_allclose(chunk[valid].numpy(), expected[:, 4:][valid].numpy(), atol=1e-5, rtol=1e-5)

        with paddle.no_grad():
            input_ids = paddle.to_tensor([[2, 3, 4, 5, 6, 7]], dtype="int64")
            expected = model(input_ids, use_cache=False, return_dict=True).logits
            indices = paddle.full([1, 1, 6, 1], 6, dtype="int32")
            snapshot = indices.clone()
            actual = model(input_ids, attn_mask_startend_row_indices=indices, use_cache=False, return_dict=True).logits
            np.testing.assert_allclose(actual.numpy(), expected.numpy(), atol=1e-5, rtol=1e-5)
            np.testing.assert_array_equal(indices.numpy(), snapshot.numpy())


#  Generation test, marked with slow decorator, not executed by default
class Olmo3GenerationTest(unittest.TestCase):
    _MODEL_ID = "allenai/OLMo-3-7B-Instruct"

    @slow
    @require_package("transformers")
    def test_generation_capital_of_china(self):
        tokenizer = AutoTokenizer.from_pretrained(self._MODEL_ID, trust_remote_code=True, download_hub="modelscope")

        model = Olmo3ForCausalLM.from_pretrained(
            self._MODEL_ID,
            dtype="bfloat16",
            download_hub="modelscope",
            load_via_cpu=True,  # Load weights via CPU to avoid GPU OOM
        )
        model.eval()

        messages = [{"role": "user", "content": "What is the capital of China? Answer in one word."}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="np")
        input_ids = paddle.to_tensor(inputs["input_ids"])

        with paddle.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=32,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
            )[0]

        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        generated_text_lower = generated_text.lower()
        print("output:" + generated_text)
        self.assertIn(
            "beijing",
            generated_text_lower,
            f"Expected 'beijing' in generated text, but got: {generated_text}",
        )


"""
Fixed reference arrays for allenai/OLMo-3-7B-Instruct alignment.
Reproduced with Transformers 4.57.0, PyTorch 2.9.0+cu130, CPU SDPA and FP32.
The Paddle comparison also uses CPU SDPA and FP32 with unfused RMSNorm.
The local checkpoint has no recorded upstream revision; SHA256 fingerprints:
config.json: 6282ae812f9e494bd212e9a3233d68dc4d5f1e2569642423d2b79a3e80373ef8
model-00001-of-00003.safetensors: e7bd4c8ba52177a815aee5929cb458a116a65799688d3e51402500a1cb77023c
model-00002-of-00003.safetensors: ea92aa0499a2ca02400e5d87324009e172bdbfb9809f2ac501485d8224382328
model-00003-of-00003.safetensors: d50f95e4f2ec5fc4fbce6217cbf54fd1d700928b05e5550c8ee08bf4ba948925
The stored prompt asks: "What is the difference between a cat and a dog?"
The tests compare ten generated token IDs exactly and require the mean absolute
error of the first twenty logits at the last input position to be below 1e-2.
"""


_OLMO3_REF_INPUT_IDS = [
    100264,
    9125,
    198,
    2675,
    527,
    264,
    11190,
    734,
    1824,
    17157,
    15592,
    18328,
    13,
    1472,
    656,
    539,
    5131,
    617,
    2680,
    311,
    904,
    5865,
    13,
    220,
    100266,
    100267,
    100265,
    198,
    100264,
    882,
    198,
    3923,
    374,
    279,
    6811,
    1990,
    264,
    8415,
    323,
    264,
    5679,
    30,
    100265,
    198,
    100264,
    78191,
    198,
]

_OLMO3_REF_GENERATED_TOP10 = [96556, 0, 6104, 2225, 19987, 323, 12875, 527, 5526, 26159]

_OLMO3_REF_FIRST_TOKEN_LOGITS_FIRST20 = [
    3.8641438484191895,
    9.757109642028809,
    4.222537994384766,
    4.725244522094727,
    1.880645990371704,
    1.7617301940917969,
    5.4357686042785645,
    4.220963001251221,
    1.1612153053283691,
    8.618879318237305,
    2.527357578277588,
    -3.317128896713257,
    1.4765191078186035,
    -1.2547285556793213,
    0.2115684449672699,
    1.5777826309204102,
    5.592334747314453,
    1.2720606327056885,
    1.2675397396087646,
    1.0419766902923584,
]


def _build_ref_dict():
    return {
        "input_ids": _OLMO3_REF_INPUT_IDS,
        "generated_top10": _OLMO3_REF_GENERATED_TOP10,
        "first_token_logits_first20": _OLMO3_REF_FIRST_TOKEN_LOGITS_FIRST20,
    }


@slow
class Olmo3DiffTest(unittest.TestCase):
    _MODEL_ID = "allenai/OLMo-3-7B-Instruct"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._ref = _build_ref_dict()
        with device_guard("cpu"):
            config = Olmo3Config.from_pretrained(cls._MODEL_ID, download_hub="modelscope")
            config.fuse_rms_norm = False
            config._attn_implementation = "sdpa"
            cls._model = Olmo3ForCausalLM.from_pretrained(
                cls._MODEL_ID,
                dtype="float32",
                config=config,
                load_checkpoint_format="flex_checkpoint",
                download_hub="modelscope",
                load_via_cpu=True,
            )
            cls._model.eval()
        cls.addClassCleanup(delattr, cls, "_model")

    def setUp(self):
        super().setUp()
        self.addCleanup(paddle.set_device, paddle.get_device())
        paddle.set_device("cpu")

    def test_diff_generated_tokens(self):
        ref = self._ref
        pd_model = self._model

        input_ids_np = np.array(ref["input_ids"], dtype=np.int64).reshape([1, len(ref["input_ids"])])
        paddle_input = paddle.to_tensor(input_ids_np)
        with paddle.no_grad():
            output = pd_model.generate(
                paddle_input,
                max_new_tokens=10,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
            )

        output_ids = output[0]
        pd_generated = output_ids[0].tolist()
        ref_generated = ref["generated_top10"]
        print(f"Paddle generated: {pd_generated}")
        print(f"Reference:       {ref_generated}")

        self.assertEqual(
            pd_generated,
            ref_generated,
            f"Generated tokens mismatch. Paddle: {pd_generated}, Reference: {ref_generated}",
        )

    def test_diff_first_token_logits(self):
        ref = self._ref
        pd_model = self._model

        input_ids_np = np.array(ref["input_ids"], dtype=np.int64).reshape([1, len(ref["input_ids"])])
        paddle_input = paddle.to_tensor(input_ids_np)

        with paddle.no_grad():
            paddle_output = pd_model(paddle_input, return_dict=True)
            paddle_logits = paddle_output.logits.detach().cpu().astype("float32").numpy()

        paddle_logits_first20 = paddle_logits[0, -1, :20].astype(np.float32)

        ref_logits = np.array(ref["first_token_logits_first20"], dtype=np.float32)
        mean_diff = float(np.mean(np.abs(paddle_logits_first20 - ref_logits)))

        print(f"Paddle first token logits first 20: {paddle_logits_first20.tolist()}")
        print(f"Reference first token logits first 20: {ref_logits.tolist()}")
        print(f"Mean diff: {mean_diff:.6f}")

        self.assertLess(
            mean_diff,
            1e-2,
            f"Mean logit diff ({mean_diff:.6f}) must be below 1e-2.",
        )


if __name__ == "__main__":
    unittest.main()
