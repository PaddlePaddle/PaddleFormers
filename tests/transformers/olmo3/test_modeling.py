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

import os
import re
import shutil
import tempfile
import unittest

import numpy as np
import paddle

from paddleformers.transformers import (
    AutoTokenizer,
    Olmo3Config,
    Olmo3ForCausalLM,
    Olmo3Model,
)
from tests.testing_utils import require_package, slow
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

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_lm_head_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_lm_head_model(*config_and_inputs)

    def test_config(self):
        """Test that Olmo3Config can be created with default values."""
        config = Olmo3Config(num_hidden_layers=4)
        self.assertEqual(config.model_type, "olmo3")
        self.assertEqual(config.vocab_size, 50304)
        self.assertEqual(config.hidden_size, 4096)
        self.assertEqual(config.sliding_window, 4096)
        self.assertEqual(config.attention_bias, False)
        self.assertEqual(config.rms_norm_eps, 1e-5)

    def test_config_layer_types_default(self):
        """Test that default layer_types are correctly generated."""
        config = Olmo3Config(num_hidden_layers=8)
        # Default: 3 out of 4 layers use sliding_attention, every 4th uses full_attention
        expected = [
            "sliding_attention",  # layer 0: (0+1)%4=1 != 0
            "sliding_attention",  # layer 1: (1+1)%4=2 != 0
            "sliding_attention",  # layer 2: (2+1)%4=3 != 0
            "full_attention",  # layer 3: (3+1)%4=0
            "sliding_attention",  # layer 4: (4+1)%4=1 != 0
            "sliding_attention",  # layer 5: (5+1)%4=2 != 0
            "sliding_attention",  # layer 6: (6+1)%4=3 != 0
            "full_attention",  # layer 7: (7+1)%4=0
        ]
        self.assertEqual(config.layer_types, expected)

    def test_config_custom_layer_types(self):
        """Test that custom layer_types can be set."""
        custom_types = ["full_attention", "sliding_attention"]
        config = Olmo3Config(num_hidden_layers=2, layer_types=custom_types)
        self.assertEqual(config.layer_types, custom_types)

    def test_config_inherits_olmo2(self):
        """Test that Olmo3Config inherits from Olmo2Config."""
        from paddleformers.transformers.olmo2.configuration import Olmo2Config

        config = Olmo3Config(num_hidden_layers=4)
        self.assertIsInstance(config, Olmo2Config)


#  Generation test, marked with slow decorator, not executed by default
class Olmo3GenerationTest(unittest.TestCase):
    _MODEL_ID = "allenai/OLMo-3-7B-Instruct"

    def setUp(self):
        pass

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
Due to olmo3 depending on an old version of transformers, downgrading would cause tokenizer errors
in paddleformers. Therefore, olmo3 and paddle alignment uses direct array generation for comparison.

The original 8b model input prompt:

input_question = "What is the capital of China?"
messages = [{"role": "user", "content": input_question}]
input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

Proceed with subsequent inference processing comparison.

Due to slight precision differences in each layer and this being bf16 precision, the final diff is
2e-2, just slightly below the mandatory 1e-2 threshold.

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
    3.90625,
    9.8125,
    4.25,
    4.71875,
    1.890625,
    1.765625,
    5.46875,
    4.21875,
    1.1875,
    8.625,
    2.53125,
    -3.3125,
    1.4765625,
    -1.25,
    0.2275390625,
    1.5859375,
    5.625,
    1.28125,
    1.2734375,
    1.0546875,
]
_OLMO3_REF_FIRST_TOKEN_TOP1 = 96556
_OLMO3_REF_FIRST_TOKEN_LOGITS_MEAN = 2.816846


def _build_ref_dict():
    return {
        "input_ids": _OLMO3_REF_INPUT_IDS,
        "generated_top10": _OLMO3_REF_GENERATED_TOP10,
        "first_token_logits_first20": _OLMO3_REF_FIRST_TOKEN_LOGITS_FIRST20,
        "first_token_top1": _OLMO3_REF_FIRST_TOKEN_TOP1,
        "first_token_logits_mean": _OLMO3_REF_FIRST_TOKEN_LOGITS_MEAN,
    }


def _transpose_to_pt_layout(state_dict):
    transpose_names = Olmo3ForCausalLM.transpose_weight_keys or []
    converted = {}
    for key, value in state_dict.items():
        should_transpose = value.ndim == 2 and any(
            re.search(rf"\.{name}\.weight$", key) or re.fullmatch(rf"^{name}\.weight$", key)
            for name in transpose_names
        )
        converted[key] = value.transpose([1, 0]) if should_transpose else value
    return converted


def _prepare_pt_layout_pd_checkpoint(src_model_path: str) -> str:
    tmp_root = tempfile.mkdtemp(prefix="olmo3_pd_pt_layout_")
    dst_model_path = os.path.join(tmp_root, "model")
    os.makedirs(dst_model_path, exist_ok=True)

    pd_weight_path = os.path.join(dst_model_path, "model_state.pdparams")
    if not os.path.exists(pd_weight_path):
        model = Olmo3ForCausalLM.from_pretrained(
            src_model_path,
            dtype="bfloat16",
            load_checkpoint_format="flex_checkpoint",
            _attn_implementation="eager",
            download_hub="modelscope",
            load_via_cpu=True,
        )
        paddle.save(model.state_dict(), pd_weight_path)
        del model
        if paddle.is_compiled_with_cuda():
            paddle.cuda.empty_cache()

    state_dict = paddle.load(pd_weight_path)
    state_dict = _transpose_to_pt_layout(state_dict)
    paddle.save(state_dict, pd_weight_path)
    return tmp_root


class Olmo3DiffTest(unittest.TestCase):
    _MODEL_ID = "allenai/OLMo-3-7B-Instruct"

    def setUp(self):
        pass

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._ref = _build_ref_dict()
        cls._pt_layout_pd_tmp_root = None
        cls._pt_layout_pd_model_path = None

    @classmethod
    def tearDownClass(cls):
        if (
            hasattr(cls, "_pt_layout_pd_tmp_root")
            and cls._pt_layout_pd_tmp_root
            and os.path.isdir(cls._pt_layout_pd_tmp_root)
        ):
            shutil.rmtree(cls._pt_layout_pd_tmp_root, ignore_errors=True)
        super().tearDownClass()

    @slow
    def test_diff_generated_tokens(self):
        ref = self._ref
        pd_model = Olmo3ForCausalLM.from_pretrained(
            self._MODEL_ID,
            dtype="bfloat16",
            load_checkpoint_format="flex_checkpoint",
            _attn_implementation="eager",
            download_hub="modelscope",
            load_via_cpu=True,
        )
        pd_model.eval()

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

    @slow
    def test_diff_first_token_logits(self):
        ref = self._ref
        pd_model = Olmo3ForCausalLM.from_pretrained(
            self._MODEL_ID,
            dtype="bfloat16",
            load_checkpoint_format="flex_checkpoint",
            _attn_implementation="eager",
            download_hub="modelscope",
            load_via_cpu=True,
        )
        pd_model.eval()

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
            3e-2,
            f"Mean logit diff ({mean_diff:.6f}) exceeds 3e-2 threshold.",
        )


if __name__ == "__main__":
    unittest.main()
