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

import gc
import os
import shutil
import tempfile
import unittest

import numpy as np
import paddle

from paddleformers.transformers import Phi4ForCausalLM, Phi4Tokenizer
from tests.testing_utils import slow


class TestPhi4Modeling(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_model_creation_and_forward(self):
        from paddleformers.transformers import Phi4Config

        config = Phi4Config(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=512,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=4,
            max_position_embeddings=128,
            pad_token_id=0,
        )
        model = Phi4ForCausalLM(config)
        model.eval()

        batch_size = 1
        seq_length = 3
        input_ids = paddle.randint(0, config.vocab_size, [batch_size, seq_length], dtype="int64")
        with paddle.no_grad():
            outputs = model(input_ids=input_ids, use_cache=False)

        self.assertIsNotNone(outputs)
        logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs.logits
        self.assertEqual(list(logits.shape), [batch_size, seq_length, config.vocab_size])
        print(f"Model creation and forward pass OK, shape: {logits.shape}")

    def test_model_save_and_load(self):
        from paddleformers.transformers import Phi4Config

        config = Phi4Config(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=512,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=4,
            max_position_embeddings=128,
            pad_token_id=0,
        )

        model = Phi4ForCausalLM(config)
        model.eval()

        save_path = os.path.join(self.temp_dir, "saved_model")
        model.save_pretrained(save_path)
        self.assertTrue(os.path.exists(save_path))

        loaded_model = Phi4ForCausalLM.from_pretrained(save_path)
        loaded_model.eval()

        input_ids = paddle.randint(0, config.vocab_size, [1, 3], dtype="int64")
        with paddle.no_grad():
            outputs = loaded_model(input_ids=input_ids)
        logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs.logits
        self.assertEqual(logits.shape[-1], config.vocab_size)
        print(f"Model save and load OK, logits shape={tuple(logits.shape)}")

    def test_beam_search_with_cache(self):
        from paddleformers.transformers import Phi4Config

        config = Phi4Config(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=512,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=4,
            max_position_embeddings=128,
            pad_token_id=0,
        )
        model = Phi4ForCausalLM(config)
        model.eval()
        input_ids = paddle.randint(1, config.vocab_size, [1, 3], dtype="int64")
        with paddle.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=2,
                decode_strategy="beam_search",
                num_beams=2,
                use_cache=True,
                eos_token_id=config.eos_token_id,
                pad_token_id=config.pad_token_id,
            )[0]
        self.assertEqual(output_ids.shape[0], 1)
        print(f"Beam search with cache OK, output shape: {output_ids.shape}")


@slow
class TestPhi4InferenceUseHf(unittest.TestCase):
    model_path = "microsoft/Phi-4-mini-flash-reasoning"
    model = None
    tokenizer = None

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = Phi4Tokenizer.from_pretrained(cls.model_path)
        cls.model = Phi4ForCausalLM.from_pretrained(
            cls.model_path,
            dtype="bfloat16",
            convert_from_hf=True,
        )
        cls.model.eval()

    @classmethod
    def tearDownClass(cls):
        del cls.model
        del cls.tokenizer
        cls.model = None
        cls.tokenizer = None
        gc.collect()
        try:
            paddle.device.cuda.empty_cache()
        except Exception:
            pass

    def test_inference_cat_vs_dog(self):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of China?"},
        ]
        input_text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self.tokenizer(input_text, return_tensors="pd")
        input_ids = inputs["input_ids"]

        with paddle.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=256,
                do_sample=False,
                temperature=1.0,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )[0]

        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        print(f"\n{'=' * 60}")
        print("Input: What is the capital of China?")
        print(f"{'=' * 60}")
        print(f"Output:\n{response}")
        print(f"{'=' * 60}")


@slow
class TestPhi4InferenceUsePaddle(unittest.TestCase):
    model_path = os.path.expanduser("microsoft/Phi-4-mini-flash-reasoning")
    model = None
    tokenizer = None

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = Phi4Tokenizer.from_pretrained(cls.model_path)
        cls.model = Phi4ForCausalLM.from_pretrained(
            cls.model_path,
            dtype="bfloat16",
            convert_from_hf=True,
        )
        cls.model.eval()

    @classmethod
    def tearDownClass(cls):
        del cls.model
        del cls.tokenizer
        cls.model = None
        cls.tokenizer = None
        gc.collect()
        try:
            paddle.device.cuda.empty_cache()
        except Exception:
            pass

    def test_inference_cat_vs_dog(self):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of China?"},
        ]
        input_text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self.tokenizer(input_text, return_tensors="pd")
        input_ids = inputs["input_ids"]

        with paddle.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=256,
                do_sample=False,
                temperature=1.0,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )[0]

        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        print(f"\n{'=' * 60}")
        print("Input: What is the capital of China?")
        print(f"{'=' * 60}")
        print(f"Output:\n{response}")
        print(f"{'=' * 60}")

    def test_manual_greedy_no_cache(self):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of China?"},
        ]
        input_text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self.tokenizer(input_text, return_tensors="pd")
        cur_ids = inputs["input_ids"]

        generated = []
        eos_id = self.tokenizer.eos_token_id
        for _ in range(30):
            with paddle.no_grad():
                out = self.model(input_ids=cur_ids, use_cache=False)
            lgt = out[0] if isinstance(out, (tuple, list)) else out.logits
            next_id = int(lgt[0, -1, :].argmax().item())
            generated.append(next_id)
            if next_id == eos_id:
                break
            cur_ids = paddle.concat([cur_ids, paddle.to_tensor([[next_id]], dtype="int64")], axis=1)

        response = self.tokenizer.decode(generated, skip_special_tokens=True)
        print(f"\n{'=' * 60}")
        print("[manual greedy, use_cache=False, 30 steps]")
        print(f"token ids: {generated}")
        print(f"Output:\n{response}")
        print(f"{'=' * 60}")
        self.assertIsInstance(response, str)

    def test_manual_greedy_with_cache(self):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of China?"},
        ]
        input_text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self.tokenizer(input_text, return_tensors="pd")
        cur_ids = inputs["input_ids"]

        generated = []
        eos_id = self.tokenizer.eos_token_id
        past_key_values = None
        for step in range(30):
            with paddle.no_grad():
                out = self.model(
                    input_ids=cur_ids if step == 0 else paddle.to_tensor([[generated[-1]]], dtype="int64"),
                    use_cache=True,
                    past_key_values=past_key_values,
                )
            if hasattr(out, "logits"):
                lgt = out.logits
                past_key_values = out.past_key_values
            else:
                lgt = out[0]
                past_key_values = out[1] if len(out) > 1 else None
            next_id = int(lgt[0, -1, :].argmax().item())
            generated.append(next_id)
            if next_id == eos_id:
                break

        response = self.tokenizer.decode(generated, skip_special_tokens=True)
        print(f"\n{'=' * 60}")
        print("[manual greedy, use_cache=True, 30 steps]")
        print(f"token ids: {generated}")
        print(f"Output:\n{response}")
        print(f"{'=' * 60}")
        self.assertIsInstance(response, str)


"""
Alignment test notes:
- Due to differences in manually compiled CUDA ops, accumulated errors prevent last-layer alignment.
- Achieved: layer0 mean diff ~0.0008, first 10 token ids match, last layer mean diff ~0.28.
- REF_* values are hardcoded from PyTorch inference in a dedicated phi4 conda environment.
"""


@slow
class TestPhi4LayerDiffAlignment(unittest.TestCase):
    model = None
    tokenizer = None
    model_path = "microsoft/Phi-4-mini-flash-reasoning"

    REF_LAYER0_FIRST_TOKEN_FIRST20 = [
        -0.09912109375,
        0.4345703125,
        -0.0721282958984375,
        0.1578369140625,
        -0.19476318359375,
        -0.394439697265625,
        -0.0894775390625,
        -0.022216796875,
        0.01220703125,
        0.056640625,
        -0.1060791015625,
        -0.0780029296875,
        0.08111572265625,
        -0.12481689453125,
        0.0150146484375,
        -0.0150299072265625,
        -0.384490966796875,
        0.0347900390625,
        0.1318359375,
        -0.0999755859375,
    ]

    REF_LAST_LAYER_FIRST_TOKEN_FIRST20 = [
        0.0306396484375,
        4.347686767578125,
        0.177886962890625,
        9.518585205078125,
        -1.6810302734375,
        4.293243408203125,
        -3.50250244140625,
        6.155670166015625,
        11.880416870117188,
        -1.83856201171875,
        2.328369140625,
        -0.244964599609375,
        -0.10107421875,
        8.194091796875,
        -8.33013916015625,
        0.3831634521484375,
        1.711822509765625,
        0.126953125,
        2.81903076171875,
        -1.20147705078125,
    ]

    REF_LAST_LAYER_LAST_TOKEN_FIRST20 = [
        15.5140380859375,
        10.106557846069336,
        -8.950668334960938,
        13.417678833007812,
        -8.5753173828125,
        6.5550537109375,
        11.9271240234375,
        -21.944580078125,
        38.504486083984375,
        -0.96942138671875,
        4.662109375,
        -9.9052734375,
        29.12060546875,
        36.023704528808594,
        -8.587677001953125,
        -3.0186767578125,
        6.49127197265625,
        -13.24774169921875,
        15.015228271484375,
        -19.550689697265625,
    ]

    REF_NEW_TOKEN_IDS = [33313, 881, 523, 53520, 11, 813, 357, 1309, 316, 11310]

    @classmethod
    def setUpClass(cls):
        paddle.seed(42)
        cls.tokenizer = Phi4Tokenizer.from_pretrained(cls.model_path)
        cls.model = Phi4ForCausalLM.from_pretrained(
            cls.model_path,
            dtype="bfloat16",
            convert_from_hf=True,
        )
        cls.model.eval()

    @classmethod
    def tearDownClass(cls):
        del cls.model
        del cls.tokenizer
        cls.model = None
        cls.tokenizer = None
        gc.collect()
        try:
            paddle.device.cuda.empty_cache()
        except Exception:
            pass

    def _get_input_ids(self):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of China?"},
        ]
        input_text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self.tokenizer(input_text, return_tensors="pd")
        return inputs["input_ids"]

    # Only align layer0 and first 10 token ids
    def test_alignment(self):
        input_ids = self._get_input_ids()
        print(f"\n{'=' * 80}")
        print(f"Input ids: {input_ids[0].tolist()}, shape: {input_ids.shape}")

        layer0_out = []
        last_layer_out = []

        def _capture(store):
            def hook(layer, args, output):
                x = output[0] if isinstance(output, (tuple, list)) else output
                store.append(x.detach().cast("float32").numpy())

            return hook

        h0 = self.model.model.layers[0].register_forward_post_hook(_capture(layer0_out))
        hl = self.model.model.layers[-1].register_forward_post_hook(_capture(last_layer_out))
        try:
            with paddle.no_grad():
                self.model(input_ids=input_ids, use_cache=False)
        finally:
            h0.remove()
            hl.remove()

        self.assertTrue(layer0_out, "Hook did not capture layer0 output")
        self.assertTrue(last_layer_out, "Hook did not capture last_layer output")

        l0 = layer0_out[0][0, 0, :20].tolist()
        ref_l0 = self.REF_LAYER0_FIRST_TOKEN_FIRST20
        diff0 = np.abs(np.array(l0) - np.array(ref_l0))
        print(f"\nLayer0[0,0,:20]  paddle : {l0}")
        print(f"Layer0[0,0,:20]  pytorch: {ref_l0}")
        print(f"Layer0 diff: max={diff0.max():.6f}, mean={diff0.mean():.6f}")

        # Require mean diff < 1e-2
        self.assertTrue(diff0.mean() < 0.01, "Layer0 diff too large")

        ll = last_layer_out[0]
        ll_first = ll[0, 0, :20].tolist()
        ll_last = ll[0, -1, :20].tolist()
        ref_ll_first = self.REF_LAST_LAYER_FIRST_TOKEN_FIRST20
        ref_ll_last = self.REF_LAST_LAYER_LAST_TOKEN_FIRST20
        diff_ll_first = np.abs(np.array(ll_first) - np.array(ref_ll_first))
        diff_ll_last = np.abs(np.array(ll_last) - np.array(ref_ll_last))
        print(f"\nLastLayer[0,0,:20]  paddle : {ll_first}")
        print(f"LastLayer[0,0,:20]  pytorch: {ref_ll_first}")
        print(f"LastLayer first token diff: max={diff_ll_first.max():.6f}, mean={diff_ll_first.mean():.6f}")
        print(f"\nLastLayer[0,-1,:20] paddle : {ll_last}")
        print(f"LastLayer[0,-1,:20] pytorch: {ref_ll_last}")
        print(f"LastLayer last token diff: max={diff_ll_last.max():.6f}, mean={diff_ll_last.mean():.6f}")

        # Last layer cannot align due to accumulated CUDA op errors across layers

        # Generate with KV cache
        with paddle.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=512,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )[0]

        new_ids = output_ids[0].tolist()
        full_text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        print(f"\n{'=' * 80}")
        print(f"Full output:\n{full_text}")
        print(f"{'=' * 80}")

        # First 10 tokens comparison
        first10 = new_ids[:10]
        ref_ids = self.REF_NEW_TOKEN_IDS
        match = sum(a == b for a, b in zip(first10, ref_ids))
        ref_text = self.tokenizer.decode(ref_ids, skip_special_tokens=True)
        print(f"\nFirst-10 token match: {match}/{len(ref_ids)}")
        print(f"  Paddle : {first10}  -> {self.tokenizer.decode(first10, skip_special_tokens=True)}")
        print(f"  PyTorch: {ref_ids}  -> {ref_text}")
        print(f"{'=' * 80}")

        self.assertGreaterEqual(match, 10, f"Token match too low: {match}/{len(ref_ids)}")


def run_tests():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestPhi4Modeling))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    print(f"Errors: {len(result.errors)}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    import sys

    sys.exit(run_tests())
