# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

"""Tests for _sample_next_token and GreedyGenerator.generate sampling params."""

import functools
import unittest

import paddle
from paddle.distributed import fleet

from paddleformers.fleet.generation.greedy_generator import (
    GreedyGenerator,
    _sample_next_token,
)
from paddleformers.fleet.gpt_builders import gpt_builder
from paddleformers.fleet.models.gpt import GPTConfig

# ---------------------------------------------------------------------------
# Shared fleet init (runs once per process)
# ---------------------------------------------------------------------------


def _fleet_init():
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": 1,
        "pp_degree": 1,
        "sharding_degree": 1,
        "sep_degree": 1,
        "cp_degree": 1,
        "ep_degree": 1,
        "moe_sharding_degree": 1,
        "order": [
            "sharding",
            "moe_sharding",
            "pp",
            "sep",
            "cp",
            "dp",
            "ep",
            "mp",
        ],
    }
    fleet.init(is_collective=True, strategy=strategy)


_fleet_init()


def _make_model(
    vocab_size: int = 64,
    hidden_size: int = 64,
    num_layers: int = 2,
    max_seq_len: int = 32,
):
    """Build a tiny GPTModel suitable for inference tests."""
    paddle.manual_seed(0)
    config = GPTConfig(
        num_hidden_layers=num_layers,
        hidden_size=hidden_size,
        rotary_base=10000,
        vocab_size=vocab_size,
        rotary_percent=1.0,
        rope_scaling=1.0,
        position_embedding_type="rope",
        num_attention_heads=4,
        intermediate_size=hidden_size * 2,
        max_sequence_length=max_seq_len,
        normalization="RMSNorm",
        hidden_dropout_prob=0.0,
        attention_dropout=0.0,
        init_method=functools.partial(paddle.nn.init.xavier_uniform_, gain=1.0),
        output_layer_init_method=functools.partial(
            paddle.nn.init.xavier_uniform_, gain=1.0
        ),
        tie_word_embeddings=True,
    )
    return gpt_builder(config, num_stages=1), config


# ===========================================================================
# Tests for _sample_next_token (no model needed)
# ===========================================================================


class TestSampleNextTokenGreedy(unittest.TestCase):
    def _logits(self, peak_idx: int, vocab: int = 10, batch: int = 1):
        logits = paddle.zeros([batch, vocab])
        logits[:, peak_idx] = 10.0
        return logits

    def test_temperature_zero_returns_argmax(self):
        tok = _sample_next_token(
            self._logits(3), temperature=0.0, top_k=0, top_p=0.0
        )
        self.assertEqual(tok.shape, [1, 1])
        self.assertEqual(int(tok[0, 0]), 3)

    def test_negative_temperature_returns_argmax(self):
        tok = _sample_next_token(
            self._logits(7), temperature=-1.0, top_k=0, top_p=0.0
        )
        self.assertEqual(int(tok[0, 0]), 7)

    def test_top_k_one_returns_argmax(self):
        tok = _sample_next_token(
            self._logits(5), temperature=1.0, top_k=1, top_p=0.0
        )
        self.assertEqual(int(tok[0, 0]), 5)

    def test_greedy_batch(self):
        logits = paddle.zeros([3, 10])
        peaks = [2, 6, 9]
        for i, p in enumerate(peaks):
            logits[i, p] = 10.0
        tok = _sample_next_token(logits, temperature=0.0, top_k=0, top_p=0.0)
        self.assertEqual(tok.shape, [3, 1])
        for i, p in enumerate(peaks):
            self.assertEqual(int(tok[i, 0]), p)


class TestSampleNextTokenTopK(unittest.TestCase):
    def test_top_k_constrains_output(self):
        vocab = 20
        logits = paddle.full([1, vocab], -100.0)
        logits[0, 0] = 3.0
        logits[0, 1] = 2.0
        logits[0, 2] = 1.0
        allowed = {0, 1, 2}
        for _ in range(50):
            tok = _sample_next_token(
                logits, temperature=1.0, top_k=3, top_p=0.0
            )
            self.assertIn(int(tok[0, 0]), allowed)

    def test_top_k_equal_vocab_no_filtering(self):
        vocab = 8
        logits = paddle.randn([1, vocab])
        tok = _sample_next_token(
            logits, temperature=1.0, top_k=vocab, top_p=0.0
        )
        self.assertTrue(0 <= int(tok[0, 0]) < vocab)

    def test_top_k_one_is_deterministic(self):
        logits = paddle.randn([1, 16])
        expected = int(logits[0].argmax())
        tok = _sample_next_token(logits, temperature=1.0, top_k=1, top_p=0.0)
        self.assertEqual(int(tok[0, 0]), expected)


class TestSampleNextTokenTopP(unittest.TestCase):
    def test_top_p_near_zero_picks_top_token(self):
        vocab = 20
        logits = paddle.full([1, vocab], -100.0)
        logits[0, 4] = 10.0
        for _ in range(20):
            tok = _sample_next_token(
                logits, temperature=1.0, top_k=0, top_p=0.01
            )
            self.assertEqual(int(tok[0, 0]), 4)

    def test_top_p_one_allows_all_tokens(self):
        logits = paddle.randn([1, 10])
        tok = _sample_next_token(logits, temperature=1.0, top_k=0, top_p=1.0)
        self.assertTrue(0 <= int(tok[0, 0]) < 10)

    def test_top_p_disabled_when_zero(self):
        logits = paddle.randn([1, 10])
        tok = _sample_next_token(logits, temperature=1.0, top_k=0, top_p=0.0)
        self.assertTrue(0 <= int(tok[0, 0]) < 10)


class TestSampleNextTokenTemperature(unittest.TestCase):
    def _unique_count(self, temperature: float, n_runs: int = 200) -> int:
        vocab = 10
        logits = paddle.zeros([1, vocab])
        logits[0, 0] = 2.0
        counts = set()
        for _ in range(n_runs):
            tok = _sample_next_token(
                logits, temperature=temperature, top_k=0, top_p=0.0
            )
            counts.add(int(tok[0, 0]))
        return len(counts)

    def test_high_temperature_more_diverse(self):
        self.assertGreaterEqual(
            self._unique_count(5.0), self._unique_count(0.1)
        )

    def test_output_shape_preserved(self):
        logits = paddle.randn([4, 32])
        tok = _sample_next_token(logits, temperature=0.7, top_k=0, top_p=0.0)
        self.assertEqual(tok.shape, [4, 1])


class TestSampleNextTokenEdgeCases(unittest.TestCase):
    def test_single_token_vocab(self):
        tok = _sample_next_token(
            paddle.zeros([1, 1]), temperature=1.0, top_k=0, top_p=0.0
        )
        self.assertEqual(int(tok[0, 0]), 0)

    def test_output_within_vocab(self):
        vocab = 50
        tok = _sample_next_token(
            paddle.randn([2, vocab]), temperature=0.8, top_k=10, top_p=0.0
        )
        for i in range(2):
            self.assertTrue(0 <= int(tok[i, 0]) < vocab)

    def test_top_k_and_top_p_combined(self):
        tok = _sample_next_token(
            paddle.randn([1, 30]), temperature=1.0, top_k=10, top_p=0.9
        )
        self.assertTrue(0 <= int(tok[0, 0]) < 30)


# ===========================================================================
# Tests for GreedyGenerator.generate with real model
# ===========================================================================


class TestGreedyGeneratorGenerate(unittest.TestCase):
    """End-to-end tests for generate() with temperature/top_k/top_p."""

    @classmethod
    def setUpClass(cls):
        cls.vocab_size = 64
        cls.model, cls.config = _make_model(vocab_size=cls.vocab_size)
        cls.gen = GreedyGenerator(cls.model)
        cls.eos_id = 2
        cls.input_ids = paddle.to_tensor(
            [[1, 5, 10, 3]], dtype="int64"
        )  # [1, 4]

    def _run(self, **kwargs):
        return self.gen.generate(
            self.input_ids, max_new_tokens=8, eos_token_id=self.eos_id, **kwargs
        )

    # --- output shape ---

    def test_greedy_output_shape(self):
        out = self._run()
        self.assertEqual(out.shape[0], 1)
        self.assertGreaterEqual(out.shape[1], self.input_ids.shape[1] + 1)

    def test_temperature_output_shape(self):
        out = self._run(temperature=0.8)
        self.assertEqual(out.shape[0], 1)
        self.assertGreaterEqual(out.shape[1], self.input_ids.shape[1] + 1)

    def test_top_k_output_shape(self):
        out = self._run(temperature=1.0, top_k=5)
        self.assertEqual(out.shape[0], 1)
        self.assertGreaterEqual(out.shape[1], self.input_ids.shape[1] + 1)

    def test_top_p_output_shape(self):
        out = self._run(temperature=1.0, top_p=0.9)
        self.assertEqual(out.shape[0], 1)
        self.assertGreaterEqual(out.shape[1], self.input_ids.shape[1] + 1)

    # --- prompt tokens preserved ---

    def test_prompt_tokens_preserved_greedy(self):
        out = self._run()
        prompt_len = self.input_ids.shape[1]
        self.assertEqual(out[:, :prompt_len].tolist(), self.input_ids.tolist())

    def test_prompt_tokens_preserved_sampling(self):
        out = self._run(temperature=0.7, top_k=10)
        prompt_len = self.input_ids.shape[1]
        self.assertEqual(out[:, :prompt_len].tolist(), self.input_ids.tolist())

    # --- generated tokens within vocab ---

    def test_generated_tokens_in_vocab(self):
        out = self._run(temperature=1.0, top_k=8)
        gen_tokens = out[0, self.input_ids.shape[1] :].tolist()
        for t in gen_tokens:
            self.assertTrue(
                0 <= t < self.vocab_size, f"token {t} out of vocab range"
            )

    # --- greedy is deterministic ---

    def test_greedy_deterministic(self):
        """Two greedy runs with the same input must produce identical output."""
        out1 = self._run()
        out2 = self._run()
        self.assertEqual(out1.tolist(), out2.tolist())

    # --- sampling produces variation ---

    def test_sampling_produces_variation(self):
        """High-temperature sampling over many runs should see at least 2 unique sequences."""
        results = set()
        for _ in range(20):
            out = self._run(temperature=2.0, top_k=0, top_p=0.0)
            results.add(tuple(out[0].tolist()))
        self.assertGreater(
            len(results),
            1,
            "sampling with high temperature should produce diverse outputs",
        )

    # --- max_new_tokens respected ---

    def test_max_new_tokens_not_exceeded(self):
        max_new = 5
        out = self.gen.generate(
            self.input_ids,
            max_new_tokens=max_new,
            temperature=1.0,
            top_k=0,
            top_p=0.0,
        )
        generated = out.shape[1] - self.input_ids.shape[1]
        self.assertLessEqual(generated, max_new)

    # --- eos stops generation ---

    def test_eos_stops_generation(self):
        """If the model happens to emit eos quickly, output should be shorter than max."""
        # We can't control when eos fires, but we can verify the function terminates.
        out = self.gen.generate(
            self.input_ids,
            max_new_tokens=32,
            eos_token_id=self.eos_id,
            temperature=0.0,
        )
        self.assertLessEqual(out.shape[1], self.input_ids.shape[1] + 32)


if __name__ == "__main__":
    unittest.main()
