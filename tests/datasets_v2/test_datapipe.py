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

"""Tests for datasets_v2/datapipe and datasets_v2/dataset."""

import pytest

# ============================================================
# Mock tokenizer for testing (no real model needed)
# ============================================================


class MockTokenizer:
    """Minimal tokenizer mock that maps chars to IDs."""

    def __init__(self):
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.pad_token_id = 0
        self._vocab = {"<|im_start|>": 100, "<|im_end|>": 101, "<|endoftext|>": 102}

    def encode(self, text, add_special_tokens=False):
        # Simple: each character becomes an ID (ord + 1000)
        return [ord(c) + 1000 for c in text]

    def convert_tokens_to_ids(self, token):
        return self._vocab.get(token, 999)


# ============================================================
# Test template.py
# ============================================================


class TestTemplate:
    def test_register_and_get(self):
        from paddleformers.datasets_v2.datapipe.template import (
            get_template,
            list_templates,
            register_template,
        )

        register_template(
            "test_tpl",
            user=["USER:{{content}}\nBOT:"],
            assistant=["{{content}}\n"],
            exist_ok=True,
        )
        tpl = get_template("test_tpl")
        assert tpl.name == "test_tpl"
        assert "test_tpl" in list_templates()

    def test_get_nonexistent_raises(self):
        from paddleformers.datasets_v2.datapipe.template import get_template

        with pytest.raises(KeyError):
            get_template("nonexistent_xyz")

    def test_builtin_templates_exist(self):
        from paddleformers.datasets_v2.datapipe.template import list_templates

        templates = list_templates()
        assert "chatml" in templates
        assert "llama3" in templates
        assert "deepseek3" in templates

    def test_encode_multiturn_basic(self):
        from paddleformers.datasets_v2.datapipe.template import (
            encode_multiturn,
            get_template,
        )

        tpl = get_template("chatml")
        tokenizer = MockTokenizer()
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        pairs = encode_multiturn(tpl, tokenizer, messages)
        assert len(pairs) == 1
        prompt_ids, response_ids = pairs[0]
        assert len(prompt_ids) > 0
        assert len(response_ids) > 0

    def test_encode_multiturn_with_system(self):
        from paddleformers.datasets_v2.datapipe.template import (
            encode_multiturn,
            get_template,
        )

        tpl = get_template("chatml")
        tokenizer = MockTokenizer()
        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        pairs = encode_multiturn(tpl, tokenizer, messages)
        assert len(pairs) == 1
        # System is prepended to first prompt
        prompt_ids, _ = pairs[0]
        assert len(prompt_ids) > 0

    def test_encode_multiturn_multi_turn(self):
        from paddleformers.datasets_v2.datapipe.template import (
            encode_multiturn,
            get_template,
        )

        tpl = get_template("chatml")
        tokenizer = MockTokenizer()
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "Bye"},
            {"role": "assistant", "content": "See ya"},
        ]
        pairs = encode_multiturn(tpl, tokenizer, messages)
        assert len(pairs) == 2

    def test_slot_with_dict_token(self):
        from paddleformers.datasets_v2.datapipe.template import (
            encode_multiturn,
            get_template,
            register_template,
        )

        register_template(
            "test_dict_slot",
            prefix=[{"token": "<|im_start|>"}],
            user=["{{content}}"],
            assistant=["{{content}}"],
            exist_ok=True,
        )
        tpl = get_template("test_dict_slot")
        tokenizer = MockTokenizer()
        messages = [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "B"},
        ]
        pairs = encode_multiturn(tpl, tokenizer, messages)
        prompt_ids, _ = pairs[0]
        # First token should be the dict-token ID (100 for <|im_start|>)
        assert prompt_ids[0] == 100

    def test_slot_with_set_bos(self):
        from paddleformers.datasets_v2.datapipe.template import (
            encode_multiturn,
            get_template,
            register_template,
        )

        register_template(
            "test_set_slot",
            prefix=[{"bos_token"}],
            user=["{{content}}"],
            assistant=["{{content}}"],
            exist_ok=True,
        )
        tpl = get_template("test_set_slot")
        tokenizer = MockTokenizer()
        messages = [
            {"role": "user", "content": "X"},
            {"role": "assistant", "content": "Y"},
        ]
        pairs = encode_multiturn(tpl, tokenizer, messages)
        prompt_ids, _ = pairs[0]
        assert prompt_ids[0] == tokenizer.bos_token_id


# ============================================================
# Test encode.py
# ============================================================


class TestEncode:
    def test_encode_sft_basic(self):
        from paddleformers.datasets_v2.datapipe import (
            EncodeConfig,
            encode_sft,
            get_template,
        )

        tokenizer = MockTokenizer()
        tpl = get_template("chatml")
        config = EncodeConfig(max_seq_len=4096)
        example = {
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ]
        }
        sample = encode_sft(example, tokenizer, tpl, config)
        assert sample is not None
        assert len(sample.input_ids) == len(sample.labels)
        assert sample.seq_len == len(sample.input_ids)
        # Prompt positions should have -100 labels
        assert -100 in sample.labels

    def test_encode_sft_truncation(self):
        from paddleformers.datasets_v2.datapipe import (
            EncodeConfig,
            encode_sft,
            get_template,
        )

        tokenizer = MockTokenizer()
        tpl = get_template("chatml")
        config = EncodeConfig(max_seq_len=10)
        example = {
            "messages": [
                {"role": "user", "content": "This is a long message"},
                {"role": "assistant", "content": "This is also long"},
            ]
        }
        sample = encode_sft(example, tokenizer, tpl, config)
        assert sample is not None
        assert sample.seq_len <= 10

    def test_encode_sft_loss_mask(self):
        from paddleformers.datasets_v2.datapipe import EncodeConfig, encode_sft
        from paddleformers.datasets_v2.datapipe.template import (
            get_template,
            register_template,
        )

        register_template(
            "test_no_eos",
            user=["{{content}}"],
            assistant=["{{content}}"],
            efficient_eos=False,
            exist_ok=True,
        )
        tokenizer = MockTokenizer()
        tpl = get_template("test_no_eos")
        config = EncodeConfig(max_seq_len=4096, label_shift=False)
        example = {
            "messages": [
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": "A", "loss": False},
            ]
        }
        sample = encode_sft(example, tokenizer, tpl, config)
        assert sample is not None
        # All labels should be -100 (user has no loss, assistant has loss=False)
        assert all(l == -100 for l in sample.labels)

    def test_encode_sft_empty_messages_returns_none(self):
        from paddleformers.datasets_v2.datapipe import (
            EncodeConfig,
            encode_sft,
            get_template,
        )

        tokenizer = MockTokenizer()
        tpl = get_template("chatml")
        config = EncodeConfig(max_seq_len=4096)
        assert encode_sft({"messages": []}, tokenizer, tpl, config) is None
        assert encode_sft({"messages": [{"role": "user", "content": "hi"}]}, tokenizer, tpl, config) is None

    def test_encode_sft_eos_appended(self):
        from paddleformers.datasets_v2.datapipe import (
            EncodeConfig,
            encode_sft,
            get_template,
        )

        tokenizer = MockTokenizer()
        tpl = get_template("chatml")  # efficient_eos=True
        config = EncodeConfig(max_seq_len=4096, label_shift=False)
        example = {
            "messages": [
                {"role": "user", "content": "A"},
                {"role": "assistant", "content": "B"},
            ]
        }
        sample = encode_sft(example, tokenizer, tpl, config)
        # Last token should be EOS
        assert sample.input_ids[-1] == tokenizer.eos_token_id


# ============================================================
# Test packing.py
# ============================================================


class TestPacking:
    def test_greedy_pack_basic(self):
        from paddleformers.datasets_v2.datapipe import EncodedSample, greedy_pack

        samples = [
            EncodedSample(input_ids=[1, 2, 3], labels=[1, 2, 3], seq_len=3),
            EncodedSample(input_ids=[4, 5], labels=[4, 5], seq_len=2),
            EncodedSample(input_ids=[6, 7, 8], labels=[6, 7, 8], seq_len=3),
        ]
        groups = greedy_pack(samples, max_seq_len=5)
        # Sample 0 (3) + Sample 1 (2) = 5, fits in one bin
        # Sample 2 (3) in another bin
        assert len(groups) == 2
        total_samples = sum(len(g) for g in groups)
        assert total_samples == 3

    def test_greedy_pack_all_fit(self):
        from paddleformers.datasets_v2.datapipe import EncodedSample, greedy_pack

        samples = [
            EncodedSample(input_ids=[1], labels=[1], seq_len=1),
            EncodedSample(input_ids=[2], labels=[2], seq_len=1),
            EncodedSample(input_ids=[3], labels=[3], seq_len=1),
        ]
        groups = greedy_pack(samples, max_seq_len=10)
        # All fit in one bin
        assert len(groups) == 1
        assert len(groups[0]) == 3

    def test_greedy_pack_each_alone(self):
        from paddleformers.datasets_v2.datapipe import EncodedSample, greedy_pack

        samples = [
            EncodedSample(input_ids=[1, 2, 3, 4, 5], labels=[0] * 5, seq_len=5),
            EncodedSample(input_ids=[1, 2, 3, 4, 5], labels=[0] * 5, seq_len=5),
        ]
        groups = greedy_pack(samples, max_seq_len=5)
        # Each sample fills a bin exactly
        assert len(groups) == 2

    def test_greedy_pack_empty(self):
        from paddleformers.datasets_v2.datapipe import greedy_pack

        assert greedy_pack([], max_seq_len=10) == []


# ============================================================
# Test collate.py
# ============================================================


class TestCollate:
    def test_collate_simple(self):
        from paddleformers.datasets_v2.datapipe import EncodedSample, collate_sft

        batch = [
            EncodedSample(input_ids=[1, 2, 3], labels=[-100, 2, 3], seq_len=3),
            EncodedSample(input_ids=[4, 5], labels=[-100, 5], seq_len=2),
        ]
        out = collate_sft(batch, pad_token_id=0, max_seq_len=10, packing=False)
        assert out["input_ids"].shape == (2, 3)  # padded to max in batch
        assert out["labels"].shape == (2, 3)
        assert out["position_ids"].shape == (2, 3)
        assert out["attention_mask"].shape == (2, 1, 3, 3)
        # Check padding
        assert out["input_ids"][1, 2] == 0  # pad_token_id
        assert out["labels"][1, 2] == -100

    def test_collate_packed(self):
        from paddleformers.datasets_v2.datapipe import EncodedSample, collate_sft

        batch = [
            EncodedSample(input_ids=[1, 2, 3], labels=[-100, 2, 3], seq_len=3),
            EncodedSample(input_ids=[4, 5], labels=[-100, 5], seq_len=2),
            EncodedSample(input_ids=[6, 7, 8], labels=[-100, 7, 8], seq_len=3),
        ]
        out = collate_sft(batch, pad_token_id=0, max_seq_len=8, packing=True)
        # With max_seq_len=8: samples 0(3)+1(2)=5 fit together, sample 2(3) alone
        assert out["input_ids"].shape[1] == 8
        assert out["attention_mask"].shape == (out["input_ids"].shape[0], 1, 8, 8)

    def test_collate_packed_block_diagonal(self):
        from paddleformers.datasets_v2.datapipe import EncodedSample, collate_sft

        batch = [
            EncodedSample(input_ids=[1, 2], labels=[1, 2], seq_len=2),
            EncodedSample(input_ids=[3, 4], labels=[3, 4], seq_len=2),
        ]
        out = collate_sft(batch, pad_token_id=0, max_seq_len=4, packing=True)
        # Both samples packed into one group (2+2=4)
        mask = out["attention_mask"][0, 0]  # [4, 4]
        # Position [2,0] should be 0 (second sub-seq can't attend to first)
        assert mask[2, 0] == 0.0
        assert mask[2, 1] == 0.0
        # Position [2,2] and [3,2] should be 1 (within second block)
        assert mask[2, 2] == 1.0
        assert mask[3, 2] == 1.0

    def test_collate_position_ids_reset(self):
        from paddleformers.datasets_v2.datapipe import EncodedSample, collate_sft

        batch = [
            EncodedSample(input_ids=[1, 2, 3], labels=[1, 2, 3], seq_len=3),
            EncodedSample(input_ids=[4, 5], labels=[4, 5], seq_len=2),
        ]
        out = collate_sft(batch, pad_token_id=0, max_seq_len=5, packing=True)
        pos = out["position_ids"][0]  # packed: [0,1,2, 0,1]
        assert pos[0] == 0
        assert pos[1] == 1
        assert pos[2] == 2
        assert pos[3] == 0  # resets for second sub-sequence
        assert pos[4] == 1


# ============================================================
# Test dataset/lazy_dataset.py
# ============================================================


class TestLazyDataset:
    def test_basic_getitem(self):
        from paddleformers.datasets_v2.datapipe import EncodedSample
        from paddleformers.datasets_v2.dataset import LazyEncodeDataset

        class FakeDataset:
            def __len__(self):
                return 3

            def __getitem__(self, idx):
                return {
                    "messages": [
                        {"role": "user", "content": f"Q{idx}"},
                        {"role": "assistant", "content": f"A{idx}"},
                    ]
                }

        def fake_encode(row):
            return EncodedSample(input_ids=[1, 2, 3], labels=[-100, 2, 3], seq_len=3)

        ds = LazyEncodeDataset(FakeDataset(), fake_encode)
        assert len(ds) == 3
        sample = ds[0]
        assert sample.seq_len == 3

    def test_retry_on_failure(self):
        from paddleformers.datasets_v2.datapipe import EncodedSample
        from paddleformers.datasets_v2.dataset import LazyEncodeDataset

        call_count = [0]

        class FakeDataset:
            def __len__(self):
                return 5

            def __getitem__(self, idx):
                return {"idx": idx}

        def flaky_encode(row):
            call_count[0] += 1
            if call_count[0] <= 1:  # first call fails
                raise ValueError("bad sample")
            return EncodedSample(input_ids=[1], labels=[1], seq_len=1)

        ds = LazyEncodeDataset(FakeDataset(), flaky_encode, seed=42)
        sample = ds[0]
        assert sample is not None
        assert call_count[0] >= 2

    def test_all_retries_fail(self):
        from paddleformers.datasets_v2.dataset import LazyEncodeDataset

        class FakeDataset:
            def __len__(self):
                return 3

            def __getitem__(self, idx):
                return {}

        def always_fail(row):
            raise ValueError("always fails")

        ds = LazyEncodeDataset(FakeDataset(), always_fail, n_try_fetch=3)
        with pytest.raises(RuntimeError, match="Failed to encode"):
            ds[0]


# ============================================================
# Integration test
# ============================================================


class TestEndToEnd:
    def test_full_pipeline(self):
        """Test the complete flow: messages → encode → collate."""
        from functools import partial

        from paddleformers.datasets_v2.datapipe import (
            EncodeConfig,
            collate_sft,
            encode_sft,
            get_template,
        )

        tokenizer = MockTokenizer()
        tpl = get_template("chatml")
        config = EncodeConfig(max_seq_len=256)
        encode_fn = partial(encode_sft, tokenizer=tokenizer, template=tpl, config=config)

        examples = [
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": "4"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "Bye"},
                    {"role": "assistant", "content": "Goodbye"},
                ]
            },
        ]

        samples = [encode_fn(ex) for ex in examples]
        assert all(s is not None for s in samples)

        # Test without packing
        out = collate_sft(samples, pad_token_id=0, max_seq_len=256, packing=False)
        assert out["input_ids"].shape[0] == 3
        assert out["input_ids"].shape[1] <= 256
        assert out["attention_mask"].shape == (3, 1, out["input_ids"].shape[1], out["input_ids"].shape[1])

        # Test with packing
        out_packed = collate_sft(samples, pad_token_id=0, max_seq_len=256, packing=True)
        assert out_packed["input_ids"].shape[1] == 256
        assert "attention_mask" in out_packed

    def test_with_lazy_dataset(self):
        """Test LazyEncodeDataset integration."""
        from functools import partial

        from paddleformers.datasets_v2.datapipe import (
            EncodeConfig,
            collate_sft,
            encode_sft,
            get_template,
        )
        from paddleformers.datasets_v2.dataset import LazyEncodeDataset

        tokenizer = MockTokenizer()
        tpl = get_template("chatml")
        config = EncodeConfig(max_seq_len=256)
        encode_fn = partial(encode_sft, tokenizer=tokenizer, template=tpl, config=config)

        class FakeHFDataset:
            def __init__(self):
                self.data = [
                    {
                        "messages": [
                            {"role": "user", "content": f"Q{i}"},
                            {"role": "assistant", "content": f"A{i}"},
                        ]
                    }
                    for i in range(10)
                ]

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        ds = LazyEncodeDataset(FakeHFDataset(), encode_fn)
        assert len(ds) == 10

        # Simulate a mini batch
        batch = [ds[i] for i in range(4)]
        out = collate_sft(batch, pad_token_id=0, max_seq_len=256, packing=True)
        assert out["input_ids"].ndim == 2
        assert out["labels"].ndim == 2
        assert out["position_ids"].ndim == 2
        assert out["attention_mask"].ndim == 4
