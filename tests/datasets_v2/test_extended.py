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

"""Extended tests for datasets_v2 preprocessors and encode_pt.

Covers: TextPreprocessor, edge cases, encode_pt, collate dict input,
and streaming dataset integration.

Run with: python -m pytest tests/datasets_v2/test_extended.py -v
"""

import importlib

# Workaround: broken torchcodec residual in env
_original_find_spec = importlib.util.find_spec


def _patched_find_spec(name, *args, **kwargs):
    if name == "torchcodec":
        return None
    return _original_find_spec(name, *args, **kwargs)


importlib.util.find_spec = _patched_find_spec

import pytest
from datasets import Dataset

from paddleformers.datasets_v2.preprocessors import (
    AutoPreprocessor,
    MessagesPreprocessor,
    ResponsePreprocessor,
)
from paddleformers.datasets_v2.preprocessors.extra import TextPreprocessor


def make_dataset(data: dict) -> Dataset:
    """Helper: create a HF Dataset from a dict of columns."""
    return Dataset.from_dict(data)


# ============================================================
# TextPreprocessor
# ============================================================


class TestTextPreprocessor:
    def test_basic_text(self):
        """Basic text column -> messages."""
        ds = make_dataset({"text": ["Hello world", "Another text"]})
        preprocessor = TextPreprocessor()
        result = preprocessor(ds)

        assert "messages" in result.column_names
        row = result[0]
        assert row["messages"] == [{"role": "user", "content": "Hello world"}]

    def test_empty_text_filtered(self):
        """Empty text rows should be filtered out."""
        ds = make_dataset({"text": ["Valid text", "", "   ", "Also valid"]})
        preprocessor = TextPreprocessor()
        result = preprocessor(ds)
        assert len(result) == 2

    def test_none_text_filtered(self):
        """None text rows should be filtered out."""
        ds = make_dataset({"text": ["Valid", None]})
        preprocessor = TextPreprocessor()
        result = preprocessor(ds)
        assert len(result) == 1

    def test_content_column(self):
        """Should also work with 'content' column."""
        ds = make_dataset({"content": ["Text from content column"]})
        preprocessor = TextPreprocessor()
        result = preprocessor(ds)
        assert len(result) == 1
        assert result[0]["messages"][0]["content"] == "Text from content column"

    def test_non_string_filtered(self):
        """Non-string values should be filtered."""
        # HF Dataset enforces uniform types, so test via the preprocessor directly
        preprocessor = TextPreprocessor()
        assert preprocessor.preprocess({"text": 123}) is None
        assert preprocessor.preprocess({"text": ["list"]}) is None
        assert preprocessor.preprocess({"text": "valid"}) is not None


class TestAutoDetectsText:
    def test_auto_detects_text_only(self):
        """AutoPreprocessor should pick TextPreprocessor for text-only dataset."""
        ds = make_dataset({"text": ["Hello world"], "id": ["001"], "url": ["http://example.com"]})
        result = AutoPreprocessor()(ds)
        assert "messages" in result.column_names
        assert result[0]["messages"][0]["content"] == "Hello world"

    def test_auto_text_with_sft_indicators_uses_response(self):
        """If 'text' exists but also SFT indicators, should NOT pick TextPreprocessor."""
        ds = make_dataset({"text": ["answer text"], "query": ["question"]})
        result = AutoPreprocessor()(ds)
        # Should use ResponsePreprocessor, not TextPreprocessor
        assert "messages" in result.column_names
        # The query should be the user message
        assert result[0]["messages"][0]["content"] == "question"


# ============================================================
# MessagesPreprocessor edge cases
# ============================================================


class TestMessagesEdgeCases:
    def test_stringified_messages(self):
        """Messages stored as string should be parsed via ast.literal_eval."""
        ds = make_dataset(
            {"messages": ['[{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]']}
        )
        preprocessor = MessagesPreprocessor()
        result = preprocessor(ds)
        assert result[0]["messages"][0]["content"] == "Hi"

    def test_system_in_row(self):
        """System prompt from row-level 'system' field."""
        ds = make_dataset(
            {
                "messages": [[{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]],
                "system": ["Be concise"],
            }
        )
        preprocessor = MessagesPreprocessor()
        result = preprocessor(ds)
        row = result[0]
        assert row["messages"][0] == {"role": "system", "content": "Be concise"}
        assert row["messages"][1] == {"role": "user", "content": "Hi"}

    def test_empty_messages_filtered(self):
        """Empty/null messages should be filtered."""
        ds = make_dataset(
            {
                "messages": [
                    [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}],
                    [],
                ]
            }
        )
        preprocessor = MessagesPreprocessor()
        result = preprocessor(ds)
        assert len(result) == 1


# ============================================================
# ResponsePreprocessor edge cases
# ============================================================


class TestResponseEdgeCases:
    def test_response_is_list(self):
        """Response as list should take first element."""
        ds = make_dataset(
            {
                "query": ["What?"],
                "response": [["Answer 1", "Answer 2"]],
            }
        )
        preprocessor = ResponsePreprocessor()
        result = preprocessor(ds)
        assert result[0]["messages"][1]["content"] == "Answer 1"

    def test_multiple_key_aliases(self):
        """Various column name aliases should work."""
        # 'prompt' -> query, 'answer' -> response
        ds = make_dataset({"prompt": ["Q"], "answer": ["A"]})
        preprocessor = ResponsePreprocessor()
        result = preprocessor(ds)
        assert result[0]["messages"][0]["content"] == "Q"
        assert result[0]["messages"][1]["content"] == "A"


# ============================================================
# BasePreprocessor internals
# ============================================================


class TestBasePreprocessorInternals:
    def test_batched_to_rows_roundtrip(self):
        """_batched_to_rows and _rows_to_batched should roundtrip."""
        from paddleformers.datasets_v2.preprocessors.base import BasePreprocessor

        batched = {"a": [1, 2, 3], "b": ["x", "y", "z"]}
        rows = BasePreprocessor._batched_to_rows(batched)
        assert rows == [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}, {"a": 3, "b": "z"}]

        rebatched = BasePreprocessor._rows_to_batched(rows)
        assert rebatched == batched

    def test_batched_to_rows_empty(self):
        """Empty batch should return empty list."""
        from paddleformers.datasets_v2.preprocessors.base import BasePreprocessor

        batched = {"a": [], "b": []}
        rows = BasePreprocessor._batched_to_rows(batched)
        assert rows == []

    def test_rows_to_batched_empty(self):
        """Empty rows should return empty dict."""
        from paddleformers.datasets_v2.preprocessors.base import BasePreprocessor

        result = BasePreprocessor._rows_to_batched([])
        assert result == {}

    def test_columns_to_remove(self):
        """Should remove non-standard columns."""
        from paddleformers.datasets_v2.preprocessors.base import BasePreprocessor

        class FakeDataset:
            column_names = ["messages", "images", "custom_col", "id"]

        result = BasePreprocessor._columns_to_remove(FakeDataset())
        assert "custom_col" in result
        assert "id" in result
        assert "messages" not in result
        assert "images" not in result

    def test_strict_mode_raises(self):
        """Strict mode should propagate errors."""
        from paddleformers.datasets_v2.preprocessors.base import BasePreprocessor

        class BrokenPreprocessor(BasePreprocessor):
            def preprocess(self, row):
                raise ValueError("intentional error")

        ds = make_dataset({"query": ["test"], "response": ["answer"]})
        preprocessor = BrokenPreprocessor()
        with pytest.raises(ValueError, match="intentional error"):
            preprocessor(ds, strict=True)

    def test_non_strict_mode_skips_errors(self):
        """Non-strict mode should skip bad rows."""
        from paddleformers.datasets_v2.preprocessors.base import BasePreprocessor

        call_count = [0]

        class FlakeyPreprocessor(BasePreprocessor):
            def preprocess(self, row):
                call_count[0] += 1
                if call_count[0] == 1:
                    raise ValueError("bad row")
                return {"messages": [{"role": "user", "content": "ok"}, {"role": "assistant", "content": "ok"}]}

        ds = make_dataset({"data": ["a", "b"]})
        preprocessor = FlakeyPreprocessor()
        result = preprocessor(ds, strict=False)
        # First row raises (skipped), second succeeds
        assert len(result) == 1


# ============================================================
# encode_pt
# ============================================================


class MockTokenizer:
    """Minimal tokenizer mock."""

    def __init__(self, eos_token_id=2):
        self.bos_token_id = 1
        self.eos_token_id = eos_token_id
        self.pad_token_id = 0

    def tokenize(self, text):
        return list(text)

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return ord(tokens) + 100
        return [ord(t) + 100 for t in tokens]

    def encode(self, text, add_special_tokens=False):
        return [ord(c) + 100 for c in text]


class TestEncodePt:
    def test_basic(self):
        """Basic pretraining encoding."""
        from paddleformers.datasets_v2.datapipe import EncodeConfig, encode_pt

        tokenizer = MockTokenizer()
        config = EncodeConfig(max_seq_len=100)
        example = {"messages": [{"role": "user", "content": "Hello world"}]}

        result = encode_pt(example, tokenizer, config)
        assert result is not None
        assert len(result.input_ids) == len(result.labels)
        assert result.seq_len == len(result.input_ids)
        # Labels should be shifted: labels[i] == input_ids[i+1] (for original tokens)
        # Because encode_pt does: input_ids = tokens[:-1], labels = tokens[1:]

    def test_truncation(self):
        """Long text should be truncated to max_seq_len."""
        from paddleformers.datasets_v2.datapipe import EncodeConfig, encode_pt

        tokenizer = MockTokenizer()
        config = EncodeConfig(max_seq_len=5)
        example = {"messages": [{"role": "user", "content": "A very long text here"}]}

        result = encode_pt(example, tokenizer, config)
        assert result is not None
        assert result.seq_len <= 5

    def test_empty_messages_returns_none(self):
        """Empty messages should return None."""
        from paddleformers.datasets_v2.datapipe import EncodeConfig, encode_pt

        tokenizer = MockTokenizer()
        config = EncodeConfig(max_seq_len=100)
        assert encode_pt({"messages": []}, tokenizer, config) is None
        assert encode_pt({"messages": None}, tokenizer, config) is None
        assert encode_pt({}, tokenizer, config) is None

    def test_empty_content_returns_none(self):
        """Empty content should return None."""
        from paddleformers.datasets_v2.datapipe import EncodeConfig, encode_pt

        tokenizer = MockTokenizer()
        config = EncodeConfig(max_seq_len=100)
        example = {"messages": [{"role": "user", "content": ""}]}
        assert encode_pt(example, tokenizer, config) is None

    def test_eos_appended(self):
        """EOS should be appended at the end."""
        from paddleformers.datasets_v2.datapipe import EncodeConfig, encode_pt

        tokenizer = MockTokenizer(eos_token_id=2)
        config = EncodeConfig(max_seq_len=100)
        example = {"messages": [{"role": "user", "content": "AB"}]}

        result = encode_pt(example, tokenizer, config)
        # tokens = [A_id, B_id, eos_id]
        # input_ids = tokens[:-1] = [A_id, B_id]
        # labels = tokens[1:] = [B_id, eos_id]
        assert result.labels[-1] == 2  # last label is EOS

    def test_eos_token_id_none(self):
        """When eos_token_id is None, should handle gracefully."""
        from paddleformers.datasets_v2.datapipe import EncodeConfig, encode_pt

        tokenizer = MockTokenizer(eos_token_id=None)
        config = EncodeConfig(max_seq_len=100)
        example = {"messages": [{"role": "user", "content": "Hello"}]}

        # This tests the bug: eos_token_id=None appends None to the list
        # After fix, it should either skip EOS or handle gracefully
        result = encode_pt(example, tokenizer, config)
        # If not fixed, result.input_ids would contain None → downstream crash
        # If fixed, result should be valid
        if result is not None:
            assert None not in result.input_ids
            assert None not in result.labels

    def test_label_shift(self):
        """Labels should be next-token prediction (shifted by 1)."""
        from paddleformers.datasets_v2.datapipe import EncodeConfig, encode_pt

        tokenizer = MockTokenizer(eos_token_id=2)
        config = EncodeConfig(max_seq_len=100)
        example = {"messages": [{"role": "user", "content": "ABC"}]}

        result = encode_pt(example, tokenizer, config)
        # tokens = [A_id, B_id, C_id, eos_id]
        # input_ids = tokens[:-1] = [A_id, B_id, C_id]
        # labels = tokens[1:] = [B_id, C_id, eos_id]
        assert result.input_ids[0] == ord("A") + 100
        assert result.labels[0] == ord("B") + 100
        assert result.labels[1] == ord("C") + 100
        assert result.labels[2] == 2  # eos


# ============================================================
# collate_sft with dict input (streaming mode)
# ============================================================


class TestCollateDictInput:
    def test_dict_batch_converted(self):
        """Dict batch should be converted to EncodedSample."""
        from paddleformers.datasets_v2.datapipe import collate_sft

        batch = [
            {"input_ids": [1, 2, 3], "labels": [-100, 2, 3], "seq_len": 3},
            {"input_ids": [4, 5], "labels": [-100, 5], "seq_len": 2},
        ]
        out = collate_sft(batch, pad_token_id=0, max_seq_len=10, packing=False)
        assert out["input_ids"].shape == (2, 3)
        assert out["labels"].shape == (2, 3)

    def test_dict_batch_packed(self):
        """Dict batch with packing should work."""
        from paddleformers.datasets_v2.datapipe import collate_sft

        batch = [
            {"input_ids": [1, 2], "labels": [1, 2], "seq_len": 2},
            {"input_ids": [3, 4], "labels": [3, 4], "seq_len": 2},
        ]
        out = collate_sft(batch, pad_token_id=0, max_seq_len=4, packing=True)
        assert out["input_ids"].shape[1] == 4

    def test_empty_batch(self):
        """Empty batch should not crash (or raise a clear error)."""
        from paddleformers.datasets_v2.datapipe import collate_sft

        # After fix, this should either return empty arrays or raise ValueError
        with pytest.raises((ValueError, IndexError)):
            collate_sft([], pad_token_id=0, max_seq_len=10)


# ============================================================
# StreamingDataset
# ============================================================


class TestStreamingDataset:
    def test_basic_iteration(self):
        """StreamingDataset should yield all items."""
        from paddleformers.datasets_v2.dataset import StreamingDataset

        class FakeIterable:
            def __iter__(self):
                for i in range(5):
                    yield {"input_ids": [i], "labels": [i], "seq_len": 1}

        ds = StreamingDataset(FakeIterable())
        items = []
        for item in ds:
            items.append(item)
        assert len(items) == 5
        assert items[0] == {"input_ids": [0], "labels": [0], "seq_len": 1}

    def test_empty_iteration(self):
        """Empty iterable should yield nothing."""
        from paddleformers.datasets_v2.dataset import StreamingDataset

        class EmptyIterable:
            def __iter__(self):
                return iter([])

        ds = StreamingDataset(EmptyIterable())
        items = []
        for item in ds:
            items.append(item)
        assert items == []


# ============================================================
# encode_sft edge cases
# ============================================================


class TestEncodeSftEdgeCases:
    def test_left_truncation(self):
        """Left truncation should keep the end of the sequence."""
        from paddleformers.datasets_v2.datapipe import (
            EncodeConfig,
            encode_sft,
            get_template,
        )

        tokenizer = MockTokenizer()
        tpl = get_template("chatml")
        config = EncodeConfig(max_seq_len=10, truncation="left")
        example = {
            "messages": [
                {"role": "user", "content": "A very long prompt here"},
                {"role": "assistant", "content": "Response"},
            ]
        }
        result = encode_sft(example, tokenizer, tpl, config)
        assert result is not None
        assert result.seq_len <= 10

    def test_multi_turn_loss_mask(self):
        """Multi-turn with selective loss masking."""
        from paddleformers.datasets_v2.datapipe import EncodeConfig, encode_sft
        from paddleformers.datasets_v2.datapipe.template import (
            get_template,
            register_template,
        )

        register_template(
            "test_simple",
            user=["{{content}}"],
            assistant=["{{content}}"],
            efficient_eos=False,
            exist_ok=True,
        )
        tokenizer = MockTokenizer()
        tpl = get_template("test_simple")
        config = EncodeConfig(max_seq_len=4096, label_shift=False)

        example = {
            "messages": [
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "A1", "loss": False},
                {"role": "user", "content": "Q2"},
                {"role": "assistant", "content": "A2", "loss": True},
            ]
        }
        result = encode_sft(example, tokenizer, tpl, config)
        assert result is not None
        # A1 should have -100 labels, A2 should have real labels
        # Find where A2 tokens are (they come after Q2 tokens)
        a2_ids = tokenizer.encode("A2")
        # The last len(a2_ids) labels should NOT be -100
        last_labels = result.labels[-len(a2_ids) :]
        assert all(l != -100 for l in last_labels)


# ============================================================
# _extract_loss_mask
# ============================================================


class TestExtractLossMask:
    def test_default_true(self):
        """Assistant messages default to loss=True."""
        from paddleformers.datasets_v2.datapipe.encode import _extract_loss_mask

        messages = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
        ]
        mask = _extract_loss_mask(messages)
        assert mask == [True]

    def test_explicit_false(self):
        """Explicit loss=False should be respected."""
        from paddleformers.datasets_v2.datapipe.encode import _extract_loss_mask

        messages = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A", "loss": False},
        ]
        mask = _extract_loss_mask(messages)
        assert mask == [False]

    def test_multi_turn(self):
        """Multiple assistant turns."""
        from paddleformers.datasets_v2.datapipe.encode import _extract_loss_mask

        messages = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1", "loss": False},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
        ]
        mask = _extract_loss_mask(messages)
        assert mask == [False, True]

    def test_non_assistant_ignored(self):
        """Non-assistant messages don't contribute to mask."""
        from paddleformers.datasets_v2.datapipe.encode import _extract_loss_mask

        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
        ]
        mask = _extract_loss_mask(messages)
        assert mask == [True]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
