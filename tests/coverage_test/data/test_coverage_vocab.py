# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import collections
import json
import os
import tempfile
import unittest
import warnings

import numpy as np

from paddleformers.data.vocab import Vocab


class TestVocabInitFromCounter(unittest.TestCase):
    """Tests for Vocab initialization from counter."""

    def test_basic_init_from_counter(self):
        counter = collections.Counter(["hello", "world", "hello", "foo"])
        vocab = Vocab(counter)
        self.assertEqual(len(vocab), 3)  # hello (freq 2), foo (freq 1), world (freq 1) - alphabetically
        self.assertIn("hello", vocab)

    def test_init_with_special_tokens(self):
        counter = collections.Counter(["hello", "world"])
        vocab = Vocab(counter, unk_token="<unk>", pad_token="<pad>")
        self.assertEqual(len(vocab), 4)  # 2 special + 2 regular
        self.assertEqual(vocab.unk_token, "<unk>")
        self.assertEqual(vocab.pad_token, "<pad>")

    def test_init_with_all_special_tokens(self):
        counter = collections.Counter(["hello"])
        vocab = Vocab(
            counter,
            unk_token="<unk>",
            pad_token="<pad>",
            bos_token="<bos>",
            eos_token="<eos>",
        )
        self.assertEqual(len(vocab), 5)  # 4 special + 1 regular
        self.assertEqual(vocab.bos_token, "<bos>")
        self.assertEqual(vocab.eos_token, "<eos>")

    def test_init_with_max_size(self):
        counter = collections.Counter(["a", "b", "c", "d", "e"])
        vocab = Vocab(counter, max_size=3)
        # max_size=3 means 3 regular tokens (special tokens not counted)
        self.assertEqual(len(vocab), 3)

    def test_init_with_max_size_and_special_tokens(self):
        counter = collections.Counter(["a", "b", "c", "d", "e"])
        vocab = Vocab(counter, max_size=3, unk_token="<unk>")
        # 1 special + 3 regular
        self.assertEqual(len(vocab), 4)

    def test_init_with_min_freq(self):
        counter = collections.Counter(["a", "a", "a", "b", "b", "c"])
        vocab = Vocab(counter, min_freq=2)
        # Only a and b pass the filter
        self.assertEqual(len(vocab), 2)

    def test_init_with_token_to_idx(self):
        counter = collections.Counter(["a", "b", "c"])
        # Indices must be within range [0, len(vocab)-1] = [0, 2]
        token_to_idx = {"b": 2, "a": 0}
        vocab = Vocab(counter, token_to_idx=token_to_idx)
        self.assertIn("a", vocab)
        self.assertIn("b", vocab)

    def test_init_with_custom_special_token(self):
        counter = collections.Counter(["hello"])
        vocab = Vocab(counter, unk_token="<unk>", mask_token="<mask>")
        self.assertEqual(vocab.mask_token, "<mask>")
        self.assertIn("<mask>", vocab)

    def test_init_invalid_kwarg_name(self):
        counter = collections.Counter(["hello"])
        # "not_a_special" does not end with "_token", should raise ValueError
        with self.assertRaises(ValueError):
            Vocab(counter, not_a_special="value")

    def test_init_underscore_identifier_raises(self):
        counter = collections.Counter(["hello"])
        with self.assertRaises(ValueError):
            Vocab(counter, _private_token="<p>")

    def test_init_duplicate_identifier_raises(self):
        # unk_token is set as an attribute during init, and passing it again via
        # kwargs would raise ValueError because it's already an attribute.
        # We use **kwargs trick: pass unk_token via the combs (positional path)
        # and then try to set another attribute with the same name.
        counter = collections.Counter(["hello"])
        vocab = Vocab(counter, unk_token="<unk>")
        # Verify the attribute exists
        self.assertEqual(vocab.unk_token, "<unk>")
        # Trying to create a vocab with an existing built-in attribute name
        # Use a different identifier that would collide
        counter2 = collections.Counter(["hello"])
        with self.assertRaises(ValueError):
            Vocab(counter2, unk_token="<unk>", _identifiers_to_tokens="<v>")


class TestVocabInitFromTokenToIdx(unittest.TestCase):
    """Tests for Vocab initialization from token_to_idx dict."""

    def test_init_from_token_to_idx(self):
        token_to_idx = {"hello": 0, "world": 1, "foo": 2}
        vocab = Vocab(token_to_idx=token_to_idx)
        self.assertEqual(len(vocab), 3)
        self.assertEqual(vocab.to_indices("hello"), 0)
        self.assertEqual(vocab.to_indices("world"), 1)

    def test_init_from_token_to_idx_with_unk(self):
        token_to_idx = {"<unk>": 0, "hello": 1, "world": 2}
        vocab = Vocab(token_to_idx=token_to_idx, unk_token="<unk>")
        self.assertEqual(len(vocab), 3)
        self.assertEqual(vocab.to_indices("unknown_word"), vocab.to_indices("<unk>"))

    def test_init_from_token_to_idx_missing_special_token_raises(self):
        token_to_idx = {"hello": 0, "world": 1}
        with self.assertRaises(AssertionError):
            Vocab(token_to_idx=token_to_idx, unk_token="<unk_not_in_dict>")

    def test_init_from_token_to_idx_none_raises(self):
        with self.assertRaises(AssertionError):
            Vocab(counter=None, token_to_idx=None)


class TestVocabProperties(unittest.TestCase):
    """Tests for Vocab property accessors."""

    def setUp(self):
        counter = collections.Counter(["hello", "world"])
        self.vocab = Vocab(counter, unk_token="<unk>", pad_token="<pad>")

    def test_idx_to_token(self):
        idx_map = self.vocab.idx_to_token
        self.assertIsInstance(idx_map, dict)

    def test_token_to_idx(self):
        token_map = self.vocab.token_to_idx
        self.assertIsInstance(token_map, dict)

    def test_len(self):
        self.assertEqual(len(self.vocab), 4)  # 2 special + 2 regular

    def test_contains(self):
        self.assertIn("hello", self.vocab)
        self.assertIn("<unk>", self.vocab)
        self.assertNotIn("nonexistent", self.vocab)


class TestVocabIndexing(unittest.TestCase):
    """Tests for Vocab __getitem__ and to_indices."""

    def setUp(self):
        counter = collections.Counter(["hello", "world", "foo"])
        self.vocab = Vocab(counter, unk_token="<unk>")

    def test_getitem_single_token(self):
        idx = self.vocab["hello"]
        self.assertIsInstance(idx, int)

    def test_getitem_unknown_token(self):
        idx = self.vocab["nonexistent"]
        # Should return unk index
        self.assertEqual(idx, self.vocab["<unk>"])

    def test_getitem_list(self):
        indices = self.vocab[["hello", "world"]]
        self.assertIsInstance(indices, list)
        self.assertEqual(len(indices), 2)

    def test_getitem_tuple(self):
        indices = self.vocab[("hello", "world")]
        self.assertIsInstance(indices, list)
        self.assertEqual(len(indices), 2)

    def test_to_indices_single(self):
        idx = self.vocab.to_indices("hello")
        self.assertIsInstance(idx, int)

    def test_to_indices_list(self):
        indices = self.vocab.to_indices(["hello", "world", "foo"])
        self.assertIsInstance(indices, list)
        self.assertEqual(len(indices), 3)

    def test_call(self):
        idx = self.vocab("hello")
        self.assertIsInstance(idx, int)

    def test_call_list(self):
        indices = self.vocab(["hello", "world"])
        self.assertIsInstance(indices, list)


class TestVocabToTokens(unittest.TestCase):
    """Tests for Vocab.to_tokens."""

    def setUp(self):
        counter = collections.Counter(["hello", "world"])
        self.vocab = Vocab(counter, unk_token="<unk>")

    def test_to_tokens_single_int(self):
        token = self.vocab.to_tokens(0)
        self.assertIsInstance(token, str)

    def test_to_tokens_list(self):
        tokens = self.vocab.to_tokens([0, 1])
        self.assertIsInstance(tokens, list)
        self.assertEqual(len(tokens), 2)

    def test_to_tokens_tuple(self):
        tokens = self.vocab.to_tokens((0, 1))
        self.assertIsInstance(tokens, list)

    def test_to_tokens_numpy_1d(self):
        tokens = self.vocab.to_tokens(np.array([0, 1]))
        self.assertIsInstance(tokens, list)
        self.assertEqual(len(tokens), 2)

    def test_to_tokens_numpy_scalar(self):
        tokens = self.vocab.to_tokens(np.int64(0))
        self.assertIsInstance(tokens, str)

    def test_to_tokens_2d_raises(self):
        with self.assertRaises(ValueError):
            self.vocab.to_tokens(np.array([[0, 1], [2, 3]]))

    def test_to_tokens_invalid_index_raises(self):
        with self.assertRaises(ValueError):
            self.vocab.to_tokens(99999)

    def test_to_tokens_non_int_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            token = self.vocab.to_tokens(np.float64(0.0))
            self.assertIsInstance(token, str)
            # Should have warned about type conversion
            self.assertTrue(len(w) > 0)


class TestVocabSortIndex(unittest.TestCase):
    """Tests for _sort_index_according_to_user_specification."""

    def test_sort_with_valid_mapping(self):
        counter = collections.Counter(["a", "b", "c", "d"])
        vocab = Vocab(counter, token_to_idx={"a": 3, "b": 0})
        # a should be at index 3, b at index 0
        self.assertEqual(vocab.to_indices("b"), 0)
        self.assertEqual(vocab.to_indices("a"), 3)

    def test_sort_with_out_of_range_raises(self):
        counter = collections.Counter(["a", "b", "c"])
        with self.assertRaises(ValueError):
            Vocab(counter, token_to_idx={"a": 99})

    def test_sort_with_negative_raises(self):
        counter = collections.Counter(["a", "b", "c"])
        with self.assertRaises(ValueError):
            Vocab(counter, token_to_idx={"a": -1})

    def test_sort_with_duplicate_indices_raises(self):
        counter = collections.Counter(["a", "b", "c"])
        with self.assertRaises(ValueError):
            Vocab(counter, token_to_idx={"a": 0, "b": 0})

    def test_sort_with_unknown_token_raises(self):
        counter = collections.Counter(["a", "b"])
        with self.assertRaises(ValueError):
            Vocab(counter, token_to_idx={"nonexistent": 0})


class TestVocabToJson(unittest.TestCase):
    """Tests for Vocab.to_json and from_json."""

    def setUp(self):
        counter = collections.Counter(["hello", "world", "foo"])
        self.vocab = Vocab(counter, unk_token="<unk>", pad_token="<pad>")

    def test_to_json_returns_string(self):
        json_str = self.vocab.to_json()
        self.assertIsInstance(json_str, str)
        data = json.loads(json_str)
        self.assertIn("idx_to_token", data)
        self.assertIn("token_to_idx", data)
        self.assertEqual(data["unk_token"], "<unk>")

    def test_to_json_save_to_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json_file = f.name

        try:
            self.vocab.to_json(path=json_file)
            self.assertTrue(os.path.exists(json_file))
            with open(json_file, "r") as f:
                data = json.load(f)
            self.assertEqual(data["unk_token"], "<unk>")
        finally:
            os.unlink(json_file)

    def test_from_json_string(self):
        json_str = self.vocab.to_json()
        restored = Vocab.from_json(json_str)
        self.assertEqual(len(restored), len(self.vocab))
        self.assertEqual(restored.unk_token, "<unk>")
        self.assertEqual(restored.pad_token, "<pad>")

    def test_from_json_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(self.vocab.to_json())
            json_file = f.name

        try:
            restored = Vocab.from_json(json_file)
            self.assertEqual(len(restored), len(self.vocab))
        finally:
            os.unlink(json_file)


class TestVocabFromDict(unittest.TestCase):
    """Tests for Vocab.from_dict."""

    def test_from_dict_basic(self):
        token_to_idx = {"hello": 0, "world": 1}
        vocab = Vocab.from_dict(token_to_idx)
        self.assertEqual(len(vocab), 2)
        self.assertEqual(vocab.to_indices("hello"), 0)

    def test_from_dict_with_unk(self):
        token_to_idx = {"<unk>": 0, "hello": 1, "world": 2}
        vocab = Vocab.from_dict(token_to_idx, unk_token="<unk>")
        self.assertEqual(vocab.to_indices("unknown"), 0)

    def test_from_dict_with_all_special_tokens(self):
        token_to_idx = {"<unk>": 0, "<pad>": 1, "<bos>": 2, "<eos>": 3, "hello": 4}
        vocab = Vocab.from_dict(
            token_to_idx,
            unk_token="<unk>",
            pad_token="<pad>",
            bos_token="<bos>",
            eos_token="<eos>",
        )
        self.assertEqual(len(vocab), 5)
        self.assertEqual(vocab.bos_token, "<bos>")
        self.assertEqual(vocab.eos_token, "<eos>")


class TestVocabBuildVocab(unittest.TestCase):
    """Tests for Vocab.build_vocab."""

    def test_build_vocab_basic(self):
        tokens_list = [["hello", "world"], ["hello", "foo", "bar"]]
        vocab = Vocab.build_vocab(tokens_list)
        self.assertGreater(len(vocab), 0)
        self.assertIn("hello", vocab)
        self.assertIn("world", vocab)
        self.assertIn("foo", vocab)
        self.assertIn("bar", vocab)

    def test_build_vocab_with_max_size(self):
        tokens_list = [["a", "b", "c", "d", "e"]]
        vocab = Vocab.build_vocab(tokens_list, max_size=3)
        self.assertEqual(len(vocab), 3)

    def test_build_vocab_with_min_freq(self):
        tokens_list = [["a", "a", "a", "b", "b", "c"]]
        vocab = Vocab.build_vocab(tokens_list, min_freq=2)
        self.assertEqual(len(vocab), 2)
        self.assertIn("a", vocab)
        self.assertIn("b", vocab)
        self.assertNotIn("c", vocab)

    def test_build_vocab_with_unk_token(self):
        tokens_list = [["hello", "world"]]
        vocab = Vocab.build_vocab(tokens_list, unk_token="<unk>")
        self.assertEqual(vocab.to_indices("nonexistent"), vocab.to_indices("<unk>"))

    def test_build_vocab_with_token_to_idx(self):
        tokens_list = [["a", "b", "c", "d"]]
        vocab = Vocab.build_vocab(tokens_list, token_to_idx={"b": 2, "a": 0})
        self.assertEqual(vocab.to_indices("a"), 0)
        self.assertEqual(vocab.to_indices("b"), 2)


class TestVocabLoadSaveVocabulary(unittest.TestCase):
    """Tests for Vocab.load_vocabulary and save_vocabulary."""

    def test_save_and_load_vocabulary(self):
        counter = collections.Counter(["hello", "world", "foo"])
        vocab = Vocab(counter, unk_token="<unk>", pad_token="<pad>")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            filepath = f.name

        try:
            vocab.save_vocabulary(filepath)
            loaded = Vocab.load_vocabulary(filepath, unk_token="<unk>", pad_token="<pad>")
            self.assertEqual(len(loaded), len(vocab))
            self.assertEqual(loaded.unk_token, "<unk>")
            self.assertEqual(loaded.pad_token, "<pad>")
        finally:
            os.unlink(filepath)

    def test_load_vocabulary_basic(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello\nworld\nfoo\n")
            filepath = f.name

        try:
            vocab = Vocab.load_vocabulary(filepath)
            self.assertEqual(len(vocab), 3)
            self.assertEqual(vocab.to_indices("hello"), 0)
            self.assertEqual(vocab.to_indices("world"), 1)
            self.assertEqual(vocab.to_indices("foo"), 2)
        finally:
            os.unlink(filepath)


class TestVocabGetTokenIds(unittest.TestCase):
    """Tests for Vocab get_*_token_id methods."""

    def test_get_unk_token_id(self):
        token_to_idx = {"<unk>": 0, "hello": 1}
        vocab = Vocab.from_dict(token_to_idx, unk_token="<unk>")
        self.assertEqual(vocab.get_unk_token_id(), 0)

    def test_get_unk_token_id_none(self):
        token_to_idx = {"hello": 0, "world": 1}
        vocab = Vocab.from_dict(token_to_idx)
        self.assertIsNone(vocab.get_unk_token_id())

    def test_get_pad_token_id(self):
        token_to_idx = {"<pad>": 5, "hello": 0}
        vocab = Vocab.from_dict(token_to_idx, pad_token="<pad>")
        self.assertEqual(vocab.get_pad_token_id(), 5)

    def test_get_pad_token_id_none(self):
        token_to_idx = {"hello": 0}
        vocab = Vocab.from_dict(token_to_idx)
        self.assertIsNone(vocab.get_pad_token_id())

    def test_get_bos_token_id(self):
        token_to_idx = {"<bos>": 3, "hello": 0}
        vocab = Vocab.from_dict(token_to_idx, bos_token="<bos>")
        self.assertEqual(vocab.get_bos_token_id(), 3)

    def test_get_bos_token_id_none(self):
        token_to_idx = {"hello": 0}
        vocab = Vocab.from_dict(token_to_idx)
        self.assertIsNone(vocab.get_bos_token_id())

    def test_get_eos_token_id(self):
        token_to_idx = {"<eos>": 7, "hello": 0}
        vocab = Vocab.from_dict(token_to_idx, eos_token="<eos>")
        self.assertEqual(vocab.get_eos_token_id(), 7)

    def test_get_eos_token_id_none(self):
        token_to_idx = {"hello": 0}
        vocab = Vocab.from_dict(token_to_idx)
        self.assertIsNone(vocab.get_eos_token_id())


class TestVocabDuplicateSpecialTokens(unittest.TestCase):
    """Tests for duplicate special tokens handling."""

    def test_duplicate_special_tokens_not_added_twice(self):
        # If unk_token and pad_token are the same, only one entry
        token_to_idx = {"<special>": 0, "hello": 1}
        vocab = Vocab.from_dict(token_to_idx, unk_token="<special>", pad_token="<special>")
        self.assertEqual(len(vocab), 2)


class TestVocabEdgeCases(unittest.TestCase):
    """Tests for edge cases."""

    def test_empty_counter(self):
        counter = collections.Counter()
        vocab = Vocab(counter)
        self.assertEqual(len(vocab), 0)

    def test_empty_counter_with_special_tokens(self):
        counter = collections.Counter()
        vocab = Vocab(counter, unk_token="<unk>")
        self.assertEqual(len(vocab), 1)

    def test_special_tokens_in_counter(self):
        # Special tokens in counter should not be added again
        counter = collections.Counter(["<unk>", "hello", "<unk>"])
        vocab = Vocab(counter, unk_token="<unk>")
        # <unk> should appear only once
        self.assertEqual(len(vocab), 2)

    def test_from_json_with_identifiers_to_tokens(self):
        counter = collections.Counter(["hello"])
        vocab = Vocab(counter, unk_token="<unk>", cls_token="<cls>")
        json_str = vocab.to_json()
        restored = Vocab.from_json(json_str)
        self.assertEqual(restored.cls_token, "<cls>")


if __name__ == "__main__":
    unittest.main()
