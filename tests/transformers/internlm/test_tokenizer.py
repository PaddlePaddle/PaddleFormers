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
import itertools
from pathlib import Path

import pytest
import sentencepiece as spm

from paddleformers.transformers import (
    AutoTokenizer,
    InternLM25Tokenizer,
    InternLMTokenizer,
)


@pytest.fixture(scope="module")
def vocab_file(tmp_path_factory):
    directory = tmp_path_factory.mktemp("internlm_sentencepiece")
    corpus = directory / "corpus.txt"
    corpus.write_text("hello world\nhello tokenizer\n你好世界\nleading space and punctuation!\n", encoding="utf-8")
    prefix = str(directory / "tokenizer")
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=prefix,
        vocab_size=96,
        model_type="bpe",
        hard_vocab_limit=False,
        minloglevel=2,
        num_threads=1,
        shuffle_input_sentence=False,
        unk_id=0,
        bos_id=1,
        eos_id=2,
    )
    return prefix + ".model"


@pytest.mark.parametrize("tokenizer_class", [InternLMTokenizer, InternLM25Tokenizer])
@pytest.mark.parametrize("add_bos,add_eos", list(itertools.product([False, True], repeat=2)))
@pytest.mark.parametrize("pair", [False, True])
def test_special_tokens_mask(vocab_file, tokenizer_class, add_bos, add_eos, pair):
    tokenizer = tokenizer_class(vocab_file, add_bos_token=add_bos, add_eos_token=add_eos)
    first = tokenizer.encode("hello world", add_special_tokens=False)
    second = tokenizer.encode("你好世界", add_special_tokens=False) if pair else None
    content = first + (second or [])
    expected_ids = ([tokenizer.bos_token_id] if add_bos else []) + content
    if add_eos:
        expected_ids += [tokenizer.eos_token_id]
    ids = tokenizer.build_inputs_with_special_tokens(first, second)
    mask = tokenizer.get_special_tokens_mask(first, second)
    assert ids == expected_ids
    assert mask == ([1] if add_bos else []) + [0] * len(content) + ([1] if add_eos else [])
    assert len(ids) == len(mask)
    assert tokenizer.create_token_type_ids_from_sequences(first, second) == [0] * len(ids)
    assert tokenizer.get_special_tokens_mask(ids, already_has_special_tokens=True) == mask
    with pytest.raises(ValueError):
        tokenizer.get_special_tokens_mask(ids, first, already_has_special_tokens=True)


@pytest.mark.parametrize("text", ["hello world", " hello world", "你好世界", "leading space and punctuation!", ""])
@pytest.mark.parametrize("decode_with_prefix_space", [False, True])
def test_encode_decode(vocab_file, text, decode_with_prefix_space):
    tokenizer = InternLMTokenizer(vocab_file, decode_with_prefix_space=decode_with_prefix_space)
    ids = tokenizer.encode(text, add_special_tokens=False)
    # SentencePiece normalizes an initial space; decoding must preserve the first actual character.
    expected = tokenizer.sp_model.decode(tokenizer.sp_model.encode(text))
    assert tokenizer.decode(ids, skip_special_tokens=True) == expected
    legacy = InternLM25Tokenizer(vocab_file, decode_with_prefix_space=decode_with_prefix_space)
    assert legacy.encode(text, add_special_tokens=False) == ids
    assert legacy.decode(ids, skip_special_tokens=True) == expected


@pytest.mark.parametrize("add_bos,add_eos", list(itertools.product([False, True], repeat=2)))
def test_save_pretrained_roundtrip(vocab_file, tmp_path, add_bos, add_eos):
    tokenizer = InternLMTokenizer(vocab_file, add_bos_token=add_bos, add_eos_token=add_eos)
    tokenizer.add_tokens(["<extra_word>"])
    tokenizer.add_special_tokens({"additional_special_tokens": ["<extra_special>"]})
    tokenizer.save_pretrained(tmp_path)
    restored = InternLMTokenizer.from_pretrained(tmp_path)
    automatic = AutoTokenizer.from_pretrained(tmp_path)
    assert isinstance(automatic, InternLMTokenizer)
    for other in [restored, automatic]:
        assert other.get_vocab() == tokenizer.get_vocab()
        assert other.add_bos_token == add_bos
        assert other.add_eos_token == add_eos
        for text in ["hello world", "你好世界", "<extra_word> hello <extra_special>"]:
            assert other.encode(text) == tokenizer.encode(text)
            assert other.decode(other.encode(text)) == tokenizer.decode(tokenizer.encode(text))


@pytest.mark.parametrize("tokenizer_class", [InternLMTokenizer, InternLM25Tokenizer])
def test_save_vocabulary(vocab_file, tmp_path, tokenizer_class):
    tokenizer = tokenizer_class(vocab_file)
    (saved,) = tokenizer.save_vocabulary(str(tmp_path), filename_prefix="copy")
    assert Path(saved).name == "copy-tokenizer.model"
    assert Path(saved).read_bytes() == Path(vocab_file).read_bytes()
    # Same-path saves and serialization after the original file disappears must both work.
    restored = tokenizer_class(saved)
    assert restored.save_vocabulary(str(tmp_path), filename_prefix="copy") == (saved,)
    Path(saved).unlink()
    restored.save_vocabulary(str(tmp_path), filename_prefix="copy")
    assert Path(saved).read_bytes() == Path(vocab_file).read_bytes()


@pytest.mark.parametrize("tokenizer_class", [InternLMTokenizer, InternLM25Tokenizer])
@pytest.mark.parametrize("target", ["missing", "file"])
def test_save_vocabulary_invalid_directory(vocab_file, tmp_path, tokenizer_class, target):
    destination = tmp_path / target
    if target == "file":
        destination.write_text("not a directory")
    with pytest.raises(ValueError, match="should be a directory"):
        tokenizer_class(vocab_file).save_vocabulary(str(destination))


@pytest.mark.parametrize("add_bos,add_eos", list(itertools.product([False, True], repeat=2)))
def test_legacy_tokenizer_save_roundtrip(vocab_file, tmp_path, add_bos, add_eos):
    tokenizer = InternLM25Tokenizer(vocab_file, add_bos_token=add_bos, add_eos_token=add_eos)
    tokenizer.save_pretrained(tmp_path)
    restored = InternLM25Tokenizer.from_pretrained(str(tmp_path))
    assert restored.get_vocab() == tokenizer.get_vocab()
    assert restored.add_bos_token == add_bos
    assert restored.add_eos_token == add_eos
    assert restored.encode("hello world") == tokenizer.encode("hello world")
