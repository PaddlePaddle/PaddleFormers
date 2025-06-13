# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import tokenizers
from packaging import version
from tokenizers import (
    AddedToken,
    Regex,
    Tokenizer,
    decoders,
    normalizers,
    pre_tokenizers,
    processors,
)
from tokenizers.models import BPE, Unigram, WordPiece

from ..utils.import_utils import is_protobuf_available, is_sentencepiece_available


def import_protobuf(error_message=""):
    if is_sentencepiece_available():
        from sentencepiece import sentencepiece_model_pb2

        return sentencepiece_model_pb2
    if is_protobuf_available():
        import google.protobuf

        if version.parse(google.protobuf.__version__) < version.parse("4.0.0"):
            from transformers.utils import sentencepiece_model_pb2
        else:
            from transformers.utils import (
                sentencepiece_model_pb2_new as sentencepiece_model_pb2,
            )
        return sentencepiece_model_pb2
    else:
        raise ImportError(
            f"""
{error_message} requires the protobuf library but it was not found in your environment. Checkout the instructions on the
installation page of its repo: https://github.com/protocolbuffers/protobuf/tree/master/python#installation and follow the ones
that match your environment. Please note that you may need to restart your runtime after installation.
"""
        )


# Copied from transformers, adapted for tokenizers >= 0.19.0
def _get_prepend_scheme(add_prefix_space: bool, original_tokenizer) -> str:
    if add_prefix_space:
        prepend_scheme = "always"
        if hasattr(original_tokenizer, "legacy") and not original_tokenizer.legacy:
            prepend_scheme = "first"
    else:
        prepend_scheme = "never"
    return prepend_scheme


def generate_merges(vocab, vocab_scores):
    reverse = vocab_scores is not None
    vocab_scores = dict(vocab_scores) if reverse else vocab

    merges = []
    for merge, piece_score in vocab_scores.items():
        local = []
        for index in range(1, len(merge)):
            piece_l, piece_r = merge[:index], merge[index:]
            if piece_l in vocab and piece_r in vocab:
                local.append((piece_l, piece_r, piece_score))
        local = sorted(local, key=lambda x: (vocab[x[0]], vocab[x[1]]))
        merges.extend(local)

    merges = sorted(merges, key=lambda val: (val[2], len(val[0]), len(val[1])), reverse=reverse)
    merges = [(val[0], val[1]) for val in merges]
    return merges


# Extract the vocab and merge file from sentencepiece file
class SentencePieceExtractor:
    def __init__(self, model: str):
        from sentencepiece import SentencePieceProcessor

        self.sp = SentencePieceProcessor()
        self.sp.Load(model)

    def extract(self, vocab_scores: Optional[Tuple[str, float]] = None) -> Tuple[Dict[str, int], List[Tuple]]:
        sp = self.sp
        vocab = {sp.id_to_piece(index): index for index in range(sp.GetPieceSize())}
        if vocab_scores is not None:
            vocab_scores, reverse = dict(vocab_scores), True
        else:
            vocab_scores, reverse = vocab, False

        # Merges
        merges = []
        for merge, piece_score in vocab_scores.items():
            local = []
            for index in range(1, len(merge)):
                piece_l, piece_r = merge[:index], merge[index:]
                if piece_l in vocab and piece_r in vocab:
                    local.append((piece_l, piece_r, piece_score))
            local = sorted(local, key=lambda x: (vocab[x[0]], vocab[x[1]]))
            merges.extend(local)

        merges = sorted(merges, key=lambda val: val[2], reverse=reverse)
        merges = [(val[0], val[1]) for val in merges]

        return vocab, merges


class GemmaSentencePieceExtractor(SentencePieceExtractor):
    def extract(self, vocab_scores=None) -> Tuple[Dict[str, int], List[Tuple]]:
        """
        By default will return vocab and merges with respect to their order, by sending `vocab_scores` we're going to
        order the merges with respect to the piece scores instead.
        """
        sp = self.sp
        vocab = {sp.id_to_piece(index): index for index in range(sp.GetPieceSize())}

        # there is a missing token in the vocab. We have to do this to support merges
        # "<0x09>" is the bytefallback for `\t`
        vocab["\t"] = vocab.get("<0x09>")

        merges = generate_merges(vocab, vocab_scores)
        return vocab, merges


def check_number_comma(piece: str) -> bool:
    return len(piece) < 2 or piece[-1] != "," or not piece[-2].isdigit()


class Converter:
    def __init__(self, original_tokenizer):
        self.original_tokenizer = original_tokenizer

    def converted(self) -> Tokenizer:
        raise NotImplementedError()


class SpmConverter(Converter):
    def __init__(self, *args):

        super().__init__(*args)

        from . import sentencepiece_model_pb2 as model_pb2

        m = model_pb2.ModelProto()
        if hasattr(self.original_tokenizer, "sentencepiece_model_file"):
            spm_vocab_file = self.original_tokenizer.sentencepiece_model_file
        else:
            spm_vocab_file = self.original_tokenizer.vocab_file
        with open(spm_vocab_file, "rb") as f:
            m.ParseFromString(f.read())
        self.proto = m

        if self.proto.trainer_spec.byte_fallback:
            if not getattr(self, "handle_byte_fallback", None):
                import warnings

                warnings.warn(
                    "The sentencepiece tokenizer that you are converting to a fast tokenizer uses the byte fallback option"
                    " which is not implemented in the fast tokenizers. In practice this means that the fast version of the"
                    " tokenizer can produce unknown tokens whereas the sentencepiece version would have converted these "
                    "unknown tokens into a sequence of byte tokens matching the original piece of text."
                )

    def vocab(self, proto):
        return [(piece.piece, piece.score) for piece in proto.pieces]

    def unk_id(self, proto):
        return proto.trainer_spec.unk_id

    def tokenizer(self, proto):
        model_type = proto.trainer_spec.model_type
        vocab_scores = self.vocab(proto)
        unk_id = self.unk_id(proto)

        if model_type == 1:
            tokenizer = Tokenizer(Unigram(vocab_scores, unk_id))
        elif model_type == 2:
            _, merges = SentencePieceExtractor(self.original_tokenizer.vocab_file).extract()
            bpe_vocab = {word: i for i, (word, score) in enumerate(vocab_scores)}
            tokenizer = Tokenizer(
                BPE(
                    bpe_vocab,
                    merges,
                    unk_token=proto.trainer_spec.unk_piece,
                    fuse_unk=True,
                )
            )
        else:
            raise Exception(
                "You're trying to run a `Unigram` model but you're file was trained with a different algorithm"
            )

        return tokenizer

    def normalizer(self, proto):
        precompiled_charsmap = proto.normalizer_spec.precompiled_charsmap
        _normalizers = [
            normalizers.Strip(left=False, right=True),  # stripping is important
            normalizers.Replace(Regex(" {2,}"), "▁"),
        ]
        if not precompiled_charsmap:
            return normalizers.Sequence(_normalizers)
        else:
            return normalizers.Sequence([normalizers.Precompiled(precompiled_charsmap)] + _normalizers)

    def pre_tokenizer(self, replacement, add_prefix_space):
        prepend_scheme = "always"
        if hasattr(self.original_tokenizer, "legacy") and not self.original_tokenizer.legacy:
            prepend_scheme = "first"
        if version.parse(tokenizers.__version__) >= version.parse("0.19.0"):
            prepend_scheme = _get_prepend_scheme(add_prefix_space, self.original_tokenizer)
            return pre_tokenizers.Metaspace(replacement=replacement, prepend_scheme=prepend_scheme)
        else:
            return pre_tokenizers.Metaspace(
                replacement=replacement, add_prefix_space=add_prefix_space, prepend_scheme=prepend_scheme
            )

    def post_processor(self):
        return None

    def decoder(self, replacement, add_prefix_space):
        if version.parse(tokenizers.__version__) >= version.parse("0.19.0"):
            prepend_scheme = _get_prepend_scheme(add_prefix_space, self.original_tokenizer)
            return decoders.Metaspace(replacement=replacement, prepend_scheme=prepend_scheme)
        else:
            return decoders.Metaspace(replacement=replacement, add_prefix_space=add_prefix_space)

    def converted(self) -> Tokenizer:
        tokenizer = self.tokenizer(self.proto)

        # Tokenizer assemble
        normalizer = self.normalizer(self.proto)
        if normalizer is not None:
            tokenizer.normalizer = normalizer

        replacement = "▁"
        add_prefix_space = True
        pre_tokenizer = self.pre_tokenizer(replacement, add_prefix_space)
        if pre_tokenizer is not None:
            tokenizer.pre_tokenizer = pre_tokenizer

        tokenizer.decoder = self.decoder(replacement, add_prefix_space)
        post_processor = self.post_processor()
        if post_processor:
            tokenizer.post_processor = post_processor

        return tokenizer


def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a significant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    _chr = chr
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [_chr(n) for n in cs]
    return dict(zip(bs, cs))


class TikTokenConverter:
    """
    A general tiktoken converter.
    """

    def __init__(
        self,
        vocab_file=None,
        pattern=r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""",
        add_prefix_space=False,
        additional_special_tokens=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args)
        self.vocab_file = vocab_file
        self.pattern = pattern
        self.add_prefix_space = add_prefix_space
        self.additional_special_tokens = additional_special_tokens

    def extract_vocab_merges_from_model(self, tiktoken_url: str):
        try:
            from tiktoken.load import load_tiktoken_bpe
        except Exception:
            raise ValueError(
                "`tiktoken` is required to read a `tiktoken` file. Install it with " "`pip install tiktoken`."
            )

        bpe_ranks = load_tiktoken_bpe(tiktoken_url)
        byte_encoder = bytes_to_unicode()

        def token_bytes_to_string(b):
            return "".join([byte_encoder[ord(char)] for char in b.decode("latin-1")])

        merges = []
        vocab = {}
        for token, rank in bpe_ranks.items():
            vocab[token_bytes_to_string(token)] = rank
            if len(token) == 1:
                continue
            local = []
            for index in range(1, len(token)):
                piece_l, piece_r = token[:index], token[index:]
                if piece_l in bpe_ranks and piece_r in bpe_ranks and (piece_l + piece_r) in bpe_ranks:
                    local.append((piece_l, piece_r, rank))
            local = sorted(local, key=lambda x: (bpe_ranks[x[0]], bpe_ranks[x[1]]), reverse=False)
            merges.extend(local)
        merges = sorted(merges, key=lambda val: val[2], reverse=False)
        merges = [(token_bytes_to_string(val[0]), token_bytes_to_string(val[1])) for val in merges]
        return vocab, merges

    def tokenizer(self):
        vocab_scores, merges = self.extract_vocab_merges_from_model(self.vocab_file)
        tokenizer = Tokenizer(BPE(vocab_scores, merges, fuse_unk=False))
        if hasattr(tokenizer.model, "ignore_merges"):
            tokenizer.model.ignore_merges = True
        return tokenizer

    def converted(self) -> Tokenizer:
        tokenizer = self.tokenizer()
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(Regex(self.pattern), behavior="isolated", invert=False),
                pre_tokenizers.ByteLevel(add_prefix_space=self.add_prefix_space, use_regex=False),
            ]
        )
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.add_special_tokens(self.additional_special_tokens)

        tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

        return tokenizer


class BertConverter(Converter):
    def converted(self) -> Tokenizer:
        vocab = self.original_tokenizer.vocab
        tokenizer = Tokenizer(
            WordPiece(
                OrderedDict([(vocab._idx_to_token[i], i) for i in range(len(vocab))]),
                unk_token=str(self.original_tokenizer.unk_token),
            )
        )

        tokenize_chinese_chars = False
        strip_accents = False
        do_lower_case = False
        if hasattr(self.original_tokenizer, "basic_tokenizer"):
            tokenize_chinese_chars = self.original_tokenizer.basic_tokenizer.tokenize_chinese_chars
            strip_accents = self.original_tokenizer.basic_tokenizer.strip_accents
            do_lower_case = self.original_tokenizer.basic_tokenizer.do_lower_case

        tokenizer.normalizer = normalizers.BertNormalizer(
            clean_text=True,
            handle_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            lowercase=do_lower_case,
        )
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

        cls = str(self.original_tokenizer.cls_token)
        sep = str(self.original_tokenizer.sep_token)
        cls_token_id = self.original_tokenizer.cls_token_id
        sep_token_id = self.original_tokenizer.sep_token_id

        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{cls}:0 $A:0 {sep}:0",
            pair=f"{cls}:0 $A:0 {sep}:0 $B:1 {sep}:1",
            special_tokens=[
                (cls, cls_token_id),
                (sep, sep_token_id),
            ],
        )
        tokenizer.decoder = decoders.WordPiece(prefix="##")

        return tokenizer


class ErnieConverter(Converter):
    def converted(self) -> Tokenizer:
        vocab = self.original_tokenizer.vocab
        tokenizer = Tokenizer(
            WordPiece(
                OrderedDict([(vocab._idx_to_token[i], i) for i in range(len(vocab))]),
                unk_token=str(self.original_tokenizer.unk_token),
            )
        )
        tokenize_chinese_chars = False
        strip_accents = False
        do_lower_case = False
        if hasattr(self.original_tokenizer, "basic_tokenizer"):
            tokenize_chinese_chars = self.original_tokenizer.basic_tokenizer.tokenize_chinese_chars
            strip_accents = self.original_tokenizer.basic_tokenizer.strip_accents
            do_lower_case = self.original_tokenizer.basic_tokenizer.do_lower_case

        tokenizer.normalizer = normalizers.BertNormalizer(
            clean_text=True,
            handle_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            lowercase=do_lower_case,
        )
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

        cls = str(self.original_tokenizer.cls_token)
        sep = str(self.original_tokenizer.sep_token)
        cls_token_id = self.original_tokenizer.cls_token_id
        sep_token_id = self.original_tokenizer.sep_token_id

        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{cls}:0 $A:0 {sep}:0",
            pair=f"{cls}:0 $A:0 {sep}:0 $B:1 {sep}:1",
            special_tokens=[
                (cls, cls_token_id),
                (sep, sep_token_id),
            ],
        )
        tokenizer.decoder = decoders.WordPiece(prefix="##")

        return tokenizer


class GPTConverter(Converter):
    def converted(self, vocab: Dict[str, int] = None, merges: List[Tuple[str, str]] = None) -> Tokenizer:
        if not vocab:
            vocab = self.original_tokenizer.encoder
        if not merges:
            merges = list(self.original_tokenizer.bpe_ranks)

        tokenizer = Tokenizer(
            BPE(
                vocab=vocab,
                merges=merges,
                dropout=None,
                continuing_subword_prefix="",
                end_of_word_suffix="",
                fuse_unk=False,
            )
        )

        add_prefix_space = getattr(self.original_tokenizer, "add_prefix_space", False)
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=add_prefix_space)
        tokenizer.decoder = decoders.ByteLevel()
        if getattr(self.original_tokenizer, "add_bos_token", False):
            bos = self.original_tokenizer.bos_token
            bos_token_id = self.original_tokenizer.bos_token_id
            tokenizer.post_processor = processors.TemplateProcessing(
                single=f"{bos}:0 $A:0",
                pair=f"{bos}:0 $A:0 $B:1",
                special_tokens=[
                    (bos, bos_token_id),
                ],
            )
        else:
            # XXX trim_offsets=False actually means this post_processor doesn't
            # really do anything.
            tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
        return tokenizer


class GemmaConverter(SpmConverter):
    handle_byte_fallback = True
    SpmExtractor = GemmaSentencePieceExtractor
    # start and end of turn tokens must be marked as special
    special_tokens = {"<start_of_turn>", "<end_of_turn>"}

    """"
    split_by_unicode_script: true
    split_by_number: true
    split_by_whitespace: true
    treat_whitespace_as_suffix: false
    allow_whitespace_only_pieces: true
    split_digits: true
    byte_fallback: true
    """

    def normalizer(self, proto):
        return normalizers.Replace(" ", "▁")

    def vocab(self, proto):
        vocab = [
            (self.original_tokenizer.pad_token, 0.0),
            (self.original_tokenizer.eos_token, 0.0),
            (self.original_tokenizer.bos_token, 0.0),
        ]
        for piece in proto.pieces[3:]:
            if piece.piece == "<0x09>":
                vocab += [("\t", piece.score)]
            else:
                vocab += [(piece.piece, piece.score)]
        # vocab += [(piece.piece, piece.score) for piece in proto.pieces[3:]]
        return vocab

    def pre_tokenizer(self, replacement, add_prefix_space):
        return pre_tokenizers.Split(" ", "merged_with_previous")

    def unk_id(self, proto):
        unk_id = 3
        return unk_id

    def decoder(self, replacement, add_prefix_space):
        return decoders.Sequence(
            [
                decoders.Replace("▁", " "),
                decoders.ByteFallback(),
                decoders.Fuse(),
            ]
        )


class LlamaConverter(SpmConverter):
    handle_byte_fallback = True

    def vocab(self, proto):
        vocab = [
            ("<unk>", 0.0),
            ("<s>", 0.0),
            ("</s>", 0.0),
        ]
        vocab += [(piece.piece, piece.score) for piece in proto.pieces[3:]]
        return vocab

    def unk_id(self, proto):
        return 0

    def decoder(self, replacement, add_prefix_space):
        return decoders.Sequence(
            [
                decoders.Replace("▁", " "),
                decoders.ByteFallback(),
                decoders.Fuse(),
                decoders.Strip(content=" ", left=1),
            ]
        )

    def tokenizer(self, proto):
        model_type = proto.trainer_spec.model_type
        vocab_scores = self.vocab(proto)
        if model_type == 1:

            if version.parse(tokenizers.__version__) < version.parse("0.14.0"):
                tokenizer = Tokenizer(Unigram(vocab_scores, 0))
            else:
                tokenizer = Tokenizer(Unigram(vocab_scores, 0, byte_fallback=True))

        elif model_type == 2:
            _, merges = SentencePieceExtractor(self.original_tokenizer.vocab_file).extract(vocab_scores)
            bpe_vocab = {word: i for i, (word, _score) in enumerate(vocab_scores)}
            tokenizer = Tokenizer(
                BPE(bpe_vocab, merges, unk_token=proto.trainer_spec.unk_piece, fuse_unk=True, byte_fallback=True)
            )
            tokenizer.add_special_tokens(
                [
                    AddedToken("<unk>", normalized=False, special=True),
                    AddedToken("<s>", normalized=False, special=True),
                    AddedToken("</s>", normalized=False, special=True),
                ]
            )
        else:
            raise Exception(
                "You're trying to run a `Unigram` model but you're file was trained with a different algorithm"
            )

        return tokenizer

    def normalizer(self, proto):
        return normalizers.Sequence(
            [
                normalizers.Prepend(prepend="▁"),
                normalizers.Replace(pattern=" ", content="▁"),
            ]
        )

    def pre_tokenizer(self, replacement, add_prefix_space):
        return None


class Qwen2Converter(Converter):
    def converted(self, vocab: Dict[str, int] = None, merges: List[Tuple[str, str]] = None) -> Tokenizer:
        if not vocab:
            vocab = self.original_tokenizer.encoder
        if not merges:
            merges = list(self.original_tokenizer.bpe_ranks.keys())

        tokenizer = Tokenizer(
            BPE(
                vocab=vocab,
                merges=merges,
                dropout=None,
                unk_token=None,
                continuing_subword_prefix="",
                end_of_word_suffix="",
                fuse_unk=False,
                byte_fallback=False,
            )
        )

        tokenizer.normalizer = normalizers.NFC()

        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(
                    Regex(
                        r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
                    ),
                    behavior="isolated",
                    invert=False,
                ),
                pre_tokenizers.ByteLevel(
                    add_prefix_space=getattr(self.original_tokenizer, "add_prefix_space", False),
                    use_regex=False,
                ),
            ]
        )

        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

        return tokenizer


SLOW_TO_FAST_CONVERTERS = {
    "BertTokenizer": BertConverter,
    "ErnieTokenizer": ErnieConverter,
    "GemmaTokenizer": GemmaConverter,
    "GPTTokenizer": GPTConverter,
    "LlamaTokenizer": LlamaConverter,
    "Qwen2Tokenizer": Qwen2Converter,
}


def convert_slow_tokenizer(transformer_tokenizer, from_tiktoken=False) -> Tokenizer:
    """
    Utilities to convert a slow tokenizer instance in a fast tokenizer instance.

    Args:
        transformer_tokenizer ([`~tokenizer_utils_base.PretrainedTokenizer`]):
            Instance of a slow tokenizer to convert in the backend tokenizer for
            [`~tokenizer_utils_base.PretrainedTokenizerFast`].

    Return:
        A instance of [`~tokenizers.Tokenizer`] to be used as the backend tokenizer of a
        [`~tokenizer_utils_base.PretrainedTokenizerFast`]
    """

    tokenizer_class_name = transformer_tokenizer.__class__.__name__
    if tokenizer_class_name in SLOW_TO_FAST_CONVERTERS and not from_tiktoken:
        converter_class = SLOW_TO_FAST_CONVERTERS[tokenizer_class_name]
        return converter_class(transformer_tokenizer).converted()
    else:
        try:
            return TikTokenConverter(
                vocab_file=transformer_tokenizer.vocab_file,
                additional_special_tokens=transformer_tokenizer.additional_special_tokens,
            ).converted()
        except Exception:
            raise ValueError(
                f"Converting from Tiktoken failed, if a converter for SentencePiece is available, provide a model path "
                f"with a SentencePiece tokenizer.model file."
                f"Currently available slow->fast converters: {list(SLOW_TO_FAST_CONVERTERS.keys())}"
            )
