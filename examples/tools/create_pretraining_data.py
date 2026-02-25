# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved
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
import argparse
import io
import json
import multiprocessing
import os
import re
import sys
import time

import numpy as np
from tqdm import tqdm

from paddleformers.data import indexed_dataset
from paddleformers.transformers import AutoTokenizer
from paddleformers.utils.log import logger

try:
    import nltk

    nltk_available = True
except ImportError:
    nltk_available = False

from datetime import datetime


def print_datetime(string):
    time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("[" + string + "] datetime: {} ".format(time_str))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True, help="What model to use.")
    group = parser.add_argument_group(title="data input/output")
    group.add_argument("--input_path", type=str, required=True, help="Path to input JSON files.")
    group.add_argument("--output_prefix", type=str, required=True, help="Output prefix to store output file.")
    group.add_argument(
        "--data_format",
        type=str,
        default="text",
        choices=["JSON"],
        help="Only support json format for now. One document per line.",
    )
    group.add_argument(
        "--json_key",
        type=str,
        default="text",
        help="For JSON format. Space separate listed of keys to extract from json",
    )
    group.add_argument("--split_sentences", action="store_true", help="Split documents into sentences.")

    group.add_argument("--data_impl", type=str, default="mmap", choices=["lazy", "mmap"])

    group = parser.add_argument_group(title="chinese words")
    group.add_argument(
        "--chinese", action="store_true", help="Is corpus need words segmentation step for chinese words."
    )
    group.add_argument(
        "--cn_whole_word_segment",
        action="store_true",
        help="Is corpus need words segmentation step for chinese words WWM.",
    )
    group.add_argument(
        "--cn_seg_func",
        type=str,
        default="jieba",
        choices=["lac", "seg", "jieba"],
        help="Words segment function for chinese words.",
    )
    group.add_argument("--cn_splited", action="store_true", help="Is chinese corpus is split in to words.")
    group.add_argument("--cn_split_dimer", type=str, default=" ", help="Split dimer between chinese words.")

    group = parser.add_argument_group(title="common config")
    group.add_argument("--append_eos", action="store_true", help="Append an <eos> token to the end of a document.")
    group.add_argument("--log_interval", type=int, default=100, help="Interval between progress updates")
    group.add_argument("--workers", type=int, default=1, help="Number of worker processes to launch")
    group.add_argument("--max_doc_num", type=int, default=sys.maxsize, help="Stop when reach max_doc_num.")
    group.add_argument(
        "--max_repeated_len", type=int, default=100, help="The maximum length of the repeated characters to keep"
    )

    args = parser.parse_args()
    return args


def lexical_analysis_fn():
    from LAC import LAC

    lac = LAC(mode="lac")

    def process(line):
        words, _ = lac.run(line)
        return words

    return process


def chinese_segmentation_fn():
    from LAC import LAC

    lac_cws = LAC(mode="seg")

    def process(line):
        words = lac_cws.run(line)
        return words

    return process


def jieba_segmentation_fn():
    import jieba

    def process(line):
        words = jieba.cut(line)
        return list(words)

    return process


def get_whole_word_mask_tokens(tokens, words, max_word_length=6):
    """
    Do whole word mask on Chinese word.
    First, we do Chinese word segmentation on the sequence of tokens, which are from the WordPiece tokenization.
    Then, we add the '##' mark on chinese characters which are in the middle of Chinese words.
    And if the tokens are not chinese characters, we just exploit the results of WordPiece tokenization as words.
    Such as,
         - text line : 通过利用mercer核，将样本从输入空间映射到高维特征空间，使原来没有显现的特征突现出来，取得了很好的图像分割效果。
         - the input tokens (after WordPiece):
            ['通', '过', '利', '用', 'me', '##rc', '##er', '核', '，', '将', '样', '本', '从', '输', '入', '空', '间', '映',
            '射', '到', '高', '维', '特', '征', '空', '间', '，', '使', '原', '来', '没', '有', '显', '现', '的', '特', '征',
            '突', '现', '出', '来', '，', '取', '得', '了', '很', '好', '的', '图', '像', '分', '割', '效', '果', '。']
        - the Chinese words (after Chinese word segmentation like jieba)
            ['通过', '利用', 'mercer', '核', '，', '将', '样本', '从', '输入', '空间', '映射', '到', '高维', '特征',
            '空间', '，', '使', '原来', '没有', '显现', '的', '特征', '突现', '出来', '，', '取得', '了', '很', '好',
            '的', '图像', '分割', '效果', '。']
        - the output whole word mask tokens:
            ['通', '##过', '利', '##用', 'me', '##rc', '##er', '核', '，', '将', '样', '##本', '从', '输', '##入',
            '空', '##间', '映', '##射', '到', '高', '##维', '特', '##征', '空', '##间', '，', '使', '原', '##来',
            '没', '##有', '显', '##现', '的', '特', '##征', '突', '##现', '出', '##来', '，', '取', '##得', '了',
            '很', '好', '的', '图', '##像', '分', '##割', '效', '##果', '。']

    Args:
        tokens(list(str)): The sequence of tokens, which are from the WordPiece tokenization.
        words(list(str)): The sequence of Chinese words.
        max_word_length(int, optional):
            The maximum chinese character in Chinese words. It avoids too long Chinese word to be masked.
            Defaults as 4.

    Returns:
         new_tokens(list(str)): The new token will be done with whole word masking strategy.

    """

    new_tokens = []
    # opt for long document
    words_set = set(words)
    i = 0
    while i < len(tokens):
        # non-chinese character, then do word piece
        if len(re.findall("[\u4E00-\u9FA5]", tokens[i])) == 0:
            new_tokens.append(tokens[i])
            i += 1
            continue

        # add "##" mark on the middle tokens of Chinese words
        # such as ["通过", "利用"] -> ["通", "##过"， "利", "##用"]
        has_add = False
        for length in range(max_word_length, 0, -1):
            if i + length > len(tokens):
                continue
            if "".join(tokens[i : i + length]) in words_set:
                new_tokens.append(tokens[i])
                for l in range(1, length):
                    new_tokens.append("##" + tokens[i + l])
                i += length
                has_add = True
                break

        if not has_add:
            new_tokens.append(tokens[i])
            i += 1
    return new_tokens


class IdentitySplitter(object):
    def tokenize(self, *text):
        return text


class NewlineSplitter:
    def tokenize(self, text):
        return text.split("\n")


class Converter(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        Converter.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path)
        if self.args.cn_whole_word_segment:
            # Extend chinese char vocab for ErnieTokinzer
            Converter.tokenizer.extend_chinese_char()

        # Split document to sentence.
        if self.args.split_sentences:
            if self.args.chinese:
                Converter.splitter = NewlineSplitter()
            else:
                if not nltk_available:
                    print("NLTK is not available to split sentences.")
                    exit()
                splitter = nltk.load("tokenizers/punkt/english.pickle")
                Converter.splitter = splitter
        else:
            Converter.splitter = IdentitySplitter()

        # Split sentence whole words mask for chinese
        if self.args.cn_whole_word_segment:
            if self.args.cn_splited:
                Converter.segment_func = lambda text: text.split(self.args.cn_split_dimer)
            else:
                CHINESE_SEG_FUNC = {
                    "lac": lexical_analysis_fn(),
                    "seg": chinese_segmentation_fn(),
                    "jieba": jieba_segmentation_fn(),
                }
                Converter.segment_func = CHINESE_SEG_FUNC[self.args.cn_seg_func]
            Converter.whole_word_mask = get_whole_word_mask_tokens
        else:
            Converter.segment_func = lambda x: x
            Converter.whole_word_mask = lambda x, y: x

        def process(text):
            words = Converter.segment_func(text)
            # if there are two empty word, the should a split dimer in the pos
            if self.args.cn_splited:
                pre_dimer = False
                for index, w in enumerate(words):
                    if pre_dimer and len(w) == 0:
                        words[index] = self.args.cn_split_dimer
                        pre_dimer = False
                    elif len(w) == 0:
                        pre_dimer = True
                    else:
                        pre_dimer = False

            tokens = Converter.tokenizer.tokenize("".join(words))
            tokens = Converter.whole_word_mask(tokens, words)
            tokens = Converter.tokenizer.convert_tokens_to_ids(tokens)
            return tokens

        Converter.process = process

    def remove_repeated_chars(text, max_repeated_len=100):
        """
        Removes repeated characters from the given text, where the length of
        the repeated characters is greater than or equal to the specified length.

        Args:
            text (str): The input text from which to remove repeated characters.
            length (int, optional): The minimum length of the repeated characters. Defaults to 15.

        Returns:
            str: The modified text with the repeated characters removed.
        """
        pattern = r"(.)\1{" + str(max_repeated_len) + ",}"
        return re.sub(pattern, r"\1", text)

    def encode(self, json_line):
        text = json.loads(json_line)[self.args.json_key]
        text = Converter.remove_repeated_chars(text, self.args.max_repeated_len)
        doc_ids = []
        for sentence in Converter.splitter.tokenize(text):
            sentence_ids = Converter.process(sentence.strip())
            if len(sentence_ids) > 0:
                doc_ids.append(sentence_ids)

        if len(doc_ids) > 0 and self.args.append_eos:
            if Converter.tokenizer.eos_token_id is None:
                logger.warning(
                    "{}: eos_token_id is not set, ".format(self.args.tokenizer_name)
                    + "please set other tokenizer "
                    + "or config eos_token_id or unset append_eos."
                )
            else:
                doc_ids[-1].append(Converter.tokenizer.eos_token_id)

        return doc_ids, len(text.encode("utf-8"))


def main():
    print_datetime("start")
    args = get_args()
    file_paths = []
    if os.path.isfile(args.input_path):
        file_paths.append(args.input_path)
    else:
        for root, _, fs in os.walk(args.input_path):
            for f in fs:
                file_paths.append(os.path.join(root, f))

    convert = Converter(args)

    # Try tokenizer is available
    sample_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if sample_tokenizer.vocab_size < 2**16 - 1:
        save_dtype = np.uint16
    else:
        save_dtype = np.int32

    pool = multiprocessing.Pool(args.workers, initializer=convert.initializer)

    output_ids_files = args.output_prefix + ".bin"
    output_idx_files = args.output_prefix + ".idx"
    builder = indexed_dataset.make_builder(output_ids_files, args.data_impl, save_dtype)

    file_paths.sort()

    step = 0
    total_bytes_processed = 0
    startup_start = time.time()
    for file_path in tqdm(file_paths):
        if file_path.endswith(".zst"):
            import zstandard

            cctx = zstandard.ZstdDecompressor()
            fh = open(file_path, "rb")
            text = io.BufferedReader(cctx.stream_reader(fh))
        elif file_path.endswith(".jsonl"):
            text = open(file_path, "r", encoding="utf-8")
        else:
            print("Unexpected data format, skipped %s" % file_path)
            continue

        encoded_docs = pool.imap(convert.encode, text, 256)
        print("Processing %s" % file_path)
        for i, (doc, bytes_processed) in enumerate(encoded_docs, start=1):
            step += 1
            total_bytes_processed += bytes_processed
            if len(doc) == 0:
                continue

            for sentence in doc:
                sentence_len = len(sentence)
                if sentence_len == 0:
                    continue
                builder.add_item(sentence)

            builder.end_document()

            if step % args.log_interval == 0:
                current = time.time()
                elapsed = current - startup_start
                mbs = total_bytes_processed / elapsed / 1024 / 1024
                print(f"Processed {step} documents", f"({step/elapsed:.2f} docs/s, {mbs:.4f} MB/s).", file=sys.stderr)
            if step >= args.max_doc_num:
                break

        if step >= args.max_doc_num:
            break

    pool.close()
    print("Saving tokens to files...")
    builder.finalize(output_idx_files)
    print_datetime("end")


if __name__ == "__main__":
    main()
