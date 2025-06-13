# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
"""Embedding dataset."""

import random
from dataclasses import dataclass
from typing import List

from paddle.io import Dataset, IterableDataset

from ..utils.log import logger


@dataclass
class Example:
    """Dataset example."""

    query: str
    pos_passage: List[str]
    neg_passage: List[str] = None


@dataclass
class Sequence:
    """Sequence."""

    token_ids: List[int]
    position_ids: List[int]


@dataclass
class Pair:
    """Pair."""

    query: Sequence
    passages: List[Sequence]


class EmbeddingDatasetMixin:
    """EmbeddingDatasetMixin."""

    def convert_example(tokenizer, example):
        """Convert raw json format example to Example."""

        assert all(
            (key in example for key in ["query", "pos_passage", "neg_passage"])
        ), "query, pos_passage, neg_passage are needed"

        if not isinstance(example["query"], str):
            raise ValueError("query must be a string.")
        if isinstance(example["pos_passage"], str):
            example["pos_passage"] = [example["pos_passage"]]
        if isinstance(example["neg_passage"], str):
            example["neg_passage"] = [example["neg_passage"]]

        if len(example["neg_passage"]) > 0:
            for item in [example["query"]] + example["pos_passage"] + example["neg_passage"]:
                if not isinstance(item, str):
                    raise ValueError("The item in pos_passage / neg_passage must be a string.")
                if len(item.strip()) == 0:
                    raise ValueError("Example with empty string in query / pos_passage / neg_passage field.")

        query = example["query"]
        pos_passage = example["pos_passage"]
        neg_passage = example["neg_passage"]
        return Example(query=query, pos_passage=pos_passage, neg_passage=neg_passage)

    def tokenize_template(cls, tokenizer, template: str):
        """Tokenize a given template using the provided tokenizer."""
        assert template.count("{text}") == 1, "Template must contain exactly one {text} placeholder"

        template_prefix, template_suffix = template.split("{text}")

        prefix_tokens = tokenizer(template_prefix, add_special_tokens=False).input_ids
        suffix_tokens = tokenizer(template_suffix, add_special_tokens=False).input_ids
        return prefix_tokens, suffix_tokens

    def _process_truncation(self, tokens, text_type):
        """
        Process tokens by converting them into a complete token sequence with prefix and suffix,
        and generate corresponding position ids.
        """
        if text_type not in ["query", "passage"]:
            raise ValueError("text_type must be either 'query' or 'passage'")

        prefix_key = f"{text_type}_template_prefix"
        suffix_key = f"{text_type}_template_suffix"
        max_len_key = f"max_{text_type}_len"

        # If the template does not contain a suffix token, add the EOS token to the end
        if getattr(self, suffix_key) == []:
            setattr(self, suffix_key, [self.tokenizer.eos_token_id])
        if getattr(self, prefix_key) == []:
            setattr(self, prefix_key, [self.tokenizer.bos_token_id])

        # Calculate the available length
        max_len = getattr(self, max_len_key)
        prefix_tokens = getattr(self, prefix_key)
        suffix_tokens = getattr(self, suffix_key)
        available_len = int(max_len - len(prefix_tokens) - len(suffix_tokens))

        # Convert tokens to ids and truncate
        token_ids_converted = self.tokenizer.convert_tokens_to_ids(tokens)
        truncated_token_ids = token_ids_converted[:available_len]

        # Combine prefix, truncated tokens, and suffix
        token_ids = prefix_tokens + truncated_token_ids + suffix_tokens
        pos_ids = list(range(len(token_ids)))
        return token_ids, pos_ids

    def _postprocess_sequence(self, example: Example, rng):
        """Post process sequence: tokenization & truncation."""
        query = example.query
        pos_passage = rng.choice(example.pos_passage)
        neg_passage = example.neg_passage
        if len(neg_passage) > 0:
            if len(neg_passage) < self.group_size - 1:
                # Calculate how many full sets are needed to ensure each element appears at least once
                full_sets_needed = (self.group_size - 1) // len(neg_passage)
                remainder = (self.group_size - 1) % len(neg_passage)

                # Initialize the list and add complete sets
                selected_neg_passage = neg_passage * full_sets_needed

                # Ensure the remainder part is filled; randomly select from neg_passage
                selected_neg_passage += rng.sample(neg_passage, remainder)

                # Shuffle the result to ensure randomness
                rng.shuffle(selected_neg_passage)
            else:
                selected_neg_passage = rng.sample(neg_passage, self.group_size - 1)
        else:
            selected_neg_passage = []
        # Process query tokens
        query_tokens = self.tokenizer.tokenize(query)
        query_token_ids, query_pos_ids = self._process_truncation(query_tokens, "query")

        query = Sequence(
            token_ids=query_token_ids,
            position_ids=query_pos_ids,
        )

        # Process passage tokens
        passages = []
        for passage in [pos_passage] + selected_neg_passage:
            passage_tokens = self.tokenizer.tokenize(passage)
            passage_token_ids, passage_pos_ids = self._process_truncation(passage_tokens, "passage")
            passages.append(
                Sequence(
                    token_ids=passage_token_ids,
                    position_ids=passage_pos_ids,
                )
            )
        return Pair(query=query, passages=passages)


class EmbeddingDataset(EmbeddingDatasetMixin, Dataset):
    def __init__(
        self,
        dataset,
        tokenizer,
        max_query_len: int = 64,
        max_passage_len: int = 256,
        group_size: int = 2,
        query_template: str = "{text}",
        passage_template: str = "{text}",
    ):
        super().__init__()
        self.example_dataset = dataset
        self.tokenizer = tokenizer
        self.max_query_len = max_query_len
        self.max_passage_len = max_passage_len
        self.group_size = group_size
        self.query_template = query_template
        self.passage_template = passage_template
        self.query_template_prefix, self.query_template_suffix = self.tokenize_template(
            self.tokenizer, self.query_template
        )
        self.passage_template_prefix, self.passage_template_suffix = self.tokenize_template(
            self.tokenizer, self.passage_template
        )

        for index, data in enumerate(self.example_dataset):
            self.example_dataset[index] = self.convert_example(data)

    def __getitem__(self, index):
        return self._postprocess_sequence(self.example_dataset[index])

    def __len__(self):
        raise len(self.example_dataset)


class EmbeddingIterableDataset(EmbeddingDatasetMixin, IterableDataset):
    """Create sequences from Example Dataset.

    This is a stateful dataset.
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        max_query_len: int = 64,
        max_passage_len: int = 256,
        group_size: int = 2,
        query_template: str = "{text}",
        passage_template: str = "{text}",
    ):
        super().__init__()
        self.example_dataset = dataset
        self.tokenizer = tokenizer
        self.max_query_len = max_query_len
        self.max_passage_len = max_passage_len
        self.group_size = group_size
        self.query_template = query_template
        self.passage_template = passage_template
        self.query_template_prefix, self.query_template_suffix = self.tokenize_template(
            self.tokenizer, self.query_template
        )
        self.passage_template_prefix, self.passage_template_suffix = self.tokenize_template(
            self.tokenizer, self.passage_template
        )

        self.epoch_index = 0

    def __iter__(self):
        while True:
            logger.info(f"Start to load dataset on epoch={self.epoch_index}")
            yield from self.iter_one_epoch()

    def iter_one_epoch(self):
        """Iterates through one epoch of the dataset."""

        num_sequences = 0
        rng = random.Random()
        for _, example in enumerate(self.example_dataset):
            example = self.convert_example(example)
            rng.seed(num_sequences)
            sequence = self._postprocess_sequence(example, rng)
            if sequence is None:
                continue
            num_sequences += 1
            yield [sequence]

        self.epoch_index += 1
