# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import json
import os
import unittest

import numpy as np

from paddleformers.datasets import (
    ZeroPaddingIterableDataset,
    ZeroPaddingMapDataset,
    load_dataset,
)
from paddleformers.transformers import AutoTokenizer
from tests.testing_utils import get_tests_dir


# used to create a IterDataset that can be iterated over many times
def read_local_dataset(path):
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            yield json.loads(line.strip())


class ZeroPaddingTestCommon:
    tokenizer = AutoTokenizer.from_pretrained("test_paddleformers/micro-random-llama")
    expected_output = {
        "input_ids": [1, 29871, 30429, 1, 29871, 30429, 2, 1, 29871, 31427, 1, 29871, 31427, 2],
        "labels": [-100, -100, -100, 1, 29871, 30429, 2, -100, -100, -100, 1, 29871, 31427, 2],
        "position_ids": np.array([0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6]),
        "attention_mask": np.array(
            [
                [
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                ]
            ]
        ),
        "position_ids_2d": [[0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6]],
    }

    def preprocess_fn(
        self,
        example,
        max_src_length=3,
        max_tgt_length=3,
        return_position_ids=True,
        position_ids_2d=False,
        return_attention_mask=True,
    ):
        inputs = example["sentence"][:2]
        model_inputs = self.tokenizer(inputs, max_length=max_src_length, truncation=True, return_attention_mask=False)
        labels_input_ids = model_inputs["input_ids"] + [self.tokenizer.eos_token_id]
        model_inputs["labels"] = [-100] * len(model_inputs["input_ids"]) + labels_input_ids
        model_inputs["input_ids"] = model_inputs["input_ids"] + labels_input_ids
        seq_length = len(model_inputs["input_ids"])
        if return_position_ids:
            if position_ids_2d:
                position_ids = np.arange(seq_length, dtype=np.int64)
                # fake block_position_ids with wrong values but correct shape
                block_position_ids = np.arange(seq_length, dtype=np.int64)
                model_inputs["position_ids"] = np.stack([position_ids, block_position_ids], axis=0)
            else:
                model_inputs["position_ids"] = list(range(seq_length))
        if return_attention_mask:
            model_inputs["attention_mask"] = np.tril(np.ones([seq_length, seq_length]))
        return model_inputs


class TestZeroPaddingMapDataset(ZeroPaddingTestCommon, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture_path = get_tests_dir(os.path.join("fixtures", "dummy"))
        cls.train_ds = load_dataset("json", data_files=os.path.join(fixture_path, "tnews", "train.json"), lazy=False)[
            0
        ]
        copy_dataset_1 = copy.deepcopy(cls.train_ds)
        copy_dataset_2 = copy.deepcopy(cls.train_ds)
        cls.dataset = cls.train_ds.map(lambda example: cls.preprocess_fn(cls, example))
        cls.dataset_position_2d = copy_dataset_1.map(
            lambda example: cls.preprocess_fn(cls, example, position_ids_2d=True)
        )
        cls.dataset_input_labels_only = copy_dataset_2.map(
            lambda example: cls.preprocess_fn(cls, example, return_position_ids=False, return_attention_mask=False)
        )

    def test_long_max_length(self):
        inData = ZeroPaddingMapDataset(self.dataset, self.tokenizer, max_length=128)
        self.assertEqual(set(inData[0].keys()), {"input_ids", "labels", "position_ids", "attention_mask"})
        self.assertEqual(len(inData), 1)
        self.assertEqual(type(inData[0]["input_ids"]), list)
        self.assertEqual(np.array(inData[0]["input_ids"]).shape, (70,))

        inData_input_labels_only = ZeroPaddingMapDataset(
            self.dataset_input_labels_only, self.tokenizer, max_length=128
        )
        self.assertEqual(set(inData_input_labels_only[0].keys()), {"input_ids", "labels", "attention_mask"})
        self.assertEqual(len(inData_input_labels_only), 1)
        self.assertEqual(type(inData_input_labels_only[0]["input_ids"]), list)
        self.assertEqual(np.array(inData_input_labels_only[0]["input_ids"]).shape, (70,))

    def test_short_max_length(self):
        inData = ZeroPaddingMapDataset(self.dataset, self.tokenizer, max_length=16)
        self.assertEqual(inData[0]["input_ids"], self.expected_output["input_ids"])
        self.assertEqual(inData[0]["labels"], self.expected_output["labels"])
        self.assertTrue((inData[0]["position_ids"] == self.expected_output["position_ids"]).all())
        self.assertTrue((inData[0]["attention_mask"] == self.expected_output["attention_mask"]).all())

        inData_input_labels_only = ZeroPaddingMapDataset(self.dataset_input_labels_only, self.tokenizer, max_length=16)
        self.assertEqual(inData_input_labels_only[0]["input_ids"], self.expected_output["input_ids"])
        self.assertEqual(inData_input_labels_only[0]["labels"], self.expected_output["labels"])
        self.assertTrue(
            (inData_input_labels_only[0]["attention_mask"] == self.expected_output["attention_mask"]).all()
        )

    def test_2d_position_id(self):
        inData_2d = ZeroPaddingMapDataset(self.dataset_position_2d, self.tokenizer, max_length=16)
        self.assertTrue(inData_2d[0]["position_ids"] == self.expected_output["position_ids_2d"])

    def test_missing_data(self):
        orginal_input_ids = [item["input_ids"] for item in self.dataset]
        orginal_input_ids = [sum(orginal_input_ids, [])]
        inData = ZeroPaddingMapDataset(self.dataset, self.tokenizer, max_length=16)
        tgt_input_ids = [item["input_ids"] for item in inData]
        tgt_input_ids = [sum(tgt_input_ids, [])]
        self.assertEqual(orginal_input_ids, tgt_input_ids)


class TestZeroPaddingIterableDataset(ZeroPaddingTestCommon, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture_path = get_tests_dir(os.path.join("fixtures", "dummy"))
        cls.train_ds = load_dataset("json", data_files=os.path.join(fixture_path, "tnews", "train.json"), lazy=False)[
            0
        ]
        copy_dataset_1 = copy.deepcopy(cls.train_ds)
        copy_dataset_2 = copy.deepcopy(cls.train_ds)
        cls.dataset = cls.train_ds.map(lambda example: cls.preprocess_fn(cls, example))
        cls.dataset_position_2d = copy_dataset_1.map(
            lambda example: cls.preprocess_fn(cls, example, position_ids_2d=True)
        )
        cls.dataset_input_labels_only = copy_dataset_2.map(
            lambda example: cls.preprocess_fn(cls, example, return_position_ids=False, return_attention_mask=False)
        )

    def test_long_max_length(self):
        inData = ZeroPaddingIterableDataset(self.dataset, self.tokenizer, max_length=128)
        example = next(iter(inData))
        self.assertEqual(set(example.keys()), {"input_ids", "labels", "position_ids", "attention_mask"})
        self.assertEqual(type(example["input_ids"]), list)
        self.assertEqual(np.array(example["input_ids"]).shape, (70,))

        inData_input_labels_only = ZeroPaddingIterableDataset(
            self.dataset_input_labels_only, self.tokenizer, max_length=128
        )
        example = next(iter(inData_input_labels_only))
        self.assertEqual(set(example.keys()), {"input_ids", "labels", "attention_mask"})
        self.assertEqual(type(example["input_ids"]), list)
        self.assertEqual(np.array(example["input_ids"]).shape, (70,))

    def test_short_max_length(self):
        inData = ZeroPaddingIterableDataset(self.dataset, self.tokenizer, max_length=16)
        example = next(iter(inData))
        self.assertEqual(example["input_ids"], self.expected_output["input_ids"])
        self.assertEqual(example["labels"], self.expected_output["labels"])
        self.assertTrue((example["position_ids"] == self.expected_output["position_ids"]).all())
        self.assertTrue((example["attention_mask"] == self.expected_output["attention_mask"]).all())

        inData_input_labels_only = ZeroPaddingIterableDataset(
            self.dataset_input_labels_only, self.tokenizer, max_length=16
        )
        example = next(iter(inData_input_labels_only))
        self.assertEqual(example["input_ids"], self.expected_output["input_ids"])
        self.assertEqual(example["labels"], self.expected_output["labels"])
        self.assertTrue((example["attention_mask"] == self.expected_output["attention_mask"]).all())

    def test_2d_position_id(self):
        inData_2d = ZeroPaddingIterableDataset(self.dataset_position_2d, self.tokenizer, max_length=16)
        example = next(iter(inData_2d))
        self.assertTrue(example["position_ids"] == self.expected_output["position_ids_2d"])

    def test_missing_data(self):
        orginal_input_ids = [item["input_ids"] for item in self.dataset]
        orginal_input_ids = [sum(orginal_input_ids, [])]
        inData = ZeroPaddingIterableDataset(self.dataset, self.tokenizer, max_length=128)
        tgt_input_ids = [item["input_ids"] for item in inData]
        self.assertEqual(orginal_input_ids, tgt_input_ids)
