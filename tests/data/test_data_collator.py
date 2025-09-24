# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import os
import shutil
import tempfile
import unittest

import numpy as np
import paddle

from paddleformers.data import default_data_collator


class DataCollatorIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()

        vocab_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
        self.vocab_file = os.path.join(self.tmpdirname, "vocab.txt")
        with open(self.vocab_file, "w", encoding="utf-8") as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def test_default_with_dict(self):
        features = [{"label": i, "inputs": [0, 1, 2, 3, 4, 5]} for i in range(8)]
        batch = default_data_collator(features)

        self.assertTrue(batch["labels"].equal_all(paddle.to_tensor(list(range(8)))))
        self.assertEqual(batch["labels"].dtype, paddle.int64)
        self.assertEqual(batch["inputs"].shape, [8, 6])

        # With label_ids
        features = [{"label_ids": [0, 1, 2], "inputs": [0, 1, 2, 3, 4, 5]} for i in range(8)]
        batch = default_data_collator(features)
        self.assertTrue(batch["labels"].equal_all(paddle.to_tensor([[0, 1, 2]] * 8)))
        self.assertEqual(batch["labels"].dtype, paddle.int64)
        self.assertEqual(batch["inputs"].shape, [8, 6])

        # Features can already be tensors
        features = [{"label": i, "inputs": np.random.randint(0, 10, [10])} for i in range(8)]
        batch = default_data_collator(features)
        self.assertTrue(batch["labels"].equal_all(paddle.to_tensor(list(range(8)))))
        self.assertEqual(batch["labels"].dtype, paddle.int64)
        self.assertEqual(batch["inputs"].shape, [8, 10])

        # Labels can already be tensors
        features = [{"label": paddle.to_tensor(i), "inputs": np.random.randint(0, 10, [10])} for i in range(8)]

        batch = default_data_collator(features)
        self.assertEqual(batch["labels"].dtype, paddle.int64)
        self.assertTrue(batch["labels"].equal_all(paddle.to_tensor(list(range(8)))))
        self.assertEqual(batch["labels"].dtype, paddle.int64)
        self.assertEqual(batch["inputs"].shape, [8, 10])

    def test_default_classification_and_regression(self):
        data_collator = default_data_collator

        features = [{"input_ids": [0, 1, 2, 3, 4], "label": i} for i in range(4)]
        batch = data_collator(features)
        self.assertEqual(batch["labels"].dtype, paddle.int64)

        features = [{"input_ids": [0, 1, 2, 3, 4], "label": float(i)} for i in range(4)]
        batch = data_collator(features)
        self.assertEqual(batch["labels"].dtype, paddle.float32)

    def test_default_with_no_labels(self):
        features = [{"label": None, "inputs": [0, 1, 2, 3, 4, 5]} for i in range(8)]
        batch = default_data_collator(features)
        self.assertTrue("labels" not in batch)
        self.assertEqual(batch["inputs"].shape, [8, 6])

        # With label_ids
        features = [{"label_ids": None, "inputs": [0, 1, 2, 3, 4, 5]} for i in range(8)]
        batch = default_data_collator(features)
        self.assertTrue("labels" not in batch)
        self.assertEqual(batch["inputs"].shape, [8, 6])
