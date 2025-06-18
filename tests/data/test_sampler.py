# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import unittest

from paddleformers.data import SamplerHelper
from paddleformers.datasets import load_dataset
from tests.common_test import CpuCommonTest
from tests.testing_utils import assert_raises, get_tests_dir


def cmp(x, y):
    return -1 if x < y else 1 if x > y else 0


class TestSampler(CpuCommonTest):
    @classmethod
    def setUpClass(cls):
        fixture_path = get_tests_dir(os.path.join("fixtures", "dummy"))
        cls.train_ds = load_dataset("json", data_files=os.path.join(fixture_path, "tnews", "train.json"))[0]

    def test_length(self):
        train_batch_sampler = SamplerHelper(self.train_ds)
        self.check_output_equal(len(train_batch_sampler), 10)
        self.check_output_equal(len(train_batch_sampler), train_batch_sampler.length)

        train_batch_sampler.length = 5
        self.check_output_equal(len(train_batch_sampler), 5)

    def test_iter1(self):
        train_ds_len = len(self.train_ds)
        ds_iter = iter(range(train_ds_len - 1, -1, -1))
        train_batch_sampler = SamplerHelper(self.train_ds, ds_iter)
        for i, sample in enumerate(train_batch_sampler):
            self.check_output_equal(i, train_ds_len - 1 - sample)

    def test_iter2(self):
        train_batch_sampler = SamplerHelper(self.train_ds)
        for i, sample in enumerate(train_batch_sampler):
            self.check_output_equal(i, sample)

    def test_list(self):
        train_batch_sampler = SamplerHelper(self.train_ds)
        list_sampler = train_batch_sampler.list()
        self.check_output_equal(type(iter(list_sampler)).__name__, "list_iterator")
        for i, sample in enumerate(list_sampler):
            self.check_output_equal(i, sample)

    def test_shuffle_no_buffer_size(self):
        train_batch_sampler = SamplerHelper(self.train_ds)
        shuffle_sampler = train_batch_sampler.shuffle(seed=102)
        expected_result = {0: 4, 1: 9}
        for i, sample in enumerate(shuffle_sampler):
            if i in expected_result.keys():
                self.check_output_equal(sample, expected_result[i])

    def test_shuffle_buffer_size(self):
        train_batch_sampler = SamplerHelper(self.train_ds)
        shuffle_sampler = train_batch_sampler.shuffle(buffer_size=10, seed=102)
        expected_result = {0: 4, 1: 9}
        for i, sample in enumerate(shuffle_sampler):
            if i in expected_result.keys():
                self.check_output_equal(sample, expected_result[i])

    def test_sort_buffer_size(self):
        train_ds_len = len(self.train_ds)
        ds_iter = iter(range(train_ds_len - 1, -1, -1))
        train_batch_sampler = SamplerHelper(self.train_ds, ds_iter)
        sort_sampler = train_batch_sampler.sort(cmp=lambda x, y, dataset: cmp(x, y), buffer_size=5)
        for i, sample in enumerate(sort_sampler):
            if i < 5:
                self.check_output_equal(i + 5, sample)
            else:
                self.check_output_equal(i - 5, sample)

    def test_sort_no_buffer_size(self):
        train_ds_len = len(self.train_ds)
        ds_iter = iter(range(train_ds_len - 1, -1, -1))
        train_batch_sampler = SamplerHelper(self.train_ds, ds_iter)
        sort_sampler = train_batch_sampler.sort(cmp=lambda x, y, dataset: cmp(x, y))
        for i, sample in enumerate(sort_sampler):
            self.check_output_equal(i, sample)

    def test_batch(self):
        train_batch_sampler = SamplerHelper(self.train_ds)
        batch_size = 3
        batch_sampler = train_batch_sampler.batch(batch_size)
        for i, sample in enumerate(batch_sampler):
            for j, minibatch in enumerate(sample):
                self.check_output_equal(i * batch_size + j, minibatch)

    @assert_raises(ValueError)
    def test_batch_oversize(self):
        train_batch_sampler = SamplerHelper(self.train_ds)
        batch_size = 3

        batch_sampler = train_batch_sampler.batch(
            batch_size,
            key=lambda size_so_far, minibatch_len: max(size_so_far, minibatch_len),
            batch_size_fn=lambda new, count, sofar, data_source: len(data_source),
        )
        for i, sample in enumerate(batch_sampler):
            for j, minibatch in enumerate(sample):
                self.check_output_equal(i * batch_size + j, minibatch)

    def test_shard(self):
        train_batch_sampler = SamplerHelper(self.train_ds)
        shard_sampler1 = train_batch_sampler.shard(2, 0)
        shard_sampler2 = train_batch_sampler.shard(2, 1)
        for i, sample in enumerate(shard_sampler1):
            self.check_output_equal(i * 2, sample)

        for i, sample in enumerate(shard_sampler2):
            self.check_output_equal(i * 2 + 1, sample)

    def test_shard_default(self):
        train_batch_sampler = SamplerHelper(self.train_ds)
        shard_sampler1 = train_batch_sampler.shard()
        for i, sample in enumerate(shard_sampler1):
            self.check_output_equal(i, sample)

    def test_apply(self):
        train_ds_len = len(self.train_ds)
        ds_iter = iter(range(train_ds_len - 1, -1, -1))
        train_batch_sampler = SamplerHelper(self.train_ds, ds_iter)
        apply_sampler = train_batch_sampler.apply(
            lambda sampler: SamplerHelper.sort(sampler, cmp=lambda x, y, dataset: cmp(x, y))
        )
        for i, sample in enumerate(apply_sampler):
            self.check_output_equal(i, sample)


if __name__ == "__main__":
    unittest.main()
