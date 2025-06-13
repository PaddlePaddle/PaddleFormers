#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import division, print_function

import paddle

__all__ = ["DistributedBatchSampler"]


class DistributedBatchSampler(paddle.io.BatchSampler):
    """Sampler that restricts data loading to a subset of the dataset.

    In such case, each process can pass a DistributedBatchSampler instance
    as a DataLoader sampler, and load a subset of the original dataset that
    is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Args:
        dataset(paddle.io.Dataset): this could be a `paddle.io.Dataset` implement
                     or other python object which implemented
                     `__len__` for BatchSampler to get sample
                     number of data source.
        batch_size(int): sample indice number in a mini-batch indices.
        num_replicas(int, optional): process number in distributed training.
            If :attr:`num_replicas` is None, :attr:`num_replicas` will be
            retrieved from :code:`paddle.distributed.ParallenEnv`.
            Default None.
        rank(int, optional): the rank of the current process among :attr:`num_replicas`
            processes. If :attr:`rank` is None, :attr:`rank` is retrieved from
            :code:`paddle.distributed.ParallenEnv`. Default None.
        shuffle(bool): whether to shuffle indices order before generating
            batch indices. Default False.
        drop_last(bool): whether drop the last incomplete batch dataset size
            is not divisible by the batch size. Default False

    Examples:
        .. code-block:: python

            import numpy as np

            from paddle.io import Dataset, DistributedBatchSampler

            # init with dataset
            class RandomDataset(Dataset):
                def __init__(self, num_samples):
                    self.num_samples = num_samples

                def __getitem__(self, idx):
                    image = np.random.random([784]).astype('float32')
                    label = np.random.randint(0, 9, (1, )).astype('int64')
                    return image, label

                def __len__(self):
                    return self.num_samples

            dataset = RandomDataset(100)
            sampler = DistributedBatchSampler(dataset, batch_size=64)

            for data in sampler:
                # do something
                break
    """

    def __init__(
        self, dataset, batch_size, num_replicas=None, rank=None, shuffle=False, drop_last=False, consumed_samples=0
    ):
        self.dataset = dataset

        assert isinstance(batch_size, int) and batch_size > 0, "batch_size should be a positive integer"
        self.batch_size = batch_size
        assert isinstance(shuffle, bool), "shuffle should be a boolean value"
        self.shuffle = shuffle
        assert isinstance(drop_last, bool), "drop_last should be a boolean number"

        from paddle.distributed import ParallelEnv

        if num_replicas is not None:
            assert isinstance(num_replicas, int) and num_replicas > 0, "num_replicas should be a positive integer"
            self.nranks = num_replicas
        else:
            self.nranks = ParallelEnv().nranks

        if rank is not None:
            assert isinstance(rank, int) and rank >= 0, "rank should be a non-negative integer"
            self.local_rank = rank
        else:
            self.local_rank = ParallelEnv().local_rank

        self.drop_last = drop_last
        self.epoch = 0

        self.consumed_samples = consumed_samples
        if self.dataset is None:
            # In pre-training mode when using distributed dataloader, the input dataset can be None. We should handle this situation.
            self.num_samples = 0
        else:
            self.num_samples = int(len(self.dataset) * 1.0 / self.nranks)
        self.total_size = self.num_samples * self.nranks

    def get_start_end_idx(self):
        start_idx = self.local_rank * self.batch_size
        end_idx = start_idx + self.batch_size
        return start_idx, end_idx

    def __iter__(self):
        assert (
            self.consumed_samples % self.nranks == 0
        ), "The consumed_samples should be divided by nranks. consumed_samples=%d, nranks=%s" % (
            self.consumed_samples,
            self.nranks,
        )
        self.remain_num_samples = int((len(self.dataset) - self.consumed_samples) * 1.0 / self.nranks)
        self.remain_total_size = self.remain_num_samples * self.nranks
        self.batch_size_times_rank_size = self.batch_size * self.nranks

        batch_indices = []
        for idx in range(self.consumed_samples, self.total_size):
            batch_indices.append(idx)
            if len(batch_indices) == self.batch_size_times_rank_size:
                start_idx, end_idx = self.get_start_end_idx()
                yield batch_indices[start_idx:end_idx]
                batch_indices = []
        if not self.drop_last and len(batch_indices) > 0:
            yield batch_indices

    def __len__(self):
        num_samples = self.num_samples
        num_samples += int(not self.drop_last) * (self.batch_size - 1)
        return num_samples // self.batch_size

    def set_epoch(self, epoch=0, consumed_samples=0):
        """
        Sets the epoch number. When :attr:`shuffle=True`, this number is used
        as seeds of random numbers. By default, users may not set this, all
        replicas (workers) use a different random ordering for each epoch.
        If set same number at each epoch, this sampler will yield the same
        ordering at all epoches.

        Arguments:
            epoch (int): Epoch number.

        Examples:
            .. code-block:: python

                from paddle.io import Dataset, DistributedBatchSampler

                # init with dataset
                class RandomDataset(Dataset):
                    def __init__(self, num_samples):
                        self.num_samples = num_samples

                    def __getitem__(self, idx):
                        image = np.random.random([784]).astype('float32')
                        label = np.random.randint(0, 9, (1, )).astype('int64')
                        return image, label

                    def __len__(self):
                        return self.num_samples

                dataset = RandomDataset(100)
                sampler = DistributedBatchSampler(dataset, batch_size=64)

                for epoch in range(10):
                    sampler.set_epoch(epoch)
        """
        self.epoch = epoch
        # if we reset the epoch, the consumed_samples should be set to 0.
        self.consumed_samples = consumed_samples
