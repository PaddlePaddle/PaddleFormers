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

    Uses interleaved sharding (indices[rank::world_size]) to align with
    ms-swift's BatchSamplerShard, instead of the contiguous-block strategy
    used by paddle.io.DistributedBatchSampler.

    Supports consumed_samples for checkpoint-resume (resume-from-middle-of-epoch).

    Args:
        dataset(paddle.io.Dataset): dataset to sample from.
        batch_size(int): sample indices number in a mini-batch.
        num_replicas(int, optional): number of processes. Defaults to ParallelEnv().nranks.
        rank(int, optional): rank of current process. Defaults to ParallelEnv().local_rank.
        shuffle(bool): whether to shuffle. Uses paddle.randperm with seed=base_seed+epoch.
        drop_last(bool): whether to drop the last incomplete batch.
        consumed_samples(int): total number of samples already consumed across all ranks
            (used to resume from a checkpoint mid-epoch).
        data_seed(int): base random seed. Actual seed per epoch = data_seed + epoch.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        num_replicas=None,
        rank=None,
        shuffle=False,
        drop_last=False,
        consumed_samples=0,
        data_seed=0,
    ):
        self.dataset = dataset

        assert isinstance(batch_size, int) and batch_size > 0, "batch_size should be a positive integer"
        self.batch_size = batch_size
        assert isinstance(shuffle, bool), "shuffle should be a boolean value"
        self.shuffle = shuffle
        assert isinstance(drop_last, bool), "drop_last should be a boolean number"
        self.drop_last = drop_last

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

        self.consumed_samples = consumed_samples
        self.base_seed = data_seed
        self.curr_seed = data_seed
        self.epoch = 0
        self._acc_steps = 1  # Required by LastBatchPaddingSampler and compatible with paddle.io.BatchSampler

        if self.dataset is None:
            # In pre-training mode when using distributed dataloader, the input dataset can be None. We should handle this situation.
            self.num_samples = 0
        else:
            # floor truncation (no padding)
            total_size = (len(self.dataset) // self.nranks) * self.nranks
            self.num_samples = total_size // self.nranks
        self.total_size = self.num_samples * self.nranks

    def __iter__(self):
        assert (
            self.consumed_samples % self.nranks == 0
        ), "consumed_samples should be divisible by nranks. consumed_samples=%d, nranks=%d" % (
            self.consumed_samples,
            self.nranks,
        )

        # floor truncation (no padding)
        total_size = (len(self.dataset) // self.nranks) * self.nranks

        if self.shuffle:
            # need align with torch
            paddle.seed(self.curr_seed)
            total_idx = paddle.randperm(total_size).tolist()
            indices = total_idx[self.local_rank :: self.nranks]  # interleaved shard
        else:
            indices = list(range(self.local_rank, total_size, self.nranks))  # interleaved shard

        # Resume from checkpoint: skip already-consumed samples for this rank
        consumed_per_rank = self.consumed_samples // self.nranks
        indices = indices[consumed_per_rank:]

        batch = []
        for idx in indices:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if not self.drop_last and len(batch) > 0:
            yield batch

    def __len__(self):
        total_size = (len(self.dataset) // self.nranks) * self.nranks
        per_rank = total_size // self.nranks
        consumed_per_rank = self.consumed_samples // self.nranks
        remaining = per_rank - consumed_per_rank
        if self.drop_last:
            return remaining // self.batch_size
        else:
            return (remaining + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch=0, consumed_samples=0):
        """
        Update epoch and consumed_samples.

        When shuffle=True, the seed for the next iteration will be base_seed + epoch,
        consistent with ms-swift's BatchSamplerShard.set_epoch().

        Args:
            epoch(int): current epoch number.
            consumed_samples(int): total samples consumed across all ranks so far.
                Used to resume from a checkpoint mid-epoch.
        """
        self.curr_seed = self.base_seed + epoch
        self.epoch = epoch
        # if we reset the epoch, the consumed_samples should be set to 0.
        self.consumed_samples = consumed_samples
