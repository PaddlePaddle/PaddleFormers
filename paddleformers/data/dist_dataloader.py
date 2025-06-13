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

import paddle
from paddle.distributed import fleet

from ..utils.log import logger
from ..utils.nested import (
    nested_broadcast_tensor,
    nested_copy_place,
    nested_empty_tensor,
    nested_reduce_tensor,
)


class DummyDataset(paddle.io.Dataset):
    """
    A dummy dataset.
    """

    def __len__(self):
        return 0


class IterableDummyDataset(paddle.io.IterableDataset):
    def __iter__(self):
        return None


class DistDataLoader(paddle.io.DataLoader):
    """
    DistDataLoader is a wrapper of paddle.io.DataLoader.
    """

    def __init__(
        self,
        dataset,
        feed_list=None,
        places=None,
        return_list=True,
        batch_sampler=None,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        collate_fn=None,
        num_workers=0,
        use_buffer_reader=True,
        prefetch_factor=2,
        use_shared_memory=True,
        timeout=0,
        worker_init_fn=None,
        persistent_workers=False,
        **kwargs,
    ):

        eval = kwargs.pop("eval", False)
        is_iterable_dataset = kwargs.pop("is_iterable_dataset", False)
        self._pp_data_group = kwargs.pop("pp_data_group", None)

        if dataset is None:
            dataset = DummyDataset() if not is_iterable_dataset else IterableDummyDataset()
            logger.info("rank has no data, use Dummpy dataset")

        super().__init__(dataset=dataset, batch_sampler=batch_sampler, collate_fn=collate_fn, num_workers=num_workers)

        self._hcg = fleet.get_hybrid_communicate_group()
        self.eval = eval

        # Init pp data comm group.
        if self._hcg.get_pipe_parallel_world_size() > 1:
            self._pp_group = self._hcg.get_pipe_parallel_group()
        else:
            self._pp_group = None

        self.mp_group = self._hcg.get_model_parallel_group()
        self.mp_rank = self._hcg.get_model_parallel_rank()
        self.mp_src_rank = self._hcg.get_model_parallel_group_src_rank()

        self.pp_rank = self._hcg.get_stage_id()
        self.dp_rank = self._hcg.get_data_parallel_rank()
        sharding_rank = self._hcg.get_sharding_parallel_rank()
        self._need_data = (self.mp_rank == 0) and (self.pp_rank == 0)

        if self._need_data:
            self._dataloader = paddle.io.DataLoader(
                dataset,
                feed_list,
                places,
                return_list,
                batch_sampler,
                batch_size,
                shuffle,
                drop_last,
                collate_fn,
                num_workers,
                use_buffer_reader,
                prefetch_factor,
                use_shared_memory,
                timeout,
                worker_init_fn,
                persistent_workers,
            )

            self._lazy_dataloader_iter = None
        else:
            logger.info(
                "mp{}_pp{}_sharding{}_dp{} no data needed, "
                "skip init dataloader.".format(self.mp_rank, self.pp_rank, sharding_rank, self.dp_rank)
            )

    @property
    def _dataloader_iter(self):
        if self._lazy_dataloader_iter is None:
            self._lazy_dataloader_iter = iter(self._dataloader)
        return self._lazy_dataloader_iter

    def __len__(self):
        if self._need_data:
            return super().__len__()
        else:
            raise ValueError("raise error for `paddleformers.trainer.trainer_utils.has_length`")

    def __iter__(self):
        return self

    def _broadcast_data(self, data):
        process_rank = paddle.distributed.get_rank()
        if self.mp_group.nranks > 1:
            if process_rank == self.mp_src_rank:
                fake_data = [nested_reduce_tensor(data)]
            else:
                if data is not None:
                    logger.warning(
                        f"Your local rank {paddle.distributed.get_rank()} are forbidden to have a state_dict."
                    )
                fake_data = [None]
        if self._pp_group is not None:
            if process_rank == self._pp_group.ranks[0]:
                fake_data = [nested_reduce_tensor(data)]
            else:
                if data is not None:
                    logger.warning(
                        f"Your local rank {paddle.distributed.get_rank()} are forbidden to have a state_dict."
                    )
                fake_data = [None]
        if self.mp_group.nranks > 1 and self.pp_rank == 0:
            paddle.distributed.broadcast_object_list(
                fake_data,
                src=self.mp_src_rank,
                group=self.mp_group,
            )
        if self._pp_group is not None:
            paddle.distributed.broadcast_object_list(
                fake_data,
                src=self._pp_group.ranks[0],
                group=self._pp_group,
            )

        fake_data = fake_data[0]
        if fake_data is None:
            raise StopIteration

        dst_pp_group = self._pp_group if self.eval else self._pp_data_group
        if self.mp_group.nranks > 1:
            if process_rank != self.mp_src_rank:
                data = nested_empty_tensor(fake_data)
        if dst_pp_group is not None:
            if process_rank != dst_pp_group.ranks[0]:
                data = nested_empty_tensor(fake_data)

        if self.mp_group.nranks > 1 and self.pp_rank == 0:
            data = nested_broadcast_tensor(data, src=self.mp_src_rank, group=self.mp_group)
        if dst_pp_group is not None:
            data = nested_broadcast_tensor(data, src=dst_pp_group.ranks[0], group=dst_pp_group)
        # for pp1 - pp_{n-1}, Paddle need to receive empty dict for pipeline parallel.
        if data is None:
            data = {}

        return data

    def __next__(self):
        data = None
        if self._need_data:
            try:
                data = next(self._dataloader_iter)
                data = nested_copy_place(data, place=paddle.framework._current_expected_place())
            except Exception as e:
                logger.debug(e)
        data = self._broadcast_data(data)
        return data


def init_dataloader_comm_group():
    hcg = fleet.get_hybrid_communicate_group()
    topo = hcg._topo
    parallel_groups = topo.get_comm_list("pipe")
    parallel_comm_group = None

    for group in parallel_groups:
        ranks = [group[0], group[-1]]
        comm_group = paddle.distributed.new_group(ranks=ranks)
        if paddle.distributed.get_rank() in ranks:
            parallel_comm_group = comm_group
    return parallel_comm_group
