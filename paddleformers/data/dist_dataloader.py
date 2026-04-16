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
    nested_scatter_tensor,
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
        reader_buffer_size=2,
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

        if hasattr(self._hcg, "get_context_parallel_world_size") and self._hcg.get_context_parallel_world_size() > 1:
            self.dist_data_loader_group = self._hcg.get_cp_mp_parallel_group()
            self.dist_data_loader_src_rank = self._hcg.get_cp_mp_parallel_group_src_rank()
        else:
            self.dist_data_loader_group = self.mp_group
            self.dist_data_loader_src_rank = self.mp_src_rank

        self.pp_rank = self._hcg.get_stage_id()
        self.dp_rank = self._hcg.get_data_parallel_rank()
        sharding_rank = self._hcg.get_sharding_parallel_rank()
        self._need_data = (self.mp_rank == 0) and (self.pp_rank == 0)

        if self._need_data:
            self._dataloader = paddle.io.DataLoader(
                dataset=dataset,
                feed_list=feed_list,
                places=places,
                return_list=return_list,
                batch_sampler=batch_sampler,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
                collate_fn=collate_fn,
                num_workers=num_workers,
                use_buffer_reader=use_buffer_reader,
                prefetch_factor=prefetch_factor,
                use_shared_memory=use_shared_memory,
                timeout=timeout,
                worker_init_fn=worker_init_fn,
                persistent_workers=persistent_workers,
                reader_buffer_size=reader_buffer_size,
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
        if self.dist_data_loader_group.nranks > 1:
            if process_rank == self.dist_data_loader_src_rank:
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
        if self.dist_data_loader_group.nranks > 1 and self.pp_rank == 0:
            paddle.distributed.broadcast_object_list(
                fake_data,
                src=self.dist_data_loader_src_rank,
                group=self.dist_data_loader_group,
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
        if self.dist_data_loader_group.nranks > 1:
            if process_rank != self.dist_data_loader_src_rank:
                data = nested_empty_tensor(fake_data)
        if dst_pp_group is not None:
            if process_rank != dst_pp_group.ranks[0]:
                data = nested_empty_tensor(fake_data)

        if self.dist_data_loader_group.nranks > 1 and self.pp_rank == 0:
            data = nested_broadcast_tensor(data, src=self.dist_data_loader_src_rank, group=self.dist_data_loader_group)
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


def init_stream_data_group():
    """
    Create a communication group spanning all data-reading nodes
    (ranks where model==0 and pipe==0 and sep==0).

    Returns:
        (group, src_rank): The communication group and global rank 0 as source.
                           Returns (None, 0) if only 1 data-reading node exists.
    """
    hcg = fleet.get_hybrid_communicate_group()
    topo = hcg._topo
    world_size = paddle.distributed.get_world_size()

    dataset_ranks = []
    for rank in range(world_size):
        coord = topo.get_coord(rank)
        logger.info(
            "[dataflow] rank: {}, model: {}, pipe: {}, sep: {}".format(rank, coord.model, coord.pipe, coord.sep)
        )
        if coord.model == 0 and coord.pipe == 0 and getattr(coord, "sep", 0) == 0:
            dataset_ranks.append(rank)

    if len(dataset_ranks) <= 1:
        return None, 0

    group = paddle.distributed.new_group(ranks=dataset_ranks)
    return group, 0


class StreamDistDataLoader:
    """
    Distributed data loader for streaming (iterable) datasets.

    Only global rank 0 reads from the data source and distributes
    batches to all dataset ranks (dp/sharding) via scatter. Each dataset
    rank then broadcasts to its TP/PP/CP peers using the same mechanism
    as DistDataLoader.

    This eliminates the need for IterableDatasetShard, which wastefully
    iterates the full data stream on every rank.
    """

    def __init__(
        self,
        dataset,
        batch_size=1,
        collate_fn=None,
        num_workers=0,
        prefetch_factor=2,
        use_shared_memory=True,
        persistent_workers=False,
        reader_buffer_size=2,
        stream_data_group=None,
        pp_data_group=None,
        eval=False,
    ):
        self._hcg = fleet.get_hybrid_communicate_group()
        self.eval = eval
        self._pp_data_group = pp_data_group
        self._batch_size = batch_size

        # Init pp data comm group.
        if self._hcg.get_pipe_parallel_world_size() > 1:
            self._pp_group = self._hcg.get_pipe_parallel_group()
        else:
            self._pp_group = None

        self.mp_group = self._hcg.get_model_parallel_group()
        self.mp_rank = self._hcg.get_model_parallel_rank()
        self.mp_src_rank = self._hcg.get_model_parallel_group_src_rank()

        if hasattr(self._hcg, "get_context_parallel_world_size") and self._hcg.get_context_parallel_world_size() > 1:
            self.dist_data_loader_group = self._hcg.get_cp_mp_parallel_group()
            self.dist_data_loader_src_rank = self._hcg.get_cp_mp_parallel_group_src_rank()
        else:
            self.dist_data_loader_group = self.mp_group
            self.dist_data_loader_src_rank = self.mp_src_rank

        self.pp_rank = self._hcg.get_stage_id()
        self.dp_rank = self._hcg.get_data_parallel_rank()
        self._need_data = (self.mp_rank == 0) and (self.pp_rank == 0)

        # Stream data group: spans all data-reading nodes for scatter
        self._stream_data_group = stream_data_group
        self._stream_data_src = 0  # global rank 0

        if self._stream_data_group is not None:
            self._dataset_world_size = self._stream_data_group.nranks
        else:
            self._dataset_world_size = 1

        self._is_global_rank_0 = paddle.distributed.get_rank() == 0
        self.dataset = dataset

        if self._is_global_rank_0:
            if dataset is None:
                dataset = IterableDummyDataset()
                logger.info("rank 0 has no data, use Dummy dataset")

            self._dataloader = paddle.io.DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                collate_fn=collate_fn,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                use_shared_memory=use_shared_memory,
                persistent_workers=persistent_workers,
                reader_buffer_size=reader_buffer_size,
            )
        else:
            logger.info(
                "StreamDistDataLoader: rank {} does not read data, "
                "only global rank 0 reads.".format(paddle.distributed.get_rank())
            )

        if self._need_data:
            self._lazy_dataloader_iter = None

    @property
    def _dataloader_iter(self):
        if self._lazy_dataloader_iter is None:
            self._lazy_dataloader_iter = iter(self._dataloader)
        return self._lazy_dataloader_iter

    def __len__(self):
        raise ValueError("raise error for `paddleformers.trainer.trainer_utils.has_length`")

    def __iter__(self):
        return self

    def _scatter_data(self):
        """
        Global rank 0 reads dataset_world_size batches and scatters
        one batch to each data-reading node via the stream_data_group.

        Returns the batch for this rank, or None if data is exhausted.
        """
        if self._stream_data_group is None or self._dataset_world_size <= 1:
            # Only 1 dataset rank, no scatter needed. This rank is global rank 0.
            try:
                data = next(self._dataloader_iter)
                data = nested_copy_place(data, place=paddle.framework._current_expected_place())
                return data
            except StopIteration:
                return None

        if self._is_global_rank_0:
            batches = []
            exhausted = False
            try:
                for _ in range(self._dataset_world_size):
                    batch = next(self._dataloader_iter)
                    batch = nested_copy_place(batch, place=paddle.framework._current_expected_place())
                    batches.append(batch)
            except StopIteration:
                exhausted = True

            if exhausted or len(batches) < self._dataset_world_size:
                # Not enough data for a full round, signal stop
                fake_data = [None] * self._dataset_world_size
                paddle.distributed.broadcast_object_list(
                    fake_data, src=self._stream_data_src, group=self._stream_data_group
                )
                return None

            # Broadcast per-rank metadata (supports variable tensor shapes across batches)
            all_meta = [nested_reduce_tensor(b) for b in batches]
            paddle.distributed.broadcast_object_list(
                all_meta, src=self._stream_data_src, group=self._stream_data_group
            )

            # Use rank 0's own metadata to allocate output
            src_index = self._stream_data_group.ranks.index(self._stream_data_src)
            out_data = nested_empty_tensor(all_meta[src_index])
            out_data = nested_scatter_tensor(
                batches, out_data, src=self._stream_data_src, group=self._stream_data_group
            )
            return out_data
        else:
            # Non-rank-0 dataset rank: receive metadata then recv tensors
            all_meta = [None] * self._dataset_world_size
            paddle.distributed.broadcast_object_list(
                all_meta, src=self._stream_data_src, group=self._stream_data_group
            )

            if all_meta[0] is None:
                return None

            # Pick this rank's metadata
            my_index = self._stream_data_group.ranks.index(paddle.distributed.get_rank())
            out_data = nested_empty_tensor(all_meta[my_index])
            out_data = nested_scatter_tensor(None, out_data, src=self._stream_data_src, group=self._stream_data_group)
            return out_data

    def _broadcast_data(self, data):
        """
        Broadcast data from each data-reading node to its TP/PP/CP peers.
        Same logic as DistDataLoader._broadcast_data.
        """
        # No TP/CP/PP peers to broadcast to, return directly
        if self.dist_data_loader_group.nranks <= 1 and self._pp_group is None:
            if data is None:
                raise StopIteration
            return data

        process_rank = paddle.distributed.get_rank()
        if self.dist_data_loader_group.nranks > 1:
            if process_rank == self.dist_data_loader_src_rank:
                fake_data = [nested_reduce_tensor(data)]
            else:
                if data is not None:
                    logger.warning(f"Your local rank {process_rank} are forbidden to have a state_dict.")
                fake_data = [None]
        if self._pp_group is not None:
            if process_rank == self._pp_group.ranks[0]:
                fake_data = [nested_reduce_tensor(data)]
            else:
                if data is not None:
                    logger.warning(f"Your local rank {process_rank} are forbidden to have a state_dict.")
                fake_data = [None]
        if self.dist_data_loader_group.nranks > 1 and self.pp_rank == 0:
            paddle.distributed.broadcast_object_list(
                fake_data,
                src=self.dist_data_loader_src_rank,
                group=self.dist_data_loader_group,
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
        if self.dist_data_loader_group.nranks > 1:
            if process_rank != self.dist_data_loader_src_rank:
                data = nested_empty_tensor(fake_data)
        if dst_pp_group is not None:
            if process_rank != dst_pp_group.ranks[0]:
                data = nested_empty_tensor(fake_data)

        if self.dist_data_loader_group.nranks > 1 and self.pp_rank == 0:
            data = nested_broadcast_tensor(data, src=self.dist_data_loader_src_rank, group=self.dist_data_loader_group)
        if dst_pp_group is not None:
            data = nested_broadcast_tensor(data, src=dst_pp_group.ranks[0], group=dst_pp_group)
        if data is None:
            data = {}

        return data

    def __next__(self):
        data = None
        if self._need_data:
            try:
                data = self._scatter_data()
            except Exception as e:
                logger.debug(e)
        data = self._broadcast_data(data)
        return data
