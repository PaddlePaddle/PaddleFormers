# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""Pipeline Parallelism communication utilities (MTP Magic Send).

Migrated broadcast_data_obj from ernie_core/models/ernie5/pp_need_data.py,
and added init_magic_send_comm_group for creating 3-rank communication subgroups.
"""

from dataclasses import dataclass
from functools import reduce
from itertools import groupby

import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.utils.layers_utils import flatten, map_structure, pack_sequence_as

# Sentinel object used as placeholder in ret_flat before tensors are filled.
_UNFILLED = object()


@dataclass
class _DtypeSndShape:
    """Describes a tensor's dtype and shape, used as broadcast metadata template."""

    dtype: str
    shape: list

    def size(self):
        """Return total number of elements in the tensor."""
        return reduce(lambda x, y: x * y, self.shape)


def split_group(grouped, split_size):
    """Split grouped list into chunks by cumulative size to avoid single concat exceeding 2^31.

    Args:
        grouped: list of (index, _DtypeSndShape) tuples
        split_size: max number of elements per chunk

    Yields:
        list: one chunk
    """
    ret = []
    while grouped:
        if sum([r[1].size() for r in ret]) > split_size:
            yield ret
            ret = []
        ret.append(grouped.pop())
    if ret:
        yield ret


def broadcast_data_obj(data, src_rank, group):
    """Broadcast an arbitrarily nested data structure where each leaf is a paddle.Tensor or None.

    Core logic:
    1. src_rank extracts dtype+shape metadata from all tensors, sends via broadcast_object_list
    2. Flatten the nested structure, group by dtype
    3. Concat same-dtype tensors into one buffer for a single broadcast (reduce comm calls)
    4. Receiver splits+reshapes buffer back to original structure

    Args:
        data: arbitrarily nested structure (list/tuple/dict) with paddle.Tensor or None leaves
        src_rank (int): global rank of the sender
        group (ProcessGroup): communication group

    Returns:
        data: broadcasted nested structure (same structure as input)
    """
    this_rank = dist.get_rank()

    # 1. Broadcast metadata template
    if this_rank == src_rank:
        template = [
            map_structure(
                lambda x: _DtypeSndShape(dtype=x.dtype, shape=x.shape)
                if x is not None
                else _DtypeSndShape(dtype="", shape=[0]),
                data,
            )
        ]
    else:
        template = [None]
    dist.broadcast_object_list(template, src_rank, group)
    template = template[0]

    # 2. Flatten nested structure
    temp_flat = flatten(template)
    data_flat = flatten(data)

    def keyfn(i):
        return str(i[1].dtype)

    # 3. Group by dtype and broadcast
    ret_flat = [_UNFILLED for _ in range(len(temp_flat))]
    for dtype, grouped in groupby(
        sorted(enumerate(temp_flat), key=keyfn), keyfn
    ):
        grouped = list(grouped)
        for grouped_chunk in split_group(grouped, 2**18):
            idxs = [g[0] for g in grouped_chunk]
            if not dtype:
                # Skip None values
                for id in idxs:
                    ret_flat[id] = None
                continue

            data_buf_shapes = [
                reduce(lambda x, y: x * y, g[1].shape) for g in grouped_chunk
            ]
            if this_rank == src_rank:
                data_buf = paddle.concat(
                    [data_flat[i].reshape([-1]) for i in idxs], 0
                )
            else:
                data_buf = paddle.empty(
                    [sum(data_buf_shapes)], dtype=grouped_chunk[0][1].dtype
                )
            dist.broadcast(data_buf, src_rank, group)

            # 4. Receiver reconstructs tensors
            if this_rank != src_rank:
                if len(data_buf_shapes) == 1:
                    data_buf = [data_buf]
                else:
                    data_buf = data_buf.split(data_buf_shapes, axis=0)
                for g, data_chunk in zip(grouped_chunk, data_buf):
                    ret_flat[g[0]] = data_chunk.reshape(g[1].shape)

    if this_rank != src_rank:
        unfilled = [i for i, r in enumerate(ret_flat) if r is _UNFILLED]
        if unfilled:
            raise RuntimeError(
                f"broadcast_data_obj: tensor(s) at index {unfilled} were not filled after broadcast."
            )
        data = pack_sequence_as(template, ret_flat)
    return data


def init_magic_send_comm_group():
    """Create MTP magic send 3-rank communication group [rank_0, rank_{N-2}, rank_{N-1}].

    Aligned with ernie5 implementation: forms a subgroup from the first and last
    two ranks of each PP group, used to broadcast input_ids from the first stage
    to the last two stages.

    Returns:
        ProcessGroup or None: the communication subgroup for this rank, or None if not a member
    """
    topo = fleet.get_hybrid_communicate_group()._topo
    parallel_groups = topo.get_comm_list("pipe")

    comm_group = None
    for group in parallel_groups:
        if len(group) > 2:
            ranks = [group[0], group[-2], group[-1]]
        else:
            ranks = [group[0], group[-1]]
        new_group = dist.new_group(ranks=ranks)
        if dist.get_rank() in ranks:
            comm_group = new_group
    return comm_group
