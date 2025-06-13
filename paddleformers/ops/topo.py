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

from collections import namedtuple

import numpy as np

GroupInfo = namedtuple("GroupInfo", ["size", "rank", "world"])


class Topology:
    def __init__(
        self,
        device_rank,
        world_size,
        dp_degree=None,
        pp_degree=1,
        sharding_degree=1,
        mp_degree=1,
        sep_degree=1,
        order=["dp", "pp", "sharding", "mp", "sep"],
    ):
        assert set(order) == {"dp", "pp", "sharding", "mp", "sep"}, f"Illegal order : {order}"
        self.order = order

        degree_map = {
            "dp": dp_degree,
            "pp": pp_degree,
            "sharding": sharding_degree,
            "mp": mp_degree,
            "sep": sep_degree,
        }
        shape = [degree_map[key] for key in self.order]

        arr = np.arange(0, dp_degree * pp_degree * sharding_degree * mp_degree * sep_degree).reshape(shape)
        ranks = [rank[0] for rank in np.where(arr == device_rank)]

        self.world = GroupInfo(size=world_size, rank=device_rank, world=list(range(0, world_size)))
        worlds = []
        for i in range(len(ranks)):
            indexes = tuple(ranks[:i] + [slice(None)] + ranks[(i + 1) :])
            worlds.append(arr[indexes])

        for i, key in enumerate(self.order):
            if key == "dp":
                self.dp_info = GroupInfo(size=len(worlds[i]), rank=ranks[i], world=worlds[i].tolist())
            elif key == "pp":
                self.pp_info = GroupInfo(size=len(worlds[i]), rank=ranks[i], world=worlds[i].tolist())
            elif key == "sharding":
                self.sharding_info = GroupInfo(size=len(worlds[i]), rank=ranks[i], world=worlds[i].tolist())
            elif key == "mp":
                self.mp_info = GroupInfo(size=len(worlds[i]), rank=ranks[i], world=worlds[i].tolist())
            elif key == "sep":
                self.sep_info = GroupInfo(size=len(worlds[i]), rank=ranks[i], world=worlds[i].tolist())

        self.is_last = self.pp_info.rank == self.pp_info.size - 1

        data_arr = np.arange(0, dp_degree * sharding_degree).reshape([dp_degree, sharding_degree])
        for i, key in enumerate(self.order):
            if key != "dp" and key != "sharding":
                data_arr = np.expand_dims(data_arr, axis=i).repeat(degree_map[key], axis=i)

        self.data_info = GroupInfo(
            size=int(self.dp_info.size * self.sharding_info.size),
            rank=int(self.dp_info.rank * self.sharding_info.size + self.sharding_info.rank),
            world=data_arr.reshape(-1).tolist(),
        )

        assert self.data_info.world[device_rank] == self.data_info.rank, "Data rank calculate error!"
        self.data_inner_times = self.world.size // self.data_info.size

    def __repr__(self):
        return f"dp_info:\n\t {self.dp_info}, \npp_info:\n\t {self.pp_info}, \nsharding_info:\n\t {self.sharding_info}, \nmp_info:\n\t {self.mp_info}, \nsep_info:\n\t {self.sep_info}, \ndata_info:\n\t {self.data_info}, \norder:\n\t {self.order}"
