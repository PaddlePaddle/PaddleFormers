# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

"""
Selective launch script.

Usage: python script/selective_launch.py <port> <ranks> <ranks> <ranks> ...
"""
import os
import sys


def parse_ranks(ranks_strs):
    """
    parse_ranks
    """
    # NOTE: You can return ranks directly here to change script/train_gpu.sh
    # and script/kill_process.sh together

    # Example 1: Use contiguous nodes [8, 16)
    return range(6, 7)

    # Example 2: Use non-contiguous nodes [4, 8) + {10} + [30, 32), i.e., [4, 5, 6, 7, 10, 30, 31]
    # return list(range(0, 16)) + list(range(24, 40))

    # Example 3:
    # Just Python code, return any nodes you want!

    if not ranks_strs:
        return None

    ranks = []
    for r in ranks_strs:
        r = eval(r)
        if isinstance(r, int):
            ranks.append(r)
        else:
            ranks.extend(r)
    return ranks


def main(port, ranks):
    """
    main
    """
    ips = [ip.strip() for ip in os.getenv("TRAINER_INSTANCES").split(",") if ip.strip()]
    if ranks is None:
        ranks = list(range(len(ips)))
    ranks = sorted(list(set(ranks)))
    my_rank = int(os.getenv("POD_INDEX", "0"))
    if my_rank not in ranks:
        return

    rank = ranks.index(my_rank)
    nranks = len(ranks)

    master = ips[ranks[0]]
    print(f"--master {master}:{port} --rank {rank} --nnodes {nranks}")


if __name__ == "__main__":
    main(int(sys.argv[1]), parse_ranks(sys.argv[2:]))
