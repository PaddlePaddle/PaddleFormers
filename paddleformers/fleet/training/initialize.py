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

import logging

import paddle.distributed as dist
from paddle.distributed import fleet

from paddleformers.fleet import parallel_state as ps
from paddleformers.fleet.training.arguments import parse_args
from paddleformers.fleet.training.global_vars import set_global_variables


def initialize_fleet(
    strategy: fleet.DistributedStrategy,
    parsed_args=None,
    **kwargs,
):
    # Parse arguments
    if parsed_args is None:
        args = parse_args(**kwargs, ignore_unknown_args=True)
    else:
        args = parsed_args

    set_global_variables(args)
    set_logging(args)

    fleet.init(is_collective=True, strategy=strategy)
    hcg = fleet.get_hybrid_communicate_group()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    ps.initialize_model_parallel(hcg)
    print(f"fleet initialize successfully: {rank=} {world_size=}")


def set_logging(args):
    """Set the logging level and logging format"""
    logger = logging.getLogger("paddleformers.fleet")

    # set logging level
    logging_level = getattr(args, "logging_level", logging.INFO)
    logger.setLevel(logging_level)

    # set logging format
    import colorlog

    format_string = "%(log_color)s[%(asctime)-15s] [%(levelname)8s] %(filename)s:%(lineno)d%(reset)s - %(message)s"
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(format_string))
    logger.addHandler(handler)
