# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
"""Unfied checkpoint check functions."""

import json
import os

import paddle
import paddle.distributed as dist
from paddle.distributed import fleet

from ...utils.env import (
    PADDLE_MASTER_WEIGHTS_INDEX_NAME,
    PADDLE_OPTIMIZER_INDEX_NAME,
    SAFE_MASTER_WEIGHTS_INDEX_NAME,
    SAFE_OPTIMIZER_INDEX_NAME,
)
from ...utils.log import logger
from ...utils.nested import flatten_list
from ..utils.helper import distributed_file, distributed_isfile
from .utils import (
    get_expected_state_dict,
    is_sharding_split_param_mode,
    select_model_weight_index,
    update_master_weight_status,
)

__all__ = ["check_unified_checkpoint", "check_unified_optimizer"]


def check_unified_checkpoint(args, model, resume_from_checkpoint, safe_serialization=False):
    index_filename = select_model_weight_index(model, resume_from_checkpoint, safe_serialization, local=False)
    index_filename = os.path.join(resume_from_checkpoint, index_filename)
    # Find index json file and distribute this file in global group.
    if distributed_isfile(index_filename):
        distributed_file(index_filename)
    else:
        raise Exception(
            f"Sorry, we can not find {index_filename}. This file should be appear at least on one machine."
        )

    with open(index_filename, "r") as f:
        index = json.loads(f.read())
    all_weight_filenames = sorted(set(index["weight_map"].values()))

    # Get existed weight file list on current machine.
    existed_filelist = []
    existed_files = []
    for filename in os.listdir(resume_from_checkpoint):
        if filename in all_weight_filenames:
            existed_files.append(filename)

    # Gather all the existed files in global group.
    dist.all_gather_object(existed_filelist, existed_files)
    flatten_existed_filelist = flatten_list(existed_filelist)
    diff_filelist = list(set(all_weight_filenames).difference(set(flatten_existed_filelist)))
    if len(diff_filelist) != 0:
        raise Exception(f"Sorry, the weight file list on the machines is not complete!, missing {diff_filelist}")

    # To decide whether to load the checkpoint locally, or need to dynamically send tensors across machines.
    local_resume = True
    if args.dataset_rank == 0 or args.use_expert_parallel:
        hcg = fleet.get_hybrid_communicate_group()
        tp_group = hcg.get_model_parallel_group()
        pp_group = hcg.get_pipe_parallel_group()
        dp_group = hcg.get_data_parallel_group()
        dp_rank = dp_group.rank if dp_group.nranks > 1 else 0

        need_files = set()
        state_dict = get_expected_state_dict(model)
        for key in state_dict.keys():
            filename = index["weight_map"][key]
            # When using expert parallel, there's no need to check tensors with `no_sync=False` when dp_rank > 0.
            if args.use_expert_parallel and dp_rank > 0 and not getattr(state_dict[key], "no_sync", False):
                continue
            need_files.add(filename)
        diff_filelist = list(need_files.difference(set(existed_files)))
        num_diff = paddle.to_tensor([len(diff_filelist)])
        if tp_group.nranks > 1:
            dist.all_reduce(num_diff, op=dist.ReduceOp.MAX, group=tp_group)
        if pp_group.nranks > 1:
            dist.all_reduce(num_diff, op=dist.ReduceOp.MAX, group=pp_group)
        if args.use_expert_parallel and dp_group.nranks > 1:
            dist.all_reduce(num_diff, op=dist.ReduceOp.MAX, group=dp_group)
        if num_diff.item() == 0:
            local_resume = True
        else:
            local_resume = False
    local_resume = paddle.to_tensor([local_resume])
    dist.all_reduce(local_resume, op=dist.ReduceOp.MIN)
    local_resume = local_resume.item()
    return local_resume


def check_unified_optimizer(args, model, optimizer, resume_from_checkpoint, safe_serialization=False):
    if not safe_serialization:
        index_filename, index_filename_master_weights = PADDLE_OPTIMIZER_INDEX_NAME, PADDLE_MASTER_WEIGHTS_INDEX_NAME
    else:
        index_filename, index_filename_master_weights = SAFE_OPTIMIZER_INDEX_NAME, SAFE_MASTER_WEIGHTS_INDEX_NAME
    index_filename = os.path.join(resume_from_checkpoint, index_filename)
    index_filename_master_weights = os.path.join(resume_from_checkpoint, index_filename_master_weights)

    # Find index json file and distribute the file in global group.
    if distributed_isfile(index_filename):
        distributed_file(index_filename)
    else:
        raise Exception(
            f"Sorry, we can not find {index_filename}. This file should be appear at least on one machine."
        )

    with open(index_filename, "r") as f:
        index = json.loads(f.read())
    all_optimizer_filenames = sorted(set(index["weight_map"].values()))

    has_master_weights = index["master_weights"]
    # update has_master_weights and index_filename_master_weights
    # 1. if the master weight exists, only has_master_weights is set True and loaded when needed
    # 2. if master weight does not exist, convert model weight to master weight when needed
    has_master_weights, index_filename_master_weights = update_master_weight_status(
        args, optimizer, has_master_weights, safe_serialization
    )
    if has_master_weights:
        index_filename_master_weights = os.path.join(resume_from_checkpoint, index_filename_master_weights)
        if distributed_isfile(index_filename_master_weights):
            distributed_file(index_filename_master_weights)
        else:
            raise Exception(
                f"Sorry, we can not find {index_filename_master_weights}. This file should be appear at least on one machine."
            )
        with open(index_filename_master_weights, "r") as f:
            index_mw = json.loads(f.read())
        all_mw_filenames = sorted(set(index_mw["weight_map"].values()))

    hcg = fleet.get_hybrid_communicate_group()
    tp_group = hcg.get_model_parallel_group()
    pp_group = hcg.get_pipe_parallel_group()
    dp_group = hcg.get_data_parallel_group()
    sharding_group = hcg.get_sharding_parallel_group()
    sharding_rank = sharding_group.rank
    dp_rank = dp_group.rank if dp_group.nranks > 1 else 0
    model_state_dict = get_expected_state_dict(model)
    struct2static_name_mappings = {k: v.name for k, v in model_state_dict.items()}

    if is_sharding_split_param_mode(args):
        # We do not check optimizer files completion for split_param, since it is very complicated. Directly support local resume.
        logger.warning("We only support local resume for split_param mode, do not support dynamically loading.")
        return True

    if sharding_group.nranks > 1:
        param2rank = optimizer._param2rank

    def check_complete(all_filenames):
        # Check whether the checkpoint files on machines are complete. If not complete, raise Exception.
        existed_filelist = []
        existed_files = []
        for filename in os.listdir(resume_from_checkpoint):
            if filename in all_filenames:
                existed_files.append(filename)

        dist.all_gather_object(existed_filelist, existed_files)
        flatten_existed_filelist = flatten_list(existed_filelist)
        diff_filelist = list(set(all_filenames).difference(set(flatten_existed_filelist)))
        if len(diff_filelist) != 0:
            raise Exception(
                f"Sorry, the optimizer file list on `data_parallel_rank==0` machines is not complete!, missing {diff_filelist}"
            )
        return existed_files

    def check_dynamic_load(args, weight_map, existed_files, is_master_weights=False, typename_set=None):
        # To decide whether to load the checkpoint locally, or need to dynamically distribute the checkpoint.
        local_resume = True
        if args.data_parallel_rank == 0 or args.use_expert_parallel:
            need_files = set()
            state_dict = get_expected_state_dict(model)

            for key in state_dict.keys():
                if state_dict[key].stop_gradient:
                    continue
                if model._keys_to_ignore_on_load_missing is not None and key in model._keys_to_ignore_on_load_missing:
                    continue
                if sharding_group.nranks > 1:
                    static_name = struct2static_name_mappings.get(key, None)
                    param_rank = param2rank.get(static_name, None)
                    if param_rank != sharding_rank:
                        continue

                # When using expert parallel, there's no need to check tensors with `no_sync=False` when dp_rank > 0.
                if args.use_expert_parallel and dp_rank > 0 and not getattr(state_dict[key], "no_sync", False):
                    continue

                if is_master_weights and state_dict[key].dtype == paddle.float32:
                    continue

                if not is_master_weights:
                    for type_name in typename_set:
                        type_key = key + "/" + type_name
                        filename = weight_map[type_key]
                        need_files.add(filename)
                else:
                    filename = weight_map[key]
                    need_files.add(filename)

            diff_filelist = list(need_files.difference(set(existed_files)))
            num_diff = paddle.to_tensor([len(diff_filelist)])
            if tp_group.nranks > 1:
                dist.all_reduce(num_diff, op=dist.ReduceOp.MAX, group=tp_group)
            if pp_group.nranks > 1:
                dist.all_reduce(num_diff, op=dist.ReduceOp.MAX, group=pp_group)
            if sharding_group.nranks > 1:
                dist.all_reduce(num_diff, op=dist.ReduceOp.MAX, group=sharding_group)
            if args.use_expert_parallel and dp_group.nranks > 1:
                dist.all_reduce(num_diff, op=dist.ReduceOp.MAX, group=dp_group)

            if num_diff.item() == 0:
                local_resume = True
            else:
                local_resume = False
        local_resume = paddle.to_tensor([local_resume])
        dist.all_reduce(local_resume, op=dist.ReduceOp.MIN)
        return local_resume.item()

    # check whether the optimizer checkpoint files are complete.
    existed_files = check_complete(all_optimizer_filenames)
    if has_master_weights:
        existed_files_mw = check_complete(all_mw_filenames)
    # get optimizer's param type name, like moment1_0.
    typename_set = set()
    for key in index["weight_map"].keys():
        _, typename = key.split("/")
        typename_set.add(typename)
    local_resume = check_dynamic_load(
        args, index["weight_map"], existed_files, is_master_weights=False, typename_set=typename_set
    )
    local_resume_rw = True
    if has_master_weights:
        local_resume_rw = check_dynamic_load(args, index_mw["weight_map"], existed_files_mw, is_master_weights=True)
    return local_resume & local_resume_rw
