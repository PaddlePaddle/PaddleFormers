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

import json
import os
import re
from functools import reduce
from typing import List, Union

import paddle
from paddle.distributed.fleet.utils.log_util import logger

try:
    from paddle.distributed.flex_checkpoint.dcp.load_state_dict import (
        _load_state_dict,
        get_rank_to_read_files,
    )
except ModuleNotFoundError:
    try:
        from paddle.distributed.checkpoint.load_state_dict import (
            _load_state_dict,
            get_rank_to_read_files,
        )
    except ModuleNotFoundError:
        _load_state_dict = None
        get_rank_to_read_files = None


try:
    from paddle.distributed.flex_checkpoint.dcp.metadata import (
        LocalTensorIndex,
        LocalTensorMetadata,
        Metadata,
    )
except ModuleNotFoundError:
    try:
        from paddle.distributed.checkpoint.metadata import (
            LocalTensorIndex,
            LocalTensorMetadata,
            Metadata,
        )
    except ModuleNotFoundError:
        LocalTensorIndex = None
        LocalTensorMetadata = None
        Metadata = None

try:
    from paddle.distributed.flex_checkpoint.dcp.utils import flatten_state_dict
except ModuleNotFoundError:
    try:
        from paddle.distributed.checkpoint.utils import flatten_state_dict
    except ModuleNotFoundError:
        flatten_state_dict = None

MODEL_WEIGHT_SUFFIX = ".pdparams"
OPTIMIZER_WEIGHT_SUFFIX = ".pdopt"
SCHEDULER_NAME = "scheduler.pdparams"
SCALAR_NAME = "scalar.pdparams"
MODEL_META_FILE_NAME = "model_meta.json"
OPTIMIZER_STATE_NAME_SUFFIX = [".moment1", ".moment2", ".beta1_pow_acc", ".beta2_pow_acc", ".master_weight"]
MODEL_STATE_FILE_MIN_SIZE = 512


class CheckpointConverter:
    def __init__(
        self,
        hybrid_parallel_ckpt_path,
        state_dict,
        parameter_to_structured_name,
        trainging_args=None,
        patch_dict=None,
        local_view_pattern: Union[List, bool] = None,
    ):
        self.use_dist = True if paddle.distributed.get_world_size() > 1 else False
        self.path = hybrid_parallel_ckpt_path

        if trainging_args.ignore_load_lr_and_optim:
            state_dict.pop("optimizer")

        self.auto_parallel_state_dict = self.flatten_state_dict(state_dict)
        self.parameter_to_structured_name = self.gather_global_object(parameter_to_structured_name)
        model_state_global_shape = {}
        for k, v in self.auto_parallel_state_dict.items():
            model_state_global_shape[k] = v.shape
        self.model_state_global_shape = self.gather_global_object(model_state_global_shape)
        self.cur_rank = paddle.distributed.get_rank()

        (
            self.cur_rank_model_state_file_names,
            self.cur_rank_optimizer_state_file_names,
        ) = self.get_local_checkpoint_file_names()

        self.global_model_state_file_names = self.gather_global_object(self.cur_rank_model_state_file_names)

        self.global_optimizer_state_file_names = self.gather_global_object(self.cur_rank_optimizer_state_file_names)

        self.is_model_meta_exists = self.get_is_model_meta_exists_flag()
        self.is_model_state_stored = self.get_is_model_state_stored_flag()

        self.initial_distributed_configuration()

        if patch_dict is not None:
            self.patch_dict = patch_dict
            for k, v in self.parameter_to_structured_name.items():
                if v in self.patch_dict:
                    self.parameter_to_structured_name[k] = self.patch_dict[v]

            del_keys = []
            for k, v in self.auto_parallel_state_dict.items():
                if k in self.patch_dict:
                    del_keys.append(k)
            for k in del_keys:
                self.auto_parallel_state_dict[self.patch_dict[k]] = self.auto_parallel_state_dict[k]
            for k in del_keys:
                self.auto_parallel_state_dict.pop(k)
        # solve the problem of inconsistent parameter names in moe automatic parallel mode.
        if hasattr(trainging_args, "moe_group") and trainging_args.moe_group:
            if local_view_pattern is False:
                self.local_view_pattern_list = None
            else:
                if isinstance(local_view_pattern, list):
                    self.local_view_pattern_list = local_view_pattern
                else:
                    self.local_view_pattern_list = ["experts"]
        else:
            self.local_view_pattern_list = None

        flags = [
            ["tp degree", self.tp_degree],
            ["pp degree", self.pp_degree],
            ["sharding degree", self.sharding_degree],
            ["is model_meta exists", self.is_model_meta_exists],
            ["is model_state stored", self.is_model_state_stored],
        ]
        self.print_checkpoint_file_info(flags)

    def load_from_hybrid_parallel_checkpoint(self):
        """
        Automatically and inplace load the distributed checkpoint stored in hybrid parallel mode into the auto parallel state_dict.
        The main logic is as follows:
            1. Call rename_semi_auto_state_dict: Rename the keys of the auto parallel state_dict according to certain rules.
               (Why rename? To facilitate the subsequent correspondence between the optimizer state names of the semi-automatic and static optimizers.)
            2. Call gen_metadata_and_prepare_source_state_dict: Automatically parse the manual checkpoint file based on the state_dict information
               provided by auto parallel, obtaining the Metadata and state_dict required for auto parallel to load the checkpoint.
            3. Call load_state_dict: Automatically reshard and load.
            4. Special logic adaptation: In the save_sharded_model mode, the weights are obtained through the master_weight cast in the checkpoint.
        """
        self.rename_auto_parallel_state_dict()

        metadata, source_state_dict = self.gen_metadata_and_prepare_source_state_dict()
        logger.info("Generated the checkpoint’s metadata.")
        logger.debug(f"The checkpoint's metadata is {metadata}.")
        if not self.is_model_state_stored:
            assert self.optimizer_state_with_master_weights
            model_params = {}
            for state_name, state_value in self.auto_parallel_state_dict.items():
                self.auto_parallel_state_dict[state_name] = state_value.cuda()
                if state_name in self.parameter_to_structured_name.values():
                    model_params[state_name] = state_value
            for param_name in model_params.keys():
                self.auto_parallel_state_dict.pop(param_name)

            logger.info("Requesting GPU memory space to load master_weights.")
            appended_master_weight_names = []
            for param_name, param_value in model_params.items():
                master_weight = param_name + ".master_weight"
                if master_weight not in self.auto_parallel_state_dict:
                    appended_master_weight_names.append(master_weight)
                    if param_value.is_dist():
                        param_shape = param_value._local_value().shape
                    else:
                        param_shape = param_value.shape

                    tmp_tensor = paddle.zeros(param_shape, dtype="float32")
                    with paddle.base.dygraph.guard():
                        if param_value.is_dist():
                            self.auto_parallel_state_dict[
                                master_weight
                            ] = paddle.distributed.auto_parallel.api.dtensor_from_local(
                                tmp_tensor, param_value.process_mesh, param_value.placements
                            )
                        else:
                            self.auto_parallel_state_dict[master_weight] = tmp_tensor

            logger.info("Calling _load_state_dict to load the required weights.")
            _load_state_dict(self.auto_parallel_state_dict, source_state_dict, [metadata], offload=True)
            logger.info("Calling _load_state_dict completed, restored the required weights.")

            # In this scenario, the data type of the model state is bfloat16.
            for param_name, param_value in model_params.items():
                if param_value._is_initialized():
                    # These codes are compatible for both dense tensor and dist tensor
                    master_weight = self.auto_parallel_state_dict[param_name + ".master_weight"]
                    cast_master_weight = paddle.cast(master_weight, param_value.dtype)
                    paddle.assign(cast_master_weight, param_value)
            for master_weight_name in appended_master_weight_names:
                self.auto_parallel_state_dict.pop(master_weight_name)
        else:
            logger.info("Calling _load_state_dict to load the required weights.")
            _load_state_dict(self.auto_parallel_state_dict, source_state_dict, [metadata], offload=True)
            logger.info("Calling _load_state_dict completed, restored the required weights.")
        logger.info("Successfully loaded hybrid_parallel checkpoint!")

    def gen_metadata_and_prepare_source_state_dict(self):
        """
        Automatically parse the manual checkpoint file based on the state_dict information provided by auto parallel,
        obtaining the Metadata and state_dict required for auto parallel to load the checkpoint:
            1. Call load_state_dict_and_rename: Parse the distributed information from the names of the checkpoint files, and evenly parse out the distributed
               information for each weight/optimizer state into self.global_sharded_tensor_infos(data structure:param_name -> [{tp_rank: 1, sharding_rank: 1}, shape, dtype, file_name]).
               Modify the names of the optimizer states in the form ofparameter+suffixand record them in self.cur_rank_loaded_state_dict(data structure:file_name -> renamed_state_dict).
            2. Construct the Metadata and state_dict based on the distributed information obtained in the previous step for the final load.
            3. Special logic adaptation: When sharding is enabled, the optimizer states are also split. In this step, the optimizer states need to be concatenated back according to the sharding dimension:
                * Construct the Metadata for concatenating the sharded states back based on the characteristics of sharding.
                * Construct a temporaryopt_state_dictand use the_load_state_dictinterface to obtain the state_dict with the sharded states concatenated back.
                * Reshape the optimizer states back to the shape of the weights.
        """
        self.load_state_dict_and_rename()
        logger.info("Complete the loading and renaming of state_dict.")
        if self.sharding_degree > 1 and self.sharding_stage1_v == 2 and not self.is_sharding_stage3:
            for state_name, shard_info in self.global_sharded_tensor_infos.items():
                shard_info.sort(key=lambda x: x[0]["sharding_rank"])

            state_dict_metadata = {}
            storage_metadata = {}
            # After obtaining the local_shape and sharding rank of each tensor, the global offset of each tensor can be calculated.
            for state_name, shard_info in self.global_sharded_tensor_infos.items():
                global_offset = [0] * self.tp_degree
                for item in shard_info:
                    tp_rank = item[0]["tp_rank"]
                    state_name_with_tp_rank = state_name + "_tp" + "{:02d}".format(tp_rank)
                    local_tensor_meta_data = LocalTensorMetadata((global_offset[tp_rank],), item[1], item[2])
                    local_tensor_index = LocalTensorIndex(state_name_with_tp_rank, (global_offset[tp_rank],))
                    global_offset[tp_rank] += item[1][0]
                    if state_name_with_tp_rank not in state_dict_metadata:
                        state_dict_metadata[state_name_with_tp_rank] = [local_tensor_meta_data]
                    else:
                        state_dict_metadata[state_name_with_tp_rank].append(local_tensor_meta_data)
                    storage_metadata[local_tensor_index] = item[3]

            metadata_for_merge_sharding = Metadata(state_dict_metadata, storage_metadata, None)

            logger.debug(f"The metadata for merge sharding is: {metadata_for_merge_sharding}")

            source_state_dict_for_merge_sharding = {}
            for file_name, state_dict in self.cur_rank_loaded_state_dict.items():
                renamed_state_dict = {}
                (tp_rank, pp_rank, sharding_rank) = self.get_distribution_rank_from_file_name(file_name)
                for state_name, state_value in state_dict.items():
                    state_name_with_tp_rank = state_name + "_tp" + "{:02d}".format(tp_rank)
                    renamed_state_dict[state_name_with_tp_rank] = state_value

                source_state_dict_for_merge_sharding[file_name] = renamed_state_dict

            assert self.model_meta is not None
            global_model_state_shapes = []
            sharding_metas_keys = []
            for i in range(self.tp_degree):
                for j in range(self.pp_degree):
                    sharding_metas_keys.append("tp{:02d}_pp{:02d}".format(i, j))
            for key in sharding_metas_keys:
                param_meta = self.model_meta["sharding_metas"][key]["param_meta"]
                for param_name, param_shape_and_dtype in param_meta.items():
                    global_model_state_shapes.append([param_name, param_shape_and_dtype[0]])

            # Distribute all model parameters evenly across each card for loading

            world_size = paddle.distributed.get_world_size()
            partition_mapping = self.partition_parameters(global_model_state_shapes, True, world_size)

            partition_model_state_keys = []
            for cur_rank, partition_model_state in partition_mapping.items():
                partition_model_state_keys.append([item[0] for item in partition_model_state])

            all_param_meta = {}
            for i in range(self.tp_degree):
                for j in range(self.pp_degree):
                    key = "tp{:02d}_pp{:02d}".format(i, j)
                    param_meta = self.model_meta["sharding_metas"][key]["param_meta"]
                    for param_name, param_shape_and_dtype in param_meta.items():
                        all_param_meta[param_name] = param_shape_and_dtype

            param_flattened_shapes = {}
            for param_name, param_shape_and_dtype in all_param_meta.items():
                param_flattened_shapes[param_name] = reduce(lambda x, y: x * y, param_shape_and_dtype[0])

            cur_rank_need_load_model_state_keys = partition_model_state_keys[self.cur_rank]
            # Generate the optimizer states corresponding to the model weights.
            logger.info("Requesting GPU memory space to concatenate tensors split by sharding1 v2.")
            optimizer_state_dict = {}
            with paddle.base.dygraph.guard(place=paddle.CPUPlace()):
                for key in cur_rank_need_load_model_state_keys:
                    for tp_rank in range(self.tp_degree):
                        tp_rank_suffix = "_tp{:02d}".format(tp_rank)
                        optimizer_state_dict[key + ".moment1" + tp_rank_suffix] = paddle.zeros(
                            (param_flattened_shapes[key],), "float32"
                        )
                        optimizer_state_dict[key + ".moment2" + tp_rank_suffix] = paddle.zeros(
                            (param_flattened_shapes[key],), "float32"
                        )
                        if self.optimizer_state_with_master_weights:
                            optimizer_state_dict[key + ".master_weight" + tp_rank_suffix] = paddle.zeros(
                                (param_flattened_shapes[key],), "float32"
                            )
                        # When handling tensor parallelism (TP), if some tensors are replicated, we initially assume that they are partitioned.
                        # Later, when these are compared with the global shape, we realize that they are replicated.

                        optimizer_state_dict[key + ".beta1_pow_acc" + tp_rank_suffix] = paddle.zeros((1,), "float32")
                        optimizer_state_dict[key + ".beta2_pow_acc" + tp_rank_suffix] = paddle.zeros((1,), "float32")

            malloc_size = 0
            for opt_state_name, opt_state_value in optimizer_state_dict.items():
                malloc_size += opt_state_value.numel().numpy() * opt_state_value.element_size()
            malloc_size = malloc_size / 2**20
            logger.debug(f"{malloc_size} MB of GPU memory were allocated.")

            # merge sharding
            logger.info("First call _load_state_dict to stitch back the tensors split by sharding1 v2.")
            _load_state_dict(
                optimizer_state_dict, source_state_dict_for_merge_sharding, [metadata_for_merge_sharding], offload=True
            )
            logger.info("Completed the call _load_state_dict, concating back the tensors split by sharding.")

            # Reshape
            for opt_state_name, opt_state_value in optimizer_state_dict.items():
                if opt_state_value.shape[0] > 1 and "_tp" in opt_state_name:
                    param_name = self.optimizer_key_to_model_state_key(opt_state_name[:-5])
                    param_shape = all_param_meta[param_name][0]
                    assert opt_state_value.numel() == reduce(lambda x, y: x * y, param_shape)
                    reshaped_opt_state_value = opt_state_value.reshape(param_shape)
                    optimizer_state_dict[opt_state_name] = reshaped_opt_state_value
            concat_optimier_state_dict = {}

            optimizer_state_key_to_tp_keys = {}
            for opt_state_name in optimizer_state_dict.keys():
                # Count how each key is split into keys ending with ‘_tpXX’.
                # optimizer_state_key_to_tp_keys ： {key:[key_tp00,key_tp01]}
                opt_state_name_removed_tp_rank = opt_state_name[:-5]
                if opt_state_name_removed_tp_rank not in optimizer_state_key_to_tp_keys:
                    optimizer_state_key_to_tp_keys[opt_state_name_removed_tp_rank] = [opt_state_name]
                else:
                    optimizer_state_key_to_tp_keys[opt_state_name_removed_tp_rank].append(opt_state_name)

            for opt_state_name_removed_tp_rank, opt_state_name in optimizer_state_key_to_tp_keys.items():
                opt_state_name.sort(key=lambda x: int(x[-2:]))

            for opt_state_name_removed_tp_rank, opt_state_name in optimizer_state_key_to_tp_keys.items():
                model_state_name = self.optimizer_key_to_model_state_key(opt_state_name_removed_tp_rank)
                local_shape = optimizer_state_dict[opt_state_name[0]].shape
                if (
                    ".beta1_pow_acc" not in opt_state_name_removed_tp_rank
                    and ".beta2_pow_acc" not in opt_state_name_removed_tp_rank
                ):
                    global_shape = self.model_state_global_shape[model_state_name]
                else:
                    global_shape = (1,)

                if len(local_shape) != 1:
                    assert len(local_shape) == len(global_shape)

                axis = -1
                for i in range(len(local_shape)):
                    if local_shape[i] != global_shape[i]:
                        axis = i
                        break

                is_replicated = axis == -1
                tp_tensors = []
                for opt_state_name_with_tp_rank in opt_state_name:
                    tp_tensors.append(optimizer_state_dict[opt_state_name_with_tp_rank])

                if not is_replicated:
                    # Derive the partition strategy based on the global_shape, then concatenate.
                    concat_optimier_state_dict[opt_state_name_removed_tp_rank] = paddle.concat(tp_tensors, axis=axis)
                else:
                    concat_optimier_state_dict[opt_state_name_removed_tp_rank] = tp_tensors[0]

            fake_file_name = "{:02d}".format(self.cur_rank) + ".distcp"
            local_tensor_meta_data = {}
            local_tensor_index = {}
            for k, v in concat_optimier_state_dict.items():
                # Generate metadata.
                local_shape = v.shape
                global_offset = tuple([0] * len(local_shape))
                dtype = str(v.dtype).split(".")[1]
                local_tensor_meta_data[k] = LocalTensorMetadata(global_offset, local_shape, dtype)
                local_tensor_index[k] = [LocalTensorIndex(k, global_offset), fake_file_name]

            global_local_tensor_meta_data = []
            global_local_tensor_index = []

            use_dist = True if paddle.distributed.get_world_size() > 1 else False

            if use_dist:
                paddle.distributed.all_gather_object(global_local_tensor_meta_data, local_tensor_meta_data)
                paddle.distributed.all_gather_object(global_local_tensor_index, local_tensor_index)
            else:
                global_local_tensor_meta_data = [local_tensor_meta_data]
                global_local_tensor_index = [local_tensor_index]

            state_dict_metadata = {}
            for tensor_meta_data in global_local_tensor_meta_data:
                for k, v in tensor_meta_data.items():
                    if k not in state_dict_metadata:
                        state_dict_metadata[k] = [v]
                    else:
                        state_dict_metadata[k].append(v)

            storage_metadata = {}
            for tensor_index in global_local_tensor_index:
                for k, v in tensor_index.items():
                    storage_metadata[v[0]] = v[1]

            meta_data = Metadata(state_dict_metadata, storage_metadata, None)
            source_state_dict = {fake_file_name: concat_optimier_state_dict}
            return meta_data, source_state_dict

        elif self.sharding_degree > 1 and self.sharding_stage1_v == 1 and not self.is_sharding_stage3:
            return self.gen_metadata_for_tp_sharded_tensor()
        else:
            if self.is_sharding_stage3:
                for state_name, shard_info in self.global_sharded_tensor_infos.items():
                    shard_info.sort(key=lambda x: x[0]["sharding_rank"])
                state_dict_metadata = {}
                storage_metadata = {}
                # After obtaining the local_shape and sharding rank of each tensor, the global offset of each tensor can be calculated.
                for state_name, shard_info in self.global_sharded_tensor_infos.items():
                    global_offset = 0
                    for item in shard_info:
                        if len(item[1]) == 1:
                            local_tensor_meta_data = LocalTensorMetadata((global_offset,), item[1], item[2])
                            local_tensor_index = LocalTensorIndex(state_name, (global_offset,))
                            global_offset += item[1][0]
                        else:
                            global_offset = tuple([0] * len(item[1]))
                            local_tensor_meta_data = LocalTensorMetadata(global_offset, item[1], item[2])
                            local_tensor_index = LocalTensorIndex(state_name, global_offset)
                        if state_name not in state_dict_metadata:
                            state_dict_metadata[state_name] = [local_tensor_meta_data]
                        else:
                            state_dict_metadata[state_name].append(local_tensor_meta_data)
                        storage_metadata[local_tensor_index] = item[3]

                metadata_for_merge_sharding = Metadata(state_dict_metadata, storage_metadata, None)
                model_state_shapes = []
                dtype = ""
                for file, state_dict in self.cur_rank_loaded_state_dict.items():
                    if file.endswith(MODEL_WEIGHT_SUFFIX):
                        for k, v in state_dict.items():
                            model_state_shapes.append([k, v.shape])
                            dtype = str(v.dtype).split(".")[1]

                dtypes = self.gather_global_object([dtype])
                for dtype_s in dtypes:
                    if len(dtype_s) > 0:
                        dtype = dtype_s

                assert len(dtype) > 0

                global_model_state_shapes = self.gather_global_object(model_state_shapes)

                partition_result = self.partition_parameters(
                    global_model_state_shapes, True, paddle.distributed.get_world_size()
                )

                cur_rank_merger_model_params = partition_result[self.cur_rank]
                target_state_dict = {}
                for item in cur_rank_merger_model_params:
                    key = item[0]
                    shape = item[1]
                    flatten_shape = reduce(lambda a, b: a * b, item[1])
                    target_state_dict[key] = paddle.zeros(shape, dtype)
                    target_state_dict[key + ".moment1"] = paddle.zeros((flatten_shape,), "float32")
                    target_state_dict[key + ".moment2"] = paddle.zeros((flatten_shape,), "float32")
                    if self.optimizer_state_with_master_weights:
                        target_state_dict[key + ".master_weight"] = paddle.zeros((flatten_shape,), "float32")
                    # When handling tensor parallelism (TP), if some tensors are replicated, we initially assume that they are partitioned.
                    # Later, when these are compared with the global shape, we realize that they are replicated.

                    target_state_dict[key + ".beta1_pow_acc"] = paddle.zeros((1,), "float32")
                    target_state_dict[key + ".beta2_pow_acc"] = paddle.zeros((1,), "float32")

                _load_state_dict(
                    target_state_dict, self.cur_rank_loaded_state_dict, [metadata_for_merge_sharding], offload=True
                )

                # Reshape
                for item in cur_rank_merger_model_params:
                    key = item[0]
                    shape = item[1]
                    for k, v in target_state_dict.items():
                        if key == self.optimizer_key_to_model_state_key(k):
                            if tuple(shape) != tuple(v.shape) and v.numel() == reduce(lambda x, y: x * y, shape):
                                reshaped_v = v.reshape(shape)
                                target_state_dict[k] = reshaped_v

                fake_file_name = "{:02d}".format(self.cur_rank) + ".distcp"
                local_tensor_meta_data = {}
                local_tensor_index = {}
                for k, v in target_state_dict.items():
                    # Generate metadata.
                    local_shape = v.shape
                    global_offset = tuple([0] * len(local_shape))
                    dtype = str(v.dtype).split(".")[1]
                    local_tensor_meta_data[k] = LocalTensorMetadata(global_offset, local_shape, dtype)
                    local_tensor_index[k] = [LocalTensorIndex(k, global_offset), fake_file_name]

                global_local_tensor_meta_data = []
                global_local_tensor_index = []

                use_dist = True if paddle.distributed.get_world_size() > 1 else False

                if use_dist:
                    paddle.distributed.all_gather_object(global_local_tensor_meta_data, local_tensor_meta_data)
                    paddle.distributed.all_gather_object(global_local_tensor_index, local_tensor_index)
                else:
                    global_local_tensor_meta_data = [local_tensor_meta_data]
                    global_local_tensor_index = [local_tensor_index]

                state_dict_metadata = {}
                for tensor_meta_data in global_local_tensor_meta_data:
                    for k, v in tensor_meta_data.items():
                        if k not in state_dict_metadata:
                            state_dict_metadata[k] = [v]
                        else:
                            state_dict_metadata[k].append(v)

                storage_metadata = {}
                for tensor_index in global_local_tensor_index:
                    for k, v in tensor_index.items():
                        storage_metadata[v[0]] = v[1]

                meta_data = Metadata(state_dict_metadata, storage_metadata, None)
                source_state_dict = {fake_file_name: target_state_dict}

                return meta_data, source_state_dict
            else:
                return self.gen_metadata_for_tp_sharded_tensor()

    def rename_local_view_state_dict(self, state_dict, file_name):
        """
        Rename the key for local views to the key for global views, and return the renamed `state_dict`.
        """
        if self.local_view_pattern_list is None:
            return state_dict
        # case 1: moe_group is mp_group
        if self.tp_degree > 1 and self.sharding_degree <= 1:
            (tp_rank, pp_rank, sharding_rank) = self.get_distribution_rank_from_file_name(file_name)
            expert_name_old2new = {}
            for pattern in self.local_view_pattern_list:
                expert_pattern = rf"({pattern}\.)(\d+)"
                # extract all experts IDs
                expert_ids = set()
                for state_name in state_dict.keys():
                    res = re.search(expert_pattern, state_name)
                    if res:
                        expert_ids.add(int(res.group(2)))
                expert_num = len(expert_ids)
                # construct old name to new name mapping
                for state_name in state_dict.keys():
                    res = re.search(expert_pattern, state_name)
                    if res:
                        new_expert_id = int(res.group(2)) % expert_num + tp_rank * expert_num
                        expert_name_old2new[state_name] = re.sub(
                            expert_pattern, f"{res.group(1)}{new_expert_id}", state_name
                        )
            # rename state_dict
            renamed_state_dict = {
                expert_name_old2new[state_name]
                if state_name in expert_name_old2new
                else state_name: state_dict[state_name]
                for state_name in state_dict.keys()
            }

            return renamed_state_dict
        # TODO: add support for sharding
        else:
            return state_dict

    def load_state_dict_and_rename(self):
        """
        Parse the distributed information from the names of the checkpoint files and evenly parse out the distributed information for each weight/optimizer state
        into self.global_sharded_tensor_infos (data structure: param_name -> [{tp_rank: 1, sharding_rank: 1}, shape, dtype, file_name]). Modify the names of the
        optimizer states in the form of parameter+suffix and record them in self.cur_rank_loaded_state_dict (data structure: file_name -> renamed_state_dict).
            1. Load balancing: Each rank parses a portion of the checkpoint files.
            2. Flatten master_weights in opt_state into opt_state.
            3. Rename the keys in opt_state according to the rule: adamw_optimizer_param_suffix_name_mapping.
            4. Optimizer state renaming and distributed information extraction:
                * If it is sharding_stage1/2_v2 version:
                    * Renaming: rename_using_model_meta: In this case, a model_meta file is required. According to this file,
                      obtain the name mapping of weights and optimizer parameters, so that the optimizer states of manual and static partitions can correspond.
                    * Distributed information extraction: Record the distributed information of parameters: name -> [{tp_rank, sharding_rank}, shape, dtype, file_name].
                * If it is sharding_stage1/2_v1 version:
                    * Renaming:
                        * If a model_meta file exists:
                            * rename_using_model_meta
                        * If a model_meta file does not exist:
                            * According to the characteristics of v1 partitioning, infer the mapping relationship between optimizer states and weights (partition_result): master_weight_name_to_model_weight_name_mapping.
                        * Distributed information extraction: Record the distributed information of parameters: name -> [{tp_rank}, shape, dtype, file_name] (parameters will not be sharded).
                * If it is sharding_stage3:
                    * Renaming:
                        * If a model_meta file exists:
                            * rename_using_model_meta
                        * If a model_meta file does not exist:
                            * Establish the mapping between weights and optimizer names according to the order of optimizer states and weights: rename_using_optimizer_state_order.
                        * Distributed information extraction: Record the distributed information of parameters: name -> [{tp_rank, sharding_rank}, shape, dtype, file_name].
        """
        rank_access_files = {}
        if self.is_model_state_stored:
            rank_access_files[self.cur_rank] = (
                self.cur_rank_model_state_file_names + self.cur_rank_optimizer_state_file_names
            )
        else:
            rank_access_files[self.cur_rank] = self.cur_rank_optimizer_state_file_names

        global_rank_access_files = self.gather_global_object(rank_access_files)
        logger.info(f"The file(s) to be loaded for the global rank are: {global_rank_access_files}")
        need_read_files = get_rank_to_read_files(global_rank_access_files, global_rank_access_files)
        logger.info(f"The file(s) to be loaded for the current rank are: {need_read_files}")
        self.cur_rank_loaded_state_dict = {}

        for file in need_read_files:
            self.cur_rank_loaded_state_dict[file] = paddle.load(os.path.join(self.path, file), return_numpy=True)

        self.optimizer_state_with_master_weights = False

        for file, state_dict in self.cur_rank_loaded_state_dict.items():
            if file.endswith(OPTIMIZER_WEIGHT_SUFFIX):
                state_dict.pop("LR_Scheduler")
                if "master_weights" in state_dict:
                    self.optimizer_state_with_master_weights = True
                    master_weights = state_dict.pop("master_weights")
                    for master_weight_name, master_weight_value in master_weights.items():
                        # In sharding stage3, ‘@slice’ will be added in front of the key for master_weight, which is removed here.
                        state_dict[master_weight_name.replace("slice@", "") + ".master_weight"] = master_weight_value

                self.cur_rank_loaded_state_dict[file] = state_dict

        memory_size = 0
        for file, state_dict in self.cur_rank_loaded_state_dict.items():
            for k, v in state_dict.items():
                memory_size += v.size * v.itemsize

        memory_size = memory_size / 2**20
        logger.debug(
            f"The current rank has finished loading the checkpoint file and has allocated {memory_size} MB of GPU memory."
        )

        # After the rank has finished loading the files it needs, it can infer sharding_stage1_v and is_sharding_stage3.
        self.sharding_stage1_v = self.infer_sharding_stage1_v()
        self.is_sharding_stage3 = self.infer_is_sharding_stage3()

        flags = [
            ["is sharding stage1/2", (not self.is_sharding_stage3) and self.sharding_degree > 1],
            ["sharding stage1/2 version", self.sharding_stage1_v],
            ["is sharding stage3", self.is_sharding_stage3],
            ["master_weight", self.optimizer_state_with_master_weights],
        ]
        self.print_checkpoint_file_info(flags)

        # In sharding stage3, the parameters need to be reordered based on whether they are sliced.
        # The threshold for determining whether to slice is segment_size, with a default value of 2**20.
        # However, sharding stage3 allows users to specify their own unsliced layers, which seems to be incompatible here.
        if self.is_sharding_stage3:
            logger.info("The currently loaded checkpoint file comes from sharding stage 3.")
            segment_size = 2**20
            for file, state_dict in self.cur_rank_loaded_state_dict.items():
                if file.endswith(MODEL_WEIGHT_SUFFIX):
                    sliced_prameters = []
                    unsliced_parameters = []
                    sorted_state_dict = {}
                    for k, v in state_dict.items():
                        if v.numel() > segment_size:
                            sliced_prameters.append(k)
                        else:
                            unsliced_parameters.append(k)
                    for k in sliced_prameters + unsliced_parameters:
                        sorted_state_dict[k] = state_dict.pop(k)
                    self.cur_rank_loaded_state_dict[file] = sorted_state_dict

        # rename and record sharded_tensor_info
        cur_rank_sharded_tensor_infos = {}

        # 1. Handling the sharding stage1 v2 scenario, where the save_sharded_model flag must be enabled, independent of master_weights.
        if self.sharding_degree > 1 and self.sharding_stage1_v == 2 and not self.is_sharding_stage3:
            logger.info("The currently loaded checkpoint file comes from sharding stage1 v2.")
            assert self.is_model_meta_exists
            for file, state_dict in self.cur_rank_loaded_state_dict.items():
                # The rule for renaming is to change the master_weights name in the optimizer state to the model weight name,
                # and then append the tp_degree.
                renamed_state_dict = self.rename_using_model_meta(file)
                self.get_sharded_tensor_infos(file, renamed_state_dict, cur_rank_sharded_tensor_infos)
                self.cur_rank_loaded_state_dict[file] = renamed_state_dict
        # 2. In handling the sharding stage1 v1 and stage2 scenario, the optimizer states are distributed across different ranks.
        # We need to obtain the name mapping by simulating the partitioning method, without concern for the presence of master_weights.
        elif self.sharding_degree > 1 and self.sharding_stage1_v == 1 and not self.is_sharding_stage3:
            logger.info("The currently loaded checkpoint file comes from sharding stage1/2 v1.")
            if not self.is_model_meta_exists:
                file_to_state_dict_shapes_mapping = {}
                for file, state_dict in self.cur_rank_loaded_state_dict.items():
                    shapes = []
                    for state_name, state_value in state_dict.items():
                        shapes.append([state_name, state_value.shape])
                    file_to_state_dict_shapes_mapping[file] = shapes

                global_file_to_state_dict_shapes_mapping = self.gather_global_object(file_to_state_dict_shapes_mapping)

                for file, state_dict in self.cur_rank_loaded_state_dict.items():
                    (tp_rank, pp_rank, sharding_rank) = self.get_distribution_rank_from_file_name(file)
                    sharding_optimizer_state_shards = []
                    if file.endswith(OPTIMIZER_WEIGHT_SUFFIX):
                        for k, v in global_file_to_state_dict_shapes_mapping.items():
                            (tp_rank_, pp_rank_, sharding_rank_) = self.get_distribution_rank_from_file_name(k)
                            if tp_rank == tp_rank_ and pp_rank == pp_rank_ and k.endswith(OPTIMIZER_WEIGHT_SUFFIX):
                                sharding_optimizer_state_shards.append([v, sharding_rank_])
                        model_state_file_name = self.get_model_state_file_from(file)
                        model_state_shapes = global_file_to_state_dict_shapes_mapping[model_state_file_name]
                        sharding_optimizer_state_shards.sort(key=lambda x: x[1])

                        partition_result_0 = self.partition_parameters(model_state_shapes, False, self.sharding_degree)
                        partition_result_1 = self.partition_parameters(model_state_shapes, True, self.sharding_degree)

                        for rank, portion in partition_result_0.items():
                            portion = sorted(portion, key=model_state_shapes.index)
                            partition_result_0[rank] = portion

                        for rank, portion in partition_result_1.items():
                            portion = sorted(portion, key=model_state_shapes.index)
                            partition_result_1[rank] = portion

                        sharding_sort_parameters = False

                        for i in range(len(sharding_optimizer_state_shards)):
                            if not sharding_sort_parameters:
                                state_shard = sharding_optimizer_state_shards[i][0]
                                partitioned_shard = partition_result_0[i]
                                for j in range(len(partitioned_shard)):
                                    if partitioned_shard[j][1] != state_shard[j][1]:
                                        sharding_sort_parameters = True
                                        break

                        if sharding_sort_parameters:
                            for i in range(len(sharding_optimizer_state_shards)):
                                state_shard = sharding_optimizer_state_shards[i][0]
                                partitioned_shard = partition_result_1[i]
                                for j in range(len(partitioned_shard)):
                                    assert partitioned_shard[j][1] == state_shard[j][1]

                        if sharding_sort_parameters:
                            partition_result = partition_result_1
                        else:
                            partition_result = partition_result_0

                        name_mapping = {}
                        for i in range(len(sharding_optimizer_state_shards)):
                            state_shard = sharding_optimizer_state_shards[i][0]
                            partitioned_shard = partition_result[i]
                            suffix_bucket = {}
                            for suffix in OPTIMIZER_STATE_NAME_SUFFIX:
                                suffix_bucket[suffix] = []
                            for j in range(len(state_shard)):
                                optimizer_state_name = state_shard[j][0]
                                if "moment1" in optimizer_state_name:
                                    suffix_bucket[".moment1"].append(optimizer_state_name)
                                elif "moment2" in optimizer_state_name:
                                    suffix_bucket[".moment2"].append(optimizer_state_name)
                                elif "beta1_pow_acc" in optimizer_state_name:
                                    suffix_bucket[".beta1_pow_acc"].append(optimizer_state_name)
                                elif "beta2_pow_acc" in optimizer_state_name:
                                    suffix_bucket[".beta2_pow_acc"].append(optimizer_state_name)
                                else:
                                    suffix_bucket[".master_weight"].append(optimizer_state_name)

                            # In this scenario, the order of master_weights might differ from the order of the regular optimizer states and needs to be reordered.
                            if len(suffix_bucket[".master_weight"]) != 0:
                                master_weight_keys = []
                                for master_weight_key in suffix_bucket[".master_weight"]:
                                    for index in range(len(state_shard)):
                                        if master_weight_key[: -len(".master_weight")] in state_shard[index][0]:
                                            # Find the first match
                                            master_weight_keys.append([master_weight_key, index])
                                            break

                                master_weight_keys = sorted(master_weight_keys, key=lambda x: x[1])
                                suffix_bucket[".master_weight"] = [x[0] for x in master_weight_keys]

                            for suffix, old_names in suffix_bucket.items():
                                assert len(old_names) == len(partitioned_shard)
                                for k in range(len(old_names)):
                                    name_mapping[old_names[k]] = partitioned_shard[k][0] + suffix

                        renamed_state_dict = {}
                        # In this branch, sharding does not split the optimizer states; it merely relocates them to different cards.
                        # Therefore, the sharding information can now be directly removed.
                        for opt_state_name, opt_state_value in state_dict.items():
                            renamed_state_dict[name_mapping[opt_state_name]] = opt_state_value

                        self.get_sharded_tensor_infos(file, renamed_state_dict, cur_rank_sharded_tensor_infos)

                        self.cur_rank_loaded_state_dict[file] = renamed_state_dict
                    else:
                        self.get_sharded_tensor_infos(file, state_dict, cur_rank_sharded_tensor_infos)
            else:
                for file, state_dict in self.cur_rank_loaded_state_dict.items():
                    renamed_state_dict = self.rename_using_model_meta(file)
                    self.get_sharded_tensor_infos(file, renamed_state_dict, cur_rank_sharded_tensor_infos)

                    self.cur_rank_loaded_state_dict[file] = renamed_state_dict
        else:
            # 3. Handling the sharding stage3 and non-sharding scenario

            file_to_state_dict_keys_mapping = {}
            for file_name, state_dict in self.cur_rank_loaded_state_dict.items():
                file_to_state_dict_keys_mapping[file_name] = list(state_dict.keys())
            global_file_to_state_dict_keys_mapping = self.gather_global_object(file_to_state_dict_keys_mapping)

            logger.info("The current checkpoint comes from either sharding stage 3 or non-sharding.")
            if not self.is_model_meta_exists:
                for file_name, state_dict in self.cur_rank_loaded_state_dict.items():
                    if file_name.endswith(OPTIMIZER_WEIGHT_SUFFIX):
                        model_state_file_name = self.get_model_state_file_from(file_name)
                        assert model_state_file_name is not None
                        model_state_keys = global_file_to_state_dict_keys_mapping[model_state_file_name]
                        state_dict = self.rename_using_optimizer_state_order(model_state_keys, state_dict)
                    renamed_state_dict = self.rename_local_view_state_dict(state_dict, file_name)
                    self.get_sharded_tensor_infos(file_name, renamed_state_dict, cur_rank_sharded_tensor_infos)
                    self.cur_rank_loaded_state_dict[file_name] = renamed_state_dict
            else:
                for file, state_dict in self.cur_rank_loaded_state_dict.items():
                    # The rule for renaming is to change the master_weights name in the optimizer state to the model weight name,
                    # and then append the tp_degree.
                    renamed_state_dict = self.rename_using_model_meta(file)
                    self.get_sharded_tensor_infos(file, renamed_state_dict, cur_rank_sharded_tensor_infos)
                    self.cur_rank_loaded_state_dict[file] = renamed_state_dict

        # gather global sharded tensor infos
        sharded_tensor_infos = self.gather_global_object({self.cur_rank: cur_rank_sharded_tensor_infos})
        self.global_sharded_tensor_infos = {}
        for rank, sharded_tensor_info in sharded_tensor_infos.items():
            for state_name, shard_info in sharded_tensor_info.items():
                if state_name not in self.global_sharded_tensor_infos:
                    self.global_sharded_tensor_infos[state_name] = shard_info
                else:
                    self.global_sharded_tensor_infos[state_name] += shard_info
        logger.debug(f"global_sharded_tensor_infos: {self.global_sharded_tensor_infos}")

    def get_sharded_tensor_infos(self, file, state_dict, cur_rank_sharded_tensor_infos):
        (tp_rank, pp_rank, sharding_rank) = self.get_distribution_rank_from_file_name(file)
        for state_name, state_value in state_dict.items():
            if state_name not in cur_rank_sharded_tensor_infos:
                cur_rank_sharded_tensor_infos[state_name] = [
                    [
                        {"tp_rank": tp_rank, "sharding_rank": sharding_rank},
                        state_value.shape,
                        str(state_value.dtype),
                        file,
                    ]
                ]
            else:
                cur_rank_sharded_tensor_infos[state_name].append(
                    [
                        {"tp_rank": tp_rank, "sharding_rank": sharding_rank},
                        state_value.shape,
                        str(state_value.dtype),
                        file,
                    ]
                )

    def gen_metadata_for_tp_sharded_tensor(self):
        """
        Based on the distributed information of each weight/optimizer state (global_sharded_tensor_infos), construct Metadata
        information: LocalTensorMetadata,LocalTensorIndex
        """
        for state_name, shard_info in self.global_sharded_tensor_infos.items():
            shard_info.sort(key=lambda x: x[0]["tp_rank"])

        state_dict_metadata = {}
        storage_metadata = {}

        # After obtaining the local_shape and sharding rank of each tensor, the global offset of each tensor can be calculated.
        for state_name, shard_info in self.global_sharded_tensor_infos.items():

            global_offset = 0
            local_shape = shard_info[0][1]

            model_state_name = self.optimizer_key_to_model_state_key(state_name)
            if ".beta1_pow_acc" not in state_name and ".beta2_pow_acc" not in state_name:
                global_shape = self.model_state_global_shape[model_state_name]
            else:
                global_shape = (1,)
            assert len(local_shape) == len(global_shape)
            axis = -1
            for i in range(len(local_shape)):
                if local_shape[i] != global_shape[i]:
                    axis = i
                    break

            is_replicated = axis == -1
            global_offset = [0] * len(local_shape)

            if is_replicated:
                shard_info = [shard_info[0]]

            for item in shard_info:
                local_tensor_meta_data = LocalTensorMetadata(tuple(global_offset), item[1], item[2])
                local_tensor_index = LocalTensorIndex(state_name, tuple(global_offset))
                global_offset[axis] += item[1][axis]
                if state_name not in state_dict_metadata:
                    state_dict_metadata[state_name] = [local_tensor_meta_data]
                else:
                    state_dict_metadata[state_name].append(local_tensor_meta_data)
                storage_metadata[local_tensor_index] = item[3]

            metadata = Metadata(state_dict_metadata, storage_metadata, None)
            source_state_dict = self.cur_rank_loaded_state_dict

        return metadata, source_state_dict

    def rename_using_model_meta(self, file_name):
        """
        Rename the keys in opt_state_dict based on the following rule: model_meta records a mapping of parameter names to optimizer names.
        Here, we unify the optimizer state names to parameter names directly. For example:
            * model_meta: linear0 -> param0
            * opt_state: param0.w0
            * Renamed opt_state: linear0.w0
        NOTE：The reason for renaming is that there is a difference in the naming of optimizer parameters between dynamic and static partitions,
        making it difficult to match optimizer parameters directly by name. Therefore, we unify them to the weight names.
        """
        if not hasattr(self, "model_meta"):
            meta_file_path = os.path.join(self.path, MODEL_META_FILE_NAME)
            assert os.path.exists(meta_file_path)
            with open(meta_file_path, "r") as file:
                self.model_meta = json.load(file)

        (tp_rank, pp_rank, sharding_rank) = self.get_distribution_rank_from_file_name(file_name)
        dist_strategy_key = "tp" + "{:02d}".format(tp_rank) + "_" + "pp" + "{:02d}".format(pp_rank)
        # Map model weight names to their corresponding names of master_weights in the optimizer state.
        if file_name.endswith(OPTIMIZER_WEIGHT_SUFFIX):
            structure_name_mapping = self.model_meta["sharding_metas"][dist_strategy_key]["structure_name_mapping"]
            parameter_to_structured_name = {}
            for k, v in structure_name_mapping.items():
                parameter_to_structured_name[v] = k
            state_dict = self.cur_rank_loaded_state_dict[file_name]
            return self.rename_using_parameter_to_structured_name_mapping(state_dict, parameter_to_structured_name)
        else:
            return self.cur_rank_loaded_state_dict[file_name]

    def rename_auto_parallel_state_dict(self):
        """
        Rename the keys of the auto parallel state_dict according to certain rules:
            1. Rename the suffixes of the optimizer states to a unified format: adamw_optimizer_status_name_suffix_mappings
        """
        self.auto_parallel_state_dict = self.rename_using_parameter_to_structured_name_mapping(
            self.auto_parallel_state_dict, self.parameter_to_structured_name
        )

    def rename_using_parameter_to_structured_name_mapping(self, state_dict, parameter_to_structured_name):
        renamed_state_dict = {}

        def rename(old_name, parameter_to_structured_name):
            for i in range(1, len(old_name) + 1):
                param_name = old_name[:i]  # param_name
                suffix = old_name[i:]  # suffix
                if param_name in parameter_to_structured_name:
                    structure_name = parameter_to_structured_name[param_name]
                    if "moment1" in suffix:
                        return structure_name + ".moment1"
                    elif "moment2" in suffix:
                        return structure_name + ".moment2"
                    elif "beta1_pow_acc" in suffix:
                        return structure_name + ".beta1_pow_acc"
                    elif "beta2_pow_acc" in suffix:
                        return structure_name + ".beta2_pow_acc"
                    else:
                        return structure_name + ".master_weight"
            return None

        for key, value in state_dict.items():
            # NOTE: Skip the parameters that are not initialized，which are not in the current rank.
            if value is None or (isinstance(value, paddle.Tensor) and not value._is_initialized()):
                continue
            if key in parameter_to_structured_name.values():
                new_name = key
            else:
                new_name = rename(key, parameter_to_structured_name)
            assert new_name is not None
            renamed_state_dict[new_name] = value

        return renamed_state_dict

    def rename_using_optimizer_state_order(self, model_state_keys, optimizer_state_dict):
        name_mapping = {}
        suffix_bucket = {}
        # TODO: After adapting to sharding, remove the code below.
        if self.is_sharding_stage3 or (self.sharding_degree > 1 and self.sharding_stage1_v == 2):
            assert len(optimizer_state_dict) % len(model_state_keys) == 0
        for suffix in OPTIMIZER_STATE_NAME_SUFFIX:
            suffix_bucket[suffix] = []
        for opt_name, opt_value in optimizer_state_dict.items():
            if "moment1" in opt_name:
                suffix_bucket[".moment1"].append(opt_name)
            elif "moment2" in opt_name:
                suffix_bucket[".moment2"].append(opt_name)
            elif "beta1_pow_acc" in opt_name:
                suffix_bucket[".beta1_pow_acc"].append(opt_name)
            elif "beta2_pow_acc" in opt_name:
                suffix_bucket[".beta2_pow_acc"].append(opt_name)
            else:
                suffix_bucket[".master_weight"].append(opt_name)

        for suffix, old_names in suffix_bucket.items():
            if len(old_names) == 0:
                continue
            # TODO: After adapting to sharding, remove the code below.
            if self.is_sharding_stage3 or (self.sharding_degree > 1 and self.sharding_stage1_v == 2):
                assert len(old_names) == len(model_state_keys)

            # NOTE: Handle the case where the number of master_weight elements is not equal to the number of model_state_keys.
            if suffix != ".master_weight":
                for i in range(len(old_names)):
                    name_mapping[old_names[i]] = model_state_keys[i] + suffix
            else:
                for i in range(len(old_names)):
                    param = old_names[i][:-14]
                    index = -1
                    for idx, opt_name in enumerate(suffix_bucket[".moment1"]):
                        if param == opt_name[:-24]:
                            index = idx
                            break
                    if index >= 0:
                        name_mapping[old_names[i]] = model_state_keys[index] + suffix
                    else:
                        raise RuntimeError(f"Can't find {param} in optimizer state dict.")
        # rename state dict
        renamed_state_dict = {}
        for k, v in optimizer_state_dict.items():
            renamed_state_dict[name_mapping[k]] = v
        return renamed_state_dict

    def partition_parameters(self, model_state_shapes, is_sort, shard_num):
        """
        In sharding_stage3 and sharding_stage1_v1, parameters and optimizer states will be assigned to different ranks. This function defines the allocation rules.
        For details, refer to: python/paddle/distributed/fleet/meta_optimizers/dygraph_optimizer/dygraph_sharding_optimizer.py.
        """
        mapping = {}
        for rank_ in range(shard_num):
            mapping[rank_] = []
        sizes = [0] * shard_num

        parameters = model_state_shapes.copy()

        if is_sort:
            parameters.sort(key=lambda p: reduce(lambda x, y: x * y, p[1]), reverse=True)

        for param in parameters:
            rank = sizes.index(min(sizes))
            mapping[rank].append(param)
            numel = reduce(lambda x, y: x * y, param[1], 1)
            assert numel > 0, f"param [{param[0]}] should larger than 0, but it is [{numel}]"
            sizes[rank] += numel

        return mapping

    def get_is_model_meta_exists_flag(self):
        save_sharded_model_flag = self.gather_global_object(
            [os.path.exists(os.path.join(self.path, MODEL_META_FILE_NAME))]
        )
        return True in save_sharded_model_flag

    def get_is_model_state_stored_flag(self):
        if len(self.global_model_state_file_names) == 0:
            return False
        model_state_file_name = self.global_model_state_file_names[0]
        file_readable = model_state_file_name in self.cur_rank_model_state_file_names
        file_readables = self.gather_global_object([file_readable])
        coordinator_rank = file_readables.index(True)
        is_model_state_stored = False
        if self.cur_rank == coordinator_rank:
            model_state_file_size = os.path.getsize(os.path.join(self.path, model_state_file_name))
            if model_state_file_size > MODEL_STATE_FILE_MIN_SIZE:
                is_model_state_stored = True

        is_model_state_stored_flags = self.gather_global_object([is_model_state_stored])
        return True in is_model_state_stored_flags

    def flatten_state_dict(self, state_dict):
        flattened_state_dict = {}
        flat_state_dict, mapping = flatten_state_dict(state_dict)
        for k, v in flat_state_dict.items():
            last_level_key = mapping[k][-1]
            assert last_level_key not in flattened_state_dict
            flattened_state_dict[last_level_key] = v
        return flattened_state_dict

    def gather_global_object(self, cur_rank_object):
        all_rank_objects = []
        if self.use_dist:
            paddle.distributed.all_gather_object(all_rank_objects, cur_rank_object)
        else:
            all_rank_objects = [all_rank_objects]

        if isinstance(cur_rank_object, list):
            for obj in all_rank_objects:
                assert isinstance(obj, list)
            return [item for sublist in all_rank_objects for item in sublist]
        elif isinstance(cur_rank_object, dict):
            for obj in all_rank_objects:
                assert isinstance(obj, dict)
            global_map = {}
            for rank_map in all_rank_objects:
                global_map.update(rank_map)
            return global_map
        else:
            raise ValueError("cur_rank_object should be either a list or a dict")

    def get_local_checkpoint_file_names(self):
        cur_rank_files = os.listdir(self.path)
        cur_rank_model_state_file_names = []
        cur_rank_optimizer_state_file_names = []
        for file_name in cur_rank_files:
            if file_name.endswith(MODEL_WEIGHT_SUFFIX):
                cur_rank_model_state_file_names.append(file_name)
            elif file_name.endswith(OPTIMIZER_WEIGHT_SUFFIX):
                cur_rank_optimizer_state_file_names.append(file_name)
        if SCHEDULER_NAME in cur_rank_model_state_file_names:
            cur_rank_model_state_file_names.remove(SCHEDULER_NAME)
        if SCALAR_NAME in cur_rank_model_state_file_names:
            cur_rank_model_state_file_names.remove(SCALAR_NAME)

        return cur_rank_model_state_file_names, cur_rank_optimizer_state_file_names

    def get_distribution_rank_from_file_name(self, file_name):
        pp_degree = 0
        tp_degree = 0
        sharding_degree = 0
        pattern_pp = r"pp(\d+)"
        pattern_tp = r"tp(\d+)"
        pattern_shard = r"shard(\d+)"
        match_pp = re.search(pattern_pp, file_name)
        if match_pp:
            pp_degree = int(match_pp.group(1))
        match_tp = re.search(pattern_tp, file_name)
        if match_tp:
            tp_degree = int(match_tp.group(1))
        match_shard = re.search(pattern_shard, file_name)
        if match_shard:
            sharding_degree = int(match_shard.group(1))
        return (tp_degree, pp_degree, sharding_degree)

    def initial_distributed_configuration(self):
        self.pp_degree = 0
        self.tp_degree = 0
        self.sharding_degree = 0

        all_files = self.global_model_state_file_names + self.global_optimizer_state_file_names

        for file in all_files:
            (tp_degree, pp_degree, sharding_degree) = self.get_distribution_rank_from_file_name(file)
            self.pp_degree = max(self.pp_degree, pp_degree)
            self.tp_degree = max(self.tp_degree, tp_degree)
            self.sharding_degree = max(self.sharding_degree, sharding_degree)

        self.pp_degree = self.pp_degree + 1
        self.tp_degree = self.tp_degree + 1
        self.sharding_degree = self.sharding_degree + 1

    def infer_sharding_stage1_v(self):
        sharding_stage1_v = [2]
        for file, state_dict in self.cur_rank_loaded_state_dict.items():
            if file.endswith(OPTIMIZER_WEIGHT_SUFFIX) and sharding_stage1_v[0] == 2:
                for k, v in state_dict.items():
                    # Under shardingv2, the optimizer state is first flattened and then split.
                    if len(v.shape) != 1:
                        sharding_stage1_v = [1]
                        break

        sharding_stage1_v = self.gather_global_object(sharding_stage1_v)
        if 1 in sharding_stage1_v:
            return 1
        return 2

    def infer_is_sharding_stage3(self):
        if self.sharding_degree == 1:
            return False
        if self.pp_degree > 1 or self.tp_degree > 1:
            # Currently, sharding stage 3 does not support concurrent use with tensor parallelism (TP) and pipeline parallelism (PP).
            return False

        is_sharding_stage3 = True

        file_to_state_shape_mapping = {}
        for file, state_dict in self.cur_rank_loaded_state_dict.items():
            if file.endswith(OPTIMIZER_WEIGHT_SUFFIX):
                state_shape_mapping = {}
                for k, v in state_dict.items():
                    state_shape_mapping[k] = v.shape
                    if len(v.shape) != 1:
                        return False
                file_to_state_shape_mapping[file] = state_shape_mapping
        global_file_to_state_shape_mapping = self.gather_global_object(file_to_state_shape_mapping)

        state_dict_std = global_file_to_state_shape_mapping[list(global_file_to_state_shape_mapping.keys())[0]]

        for file, state_dict in global_file_to_state_shape_mapping.items():
            if state_dict != state_dict_std:
                is_sharding_stage3 = False
                break
        return is_sharding_stage3

    def get_model_state_file_from(self, optimizer_state_file_name):
        (tp_rank, pp_rank, sharding_rank) = self.get_distribution_rank_from_file_name(optimizer_state_file_name)
        for model_state_file in self.global_model_state_file_names:
            distributed_rank = self.get_distribution_rank_from_file_name(model_state_file)
            if tp_rank == distributed_rank[0] and pp_rank == distributed_rank[1]:
                return model_state_file
        return None

    def optimizer_key_to_model_state_key(self, optimizer_key):
        model_state_key = optimizer_key
        for suffix in OPTIMIZER_STATE_NAME_SUFFIX:
            if model_state_key.endswith(suffix):
                # Remove the suffix from model_state_key
                model_state_key = model_state_key[: -len(suffix)]
                break
        return model_state_key

    def print_checkpoint_file_info(self, flags):
        processed_flags = [
            [str(item) if not isinstance(item, bool) else "True" if item else "False" for item in row] for row in flags
        ]

        logger.info("Checkpoint file info:")
        headers = ["Flag", "Value"]
        col_widths = [max(len(str(item)) for item in column) for column in zip(headers, *flags)]
        format_str = "| " + " | ".join(f"{{:<{width}}}" for width in col_widths) + " |"
        separator_line = "+-" + "-+-".join("-" * width for width in col_widths) + "-+"

        logger.info(separator_line)
        logger.info(format_str.format(*headers))
        logger.info(separator_line)
        for row in processed_flags:
            logger.info(format_str.format(*row))
        logger.info(separator_line)
