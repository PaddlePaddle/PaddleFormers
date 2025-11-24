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

import copy
import json
import os
from collections import OrderedDict

import numpy
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer import (
    DygraphShardingOptimizer,
)

try:
    from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer.dygraph_sharding_optimizer import (
        DygraphShardingOptimizerV2,
    )
except:
    DygraphShardingOptimizerV2 = None

from ...transformers.model_utils import (
    _add_variant,
    get_parameter_dtype,
    unwrap_optimizer,
)
from ...transformers.utils import paddleformers_load
from ...utils.env import MODEL_META_NAME, SHARDING_META_NAME
from ...utils.log import logger
from ...utils.tools import get_env_device
from . import reshard as reshard_util
from .reshard import (
    SHARDING_STRATEGY_V1,
    SHARDING_STRATEGY_V2,
    get_param_sharding_group,
    merge_model_state,
    merge_opt_state,
    pp_reshard,
    split_model_state,
    split_opt_state,
    split_structure_name_mapping,
)


def to_device(tensor, place=None):
    if place is None:
        place = get_env_device()

    if isinstance(place, str):
        place = paddle.device._convert_to_place(place)

    if not tensor.place._equals(place):
        new_t = tensor._copy_to(place, True)
        dst_tensor = tensor.value().get_tensor()
        src_tensor = new_t.value().get_tensor()
        dst_tensor._share_data_with(src_tensor)

    return tensor


def filter_sharded_params(state_dict, optimizer, sharding_group, include_freeze_params=False):

    sharding_rank = max(sharding_group.rank, 0)
    sharding_world_size = sharding_group.nranks
    from ...trainer.utils import reshard as reshard_util

    logger.info(f"filter sharded_params not placed in sharding_rank {sharding_rank} .")
    if not reshard_util.is_sharding_opt(optimizer):
        return state_dict

    filtered_state_dict = OrderedDict()
    if reshard_util.get_sharding_strategy(optimizer) == reshard_util.SHARDING_STRATEGY_V1:
        optimizer = unwrap_optimizer(optimizer, DygraphShardingOptimizer)
        for (k, v) in state_dict.items():
            if v.name in optimizer._param2rank:
                sharded_rank = optimizer._param2rank[v.name]
                if sharded_rank != sharding_rank:
                    continue
                filtered_state_dict[k] = v
            elif include_freeze_params:
                if sharding_rank == 0:
                    filtered_state_dict[k] = v
    else:
        optimizer = unwrap_optimizer(optimizer, DygraphShardingOptimizerV2)
        parameters = optimizer._parameter_list
        filtered_parameters = [p.name for (i, p) in enumerate(parameters) if i % sharding_world_size == sharding_rank]
        filtered_parameters = set(filtered_parameters)
        for (k, v) in state_dict.items():
            if v.name in filtered_parameters:
                filtered_state_dict[k] = v
            elif include_freeze_params and (v.name not in [p.name for p in parameters]):
                if sharding_rank == 0:
                    filtered_state_dict[k] = v
    return filtered_state_dict


def exclude_parameters_in_state_dict(
    model_state_dict, param_names_in_master_weights, sharding_group, should_save_sharding_stage1_model=True
):
    assert sharding_group is not None
    assert isinstance(model_state_dict, dict) and isinstance(
        param_names_in_master_weights, (list, set)
    ), "param_names_in_master_weights type:{}".format(type(param_names_in_master_weights))
    state_param_names = [v.name for k, v in model_state_dict.items()]
    logger.debug(
        "param_names_in_master_weights:{}, state_param_names:{}".format(
            param_names_in_master_weights, state_param_names
        )
    )
    # allgather parameter names in sharding group
    tmp = []
    if sharding_group.nranks > 1:
        paddle.distributed.all_gather_object(tmp, param_names_in_master_weights, group=sharding_group)
    else:
        tmp = [param_names_in_master_weights]
    param_names_in_master_weights = set([v for item in tmp for v in item])
    logger.info("sharding_group_param_names:{}".format(param_names_in_master_weights))
    non_parameters_state_dict = copy.copy(model_state_dict)
    for k, v in model_state_dict.items():
        if v.name in param_names_in_master_weights:
            non_parameters_state_dict.pop(k)

    return non_parameters_state_dict


class ParameterNameRemapper:
    def __init__(self, old_mapping, new_mapping, checkpoint):
        self.checkpoint = checkpoint
        self.p_name_map = {}
        for k, v in old_mapping.items():
            assert k in new_mapping, f"structure name not found: {k} {new_mapping.keys()}"
            new_v = new_mapping[k]
            if v not in self.p_name_map:
                self.p_name_map[v] = new_v
            else:
                old_v = self.p_name_map[v]
                assert old_v == new_v, f"structure name {k} has different parameter name {new_v} {self.p_name_map[v]}"
        self.old_p_names = list(self.p_name_map.keys())
        self.new_mapping = dict([(k, new_mapping[k]) for k in old_mapping.keys()])

    def _map_tensor(self, tensor, old_p_name=None):
        if old_p_name is not None:
            new_p_name = self.p_name_map.get(old_p_name)
            assert new_p_name is not None, f"parameter name {old_p_name} not found"
        else:
            new_p_name = None

        def _map_name(old_name):
            if new_p_name is not None:
                assert old_name.startswith(old_p_name)
                return new_p_name + old_name[len(old_p_name) :]
            else:
                new_name = self.p_name_map.get(old_name)
                assert new_name is not None, f"parameter name {old_name} not found"
                return new_name

        if isinstance(tensor, paddle.Tensor):
            new_name = _map_name(tensor.name)
            tensor.name = new_name
            return new_name, tensor
        else:
            assert isinstance(tensor, (list, tuple)), type(tensor)
            old_name, value = tensor
            new_name = self._map_name(old_name)
            return new_name, (new_name, value)

    def remap_model_state(self, model_state):
        for k, v in model_state.items():
            if not isinstance(v, numpy.ndarray):
                model_state[k] = self._map_tensor(v)[1]
        return model_state

    def remap_optimizer_state(self, opt_state):
        lr_scheduler_key = "LR_Scheduler"
        master_weight_key = "master_weights"

        new_opt_state = {}
        new_master_weights = None
        opt_names = []
        for k, v in opt_state.items():
            if k == lr_scheduler_key:
                new_opt_state[k] = v
            elif k == master_weight_key:
                for kk, vv in v.items():
                    new_kk = self.p_name_map[kk]
                    if new_master_weights is None:
                        new_opt_state[master_weight_key] = {}
                        new_master_weights = new_opt_state[master_weight_key]
                    new_master_weights[new_kk] = self._map_tensor(vv, kk)[1]
            else:
                assert isinstance(v, paddle.Tensor), type(v)
                opt_names.append(v.name)

        opt_to_pname = reshard_util.convert_opt_name_to_tname(self.old_p_names, opt_names)
        for opt_name in opt_names:
            v = opt_state[opt_name]
            new_opt_name, new_v = self._map_tensor(v, opt_to_pname[opt_name])
            new_opt_state[new_opt_name] = new_v

        opt_state.clear()
        opt_state.update(new_opt_state)
        return opt_state


class GroupGetter:
    def __init__(self, model, hcg=None):
        self.structure_name_mapping = {}
        self.structure_name_to_group = {}
        self.tensor_name_to_group = {}
        self.parameter_names = []
        self.group_map = OrderedDict()
        self.hcg = hcg or fleet.get_hybrid_communicate_group()
        for k, v in model.state_dict().items():
            self.structure_name_mapping[k] = v.name
            group = get_param_sharding_group(v, self.hcg)
            self.structure_name_to_group[k] = group
            self.tensor_name_to_group[v.name] = group
            self.group_map[group.id] = group
            self.parameter_names.append(v.name)

    def _get_parameter_name(self, name):
        if name in self.tensor_name_to_group:
            return name

        suffix = [
            "_fp32_master_0_beta1_pow_acc_0",
            "_fp32_master_0_beta2_pow_acc_0",
            "_fp32_master_0_moment1_0",
            "_fp32_master_0_moment2_0",
            "_beta1_pow_acc_0",
            "_beta2_pow_acc_0",
            "_moment1_0",
            "_moment2_0",
        ]

        for s in suffix:
            if name.endswith(s):
                tmp = name[: -len(s)]
                assert tmp in self.tensor_name_to_group, f"cannot find {name}"
                return tmp

        raise ValueError(f"cannot find {name}")

    def get_group(self, name):
        if name in self.structure_name_to_group:
            assert name not in self.tensor_name_to_group, name
            return self.structure_name_to_group[name]
        else:
            return self.tensor_name_to_group[self._get_parameter_name(name)]

    def get_group_by_id(self, gid):
        return self.group_map[gid]

    def get_group_ids(self):
        return list(self.group_map.keys())


class ShardingIO:
    def __init__(self, args, model, optimizer=None, hcg=None, remap_parameter_name=False, is_ema=False):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.hcg = hcg
        self.sharding_group = None
        if self.hcg is None and paddle.distributed.get_world_size() > 1 and self.args.use_hybrid_parallel:
            self.hcg = fleet.get_hybrid_communicate_group()
            self.sharding_group = self.hcg.get_sharding_parallel_group()

        self.remap_parameter_name = remap_parameter_name
        self.remapper = None
        self.is_ema = is_ema

    def _get_remapper(self, checkpoint):
        if not self.remap_parameter_name:
            return None

        if self.remapper is None or self.remapper.checkpoint != checkpoint:
            new_mapping = {}
            for k, v in self.model.state_dict().items():
                new_mapping[k] = v.name

            suffix = self._sharding_meta_suffix()
            model_meta = self._load_model_meta_impl(checkpoint)
            old_mapping = model_meta["sharding_metas"][suffix]["structure_name_mapping"]
            self.remapper = ParameterNameRemapper(old_mapping, new_mapping, checkpoint)
        return self.remapper

    def _remap_parameter_name(self, checkpoint, state_dict, is_opt):
        remapper = self._get_remapper(checkpoint)
        if remapper is None:
            return state_dict
        if is_opt:
            return remapper.remap_optimizer_state(state_dict)
        else:
            return remapper.remap_model_state(state_dict)

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def load_state_dict_from_checkpoint_with_reshard(
        self, checkpoint, base_weight_name, model_wrapped, opt_state_dict=None
    ):
        """load state_dict from_checkpoint with reshard, Only load model state dict.
        Args:
            checkpoint (str): The directory of the checkpoint.
            base_weight_name (str): The name of the checkpoint file.
            model_wrapped (nn.Layer): The wrapped model.
        """
        group_getter = GroupGetter(self.model)
        gids = group_getter.get_group_ids()
        parallel_config = self._load_distributed_strategy(checkpoint)
        pp_degree = parallel_config["pp_degree"]
        mp_degree = parallel_config["mp_degree"]
        sharding_degree = parallel_config["sharding_degree"]
        assert (
            self.args.tensor_parallel_degree == mp_degree
        ), f"mp_degree of the script {self.args.tensor_parallel_degree} and mp of the model {mp_degree} are not matched"
        cur_sharding_degree = self.args.sharding_parallel_degree
        cur_pp_degree = self.args.pipeline_parallel_degree
        if pp_degree > 1:
            assert cur_pp_degree > 1, "can not reshard from pp to non pp"
        if pp_degree <= 1:
            assert cur_pp_degree <= 1, "can not reshard from non pp to pp"

        def print_ckpt(state_dict):
            for k, v in state_dict["master_weights"].items():
                if not isinstance(v, paddle.Tensor):
                    v_t = paddle.to_tensor(v)
                else:
                    v_t = v
                print(k, v_t.name, v_t.shape, v_t._md5sum())

        def load_model_slices():
            model_state = {gid: reshard_util.NodeModelState(group=group_getter.get_group_by_id(gid)) for gid in gids}
            for j in range(self.args.pipeline_parallel_rank, pp_degree, cur_pp_degree):
                cur_sharding_meta = self._load_sharding_meta(checkpoint, j)
                assert "structure_name_mapping" in cur_sharding_meta
                structure_name_map = cur_sharding_meta["structure_name_mapping"]
                structure_name_map = split_structure_name_mapping(structure_name_map, group_getter)
                for i in range(self.args.sharding_parallel_rank, sharding_degree, cur_sharding_degree):
                    tmp = self._load_one_state_dict_from_checkpoint(
                        checkpoint,
                        base_weight_name,
                        self.args.sharded_name_suffix(i, j, sharding_parallel_degree=sharding_degree),
                    )
                    tmp = split_model_state(tmp, group_getter)
                    for gid in gids:
                        sub_tmp = tmp.get(gid, {})
                        node_model_state_tmp = reshard_util.NodeModelState(group=group_getter.get_group_by_id(gid))
                        node_model_state_tmp.add_weights(sub_tmp)
                        node_model_state_tmp.pack_keys(structure_name_map.get(gid, {}))
                        model_state[gid].merge_from(node_model_state_tmp, i)
            return model_state

        node_model_state = load_model_slices()

        if self._need_reshard_pp(checkpoint):
            meta = self._load_model_meta(checkpoint)
            reshard_context = pp_reshard.build_pipeline_context(meta, model_wrapped)
            node_model_state = pp_reshard.reshard(node_model_state, reshard_context, self.hcg)

        if opt_state_dict is None:
            opt_state_dict = self.optimizer.state_dict()
        opt_state_dict = split_opt_state(opt_state_dict, group_getter)

        res_state_dict = OrderedDict()
        for gid, nms in node_model_state.items():
            nms.drop_rank()
            nms.unpack_keys()
            state_dict = nms.model_weights

            def filter_func(name):
                return True

            state_dict = reshard_util.all_gather_state_dict(state_dict, filter_func, nms.group)

            if self.args.bf16:
                state_dict = self._recover_params_from_master_weights(
                    state_dict,
                    opt_state_dict=opt_state_dict.get(gid, {}),
                    group=group_getter.get_group_by_id(gid),
                )

            res_state_dict.update(state_dict)

        return res_state_dict

    def _load_one_state_dict_from_checkpoint(self, resume_from_checkpoint, base_weight_name, weight_name_suffix):
        """
        load state_dict of one shard from_checkpoint, Only load model state dict.
        """
        if self.is_ema:
            base_weight_name = base_weight_name.replace("model_state", "ema").replace("pdparams", "pdopt")
        file_path = os.path.join(resume_from_checkpoint, _add_variant(base_weight_name, weight_name_suffix))
        if not os.path.isfile(file_path):
            raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}, no {file_path}")

        logger.info(f"Loading model from {file_path} .")
        # We load the model state dict on the CPU to avoid an OOM error.
        state_dict = paddle.load(file_path, return_numpy=True)
        if self.is_ema:
            state_dict.pop("master_weights", None)
        state_dict = self._remap_parameter_name(resume_from_checkpoint, state_dict, is_opt=False)
        return state_dict

    def _load_optimizer_state_of_one_shard(self, checkpoint, base_opt_name, optimizer_name_suffix, group_getter=None):
        if self.is_ema:
            base_opt_name = base_opt_name.replace("optimizer", "ema")
        optimizer_name = _add_variant(base_opt_name, optimizer_name_suffix)
        path = os.path.join(checkpoint, optimizer_name)
        logger.info(f"load optimizer state from {path}")
        if os.path.isfile(path):
            opt_state = paddleformers_load(path, map_location="cpu")
            if self.is_ema:
                opt_state = {"master_weights": opt_state.get("master_weights", {})}
            return self._remap_parameter_name(
                checkpoint,
                self._modify_ckpt_for_compatibility(opt_state),
                is_opt=True,
            )
        logger.info(f"{path} not exists")
        return None

    def _modify_ckpt_for_compatibility(self, ckpt):
        master_weights = ckpt.get("master_weights", None)
        if master_weights:
            for k, v in master_weights.items():
                assert isinstance(v, paddle.Tensor), v
                if not v.name.startswith(k):
                    new_name = k + "_fp32_master_0"
                    logger.info(f"Modify master weights {v.name} -> {new_name}")
                    v.name = new_name
        return ckpt

    def _need_reshard(self, checkpoint):
        if self._need_reshard_pp(checkpoint):
            return True
        parallel_config = self._load_distributed_strategy(checkpoint)
        sharding_meta = self._load_sharding_meta(checkpoint)
        sharding_degree = parallel_config["sharding_degree"]
        sharding_strategy = SHARDING_STRATEGY_V1
        if "sharding_strategy" in sharding_meta:
            sharding_strategy = sharding_meta["sharding_strategy"]
        cur_sharding_degree = self.args.sharding_parallel_degree
        cur_sharding_strategy = reshard_util.get_sharding_strategy(self.optimizer)
        if sharding_degree != cur_sharding_degree or sharding_strategy != cur_sharding_strategy:
            return True
        if sharding_strategy == SHARDING_STRATEGY_V1:
            param2rank = sharding_meta["param2rank"]
            optimizer = unwrap_optimizer(self.optimizer, DygraphShardingOptimizer)
            if self.args.sharding_parallel_degree > 1:
                assert optimizer is not None
            else:
                assert optimizer is None
            if len(param2rank) == 0 or optimizer is None:
                logger.warning("The param2rank is empty or sharding degree is 1. Force reshard would be performed.")
                return True
            assert len(param2rank) == len(optimizer._param2rank)
            for (k, v) in param2rank.items():
                assert k in optimizer._param2rank
                if optimizer._param2rank[k] != int(v):
                    return True
        else:
            pp_overlap = None
            # backward compatibility
            if "enable_overlap" in sharding_meta:
                pp_overlap = sharding_meta["enable_overlap"]

            cur_pp_overlap = unwrap_optimizer(self.optimizer, DygraphShardingOptimizerV2).pp_overlap
            return pp_overlap != cur_pp_overlap

        return False

    def _need_reshard_pp(self, checkpoint):
        parallel_config = self._load_distributed_strategy(checkpoint)
        pp_degree = parallel_config["pp_degree"]
        cur_pp_degree = self.args.pipeline_parallel_degree
        if pp_degree != cur_pp_degree:
            return True
        # vpp、segment method changes is not auto supported yet
        return self.args.force_reshard_pp

    def load_optimizer_state_with_reshard(self, checkpoint, base_opt_name, model_wrapped):
        """load state_dict of multiple shard from_checkpoint, Only load model state dict."""

        parallel_config = self._load_distributed_strategy(checkpoint)
        sharding_meta = self._load_sharding_meta(checkpoint)
        pp_degree = parallel_config["pp_degree"]
        mp_degree = parallel_config["mp_degree"]
        sharding_degree = parallel_config["sharding_degree"]
        assert sharding_degree > 1, "sharding degree of the checkpoint should be larger than 1"
        sharding_strategy = SHARDING_STRATEGY_V1
        if "sharding_strategy" in sharding_meta:
            sharding_strategy = sharding_meta["sharding_strategy"]
        assert self.args.tensor_parallel_degree == mp_degree
        cur_pp_degree = self.args.pipeline_parallel_degree

        if pp_degree > 1:
            assert cur_pp_degree > 1, "can not reshard from pp to non pp"
        if pp_degree <= 1:
            assert cur_pp_degree <= 1, "can not reshard from non pp to pp"

        cur_sharding_degree = self.args.sharding_parallel_degree
        cur_sharding_strategy = reshard_util.get_sharding_strategy(self.optimizer)

        group_getter = GroupGetter(self.model)

        if not self._need_reshard(checkpoint):
            one_shard_opt_state_dict = self._load_optimizer_state_of_one_shard(
                checkpoint,
                base_opt_name,
                self.args.sharded_name_suffix(sharding_parallel_degree=sharding_degree),
                group_getter=group_getter,
            )

            if sharding_strategy == SHARDING_STRATEGY_V2 and cur_sharding_strategy == SHARDING_STRATEGY_V2:
                is_matched = reshard_util.sharding_v2.is_matched_optimizer_state_dict(
                    one_shard_opt_state_dict, self.optimizer, model_wrapped
                )
                is_matched = paddle.to_tensor([is_matched], dtype=paddle.int32)
                dp_group = fleet.get_hybrid_communicate_group().get_data_parallel_group()
                dp_src_rank = fleet.get_hybrid_communicate_group().get_data_parallel_group_src_rank()
                dist.broadcast(is_matched, src=dp_src_rank, group=dp_group)
                is_matched = bool(is_matched[0])
            else:
                is_matched = True

            if is_matched:
                logger.info("do not need reshard")
                return one_shard_opt_state_dict
        else:
            one_shard_opt_state_dict = None

        logger.info("reshard optimizer state")
        gids = group_getter.get_group_ids()

        def load_model_slices():
            model_state = {gid: reshard_util.NodeModelState(group=group_getter.get_group_by_id(gid)) for gid in gids}
            for j in range(self.args.pipeline_parallel_rank, pp_degree, cur_pp_degree):
                cur_sharding_meta = self._load_sharding_meta(checkpoint, j)
                assert "structure_name_mapping" in cur_sharding_meta
                structure_name_map = cur_sharding_meta["structure_name_mapping"]
                structure_name_map = split_structure_name_mapping(structure_name_map, group_getter)
                for i in range(self.args.sharding_parallel_rank, sharding_degree, cur_sharding_degree):
                    sharded_name_suffix = self.args.sharded_name_suffix(i, j, sharding_parallel_degree=sharding_degree)
                    if one_shard_opt_state_dict is None:
                        tmp = self._load_optimizer_state_of_one_shard(checkpoint, base_opt_name, sharded_name_suffix)
                    else:
                        assert (
                            self.args.optimizer_name_suffix == sharded_name_suffix
                        ), f"{self.args.optimizer_name_suffix} vs {sharded_name_suffix}"
                        tmp = one_shard_opt_state_dict
                    tmp = split_opt_state(tmp, group_getter)
                    for gid in gids:
                        sub_tmp = tmp.get(gid, {})
                        node_model_state_tmp = reshard_util.NodeModelState(group=group_getter.get_group_by_id(gid))
                        node_model_state_tmp.add_opts(sub_tmp)
                        node_model_state_tmp.pack_keys(structure_name_map.get(gid, {}))
                        model_state[gid].merge_from(node_model_state_tmp, i)
            return model_state

        def reshard_pp(model_state):
            # pp reshard
            if self._need_reshard_pp(checkpoint):
                assert len(model_state) == 1, "only support one group reshard"
                key = list(model_state.keys())[0]
                tmp = model_state[key]
                meta = self._load_model_meta(checkpoint)
                reshard_context = pp_reshard.build_pipeline_context(meta, model_wrapped)
                model_state = {key: pp_reshard.reshard(tmp, reshard_context, self.hcg)}
            return model_state

        def reshard_sharding(node_model_state):
            # shard reshard
            restore_func = (
                reshard_util.sharding_v1.restore
                if sharding_strategy == SHARDING_STRATEGY_V1
                else reshard_util.sharding_v2.restore
            )
            for gid in gids:
                node_model_state[gid] = restore_func(node_model_state[gid], self.model, self.optimizer)
            shard_func = (
                reshard_util.sharding_v1.shard
                if cur_sharding_strategy == SHARDING_STRATEGY_V1
                else reshard_util.sharding_v2.shard
            )
            ret_opt_state_dict = OrderedDict()
            for gid in gids:
                node_model_state[gid] = shard_func(node_model_state[gid], model_wrapped, self.optimizer)
                # drop structural name in the key
                node_model_state[gid].unpack_keys()
                ret_opt_state_dict[gid] = node_model_state[gid].get_opt_state_dict()
            return merge_opt_state(ret_opt_state_dict)

        node_model_state = load_model_slices()
        node_model_state = reshard_pp(node_model_state)
        return reshard_sharding(node_model_state)

    def manipulate_state_dict_and_config(self, model_to_save, merge_tensor_parallel=False, state_dict=None):
        weight_name_suffix = self.args.sharded_name_suffix()
        group_getter = GroupGetter(model_to_save)
        gids = group_getter.get_group_ids()
        if state_dict is None:
            state_dict = model_to_save.state_dict()
            if self.args.should_save_sharding_stage1_model:
                state_dict = split_model_state(state_dict, group_getter)
                for gid in gids:
                    state_dict[gid] = filter_sharded_params(
                        state_dict.get(gid, {}),
                        self.optimizer,
                        self.sharding_group,
                        self.args.save_sharding_stage1_model_include_freeze_params,
                    )
                state_dict = merge_model_state(state_dict)

        config_to_save = None
        merge_tensor_parallel = merge_tensor_parallel and self.args.use_hybrid_parallel
        if merge_tensor_parallel:
            dtype = get_parameter_dtype(model_to_save)
            assert hasattr(model_to_save, "config")
            model_to_save.config.dtype = str(dtype).split(".")[1]
            config_to_save = copy.deepcopy(model_to_save.config)
            if config_to_save.tensor_parallel_degree > 1:
                state_dict = model_to_save.merge_tensor_parallel(state_dict, config_to_save)
                config_to_save.tensor_parallel_degree = 1
                if config_to_save.tensor_parallel_rank != 0:
                    logger.info("Saving with merge_tensor_parallel, tensor_parallel_rank > 0 don't need save")
                    return
                # if variant is not None and "tp" in variant:
                if "tp" in weight_name_suffix:
                    weight_name_suffix = "_".join([x for x in weight_name_suffix.split("_") if "tp" not in x])

        if self.args.bf16 and self.args.should_save_sharding_stage1_model:
            param_names_in_master_weights = []
            optimzier_state_dict = self.optimizer.state_dict()
            optimzier_state_dict = split_opt_state(optimzier_state_dict, group_getter)
            state_dict = split_model_state(state_dict, group_getter)
            for gid in gids:
                sub_opt_state = optimzier_state_dict.get(gid, {})
                param_names_in_master_weights = list(sub_opt_state.get("master_weights", {}).keys())
                state_dict[gid] = exclude_parameters_in_state_dict(
                    state_dict.get(gid, {}),
                    param_names_in_master_weights,
                    group_getter.get_group_by_id(gid),
                )
            state_dict = merge_model_state(state_dict)
            logger.info(
                "param_names_in_master_weights len:{}, bf16 state_dict len:{}, :{}".format(
                    len(param_names_in_master_weights), len(state_dict), state_dict.keys()
                )
            )
        return state_dict, config_to_save, weight_name_suffix

    def gather_distributed_model_meta(self):
        if not self.args.use_hybrid_parallel:
            return None

        if not self.args.should_save_sharding_stage1_model:
            return None

        nranks = dist.get_world_size()
        if nranks <= 1:
            return None

        model_meta = {}
        model_meta["parallel_config"] = self._get_distributed_strategy()
        model_meta["sharding_metas"] = self._gather_sharding_metas()

        return model_meta

    def _check_distributed_strategy(self, parallel_config):
        ep_degree = parallel_config.get("ep_degree", 1)
        if ep_degree > 1:
            tp_degree = parallel_config["mp_degree"]
            sharding_degree = parallel_config["sharding_degree"]
            moe_sharding_degree = parallel_config.get("moe_sharding_degree", 1)
            assert tp_degree * sharding_degree == ep_degree * moe_sharding_degree, "mismatch parallel degree settings"

    def check_same_strategy(self, resume_from_checkpoint=None):
        if resume_from_checkpoint:
            cur_config = self._get_distributed_strategy()
            old_config = self._load_model_meta_impl(resume_from_checkpoint)["parallel_config"]
            keys = list(old_config.keys())
            for key in keys:
                if key not in cur_config:
                    return False, f"missing {key}"
                else:
                    old_value = old_config[key]
                    cur_value = cur_config[key]
                    if old_value != cur_value:
                        return False, f"{key} not match: {old_value} vs {cur_value}"
        return True, None

    def _get_distributed_strategy(self):
        pp_degree = 1
        mp_degree = 1
        sharding_degree = 1
        ep_degree = 1
        moe_sharding_degree = 1
        nranks = dist.get_world_size()
        if self.args.use_hybrid_parallel and nranks > 1:
            hcg = fleet.get_hybrid_communicate_group()
            mp_degree = hcg.get_model_parallel_world_size()
            pp_degree = hcg.get_pipe_parallel_world_size()
            sharding_degree = hcg.get_sharding_parallel_world_size()
            if hasattr(hcg, "get_expert_parallel_world_size"):
                ep_degree = hcg.get_expert_parallel_world_size()
            if hasattr(hcg, "get_moe_sharding_parallel_world_size"):
                moe_sharding_degree = hcg.get_moe_sharding_parallel_world_size()
        parallel_config = {
            "pp_degree": pp_degree,
            "mp_degree": mp_degree,
            "sharding_degree": sharding_degree,
            "ep_degree": ep_degree,
            "moe_sharding_degree": moe_sharding_degree,
        }
        self._check_distributed_strategy(parallel_config)
        return parallel_config

    def _recover_params_from_master_weights(self, state_dict, opt_state_dict=None, group=None):
        if group is None:
            group = self.sharding_group
        if opt_state_dict is None:
            opt_state_dict = self.optimizer.state_dict()
        assert "master_weights" in opt_state_dict, opt_state_dict.keys()
        master_weights = opt_state_dict["master_weights"]
        tmp = OrderedDict()
        (master_weights, tmp) = (tmp, master_weights)
        # cast to before
        for (k, v) in tmp.items():
            name = v.name
            master_weights[k] = paddle.cast(to_device(v), paddle.bfloat16).cpu()
            master_weights[k].name = name

        structure_name_map = {k: v.name for (k, v) in self.model.state_dict().items()}
        node_model_state = reshard_util.NodeModelState(group=group)
        node_model_state_tmp = reshard_util.NodeModelState(group=group)
        node_model_state_tmp.add_master_weights(master_weights)
        node_model_state_tmp.pack_keys(structure_name_map)
        node_model_state.merge_from(node_model_state_tmp, max(group.rank, 0))
        del node_model_state_tmp
        sharding_strategy = reshard_util.get_sharding_strategy(self.optimizer)
        restore_func = (
            reshard_util.sharding_v1.restore
            if sharding_strategy == SHARDING_STRATEGY_V1
            else reshard_util.sharding_v2.restore
        )
        node_model_state = restore_func(node_model_state, self.model, self.optimizer)
        node_model_state.unpack_keys()
        master_weights = node_model_state.master_weights

        def filter_func(name):
            return True

        master_weights = reshard_util.all_gather_state_dict(master_weights, filter_func, group)
        model_state_dict = self.model.state_dict()
        logger.info(f"state-dict-keys: {state_dict.keys()}, nums: {len(state_dict.keys())}")
        logger.info("before recover, model_state_dict number: {}".format(len(model_state_dict)))
        for key, param in model_state_dict.items():
            if param.name in master_weights:
                assert param.shape == master_weights[param.name].shape
                paddle.assign(
                    paddle.cast(to_device(master_weights[param.name]), paddle.bfloat16), model_state_dict[key]
                )
            elif key in state_dict:
                logger.info(f"key: {key} is in state_dict, but not in master_weights")
                paddle.assign(state_dict[key], model_state_dict[key])
            else:
                logger.info(f"key: {key} is not in state_dict and master_weights")
        logger.info("after recover, casted model_state_dict number: {}".format(len(model_state_dict)))
        state_dict.update(model_state_dict)
        return state_dict

    def _all_gather_simple_object(self, obj, group=None):
        if group is None:
            group = self.hcg.get_sharding_parallel_group()
        res = []
        if group.nranks < 2:
            return [obj]
        paddle.distributed.all_gather_object(res, obj, group)
        return res

    def _load_model_meta_impl(self, dir):
        meta_path = os.path.join(dir, MODEL_META_NAME)
        assert os.path.exists(meta_path), f"{meta_path} not exist"
        with open(meta_path, "r") as handle:
            model_dist_meta = json.load(handle)
        assert "parallel_config" in model_dist_meta
        self._check_distributed_strategy(model_dist_meta["parallel_config"])
        return model_dist_meta

    def _load_model_meta(self, dir):
        model_meta = self._load_model_meta_impl(dir)
        remapper = self._get_remapper(dir)
        if remapper is not None:
            suffix = self._sharding_meta_suffix()
            sharding_metas = model_meta["sharding_metas"]
            cur_sharding_metas = sharding_metas.pop(suffix)
            sharding_metas.clear()
            sharding_metas[suffix] = cur_sharding_metas
            cur_sharding_metas["structure_name_mapping"] = remapper.new_mapping
            if "param2rank" in cur_sharding_metas:
                new_param2rank = {}
                for k, rank in cur_sharding_metas["param2rank"].items():
                    new_k = remapper.p_name_map[k]
                    new_param2rank[new_k] = rank
                cur_sharding_metas["param2rank"] = new_param2rank
        return model_meta

    def _sharding_meta_suffix(self, tp_rank=None, pp_rank=None):
        if tp_rank is None:
            tp_rank = self.args.tensor_parallel_rank
        if pp_rank is None:
            pp_rank = self.args.pipeline_parallel_rank
        suffix = f"tp{tp_rank:0>2d}_pp{pp_rank:0>2d}"
        if self.args.expert_parallel_degree > 1:
            ep_rank = self.args.expert_parallel_rank
            return f"{suffix}_ep{ep_rank:0>2d}"
        else:
            return suffix

    def _load_distributed_strategy(self, dir):
        model_dist_meta = self._load_model_meta(dir)
        parallel_config = model_dist_meta["parallel_config"]
        assert "pp_degree" in parallel_config
        assert "mp_degree" in parallel_config
        assert "sharding_degree" in parallel_config
        return parallel_config

    def _load_sharding_meta(self, dir, pp_rank=None):
        suffix = self._sharding_meta_suffix(pp_rank=pp_rank)
        distributed_model_meta = self._load_model_meta(dir)
        if "sharding_metas" in distributed_model_meta:
            sharding_metas = distributed_model_meta["sharding_metas"]
            assert suffix in sharding_metas
            sharding_meta = sharding_metas[suffix]
            assert "param2rank" in sharding_meta
            return sharding_meta

        # for backward compatibility
        meta_path = os.path.join(dir, _add_variant(SHARDING_META_NAME, suffix))
        assert os.path.exists(meta_path), f"{meta_path} not exist"
        with open(meta_path, "r") as f:
            sharding_meta = json.load(f)
        assert "param2rank" in sharding_meta
        return sharding_meta

    def _map_optimizer_state_to_param(self, optimizer_state_names):
        optimizer = unwrap_optimizer(self.optimizer, DygraphShardingOptimizer)
        all_names = list(optimizer._param2rank.keys())
        all_names.extend(list(optimizer_state_names))
        all_names.sort()
        pre_p_name = ""
        opt_to_p = {}
        for n in all_names:
            if n in optimizer._param2rank:
                # we get a param
                pre_p_name = n
            else:
                assert pre_p_name, n
                opt_to_p[n] = pre_p_name
        return opt_to_p

    def _gather_sharding_metas(self):
        nranks = dist.get_world_size()
        if not self.args.use_hybrid_parallel or nranks <= 1:
            return None
        if not reshard_util.is_sharding_opt(self.optimizer):
            return None

        sharding_strategy = reshard_util.get_sharding_strategy(self.optimizer)
        param2rank = {}
        pp_overlap = False
        if sharding_strategy == SHARDING_STRATEGY_V1:
            optimizer = unwrap_optimizer(self.optimizer, DygraphShardingOptimizer)
            param2rank = {k: v for (k, v) in optimizer._param2rank.items()}
        else:
            pp_overlap = unwrap_optimizer(self.optimizer, DygraphShardingOptimizerV2).pp_overlap

        model = self.model
        structure_name_mapping = {}
        param_meta = {}
        for k, v in model.state_dict().items():
            structure_name_mapping[k] = v.name
            is_distributed = getattr(v, "is_distributed", False)
            no_sync = getattr(v, "no_sync", False)
            param_meta[k] = (v.shape, int(v.dtype), is_distributed, no_sync)

        sharding_metas = {}
        sharding_meta = {}

        sharding_meta["param2rank"] = param2rank
        sharding_meta["structure_name_mapping"] = structure_name_mapping
        sharding_meta["param_meta"] = param_meta
        sharding_meta["param_meta_keys"] = ["shape", "dtype", "is_distributed", "no_sync"]
        sharding_meta["sharding_strategy"] = sharding_strategy
        sharding_meta["enable_overlap"] = pp_overlap
        dp_metas_list = self._all_gather_simple_object(sharding_meta, self.hcg.get_data_parallel_group())
        for e in dp_metas_list:
            for key in ["structure_name_mapping", "param_meta"]:
                sharding_meta[key].update(e[key])
        suffix = self._sharding_meta_suffix()
        sharding_metas[suffix] = sharding_meta
        sharding_metas_list = self._all_gather_simple_object(sharding_metas, self.hcg.get_model_parallel_group())
        sharding_metas = {k: v for e in sharding_metas_list for (k, v) in e.items()}
        sharding_metas_list = self._all_gather_simple_object(sharding_metas, self.hcg.get_pipe_parallel_group())
        sharding_metas = {k: v for e in sharding_metas_list for (k, v) in e.items()}
        if self.args.expert_parallel_degree > 1:
            sharding_metas_list = self._all_gather_simple_object(sharding_metas, self.hcg.get_expert_parallel_group())
            sharding_metas = {k: v for e in sharding_metas_list for (k, v) in e.items()}
        return sharding_metas
