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

import time
from collections import OrderedDict

import numpy as np
import paddle
import paddle.distributed.fleet as fleet
from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer.dygraph_sharding_optimizer import (
    DygraphShardingOptimizer,
    DygraphShardingOptimizerV2,
)

try:
    from paddle.distributed.fleet.meta_optimizers.muon_sharding_optimizer import (
        MuonShardingOptimizer,
    )
except (ImportError, ModuleNotFoundError):
    MuonShardingOptimizer = None

try:
    from paddle.distributed.communication.batch_isend_irecv import _coalescing_manager
except (ImportError, ModuleNotFoundError):
    _coalescing_manager = None

from paddle.distributed.fleet.utils.log_util import logger

from paddleformers.utils.tools import get_env_device

from ....transformers.model_utils import unwrap_optimizer

SHARDING_STRATEGY_V1 = "ShardingV1"
SHARDING_STRATEGY_V2 = "ShardingV2"

_STATE_DICT_BROADCAST_BUCKET_SIZE_BYTES = 128 * 1024 * 1024
_STATE_DICT_BROADCAST_CHUNK_SIZE = 256
_STATE_DICT_BROADCAST_MAX_CHUNK_BYTES = 2 * 1024 * 1024 * 1024
_STATE_DICT_BROADCAST_LOG_INTERVAL_SECONDS = 10

# Mutable peak-memory knob for the bucketed broadcast path: the max bytes of a
# chunk kept resident on GPU at once. Sourced from TrainingArguments
# .reshard_bucketed_broadcast_max_chunk_gb via set_broadcast_max_chunk_bytes()
# at the reshard entry points; defaults to the constant above. Difers
_broadcast_max_chunk_bytes = _STATE_DICT_BROADCAST_MAX_CHUNK_BYTES


def is_sharding_opt(optimizer):
    def check(cls):
        tmp = unwrap_optimizer(optimizer, cls)
        if tmp is not None:
            return True
        return False

    if check(DygraphShardingOptimizer):
        return True

    if DygraphShardingOptimizerV2 is not None:
        if check(DygraphShardingOptimizerV2):
            return True

    if MuonShardingOptimizer is not None:
        if check(MuonShardingOptimizer):
            return True

    return False


def get_sharding_strategy(optimizer):
    if DygraphShardingOptimizerV2 is not None:
        tmp = unwrap_optimizer(optimizer, DygraphShardingOptimizerV2)
        if tmp is not None:
            return SHARDING_STRATEGY_V2
    return SHARDING_STRATEGY_V1


def convert_opt_name_to_tname(tensor_names, opt_names):
    tensor_names = set(tensor_names)
    all_names = []
    all_names.extend(list(tensor_names))
    all_names.extend(opt_names)
    all_names.sort()
    pre_t_name = ""
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
    opt_to_t = {}
    for n in all_names:
        if n in tensor_names:
            # we get a param
            pre_t_name = n
        else:
            assert pre_t_name
            opt_to_t[n] = pre_t_name

    for t in opt_names:
        _find = False
        for s in suffix:
            if get_env_device() == "xpu" and t.endswith(s + ".SCALE_VALUE"):
                # NOTE: for xpu adamw, all optimizer state will have an extra attribute end with SCALE_VALUE.
                # This extra attribute won't be used, just skip it.
                _find = True
                break
            if t.endswith(s):
                logger.info(f"{t}-{t[:-len(s)]}--{t[:-len(s)] in tensor_names}")
                opt_to_t[t] = t[: -len(s)]
                _find = True
                break
        assert _find, t
    return opt_to_t


class NodeModelState:
    def __init__(self, group):
        self._model_weights = OrderedDict()
        self._opt_state = OrderedDict()
        self._master_weights = OrderedDict()
        self._lr_scheduler = None
        self._group = group

    @property
    def group(self):
        return self._group

    def _add_kv(self, d, k, v):
        assert k not in d
        d[k] = v

    @property
    def model_weights(self):
        return self._model_weights

    def add_weight(self, k, v):
        self._add_kv(self._model_weights, k, v)

    def add_weights(self, model_state_dict, rank=None):
        for (k, v) in model_state_dict.items():
            if rank is not None:
                k = (k, rank)
            self.add_weight(k, v)

    def set_weights(self, model_state_dict):
        self._model_weights = model_state_dict

    def set_opt_state(self, opt_state_dict):
        self._opt_state = opt_state_dict

    def set_master_weights(self, master_weights):
        self._master_weights = master_weights

    @property
    def opt_state(self):
        return self._opt_state

    def add_opt(self, k, v):
        self._add_kv(self._opt_state, k, v)

    def add_opts(self, opts, rank=None):
        if "master_weights" in opts:
            s_master = opts["master_weights"]
            opts.pop("master_weights")
            self.add_master_weights(s_master, rank)

        if "LR_Scheduler" in opts:
            lr_scheduler = opts["LR_Scheduler"]
            opts.pop("LR_Scheduler")
            self.set_lr_scheduler(lr_scheduler)

        for (k, v) in opts.items():
            if rank is not None:
                k = (k, rank)
            self.add_opt(k, v)

    @property
    def master_weights(self):
        return self._master_weights

    def add_master_weight(self, k, v):
        self._add_kv(self._master_weights, k, v)

    def add_master_weights(self, master, rank=None):
        for (k, v) in master.items():
            if rank is not None:
                k = (k, rank)
            self.add_master_weight(k, v)

    @property
    def lr_scheduler(self):
        return self._lr_scheduler

    def set_lr_scheduler(self, lr_scheduler):
        if lr_scheduler is not None:
            self._lr_scheduler = lr_scheduler

    def map_names(self, map_func):
        """
        rename param names and change the keys of the dicts(model_weights, opt, master_weights) accordingly
        """

        def map_key(state_dict, map_key_func):
            state_dict_tmp = OrderedDict()
            (state_dict_tmp, state_dict) = (state_dict, state_dict_tmp)
            for key in list(state_dict_tmp.keys()):
                key_new = map_key_func(key)
                state_dict[key_new] = state_dict_tmp[key]
                del state_dict_tmp[key]
            return state_dict

        def map_model_state_key(key):
            packed = isinstance(key[0], tuple)
            structure_name, t_name = key[0] if packed else key
            t_name_new = map_func(structure_name, t_name)
            key_new = ((structure_name, t_name_new), key[1]) if packed else (structure_name, t_name_new)
            return key_new

        def map_opt_key(key):
            packed = isinstance(key[0], tuple)
            structure_name, t_name, opt_name = key[0] if packed else key
            t_name_new = map_func(structure_name, t_name)
            opt_name_new = t_name_new + opt_name[len(t_name) :]
            key_new = (
                ((structure_name, t_name_new, opt_name_new), key[1])
                if packed
                else (structure_name, t_name_new, opt_name_new)
            )
            return key_new

        self._model_weights = map_key(self._model_weights, map_model_state_key)
        self._opt_state = map_key(self._opt_state, map_opt_key)
        self._master_weights = map_key(self._master_weights, map_opt_key)
        return self

    def drop_rank(self):
        """
        drop rank in the keys of the state dict
        change dict of (key, rank)=>tensor to dict of key =>tensor
        """

        def drop(state, l=2):
            tmp_state = OrderedDict()
            (state, tmp_state) = (tmp_state, state)
            for key in list(tmp_state.keys()):
                k, rank = key
                assert len(key) == 2
                assert len(k) == l
                state[k] = tmp_state[key]
                del tmp_state[key]
            return state

        self._model_weights = drop(self._model_weights, 2)
        self._opt_state = drop(self._opt_state, 3)
        self._master_weights = drop(self._master_weights, 3)
        return self

    def collapse_key(self):
        """
        collapse dict of (key, rank)=>tensor to dict of key=>list[(rank, tensor)]
        """

        def collapse(state, l):
            tmp_state = OrderedDict()
            (state, tmp_state) = (tmp_state, state)
            state_keys = list(tmp_state.keys())
            state_keys = sorted(state_keys)
            pre = None
            for key in state_keys:
                assert len(key) == 2
                k, rank = key
                if isinstance(k, tuple):
                    assert len(k) == l
                if k != pre:
                    pre = k
                    state[k] = []
                state[k].append((rank, tmp_state[key]))
                del tmp_state[key]
            return state

        self._model_weights = collapse(self._model_weights, 2)
        self._opt_state = collapse(self._opt_state, 3)
        self._master_weights = collapse(self._master_weights, 3)
        return self

    def flatten_key(self):
        """
        flatten dict of key=>list[(rank, tensor)], to dict of (key, rank)=>tensor
        """

        def flatten(state, l):
            tmp_state = OrderedDict()
            (state, tmp_state) = (tmp_state, state)
            state_keys = list(tmp_state.keys())
            for key in state_keys:
                assert len(key) == l
                for (rank, items) in tmp_state[key]:
                    state[(key, rank)] = items
                del tmp_state[key]
            return state

        self._model_weights = flatten(self._model_weights, 2)
        self._opt_state = flatten(self._opt_state, 3)
        self._master_weights = flatten(self._master_weights, 3)
        return self

    def pack_keys(self, structure_name_mapping=None):
        """
        change the key of model_weights dict from param_name to (structure_name, param_name);
        change the key of opt dict from opt_name to (structure_name, param_name, opt_name);
        change the key of master weights dict from param_name to (structure_name, param_name)
        """
        # pack key for pp convert
        if structure_name_mapping is not None:
            tname_to_structure_name = {v: k for (k, v) in structure_name_mapping.items()}
        else:
            structure_name_mapping = {k: v.name for (k, v) in self._model_weights.items()}
            tname_to_structure_name = {v: k for (k, v) in structure_name_mapping.items()}

        tensor_names = list(tname_to_structure_name.keys())
        opt_names = list(self._opt_state.keys())
        opt_name_to_tname = convert_opt_name_to_tname(tensor_names, opt_names)

        # model state
        model_weights_tmp = OrderedDict()
        (self._model_weights, model_weights_tmp) = (model_weights_tmp, self._model_weights)
        for k in list(model_weights_tmp.keys()):
            t_name = structure_name_mapping[k]
            self._model_weights[(k, t_name)] = paddle.to_tensor(model_weights_tmp[k]).cpu()
            del model_weights_tmp[k]

        # opt
        opt_tmp = OrderedDict()
        (self._opt_state, opt_tmp) = (opt_tmp, self._opt_state)
        for opt_name in list(opt_tmp.keys()):
            assert opt_name in opt_name_to_tname
            t_name = opt_name_to_tname[opt_name]
            assert t_name in tname_to_structure_name
            structure_name = tname_to_structure_name[t_name]
            self._opt_state[(structure_name, t_name, opt_name)] = opt_tmp[opt_name].cpu()
            del opt_tmp[opt_name]

        # master weights
        master_weights_tmp = OrderedDict()
        (self._master_weights, master_weights_tmp) = (master_weights_tmp, self._master_weights)
        for t_name in list(master_weights_tmp.keys()):
            assert t_name in tname_to_structure_name
            structure_name = tname_to_structure_name[t_name]
            master_name = getattr(master_weights_tmp[t_name], "name", "")
            self._master_weights[(structure_name, t_name, master_name)] = master_weights_tmp[t_name].cpu()
            del master_weights_tmp[t_name]

        return self

    def unpack_keys(self):
        """
        the opposite of pack_keys,
        revert the key of model_weights dict from  (structure_name, param_name) to param_name
        revert the key of opt dict from  (structure_name, param_name, opt_name) to opt_name
        revert the key of master weights dict from (structure_name, param_name) to param_name
        """
        # model weights
        model_weights_tmp = OrderedDict()
        (self._model_weights, model_weights_tmp) = (model_weights_tmp, self._model_weights)
        for key in list(model_weights_tmp.keys()):
            structure_name, t_name = key
            self._model_weights[structure_name] = model_weights_tmp[key]
            self._model_weights[structure_name].name = t_name
            del model_weights_tmp[key]
        # opt
        opt_tmp = OrderedDict()
        (self._opt_state, opt_tmp) = (opt_tmp, self._opt_state)
        for key in list(opt_tmp.keys()):
            structure_name, t_name, opt_name = key
            if structure_name in self._model_weights:
                assert self._model_weights[structure_name].name == t_name
            self._opt_state[opt_name] = opt_tmp[key]
            self._opt_state[opt_name].name = opt_name
            del opt_tmp[key]

        # master weights
        master_weights_tmp = OrderedDict()
        (self._master_weights, master_weights_tmp) = (master_weights_tmp, self._master_weights)
        for key in list(master_weights_tmp.keys()):
            structure_name, t_name, master_name = key
            if structure_name in self._model_weights:
                assert self._model_weights[structure_name].name == t_name
            self._master_weights[t_name] = master_weights_tmp[key]
            self._master_weights[t_name].name = master_name
        return self

    def split_state(self, split_func):
        """
        split this node state to multiple node state according to the passed in split_func
        """
        node_model_states = {}
        for (k, v) in self._model_weights.items():
            rank = split_func(k)
            if rank not in node_model_states:
                node_model_states[rank] = NodeModelState()
            node_model_states[rank].add_weight(k, v)

        for (k, v) in self._opt_state.items():
            rank = split_func(k)
            if rank not in node_model_states:
                node_model_states[rank] = NodeModelState()
            node_model_states[rank].add_opt(k, v)

        for (k, v) in self._master_weights.items():
            rank = split_func(k)
            if rank not in node_model_states:
                node_model_states[rank] = NodeModelState()
            node_model_states[rank].add_master_weight(k, v)

        return node_model_states

    def even_distribute(self):
        """
        distribute the node state evenly among all workers in group， and make sure
        in the dicts of (key, rank)=>tensor, items keys of the same key but different rank are distributed to the
        same worker
        """
        group = self.group
        # sharding degree == 1
        if group is None or group.nranks < 2:
            return self

        def build_router(state_dict):
            state_keys_list = all_gather_simple_object([(k, v.shape) for (k, v) in state_dict.items()], group)

            key_to_size = {}
            for l in state_keys_list:
                for (k, shape) in l:
                    key, rank = k
                    if key not in key_to_size:
                        key_to_size[key] = 0
                    key_to_size[key] = key_to_size[key] + np.prod(shape)

            key_to_size = sorted(list(key_to_size.items()), key=lambda x: x[1], reverse=True)
            node_distributed = [0 for _ in range(group.nranks)]
            key_to_rank = {}
            for (k, v) in key_to_size:
                min_val = min(node_distributed)
                min_index = node_distributed.index(min_val)
                key_to_rank[k] = min_index
                node_distributed[min_index] = node_distributed[min_index] + v

            return key_to_rank

        def distribute(state_dict):

            key_to_rank = build_router(state_dict)

            def filter_func(key):
                assert key[0] in key_to_rank, key
                dst_rank = key_to_rank[key[0]]
                return dst_rank == max(group.rank, 0)

            return _all_gather_state_dict(state_dict, filter_func, group)

        self._model_weights = distribute(self._model_weights)
        self._opt_state = distribute(self._opt_state)
        self._master_weights = distribute(self._master_weights)
        return self

    def reshard(self, filter_func):
        """
        reshard according to the passed in filter_func
        """
        group = self.group
        self._model_weights = _all_gather_state_dict(self._model_weights, filter_func, group)
        self._opt_state = _all_gather_state_dict(self._opt_state, filter_func, group)
        self._master_weights = _all_gather_state_dict(self._master_weights, filter_func, group)
        lr_schedulers = all_gather_simple_object(self._lr_scheduler, group)
        self._lr_scheduler = lr_schedulers[0]
        return self

    def split_items(self, split_func):
        """
        split tensor in the dicts of key=tensor, change the dicts to dicts of key=>list[(rank, tensor)]
        """

        def split(state, l):
            tmp_state = OrderedDict()
            (state, tmp_state) = (tmp_state, state)
            state_keys = list(tmp_state.keys())
            for key in state_keys:
                assert len(key) == l
                v = tmp_state[key]
                state[key] = split_func(key, v)
                del tmp_state[key]
            return state

        self._model_weights = split(self._model_weights, 2)
        self._opt_state = split(self._opt_state, 3)
        self._master_weights = split(self._master_weights, 3)
        return self

    def merge_items(self, merge_func):
        """
        merge list in the dicts of key=>list[(rank, tensor)]  a tensor, change the dicts to dicts of key=>tensor
        """

        def merge(state, l):
            tmp_state = OrderedDict()
            (state, tmp_state) = (tmp_state, state)
            state_keys = list(tmp_state.keys())
            for key in state_keys:
                if isinstance(key, tuple):
                    assert len(key) == l
                v = tmp_state[key]
                v = sorted(v, key=lambda x: x[0])
                state[key] = merge_func(key, v)
                del tmp_state[key]
            return state

        self._model_weights = merge(self._model_weights, 2)
        self._opt_state = merge(self._opt_state, 3)
        self._master_weights = merge(self._master_weights, 3)
        return self

    def merge_from(self, other, rank=None):
        assert other.group is self.group
        self.add_weights(other.model_weights, rank)
        self.add_opts(other.opt_state, rank)
        self.add_master_weights(other.master_weights, rank)
        if other.lr_scheduler is not None:
            self.set_lr_scheduler(other.lr_scheduler)
        return self

    def get_opt_state_dict(self):
        opt_state_dict = OrderedDict()
        for (k, v) in self.opt_state.items():
            opt_state_dict[k] = v
        if self._lr_scheduler is not None:
            opt_state_dict["LR_Scheduler"] = self._lr_scheduler
        opt_state_dict["master_weights"] = self._master_weights
        return opt_state_dict


def split_model_state(model_state, group_getter):
    res = OrderedDict()
    for k, v in model_state.items():
        group = group_getter.get_group(k)
        if group.id not in res:
            res[group.id] = OrderedDict()
        res[group.id][k] = v
    return res


def merge_model_state(model_state_map):
    res = OrderedDict()
    for gid, model_state in model_state_map.items():
        res.update(model_state)
    return res


def split_opt_state(opt_state, group_getter):
    res = OrderedDict()
    lr_scheduler = opt_state.get("LR_Scheduler", None)
    for k, v in opt_state.items():
        if k == "LR_Scheduler":
            continue
        elif k == "master_weights":
            for kk, vv in v.items():
                group = group_getter.get_group(kk)
                if group.id not in res:
                    res[group.id] = {"master_weights": OrderedDict(), "LR_Scheduler": lr_scheduler}
                res[group.id]["master_weights"][kk] = vv
        else:
            assert isinstance(v, paddle.Tensor), type(v)
            group = group_getter.get_group(k)
            if group.id not in res:
                res[group.id] = {"master_weights": OrderedDict(), "LR_Scheduler": lr_scheduler}
            res[group.id][k] = v
    return res


def merge_opt_state(opt_state_map):
    res = {"LR_Scheduler": None, "master_weights": OrderedDict()}
    for gid, opt_state in opt_state_map.items():
        for k, v in opt_state.items():
            if k == "LR_Scheduler":
                if v is not None:
                    res["LR_Scheduler"] = v
            elif k == "master_weights":
                res["master_weights"].update(v)
            else:
                res[k] = v
    return res


def split_structure_name_mapping(structure_name_mapping, group_getter):
    res = OrderedDict()
    for k, v in structure_name_mapping.items():
        group = group_getter.get_group(k)
        if group.id not in res:
            res[group.id] = OrderedDict()
        res[group.id][k] = v
    return res


def all_gather_simple_object(obj, group):
    res = []
    if group.nranks < 2:
        return [obj]
    paddle.distributed.all_gather_object(res, obj, group)
    return res


def _shape_numel(shape):
    if len(shape) == 0:
        return 1
    return int(np.prod(shape, dtype=np.int64))


def _dtype_itemsize(dtype):
    return paddle.empty([0], dtype=dtype, device="cpu").element_size()


def _build_state_dict_broadcast_buckets(meta_list, bucket_size_bytes):
    assert bucket_size_bytes > 0

    grouped_items = OrderedDict()
    empty_items = []
    for k, (dtype, shape, rank) in meta_list:
        numel = _shape_numel(shape)
        item = (k, shape, numel)
        if numel == 0:
            empty_items.append((k, (dtype, shape, rank)))
            continue
        grouped_items.setdefault((rank, dtype), []).append(item)

    buckets = []
    for (rank, dtype), items in grouped_items.items():
        itemsize = _dtype_itemsize(dtype)
        bucket_items = []
        bucket_numel = 0

        def flush_bucket():
            nonlocal bucket_items, bucket_numel
            if not bucket_items:
                return
            buckets.append(
                {
                    "rank": rank,
                    "dtype": dtype,
                    "numel": bucket_numel,
                    "nbytes": bucket_numel * itemsize,
                    "items": bucket_items,
                }
            )
            bucket_items = []
            bucket_numel = 0

        for k, shape, numel in items:
            item_nbytes = numel * itemsize
            if bucket_items and (bucket_numel * itemsize + item_nbytes > bucket_size_bytes):
                flush_bucket()

            begin = bucket_numel
            bucket_items.append((k, shape, begin, begin + numel))
            bucket_numel += numel

            # Oversized tensors stay in their own bucket instead of increasing
            # the peak memory of neighboring tensors.
            if item_nbytes >= bucket_size_bytes:
                flush_bucket()

        flush_bucket()

    return buckets, empty_items


def _iter_state_dict_bucket_chunks(buckets, chunk_size, max_chunk_bytes):
    assert chunk_size > 0
    assert max_chunk_bytes > 0

    chunk = []
    chunk_bytes = 0
    for bucket in buckets:
        bucket_bytes = bucket["nbytes"]
        if chunk and (len(chunk) >= chunk_size or chunk_bytes + bucket_bytes > max_chunk_bytes):
            yield chunk
            chunk = []
            chunk_bytes = 0
        chunk.append(bucket)
        chunk_bytes += bucket_bytes
    if chunk:
        yield chunk


def _pack_state_dict_bucket(bucket, state_dict):
    items = bucket["items"]
    first_key = items[0][0]
    assert first_key in state_dict
    np_dtype = np.asarray(state_dict[first_key]).dtype

    # Scatter the whole bucket on the host first so it takes a single
    # host-to-device copy instead of one small copy per state tensor.
    staged = np.empty([bucket["numel"]], dtype=np_dtype)
    for k, _, begin, end in items:
        assert k in state_dict
        value = np.asarray(state_dict.pop(k)).reshape([-1])
        assert value.shape[0] == end - begin
        assert value.dtype == np_dtype
        staged[begin:end] = value
        del value

    tensor = paddle.to_tensor(staged)
    assert tensor.shape[0] == bucket["numel"]
    assert str(tensor.dtype).split(".")[-1] == bucket["dtype"]
    return tensor


def _unpack_state_dict_bucket(bucket, tensor, selected_keys, gathered):
    selected_items = [item for item in bucket["items"] if item[0] in selected_keys]
    if not selected_items:
        return

    if len(selected_items) == len(bucket["items"]):
        cpu_tensor = tensor.cpu()
        for k, shape, begin, end in selected_items:
            # Every item of the bucket is selected, so the slices exactly cover
            # the bucket storage. Hand out views instead of per-item copies:
            # cloning here would duplicate the whole bucket for no gain.
            gathered[k] = cpu_tensor[begin:end].reshape(shape)
        del cpu_tensor
        return

    # Sharded restore normally keeps only a small subset of each bucket on a
    # rank. Copy just those slices instead of staging the complete bucket on
    # every rank. ``.cpu()`` already allocates fresh host storage sized to the
    # slice, so no extra clone is needed.
    for k, shape, begin, end in selected_items:
        gathered[k] = tensor[begin:end].cpu().reshape(shape)


def _broadcast_state_dict_chunk(gpu_buckets, group):
    if group.nranks < 2:
        return

    if _coalescing_manager is None:
        for bucket, tensor in gpu_buckets:
            paddle.distributed.broadcast(
                tensor,
                src=group.ranks[bucket["rank"]],
                group=group,
                sync_op=True,
            )
        return

    # Every bucket has its own broadcast root, so a chunk is a batch of small
    # multi-root messages. At large nranks the per-call launch and rendezvous
    # cost dominates the payload, so aggregate the whole chunk into a single
    # ncclGroupStart/End instead of paying it once per bucket.
    tasks = []
    with _coalescing_manager(group, tasks):
        for bucket, tensor in gpu_buckets:
            paddle.distributed.stream.broadcast(
                tensor,
                src=group.ranks[bucket["rank"]],
                group=group,
                sync_op=True,
                use_calc_stream=True,
            )


# The bucketed path packs many small state tensors into large buckets and
# coalesces their broadcasts, cutting NCCL/H2D calls from O(#tensors) to
# O(#buckets). The tradeoff is that a whole ~2GiB chunk must stay resident on
# GPU at once, so it raises peak device memory for small-tensor-heavy reshards
# (see _STATE_DICT_BROADCAST_MAX_CHUNK_BYTES). Default to the original
# per-tensor path. The value is driven by TrainingArguments
# .use_reshard_bucketed_broadcast, applied via set_bucketed_broadcast() at the
# reshard entry points that hold args, so the deep all_gather_state_dict call
# chain does not have to thread the flag through every function. Difers
_USE_BUCKETED_BROADCAST = False


def set_bucketed_broadcast(enabled):
    global _USE_BUCKETED_BROADCAST
    _USE_BUCKETED_BROADCAST = bool(enabled)


def set_broadcast_max_chunk_bytes(nbytes):
    global _broadcast_max_chunk_bytes
    nbytes = int(nbytes)
    if nbytes <= 0:
        _broadcast_max_chunk_bytes = _STATE_DICT_BROADCAST_MAX_CHUNK_BYTES
        return
    # A chunk must hold at least one full bucket, otherwise every bucket lands in
    # its own chunk and the broadcast coalescing is lost. Floor at bucket size.
    _broadcast_max_chunk_bytes = max(nbytes, _STATE_DICT_BROADCAST_BUCKET_SIZE_BYTES)


def all_gather_state_dict(state_dict, filter_func, group):
    if _USE_BUCKETED_BROADCAST:
        return _all_gather_state_dict_bucketed(state_dict, filter_func, group)
    return _all_gather_state_dict_legacy(state_dict, filter_func, group)


def _all_gather_state_dict_legacy(state_dict, filter_func, group):
    res = OrderedDict()

    group_rank = max(group.rank, 0)

    # Convert tensors to numpy upfront to free GPU memory.
    # This bounds peak GPU memory to chunk_size tensors during broadcast.
    meta_dict = {}
    for (k, v) in state_dict.items():
        if isinstance(v, paddle.Tensor):
            meta_dict[k] = (str(v.dtype).split(".")[-1], list(v.shape), group_rank)
            state_dict[k] = v.numpy()
        else:
            meta_dict[k] = (str(v.dtype), list(v.shape), group_rank)

    meta_dict_list = all_gather_simple_object(meta_dict, group)

    total_meta_dict = {}
    for meta_dict in meta_dict_list:
        for (k, v) in meta_dict.items():
            assert k not in total_meta_dict
            total_meta_dict[k] = v

    meta_list = list(total_meta_dict.items())
    meta_list = sorted(meta_list, key=lambda x: (x[1][2], x[0]))

    # Process in chunks to balance broadcast throughput and GPU memory usage.
    # Within a chunk, all broadcasts are done first (no CPU sync in between),
    # then results are moved to CPU together.
    chunk_size = 8
    for chunk_start in range(0, len(meta_list), chunk_size):
        chunk = meta_list[chunk_start : chunk_start + chunk_size]

        # Phase 1: prepare all tensors on GPU (batch CPU->GPU transfers)
        gpu_tensors = []
        for (k, meta) in chunk:
            dtype, shape, rank = meta
            if rank == group_rank:
                assert k in state_dict
                tensor = paddle.to_tensor(state_dict[k])
                del state_dict[k]
            else:
                tensor = paddle.empty(shape=shape, dtype=dtype)
            gpu_tensors.append((k, meta, tensor))

        # Phase 2: broadcast all tensors continuously without interruption
        for (k, meta, tensor) in gpu_tensors:
            _, _, rank = meta
            logger.info(f"broadcast {k} from {rank}, group {group}")
            if group.nranks > 1:
                paddle.distributed.broadcast(
                    tensor,
                    src=group.ranks[rank],
                    group=group,
                    sync_op=True,
                )

        # Phase 3: move to CPU and release GPU memory
        for (k, _, tensor) in gpu_tensors:
            if filter_func(k):
                res[k] = tensor.cpu()
            del tensor

    return res


def _all_gather_state_dict_bucketed(state_dict, filter_func, group):
    group_rank = max(group.rank, 0)

    # Convert source tensors to numpy first so packing does not retain their
    # original GPU allocations.
    meta_dict = {}
    for (k, v) in state_dict.items():
        if isinstance(v, paddle.Tensor):
            meta_dict[k] = (str(v.dtype).split(".")[-1], list(v.shape), group_rank)
            state_dict[k] = v.numpy()
        else:
            meta_dict[k] = (str(v.dtype), list(v.shape), group_rank)

    meta_dict_list = all_gather_simple_object(meta_dict, group)

    total_meta_dict = {}
    for meta_dict in meta_dict_list:
        for (k, v) in meta_dict.items():
            assert k not in total_meta_dict
            total_meta_dict[k] = v

    meta_list = list(total_meta_dict.items())
    meta_list = sorted(meta_list, key=lambda x: (x[1][2], x[0]))
    selected_keys = {k for k, _ in meta_list if filter_func(k)}
    buckets, empty_items = _build_state_dict_broadcast_buckets(meta_list, _STATE_DICT_BROADCAST_BUCKET_SIZE_BYTES)
    gathered = {}

    for k, (dtype, shape, rank) in empty_items:
        if rank == group_rank:
            assert k in state_dict
            del state_dict[k]
        if k in selected_keys:
            gathered[k] = paddle.empty(shape, dtype=dtype, device="cpu")

    if group_rank == 0:
        logger.info(
            f"broadcast {len(meta_list)} state tensors in {len(buckets)} buckets, "
            f"bucket_size={_STATE_DICT_BROADCAST_BUCKET_SIZE_BYTES // (1024 * 1024)} MiB, "
            f"chunk_size={_STATE_DICT_BROADCAST_CHUNK_SIZE}, "
            f"max_chunk={_broadcast_max_chunk_bytes // (1024 * 1024)} MiB, "
            f"coalescing={_coalescing_manager is not None}, "
            f"nranks={group.nranks}, group_id={group.id}"
        )

    bucket_chunks = _iter_state_dict_bucket_chunks(
        buckets,
        _STATE_DICT_BROADCAST_CHUNK_SIZE,
        _broadcast_max_chunk_bytes,
    )
    start = time.time()
    last_log = start
    pack_seconds = 0.0
    bcast_seconds = 0.0
    unpack_seconds = 0.0
    done_buckets = 0
    done_chunks = 0
    for chunk in bucket_chunks:
        t0 = time.time()
        gpu_buckets = []
        for bucket in chunk:
            if bucket["rank"] == group_rank:
                tensor = _pack_state_dict_bucket(bucket, state_dict)
            else:
                tensor = paddle.empty([bucket["numel"]], dtype=bucket["dtype"])
            gpu_buckets.append((bucket, tensor))

        t1 = time.time()
        _broadcast_state_dict_chunk(gpu_buckets, group)

        t2 = time.time()
        for bucket, tensor in gpu_buckets:
            _unpack_state_dict_bucket(bucket, tensor, selected_keys, gathered)
        # Release the chunk before packing the next one; keeping the list alive
        # would hold every bucket of this chunk in device memory.
        del gpu_buckets

        t3 = time.time()
        pack_seconds += t1 - t0
        bcast_seconds += t2 - t1
        unpack_seconds += t3 - t2
        done_buckets += len(chunk)
        done_chunks += 1

        # This loop can run for minutes without emitting anything at large
        # nranks, which is indistinguishable from a hang. Report progress.
        if group_rank == 0 and (
            done_buckets == len(buckets) or t3 - last_log >= _STATE_DICT_BROADCAST_LOG_INTERVAL_SECONDS
        ):
            last_log = t3
            elapsed = t3 - start
            logger.info(
                f"broadcast progress {done_buckets}/{len(buckets)} buckets "
                f"({100.0 * done_buckets / len(buckets):.1f}%), chunks={done_chunks}, "
                f"elapsed={elapsed:.1f}s, "
                f"eta={elapsed * (len(buckets) - done_buckets) / done_buckets:.1f}s, "
                f"pack={pack_seconds:.1f}s, bcast={bcast_seconds:.1f}s, "
                f"unpack={unpack_seconds:.1f}s, group_id={group.id}"
            )

    if group_rank == 0 and buckets:
        logger.info(
            f"broadcast done: {len(buckets)} buckets, {len(meta_list)} tensors, "
            f"total={time.time() - start:.1f}s, pack={pack_seconds:.1f}s, "
            f"bcast={bcast_seconds:.1f}s, unpack={unpack_seconds:.1f}s, "
            f"nranks={group.nranks}, group_id={group.id}"
        )

    assert not state_dict
    return OrderedDict((k, gathered[k]) for k, _ in meta_list if k in selected_keys)


def _all_gather_state_dict(state_dict, filter_func, group):
    remote_state_dict_keys = [k for k in state_dict.keys() if not filter_func(k)]
    tmp_state_dict = OrderedDict()
    for k in remote_state_dict_keys:
        tmp_state_dict[k] = state_dict[k]
        state_dict.pop(k)
    tmp_state_dict = all_gather_state_dict(tmp_state_dict, filter_func, group)
    for (k, v) in tmp_state_dict.items():
        state_dict[k] = v
    return state_dict


def get_moe_sharding_group(hcg=None):
    if hcg is None:
        hcg = fleet.get_hybrid_communicate_group()
    if hasattr(hcg, "get_moe_sharding_parallel_group"):
        return hcg.get_moe_sharding_parallel_group()
    else:
        return None


def get_param_sharding_group(param, hcg=None):
    if hcg is None:
        hcg = fleet.get_hybrid_communicate_group()
    default_group = hcg.get_sharding_parallel_group()
    ep_sharding_group = get_moe_sharding_group(hcg)

    if not hasattr(param, "color"):
        return default_group
    color = getattr(param, "color")
    if isinstance(color, dict):
        group = color.get("group", default_group)
        assert group is default_group or group is ep_sharding_group, f"unsupported group: {group}"
        return group
    else:
        return default_group
