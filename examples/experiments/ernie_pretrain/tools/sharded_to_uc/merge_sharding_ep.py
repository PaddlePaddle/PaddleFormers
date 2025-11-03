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

import os
import subprocess

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import itertools
import json
import logging
import pickle
import re
import time
from multiprocessing import Pool
from shlex import quote

import numpy as np
import paddle

logger = logging.getLogger(__name__)

LR_KEY = "LR_Scheduler"
MASTER_WEIGHT_KEY = "master_weights"
MOMENT1_KEY = "moment1"
MOMENT2_KEY = "moment2"
BETA1_POW_KEY = "beta1_pow_acc"
BETA2_POW_KEY = "beta2_pow_acc"
FP32_MASTER_WEIGHT_SUFFIX = "fp32_master_0"
NAME_MAPPING_KEY = "StructuredToParameterName@@"

USEFUL_FILES = [
    "added_tokens.json",
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "tokenizer.model",
]


class Timer:
    def __init__(self, name="name"):
        self.name = name

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.perf_counter()
        print(f"{self.name}Time consumed: {end - self.start:.6f} s")


def strtobool(s):
    s_lower = s.lower()
    if s_lower in ["1", "on", "true", "y", "yes"]:
        return True
    elif s_lower in ["0", "off", "false", "n", "no"]:
        return False
    else:
        raise ValueError(s)


def execute_cmd(cmd, ignore_error=False):
    if not isinstance(cmd, str):
        cmd = " ".join([quote(str(c)) for c in cmd])
    exitcode = os.system(cmd)
    if not ignore_error:
        assert exitcode == 0, f"Exitcode: {exitcode}, Command: {cmd}"
    return exitcode


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mp_rank", type=int)
    parser.add_argument("--base_path", type=str)
    parser.add_argument("--output_dir_path", type=str, default="./merged_models")
    parser.add_argument("--include_opt_state", type=strtobool, default=False)
    parser.add_argument("--ignore_padding_nonzero", type=strtobool, default=False)

    parser.add_argument("--node_rank", type=int, default=0)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--dst_mp_degree", type=int, default=1)

    return parser.parse_args()


def save_ckpt(ckpt, save_dir, rank_info, mp_degree, pp_degree=0, ep_degree=0, is_opt=False):
    if is_opt:
        prefix = "optimizer"
        suffix = "pdopt"
    else:
        prefix = "model_state"
        suffix = "pdparams"

    names = ["tp", "pp", "ep"]
    degrees = [mp_degree, pp_degree, ep_degree]
    mid = ""
    for i, rank in enumerate(rank_info):
        if degrees[i] > 1:
            mid += f".{names[i]}{rank:02d}"

    save_path = prefix + mid + "." + suffix

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_path)
    print(f"Saving save_path = {save_path} ...")
    start_t = time.time()
    with open(save_path, "wb") as f:
        pickle.dump(ckpt, f)
    end_t = time.time()
    print(f"Saved save_path = {save_path} cost_time = {end_t - start_t} is_opt = {is_opt}")


class Client:
    def __init__(self, args, base_path, nproc_per_node=8, nnodes=1, node_rank=0):
        self.args = args
        self.base_path = base_path
        self.nproc_per_node = nproc_per_node
        self.meta, self.config = self.load_model_meta()
        with open("model_meta.json", "w") as f:
            json.dump(self.meta, f, indent=2)
        parallel_config = self.meta["parallel_config"]
        self.mp_degree = parallel_config["mp_degree"]
        self.pp_degree = parallel_config["pp_degree"]
        self.sharding_degree = parallel_config["sharding_degree"]
        self.moe_sharding_degree = -1
        self.ep_degree = -1
        self.nnodes = nnodes
        self.node_rank = node_rank
        assert self.nnodes == 1
        assert self.node_rank == 0
        self.pp_stage_num_per_node = [self.pp_degree // self.nnodes] * self.nnodes
        self.pp_stage_num_per_node[-1] = self.pp_degree - ((self.nnodes - 1) * (self.pp_degree // self.nnodes))
        self.num_attention_heads = self.config["num_attention_heads"]
        self.num_key_value_heads = self.config["num_key_value_heads"]
        if "num_merge_attn_heads" in self.config:
            self.num_merge_attn_heads = self.config["num_merge_attn_heads"]

        self.stage_num_prefix = []
        current_sum = 0
        for i, num in enumerate(self.pp_stage_num_per_node):
            current_sum += num
            self.stage_num_prefix.append(current_sum)

        assert args.dst_mp_degree == 1
        self.dst_ep_degree = args.dst_mp_degree
        self.dst_mp_degree = args.dst_mp_degree
        if "moe_sharding_degree" in parallel_config and "ep_degree" in parallel_config:
            self.moe_sharding_degree = parallel_config["moe_sharding_degree"]
            self.ep_degree = parallel_config["ep_degree"]
            (
                self.parallel_2_node_map,
                self.node_2_parallel_map,
                self.pp_stage_2_nodes_map,
            ) = self._gen_node_id_map()
            self.num_experts_per_rank = self._get_num_experts_per_rank()
            self.num_experts = self.num_experts_per_rank * self.ep_degree

        self.num_process = 20

    def _get_expert_param_shape(self, meta):
        expert_param_shape = {}
        for s_name, meta_info in meta["sharding_metas"]["tp00_pp00_ep00"]["param_meta"].items():
            if "mlp.experts" in s_name:
                parts = s_name.split(".")
                suffix = ".".join(parts[-2:])
                expert_param_shape[suffix] = meta_info[0]
        return expert_param_shape

    def _expert_id(self, s_name):
        pattern = r"(\d+\.mlp\.experts\.)(\d+)"
        matches = re.findall(pattern, s_name)
        local_expert_id = -1
        if matches:
            local_expert_id = int(matches[0][1])
        return local_expert_id

    def _global_expert_id(self, local_id, ep_rank):
        return ep_rank * self.num_experts_per_rank + local_id

    def _get_num_experts_per_rank(self):
        rank_suffix = "tp00_pp00"
        if self.ep_degree > 1:
            rank_suffix += "_ep00"
        structure_name_mapping = self.meta["sharding_metas"][rank_suffix]["structure_name_mapping"]

        max_local_expert_id = -1
        for s_name in structure_name_mapping.keys():
            max_local_expert_id = max(max_local_expert_id, self._expert_id(s_name))
        return max_local_expert_id + 1

    def _gen_node_id_map(self):
        parallel_2_node_map = {}  # (sd, pp, mp)  -> node_id
        node_2_parallel_map = {}  # node_id       -> (sd, pp, mp)
        pp_stage_2_nodes_map = {}  # pp_stage      -> node_id set

        for pp in range(self.pp_degree):
            pp_stage_2_nodes_map[pp] = set()

        for sd, pp, mp in itertools.product(range(self.sharding_degree), range(self.pp_degree), range(self.mp_degree)):
            if self.ep_degree > 1:
                num_mp_per_ep = self.ep_degree // self.mp_degree
                moe_sd = sd // num_mp_per_ep
                sd_idx = sd % num_mp_per_ep
                ep = sd_idx * self.mp_degree + mp
                node_rank = (
                    moe_sd * self.pp_degree * self.ep_degree + pp * self.ep_degree + ep
                ) // self.nproc_per_node
            else:
                node_rank = (sd * self.pp_degree * self.mp_degree + pp * self.mp_degree + mp) // self.nproc_per_node
            parallel_2_node_map[(sd, pp, mp)] = node_rank
            pp_stage_2_nodes_map[pp].add(node_rank)
            if node_rank in node_2_parallel_map:
                node_2_parallel_map[node_rank].append((sd, pp, mp))
            else:
                node_2_parallel_map[node_rank] = [(sd, pp, mp)]

        return parallel_2_node_map, node_2_parallel_map, pp_stage_2_nodes_map

    def _modify_expert_id(self, s_name, new_id):
        pattern = r"(layers\.\d+\.mlp\.experts\.)(\d+)"

        def _repl(match):
            return f"{match.group(1)}{new_id}"

        new_s_name = re.sub(pattern, _repl, s_name)

        return new_s_name

    def merge_and_save(
        self,
        mp_rank,
        save_dir,
        include_opt_state=False,
        ignore_sharding_padding_nonzero=False,
    ):
        if self.ep_degree > 1:
            parallel_2_ckpt_map = {}
            for local_stage in range(self.pp_stage_num_per_node[self.node_rank]):
                pp_rank = (
                    local_stage + self.stage_num_prefix[self.node_rank - 1] if self.node_rank > 0 else local_stage
                )
                logger.warning(f"loading ckpt for pp stage {pp_rank}")
                parallel_2_ckpt_map.update(self._read_all_ckpts_by_pp_stage(pp_rank, include_opt_state))

            dense_params = self._merge_sharding_for_dense_params(parallel_2_ckpt_map, ignore_sharding_padding_nonzero)
            replicated_dnese_params, names_dense_params_to_reshard = self._replicate_dense_params(dense_params)

            expert_params = self._merge_sharding_for_expert_params(
                parallel_2_ckpt_map, ignore_sharding_padding_nonzero
            )
            expert_params = self._extend_ep_degree_for_expert_params(expert_params, self.dst_ep_degree)

            final_ckpts = self._get_final_ckpts(
                dense_params,
                replicated_dnese_params,
                names_dense_params_to_reshard,
                expert_params,
                self.dst_mp_degree,
            )

            save_ckpt_args = []
            for rank_info, param_ckpts in final_ckpts.items():
                save_ckpt_args.append(
                    (
                        param_ckpts,
                        save_dir,
                        rank_info,
                        self.dst_mp_degree,
                        self.pp_degree,
                    )
                )

            pool = Pool(self.num_process)
            try:
                pool.starmap(save_ckpt, save_ckpt_args)
            finally:
                pool.close()
                pool.join()

        else:
            if mp_rank is None:
                mp_ranks = range(self.mp_degree)
            else:
                assert mp_rank < self.mp_degree
                mp_ranks = [mp_rank]

            for mp_rank in mp_ranks:
                self._merge_and_save(
                    mp_rank,
                    save_dir,
                    include_opt_state,
                    ignore_sharding_padding_nonzero,
                )

        self.move_useful_file(self.args.output_dir_path)

    def _merge_sharding_for_dense_params(self, parallel_2_ckpt_map, ignore_sharding_padding_nonzero):
        merged_params = {}
        for local_stage, mp_rank in itertools.product(
            range(self.pp_stage_num_per_node[self.node_rank]), range(self.mp_degree)
        ):
            pp_rank = local_stage + self.stage_num_prefix[self.node_rank - 1] if self.node_rank > 0 else local_stage
            args = []
            for sharding_rank in range(self.sharding_degree):
                args.append((mp_rank, pp_rank, sharding_rank))

            dense_params = []
            for arg in args:
                ckpt = parallel_2_ckpt_map[arg]
                dense_params.append({s_name: v for s_name, v in ckpt.items() if "mlp.experts" not in s_name})
            param_ckpts = self._merge_sharding_param_ckpts(
                mp_rank,
                dense_params,
                ignore_sharding_padding_nonzero,
                check_type="dense",
            )
            merged_params[(mp_rank, pp_rank)] = param_ckpts
        return merged_params

    def _replicate_fused_param(self, local_params, indices_or_sections, concat_axis):
        splitted_params = []
        for p in local_params:
            splitted_params.append(np.split(p, indices_or_sections, axis=concat_axis))

        concated_params = []
        nparam = indices_or_sections if isinstance(indices_or_sections, int) else len(indices_or_sections)
        for i in range(nparam):
            param = [p[i] for p in splitted_params]
            concated_params.append(np.concatenate(param, axis=concat_axis))
        return np.concatenate(concated_params, axis=concat_axis)

    def _replicate_dense_params(self, dense_params):
        replicated_params = {}
        names_param_to_reshard = {}
        for local_stage in range(self.pp_stage_num_per_node[self.node_rank]):
            pp_rank = local_stage + self.stage_num_prefix[self.node_rank - 1] if self.node_rank > 0 else local_stage
            replicated_params[pp_rank] = {}
            names_param_to_reshard[pp_rank] = set()

            for s_name in dense_params[(0, pp_rank)].keys():
                replicated_params[pp_rank][s_name] = dense_params[(0, pp_rank)][s_name]

        return replicated_params, names_param_to_reshard

    def _merge_sharding_for_expert_params(self, parallel_2_ckpt_map, ignore_sharding_padding_nonzero):
        merged_params = {}
        num_mp_per_ep = self.ep_degree // self.mp_degree

        for local_stage, ep_rank in itertools.product(
            range(self.pp_stage_num_per_node[self.node_rank]), range(self.ep_degree)
        ):
            pp_rank = local_stage + self.stage_num_prefix[self.node_rank - 1] if self.node_rank > 0 else local_stage
            args = []
            mp_rank = ep_rank % self.mp_degree
            for moe_sd_rank in range(self.moe_sharding_degree):
                sharding_rank = moe_sd_rank * num_mp_per_ep + ep_rank // self.mp_degree
                args.append((mp_rank, pp_rank, sharding_rank))

            expert_params = []
            for arg in args:
                ckpt = parallel_2_ckpt_map[arg]
                expert_params.append({s_name: v for s_name, v in ckpt.items() if "mlp.experts" in s_name})
            param_ckpts = self._merge_sharding_param_ckpts(
                mp_rank, expert_params, ignore_sharding_padding_nonzero, ep_rank=ep_rank
            )
            merged_params[(ep_rank, pp_rank)] = param_ckpts
        return merged_params

    def _extend_ep_degree_for_expert_params(self, expert_params, dst_ep_degree):
        extended_experts_params = {}
        for local_stage, ep_rank in itertools.product(
            range(self.pp_stage_num_per_node[self.node_rank]), range(dst_ep_degree)
        ):
            pp_rank = local_stage + self.stage_num_prefix[self.node_rank - 1] if self.node_rank > 0 else local_stage
            extended_experts_params[(ep_rank, pp_rank)] = {}

        new_num_experts_per_rank = self.num_experts // dst_ep_degree
        for (ep_rank, pp_rank), ckpt in expert_params.items():
            for s_name, v in ckpt.items():
                if s_name == NAME_MAPPING_KEY:
                    continue
                local_expert_id = self._expert_id(s_name)
                global_expert_id = self._global_expert_id(local_expert_id, ep_rank)
                new_ep_rank = global_expert_id // new_num_experts_per_rank
                new_local_id = global_expert_id % new_num_experts_per_rank
                new_s_name = self._modify_expert_id(s_name, new_local_id)
                extended_experts_params[(new_ep_rank, pp_rank)][new_s_name] = v
        return extended_experts_params

    def _get_final_ckpts(
        self,
        dense_params,
        replicated_dense_params,
        names_dense_params_to_reshard,
        expert_params,
        dst_mp_degree,
        dst_ep_degree=None,
    ):
        final_ckpts = {}
        for local_stage in range(self.pp_stage_num_per_node[self.node_rank]):
            pp_rank = local_stage + self.stage_num_prefix[self.node_rank - 1] if self.node_rank > 0 else local_stage
            if dst_ep_degree is None:
                for tp in range(dst_mp_degree):
                    final_ckpts[(tp, pp_rank)] = {}
            else:
                for ep in range(dst_ep_degree):
                    tp = ep % dst_mp_degree
                    final_ckpts[(tp, pp_rank, ep)] = {}

        num_pieces = dst_mp_degree // self.mp_degree
        for local_stage in range(self.pp_stage_num_per_node[self.node_rank]):
            pp_rank = local_stage + self.stage_num_prefix[self.node_rank - 1] if self.node_rank > 0 else local_stage

            if dst_ep_degree is not None:
                for ep in range(dst_ep_degree):
                    tp = ep % dst_mp_degree
                    # set dense params
                    for s_name, v in replicated_dense_params[pp_rank].items():
                        final_ckpts[(tp, pp_rank, ep)][s_name] = v

                    # set expert params
                    for s_name, v in expert_params[(ep, pp_rank)].items():
                        final_ckpts[(tp, pp_rank, ep)][s_name] = v

            else:
                # set dense params
                for s_name, v in replicated_dense_params[pp_rank].items():
                    for dst_mp_rank in range(self.dst_mp_degree):
                        final_ckpts[(dst_mp_rank, pp_rank)][s_name] = v

                # set expert params
                for dst_mp_rank in range(dst_mp_degree):
                    for s_name, v in expert_params[(dst_mp_rank, pp_rank)].items():
                        final_ckpts[(dst_mp_rank, pp_rank)][s_name] = v

            # reshard embedding params
            for src_mp_rank in range(self.mp_degree):
                for s_name, v in dense_params[(src_mp_rank, pp_rank)].items():
                    if s_name not in names_dense_params_to_reshard[pp_rank]:
                        continue
                    for ipiece in range(num_pieces):
                        dst_mp_rank = src_mp_rank * num_pieces + ipiece
                        if "lm_head.weight" in s_name:
                            splitted_param = np.split(v, num_pieces, axis=1)
                        else:
                            splitted_param = np.split(v, num_pieces, axis=0)
                        for idx, piece in enumerate(splitted_param):
                            dst_mp_rank = src_mp_rank * num_pieces + idx
                            final_ckpts[(dst_mp_rank, pp_rank)][s_name] = piece

        return final_ckpts

    def _read_ckpts(self, args):
        start_t = time.time()
        pool = Pool(self.num_process)
        try:
            ckpts = pool.starmap(self.load_ckpt, args)
        finally:
            pool.close()
            pool.join()
        end_t = time.time()
        print(f"Load ckpt cost: {end_t - start_t}")
        param_ckpts = [ckpt[0] for ckpt in ckpts]
        return param_ckpts

    def _read_ckpt(self, mp, pp, sd, include_opt_state):
        return self.load_ckpt(mp, pp, sd, include_opt_state)[0]

    def _read_all_ckpts_by_pp_stage(self, pp_stage, include_opt_state=False):
        args = []
        ckpt_map = {}
        for rank in self.pp_stage_2_nodes_map[pp_stage]:
            for sd, pp, mp in self.node_2_parallel_map[rank]:
                args.append((mp, pp, sd, include_opt_state))

        print("... Arg number: ", len(args))
        pool = Pool(self.num_process)
        try:
            all_ckpts = pool.starmap(self._read_ckpt, args)
        finally:
            pool.close()
            pool.join()

        for arg, ckpt in zip(args, all_ckpts):
            mp, pp, sd, _ = arg
            ckpt_map[(mp, pp, sd)] = ckpt

        return ckpt_map

    def _merge_and_save(self, mp_rank, save_dir, include_opt_state, ignore_sharding_padding_nonzero):
        args = []
        for pp_rank in range(self.pp_degree):
            for sharding_rank in range(self.sharding_degree):
                args.append((mp_rank, pp_rank, sharding_rank, include_opt_state))

        start_t = time.time()
        pool = Pool(self.num_process)
        try:
            ckpts = pool.starmap(self.load_ckpt, args)
        finally:
            pool.close()
            pool.join()
        end_t = time.time()
        print(f"Load ckpt cost: {end_t - start_t}")

        param_ckpts = [ckpt[0] for ckpt in ckpts]
        param_ckpts = self._merge_pp_ckpts(args, param_ckpts, is_opt=False)
        param_ckpts = self._merge_sharding_param_ckpts(mp_rank, param_ckpts, ignore_sharding_padding_nonzero)
        save_ckpt(
            param_ckpts,
            save_dir,
            mp_rank,
            self.mp_degree,
            pp_degree=0,
            is_opt=False,
        )

        if not include_opt_state:
            for ckpt in ckpts:
                assert ckpt[1] is None
            assert False

        opt_ckpts = [ckpt[1] for ckpt in ckpts]
        opt_ckpts = self._merge_pp_ckpts(args, opt_ckpts, is_opt=True)
        opt_ckpts = self._merge_sharding_opt_ckpts(mp_rank, opt_ckpts, ignore_sharding_padding_nonzero)
        save_ckpt(
            opt_ckpts,
            save_dir,
            mp_rank,
            self.mp_degree,
            pp_degree=0,
            is_opt=True,
        )

    def _merge_pp_ckpts(self, rank_info, ckpts, is_opt):
        ret = [{} for _ in range(self.sharding_degree)]
        for (_, pp_rank, sharding_rank, _), ckpt in zip(rank_info, ckpts):
            d = ret[sharding_rank]
            for k, v in ckpt.items():
                if k == LR_KEY:
                    assert is_opt, "LRScheduler should not be in pdparams"
                else:
                    if k in d:
                        print(f"Duplicate keys found: {k}")
                        if not np.array_equal(v, d[k]):
                            print(f"The value of duplicate keys {k} are different")
                d[k] = v
        return ret

    def _get_param_meta(self, mp_rank, ep_rank=None):
        param_meta = {}
        mapping = {}
        for pp_rank in range(self.pp_degree):
            key = f"tp{mp_rank:02d}_pp{pp_rank:02d}"
            if ep_rank is not None:
                key += f"_ep{ep_rank:02d}"
            else:
                key += f"_ep{mp_rank:02d}"
            tmp_meta = self.meta["sharding_metas"][key]["param_meta"]
            for k, m in tmp_meta.items():
                assert k not in param_meta or "embed" in k, k
                param_meta[k] = m

            tmp_mapping = self.meta["sharding_metas"][key]["structure_name_mapping"]
            for k, m in tmp_mapping.items():
                assert k not in mapping or "embed" in k, k
                mapping[k] = m
        return param_meta, mapping

    def _merge_sharding_param_ckpts(
        self,
        mp_rank,
        ckpts,
        ignore_sharding_padding_nonzero,
        ep_rank=None,
        check_type=None,
    ):
        last_idx = {}
        merged_ckpt = {}
        last_idx = {}
        for i, ckpt in enumerate(ckpts):
            for k, v in ckpt.items():
                assert isinstance(v, np.ndarray)
                if k not in merged_ckpt:
                    merged_ckpt[k] = []
                merged_ckpt[k].append(v)
                if k in last_idx and last_idx[k] + 1 != i:
                    print(f"[WARN] {k} is not contiguous")
                    assert last_idx[k] + 1 == i, f"assertion failed for {k}, {last_idx[k]}, {i}"
                last_idx[k] = i

        param_meta, structure_name_mapping = self._get_param_meta(mp_rank, ep_rank)
        for k in merged_ckpt.keys():
            v = merged_ckpt[k]
            shape = param_meta.pop(k)[0]
            try:
                v = self._concat_crop_reshape(v, shape, k, ignore_sharding_padding_nonzero)
            except Exception:
                breakpoint()
                print("Error !!!!")
            merged_ckpt[k] = v

        assert NAME_MAPPING_KEY not in merged_ckpt
        return merged_ckpt

    def _concat_crop_reshape(self, arrs, shape, name, ignore_sharding_padding_nonzero):
        if len(arrs) > 1:
            arr = np.concatenate(arrs, axis=0)
        else:
            arr = arrs[0]
        numel = int(np.prod(shape))
        assert numel <= arr.size, f"{numel} vs {arr.size}"
        if numel < arr.size:
            padding = arr[numel:]
            if not np.all(padding == 0):
                if ignore_sharding_padding_nonzero:
                    print(f"[WARN] Non-zero value for {name}")
                else:
                    print(f"Non-zero value for {name}: ", padding.tolist())
                    raise ValueError("Padding non-zero error")

        arr = arr[:numel].reshape(shape)
        return arr

    def _get_opt_state_key_and_type(self, name):
        found = None
        for pattern in [MOMENT1_KEY, MOMENT2_KEY, BETA1_POW_KEY, BETA2_POW_KEY]:
            if pattern in name:
                assert found is None, name
                found = pattern
        assert found is not None, name

        refined_name = None
        for has_master_weight in [True, False]:
            if has_master_weight:
                prefix = f"_{FP32_MASTER_WEIGHT_SUFFIX}_{found}_"
            else:
                prefix = f"_{found}_"
            if prefix not in name:
                continue
            idx = name.find(prefix)
            left = name[idx + len(prefix) :]
            assert left.isdigit(), f"{left} in {name} {found}"
            refined_name = name[:idx]
            break

        assert refined_name is not None, name
        return found, refined_name

    def _merge_sharding_opt_ckpts(self, mp_rank, ckpts, ignore_sharding_padding_nonzero):
        param_meta, structure_name_mapping = self._get_param_meta(mp_rank)
        p_name_shape_mapping = {}
        for st_name, p_name in structure_name_mapping.items():
            shape = param_meta.pop(st_name)[0]
            assert p_name not in p_name_shape_mapping, f"{st_name} -> {p_name}"
            p_name_shape_mapping[p_name] = shape

        master_weights = {}
        lr = None
        accs = {
            MOMENT1_KEY: {},
            MOMENT2_KEY: {},
            BETA1_POW_KEY: {},
            BETA2_POW_KEY: {},
        }

        for sharding_rank, ckpt in enumerate(ckpts):
            sub_mw = ckpt.pop(MASTER_WEIGHT_KEY, {})
            cur_lr = ckpt.pop(LR_KEY, None)
            if lr is None:
                lr = cur_lr
            for k, v in sub_mw.items():
                pk = v[0]
                v = v[1]
                assert isinstance(v, np.ndarray), v

                if k not in master_weights:
                    master_weights[k] = [(sharding_rank, pk, v)]
                else:
                    prev_sharding_rank = master_weights[k][-1][0]
                    prev_pk = master_weights[k][-1][1]
                    assert prev_sharding_rank + 1 == sharding_rank, f"{k}: {prev_sharding_rank} vs {sharding_rank}"
                    assert prev_pk == pk, f"{k}: {prev_pk} vs {pk}"
                    master_weights[k].append((sharding_rank, pk, v))

            for k, v in ckpt.items():
                if isinstance(v, (list, tuple)):
                    assert v[0] == k, f"{v[0]} vs {k}"
                    v = v[1]
                assert isinstance(v, np.ndarray), v

                opt_state_key, _ = self._get_opt_state_key_and_type(k)
                assert opt_state_key in accs, opt_state_key
                sub_accs = accs[opt_state_key]
                if opt_state_key in [BETA1_POW_KEY, BETA2_POW_KEY]:
                    if k not in sub_accs:
                        sub_accs[k] = (k, v)
                    else:
                        prev_acc_value = sub_accs[k][1]
                        np.testing.assert_array_equal(prev_acc_value, v, err_msg=k)
                else:
                    if k not in sub_accs:
                        sub_accs[k] = [(sharding_rank, v)]
                    else:
                        prev_sharding_rank = sub_accs[k][-1][0]
                        assert prev_sharding_rank + 1 == sharding_rank, f"{k}: {prev_sharding_rank} vs {sharding_rank}"
                        sub_accs[k].append((sharding_rank, v))

        tmp_master_weights = {}
        for k, vs in master_weights.items():
            pk = vs[0][1]
            assert k in p_name_shape_mapping, k
            shape = p_name_shape_mapping[k]
            vs = [v[-1] for v in vs]
            v = self._concat_crop_reshape(vs, shape, pk, ignore_sharding_padding_nonzero)
            tmp_master_weights[k] = (pk, v)
        master_weights = tmp_master_weights

        tmp_accs = {
            MOMENT1_KEY: {},
            MOMENT2_KEY: {},
            BETA1_POW_KEY: accs.pop(BETA1_POW_KEY),
            BETA2_POW_KEY: accs.pop(BETA2_POW_KEY),
        }
        for moment_key in [MOMENT1_KEY, MOMENT2_KEY]:
            sub_accs = accs[moment_key]
            for key, vs in sub_accs.items():
                _, pk = self._get_opt_state_key_and_type(key)
                assert pk in p_name_shape_mapping, f"{key} {pk}"
                vs = [v[-1] for v in vs]
                shape = p_name_shape_mapping[pk]
                v = self._concat_crop_reshape(vs, shape, key, ignore_sharding_padding_nonzero)
                tmp_accs[moment_key][key] = (key, v)
        accs = tmp_accs

        final_ckpts = {MASTER_WEIGHT_KEY: master_weights}
        if lr is not None:
            final_ckpts[LR_KEY] = lr
        for _, sub_accs in accs.items():
            for k, v in sub_accs.items():
                assert k not in final_ckpts, k
                final_ckpts[k] = v
        return final_ckpts

    def _cal_ep_rank(self, sd_rank, mp_rank):
        num_mp_per_ep = self.ep_degree // self.mp_degree
        sd_idx = sd_rank % num_mp_per_ep
        return sd_idx * self.mp_degree + mp_rank

    def load_ckpt(self, mp_rank, pp_rank, sharding_rank, include_opt_state):
        start_t = time.time()

        rank_file_suffix = self.weight_suffix(mp_rank, pp_rank, sharding_rank)
        opt_path = f"optimizer.{rank_file_suffix}.pdopt"
        param_path = f"model_state.{rank_file_suffix}.pdparams"
        with open(os.path.join(self.base_path, opt_path), "rb") as f:
            opt = pickle.load(f)
        master_weights = opt[MASTER_WEIGHT_KEY]
        if not include_opt_state:
            opt = None

        with open(os.path.join(self.base_path, param_path), "rb") as f:
            params = pickle.load(f)

        if self.ep_degree > 1:
            ep_rank = self._cal_ep_rank(sharding_rank, mp_rank)
            rank_suffix = f"tp{mp_rank:02d}_pp{pp_rank:02d}_ep{ep_rank:02d}"
        else:
            rank_suffix = f"tp{mp_rank:02d}_pp{pp_rank:02d}"
        s2p_mapping = self.meta["sharding_metas"][rank_suffix]["structure_name_mapping"]
        p2s_mapping = {}
        for s_name, p_name in s2p_mapping.items():
            if p_name not in p2s_mapping:
                p2s_mapping[p_name] = []
            p2s_mapping[p_name].append(s_name)
        param_meta = self.meta["sharding_metas"][rank_suffix]["param_meta"]

        ret_params = {}
        if master_weights is not None:
            for k, v in master_weights.items():
                s_names = p2s_mapping[k]
                if isinstance(v, (list, tuple)):
                    v = v[1]
                assert isinstance(v, np.ndarray)
                for s_name in s_names:
                    dtype = paddle.dtype(param_meta[s_name][1])
                    v = paddle.to_tensor(v).astype(dtype).numpy()
                    ret_params[s_name] = v

        for k, v in params.items():
            if k == NAME_MAPPING_KEY:
                continue
            if isinstance(v, (list, tuple)):
                v = v[1]
            assert isinstance(v, np.ndarray)
            assert k not in ret_params, k
            ret_params[k] = v

        end_t = time.time()
        print(
            f"Loaded time = {end_t - start_t} mp_rank = {mp_rank} "
            f"pp_rank = {pp_rank} sharding_rank = {sharding_rank} "
            f"from: {opt_path} {param_path}"
        )
        return ret_params, opt

    def weight_suffix(self, mp_rank, pp_rank, sharding_rank):
        suffix = []
        if self.mp_degree > 1:
            assert mp_rank < self.mp_degree
            suffix.append(f"tp{mp_rank:02d}")
        if self.pp_degree > 1:
            assert pp_rank < self.pp_degree
            suffix.append(f"pp{pp_rank:02d}")
        if self.sharding_degree > 1:
            assert sharding_rank < self.sharding_degree
            suffix.append(f"shard{sharding_rank:02d}")
        return "_".join(suffix)

    def load_model_meta(self):
        with open(f"{self.base_path}/model_meta.json", "r") as f:
            meta_json = json.load(f)
        with open(f"{self.base_path}/config.json", "r") as f:
            config_json = json.load(f)
        return meta_json, config_json

    def move_useful_file(self, save_dir):
        for file in USEFUL_FILES:
            assert os.path.exists(f"{self.base_path}/{file}"), f"{self.base_path}/{file} not exist, please check"
            subprocess.run(
                ["cp", f"{self.base_path}/{file}", save_dir],
                capture_output=True,
                text=True,
            )


def merge_and_save(args):
    start_t = time.time()
    client = Client(args, args.base_path, nnodes=args.nnodes, node_rank=args.node_rank)
    client.merge_and_save(
        args.mp_rank,
        args.output_dir_path,
        args.include_opt_state,
        args.ignore_padding_nonzero,
    )
    end_t = time.time()
    print(f"Total time cost {end_t - start_t}")


if __name__ == "__main__":
    args = parse_args()
    merge_and_save(args)
