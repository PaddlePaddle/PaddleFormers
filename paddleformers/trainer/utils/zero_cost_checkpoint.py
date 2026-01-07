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

import atexit
import copy
import functools
import hashlib
import json
import multiprocessing
import os
import random
import time
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from dataclasses import replace
from enum import Enum

import numpy as np
import paddle
import paddle.autograd as imperative_base
import paddle.distributed as dist
from paddle.base import core
from paddle.distributed.communication.group import is_initialized
from paddle.distributed.fleet import fleet
from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer import (
    DygraphShardingOptimizer,
)
from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer.dygraph_sharding_optimizer import (
    DygraphShardingOptimizerV2,
)
from paddle.distributed.fleet.meta_parallel import PipelineLayer
from paddle.distributed.flex_checkpoint.dcp.metadata import (
    LocalTensorIndex,
    LocalTensorMetadata,
    Metadata,
)
from paddle.distributed.flex_checkpoint.dcp.save_state_dict import dedup_key_in_dict
from paddle.distributed.flex_checkpoint.dcp.sharded_weight import ShardedWeight
from paddle.distributed.flex_checkpoint.dcp.utils import (
    flatten_state_dict,
    merge_state_dict_metadata,
)
from paddle.incubate.tensor.manipulation import (
    async_offload_with_offset,
    create_async_load,
)
from paddle.optimizer.fusion_utils import FusionStorageHelper

from paddleformers.trainer.trainer_callback import TrainerCallback
from paddleformers.trainer.utils.sharding_io import GroupGetter
from paddleformers.utils.tools import paddle_device

from ...transformers.model_utils import (
    _add_variant,
    clean_model_class_name,
    get_parameter_dtype,
    unwrap_model,
    unwrap_optimizer,
)
from ...transformers.utils import device_guard
from ...utils.env import (
    CONFIG_NAME,
    EMA_STATE_DIC,
    MASTER_WEIGHT_DIC,
    MODEL_META_NAME,
    MODEL_STATE_DIC,
    OPTIMIZER_STATE_DIC,
    PADDLE_OPTIMIZER_NAME,
    PADDLE_WEIGHTS_NAME,
    PREFIX_CHECKPOINT_DIR,
    SCHEDULER_NAME,
    TRAINER_STATE_NAME,
    TRAINING_ARGS_NAME,
)
from ...utils.fault_tolerance import FC_DUMP_ERROR, PC_DUMP_ERROR
from ...utils.log import logger
from ...utils.pdc_sdk import FLASH_DEVICE
from . import reshard as reshard_util
from .reshard import (
    SHARDING_STRATEGY_V1,
    merge_model_state,
    split_model_state,
    split_opt_state,
)


def md5(tensor):
    """debug use"""
    numpy_array = tensor.numpy()
    array_bytes = numpy_array.tobytes()
    return hashlib.md5(array_bytes).hexdigest()


class ZCCTaskType(Enum):
    """
    TaskType defines the type of tasks that can be executed by the ZeroCostCheckpointWorker.
    """

    UPDATE = 0
    PREPARE = 1
    OFFLOAD = 2
    FINISH = 3
    SET_EMA_STATE_DICT = 5


class ZCCWorkerStatus(Enum):
    IDLE = 0
    OFFLOADING = 1
    DUMPING = 2
    ERROR = 3


def showmem(msg):
    return (
        f"{msg} mem_alloc: {paddle_device.memory_allocated():.3e}"
        f" Bytes/{paddle_device.max_memory_allocated():.3e} Bytes"
        f"mem_reserv: {paddle_device.memory_reserved():.3e} "
        f"Bytes/{paddle_device.max_memory_reserved():.3e} Bytes"
    )


# the funciotn that accept state dict as input can be decorated with this function
def sharded_state_dict_compatibility(func, *, return_sharded_state_dict=False):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        def should_convert(maybe_sharded_state_dict):
            all_shared_weights = all(isinstance(value, ShardedWeight) for value in maybe_sharded_state_dict.values())
            any_shared_weights = any(isinstance(value, ShardedWeight) for value in maybe_sharded_state_dict.values())
            logger.debug(f"all sharded weight {all_shared_weights}, any shared weight {any_shared_weights}")
            if not any_shared_weights:
                logger.debug("this is not a sharded state dict, no need to convert.")
                return False

            if any_shared_weights and (not all_shared_weights):
                logger.debug("this is a mixed state dict(normal and sharded), not support to convert.")
                return False
            logger.debug("this is a sharded state dict, will convert it to local tensor dict.")
            return True

        original_sharded_state_dict = {}
        # process args
        new_args = list(args)
        for idx, arg in enumerate(new_args):
            if not isinstance(arg, dict):
                continue
            if should_convert(arg):
                local_tensor_state_dict = {}
                for k, v in arg.items():
                    local_tensor_state_dict[k] = v.local_tensor

                original_sharded_state_dict.update(arg)
                new_args[idx] = local_tensor_state_dict

        # process kwargs
        for key, value in kwargs.items():
            if not isinstance(value, dict):
                continue
            if should_convert(value):
                local_tensor_state_dict = {}
                for k, v in value.items():
                    local_tensor_state_dict[k] = v.local_tensor

                kwargs[key] = local_tensor_state_dict
                original_sharded_state_dict.update(value)

        # original function
        result = func(*new_args, **kwargs)

        if return_sharded_state_dict:
            assert isinstance(result, dict), f"expected dict, but got {type(result)}"
            for k, v in result.items():
                sharded_sharded_weight = original_sharded_state_dict[k]
                sharded_sharded_weight.local_tensor = v
                result[k] = sharded_sharded_weight
        return result

    return wrapper


@sharded_state_dict_compatibility
def get_fused_param_mappings(optimizer, manipulated_state_dict):
    param_mappings = {}
    ipc_meta_mappings = {}
    index = 0
    sharding_comm_buffers = optimizer._comm_buffer_list
    for buffer in sharding_comm_buffers:
        ipc_meta_mappings[str(index)] = buffer.param_buffer_ipc_meta
        for k, v in manipulated_state_dict.items():
            logger.info(
                f"check vname: {v.name}; buffer._sharding_param_grad_view: {buffer._sharding_param_grad_view.keys()}"
            )
            if v.name in buffer._sharding_param_grad_view:
                assert k not in param_mappings, f"{k} has already been mapped, which is unexpected."
                param_meta = {}
                param_meta["buffer_index"] = str(index)
                param_meta["shape"] = v.shape
                param_meta["name"] = v.name
                param_meta["start"] = buffer._sharding_param_grad_view[v.name]._index
                param_meta["end"] = param_meta["start"] + v._numel()
                param_mappings[k] = param_meta
        index += 1
    assert len(manipulated_state_dict) == len(
        param_mappings
    ), f"manipulated state dict is not fully covered in param mappings, manipulated_state_dict:{manipulated_state_dict.keys()}, param_mappings:{param_mappings.keys()}"
    return param_mappings, ipc_meta_mappings


class ZeroCostCheckpointEMAProcessor:
    """
    生活在 ZCC Worker 里面的 EMA 处理模块.
    通过 `optimizer_fusion_storage_helper` 以及 `param_fusion_storage_helper` 获取主模型的参数
    """

    def __init__(self, optimizer_fusion_storage_helper, param_fusion_storage_helper, ema_coef):
        self.optimizer_fusion_storage_helper = optimizer_fusion_storage_helper
        self.param_fusion_storage_helper = param_fusion_storage_helper
        self.ema_coef = ema_coef
        (
            self.ema_buffer,
            self.ema_buffer_model_params,
            self.master_min_offset,
            self.master_max_offset,
        ) = self.build_ema_buffer()

    def status(self):
        if self.ema_buffer is None:
            return "[EMA buffer] not initizied"
        opt_md = md5(self.ema_buffer)
        param_md = {k: md5(v) for k, v in self.ema_buffer_model_params.items()}
        return f"[EMA buffer] opt:{opt_md}, param:{param_md}"

    @imperative_base.no_grad()
    def build_ema_buffer(self):
        logger.info("[ZCC EMA] build ema buffer")
        master_max_offset = max(
            self.optimizer_fusion_storage_helper.master_weights_meta.values(), key=lambda i: i["end"]
        )["end"]
        master_min_offset = min(
            self.optimizer_fusion_storage_helper.master_weights_meta.values(), key=lambda i: i["start"]
        )["start"]
        with device_guard("cpu"):
            ema_buffer = paddle.zeros(
                [master_max_offset - master_min_offset],
                dtype="float32",
            )
            # ema model params, only works on float32 model weights (aka, moe gates)
            ema_buffer_model_params = {
                k: paddle.zeros_like(cpu_buf)
                for k, (cuda_buf, cpu_buf) in self.param_fusion_storage_helper.inited_buffers.items()
                if cuda_buf.dtype == paddle.float32
            }
        logger.info(f"[ZCCworker] build buffer done:{ema_buffer.dtype} {ema_buffer.place}")
        return ema_buffer, ema_buffer_model_params, master_min_offset, master_max_offset

    def ema_reset(self):
        self.ema_buffer = None
        self.ema_buffer_modele_params = None

    @imperative_base.no_grad()
    def ema_accumulate(self, global_step, loss, zcc_ema_loss_threshold):
        """
        perform ema update : ` \alpha * EMA + (1-\alpha) + model`
        build `self.ema_buffer` if necessary
        when loss < threshold, do ema update
        """
        # logger.info(f'[ZCC EMA] wait all done, doing EMA w/ coef: {self.ema_coef}, status:{self.status()}')
        # do update: ema = alpha * ema + (1-alpha) * model
        logger.info(f"[ZCC EMA] accumulating, buffer type:{self.ema_buffer.place} {self.ema_buffer.dtype}")
        with device_guard("cpu"):
            cpu_master_weights = self.optimizer_fusion_storage_helper.cpu_buffer._slice(
                self.master_min_offset, self.master_max_offset
            ).cpu()
            if zcc_ema_loss_threshold is None or loss < zcc_ema_loss_threshold:
                self.ema_buffer = self.ema_coef * self.ema_buffer + (1 - self.ema_coef) * cpu_master_weights
                for index, ema_buf in self.ema_buffer_model_params.items():
                    _, cpu_buf = self.param_fusion_storage_helper.inited_buffers[index]
                    updated_ema = self.ema_coef * ema_buf + (1 - self.ema_coef) * cpu_buf
                    self.ema_buffer_model_params[index] = updated_ema
                logger.info(
                    f"[ZCC EMA] accmulating, buffer type:{self.ema_buffer.place} {self.ema_buffer.dtype}, done"
                )
            else:
                logger.info(
                    f"[ZCC EMA] accmulating SKIP for global_step:{global_step}, because loss:{loss} > threshold:{zcc_ema_loss_threshold}"
                )

    @imperative_base.no_grad()
    def ema_state_dict(self):
        assert self.optimizer_fusion_storage_helper is not None
        logger.info("[ZCC EMA] convert ema master weights state dict")
        with device_guard("cpu"):
            ema_state_dict = {}
            for k, tensor_meta in self.param_fusion_storage_helper.model_weights_metas.items():
                shape = tensor_meta["shape"]
                name = tensor_meta["name"]
                start = tensor_meta["start"]
                end = tensor_meta["end"]
                if tensor_meta["buffer_index"] not in self.ema_buffer_model_params:
                    continue  # non fp32 has no `self.ema_buffer_model_params`
                cpu_buffer = self.ema_buffer_model_params[tensor_meta["buffer_index"]]
                tensor = cpu_buffer._slice(start, end).clone()  # slice 出来的 tensor 在执行`paddle.save`会异常慢，此处必须clone
                tensor.get_tensor()._set_dims(shape)
                tensor.name = name
                ema_state_dict[k] = tensor
            ema_state_dict_master_weights = {}
            for k, meta in self.optimizer_fusion_storage_helper.master_weights_meta.items():
                s = meta["start"] - self.master_min_offset
                e = meta["end"] - self.master_min_offset
                t = self.ema_buffer._slice(s, e).clone()
                t.get_tensor()._set_dims(meta["shape"])
                t.name = meta["name"]
                ema_state_dict_master_weights[k] = t
            ema_state_dict["master_weights"] = ema_state_dict_master_weights
        return ema_state_dict

    def load_ema_state_dict(self, state_dict):
        for k, tensor_meta in self.param_fusion_storage_helper.model_weights_metas.items():
            logger.info(f"[ZCC EMA] load model weight key={k}")
            start = tensor_meta["start"]
            end = tensor_meta["end"]
            if tensor_meta["buffer_index"] not in self.ema_buffer_model_params:
                continue  # non fp32 has no `self.ema_buffer_model_params`
            if k in state_dict:
                cpu_buffer = self.ema_buffer_model_params[tensor_meta["buffer_index"]]
                tensor = state_dict[k].flatten()
                cpu_buffer[start:end] = tensor

        ema_master = state_dict["master_weights"]
        for k, meta in self.optimizer_fusion_storage_helper.master_weights_meta.items():
            logger.info(f"[ZCC EMA] load optimizer weight key={k}")
            s = meta["start"] - self.master_min_offset
            e = meta["end"] - self.master_min_offset
            if k in ema_master:  # state-dict is filtered
                self.ema_buffer[s:e] = ema_master[k].flatten()


class ParamFusionStorageHelper:
    def __init__(
        self,
        model_weights_metas,
        buffer_ipc_metas,
    ):
        self.async_loader = create_async_load()
        self.inited_buffers = {}
        self.all_param_numel = 0
        self.model_weights_metas = OrderedDict()
        self.current_offloaded_numel = 0
        self.reset_meta(
            model_weights_metas,
            buffer_ipc_metas,
        )
        self.tasks = []

    @imperative_base.no_grad()
    def reset_meta(
        self,
        model_weights_metas,
        buffer_ipc_metas,
    ):
        self.inited_buffers = {}
        self.all_param_numel = 0
        self.model_weights_metas = OrderedDict()
        if len(model_weights_metas) == 0:
            logger.info("No model states need to save in current worker")
            return

        for k, v in model_weights_metas.items():
            assert isinstance(v, dict), "model_weights_metas must be a dict"
            buffer_index = v["buffer_index"]
            if buffer_index not in self.inited_buffers.keys():
                buffer_tuple = self.init_buffer(buffer_ipc_metas[buffer_index])
                self.inited_buffers[buffer_index] = buffer_tuple
            v["start"] = int(v["start"])
            v["end"] = int(v["end"])
            v["logical_start"] = self.all_param_numel
            self.all_param_numel += v["end"] - v["start"]
            v["logical_end"] = self.all_param_numel
            self.model_weights_metas[k] = v

    def init_buffer(self, meta):
        cuda_buffer = paddle.to_tensor(paddle.base.core.LoDTensor._new_shared_cuda(meta))
        cpu_buffer = cuda_buffer.pin_memory()
        return (cuda_buffer, cpu_buffer)

    @imperative_base.no_grad()
    def sync_partial_param(self, numel_to_sync):
        assert (
            self.current_offloaded_numel + numel_to_sync <= self.all_param_numel
        ), f"numel_to_sync: {numel_to_sync}, current_offloaded_numel: {self.current_offloaded_numel}, all_param_numel: {self.all_param_numel}"
        next_offload_index = 0
        meta_keys_in_order = list(self.model_weights_metas.keys())
        for i, k in enumerate(meta_keys_in_order):
            if self.current_offloaded_numel >= self.model_weights_metas[k]["logical_end"]:
                continue
            next_offload_index = i
            break

        while numel_to_sync > 0:
            offloading_param_key = meta_keys_in_order[next_offload_index]
            offloading_param_meta = self.model_weights_metas[offloading_param_key]
            logical_offload_param_start = self.current_offloaded_numel
            logical_offload_param_end = min(
                offloading_param_meta["logical_end"], logical_offload_param_start + numel_to_sync
            )
            actual_offload_start = (
                logical_offload_param_start - offloading_param_meta["logical_start"]
            ) + offloading_param_meta["start"]
            actual_offload_end = (
                logical_offload_param_end - offloading_param_meta["logical_end"]
            ) + offloading_param_meta["end"]
            actual_offload_size = actual_offload_end - actual_offload_start
            current_param_buffer = self.inited_buffers[offloading_param_meta["buffer_index"]][0]
            current_param_cpu_buffer = self.inited_buffers[offloading_param_meta["buffer_index"]][1]
            task = async_offload_with_offset(
                src_tensor=current_param_buffer,
                dst_tensor=current_param_cpu_buffer,
                src_offset=actual_offload_start,
                dst_offset=actual_offload_start,
                offload_size=actual_offload_size,
                async_loader=self.async_loader,
            )
            self.tasks.append(task)
            self.current_offloaded_numel += actual_offload_size
            numel_to_sync -= actual_offload_size
            next_offload_index += 1

    def wait_all(self):
        if len(self.tasks) == 0:
            return
        last_task = self.tasks.pop(-1)
        while len(self.tasks) > 0:
            task = self.tasks.pop(0)
            task.cuda_wait()
        last_task.cpu_wait()
        self.current_offloaded_numel = 0

    def state_dict(self):
        state_dict = {}
        for k, v in self.model_weights_metas.items():
            state_dict[k] = self.restore_tensor_from_meta(v)
        return state_dict

    @imperative_base.no_grad()
    def restore_tensor_from_meta(self, tensor_meta):
        shape = tensor_meta["shape"]
        name = tensor_meta["name"]
        start = tensor_meta["start"]
        end = tensor_meta["end"]
        cpu_buffer = self.inited_buffers[tensor_meta["buffer_index"]][1]
        tensor = cpu_buffer._slice(start, end)
        tensor.get_tensor()._set_dims(shape)
        tensor.name = name
        return tensor


class ZeroCostCheckpointCallback(TrainerCallback):
    """
    call ZeroCostCheckpointManager during training in following order:

    on_step_end:
        *  call get_idle_worker_for_saving, set manager.current_worker
        *  call maybe_update_zcc_worker

    * on_substep_end(call `gradient_accumulate` times): call zcc_pipeline_hook (in non-pp model)
    * (when offload done, dump model)
    on_optimizer_begin: call sync_offload_status, unset set manager.current_worker
        maybe optimizer reload
        maybe optimizer offload
    """

    def __init__(self, args, zcc_manager, timer, sharding_io):
        self.manager = zcc_manager
        self.runtime_timer = timer
        self.user_file_list = []
        self.manipulated_state_dict = None
        self.manipulated_config_to_save = None
        self.manipulated_weight_suffix = None
        self.model_meta = None
        self.sharding_io = sharding_io
        self.zcc_ema_interval = args.zcc_ema_interval

    def on_substep_end(self, args, state, control, **kwargs):
        self.manager.zcc_pipeline_hook(0)  # only works in non-pp model

    def on_optimizer_begin(self, args, state, control, **kwargs):
        if args.enable_zero_cost_checkpoint and self.manager.current_worker is not None:
            logger.info("[ZCC manager] Start syncing checkpoints")
            assert self.manager.global_step != 0, "global_step should set, when calling `on_optimizer_begin`"
            self.manager.sync_offload_status()
            logger.info("[ZCC manager] Synced checkpoints.")

    def on_step_end(self, args, state, control, model, lr_scheduler, optimizer, **kwargs):
        if not control.should_save:
            if args.zcc_save_ema_coef is not None and state.global_step % self.zcc_ema_interval == 0:
                self.maybe_update_zcc_worker(args, model, optimizer, state.global_step)
                self.manager.get_idle_worker_for_saving(((None, None), (None, state, None)))  # prepare for dumping
        else:
            self.runtime_timer.start("checkpoint saving time")
            self.maybe_update_zcc_worker(args, model, optimizer, state.global_step)
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
            save_infos = self._get_save_infos_based_on_steps(state, args, checkpoint_folder)
            non_cached_objects = (lr_scheduler.state_dict(), state, self.get_rng_states(args))
            self.manager.get_idle_worker_for_saving((save_infos, non_cached_objects))
            self.runtime_timer.stop()
            if not isinstance(model, PipelineLayer):
                self.manager.zcc_pipeline_hook(0)

    def get_rng_states(self, args):
        if not args.save_rng_states:
            return None
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cuda": paddle.get_rng_state(),
            "cpu": paddle.framework.core.default_cpu_generator().get_state(),
            "world_size": args.world_size,
        }
        if args.use_hybrid_parallel:
            rng_states[
                "hybrid_parallel_rng_state_tracker"
            ] = dist.fleet.meta_parallel.get_rng_state_tracker().get_states_tracker()
        return rng_states

    def _get_save_infos_based_on_steps(self, state, args, checkpoint_folder):
        flash_device_checkpoint_dir = None
        persistent_checkpoint_dir = None
        if args.flash_device_save_steps > 0 and state.global_step % args.flash_device_save_steps == 0:
            flash_device_checkpoint_dir = os.path.join(FLASH_DEVICE, checkpoint_folder)
        if args.save_steps > 0 and state.global_step % args.save_steps == 0:
            persistent_checkpoint_dir = os.path.join(args.output_dir, checkpoint_folder)
        return (flash_device_checkpoint_dir, persistent_checkpoint_dir)

    def _pack_dynamic_objects(self):
        dynamic_objecs = {}
        dynamic_objecs["optimizer_states_meta"] = self.optimizer_states_meta
        dynamic_objecs["model_states_meta"] = self.model_states_meta
        dynamic_objecs["optimizer_states_name_path"] = self.optimizer_states_name_path
        dynamic_objecs["model_states_name_path"] = self.model_states_name_path

        return dynamic_objecs

    def _pack_static_objects(self, args):
        static_objects = {}
        static_objects["model_config"] = self.manipulated_config_to_save
        static_objects["training_args"] = args
        static_objects["model_meta"] = self.model_meta
        static_objects["user_file"] = self.user_file_list

        return static_objects

    def maybe_update_zcc_worker(self, args, model, optimizer, global_step):
        # logger.info(f"check should update :{optimizer.fused_buffer_version} vs {self.manager.cache_version}")
        if optimizer.fused_buffer_version == self.manager.cache_version:
            return
        logger.info("ZCC checkpoint workers need upgrade.")
        self._cache_meta_for_sharded_save(model, optimizer)
        param_mappings, ipc_meta_mappings = get_fused_param_mappings(optimizer, self.manipulated_state_dict)
        self.optimizer_states_meta = (
            optimizer.fused_states_accumulators_meta,
            optimizer.fused_states_master_weights_meta,
            None,
            optimizer.fused_states_buffer_ipc_meta,
        )
        self.model_states_meta = (param_mappings, ipc_meta_mappings)
        self.optimizer_states_name_path = _add_variant(PADDLE_OPTIMIZER_NAME, args.optimizer_name_suffix)
        self.model_states_name_path = _add_variant(PADDLE_WEIGHTS_NAME, self.manipulated_weight_suffix)

        dynamic_objects = self._pack_dynamic_objects()
        static_objects = self._pack_static_objects(args)

        self.manager.update_zcc_workers(optimizer.fused_buffer_version, dynamic_objects, static_objects, global_step)
        logger.info(f"[ZCC Callback] after first update:{optimizer.fused_states_buffer_ipc_meta}")

    def _cache_meta_for_sharded_save(self, model, unused):
        logger.info("Start caching metas for sharded save...")
        (
            self.manipulated_state_dict,
            self.manipulated_config_to_save,
            self.manipulated_weight_suffix,
        ) = self.sharding_io.manipulate_state_dict_and_config(model, merge_tensor_parallel=False)
        logger.info("Cache manipulated static dict done.")
        if self.manipulated_config_to_save is None:
            model_to_save = unwrap_model(model)
            dtype = get_parameter_dtype(model_to_save)
            model_to_save.config.dtype = str(dtype).split(".")[1]
            self.manipulated_config_to_save = copy.deepcopy(model_to_save.config)
            self.manipulated_config_to_save.architectures = [clean_model_class_name(model_to_save.__class__.__name__)]
            self.manipulated_config_to_save = self.manipulated_config_to_save.to_json_string(use_diff=True)
            logger.info("Cache manipulated model config done")
        self.model_meta = self.sharding_io.gather_distributed_model_meta()
        logger.info("Cache distributed model meta done.")


class ZeroCostCheckpointManager:
    def __init__(
        self,
        worker_num,
        pipeline_hooks_capacity,
        capacity_usage,
        use_expert_parallel,
        ema_coef=None,
        zcc_worker_class=None,
        save_hf_steps=-1,
    ):
        assert worker_num > 0, "worker_num must be greater than 0"
        assert capacity_usage <= 1.0, "capacity_usage must be less than or equal to 1.0"
        self.cache_version = 0
        self.worker_num = worker_num
        self.workers = []
        self.processes = []
        self.current_worker = None
        self.global_step = 0  # set `on-step-end`
        self.device_id = int(os.getenv("FLAGS_selected_gpus"))
        self.pipeline_hooks_steps = max(int(pipeline_hooks_capacity * capacity_usage), 1)
        logger.info(
            f"[ZCC manager] pipeline hooks capacity: {pipeline_hooks_capacity}; "
            f"pipeline hooks steps for offloading: {self.pipeline_hooks_steps} "
            f"ema coefficient: {ema_coef} "
        )
        self.current_pipeline_hook_step = 0
        ctx = multiprocessing.get_context("spawn")
        assert hasattr(fleet, "_hcg"), "ZeroCostCheckpoint Only support `use_hybrid_parallel`"
        if zcc_worker_class is None:
            zcc_worker_class = ZeroCostCheckpointWorker
        for i in range(worker_num):
            worker_task_queue = ctx.Queue()
            worker_status = ctx.Value("i", ZCCWorkerStatus.IDLE.value)
            worker_version = ctx.Value("i", 0)
            worker_step = ctx.Value("i", 0)
            worker = zcc_worker_class(
                i,
                self.device_id,
                dist.get_rank(),
                self.pipeline_hooks_steps,
                worker_task_queue,
                worker_status,
                worker_step,
                worker_version,
                use_expert_parallel,
                fleet.get_hybrid_communicate_group().get_data_parallel_rank(),
                fleet.get_hybrid_communicate_group().get_model_parallel_rank(),
                fleet.get_hybrid_communicate_group()._get_pipe_parallel_id(),
                fleet.get_hybrid_communicate_group().get_sharding_parallel_rank(),
                ema_coef,
                save_hf_steps,
            )
            p = ctx.Process(target=worker_loop, args=(worker,))
            p.start()
            self.workers.append(worker)
            self.processes.append(p)
        self.ready_to_save = False
        atexit.register(self.terminate_workers)

    def set_ema_state_dict(self, path):
        logger.info(f"[ZCC manager] setting EMA state dict: {path}")
        for worker in self.workers:
            assert worker.status.value == ZCCWorkerStatus.IDLE.value, "[ZCC manager] worker should be idle, when "
            worker.task_queue.put((ZCCTaskType.SET_EMA_STATE_DICT, path))
        logger.info("[ZCC manager] done setting EMA state dict")

    def update_zcc_workers(self, new_version, dynamic_objecs, static_object, global_step):
        self.report_error_worker()
        self.cache_version = new_version
        self.global_step = global_step
        assert self.current_worker is None, "[ZCC manager] current_worker must be None"
        task = (ZCCTaskType.UPDATE, [self.cache_version, dynamic_objecs, static_object])
        logger.info(f"[ZCC manager] updating zcc workers, version: {self.cache_version}")
        for worker in self.workers:
            worker.task_queue.put(task)
        logger.info("[ZCC manager] waiting workers update done")
        for worker in self.workers:
            while worker.version.value != self.cache_version:
                logger.info(
                    f"[ZCC manager] waiting worker{worker.worker_id} update. worker version: "
                    f"{worker.version.value}, expected version: {self.cache_version} "
                    f"step:{worker.global_step.value}"
                )
                time.sleep(1)
            logger.info(
                f"[ZCC manager] worker{worker.worker_id} updated. worker version: {worker.version.value}, "
                f"expected version: {self.cache_version} "
                f"global_step={worker.global_step.value} "
            )
        logger.info("[ZCC manager] update all zcc workers done")
        self.ready_to_save = True

    def get_idle_worker_for_saving(self, save_infos_and_non_cached_objects=None):
        """
        if `save_infos_and_non_cached_objects` is None, do offload without dumping.
        """
        self.report_error_worker()
        assert self.current_worker is None, "[ZCC manager] current_worker must be None"
        found_worker = False
        while True:
            for worker in self.workers:
                if worker.status.value == ZCCWorkerStatus.IDLE.value:
                    self.current_worker = worker
                    found_worker = True
                    break
            if found_worker:
                break
            logger.info(
                "[ZCC manager] Waiting for idle worker..., consider increase `save-step` or `global-batch-size`"
            )
            time.sleep(1)
        task = (ZCCTaskType.PREPARE, save_infos_and_non_cached_objects)
        logger.info(
            f"[ZCC manager] before putting task for prepare, dumping={save_infos_and_non_cached_objects is not None}"
        )
        self.current_worker.task_queue.put(task)
        logger.info(
            f"[ZCC manager] after putting task for prepare, dumping={save_infos_and_non_cached_objects is not None}"
        )

    def sync_offload_status(self):
        self.report_error_worker()
        assert self.current_worker is not None, "[ZCC manager] current_worker must not be None"
        while True:
            if self.current_worker.global_step.value != self.global_step:
                logger.info(
                    f"[ZCC manager] Waiting current worker offloading done., "
                    f"worker_state:{self.current_worker.status.value}, "
                    f"worker_step:{self.current_worker.global_step.value}, manager_step:{self.global_step}"
                )
                time.sleep(1)
            else:
                logger.info(
                    f"[ZCC manager] Current worker offloading done "
                    f"worker_step:{self.current_worker.global_step.value}, manager_step:{self.global_step} "
                )
                break
        self.current_pipeline_hook_step = 0
        self.current_worker = None

    def report_error_worker(self):
        for worker in self.workers:
            if worker.status.value == ZCCWorkerStatus.ERROR.value:
                logger.error(f"[ZCC manager] Worker{worker.worker_id} encountered error.")
                raise RuntimeError(f"{PC_DUMP_ERROR}")

    def zcc_pipeline_hook(self, hook_id):
        if self.current_worker is None:
            return
        if self.current_pipeline_hook_step == self.pipeline_hooks_steps:
            return
        if not self.ready_to_save:
            return
        task = (ZCCTaskType.OFFLOAD, self.global_step)
        self.current_worker.task_queue.put(task)
        self.current_pipeline_hook_step += 1

    def finalize(self):
        # clean up if the final step need to save
        if self.current_worker is not None:
            logger.info("[ZCC manager] clean up last step saving")
            # trigger offload
            for i in range(self.pipeline_hooks_steps):
                self.zcc_pipeline_hook(i)
            self.sync_offload_status()
        self.ready_to_save = False
        self.terminate_workers()

    def terminate_workers(self):
        for worker in self.workers:
            task = (ZCCTaskType.FINISH, None)
            worker.task_queue.put(task)
        for p in self.processes:
            p.join()


def worker_loop(worker):
    worker.run()


class ZeroCostCheckpointWorker:
    def __init__(
        self,
        worker_id,
        device_id,
        global_rank,
        offload_chunks,
        task_queue,
        status,
        global_step,
        version,
        use_expert_parallel,
        dp_rank,
        mp_rank,
        pp_rank,
        sd_rank,
        ema_coef=None,
        save_hf_steps=-1,
    ):
        super().__init__()
        self.worker_id = worker_id
        self.device_id = device_id
        self.global_rank = global_rank
        self.offload_chunks = offload_chunks
        self.task_queue = task_queue
        self.status = status
        self.global_step = global_step  # state value
        self.version = version
        self.ema_coef = ema_coef
        self.use_expert_parallel = use_expert_parallel
        self.dp_rank = dp_rank
        self.mp_rank = mp_rank
        self.pp_rank = pp_rank
        self.sd_rank = sd_rank
        self.save_hf_steps = save_hf_steps

        # for dynamic objects saving
        self.optimizer_fusion_storage_helper = None
        self.param_fusion_storage_helper = None
        self.all_numel = 0
        self.chunk_size_in_numel = 0
        self.offloaded_numels = 0
        self.optimizer_states_name_path = None
        self.model_states_name_path = None

        # for static objects saving
        self.model_config_content = None
        self.training_args_content = None
        self.model_meta_content = None
        self.user_file_list = None

        # for non cached objects saving
        # TODO(@gexiao): remove lr scheduler saves
        self.lr_scheduler = None
        self.trainer_state = None
        self.rng_state = None

        # for dumping
        self.flash_device_save_dir = None
        self.persistent_save_dir = None
        self.zcc_ema_processor = None

    def process_update_task(self, updates):
        """
        sync operation, main process should wait
        """
        version, dynamic_objecs, static_objects = updates

        optimizer_states_meta = dynamic_objecs["optimizer_states_meta"]
        model_states_meta = dynamic_objecs["model_states_meta"]
        self.optimizer_states_name_path = dynamic_objecs["optimizer_states_name_path"]
        self.model_states_name_path = dynamic_objecs["model_states_name_path"]
        self.build_fusion_storage_helper(optimizer_states_meta, model_states_meta)

        self.model_config_content = static_objects["model_config"]
        self.training_args_content = static_objects["training_args"]
        self.model_meta_content = static_objects["model_meta"]
        self.user_file_list = static_objects["user_file"]

        self.manage_offload_chunk()
        self.version.value = version

    def process_prepare_task(self, prepares):
        self.offloaded_numels = 0
        self.status.value = ZCCWorkerStatus.OFFLOADING.value
        if prepares is None:  # when `prepares` is None, not dumping
            return
        save_infos, non_cached_objects = prepares
        self.flash_device_save_dir, self.persistent_save_dir = save_infos
        self.lr_scheduler, self.trainer_state, self.rng_state = non_cached_objects

    def process_offload_task(self, dump, global_step):
        """
        call multipule times during model forward, return True if done dumpping
        """
        actual_offload_size = (
            min(self.offloaded_numels + self.chunk_size_in_numel, self.all_numel) - self.offloaded_numels
        )
        # Scene1: offload optimizer only
        if self.offloaded_numels + actual_offload_size <= self.optimizer_fusion_storage_helper.buffer_length:
            self.optimizer_fusion_storage_helper.sync_partial_param(
                start=self.offloaded_numels, end=self.offloaded_numels + actual_offload_size
            )
        # Scene2: offload optimizer and param
        elif self.offloaded_numels < self.optimizer_fusion_storage_helper.buffer_length:
            self.optimizer_fusion_storage_helper.sync_partial_param(
                start=self.offloaded_numels, end=self.optimizer_fusion_storage_helper.buffer_length
            )
            self.param_fusion_storage_helper.sync_partial_param(
                numel_to_sync=(
                    actual_offload_size - (self.optimizer_fusion_storage_helper.buffer_length - self.offloaded_numels)
                )
            )
        # Scene3: offload param only
        else:
            self.param_fusion_storage_helper.sync_partial_param(numel_to_sync=actual_offload_size)
        self.offloaded_numels += actual_offload_size

        # wait tasks done and change status to DUMPING at the last chunk
        if self.offloaded_numels == self.all_numel:
            self.optimizer_fusion_storage_helper.wait_all()
            self.param_fusion_storage_helper.wait_all()
            self.status.value = ZCCWorkerStatus.DUMPING.value
            self.global_step.value = global_step

            if self.ema_coef is not None:
                self.zcc_ema_processor.ema_accumulate(
                    self.trainer_state.global_step,
                    self.trainer_state.loss,
                    self.training_args_content.zcc_ema_loss_threshold,
                )

        # continue to process dumping task at the last chunk
        if self.offloaded_numels == self.all_numel:
            if dump:
                need_report_error = self.process_dump_task()
            else:
                need_report_error = False
            self.offloaded_numels = 0
            self.status.value = ZCCWorkerStatus.ERROR.value if need_report_error else ZCCWorkerStatus.IDLE.value
            return True
        return False

    def process_dump_task(self):
        """
        dump saved objects to either flash device or persistent device
        Notice:
        1. If dumping to flash device failed, the process will move on for other task
        2. If dumping to persistent device failed, the process will change status to fail, and the main process will raise Error.
        """
        need_report_error = False
        if self.flash_device_save_dir:
            try:
                self.process_dump_task_impl(self.flash_device_save_dir)
                logger.info(f"[ZCC Worker{self.worker_id}] Dumping to flash device done: {self.flash_device_save_dir}")
            except Exception as e:
                logger.error(f"{FC_DUMP_ERROR} [ZCC Worker{self.worker_id}] Failed to dump to flash device: {e}")
        if self.persistent_save_dir:
            try:
                self.process_dump_task_impl(self.persistent_save_dir)
                logger.info(
                    f"[ZCC Worker{self.worker_id}] Dumping to persistent device done: {self.persistent_save_dir}"
                )
            except Exception as e:
                logger.error(f"[ZCC Worker{self.worker_id}] Failed to dump to persistent device: {e}")
                need_report_error = True
        return need_report_error

    def _filter_moe_no_sync_optimizer_params(self, model_meta, optimzier_state_dict):
        """
        filter optimizer params which should not sync, copy from ...Trainer
        """
        filter_optimzier_state_dict = OrderedDict()
        assert "master_weights" in optimzier_state_dict, optimzier_state_dict.keys()
        param_names_in_master_weights = list(optimzier_state_dict["master_weights"].keys())
        filter_optimzier_state_dict["master_weights"] = OrderedDict()
        suffix = f"tp{self.mp_rank:0>2d}_pp{self.pp_rank:0>2d}"
        dyname_to_pname = model_meta["sharding_metas"][suffix]["structure_name_mapping"]
        dyname_to_meta = model_meta["sharding_metas"][suffix]["param_meta"]
        for k, pname in dyname_to_pname.items():
            shape, dtype, is_dist, is_no_sync = dyname_to_meta[k]
            if is_no_sync:
                if pname in param_names_in_master_weights:
                    filter_optimzier_state_dict["master_weights"][pname] = optimzier_state_dict["master_weights"][
                        pname
                    ]
                else:
                    pass
                    # logger.info(f"filter out master weight:{pname} -> {k}")
                for op_k, op_v in optimzier_state_dict.items():
                    if op_k.startswith(pname):
                        filter_optimzier_state_dict[op_k] = op_v
            else:
                # logger.info(f"filter out key={k}, when dp!=0")
                pass
        return filter_optimzier_state_dict

    def _dump_static_objects(self, output_dir):
        # Step1.1: save model config
        json_file_path = os.path.join(output_dir, CONFIG_NAME)
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.model_config_content)

        # Step1.2: save training args
        args_file_path = os.path.join(output_dir, TRAINING_ARGS_NAME)
        paddle.save(self.training_args_content, args_file_path)

        # Step1.3: save model meta
        model_meta_path = os.path.join(output_dir, MODEL_META_NAME)
        with open(model_meta_path, "w") as f:
            json.dump(self.model_meta_content, f)

        # Step1.4: save user files
        for (file_name, file_content) in self.user_file_list:
            file_path = os.path.join(output_dir, file_name)
            with open(file_path, "w") as f:
                f.write(file_content)

    def _dump_states(self, output_dir):
        # Step2.1: save model states
        with device_guard("cpu"):
            model_states_name_path = os.path.join(output_dir, self.model_states_name_path)
            state_dict = self.param_fusion_storage_helper.state_dict()
            # Step2.2: save optimizer states
            optimizer_state_name_path = os.path.join(output_dir, self.optimizer_states_name_path)
            opt_state_dict = self.optimizer_fusion_storage_helper.state_dict()
        # logger.info(showmem(f"[ZCCworker{self.worker_id}] after build state-dict"))
        if self.ema_coef is not None:
            ema_name_path = os.path.join(output_dir, self.optimizer_states_name_path).replace("optimizer", "ema")
            ema_state_dict = self.zcc_ema_processor.ema_state_dict()

        if self.dp_rank <= 0 or self.use_expert_parallel:
            if self.dp_rank > 0:  # ep
                opt_state_dict = self._filter_moe_no_sync_optimizer_params(self.model_meta_content, opt_state_dict)
                if self.ema_coef is not None:
                    # non master-weights in `ema-state-dict` when dp >1 will be filtered, which is acceptable
                    ema_state_dict = self._filter_moe_no_sync_optimizer_params(self.model_meta_content, ema_state_dict)
            paddle.save(state_dict, model_states_name_path)
            paddle.save(opt_state_dict, optimizer_state_name_path)

            if self.ema_coef is not None:
                paddle.save(ema_state_dict, ema_name_path)

    def _dump_args_and_state(self, output_dir):
        # Step2.3: save LR Scheduler (To be removed)
        lr_state_name_path = os.path.join(output_dir, SCHEDULER_NAME)
        if self.device_id == 0:
            paddle.save(self.lr_scheduler, lr_state_name_path)

        # Step2.4: save TrainerState
        trainer_state_name_path = os.path.join(output_dir, TRAINER_STATE_NAME)
        if self.device_id == 0:
            self.trainer_state.save_to_json(trainer_state_name_path)

        # Step2.5: save RNG State
        if self.rng_state is not None:
            rng_state_name_path = os.path.join(output_dir, f"rng_state_{dist.get_rank()}.pth")
            paddle.save(self.rng_state, rng_state_name_path)

    def process_dump_task_impl(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        # Step1: save static objects
        if self.device_id == 0:
            self._dump_static_objects(output_dir)
            logger.info("[ZCC worker] dump static objec done.")

        # Step2: save dynamic objects
        self._dump_states(output_dir)
        logger.info("[ZCC worker] dump model state done.")

        self._dump_args_and_state(output_dir)

        if self.save_hf_steps > 0 and self.ema_coef is not None:
            saved_signal_prefix = "_save_done_tmp"
        else:
            saved_signal_prefix = "saved_signal"

        # Step3: dump save signals
        saved_signal_path = os.path.join(output_dir, f"{saved_signal_prefix}_{self.global_rank}")
        with open(saved_signal_path, mode="w+") as f:
            f.write("1")
        logger.info("[ZCC worker] dump save signal done.")

    def run(self):
        core.set_cuda_current_device_id(self.device_id)
        paddle.set_device(f"gpu:{self.device_id}")
        logger.info(f"[ZCC Worker{self.worker_id}] Worker{self.worker_id} started.")
        ema_ckpt_path = None
        save_info_tuple = None  # save dir...
        start_time = None
        try:
            while True:
                # logger.info(f"[ZCC Worker{self.worker_id}] Wait for command")
                task = self.task_queue.get()
                task_type, task_body = task
                # logger.info(f"[ZCC Worker{self.worker_id}] Received a new task of type {task_type}")
                if task_type == ZCCTaskType.FINISH:
                    logger.info(f"[ZCC worker{self.worker_id}] exit")
                    break
                elif task_type == ZCCTaskType.UPDATE:
                    self.process_update_task(task_body)
                    if self.ema_coef is not None:
                        self.zcc_ema_processor = ZeroCostCheckpointEMAProcessor(  # 在 update task 后刷新 EMA buffer
                            self.optimizer_fusion_storage_helper, self.param_fusion_storage_helper, self.ema_coef
                        )
                        if ema_ckpt_path is not None:  # update ema if needed
                            logger.info(f"[ZCC EMA] load state dict from {ema_ckpt_path}")
                            with device_guard("cpu"):
                                state_dict = paddle.load(ema_ckpt_path)
                                if self.use_expert_parallel and self.dp_rank > 0:
                                    state_dict = self._filter_moe_no_sync_optimizer_params(
                                        self.model_meta_content, state_dict
                                    )
                                self.zcc_ema_processor.load_ema_state_dict(state_dict)
                            logger.info("[ZCC EMA] done loading")
                        ema_ckpt_path = None
                elif task_type == ZCCTaskType.PREPARE:
                    start_time = time.time()
                    save_info_tuple = task_body
                    self.process_prepare_task(task_body)
                elif task_type == ZCCTaskType.OFFLOAD:
                    dumped = self.process_offload_task(dump=save_info_tuple is not None, global_step=task_body)
                    if dumped:
                        used_time = time.time() - start_time
                        logger.info(f"[ZCC Worker{self.worker_id}] used time {used_time:.3f} sec")
                elif task_type == ZCCTaskType.SET_EMA_STATE_DICT:
                    ema_ckpt_path = task_body  # mark ema state dict path
                else:
                    raise ValueError(f"[ZCC Worker{self.worker_id}] Unknown task type: {task_type}")
        except Exception as e:
            import traceback

            logger.info(f"[ZCC Worker{self.worker_id}] failed!!, Exception:{e}\n Traceback:{traceback.format_exc()}\n")
            raise e

    def build_fusion_storage_helper(self, optimizer_states_meta, model_states_meta):
        (
            accumulators_meta,
            master_weights_meta,
            merged_model_params_meta,
            buffer_ipc_meta,
        ) = optimizer_states_meta
        if self.optimizer_fusion_storage_helper is None:
            self.optimizer_fusion_storage_helper = FusionStorageHelper(
                accumulators_meta,
                master_weights_meta,
                merged_model_params_meta,
                buffer_ipc_meta,
            )
        else:
            self.optimizer_fusion_storage_helper.reset_meta(
                accumulators_meta,
                master_weights_meta,
                merged_model_params_meta,
                buffer_ipc_meta,
            )
        model_param_mappings, model_ipc_meta_mappings = model_states_meta
        if self.param_fusion_storage_helper is None:
            self.param_fusion_storage_helper = ParamFusionStorageHelper(model_param_mappings, model_ipc_meta_mappings)
        else:
            self.param_fusion_storage_helper.reset_meta(model_param_mappings, model_ipc_meta_mappings)

    def manage_offload_chunk(self):
        # TODO(@gexiao): more precise slice for different dtype
        optimizer_offload_numel = self.optimizer_fusion_storage_helper.buffer_length
        param_offload_numel = self.param_fusion_storage_helper.all_param_numel
        self.all_numel = optimizer_offload_numel + param_offload_numel
        self.chunk_size_in_numel = (self.all_numel - 1) // self.offload_chunks + 1
        logger.info(
            f"[ZCC Worker{self.worker_id}] All numel: {self.all_numel}, Offload chunks: {self.offload_chunks}, Chunk size: {self.chunk_size_in_numel}]"
        )


class EMABuffer(ABC):
    def __init__(self, resume_from_checkpoint, args, offload=True):
        self.master_weights = {}
        self.model_params = {}
        self.args = args
        self.offload = offload
        if resume_from_checkpoint is not None:
            self._load(resume_from_checkpoint)

    def _load(self, resume_from_checkpoint):
        ema_path = self._ema_path(resume_from_checkpoint)
        if not os.path.exists(ema_path):
            return

        success, err_msg = self._check_consistent_dist_strategy(resume_from_checkpoint)
        if not success:
            logger.info(f"Cannot load EMA because: {err_msg}")
            return

        logger.info(f"Loading EMA checkpoint from {resume_from_checkpoint} ...")
        with device_guard("cpu"):
            ema_state_dict = paddle.load(ema_path)
        logger.info(f"Load EMA checkpoint from {resume_from_checkpoint} done")

        self.master_weights = ema_state_dict.pop("master_weights")
        self.model_params = ema_state_dict

    def get_ema_state_dict(self):
        ema_state_dict = {"master_weights": self.master_weights}
        ema_state_dict.update(self.model_params)
        return ema_state_dict

    def save(self, global_step):
        base_path = os.path.join(self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{global_step}")
        ema_path = self._ema_path(base_path)
        ema_state_dict = {"master_weights": self.master_weights}
        ema_state_dict.update(self.model_params)
        os.makedirs(base_path, exist_ok=True)
        logger.info(f"Saving EMA checkpoint to {base_path} ...")
        paddle.save(ema_state_dict, ema_path)
        logger.info(f"Save EMA checkpoint to {base_path} done")

    def ema_accumulate(self, global_step, loss, ema_loss_threshold):
        if ema_loss_threshold is None or loss < ema_loss_threshold:
            logger.info(f"EMA accumulating for step {global_step} ...")
            self._ema_impl(
                state_dict=self._get_master_weight(),
                ema_state_dict=self.master_weights,
            )
            self._ema_impl(
                state_dict=self._get_model_state(),
                ema_state_dict=self.model_params,
            )
            logger.info(f"EMA accumulate done for step {global_step}")

    def _ema_impl(self, state_dict, ema_state_dict):
        ema_coef = self.args.zcc_save_ema_coef
        for k, v in state_dict.items():
            if k in ema_state_dict:
                ema_tensor = ema_state_dict[k]
                ema_tensor = ema_coef * ema_tensor.cuda() + (1 - ema_coef) * v.cuda()
                ema_tensor.name = v.name
                v = ema_tensor
                del ema_tensor

            if self.offload:
                v_pin = v.pin_memory()
                v_pin.name = v.name
                v = v_pin
            ema_state_dict[k] = v

    @abstractmethod
    def _get_master_weight(self):
        pass

    @abstractmethod
    def _get_model_state(self):
        pass

    @abstractmethod
    def _check_consistent_dist_strategy(self, resume_from_checkpoint):
        pass


class EMABufferShardingIOBased(EMABuffer):
    def __init__(self, resume_from_checkpoint, args, sharding_io, offload=True):
        assert sharding_io is not None, "EMA should be only enabled when save_sharded_model is True"
        self.sharding_io = sharding_io
        super().__init__(resume_from_checkpoint, args, offload)

    def _ema_path(self, base_path):
        path = _add_variant(PADDLE_OPTIMIZER_NAME, self.args.optimizer_name_suffix)
        path = path.replace("optimizer", "ema")
        return os.path.join(base_path, path)

    def _get_model_state(self):
        return self.sharding_io.manipulate_state_dict_and_config(
            unwrap_model(self.sharding_io.model),
            merge_tensor_parallel=False,
        )[0]

    def _get_master_weight(self):
        return self.sharding_io.optimizer.state_dict()["master_weights"]

    def _check_consistent_dist_strategy(self, resume_from_checkpoint):
        return self.sharding_io.check_same_strategy(resume_from_checkpoint)


class EMABufferFcBased(EMABuffer):
    def __init__(self, resume_from_checkpoint, args, offload=True, hcg=None, model=None, optimizer=None):
        self.hcg = hcg
        self.model = model
        self.optimizer = optimizer
        self.dist_info_collector_and_validator = DistInfoCollectorValidator(args, hcg)
        self.device_id = int(os.getenv("FLAGS_selected_gpus"))

        super().__init__(resume_from_checkpoint, args, offload)

    def _get_model_meta(self):
        return self.dist_info_collector_and_validator.gather_distributed_model_meta(self.model, self.optimizer)

    def _ema_path(self, base_path):
        return os.path.join(base_path, "ema_state", f"{dist.get_rank()}_0.distcp")

    def _check_consistent_dist_strategy(self, resume_from_checkpoint):
        return self.dist_info_collector_and_validator.check_same_strategy(resume_from_checkpoint)

    def _get_model_state(self):
        assert self.model is not None, "expected model is not None"
        return self.model.state_dict()

    def _get_master_weight(self):
        assert self.optimizer is not None, "expected optimizer is not None"
        return self.optimizer.state_dict()["master_weights"]

    def save(self, global_step):
        model_meta_content = self._get_model_meta()
        base_path = os.path.join(self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{global_step}")
        os.makedirs(base_path, exist_ok=True)
        model_meta_path = os.path.join(base_path, MODEL_META_NAME)
        if self.device_id == 0:
            with open(model_meta_path, "w") as f:
                json.dump(model_meta_content, f)

        super().save(global_step)


class NonZCCEMACallback(TrainerCallback):
    def __init__(self, ema_buffer: EMABuffer, ema_state_assembler=None):
        self.buffer = ema_buffer
        self.ema_state_assembler = ema_state_assembler

    @staticmethod
    def create_nonzcc_callback(
        args,
        resume_from_checkpoint,
        sharding_io=None,
        model=None,
        optimizer=None,
        hcg=None,
        offload=True,
        ema_state_assembler=None,
    ):
        if args.save_checkpoint_format == "flex_checkpoint":
            ema_buffer = EMABufferFcBased(
                resume_from_checkpoint, args, offload=offload, hcg=hcg, model=model, optimizer=optimizer
            )
        else:
            assert sharding_io is not None, "EMA should be only enabled when save_sharded_model is True"
            ema_buffer = EMABufferShardingIOBased(resume_from_checkpoint, args, sharding_io, offload=offload)

        return NonZCCEMACallback(ema_buffer, ema_state_assembler)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % args.zcc_ema_interval == 0:
            self.buffer.ema_accumulate(state.global_step, state.loss, args.zcc_ema_loss_threshold)
        if control.should_save:
            self.buffer.save(state.global_step)
        if control.should_save_hf:
            assert (
                self.ema_state_assembler is not None
            ), "When save_hf_steps is enabled, if ZCC is not used, ema_state_assembler must be initialized."
            logger.info("Starting to save unsplit EMA states!")
            start_time = time.time()
            ema_state_dict = self.buffer.get_ema_state_dict()
            ema_state_dict_cuda = {}
            ema_state_dict_cuda["master_weights"] = {}
            master_weights = ema_state_dict.pop("master_weights")
            for k, v in master_weights.items():
                ema_state_dict_cuda["master_weights"][k] = v.cuda()
            for k, v in ema_state_dict.items():
                if v.dtype != paddle.bfloat16:
                    ema_state_dict_cuda[k] = v.cuda()
            ema_sharded_state_dict = self.ema_state_assembler._build_ema_sharded_state_dict(ema_state_dict_cuda)
            self.ema_state_assembler._save_full_ema_states(state.global_step, ema_sharded_state_dict)
            del ema_sharded_state_dict
            del ema_state_dict_cuda
            end_time = time.time()
            logger.info(f"Unsplit EMA states saved successfully, time taken: {end_time - start_time} s")


class DistInfoCollectorValidator:
    def __init__(self, args, hcg=None):
        self.args = args
        self.hcg = hcg
        if self.hcg is None:
            self.hcg = fleet.get_hybrid_communicate_group()

    def _load_model_meta_impl(self, dir):
        meta_path = os.path.join(dir, MODEL_META_NAME)
        assert os.path.exists(meta_path), f"{meta_path} not exist"
        with open(meta_path, "r") as handle:
            model_dist_meta = json.load(handle)
        assert "parallel_config" in model_dist_meta
        self._check_distributed_strategy(model_dist_meta["parallel_config"])
        return model_dist_meta

    def _all_gather_simple_object(self, obj, group=None):
        if group is None:
            group = self.hcg.get_sharding_parallel_group()
        res = []
        if group.nranks < 2:
            return [obj]
        paddle.distributed.all_gather_object(res, obj, group)
        return res

    def _sharding_meta_suffix(self, tp_rank=None, pp_rank=None):
        if tp_rank is None:
            tp_rank = self.args.tensor_parallel_rank
        if pp_rank is None:
            pp_rank = self.args.pipeline_parallel_rank
        suffix = f"tp{tp_rank:0>2d}_pp{pp_rank:0>2d}"
        if self.args.expert_model_parallel_size > 1:
            ep_rank = self.args.expert_parallel_rank
            return f"{suffix}_ep{ep_rank:0>2d}"
        else:
            return suffix

    def _gather_sharding_metas(self, model, optimizer):
        nranks = dist.get_world_size()
        if not self.args.use_hybrid_parallel or nranks <= 1:
            return None
        if not reshard_util.is_sharding_opt(optimizer):
            return None

        sharding_strategy = reshard_util.get_sharding_strategy(optimizer)
        param2rank = {}
        pp_overlap = False
        if sharding_strategy == SHARDING_STRATEGY_V1:
            optimizer = unwrap_optimizer(optimizer, DygraphShardingOptimizer)
            param2rank = {k: v for (k, v) in optimizer._param2rank.items()}
        else:
            pp_overlap = unwrap_optimizer(optimizer, DygraphShardingOptimizerV2).pp_overlap

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
        suffix = self._sharding_meta_suffix()
        sharding_metas[suffix] = sharding_meta
        sharding_metas_list = self._all_gather_simple_object(sharding_metas, self.hcg.get_model_parallel_group())
        sharding_metas = {k: v for e in sharding_metas_list for (k, v) in e.items()}
        sharding_metas_list = self._all_gather_simple_object(sharding_metas, self.hcg.get_pipe_parallel_group())
        sharding_metas = {k: v for e in sharding_metas_list for (k, v) in e.items()}
        if self.args.expert_model_parallel_size > 1:
            sharding_metas_list = self._all_gather_simple_object(sharding_metas, self.hcg.get_expert_parallel_group())
            sharding_metas = {k: v for e in sharding_metas_list for (k, v) in e.items()}
        return sharding_metas

    def _check_distributed_strategy(self, parallel_config):
        ep_degree = parallel_config.get("ep_degree", 1)
        if ep_degree > 1:
            tp_degree = parallel_config["mp_degree"]
            sharding_degree = parallel_config["sharding_degree"]
            moe_sharding_degree = parallel_config.get("moe_sharding_degree", 1)
            assert tp_degree * sharding_degree == ep_degree * moe_sharding_degree, "mismatch parallel degree settings"

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

    def gather_distributed_model_meta(self, model, optimizer):
        if not self.args.use_hybrid_parallel:
            return None

        if not self.args.should_save_sharding_stage1_model:
            return None

        nranks = dist.get_world_size()
        if nranks <= 1:
            return None

        model_meta = {}
        model_meta["parallel_config"] = self._get_distributed_strategy()
        model_meta["sharding_metas"] = self._gather_sharding_metas(model, optimizer)

        return model_meta

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


def saved_ckptmeta(state_dict, ckpt_file_name, process_group=None, replicate_saved_into_local=False):
    with paddle.base.dygraph.guard():
        assert isinstance(state_dict, dict), "The state_dict should be a dictionary."
        flat_state_dict, mapping = flatten_state_dict(state_dict)
        if len(flat_state_dict) > 0:
            for val in flat_state_dict.values():
                assert isinstance(
                    val, (paddle.Tensor, ShardedWeight)
                ), f"The value of state_dict should be a paddle.Tensor or ShardedWeight, but got: {val}."

        use_dist = True if paddle.distributed.get_world_size() > 1 else False

        if use_dist and process_group is None and not is_initialized():
            # Init the default global process group
            paddle.distributed.init_parallel_env()

        metadata = Metadata()
        local_state_dict_filter_map = {}
        local_state_dict_metadata = {}
        local_storage_metadata = {}
        global_shape = None
        for key, val in flat_state_dict.items():
            assert isinstance(val, ShardedWeight), f"expected ShardedWeight, but got {type(val)}"
            local_tensor = val.local_tensor
            local_shape = val.local_shape
            global_offset = val.global_offset
            global_shape = val.global_shape
            is_flattened = val.is_flattened
            flattened_range = val.flattened_range

            local_tensor_dtype = str(local_tensor.dtype).split(".")[1]
            if flattened_range is not None:
                flattened_range = (flattened_range.start, flattened_range.stop)
            else:
                flattened_range = None
            local_state_dict_metadata[key] = LocalTensorMetadata(
                tuple(global_offset),
                tuple(local_shape),
                local_tensor_dtype,
                tuple(global_shape),
                is_flattened,
                flattened_range,
            )
            local_storage_metadata[
                LocalTensorIndex(
                    key, tuple(global_offset), is_flattened, flattened_range, local_shape=tuple(local_shape)
                )
            ] = ckpt_file_name

            local_state_dict_filter_map[key] = False

        global_state_dict_metadata = []
        global_storage_metadata = []
        global_flatten_mapping = []
        if use_dist:
            paddle.distributed.all_gather_object(
                global_state_dict_metadata,
                local_state_dict_metadata,
                process_group,
            )
            paddle.distributed.all_gather_object(global_storage_metadata, local_storage_metadata, process_group)
            paddle.distributed.all_gather_object(global_flatten_mapping, mapping, process_group)
        else:
            global_state_dict_metadata.append(local_state_dict_metadata)
            global_storage_metadata.append(local_storage_metadata)
            global_flatten_mapping.append(mapping)

        def balanced_dedup_key_in_dict(global_storage_metadata):
            lti_to_files = defaultdict(set)
            for storage_metadata in global_storage_metadata:
                for lti, fname in storage_metadata.items():
                    lti_to_files[lti].add(fname)

            file_load = defaultdict(int)
            out = {}
            for lti, file_candidates in lti_to_files.items():
                candidates = sorted(file_candidates)
                selected_main_file = min(candidates, key=lambda f: file_load[f])
                file_load[selected_main_file] += 1

                if replicate_saved_into_local:
                    lti_main = replace(lti, replica_id=0)
                    out[lti_main] = selected_main_file
                    replica_id = 1
                    for fname in candidates:
                        if fname == selected_main_file:
                            continue
                        lti_replica = replace(lti, replica_id=replica_id)
                        out[lti_replica] = fname
                        replica_id += 1
                else:
                    out[lti] = selected_main_file

            return out

        metadata.state_dict_metadata = merge_state_dict_metadata(global_state_dict_metadata)
        metadata.storage_metadata = balanced_dedup_key_in_dict(global_storage_metadata)
        metadata.flat_mapping = dedup_key_in_dict(global_flatten_mapping)
        # logger.debug(f"metadata:{metadata}")

        def _gen_filter_map():
            for tensor_index, file_name in metadata.storage_metadata.items():
                rank = int(file_name.split(".")[0].split("_")[0])
                if tensor_index in local_storage_metadata and rank != paddle.distributed.get_rank():
                    # 'True' represents that this tensor is not needed by the current rank.
                    local_state_dict_filter_map[tensor_index.tensor_key] = True

        _gen_filter_map()
        # logger.debug(f"local_state_dict_filter_map:{local_state_dict_filter_map}")

        return metadata, local_state_dict_filter_map


class ZeroCostCheckpointCallbackFcBased(ZeroCostCheckpointCallback):
    def __init__(self, args, zcc_manager, timer, unused_arg):
        self.manager = zcc_manager
        self.runtime_timer = timer
        self.user_file_list = []
        self.model_meta = None
        self.zcc_ema_interval = args.zcc_ema_interval
        self.args = args

        if paddle.distributed.get_world_size() > 1 and self.args.use_hybrid_parallel:
            self.hcg = fleet.get_hybrid_communicate_group()
            self.sharding_group = self.hcg.get_sharding_parallel_group()

    def _manipulate_state_dict_and_config(self, model_to_save, optimizer):
        # return model_to_save.sharded_state_dict()

        group_getter = GroupGetter(model_to_save)
        gids = group_getter.get_group_ids()
        from paddleformers.trainer.utils.sharding_io import (
            exclude_parameters_in_state_dict,
        )

        state_dict = model_to_save.state_dict()

        # tmp wa should_save_sharding_stage1_model
        if self.args.bf16:
            param_names_in_master_weights = []
            optimzier_state_dict = optimizer.state_dict()
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

        return state_dict

    def _cache_meta_for_sharded_save(self, model, optimizer):
        logger.info("Start caching metas for sharded save...")
        (self.manipulated_state_dict) = self._manipulate_state_dict_and_config(model, optimizer)

        def recover_sharded_state_dict():
            filtered_sharded_state_dict = {}
            model_sharded_state_dict = model.sharded_state_dict()
            for k, v in self.manipulated_state_dict.items():
                filtered_sharded_state_dict[k] = model_sharded_state_dict[k]
            return filtered_sharded_state_dict

        self.manipulated_state_dict = recover_sharded_state_dict()

        logger.info("Cache manipulated static dict done.")

        model_to_save = unwrap_model(model)
        dtype = get_parameter_dtype(model_to_save)
        model_to_save.config.dtype = str(dtype).split(".")[1]
        self.manipulated_config_to_save = copy.deepcopy(model_to_save.config)
        self.manipulated_config_to_save.architectures = [clean_model_class_name(model_to_save.__class__.__name__)]
        self.manipulated_config_to_save = self.manipulated_config_to_save.to_json_string(use_diff=True)
        logger.info("Cache manipulated model config done")

        self.model_meta = DistInfoCollectorValidator(self.args, self.hcg).gather_distributed_model_meta(
            model, optimizer
        )

        def create_ckpt_file_name():
            data_file_name = f"{paddle.distributed.get_rank()}_0.distcp"
            meta_file_name = "0.metadata"
            return (data_file_name, meta_file_name)

        # model state ckpt meta and filter
        self.ckpt_data_name, self.ckpt_meta_name = create_ckpt_file_name()
        # self.model_ckpt_meta, self.model_state_filter = saved_ckptmeta(model.sharded_state_dict(), self.ckpt_data_name)
        self.model_ckpt_meta, self.model_state_filter = saved_ckptmeta(
            self.manipulated_state_dict,
            self.ckpt_data_name,
            replicate_saved_into_local=self.args.replicate_saved_into_local,
        )

        # opt state dict ckpt meta and filter
        opt_state_dict_tmp = optimizer.sharded_state_dict(model.sharded_state_dict())

        opt_state_dict = {}
        master_weights = {}
        for k, v in opt_state_dict_tmp.items():
            if k.endswith(".w_0"):
                master_weights[k] = v
            else:
                opt_state_dict[k] = v

        self.opt_ckpt_meta, self.opt_state_filter = saved_ckptmeta(
            opt_state_dict, self.ckpt_data_name, replicate_saved_into_local=self.args.replicate_saved_into_local
        )
        self.master_weight_ckpt_meta, self.master_weights_filter = saved_ckptmeta(
            master_weights, self.ckpt_data_name, replicate_saved_into_local=self.args.replicate_saved_into_local
        )

        # gen unified name mapping for optimzier
        self.unified_name_mapping, self.param_slice_info = self._gen_unified_name(
            optimizer, model.sharded_state_dict()
        )
        logger.info("Cache distributed model meta done.")

    def _gen_unified_name(self, optimizer, model_sharded_state_dict):
        param_slice_info = {}
        padded_param = set()
        for buffer in optimizer._comm_buffer_list:
            for (
                param_name,
                grad_view,
            ) in buffer._sharding_param_grad_view.items():
                numel = grad_view._param.numel().item()
                param_begin = grad_view._param_begin
                param_end = grad_view._param_end
                index = grad_view._index
                padding_begin = index + numel
                flattened_range = slice(
                    param_begin - index,
                    max(
                        min(padding_begin - index, param_end - index),
                        param_begin - index,
                    ),
                )
                if param_end > padding_begin:
                    padded_param.add(param_name)

                param_slice_info[param_name] = flattened_range

        _FP32_MASTER = "fp32_master_0"
        _optimizer_scalar_name = [
            "beta1_pow_acc_0",
            "beta2_pow_acc_0",
        ]
        _optimizer_non_scaler_name = [
            "moment1_0",
            "moment2_0",
            "velocity_0",
        ]

        def _generate_base_static_name(vname):
            if _FP32_MASTER in vname:
                return tuple(vname.split("_" + _FP32_MASTER + "_", 1))
            for name in _optimizer_scalar_name + _optimizer_non_scaler_name:
                if vname.endswith(name):
                    return vname[: -(len(name) + 1)], name
            raise ValueError(f"Cannot split variable name: {vname}.")

        model_sharded_state_dict = dict(sorted(model_sharded_state_dict.items()))
        static_to_struct_mapping = {}
        for k, v in model_sharded_state_dict.items():
            if v.local_tensor.name not in static_to_struct_mapping:
                static_to_struct_mapping[v.local_tensor.name] = k

        optimizer_state_dict = optimizer.state_dict()
        optimizer_unified_name_mapping = {}
        unified_slice_info = {}

        master_weights = optimizer_state_dict.pop("master_weights", None)
        optimizer_state_dict.pop("LR_Scheduler", None)
        for key, _ in optimizer_state_dict.items():
            static_name, optim_state_type = _generate_base_static_name(key)
            struct_name = static_to_struct_mapping[static_name]
            unified_name = f"{struct_name}.{optim_state_type}"

            flattened_range = param_slice_info[static_name]

            # if flattened_range.stop - flattened_range.start == 0:
            #     continue
            optimizer_unified_name_mapping[key] = unified_name
            unified_slice_info[unified_name] = flattened_range

        if master_weights is not None:
            for key, _ in master_weights.items():
                struct_name = static_to_struct_mapping[key]
                unified_name = f"{struct_name}.w_0"

                flattened_range = param_slice_info[key]

                # if flattened_range.stop - flattened_range.start == 0:
                #     continue

                optimizer_unified_name_mapping[key] = unified_name
                unified_slice_info[unified_name] = flattened_range

        return optimizer_unified_name_mapping, unified_slice_info

    def _pack_dynamic_objects(self):
        dynamic_objecs = {}
        dynamic_objecs["optimizer_states_meta"] = self.optimizer_states_meta
        dynamic_objecs["model_states_meta"] = self.model_states_meta

        dynamic_objecs["distcp_file_name"] = (self.ckpt_data_name, self.ckpt_meta_name)

        dynamic_objecs["model_ckpt_meta"] = self.model_ckpt_meta
        dynamic_objecs["model_state_filter"] = self.model_state_filter

        dynamic_objecs["opt_ckpt_meta"] = self.opt_ckpt_meta
        dynamic_objecs["opt_state_filter"] = self.opt_state_filter

        dynamic_objecs["master_weight_ckpt_meta"] = self.master_weight_ckpt_meta
        dynamic_objecs["master_weights_filter"] = self.master_weights_filter

        dynamic_objecs["unified_name_mapping"] = self.unified_name_mapping
        dynamic_objecs["param_slice_info"] = self.param_slice_info

        return dynamic_objecs

    def maybe_update_zcc_worker(self, args, model, optimizer, global_step):
        # logger.info(f"check should update :{optimizer.fused_buffer_version} vs {self.manager.cache_version}")
        if optimizer.fused_buffer_version == self.manager.cache_version:
            return

        logger.info("ZCC checkpoint workers need upgrade.")
        self._cache_meta_for_sharded_save(model, optimizer)
        param_mappings, ipc_meta_mappings = get_fused_param_mappings(optimizer, self.manipulated_state_dict)
        self.optimizer_states_meta = (
            optimizer.fused_states_accumulators_meta,
            optimizer.fused_states_master_weights_meta,
            None,
            optimizer.fused_states_buffer_ipc_meta,
        )

        self.model_states_meta = (param_mappings, ipc_meta_mappings)
        dynamic_objects = self._pack_dynamic_objects()
        static_objects = self._pack_static_objects(args)

        self.manager.update_zcc_workers(optimizer.fused_buffer_version, dynamic_objects, static_objects, global_step)
        logger.info(f"[ZCC Callback] after first update:{optimizer.fused_states_buffer_ipc_meta}")


class ZeroCostCheckpointWorkerFcBased(ZeroCostCheckpointWorker):
    def process_update_task(self, updates):
        """
        sync operation, main process should wait
        """
        version, dynamic_objecs, static_objects = updates
        self.distcp_file_name = dynamic_objecs["distcp_file_name"]
        self.model_ckpt_meta = dynamic_objecs["model_ckpt_meta"]
        self.model_state_filter = dynamic_objecs["model_state_filter"]
        self.opt_ckpt_meta = dynamic_objecs["opt_ckpt_meta"]
        self.opt_state_filter = dynamic_objecs["opt_state_filter"]
        self.master_weight_ckpt_meta = dynamic_objecs["master_weight_ckpt_meta"]
        self.master_weights_filter = dynamic_objecs["master_weights_filter"]

        self.unified_name_mapping = dynamic_objecs["unified_name_mapping"]
        self.param_slice_info = dynamic_objecs["param_slice_info"]

        optimizer_states_meta = dynamic_objecs["optimizer_states_meta"]
        model_states_meta = dynamic_objecs["model_states_meta"]

        self.build_fusion_storage_helper(optimizer_states_meta, model_states_meta)

        self.model_config_content = static_objects["model_config"]
        self.training_args_content = static_objects["training_args"]
        self.model_meta_content = static_objects["model_meta"]
        self.user_file_list = static_objects["user_file"]

        self.manage_offload_chunk()
        self.version.value = version

    def _replace_pname_with_unified(self, state_dict):
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            assert key in self.unified_name_mapping, f"{key} not in {self.unified_name_mapping.keys()}"
            new_key = self.unified_name_mapping[key]
            new_state_dict[new_key] = value
        return new_state_dict

    @staticmethod
    def _filter_state_dict(state_dict, filter_map):
        need_remove_keys = []
        for k, _ in state_dict.items():
            # two case:
            # 1. Mutliple key share the same tensor.
            # 2. Don't need to be saved in current rank.
            if k not in filter_map.keys():
                logger.debug(f"[ZCC worker] {k} not exist in filter map.")
            if (k not in filter_map.keys()) or filter_map[k]:
                need_remove_keys.append(k)
        for k in need_remove_keys:
            state_dict.pop(k)
        return state_dict

    @staticmethod
    def _slice_padded_tensor(static_dict, param_slice_info):
        new_static_dict = {}
        for k, v in static_dict.items():
            if k in param_slice_info:
                logger.info(f"[ZCC worker] Slice padded tensor of {k}")
                flattened_range = param_slice_info[k]
                new_static_dict[k] = paddle.slice(
                    v,
                    axes=[0],
                    starts=[0],
                    ends=[flattened_range.stop - flattened_range.start],
                )
            else:
                new_static_dict[k] = v
        return new_static_dict

    def _save_model_state(self, output_dir):
        data_file_name, meta_file_name = self.distcp_file_name
        self.model_states_path = os.path.join(output_dir, MODEL_STATE_DIC, data_file_name)
        self.model_states_meta_path = os.path.join(output_dir, MODEL_STATE_DIC, meta_file_name)

        if self.dp_rank <= 0 or self.use_expert_parallel:
            with device_guard("cpu"):
                state_dict = self.param_fusion_storage_helper.state_dict()

                state_dict = self._filter_state_dict(state_dict, self.model_state_filter)
                logger.debug(f"model states length is {len(state_dict)}")
                paddle.save(state_dict, self.model_states_path)

                if self.device_id == 0:
                    paddle.save(self.model_ckpt_meta, self.model_states_meta_path)
        logger.info("[ZCC worker] Finish model states saved.")

    def _save_opt_state(self, output_dir):
        data_file_name, meta_file_name = self.distcp_file_name
        self.opt_state_path = os.path.join(output_dir, OPTIMIZER_STATE_DIC, data_file_name)
        self.opt_state_meta_path = os.path.join(output_dir, OPTIMIZER_STATE_DIC, meta_file_name)

        self.master_weight_path = os.path.join(output_dir, MASTER_WEIGHT_DIC, data_file_name)
        self.master_weight_meta_path = os.path.join(output_dir, MASTER_WEIGHT_DIC, meta_file_name)

        if self.dp_rank <= 0 or self.use_expert_parallel:
            with device_guard("cpu"):
                opt_state_dict = self.optimizer_fusion_storage_helper.state_dict()
                master_weights = opt_state_dict.pop("master_weights", {})

                opt_state_dict = self._replace_pname_with_unified(opt_state_dict)
                logger.info("[ZCC worker] opt state dict replace pname using unified name.")

                master_weights = self._replace_pname_with_unified(master_weights)
                logger.info("[ZCC worker] master weightsdict replace pname using unified name.")

                opt_state_dict = self._slice_padded_tensor(opt_state_dict, self.param_slice_info)
                logger.info("[ZCC worker] opt state dict slice padded tensor complete.")
                master_weights = self._slice_padded_tensor(master_weights, self.param_slice_info)
                logger.info("[ZCC worker] master weights slice padded tensor complete.")

            if self.dp_rank > 0:  # ep
                opt_state_dict = self._filter_moe_no_sync_optimizer_params(self.model_meta_content, opt_state_dict)

            opt_state_dict = self._filter_state_dict(opt_state_dict, self.opt_state_filter)
            logger.info("[ZCC worker] opt state dict filter by opt_state_filter complete.")
            master_weights = self._filter_state_dict(master_weights, self.master_weights_filter)
            logger.info("[ZCC worker] master weights dict filter by master_weights_filter complete.")

            logger.debug(f"opt states length is {len(opt_state_dict)}")
            logger.debug(f"master weights length is {len(master_weights)}")
            paddle.save(opt_state_dict, self.opt_state_path)
            paddle.save(master_weights, self.master_weight_path)
            if self.device_id == 0:
                paddle.save(self.opt_ckpt_meta, self.opt_state_meta_path)
                paddle.save(self.master_weight_ckpt_meta, self.master_weight_meta_path)
            logger.info("[ZCC worker] Finish opt states and master weights saved.")

    def _save_ema_state(self, output_dir):
        data_file_name, meta_file_name = self.distcp_file_name
        if (self.dp_rank <= 0 or self.use_expert_parallel) and self.ema_coef is not None:
            self.ema_name_path = os.path.join(output_dir, EMA_STATE_DIC, data_file_name)
            ema_state_dict = self.zcc_ema_processor.ema_state_dict()

            if self.dp_rank > 0:
                ema_state_dict = self._filter_moe_no_sync_optimizer_params(self.model_meta_content, ema_state_dict)
            logger.debug(f"ema states length is {len(ema_state_dict)}")
            paddle.save(ema_state_dict, self.ema_name_path)
        logger.info("[ZCC worker] Finish ema states saved.")

    def _dump_states(self, output_dir):
        self._save_model_state(output_dir)
        self._save_opt_state(output_dir)
        self._save_ema_state(output_dir)
