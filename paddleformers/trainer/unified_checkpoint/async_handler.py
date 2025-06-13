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
"""Asynchronous unified checkpoint handler."""

import multiprocessing
import os
import time
from multiprocessing import shared_memory

import paddle
import paddle.distributed as dist

from ...transformers.utils import is_safetensors_available
from ...utils.log import logger

if is_safetensors_available():
    from safetensors.numpy import save_file as safe_save_file

from ...quantization.unified_checkpoint_quantization import quant_unified_optimizer
from .shared_memory_utils import (
    _read_state_dict_from_shm,
    _traverse_copy_to_shm,
    create_meta_dict,
)

__all__ = ["AsyncCheckpointHandler"]


class AsyncCheckpointHandler:
    def __init__(self, args):
        # Mainly for asynchronous saving.
        self.args = args
        self.global_rank = paddle.distributed.get_rank() if paddle.distributed.get_world_size() > 1 else -1

        self._shm_model_weight = None
        self._shm_master_weight = None
        self._shm_optimizer_weight = None
        self._meta_dict_model = None
        self._meta_dict_master_weight = None
        self._meta_dict_optim = None
        self._process_model_weight = None
        self._process_master_weight = None
        self._process_optimizer_weight = None
        self._lock = None
        self._shared_save_model_flag = None
        self._shared_save_master_weight_flag = None
        self._shared_save_optimizer_flag = None

        if "async_save" in self.args.unified_checkpoint_config:
            self._lock = multiprocessing.Lock()
            self._shared_save_model_path = multiprocessing.Array("c", 100000)
            self._shared_save_model_signal_path = multiprocessing.Array("c", 100000)
            self._shared_save_master_weight_path = multiprocessing.Array("c", 100000)
            self._shared_save_master_weight_signal_path = multiprocessing.Array("c", 100000)
            self._shared_save_optimizer_path = multiprocessing.Array("c", 100000)
            self._shared_save_optimizer_signal_path = multiprocessing.Array("c", 100000)
            self._shared_save_model_flag = multiprocessing.Array("i", 1)
            self._shared_save_master_weight_flag = multiprocessing.Array("i", 1)
            self._shared_save_optimizer_flag = multiprocessing.Array("i", 1)

    def _file_save_async_or_sync(
        self, state_dict, path, signal_path=None, is_sync=True, state_dict_type="model_weight", ckpt_quant_stage="O0"
    ):
        if is_sync:
            for k in list(state_dict.keys()):
                if isinstance(state_dict[k], paddle.Tensor):
                    state_dict[k] = state_dict.pop(k).cpu().numpy()

            if state_dict_type == "optimizer_weight" and ckpt_quant_stage != "O0":
                state_dict = quant_unified_optimizer(state_dict, state_dict_type, ckpt_quant_stage)
            safe_save_file(state_dict, path, metadata={"format": "np"})
        else:
            if len(state_dict.keys()) == 0:
                saved_signal_path = os.path.join(signal_path, f".{state_dict_type}.done.{self.global_rank}")
                paddle.save(self.global_rank, saved_signal_path)
                return

            if state_dict_type == "model_weight":
                if self._shm_model_weight is None:
                    self._meta_dict_model, buffer_size = create_meta_dict(state_dict)
                    self._shm_model_weight = shared_memory.SharedMemory(create=True, size=buffer_size)
                shm_state_dict = self._shm_model_weight
                meta_dict = self._meta_dict_model
                shared_save_flag = self._shared_save_model_flag
                shared_save_path = self._shared_save_model_path
                shared_save_signal_path = self._shared_save_model_signal_path
                if self._process_model_weight is None:
                    self._process_model_weight = multiprocessing.Process(
                        target=self._save_file_async_in_process,
                        args=(
                            meta_dict,
                            self._shm_model_weight.name,
                            self._shared_save_model_flag,
                            self._shared_save_model_path,
                            self._shared_save_model_signal_path,
                            self._lock,
                            state_dict_type,
                            self.global_rank,
                        ),
                    )
                    self._process_model_weight.start()
                process = self._process_model_weight
            elif state_dict_type == "master_weight":
                if self._shm_master_weight is None:
                    self._meta_dict_master_weight, buffer_size = create_meta_dict(state_dict)
                    self._shm_master_weight = shared_memory.SharedMemory(create=True, size=buffer_size)
                shm_state_dict = self._shm_master_weight
                meta_dict = self._meta_dict_master_weight
                shared_save_flag = self._shared_save_master_weight_flag
                shared_save_path = self._shared_save_master_weight_path
                shared_save_signal_path = self._shared_save_master_weight_signal_path
                if self._process_master_weight is None:
                    self._process_master_weight = multiprocessing.Process(
                        target=self._save_file_async_in_process,
                        args=(
                            meta_dict,
                            self._shm_master_weight.name,
                            self._shared_save_master_weight_flag,
                            self._shared_save_master_weight_path,
                            self._shared_save_master_weight_signal_path,
                            self._lock,
                            "model_weight"
                            if "skip_save_model_weight" in self.args.unified_checkpoint_config
                            else state_dict_type,
                            self.global_rank,
                        ),
                    )
                    self._process_master_weight.start()
                process = self._process_master_weight
            elif state_dict_type == "optimizer_weight":
                if self._shm_optimizer_weight is None:
                    self._meta_dict_optim, buffer_size = create_meta_dict(state_dict)
                    self._shm_optimizer_weight = shared_memory.SharedMemory(create=True, size=buffer_size)
                shm_state_dict = self._shm_optimizer_weight
                meta_dict = self._meta_dict_optim
                shared_save_flag = self._shared_save_optimizer_flag
                shared_save_path = self._shared_save_optimizer_path
                shared_save_signal_path = self._shared_save_optimizer_signal_path
                if self._process_optimizer_weight is None:
                    self._process_optimizer_weight = multiprocessing.Process(
                        target=self._save_file_async_in_process,
                        args=(
                            meta_dict,
                            self._shm_optimizer_weight.name,
                            self._shared_save_optimizer_flag,
                            self._shared_save_optimizer_path,
                            self._shared_save_optimizer_signal_path,
                            self._lock,
                            state_dict_type,
                            self.global_rank,
                            ckpt_quant_stage,
                        ),
                    )
                    self._process_optimizer_weight.start()
                process = self._process_optimizer_weight

            while True:  # wait until no process is saving.
                flag_value = shared_save_flag[0]
                if flag_value == 0:
                    break
                if not process.is_alive():
                    raise RuntimeError(f"The process that saves {state_dict_type} has been killed unexpectedly.")
                time.sleep(0.5)
                logger.info(f"Wait for the previous save process to finish saving {state_dict_type}")
            # only save model weight or save master weight, we enter this loop.
            self._reset_and_update(shared_save_path, path)
            self._reset_and_update(shared_save_signal_path, signal_path)
            _traverse_copy_to_shm(state_dict, meta_dict, shm_state_dict.buf)
            with self._lock:
                shared_save_flag[0] = 1

    def _save_file_async_in_process(
        self,
        meta_dict,
        shm_name,
        shared_save_flag,
        shared_save_path,
        shared_save_signal_path,
        lock,
        state_dict_type,
        global_rank,
        ckpt_quant_stage="O0",
    ):
        shm = shared_memory.SharedMemory(name=shm_name)
        while True:
            flag_value = shared_save_flag[0]  # if process uses `spawn`, cannot read this value.
            if flag_value == -1:  # stop process
                break
            if flag_value == 0:  # nothing to save
                continue
            if flag_value == 1:  # need to save
                path = shared_save_path[:].decode("utf-8").rstrip("\x00")
                signal_path = shared_save_signal_path[:].decode("utf-8").rstrip("\x00")
                logger.info(f"Start to async save {path}")
                state_dict = _read_state_dict_from_shm(meta_dict, shm)  # numpy array
                if state_dict_type == "optimizer_weight" and ckpt_quant_stage != "O0":
                    state_dict = quant_unified_optimizer(
                        state_dict, state_dict_type, ckpt_quant_stage, async_save=True
                    )  # ckpt quantization
                safe_save_file(state_dict, path, {"format": "np"})
                del state_dict
                saved_signal_path = os.path.join(signal_path, f".{state_dict_type}.done.{global_rank}")
                paddle.save(global_rank, saved_signal_path)
                with lock:
                    shared_save_flag[0] = 0
            time.sleep(0.5)
        shm.close()

    def _reset_and_update(self, shared_array, new_value):
        # clear array
        for i in range(len(shared_array)):
            shared_array[i] = b"\0"
        # update array
        encoded_value = new_value.encode("utf-8")
        shared_array[: len(encoded_value)] = encoded_value

    def unlink_shared_memory(self):
        if not ("async_save" in self.args.unified_checkpoint_config):
            return

        if self._shared_save_model_flag is not None:
            while self._shared_save_model_flag[0] > 0:  # async process is saving
                if not self._process_model_weight.is_alive():
                    raise RuntimeError("The process that saves model_weight has been killed unexpectedly.")
                time.sleep(0.5)
            self._shared_save_model_flag[0] = -1
        if self._shared_save_master_weight_flag is not None:
            while self._shared_save_master_weight_flag[0] > 0:
                if not self._process_master_weight.is_alive():
                    raise RuntimeError("The process that saves master_weight has been killed unexpectedly.")
                time.sleep(0.5)
            self._shared_save_master_weight_flag[0] = -1
        if self._shared_save_optimizer_flag is not None:
            while self._shared_save_optimizer_flag[0] > 0:
                if not self._process_optimizer_weight.is_alive():
                    raise RuntimeError("The process that saves optimizer_weight has been killed unexpectedly.")
                time.sleep(0.5)
            self._shared_save_optimizer_flag[0] = -1

        if self._shm_model_weight is not None:
            self._shm_model_weight.close()
            self._shm_model_weight.unlink()
            self._shm_model_weight = None
        if self._shm_master_weight is not None:
            self._shm_master_weight.close()
            self._shm_master_weight.unlink()
            self._shm_master_weight = None
        if self._shm_optimizer_weight is not None:
            self._shm_optimizer_weight.close()
            self._shm_optimizer_weight.unlink()
            self._shm_optimizer_weight = None

        if paddle.distributed.get_world_size() > 1:
            dist.barrier()
