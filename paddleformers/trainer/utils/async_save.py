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
import multiprocessing
import os
import time

import paddle

from ...utils.log import logger


def _save_optimizer(obj, name_mapping, path, saved_signal_path, protocol):
    start_time = time.time()
    for k, v in obj.items():
        if k == "master_weights" and isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, paddle.Tensor):
                    vv.name = name_mapping["master_weights"][kk]
        else:
            if k in name_mapping and isinstance(v, paddle.Tensor):
                v.name = name_mapping[k]
    paddle.save(obj, path, protocol)
    # dump saved_signal
    with open(saved_signal_path, mode="w+") as f:
        f.write("1")
        f.flush()
        os.fsync(f.fileno())
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Async save optimizer took {elapsed_time:.6f} seconds to execute.")


class AsyncSaver:
    def __init__(self):
        self.context = multiprocessing.get_context("spawn")
        self.cpu_optimizer_state_dict = {}
        self.pool = self.context.Pool(1)
        self.result = None
        self.name_mapping = None

        atexit.register(self.shutdown)

    def run(self, optimizer_state_dict, path, saved_signal_path, protocol=4):
        logger.info(f"Started saving optimizer_state_dict to {os.path.abspath(path)}.")
        self._wait_for_previous_result()

        self._reset_state(path, saved_signal_path, protocol)
        self._process_optimizer_state_dict(optimizer_state_dict)

        self.result = self.pool.apply_async(
            _save_optimizer,
            args=(self.cpu_optimizer_state_dict, self.name_mapping, self.path, self.saved_signal_path, self.protocol),
        )

        logger.info("Finished launching saving optimizer_state_dict process")

    def _wait_for_previous_result(self):
        if self.result is not None:
            max_retries = 5
            for retries in range(max_retries):
                try:
                    self.result.get()
                    break
                except Exception as e:
                    if retries == max_retries - 1:
                        raise RuntimeError(f"Failed after {max_retries} retries during async save.")

                    time.sleep(1 + retries * 2)
                    logger.warning(f"An error occurred during async save: {e}. Retrying...")
                    self.result = self.pool.apply_async(
                        _save_optimizer,
                        args=(
                            self.cpu_optimizer_state_dict,
                            self.name_mapping,
                            self.path,
                            self.saved_signal_path,
                            self.protocol,
                        ),
                    )

            if self.result.ready() and not self.result.successful():
                raise RuntimeError("The previous async save task failed.")
        else:
            pass

    def _reset_state(self, path, saved_signal_path, protocol):
        self.cpu_optimizer_state_dict.clear()
        self.name_mapping = {"master_weights": {}}
        self.path = path
        self.saved_signal_path = saved_signal_path
        self.protocol = protocol

    def _process_optimizer_state_dict(self, optimizer_state_dict):
        for k, v in optimizer_state_dict.items():
            if k == "master_weights":
                self.cpu_optimizer_state_dict[k] = {}
                for kk, vv in v.items():
                    self.cpu_optimizer_state_dict[k][kk] = vv.pin_memory()
                    self.name_mapping[k][kk] = vv.name
            elif k == "LR_Scheduler":
                self.cpu_optimizer_state_dict[k] = copy.deepcopy(v)
            else:
                self.cpu_optimizer_state_dict[k] = v.pin_memory()
                self.name_mapping[k] = v.name
            paddle.device.synchronize()

    def shutdown(self):
        self.pool.close()
        self.pool.join()

    def __del__(self):
        self.shutdown()
