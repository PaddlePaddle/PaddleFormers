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
FlexAsyncSaver: Two-phase async checkpoint saving for flex_checkpoint format.

Phase 1 (synchronous, all ranks participate):
  - Obtain sharded_state_dict references
  - Model state: GPU → CPU pinned copy
  - Optimizer state: zero-copy reference to existing CPU pinned data
  - allgather metadata coordination + dedup planning

Phase 2 (asynchronous, background thread):
  - Write .metadata files
  - paddle.save .distcp shard files
  - Write saved_signal completion marker
"""

import os
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional

import paddle
import paddle.distributed as dist

from ...utils.log import logger


@dataclass
class FlexSavePlan:
    """Holds all data and paths needed for background I/O write."""

    # Model state (CPU pinned copies from GPU)
    model_local_state: Optional[Dict] = None
    model_path: str = ""
    model_metadata: object = None
    model_file_name: str = ""

    # Optimizer state (zero-copy reference to CPU pinned data)
    optimizer_local_state: Optional[Dict] = None
    optimizer_path: str = ""
    optimizer_metadata: object = None
    optimizer_file_name: str = ""

    # Master weights (zero-copy reference to CPU pinned data)
    master_weights_local_state: Optional[Dict] = None
    master_weights_path: str = ""
    master_weights_metadata: object = None
    master_weights_file_name: str = ""

    # Common
    saved_signal_path: str = ""
    save_replicas: bool = False


class FlexAsyncSaver:
    """Manages the lifecycle of async flex_checkpoint saving.

    Usage:
        saver = FlexAsyncSaver()
        # At save time (non-blocking after plan phase):
        saver.save_async(trainer, output_dir)
        # Before next optimizer.step():
        saver.wait_for_completion()
    """

    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self._done_event = threading.Event()
        self._done_event.set()  # Initially: no save in progress
        self._error: Optional[Exception] = None

    @property
    def is_saving(self) -> bool:
        """Whether a background save is currently in progress."""
        return not self._done_event.is_set()

    def wait_for_completion(self, timeout: float = None):
        """Block until the background save completes.

        Should be called before optimizer.step() to ensure data consistency.
        Raises RuntimeError if the background save encountered an error.
        """
        if self._done_event.is_set():
            return  # No save in progress
        logger.info("[FlexAsyncSaver] Waiting for previous async save to complete...")
        start = time.time()
        self._done_event.wait(timeout=timeout)
        elapsed = time.time() - start
        if elapsed > 0.1:
            logger.info(f"[FlexAsyncSaver] Wait took {elapsed:.2f}s")
        if self._error:
            err = self._error
            self._error = None
            raise RuntimeError(f"[FlexAsyncSaver] Background save failed: {err}") from err

    def save_async(self, trainer, output_dir: str):
        """Main entry point: plan (synchronous) + execute (asynchronous).

        Args:
            trainer: PaddleFormers Trainer instance
            output_dir: checkpoint output directory
        """
        # Ensure previous save is complete before starting a new one
        self.wait_for_completion()

        # Phase 1: synchronous plan
        plan = self._plan(trainer, output_dir)

        # Phase 2: launch background I/O
        self._done_event.clear()
        self._error = None
        self._thread = threading.Thread(
            target=self._execute,
            args=(plan,),
            daemon=True,
        )
        self._thread.start()

    def _plan(self, trainer, output_dir: str) -> FlexSavePlan:
        """Phase 1: Data preparation + distributed coordination (synchronous).

        All ranks must call this method together (contains collective ops).
        """
        from paddle.distributed import ShardedWeight

        from paddleformers.utils.env import (
            MASTER_WEIGHT_DIC,
            MODEL_STATE_DIC,
            OPTIMIZER_STATE_DIC,
        )

        plan = FlexSavePlan(
            saved_signal_path=os.path.join(output_dir, f"saved_signal_{dist.get_rank()}"),
            save_replicas=trainer.args.replicate_saved_into_local,
        )

        # --- Model State: GPU → CPU pinned copy ---
        model_sharded = trainer.model.sharded_state_dict()
        for key, sw in model_sharded.items():
            if isinstance(sw, ShardedWeight):
                sw.local_tensor = paddle.Tensor(sw.local_tensor)

        plan.model_path = os.path.join(output_dir, MODEL_STATE_DIC)
        os.makedirs(plan.model_path, exist_ok=True)
        plan.model_local_state, plan.model_metadata, plan.model_file_name = self._coordinate(
            model_sharded, plan.model_path, plan.save_replicas
        )

        # Copy model tensors from GPU to CPU pinned memory
        for key in list(plan.model_local_state.keys()):
            t = plan.model_local_state[key]
            if isinstance(t, paddle.Tensor) and t.place.is_gpu_place():
                plan.model_local_state[key] = t.pin_memory()
        paddle.device.synchronize()

        # --- Optimizer State: zero-copy reference (already on CPU pinned) ---
        model_sharded_for_opt = trainer.model.sharded_state_dict()
        opt_sharded = trainer.optimizer.sharded_state_dict(model_sharded_for_opt)

        opt_states = {}
        mw_states = {}
        for k, v in opt_sharded.items():
            if k.endswith(".w_0"):
                mw_states[k] = v
            else:
                opt_states[k] = v

        plan.optimizer_path = os.path.join(output_dir, OPTIMIZER_STATE_DIC)
        os.makedirs(plan.optimizer_path, exist_ok=True)
        plan.optimizer_local_state, plan.optimizer_metadata, plan.optimizer_file_name = self._coordinate(
            opt_states, plan.optimizer_path, plan.save_replicas
        )

        plan.master_weights_path = os.path.join(output_dir, MASTER_WEIGHT_DIC)
        os.makedirs(plan.master_weights_path, exist_ok=True)
        (
            plan.master_weights_local_state,
            plan.master_weights_metadata,
            plan.master_weights_file_name,
        ) = self._coordinate(mw_states, plan.master_weights_path, plan.save_replicas)

        return plan

    def _coordinate(self, state_dict, path, save_replicas):
        """Execute Phase 1+2 of dist.save_state_dict: metadata extraction + allgather coordination + dedup.

        Returns:
            (local_state_dict, metadata, file_name)
        """
        from paddle.distributed.flex_checkpoint.dcp.metadata import Metadata
        from paddle.distributed.flex_checkpoint.dcp.save_state_dict import (
            balanced_dedup_key_in_dict,
            dedup_key_in_dict,
            dedup_tensor,
        )
        from paddle.distributed.flex_checkpoint.dcp.utils import (
            LocalTensorIndex,
            check_unique_id,
            extract_tensor_metadata,
            flatten_state_dict,
            get_max_id,
            merge_state_dict_metadata,
        )

        flat_state_dict, mapping = flatten_state_dict(state_dict)
        use_dist = paddle.distributed.get_world_size() > 1

        max_unique_id = get_max_id(path)
        unique_id = 0 if max_unique_id is None else max_unique_id
        if use_dist:
            check_unique_id(unique_id, None)

        file_name = f"{paddle.distributed.get_rank()}_{unique_id}.distcp"

        # Extract local metadata
        metadata = Metadata()
        local_state_dict = {}
        local_state_dict_metadata = {}
        local_storage_metadata = {}

        for key, val in flat_state_dict.items():
            local_tensor, local_tensor_metadata = extract_tensor_metadata(val)
            if local_tensor is None and local_tensor_metadata is None:
                continue
            local_state_dict[key] = local_tensor
            local_state_dict_metadata[key] = local_tensor_metadata
            local_storage_metadata[
                LocalTensorIndex(
                    tensor_key=key,
                    global_offset=local_tensor_metadata.global_offset,
                    is_flattened=local_tensor_metadata.is_flattened,
                    flattened_range=local_tensor_metadata.flattened_range,
                    local_shape=local_tensor_metadata.local_shape,
                )
            ] = file_name

        # allgather coordination
        global_state_dict_metadata = []
        global_storage_metadata = []
        global_flatten_mapping = []
        if use_dist:
            dist.all_gather_object(global_state_dict_metadata, local_state_dict_metadata)
            dist.all_gather_object(global_storage_metadata, local_storage_metadata)
            dist.all_gather_object(global_flatten_mapping, mapping)
        else:
            global_state_dict_metadata.append(local_state_dict_metadata)
            global_storage_metadata.append(local_storage_metadata)
            global_flatten_mapping.append(mapping)

        metadata.state_dict_metadata = merge_state_dict_metadata(global_state_dict_metadata)
        metadata.storage_metadata = balanced_dedup_key_in_dict(global_storage_metadata, save_replicas=save_replicas)
        metadata.flat_mapping = dedup_key_in_dict(global_flatten_mapping)

        # Dedup: remove tensors assigned to other ranks
        if not save_replicas:
            dedup_tensor(local_state_dict, local_storage_metadata, metadata.storage_metadata)

        return local_state_dict, metadata, file_name

    def _execute(self, plan: FlexSavePlan):
        """Phase 2: Background thread executes pure I/O writes."""
        try:
            # Write model state
            if plan.model_local_state:
                self._write_shard(plan.model_local_state, plan.model_path, plan.model_metadata, plan.model_file_name)

            # Write optimizer state
            if plan.optimizer_local_state:
                self._write_shard(
                    plan.optimizer_local_state, plan.optimizer_path, plan.optimizer_metadata, plan.optimizer_file_name
                )

            # Write master weights
            if plan.master_weights_local_state:
                self._write_shard(
                    plan.master_weights_local_state,
                    plan.master_weights_path,
                    plan.master_weights_metadata,
                    plan.master_weights_file_name,
                )

            # Write completion signal
            with open(plan.saved_signal_path, "w") as f:
                f.write("1")

            logger.info("[FlexAsyncSaver] Background save completed successfully.")

        except Exception as e:
            logger.error(f"[FlexAsyncSaver] Background save failed: {e}")
            self._error = e
        finally:
            # Release CPU pinned memory as soon as I/O is done.
            plan.model_local_state = None
            plan.optimizer_local_state = None
            plan.master_weights_local_state = None
            plan.model_metadata = None
            plan.optimizer_metadata = None
            plan.master_weights_metadata = None
            self._done_event.set()

    @staticmethod
    def _write_shard(local_state_dict, path, metadata, file_name):
        """Write metadata + tensor shard file for one component."""
        from paddle.distributed.flex_checkpoint.dcp.utils import (
            get_max_id,
            write_to_file_if_empty,
        )

        max_id = get_max_id(path)
        unique_id = 0 if max_id is None else max_id
        write_to_file_if_empty(metadata, os.path.join(path, f"{unique_id}.metadata"))
        paddle.save(local_state_dict, os.path.join(path, file_name))
