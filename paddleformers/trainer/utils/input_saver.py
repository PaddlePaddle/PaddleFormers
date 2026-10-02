# Copyright 2024-present the PaddlePaddle AI. All Rights Reserved.
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
from typing import Any, Dict, Optional

import numpy as np
import paddle

from paddleformers.utils.log import logger


class InputSaver:
    """
    Utility class for saving model inputs during training for debugging purposes.

    This class provides functionality to save training inputs in two modes:
    - NPZ mode: Save all inputs to a single .npz file
    - NPY mode: Save selected fields as separate .npy files

    Usage:
        # Enable input saving by setting environment variables:
        # export FLAGS_save_inputs_mode="npz" or "npy"
        # export FLAGS_save_inputs_dir="./saved_inputs/" (optional)
        # export FLAGS_save_inputs_fields="input_ids,labels" (for npy mode, optional)

        saver = InputSaver()
        saver.save_inputs(inputs, step=current_step)
    """

    def __init__(self):
        self.save_mode = os.getenv("FLAGS_save_inputs_mode", "").lower()
        self.save_dir = os.getenv("FLAGS_save_inputs_dir", f"./saved_inputs/{self.save_mode}")
        self.save_dir = os.path.join(self.save_dir, f"rank_{paddle.distributed.get_rank()}")
        self.fields_to_save = self._parse_fields_to_save()
        self._step_counter = 0
        # attention ! clear the dir first!
        if os.path.exists(self.save_dir):
            os.system(f"rm -rf {self.save_dir}")

    def _parse_fields_to_save(self) -> list:
        """Parse comma-separated fields to save from environment variable."""
        if self.save_mode != "npy":
            return []

        fields_env = os.getenv(
            "FLAGS_save_inputs_fields",
            "input_ids, labels, position_ids, image_grid_thw, pixel_values, input_features, pixel_values_videos, video_grid_thw, attn_mask_start_row_indices",
        )
        return [f.strip() for f in fields_env.split(",") if f.strip()]

    def _to_numpy(self, value: Any) -> Optional[np.ndarray]:
        """Convert tensor/array to numpy array."""
        if value is None:
            return None
        # print(value.dtype) # Optional
        if hasattr(value, "dtype") and value.dtype == paddle.bfloat16:
            value = paddle.cast(value, "float32")
        if hasattr(value, "cpu"):
            return value.cpu().numpy()
        elif hasattr(value, "numpy"):
            return value.numpy()
        elif isinstance(value, np.ndarray):
            return value
        elif isinstance(value, (list, tuple)):
            try:
                return np.array(value)
            except Exception:
                return None
        return None

    def save_inputs(self, inputs: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Save model inputs based on the configured save mode.

        Args:
            inputs: Dictionary of model inputs
            step: Current training step number. If None, auto-increments from 0.
        """
        if self.save_mode not in ("npz", "npy"):
            return

        if step is None:
            step = self._step_counter
            self._step_counter += 1

        os.makedirs(self.save_dir, exist_ok=True)
        if self.save_mode == "npz":
            self._save_as_npz(inputs, step)
        elif self.save_mode == "npy":
            self._save_as_npy(inputs, step)

    def _save_as_npz(self, inputs: Dict[str, Any], step: int) -> None:
        """Save all inputs to a single npz file."""
        dump_dict = {}
        for key, value in inputs.items():
            arr = self._to_numpy(value)
            if arr is not None:
                dump_dict[key] = arr

        if dump_dict:
            file_path = os.path.join(self.save_dir, f"inputs_step_{step}.npz")
            if os.path.exists(file_path):
                logger.warning(f"[Debug] File {file_path} already exists, skipping to avoid overwrite.")
                return
            np.savez(file_path, **dump_dict)
            logger.info(f"[Debug] Dumped all inputs to {file_path}")

    def _save_as_npy(self, inputs: Dict[str, Any], step: int) -> None:
        """Save selected fields as separate npy files."""
        for field in self.fields_to_save:
            if field in inputs and inputs[field] is not None:
                try:
                    arr = self._to_numpy(inputs[field])
                    if arr is not None:
                        file_path = os.path.join(self.save_dir, f"{step}_{field}.npy")
                        if os.path.exists(file_path):
                            logger.warning(f"[Debug] File {file_path} already exists, skipping to avoid overwrite.")
                            continue
                        np.save(file_path, arr)
                        logger.info(f"[Debug] Saved step {step} field '{field}' to {file_path}")
                except Exception as e:
                    logger.warning(f"[Debug] Failed to save {field} for step {step}: {e}")

    @classmethod
    def should_save(cls) -> bool:
        """Check if input saving is enabled."""
        save_mode = os.getenv("FLAGS_save_inputs_mode", "").lower()
        return save_mode in ("npz", "npy")


def save_inputs_decorator(func=None):
    """
    Decorator for automatically saving inputs before model forward pass.

    Usage:
        @save_inputs_decorator()
        def compute_loss(self, model, inputs, return_outputs=False):
            # Your compute_loss implementation
            pass

    Or as a method decorator with parameters:
        @save_inputs_decorator()
        def some_method(self, inputs):
            pass
    """

    def decorator(method):
        def wrapper(self, *args, **kwargs):
            # Find inputs in args or kwargs
            inputs = None
            if len(args) >= 2:
                inputs = args[1]  # Assuming compute_loss(self, model, inputs, ...)
            elif "inputs" in kwargs:
                inputs = kwargs["inputs"]

            # Save inputs if available and saving is enabled
            if inputs is not None and InputSaver.should_save():
                step = getattr(self, "state", None)
                if step is not None:
                    step = step.global_step
                else:
                    step = 0

                saver = InputSaver()
                saver.save_inputs(inputs, step)

            # Call the original method
            return method(self, *args, **kwargs)

        return wrapper

    if func is None:
        return decorator
    else:
        return decorator(func)
