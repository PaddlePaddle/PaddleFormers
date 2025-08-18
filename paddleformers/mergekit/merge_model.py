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
import math
import os
import shutil
from multiprocessing import Process

import numpy as np
import paddle
import paddle.distributed as dist
from safetensors import safe_open
from safetensors.numpy import save_file
from tqdm.auto import tqdm

from ..peft import LoRAConfig
from ..utils import device_guard
from ..utils.env import (
    LORA_WEIGHTS_NAME,
    PADDLE_MASTER_WEIGHTS_NAME,
    PADDLE_WEIGHTS_NAME,
    SAFE_MASTER_WEIGHTS_INDEX_NAME,
    SAFE_MASTER_WEIGHTS_NAME,
    SAFE_PEFT_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
)
from ..utils.log import logger
from ..utils.safetensors import fast_safe_open
from .merge_method import MergeMethod
from .merge_utils import divide_lora_key_list, divide_positions
from .sparsify_method import SparsifyMethod

SPARSIFY_MERGE_MAPPING = {
    "linear": (None, "linear"),
    "ties": ("trim", "ties"),
    "slerp": (None, "slerp"),
    "della_linear": ("magprune", "linear"),
    "della": ("magprune", "ties"),
    "dare_linear": ("dare", "linear"),
    "dare_ties": ("dare", "ties"),
}


class MergeModel:
    def __init__(self, merge_config):
        self.reset_merge_model(merge_config=merge_config)
        self.numpy_dtype_map = {"float32": 4, "float16": 2, "uint16": 2, "bfloat16": 2}
        self.is_peft = False

    def reset_merge_model(self, merge_config=None, merge_param_dict=None):
        self.is_cpu = "cpu" in paddle.device.get_device()
        if not self.is_cpu:
            if dist.get_world_size() > 1 and not paddle.distributed.is_initialized():
                dist.init_parallel_env()
        if merge_config is not None:
            self.merge_config = merge_config
        elif merge_param_dict is not None:
            for k, v in merge_param_dict.items():
                setattr(self.merge_config, k, v)
            self.merge_config.config_check()

        # map sparsify & merge type
        if self.merge_config.merge_method is not None:
            self.merge_config.sparsify_type, self.merge_config.merge_type = SPARSIFY_MERGE_MAPPING[
                self.merge_config.merge_method
            ]
        if self.merge_config.merge_type is None:
            raise ValueError("Either merge_type or merge_method must be specified.")
        # init merge method
        sparsify_method = SparsifyMethod(self.merge_config)
        self.merge_method = MergeMethod(merge_config, sparsify_method)

    def merge_model(self):
        if self.merge_config.lora_model_path is not None:
            self.merge_lora_model()
        else:
            if self.merge_config.tensor_type == "np" and not self.is_cpu:
                # Avoid memory allocated on GPU
                with device_guard():
                    self.mergekit()
            else:
                self.mergekit()
        if paddle.distributed.get_rank() == 0:
            self.copy_file()

    def copy_file(self):
        if self.merge_config.copy_file_list is not None:
            if self.merge_config.base_model_path is not None:
                src_path = self.merge_config.base_model_path
            else:
                src_path = self.merge_config.model_path_list[0]
            for file in self.merge_config.copy_file_list:
                src_file = os.path.join(src_path, file)
                dst_file = os.path.join(self.merge_config.output_path, file)
                if os.path.isfile(src_file):
                    shutil.copy2(src_file, dst_file)
                else:
                    logger.debug(f"Copy failed: {file} not found in {src_path}")

    def mergekit(self):
        # Check model file type
        file_type_list = []
        for model_path in self.merge_config.model_path_list:
            file_type_list.append(self.check_model_path(model_path))
        if self.merge_config.base_model_path is not None:
            file_type_list.append(self.check_model_path(self.merge_config.base_model_path))

        # Merge model (distinguish between safetensors and pdparams)
        if all(file_type == "safetensors" or file_type == "safetensors_without_index" for file_type in file_type_list):
            self.merge_safetensor_model(file_type_list)
        else:
            self.merge_mix_model(file_type_list)

    def merge_mix_model(self, file_type_list):
        # Load model state dict
        state_dict_list = []
        for i, model_path in enumerate(self.merge_config.model_path_list):
            state_dict_list.append(self.get_model_state_dict(model_path, file_type_list[i]))
        if self.merge_config.base_model_path is not None:
            state_dict_list.append(self.get_model_state_dict(self.merge_config.base_model_path, file_type_list[-1]))
        logger.info("Load all model state dict.")

        if not all(state_dict_list[0].keys() == state_dict.keys() for state_dict in state_dict_list):
            raise ValueError("State dict keys mismatch. Please make sure you load the correct weight file")

        # Merge state dict
        merge_state_dict = {}
        index = {"metadata": {"total_size": 0}, "weight_map": {}}

        key_list = list(state_dict_list[file_type_list.index("pdparams")].keys())
        model_num = len(state_dict_list)
        rank = dist.get_rank()
        positions = divide_positions(len(key_list), dist.get_world_size())
        local_keys = key_list[positions[rank] : positions[rank + 1]]
        for ii in range(len(positions) - 1):
            shard_file = f"{self.merge_config.merge_prefix}-{ii+1:05d}-of-{dist.get_world_size():05d}.safetensors"
            for key in key_list[positions[ii] : positions[ii + 1]]:
                index["weight_map"][key] = shard_file
                index["metadata"]["total_size"] += int(
                    np.prod(state_dict_list[0][key].shape) * self.numpy_dtype_map[str(state_dict_list[0][key].dtype)]
                )
        for key in tqdm(local_keys, desc="Merging tensor"):
            # Tensor preprocess
            is_bf16 = str(state_dict_list[0][key].dtype) in ["uint16", "bfloat16"]
            tensor_list = [state_dict_list[i].pop(key) for i in range(model_num)]
            tensor_mem = int(np.prod(tensor_list[0].shape) * self.numpy_dtype_map[str(tensor_list[0].dtype)]) / (
                1024**3
            )
            if self.merge_config.tensor_type == "pd" and tensor_mem > self.merge_config.max_tensor_mem:
                tensor_split_list = [
                    np.array_split(tensor, self.merge_config.split_pieces, axis=0) for tensor in tensor_list
                ]
                merge_split = []
                for sp in range(self.merge_config.split_pieces):
                    tensor_list = [tensor_split[sp] for tensor_split in tensor_split_list]
                    if is_bf16:
                        tensor_list = [
                            paddle.Tensor.__call__(tensor, zero_copy=True).astype("float32") for tensor in tensor_list
                        ]
                    else:
                        tensor_list = [paddle.Tensor.__call__(tensor, zero_copy=True) for tensor in tensor_list]
                    if self.merge_config.base_model_path is not None:
                        base_tensor = tensor_list.pop()
                        tensor_list = [tensor - base_tensor for tensor in tensor_list]
                    merge_tensor = self.merge_method.merge(tensor_list)
                    if self.merge_config.base_model_path is not None:
                        merge_tensor += base_tensor
                    if is_bf16:
                        merge_split.append(merge_tensor.astype("bfloat16").numpy())
                    else:
                        merge_split.append(merge_tensor.numpy())
                merge_state_dict[key] = np.concatenate(merge_split, axis=0)
            else:
                if self.merge_config.tensor_type == "pd":
                    if is_bf16:
                        tensor_list = [
                            paddle.Tensor.__call__(tensor, zero_copy=True).astype("float32") for tensor in tensor_list
                        ]
                    else:
                        tensor_list = [paddle.Tensor.__call__(tensor, zero_copy=True) for tensor in tensor_list]
                elif self.merge_config.tensor_type == "np" and is_bf16:
                    tensor_list = [
                        paddle.Tensor.__call__(tensor, zero_copy=True).astype("float32").numpy()
                        for tensor in tensor_list
                    ]

                if self.merge_config.base_model_path is not None:
                    base_tensor = tensor_list.pop()
                    tensor_list = [tensor - base_tensor for tensor in tensor_list]

                merge_tensor = self.merge_method.merge(tensor_list)
                if self.merge_config.base_model_path is not None:
                    merge_tensor += base_tensor
                if self.merge_config.tensor_type == "pd":
                    if is_bf16:
                        merge_state_dict[key] = merge_tensor.astype("bfloat16").numpy()
                    else:
                        merge_state_dict[key] = merge_tensor.numpy()
                elif self.merge_config.tensor_type == "np" and is_bf16:
                    # dtype==bfloat16: numpy(float32) -> paddle(float32) -> paddle(bfloat16) -> numpy(uint16)
                    merge_state_dict[key] = (
                        paddle.Tensor.__call__(merge_tensor, zero_copy=True).astype("bfloat16").numpy()
                    )

        logger.info("Merge tensors successfully.")
        # Save safetensor file
        save_file_name = os.path.join(
            self.merge_config.output_path,
            f"{self.merge_config.merge_prefix}-{rank+1:05d}-of-{dist.get_world_size():05d}.safetensors",
        )
        save_file(
            merge_state_dict,
            save_file_name,
            metadata={"format": "np"},
        )
        logger.info(f"Model weights saved in {save_file_name}.")
        # Save index file & merge config file
        if paddle.distributed.get_rank() == 0:
            save_index_file = os.path.join(self.merge_config.output_path, self.safe_index_name())
            with open(save_index_file, "w", encoding="utf-8") as f:
                f.write(json.dumps(index, indent=2) + "\n")
            logger.info(f"Model index file saved in {save_index_file}.")
            self.merge_config.save_pretrained(self.merge_config.output_path)

    def get_model_state_dict(self, model_path, file_type, key_list=None, file=None):
        if file_type == "safetensors":
            state_dict = {}
            with open(os.path.join(model_path, self.safe_index_name()), "r", encoding="utf-8") as f:
                index = json.load(f)
            if file is not None:
                with fast_safe_open(os.path.join(model_path, file), framework="np") as f:
                    for k in f.keys():
                        state_dict[k] = f.get_tensor(k)
            elif key_list is None:
                files = set(index["weight_map"].values())
                for file in files:
                    with fast_safe_open(os.path.join(model_path, file), framework="np") as f:
                        for k in f.keys():
                            state_dict[k] = f.get_tensor(k)
            else:
                file_map = {}
                for key in key_list:
                    if index["weight_map"][key] not in file_map:
                        file_map[index["weight_map"][key]] = [key]
                    else:
                        file_map[index["weight_map"][key]].append(key)
                for file in file_map.keys():
                    with fast_safe_open(os.path.join(model_path, file), framework="np") as f:
                        for k in file_map[file]:
                            state_dict[k] = f.get_tensor(k)
        elif file_type == "safetensors_without_index":
            state_dict = {}
            with fast_safe_open(os.path.join(model_path, self.safe_weight_name()), framework="numpy") as f:
                tgt_key_list = f.keys() if key_list is None else key_list
                for k in tgt_key_list:
                    state_dict[k] = f.get_tensor(k)
        elif file_type == "pdparams":
            state_dict = np.load(os.path.join(model_path, self.weight_name()), allow_pickle=True)
            if "StructuredToParameterName@@" in state_dict.keys():
                state_dict.pop("StructuredToParameterName@@")
        elif file_type == "lora_pdparams":
            state_dict = np.load(os.path.join(model_path, LORA_WEIGHTS_NAME), allow_pickle=True)
        elif file_type == "lora_safetensors":
            state_dict = {}
            with open(os.path.join(model_path, SAFE_PEFT_WEIGHTS_INDEX_NAME), "r", encoding="utf-8") as f:
                index = json.load(f)
            files = set(index["weight_map"].values())
            for file in files:
                with fast_safe_open(os.path.join(model_path, file), framework="np") as f:
                    for k in f.keys():
                        state_dict[k] = f.get_tensor(k)
        else:
            raise ValueError(f"Unsupported file_type: {file_type}")
        return state_dict

    def get_safetensor_index(self, model_path, file_type):
        if file_type == "safetensors":
            with open(os.path.join(model_path, self.safe_index_name()), "r", encoding="utf-8") as f:
                index = json.load(f)
        elif file_type == "safetensors_without_index":
            weight_map = {}
            total_size = 0
            with safe_open(os.path.join(model_path, self.safe_weight_name()), framework="numpy") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    total_size += int(np.prod(tensor.shape) * self.numpy_dtype_map[str(tensor.dtype)])
                    weight_map[key] = self.safe_weight_name()
            index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
        return index

    def merge_safetensor_model(self, file_type_list):
        # Load index
        index_list = []
        model_path_list = self.merge_config.model_path_list.copy()
        if self.merge_config.base_model_path is not None:
            model_path_list += [self.merge_config.base_model_path]

        for model_path, file_type in zip(model_path_list, file_type_list):
            index_list.append(self.get_safetensor_index(model_path, file_type))

        # Check index
        if not all(index_list[0]["metadata"]["total_size"] == index["metadata"]["total_size"] for index in index_list):
            raise ValueError("Weights total_size mismatch. Please make sure you load the correct weight file")
        if not all(index_list[0]["weight_map"].keys() == index["weight_map"].keys() for index in index_list):
            raise ValueError("Weights weight_map mismatch. Please make sure you load the correct weight file")
        # Initialize new index
        index = {}
        index["metadata"] = index_list[0]["metadata"]
        index["metadata"]["total_size"] = int(index["metadata"]["total_size"])
        index["weight_map"] = {}
        num = self.merge_config.n_process if self.is_cpu else dist.get_world_size()
        key_list = list(index_list[0]["weight_map"].keys())
        positions = divide_positions(len(key_list), num)
        if not self.is_cpu:
            rank = dist.get_rank()
            file_list = sorted(list(set(index_list[0]["weight_map"].values())))
            if file_type_list[0] == "safetensors" and len(file_list) >= num:
                positions = divide_positions(len(file_list), num)
                index["weight_map"] = index_list[0]["weight_map"]
                file_map = {}
                for key in key_list:
                    if index["weight_map"][key] not in file_map:
                        file_map[index["weight_map"][key]] = [key]
                    else:
                        file_map[index["weight_map"][key]].append(key)
                logger.info(f"Merging file list: {file_list[positions[rank] : positions[rank + 1]]}")
                for shard_file in file_list[positions[rank] : positions[rank + 1]]:
                    logger.info(f"Start merging tensor in {shard_file}")
                    if self.merge_config.tensor_type == "np":
                        self.shard_merge_np(file_map[shard_file], index_list, shard_file)
                    else:
                        self.shard_merge_pd(file_map[shard_file], index_list, shard_file)
            else:
                local_keys = key_list[positions[rank] : positions[rank + 1]]
                shard_file = (
                    f"{self.merge_config.merge_prefix}-{rank+1:05d}-of-{dist.get_world_size():05d}.safetensors"
                )
                if self.merge_config.tensor_type == "np":
                    self.shard_merge_np(local_keys, index_list, shard_file)
                else:
                    self.shard_merge_pd(local_keys, index_list, shard_file)

                for i in range(len(positions) - 1):
                    shard_file = (
                        f"{self.merge_config.merge_prefix}-{i+1:05d}-of-{dist.get_world_size():05d}.safetensors"
                    )
                    for k in key_list[positions[i] : positions[i + 1]]:
                        index["weight_map"][k] = shard_file
        else:
            threads = []
            for i in range(len(positions) - 1):
                shard_file = (
                    f"{self.merge_config.merge_prefix}-{i+1:05d}-of-{self.merge_config.n_process:05d}.safetensors"
                )
                t = Process(
                    target=self.shard_merge_np if self.merge_config.tensor_type == "np" else self.shard_merge_pd,
                    args=(
                        key_list[positions[i] : positions[i + 1]],  # key_list
                        index_list,  # index_list
                        shard_file,  # shard_file name
                    ),
                )
                threads.append(t)
                for k in key_list[positions[i] : positions[i + 1]]:
                    index["weight_map"][k] = shard_file

            for t in threads:
                t.start()
            for t in threads:
                t.join()
        # Save safe index file
        if paddle.distributed.get_rank() == 0:
            save_index_file = os.path.join(self.merge_config.output_path, self.safe_index_name())
            with open(save_index_file, "w", encoding="utf-8") as f:
                f.write(json.dumps(index, indent=2) + "\n")
            logger.info(f"Model index file saved in {save_index_file}.")

    def shard_merge_np(
        self,
        key_list,
        index_list,
        shard_file,
    ):
        merge_state_dict = {}
        for k in key_list:
            tensor_list = []
            for i, model_path in enumerate(self.merge_config.model_path_list):
                with fast_safe_open(os.path.join(model_path, index_list[i]["weight_map"][k]), framework="np") as w:
                    tensor = w.get_tensor(k)
                    dtype = tensor.dtype
                    # dtype==bfloat16: numpy(uint16) -> paddle(bfloat16) -> paddle(float32) -> numpy(float32)
                    if tensor.dtype == np.uint16:
                        tensor = paddle.Tensor.__call__(tensor, zero_copy=True).astype("float32").numpy()
                    tensor_list.append(tensor)
            if self.merge_config.base_model_path is not None:
                with fast_safe_open(
                    os.path.join(self.merge_config.base_model_path, index_list[-1]["weight_map"][k]),
                    framework="np",
                ) as w:
                    base_tensor = w.get_tensor(k)
                    if base_tensor.dtype == np.uint16:
                        base_tensor = paddle.Tensor.__call__(base_tensor, zero_copy=True).astype("float32").numpy()
                tensor_list = [tensor - base_tensor for tensor in tensor_list]
            merge_state_dict[k] = self.merge_method.merge(tensor_list)
            if self.merge_config.base_model_path is not None:
                merge_state_dict[k] += base_tensor
            # dtype==bfloat16: numpy(float32) -> paddle(float32) -> paddle(bfloat16) -> numpy(uint16)
            if dtype == np.uint16:
                merge_state_dict[k] = (
                    paddle.Tensor.__call__(merge_state_dict[k], zero_copy=True).astype("bfloat16").numpy()
                )
        save_file(
            merge_state_dict,
            os.path.join(self.merge_config.output_path, shard_file),
            metadata={"format": "np"},
        )

    def shard_merge_pd(
        self,
        key_list,
        index_list,
        shard_file,
    ):
        merge_state_dict = {}
        for k in tqdm(key_list, desc="Merging tensor"):
            tensor_list = []
            for i, model_path in enumerate(self.merge_config.model_path_list):
                with fast_safe_open(os.path.join(model_path, index_list[i]["weight_map"][k]), framework="np") as w:
                    tensor_list.append(w.get_tensor(k))
            if self.merge_config.base_model_path is not None:
                with fast_safe_open(
                    os.path.join(self.merge_config.base_model_path, index_list[-1]["weight_map"][k]),
                    framework="np",
                ) as w:
                    tensor_list.append(w.get_tensor(k))
            is_bf16 = str(tensor_list[0].dtype) in ["uint16", "bfloat16"]
            tensor_mem = int(np.prod(tensor_list[0].shape) * self.numpy_dtype_map[str(tensor_list[0].dtype)]) / (
                1024**3
            )
            if tensor_mem > self.merge_config.max_tensor_mem:
                tensor_split_list = [
                    np.array_split(tensor, self.merge_config.split_pieces, axis=0) for tensor in tensor_list
                ]
                merge_split = []
                for sp in range(self.merge_config.split_pieces):
                    tensor_list = [tensor_split[sp] for tensor_split in tensor_split_list]
                    if is_bf16:
                        tensor_list = [
                            paddle.Tensor.__call__(tensor, zero_copy=True).astype("float32") for tensor in tensor_list
                        ]
                    else:
                        tensor_list = [paddle.Tensor.__call__(tensor, zero_copy=True) for tensor in tensor_list]
                    if self.merge_config.base_model_path is not None:
                        base_tensor = tensor_list.pop()
                        tensor_list = [tensor - base_tensor for tensor in tensor_list]
                    merge_tensor = self.merge_method.merge(tensor_list)
                    if self.merge_config.base_model_path is not None:
                        merge_tensor += base_tensor
                    if is_bf16:
                        merge_split.append(merge_tensor.astype("bfloat16").numpy())
                    else:
                        merge_split.append(merge_tensor.numpy())
                merge_state_dict[k] = np.concatenate(merge_split, axis=0)
            else:
                if is_bf16:
                    tensor_list = [
                        paddle.Tensor.__call__(tensor, zero_copy=True).astype("float32") for tensor in tensor_list
                    ]
                else:
                    tensor_list = [paddle.Tensor.__call__(tensor, zero_copy=True) for tensor in tensor_list]
                if self.merge_config.base_model_path is not None:
                    base_tensor = tensor_list.pop()
                    tensor_list = [tensor - base_tensor for tensor in tensor_list]
                merge_tensor = self.merge_method.merge(tensor_list)
                if self.merge_config.base_model_path is not None:
                    merge_tensor += base_tensor
                if is_bf16:
                    merge_state_dict[k] = merge_tensor.astype("bfloat16").numpy()
                else:
                    merge_state_dict[k] = merge_tensor.numpy()
        logger.info("Merge tensors successfully.")
        save_file_name = os.path.join(self.merge_config.output_path, shard_file)
        save_file(
            merge_state_dict,
            save_file_name,
            metadata={"format": "np"},
        )
        logger.info(f"Model weights saved in {save_file_name}.")

    def check_model_path(self, model_path, lora_merge=False):
        if os.path.exists(os.path.join(model_path, self.safe_index_name())):
            file_type = "safetensors"
        elif os.path.exists(os.path.join(model_path, self.safe_weight_name())):
            file_type = "safetensors_without_index"
        elif os.path.exists(os.path.join(model_path, self.weight_name())):
            file_type = "pdparams"
        else:
            raise ValueError(
                f"Please check path {model_path} is correct. Support safetensors and pdparams only in complete parameter format (not TP or PP format) only."
            )
        return file_type

    def check_lora_model_path(self, model_path):
        if os.path.exists(os.path.join(model_path, SAFE_PEFT_WEIGHTS_INDEX_NAME)):
            file_type = "lora_safetensors"
        elif os.path.exists(os.path.join(model_path, LORA_WEIGHTS_NAME)):
            file_type = "lora_pdparams"
        else:
            raise ValueError(
                f"Please check lora path {model_path} is correct. Support safetensors and pdparams only in complete parameter format (not TP or PP format) only."
            )
        return file_type

    def weight_name(self):
        if self.merge_config.merge_prefix == "model":
            return PADDLE_WEIGHTS_NAME
        else:
            return PADDLE_MASTER_WEIGHTS_NAME

    def safe_weight_name(self):
        if self.merge_config.merge_prefix == "model":
            return SAFE_WEIGHTS_NAME
        else:
            return SAFE_MASTER_WEIGHTS_NAME

    def safe_index_name(self):
        if self.merge_config.merge_prefix == "model":
            return SAFE_WEIGHTS_INDEX_NAME
        else:
            return SAFE_MASTER_WEIGHTS_INDEX_NAME

    def merge_lora_model(self):
        # Check model file type
        file_type_list = []
        file_type_list.append(self.check_lora_model_path(self.merge_config.lora_model_path))
        file_type_list.append(self.check_model_path(self.merge_config.base_model_path))
        # Merge model (distinguish between safetensors and pdparams)
        if "safetensors" in file_type_list[-1]:
            self.merge_safetensor_lora_model(file_type_list)
        else:
            self.merge_pdparams_lora_model(file_type_list)

    def shard_lora_merge(self, base_index, shard_file, lora_config, file_type_list, key_list=None, file=None):
        merge_state_dict = {}
        lora_state_dict = self.get_model_state_dict(self.merge_config.lora_model_path, file_type_list[0])
        logger.info("Load LoRA weight successfully.")
        base_state_dict = self.get_model_state_dict(
            self.merge_config.base_model_path, file_type_list[1], key_list=key_list, file=file
        )
        logger.info("Load model weight successfully.")
        if not lora_config.rslora:
            scaling = lora_config.lora_alpha / lora_config.r
        else:
            scaling = lora_config.lora_alpha / math.sqrt(lora_config.r)

        model_key_list = list(base_state_dict.keys())
        for k in tqdm(model_key_list, desc="Merging tensor"):
            if lora_state_dict is not None and k in lora_state_dict.keys():
                tensor = lora_state_dict.pop(k)
            else:
                tensor = base_state_dict.pop(k)
            if "weight" in k:
                lora_A_key, lora_B_key = k.replace("weight", "lora_A"), k.replace("weight", "lora_B")
                lora_A_tensor = None
                if lora_state_dict is not None and lora_A_key in lora_state_dict.keys():
                    lora_A_tensor, lora_B_tensor = lora_state_dict.pop(lora_A_key), lora_state_dict.pop(lora_B_key)
                    is_bf16 = str(tensor.dtype) in ["uint16", "bfloat16"]
                    tensor = paddle.Tensor.__call__(tensor, zero_copy=True)
                    lora_A_tensor = paddle.Tensor.__call__(lora_A_tensor, zero_copy=True)
                    lora_B_tensor = paddle.Tensor.__call__(lora_B_tensor, zero_copy=True)
                    if self.is_cpu and is_bf16:
                        tensor = tensor.astype("float32")
                        lora_A_tensor = lora_A_tensor.astype("float32")
                        lora_B_tensor = lora_B_tensor.astype("float32")
                        tensor += lora_A_tensor @ lora_B_tensor * scaling
                        tensor = tensor.astype("bfloat16").numpy()
                    else:
                        tensor += lora_A_tensor @ lora_B_tensor * scaling
                        tensor = tensor.numpy()
            merge_state_dict[k] = tensor

        logger.info("Merge tensors successfully.")
        save_file_name = os.path.join(self.merge_config.output_path, shard_file)
        save_file(
            merge_state_dict,
            save_file_name,
            metadata={"format": "np"},
        )
        logger.info(f"Model weights saved in {save_file_name}.")

    def merge_safetensor_lora_model(self, file_type_list):
        # Load index
        base_index = self.get_safetensor_index(self.merge_config.base_model_path, file_type_list[-1])
        lora_config = LoRAConfig.from_pretrained(self.merge_config.lora_model_path)

        # Initialize new index
        index = {}
        index["metadata"] = base_index["metadata"]
        index["metadata"]["total_size"] = int(index["metadata"]["total_size"])
        index["weight_map"] = {}

        # LoRA Merge
        key_list = list(base_index["weight_map"].keys())
        if not self.is_cpu:
            rank = dist.get_rank()
            file_list = sorted(list(set(base_index["weight_map"].values())))
            if file_type_list[-1] == "safetensors" and len(file_list) >= dist.get_world_size():
                positions = divide_positions(len(file_list), dist.get_world_size())
                logger.info(f"Merging file list: {file_list[positions[rank] : positions[rank + 1]]}")
                for shard_file in file_list[positions[rank] : positions[rank + 1]]:
                    logger.info(f"Start merging tensor in {shard_file}")
                    self.shard_lora_merge(base_index, shard_file, lora_config, file_type_list, file=shard_file)
                index["weight_map"] = base_index["weight_map"]
            else:
                divided_key_list = divide_lora_key_list(key_list, dist.get_world_size(), lora_config)
                local_keys = divided_key_list[rank]
                shard_file = (
                    f"{self.merge_config.merge_prefix}-{rank+1:05d}-of-{dist.get_world_size():05d}.safetensors"
                )
                self.shard_lora_merge(base_index, shard_file, lora_config, file_type_list, key_list=local_keys)
                for i in range(len(divided_key_list)):
                    shard_file = (
                        f"{self.merge_config.merge_prefix}-{i+1:05d}-of-{dist.get_world_size():05d}.safetensors"
                    )
                    for k in divided_key_list[i]:
                        index["weight_map"][k] = shard_file
        else:
            divided_key_list = divide_lora_key_list(key_list, self.merge_config.n_process, lora_config)
            threads = []
            for i in range(len(divided_key_list)):
                shard_file = (
                    f"{self.merge_config.merge_prefix}-{i+1:05d}-of-{self.merge_config.n_process:05d}.safetensors"
                )
                t = Process(
                    target=self.shard_lora_merge,
                    args=(
                        base_index,  # base index
                        shard_file,  # shard_file name
                        lora_config,
                        file_type_list,
                        divided_key_list[i],  # key_list
                    ),
                )
                threads.append(t)
                for k in divided_key_list[i]:
                    index["weight_map"][k] = shard_file

            for t in threads:
                t.start()
            for t in threads:
                t.join()

        # Save safe index file
        if paddle.distributed.get_rank() == 0:
            save_index_file = os.path.join(self.merge_config.output_path, self.safe_index_name())
            with open(save_index_file, "w", encoding="utf-8") as f:
                f.write(json.dumps(index, indent=2) + "\n")
            logger.info(f"Model index file saved in {save_index_file}.")
            self.merge_config.save_pretrained(self.merge_config.output_path)

    def merge_pdparams_lora_model(self, file_type_list):
        # Load & check state dict
        lora_state_dict = self.get_model_state_dict(self.merge_config.lora_model_path, file_type_list[0])
        logger.info("Load LoRA weight successfully.")
        base_state_dict = self.get_model_state_dict(self.merge_config.base_model_path, file_type_list[1])
        logger.info("Load model weight successfully.")
        for key in lora_state_dict.keys():
            if "lora_A" in key:
                if key.replace("lora_A", "lora_B") not in lora_state_dict.keys():
                    raise ValueError(f"{key} is not paired with {key.replace('lora_A', 'lora_B')}")
                if key.replace("lora_A", "weight") not in base_state_dict.keys():
                    raise ValueError(f'{key.replace("lora_A", "weight")} does not exist in base model.')

        # Load lora config
        lora_config = LoRAConfig.from_pretrained(self.merge_config.lora_model_path)
        if not lora_config.rslora:
            scaling = lora_config.lora_alpha / lora_config.r
        else:
            scaling = lora_config.lora_alpha / math.sqrt(lora_config.r)

        # Create index
        merge_state_dict = {}
        index = {"metadata": {"total_size": 0}, "weight_map": {}}
        key_list = list(base_state_dict.keys())
        positions = divide_positions(len(key_list), dist.get_world_size())
        for ii in range(len(positions) - 1):
            shard_file = f"{self.merge_config.merge_prefix}-{ii+1:05d}-of-{dist.get_world_size():05d}.safetensors"
            for key in key_list[positions[ii] : positions[ii + 1]]:
                index["weight_map"][key] = shard_file
                index["metadata"]["total_size"] += int(
                    np.prod(base_state_dict[key].shape) * self.numpy_dtype_map[str(base_state_dict[key].dtype)]
                )

        # Merge state dict
        rank = dist.get_rank()
        local_keys = key_list[positions[rank] : positions[rank + 1]]
        for k in tqdm(local_keys, desc="Merging tensor"):
            if k in lora_state_dict.keys():
                tensor = lora_state_dict[k]
            else:
                tensor = base_state_dict[k]
            if "weight" in k:
                lora_A_key, lora_B_key = k.replace("weight", "lora_A"), k.replace("weight", "lora_B")
                if lora_A_key in lora_state_dict.keys():
                    lora_A_tensor = lora_state_dict[lora_A_key]
                    lora_B_tensor = lora_state_dict[lora_B_key]
                    is_bf16 = str(tensor.dtype) in ["uint16", "bfloat16"]

                    tensor = paddle.Tensor.__call__(tensor, zero_copy=True)
                    lora_A_tensor = paddle.Tensor.__call__(lora_A_tensor, zero_copy=True)
                    lora_B_tensor = paddle.Tensor.__call__(lora_B_tensor, zero_copy=True)
                    if self.is_cpu and is_bf16:
                        tensor = tensor.astype("float32")
                        lora_A_tensor = lora_A_tensor.astype("float32")
                        lora_B_tensor = lora_B_tensor.astype("float32")
                        tensor += lora_A_tensor @ lora_B_tensor * scaling
                        tensor = tensor.astype("bfloat16")
                    else:
                        tensor += lora_A_tensor @ lora_B_tensor * scaling
                    tensor = tensor.numpy()
            merge_state_dict[k] = tensor

        # Save safetensor file
        save_file_name = os.path.join(
            self.merge_config.output_path,
            f"{self.merge_config.merge_prefix}-{rank+1:05d}-of-{dist.get_world_size():05d}.safetensors",
        )
        save_file(
            merge_state_dict,
            save_file_name,
            metadata={"format": "np"},
        )
        logger.info(f"Model weights saved in {save_file_name}.")
        # Save index file & merge config file
        if paddle.distributed.get_rank() == 0:
            save_index_file = os.path.join(self.merge_config.output_path, self.safe_index_name())
            with open(save_index_file, "w", encoding="utf-8") as f:
                f.write(json.dumps(index, indent=2) + "\n")
            logger.info(f"Model index file saved in {save_index_file}.")
            self.merge_config.save_pretrained(self.merge_config.output_path)
