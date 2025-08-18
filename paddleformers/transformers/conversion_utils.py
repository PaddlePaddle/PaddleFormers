# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import annotations

import inspect
import json
import os
import re
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np
import paddle
from numpy import allclose, ndarray, transpose
from paddle import Tensor
from paddle.nn import Layer

from ..quantization.quantization_utils import parse_weight_quantize_algo
from ..utils.distributed import distributed_allgather, distributed_gather
from ..utils.env import CONFIG_NAME, PYTORCH_WEIGHTS_NAME
from ..utils.import_utils import (
    is_package_available,
    is_torch_available,
    is_transformers_available,
)
from ..utils.log import logger
from ..utils.serialization import load_torch
from ..utils.tools import get_env_device

if TYPE_CHECKING:
    from .configuration_utils import PretrainedConfig
    from .model_utils import PretrainedModel

from ..utils import device_guard

# the type hinting for pytorch model & layer & tensor
Module = TypeVar("Module")
PytorchTensor = TypeVar("PytorchTensor")


def add_quant_mapping(name_action_mappings, quantization_config):
    if isinstance(quantization_config.weight_quantize_algo, str):
        post_quantize = quantization_config.weight_quantize_algo in [
            "weight_only_int4",
            "weight_only_int8",
        ]
    elif isinstance(quantization_config.weight_quantize_algo, dict):
        post_quantize = any(
            key in ["weight_only_int4", "weight_only_int8"] for key in quantization_config.weight_quantize_algo.keys()
        )
    else:
        post_quantize = False
    if not post_quantize:
        mapping_keys = list(name_action_mappings.keys())
        pattern = r"^(?:.*\.)?layers(\.[a-zA-Z0-9_]+)*\.weight$"
        for key in mapping_keys:
            if re.match(pattern, key):
                weight_quantize_algo = parse_weight_quantize_algo(quantization_config, key)
                quant_key = key.replace("weight", "quant_weight")
                weight_scale_key = key.replace("weight", "weight_scale")
                fn = name_action_mappings.pop(key)
                name_action_mappings[quant_key] = fn
                if (
                    weight_quantize_algo in ["a8w8linear", "a8w4linear"]
                    and "is_column" in fn.keywords
                    and fn.keywords["is_column"]
                ):
                    name_action_mappings[weight_scale_key] = partial(
                        fn.func, *fn.args, **{**fn.keywords, "is_column": True}
                    )

    return name_action_mappings


def tensor_summary(tensor: Union[str, Tensor, PytorchTensor, tuple, list, ndarray]):
    """get summary of values which can be some of different values

    Args:
        tensor (ndarray): the source data of tensor which can be: string, Paddle Tensor, Pytorch Tensor, tuple/list tensor, ndarray

    Returns:
        str: the summary info
    """
    if tensor is None:
        return "None"

    if isinstance(tensor, str):
        return tensor

    # Modeling Output from paddleformers/transformers
    if isinstance(tensor, dict):
        tensor = list(tensor.values())

    if isinstance(tensor, (tuple, list)):
        infos = []
        for item in tensor:
            infos.append(tensor_summary(item))
        return "\n".join(infos)

    # check whether contains `.numpy` method
    # numpy is wrapped from C++, so it will be the `builtin` method
    if hasattr(tensor, "numpy") and inspect.isbuiltin(getattr(tensor, "numpy")):
        tensor = tensor.detach().cpu().numpy()
        tensor = np.reshape(tensor, [-1])
        top_3_tensor = str(tensor[1:4])
        return top_3_tensor

    return str(tensor)


def compare_model_weights(first_state_dict: Dict[str, ndarray], second_state_dict: Dict[str, ndarray]) -> List[str]:
    """compare the values of two state_dict.
       This function has an assumption: the keys between `first_state_dict` and `second_state_dict` are exactly the same.

    Args:
        first_state_dict (Dict[str, ndarray]): first state_dict
        second_state_dict (Dict[str, ndarray]): second state_dict

    Returns:
        mismatched keys (List[str]): the mismatched keys of state_dict because of some reason
    """
    mismatched_keys = []
    for key in first_state_dict.keys():
        is_close = np.allclose(first_state_dict[key], second_state_dict[key], atol=1e-4)
        if not is_close:
            mismatched_keys.append(key)
    return mismatched_keys


def state_dict_contains_prefix(state_dict: Dict[str, ndarray], prefix: str) -> bool:
    """check whether state-dict contains `prefix`"""
    prefix_count = sum([1 for key in state_dict.keys() if key.startswith(prefix)])
    return prefix_count > 0


def init_name_mappings(mappings: list[StateDictNameMapping]) -> list[StateDictNameMapping]:
    """init name mapping which are simple mappings"""
    for index in range(len(mappings)):
        sub_mapping = mappings[index]

        # if sub_mapping is `str`, so repeat it. eg: [ "word_embedding.weight", ["layer_norm", "LayerNorm"] ]
        if isinstance(sub_mapping, str):
            sub_mapping = [sub_mapping]

        if len(sub_mapping) == 1:
            sub_mapping = sub_mapping * 2

        elif sub_mapping[1] is None:
            sub_mapping[1] = sub_mapping[0]

        mappings[index] = sub_mapping


class StateDictKeysChecker:
    """State Dict Keys Checker"""

    def __init__(
        self,
        model_or_state_dict: Union[Layer, Dict[str, ndarray]],
        loaded_state_dict: Dict[str, ndarray],
        check_shape: bool = True,
        base_model_prefix: Optional[str] = None,
        ignore_keys: Optional[List[str]] = None,
    ) -> None:
        if isinstance(model_or_state_dict, Layer):
            base_model_prefix = base_model_prefix or getattr(model_or_state_dict, "base_model_prefix", None)
            model_or_state_dict = {
                key: value.detach().cpu().numpy() for key, value in model_or_state_dict.state_dict().items()
            }

        self.model_state_dict = model_or_state_dict
        self.loaded_state_dict = loaded_state_dict
        self.check_shape = check_shape
        self.ignore_keys = ignore_keys or []
        self.base_model_prefix = base_model_prefix

    def change_base_downstream_mismatched_keys(self):
        """when model is base-model, loaded state-dict is downstream-model,
        it should re-change the downstream state-dict.

        eg: init `BertModel` with `BertForTokenClassification` state-dict

        # <model-base>-<loaded-downstream>
        # remove base-prefix
        """
        for key in list(self.loaded_state_dict.keys()):
            if key.startswith(self.base_model_prefix):
                value = self.loaded_state_dict.pop(key)
                new_key = key.replace(f"{self.base_model_prefix}.", "")
                self.loaded_state_dict[new_key] = value

    def change_downstream_base_mismatched_keys(self):
        """when model is downstream-model, loaded state-dict is base-model,
        it should re-change the downstream state-dict.

        eg: init `BertModel` with `BertForTokenClassification` state-dict

        # <model>-<loaded>: <downstream>-<base>
        """
        for key in list(self.model_state_dict.keys()):
            if key.startswith(self.base_model_prefix):

                key_in_loaded = key.replace(f"{self.base_model_prefix}.", "")
                assert key_in_loaded in self.loaded_state_dict
                # check loaded keys
                value = self.loaded_state_dict.pop(key_in_loaded)
                self.loaded_state_dict[key] = value

    def change_diff_keys(self) -> List[str]:
        """change the loaded-state-dict by base-model & base_model_prefix

        Returns:
            List[str]: the diff keys between models and loaded-state-dict
        """
        # 1. is absolute same
        all_diff_keys, not_in_model_keys, not_in_loaded_keys = self.get_diff_keys(return_all_diff=True)
        if len(all_diff_keys) == 0:
            return []

        if self.base_model_prefix is None:
            return all_diff_keys

        # 2. <model>-<loaded>: <base>-<downstream>
        if not state_dict_contains_prefix(self.model_state_dict, self.base_model_prefix):

            # the base-static must be same
            if not state_dict_contains_prefix(self.loaded_state_dict, self.base_model_prefix):
                error_msg = ["also the base model, but contains the diff keys: \n"]
                if not_in_model_keys:
                    error_msg.append(f"in loaded state-dict, not in model keys: <{not_in_model_keys}>\n")
                if not_in_loaded_keys:
                    error_msg.append(f"in model keys, not in loaded state-dict keys: <{not_in_model_keys}>\n")
                logger.error(error_msg)
                return []
            self.change_base_downstream_mismatched_keys()
        elif not state_dict_contains_prefix(self.loaded_state_dict, self.base_model_prefix):
            # <model>-<loaded>: <downstream>-<base>
            self.change_downstream_base_mismatched_keys()

    def get_unexpected_keys(self):
        """get unexpected keys which are not in model"""
        self.change_diff_keys()
        _, unexpected_keys, _ = self.get_diff_keys(True)
        return unexpected_keys

    def get_mismatched_keys(self):
        """get mismatched keys which not found in loaded state-dict"""
        self.change_diff_keys()
        _, _, mismatched_keys = self.get_diff_keys(True)
        return mismatched_keys

    def get_diff_keys(self, return_all_diff: bool = False) -> List[str]:
        """get diff keys

        Args:
            return_all_diff (bool, optional): return. Defaults to False.

        Returns:
            List[str]: the diff keys betweens model and loaded state-dict
        """
        mismatched_keys = set(self.model_state_dict.keys()) - set(self.loaded_state_dict.keys())
        unexpected_keys = set(self.loaded_state_dict.keys()) - set(self.model_state_dict.keys())

        all_diff_keys = mismatched_keys | unexpected_keys
        if return_all_diff:
            return all_diff_keys, unexpected_keys, mismatched_keys
        return all_diff_keys


def naive_fuse_merge_tp(weight_list, is_column=True, fuse_tensor_parts=2):
    """

    [A1 B1],[A2 B2]  => [A1, A2, B1, B2]

    Args:
        weight_list (List[np.ndarray]): The splited tensor parallel weight list.
        is_column (bool, optional): Is ColumnLinear or RowLinear. Defaults to True.

    Returns:
        weight (np.ndarray): the merged weight.
    """
    if is_column:
        axis = -1
    else:
        axis = 0

    reorder = []
    if isinstance(weight_list[0], np.ndarray):
        for item in weight_list:
            reorder.extend(np.split(item, fuse_tensor_parts, axis=axis))
    else:
        for item in weight_list:
            reorder.extend(paddle.split(item, fuse_tensor_parts, axis=axis))
    # 0 1 2 3 -> 0 2 1 3
    index = (
        np.transpose(np.arange(len(reorder)).reshape([len(weight_list), fuse_tensor_parts]), [1, 0])
        .reshape(-1)
        .tolist()
    )

    if isinstance(weight_list[0], np.ndarray):
        return np.concatenate([reorder[i] for i in index], axis=axis)
    else:
        tensor = paddle.concat([reorder[i] for i in index], axis=axis)

        if tensor.place.is_gpu_place():
            tensor = tensor._copy_to(paddle.CUDAPinnedPlace(), False)
        return tensor


def naive_fuse_split_tp(
    weight, tensor_parallel_degree, tensor_parallel_rank=None, is_column=True, fuse_tensor_parts=2
):
    """

    [A1, A2, B1, B2] => [A1 B1],[A2 B2]

    Args:
        weight (numpy.ndarray): the tensor weight,
        tensor_parallel_degree (int): tensor_parallel_degree
        tensor_parallel_rank (int): tensor_parallel_rank
        is_column (bool, optional): is ColumnLinear . Defaults to True.

    Returns:
        tensor (numpy.ndarray): splited weight.

    """
    axis = -1 if is_column else 0
    if "PySafeSlice" in str(type(weight)):
        size = weight.get_shape()[axis]
        block_size = size // (fuse_tensor_parts * tensor_parallel_degree)

        splited = []
        if tensor_parallel_rank is None:
            begin, end, step = 0, fuse_tensor_parts * tensor_parallel_degree, 1
        else:
            begin, end, step = tensor_parallel_rank, fuse_tensor_parts * tensor_parallel_degree, tensor_parallel_degree
        for rank in range(begin, end, step):
            start = rank * block_size
            stop = (rank + 1) * block_size
            if axis == 0 or len(weight.get_shape()) == 1:
                tensor = weight[start:stop]
            else:
                tensor = weight[:, start:stop]
            splited.append(tensor)

        if tensor_parallel_rank is None:
            ret = []
            for tensor_parallel_rank in range(tensor_parallel_degree):
                ret.append(np.concatenate(splited[tensor_parallel_rank::tensor_parallel_degree], axis=axis))
            return ret

        return np.concatenate(splited, axis=axis)

    if isinstance(weight, paddle.Tensor):

        def slice_concat_by_axis(weight, fuse_tensor_parts, tensor_parallel_degree, tensor_parallel_rank, axis=0):
            total_splits = fuse_tensor_parts * tensor_parallel_degree
            dim_size = weight.shape[axis]
            split_size = dim_size // total_splits

            slices = []
            for idx in range(tensor_parallel_rank, total_splits, tensor_parallel_degree):
                start = idx * split_size
                end = (start + split_size) if (idx != total_splits - 1) else dim_size
                slice_idx = [slice(None)] * len(weight.shape)
                slice_idx[axis] = slice(start, end)
                block = weight[tuple(slice_idx)]
                slices.append(block)
            result = paddle.concat(slices, axis=axis)
            return result

        if tensor_parallel_rank is not None:
            return slice_concat_by_axis(
                weight, fuse_tensor_parts, tensor_parallel_degree, tensor_parallel_rank, axis=axis
            )
        else:
            splited = paddle.split(weight, fuse_tensor_parts * tensor_parallel_degree, axis=axis)
            ret = []
            for tensor_parallel_rank in range(tensor_parallel_degree):
                ret.append(paddle.concat(splited[tensor_parallel_rank::tensor_parallel_degree], axis=axis))
            return ret
    else:
        splited = np.split(weight, fuse_tensor_parts * tensor_parallel_degree, axis=axis)

        if tensor_parallel_rank is None:
            ret = []
            for tensor_parallel_rank in range(tensor_parallel_degree):
                ret.append(np.concatenate(splited[tensor_parallel_rank::tensor_parallel_degree], axis=axis))
            return ret

        return np.concatenate(splited[tensor_parallel_rank::tensor_parallel_degree], axis=axis)


def normal_fuse_merge_tp(weight_list, is_column=True):
    """

    [A1],[A2]  => [A1, A2]

    Args:
        weight_list (List[np.ndarray]): The splited tensor parallel weight list.
        is_column (bool, optional): Is ColumnLinear or RowLinear. Defaults to True.

    Returns:
        weight (np.ndarray): the merged weight.
    """

    if is_column:
        if isinstance(weight_list[0], np.ndarray):
            return np.concatenate(weight_list, axis=-1)
        else:
            tensor = paddle.concat(weight_list, axis=-1)
            if tensor.place.is_gpu_place():
                tensor = tensor._copy_to(paddle.CUDAPinnedPlace(), False)
            return tensor
    else:
        if isinstance(weight_list[0], np.ndarray):
            return np.concatenate(weight_list, axis=0)
        else:
            tensor = paddle.concat(weight_list, axis=0)
            if tensor.place.is_gpu_place():
                tensor = tensor._copy_to(paddle.CUDAPinnedPlace(), False)
            return tensor


def normal_fuse_split_tp(weight, tensor_parallel_degree, tensor_parallel_rank=None, is_column=True):
    """

    [A1, A2]  =>  [A1],[A2]

    Args:
        weight (numpy.ndarray): the tensor weight,
        tensor_parallel_degree (int): tensor_parallel_degree
        tensor_parallel_rank (int): tensor_parallel_rank
        is_column (bool, optional): is ColumnLinear . Defaults to True.

    Returns:
        tensor (numpy.ndarray): splited weight.
    """
    dim = -1 if is_column else 0
    if "PySafeSlice" in str(type(weight)):
        size = weight.get_shape()[dim]
        block_size = size // tensor_parallel_degree

        if tensor_parallel_rank is None:
            begin, end, step = 0, tensor_parallel_degree, 1
        else:
            begin, end, step = tensor_parallel_rank, tensor_parallel_rank + 1, 1

        splited = []
        for rank in range(begin, end, step):
            start = rank * block_size
            stop = (rank + 1) * block_size

            if dim == 0 or len(weight.get_shape()) == 1:
                tensor = weight[start:stop]
            elif dim == -1:
                tensor = weight[:, start:stop]
            else:
                raise NotImplementedError("Let's make that generic when needed")
            if tensor_parallel_rank is not None:
                return tensor

            splited.append(tensor)

        return splited

    size = weight.shape[dim]
    assert (
        size % tensor_parallel_degree == 0
    ), f"The chosen size {size} is not compatible with sharding on {tensor_parallel_degree} shards. for tensor shape {weight.shape}"
    if is_column:
        total_size = weight.shape[-1]
        chunk_size = total_size // tensor_parallel_degree
        if tensor_parallel_rank is not None:
            start = tensor_parallel_rank * chunk_size
            end = (tensor_parallel_rank + 1) * chunk_size
            if isinstance(weight, paddle.Tensor):
                splited_weights = weight[..., start:end].clone()
            else:
                splited_weights = weight[..., start:end]
            return splited_weights
        else:
            splited_weights = [
                weight[..., i * chunk_size : (i + 1) * chunk_size] for i in range(tensor_parallel_degree)
            ]
            return splited_weights
    else:
        total_size = weight.shape[0]
        chunk_size = total_size // tensor_parallel_degree
        if tensor_parallel_rank is not None:
            start = tensor_parallel_rank * chunk_size
            end = (tensor_parallel_rank + 1) * chunk_size
            if isinstance(weight, paddle.Tensor):
                splited_weights = weight[start:end, ...].clone()
            else:
                splited_weights = weight[start:end, ...]
            return splited_weights
        else:
            splited_weights = [
                weight[i * chunk_size : (i + 1) * chunk_size, ...] for i in range(tensor_parallel_degree)
            ]
            return splited_weights


"""
There're three types of MultiHeadAttention QKV Layout in Transformers

tensor_parallel_qkv = [q1, k1, v1, q2, k2, v2]
naive_merged_qkv    = [q1, q1, k1, k2, v1, v2]
splited_qkv         = [q1, q1], [k1, k2], [v1, v2]

naive_merged_qkv -> tensor_parallel_qkv
    : naive_merged_qkv_to_tensor_parallel_qkv

splited_qkv -> tensor_parallel_qkv
    : splited_qkv_to_tensor_parallel_qkv


"""


def tensor_parallel_qkv_to_naive_merged_qkv(weight, num_attention_heads):
    """
    [q1, k1, v1, q2, k2, v2] => [q1, q1, k1, k2, v1, v2]
    """
    qkvs = []
    partition_dim = -1
    split_heads = np.split(weight, 3 * num_attention_heads, axis=partition_dim)
    qkv_weight_num = 3

    for i in range(qkv_weight_num):
        qkv = np.concatenate(split_heads[i::qkv_weight_num], axis=partition_dim)
        qkvs.append(qkv)

    return np.concatenate(qkvs, axis=partition_dim)


def naive_merged_qkv_to_tensor_parallel_qkv(weight, num_attention_heads):
    """
    [q1, q1, k1, k2, v1, v2] => [q1, k1, v1, q2, k2, v2]
    """
    qkv_pairs = []
    partition_dim = -1
    if isinstance(weight, paddle.Tensor):
        split_heads = paddle.split(weight, 3 * num_attention_heads, axis=partition_dim)

        for i in range(num_attention_heads):
            qkv_pair = paddle.concat(split_heads[i::num_attention_heads], axis=partition_dim)
            qkv_pairs.append(qkv_pair)
        return paddle.concat(qkv_pairs, axis=partition_dim)
    else:
        split_heads = np.split(weight, 3 * num_attention_heads, axis=partition_dim)

        for i in range(num_attention_heads):
            qkv_pair = np.concatenate(split_heads[i::num_attention_heads], axis=partition_dim)
            qkv_pairs.append(qkv_pair)

        return np.concatenate(qkv_pairs, axis=partition_dim)


def splited_qkv_to_tensor_parallel_qkv(weight_list, num_attention_heads):
    """
    [q1, k1, v1], [q2, k2, v2] => [q1, q1, k1, k2, v1, v2]

    Args:
        weight_list (_type_): [Q,K,V] tensor list
    """
    assert len(
        weight_list
    ), f"weight_list length is not equal 3, it should be Q K V list. but got length {len(weight_list)}"
    weight = np.concatenate(weight_list, axis=-1)
    return naive_merged_qkv_to_tensor_parallel_qkv(weight)


def fuse_param_func():
    def fn(fuse_params, is_qkv=False, num_heads=None, num_key_value_heads=None):
        """fuse function for fusing weights

        (1) fuse_attention_qkv
            q => [q1,q2,q3,q4]
            k => [k1,k2,k3,k4] or [k1,k2] for GQA
            v => [v1,v2,v3,v4] or [v1,v2] for GQA
            fused weight => [q1,k1,v1,q2,k2,v2,q3,k3,v3,q4,k4,v4]
                 or for GQA [q1,q2,k1,v1,q3,q4,k2,v2]
        (2) fuse_attention_ffn
            directly fuse weights to 1 parts
            [gate_weight], [up_weight] => [gate_weight, up_weight]

        Args:
            fuse_params (_type_): to be fused weights
            is_qkv (bool, optional): for attention qkv weights. Defaults to False.
            num_heads (_type_, optional): query heads. Defaults to None.
            num_key_value_heads (_type_, optional): key and value heads. Defaults to None.

        Returns:
            _type_: fused weights
        """
        concat_fn = np.concatenate
        split_fn = np.split
        if isinstance(fuse_params[0], paddle.Tensor):
            concat_fn = paddle.concat
            split_fn = paddle.split

        if is_qkv:
            # fuse_attention_qkv
            assert num_heads, f"num_heads should be number of heads for Q, but got {num_heads}"
            assert (
                num_key_value_heads
            ), f"num_key_value_heads should be number of key_value_heads for K and V, but got {num_key_value_heads}"
            assert (
                len(fuse_params) == 3
            ), f"fuse_params length is not equal 3, it should be Q K V list. but got length {len(fuse_params)}"
            num_query_groups = num_heads // num_key_value_heads
            q_list = split_fn(fuse_params[0], num_heads, axis=-1)
            k_list = split_fn(fuse_params[1], num_key_value_heads, axis=-1)
            v_list = split_fn(fuse_params[2], num_key_value_heads, axis=-1)

            qkv_pairs = []
            for i in range(num_key_value_heads):
                qkv_pairs += q_list[i * num_query_groups : (i + 1) * num_query_groups]
                qkv_pairs.append(k_list[i])
                qkv_pairs.append(v_list[i])
            return concat_fn(qkv_pairs, axis=-1)
        else:
            # fuse_attention_ffn
            return concat_fn(fuse_params, axis=-1)

    return fn


def split_param_func():
    def fn(fused_param, split_nums=2, is_qkv=False, num_heads=None, num_key_value_heads=None):
        """split function for splitting weights

        (1) fuse_attention_qkv
            fused weight => [q1,k1,v1,q2,k2,v2,q3,k3,v3,q4,k4,v4]
                 or for GQA [q1,q2,k1,v1,q3,q4,k2,v2]
            after split
            q => [q1,q2,q3,q4]
            k => [k1,k2,k3,k4] or [k1,k2] for GQA
            v => [v1,v2,v3,v4] or [v1,v2] for GQA
        (2) fuse_attention_ffn
            directly split weight to 2 parts
            [gate_weight, up_weight] => [gate_weight], [up_weight]

        Args:
            fused_param (_type_): len(fused_param)=1, only one weight to be split
            split_nums (int, optional): split_nums. Defaults to 2.
            is_qkv (bool, optional): for attention qkv weights. Defaults to False.
            num_heads (_type_, optional): query heads. Defaults to None.
            num_key_value_heads (_type_, optional): key and value heads. Defaults to None.

        Returns:
            _type_: split weights
        """
        concat_fn = np.concatenate
        split_fn = np.split
        if isinstance(fused_param, paddle.Tensor):
            concat_fn = paddle.concat
            split_fn = paddle.split

        if is_qkv:
            # fuse_attention_qkv
            assert num_heads, f"num_heads should be number of heads for Q, but got {num_heads}"
            assert (
                num_key_value_heads
            ), f"num_key_value_heads should be number of key_value_heads for K and V, but got {num_key_value_heads}"
            num_query_groups = num_heads // num_key_value_heads
            q_list, k_list, v_list = [], [], []
            split_heads = split_fn(fused_param, num_heads + 2 * num_key_value_heads, axis=-1)
            for i in range(num_key_value_heads):
                q_list += split_heads[i * (num_query_groups + 2) : (i + 1) * (num_query_groups + 2) - 2]
                k_list.append(split_heads[(i + 1) * (num_query_groups + 2) - 2])
                v_list.append(split_heads[(i + 1) * (num_query_groups + 2) - 1])
            return concat_fn(q_list, axis=-1), concat_fn(k_list, axis=-1), concat_fn(v_list, axis=-1)
        else:
            # fuse_attention_ffn
            return split_fn(fused_param, split_nums, axis=-1)

    return fn


def split_or_fuse_func(is_fuse=True):
    return fuse_param_func() if is_fuse else split_param_func()


def get_tensor_parallel_merge_func(tensor_parallel_degree, tensor_parallel_rank, num_attention_heads=None):
    def fn(
        x,
        is_column=True,
        transpose=False,
        is_old_qkv=False,
        is_naive_2fuse=False,
        is_naive_3fuse=False,
    ):
        if x is None:
            return None

        if is_naive_2fuse:
            return naive_fuse_merge_tp(x, is_column=is_column, fuse_tensor_parts=2)
        elif is_naive_3fuse:
            return naive_fuse_merge_tp(x, is_column=is_column, fuse_tensor_parts=3)
        else:
            x = normal_fuse_merge_tp(x, is_column=is_column)

        if is_old_qkv:
            assert is_column, "QKV tensor should be column parallel linear."
            assert num_attention_heads is not None, "is_old_qkv need num_attention_heads"
            x = tensor_parallel_qkv_to_naive_merged_qkv(x, num_attention_heads)
        if transpose:
            x = np.transpose(x, [1, 0])

        return x

    return fn


def get_tensor_parallel_split_func(tensor_parallel_degree, tensor_parallel_rank, num_attention_heads=None):
    def fn(x, is_column=True, transpose=False, is_old_qkv=False, is_naive_2fuse=False, is_naive_3fuse=False):
        if x is None:
            return None
        if transpose:
            if isinstance(x, paddle.Tensor):
                x = paddle.transpose(x, [1, 0])
            else:
                x = np.transpose(x, [1, 0])
        if is_old_qkv:
            assert is_column, "QKV tensor should be column parallel linear."
            assert num_attention_heads is not None, "is_old_qkv need num_attention_heads"
            x = naive_merged_qkv_to_tensor_parallel_qkv(x, num_attention_heads)
        if is_naive_2fuse:
            return naive_fuse_split_tp(
                x, tensor_parallel_degree, tensor_parallel_rank, is_column=is_column, fuse_tensor_parts=2
            )
        if is_naive_3fuse:
            return naive_fuse_split_tp(
                x, tensor_parallel_degree, tensor_parallel_rank, is_column=is_column, fuse_tensor_parts=3
            )

        return normal_fuse_split_tp(x, tensor_parallel_degree, tensor_parallel_rank, is_column=is_column)

    return fn


def split_or_merge_func(is_split, tensor_parallel_degree, tensor_parallel_rank, num_attention_heads=None):
    if is_split:
        return get_tensor_parallel_split_func(tensor_parallel_degree, tensor_parallel_rank, num_attention_heads)
    return get_tensor_parallel_merge_func(tensor_parallel_degree, tensor_parallel_rank, num_attention_heads)


@dataclass
class StateDictNameMapping:
    """NameMapping of StateDict between two models"""

    source_name: str
    target_name: str = None

    action: Optional[str] = None  # the value can be: transpose, merge_last_two_dim
    index: Optional[int] = None

    slots: list[str] = None

    def __post_init__(self):
        self.target_name = self.target_name or self.source_name

    def should_transpose(self) -> bool:
        return self.action == "transpose"

    def should_merge_last_two_dim(self) -> bool:
        """check that whether merge last two dim"""
        return self.action == "merge_last_two_dim"

    def run(self, state_dict: dict[str, ndarray], name: str) -> ndarray:
        """run some custom operation on ndarray, eg: transpose, merge_last_two_dim

        Args:
            tensor (ndarray): the source of the tensor data

        Returns:
            ndarray: the final tensor
        """
        tensor = state_dict.pop(name)
        if callable(self.action):
            return self.action(tensor)
        if self.action == "transpose":
            return transpose(tensor, [1, 0])
        if self.action == "merge_last_two_dim":
            shape = tensor.shape
            assert len(shape) == 3
            return np.reshape(tensor, [shape[0], -1])
        if self.action == "split":
            assert self.index is not None, "when action is `split`, index field is required."
            # FIXME if the order of split starts from index=2, no tensor left.
            if self.index < 2:
                state_dict[name] = tensor
            # qkv is stored in same tensor, so it should be split into 3 arr
            tensors = np.split(tensor, 3, axis=-1)
            return tensors[self.index]

        return tensor

    def matched(self, text: str) -> bool:
        """check whether the layer_name match the current pattern

        Args:
            text (str): the name of layer

        Returns:
            bool: whether the
        """
        if text == self.source_name:
            return True

        if not self.slots:
            return False


class TensorInfoSaver:
    def __init__(self) -> None:
        self.series = {}

    def add(self, state_dict_key: str, key: str, values: Union[float, ndarray, Tensor, PytorchTensor]):
        """add

        Args:
            state_dict_key (str): the state_dict key to compare, eg: embedding.weight
            key (str): the field to compare, eg: paddle_input
            values (Union[float, ndarray, Tensor]): the tensor
        """
        if state_dict_key not in self.series:
            self.series[state_dict_key] = {}

        if state_dict_key not in self.series[state_dict_key]:
            self.series[state_dict_key]["state_dict_key"] = state_dict_key

        self.series[state_dict_key][key] = tensor_summary(values)

    def summary(self, output_path: Optional[str] = None):
        """output the summary info into different terminal

        Args:
            output_path (Optional[str], optional): the dir/file of summary file. Defaults to None.
        """
        if output_path and os.path.isdir(output_path):
            output_path = os.path.join(output_path, "tensor_summary.xlsx")
            self.summary_to_excel(output_path)

        self.summary_to_terminal()

    def summary_to_excel(self, file: str):
        if not is_package_available("pandas"):
            return False
        if not is_package_available("openpyxl"):
            logger.warning(
                "detect that pandas is installed, but openpyxl is not installed so can't save info into excel file. "
                "you can run command: `pip install openpyxl` to get the great feature"
            )
            return False

        import pandas as pd

        with pd.ExcelWriter(file, "a", engine="openpyxl", if_sheet_exists="new") as writer:
            pd.DataFrame(list(self.series.values())).to_excel(writer, index=False)

    def summary_to_terminal(self):
        """print table info into terminal with tabulate"""
        from tabulate import tabulate

        headers = {key: key for key in self.series.keys()}
        print(tabulate(list(self.series.values()), tablefmt="grid", headers=headers))

    def clear(self):
        """clear the series data"""
        self.series.clear()


class LogitHooker:
    """hooks for pytorch model and paddle model, used to generate the logits of element layers"""

    def __init__(self, mappings: List[StateDictNameMapping], tensor_info_saver: Optional[TensorInfoSaver] = None):
        """register the logit hooks to compare the inputs * outputs model

        Args:
            mappings (List[StateDictNameMapping]): the mappings between paddle & pytorch model
            tensor_info_saver (Optional[TensorInfoSaver], optional): the saver for model logit. Defaults to None.
        """
        self.mappings = mappings
        self.tensor_info_saver = tensor_info_saver or TensorInfoSaver()

    def _paddle_hooks(self, layer: Layer, inputs: Tuple[Tensor], outputs: Union[Tensor, Tuple[Tensor]]):
        """internal paddle hooks to save the logit of paddle layer

        Args:
            layer (Layer): the layer of paddle element
            inputs (Tuple[Tensor]): the inputs of paddle layer
            outputs (Union[Tensor, Tuple[Tensor]]): the outputs of paddle layer
        """
        state_dict_name = layer.__state_dict_name__

        self.tensor_info_saver.add(state_dict_name, "paddle-input", inputs)

        self.tensor_info_saver.add(state_dict_name, "paddle-outputs", outputs)

    def _pytorch_hooks(
        self,
        layer: Layer,
        inputs: Tuple[PytorchTensor],
        outputs: Union[Dict[str, PytorchTensor], Tuple[PytorchTensor]],
    ):
        """internal pytorch hooks to save the logit of pytorch module

        Args:
            layer (torch.nn.Module): the module of pytorch model
            inputs (Tuple[PytorchTensor]): the inputs of pytorch layer
            outputs (Union[Dict[str, PytorchTensor], Tuple[PytorchTensor]]): the outputs of pytorch layer
        """
        state_dict_name = layer.__state_dict_name__

        self.tensor_info_saver.add(
            state_dict_name,
            "pytorch-input",
            inputs,
        )

        self.tensor_info_saver.add(state_dict_name, "pytorch-outputs", outputs)

    def register_paddle_model_hooks(self, model: Layer):
        """register post forward hook to save the inputs & outputs of paddle model

        Args:
            model (Layer): paddle model
        """

        # 1. register paddle model hook to save the logits of target layer
        def register_hook_by_name(model: Layer, mapping: StateDictNameMapping, hook: Callable[..., None]):
            """register hook by name of state_dict, eg: encoder.layers.0.linear1.bias

            Args:
                model (Layer): the source model
                mapping (StateDictNameMapping): the name mapping object
                hook (Callable[..., None]): the hook for paddle model
            """
            name = mapping.target_name
            attributes = name.split(".")
            last_layer: Layer = model
            for attribute in attributes:
                if getattr(model, attribute, None) is not None:
                    model = getattr(model, attribute)
                    if isinstance(model, Layer):
                        last_layer = model
            if (
                hasattr(last_layer, "register_forward_post_hook")
                and getattr(last_layer, "__state_dict_name__", None) is None
            ):
                last_layer.register_forward_post_hook(hook)
                # set state_dict key into layer as the private attribute
                last_layer.__state_dict_name__ = name

        for mapping in self.mappings:
            register_hook_by_name(model, mapping, self._paddle_hooks)

    def register_pytorch_model_hooks(self, model: Module):
        """register hook for pytorch model to save the inputs & outputs of pytorch model

        Args:
            model (_type_): pytorch model
        """
        from torch import nn

        # 1. register paddle model hook to save the logits of target layer
        def register_hook_by_name(model: Module, mapping: StateDictNameMapping, hook: Callable[..., None]):
            name = mapping.source_name
            attributes, index = name.split("."), 0
            last_layer: Module = model
            while index < len(attributes):
                attribute = attributes[index]
                if getattr(model, attribute, None) is not None:
                    if isinstance(model, nn.ModuleList) and attribute.isdigit():
                        model = model[int(attribute)]
                        last_layer = model
                    else:
                        model = getattr(model, attribute)
                        if isinstance(model, nn.Module):
                            last_layer = model
                index += 1
            if (
                hasattr(last_layer, "register_forward_hook")
                and getattr(last_layer, "__state_dict_name__", None) is None
            ):
                last_layer.register_forward_hook(hook)
                # set state_dict key into layer as the private attribute
                last_layer.__state_dict_name__ = mapping.target_name

        for mapping in self.mappings:
            register_hook_by_name(model, mapping, self._pytorch_hooks)

    def summary(self):
        """print the summary info to terminal/excel to analysis"""
        self.tensor_info_saver.summary()


class LogitComparer:
    """Model Weight Converter for developer to convert pytorch/tensorflow/jax pretrained model weight to paddle.

    * you can convert model weight in online/offline mode.
    * you can convert weight and config file.
    * you can convert weight/config file in some customization ways.
    """

    _ignore_state_dict_keys = []
    num_layer_regex = r"\.\d+\."

    num_layer_key: str = "num_hidden_layers"

    # when field-name is same as hf models, so you only need to
    # change this attribute to map the configuration
    config_fields_to_be_removed: List[str] = ["transformers_version"]
    architectures: Dict[str, Type[PretrainedModel]] = {}

    def __init__(self, input_dir: str) -> None:
        self.input_dir = input_dir

    def get_paddle_pytorch_model_classes(self) -> Tuple[object, object]:
        """return the [PaddleModelClass, PytorchModelClass] to
            1. generate paddle model automatically
            2. compare the logits from pytorch model and paddle model automatically

        Returns:
            Tuple[object, object]: [PaddleModelClass, PytorchModelClass]
        """
        raise NotImplementedError

    def get_inputs(self):
        """the numpy inputs for paddle & pytorch model"""
        input_ids = paddle.arange(600, 700)
        input_ids = paddle.unsqueeze(input_ids, axis=0).detach().cpu().numpy()
        return [input_ids]

    def resolve_paddle_output_logits(self, paddle_outputs: Tuple[Tensor]):
        """resolve the logit from paddle model which can be `last_hidden_state`"""
        output = None
        if isinstance(paddle_outputs, (tuple, list)):
            output = paddle_outputs[0]
        elif paddle.is_tensor(paddle_outputs):
            output = paddle_outputs

        if output is None:
            raise NotImplementedError("can't resolve paddle model outputs")

        return output.detach().cpu().reshape([-1]).numpy()

    def resolve_pytorch_output_logits(self, pytorch_outputs: Module):
        """resolve the logit from pytorch model which can be `last_hidden_state`"""
        output = pytorch_outputs[0]
        if output is None:
            raise NotImplementedError("can't resolve paddle model outputs")

        return output.detach().cpu().reshape([-1]).numpy()

    @staticmethod
    def get_model_state_dict(model: Union[Layer, Module], copy: bool = False) -> Dict[str, ndarray]:
        """get the state_dict of pytorch/paddle model

        Args:
            model (Union[Layer, Module]): can be paddle/pytorch model

        Returns:
            Dict[str, ndarray]: the final state_dict data
        """
        from torch import nn

        assert isinstance(model, (Layer, nn.Module))
        state_dict = {key: value.detach().cpu().numpy() for key, value in model.state_dict().items()}
        if copy:
            state_dict = deepcopy(state_dict)
        return state_dict

    def compare_model_state_dicts(
        self,
        paddle_model: Union[Layer, Dict[str, ndarray]],
        pytorch_model: Union[Module, Dict[str, ndarray]],
        name_mappings: List[StateDictNameMapping],
    ):
        """compare the pytorch and paddle model state with name mappings

        Args:
            paddle_model (Union[Layer, Dict[str, ndarray]]): paddle model instance
            pytorch_model (Union[Module, Dict[str, ndarray]]): pytorch model instance
            name_mappings (List[StateDictNameMapping]): the name mappings
        """
        if not isinstance(paddle_model, dict):
            paddle_state_dict = {key: value.detach().cpu().numpy() for key, value in paddle_model.state_dict().items()}
        else:
            paddle_state_dict = paddle_model

        if not isinstance(pytorch_model, dict):
            pytorch_state_dict = {
                key: value.detach().cpu().numpy() for key, value in pytorch_model.state_dict().items()
            }
        else:
            pytorch_state_dict = pytorch_model

        model_state_saver = TensorInfoSaver()
        for name_mapping in name_mappings:
            model_state_saver.add(name_mapping.target_name, "pytorch_key", name_mapping.source_name)

            if name_mapping.target_name in paddle_state_dict:
                paddle_numpy = paddle_state_dict.pop(name_mapping.target_name)
                model_state_saver.add(name_mapping.target_name, "paddle", paddle_numpy)
                model_state_saver.add(name_mapping.target_name, "paddle-shape", str(paddle_numpy.shape))

            if name_mapping.source_name in pytorch_state_dict:
                pytorch_numpy = pytorch_state_dict.pop(name_mapping.source_name)
                model_state_saver.add(name_mapping.target_name, "pytorch", pytorch_numpy)
                model_state_saver.add(name_mapping.target_name, "pytorch-shape", str(pytorch_numpy.shape))

        model_state_saver.summary()

    def compare_logits(self) -> bool:
        """compare the logit of pytorch & paddle model

        Returns:
            bool: if the logits is absolutely same
        """
        PaddleModel, PytorchModel = self.get_paddle_pytorch_model_classes()
        paddle_model = PaddleModel.from_pretrained(self.input_dir)

        # 0. init the name_mapping & tensor_info_saver & logit_hooker
        name_mappings = self.get_name_mapping(paddle_model.config)
        tensor_info_saver = TensorInfoSaver()

        logit_hooker = LogitHooker(name_mappings, tensor_info_saver)
        inputs = self.get_inputs()

        # 1. get the logits of paddle model
        logit_hooker.register_paddle_model_hooks(paddle_model)
        paddle_inputs = [paddle.to_tensor(input_item) for input_item in inputs]
        paddle_model.eval()

        paddle_outputs = paddle_model(*paddle_inputs)
        # remove paddle_model and free gpu memory
        paddle_model_state_dict = self.get_model_state_dict(paddle_model)
        del paddle_model
        paddle_logits = self.resolve_paddle_output_logits(paddle_outputs)

        logger.info("===============the summary of paddle Model logits: ===============")
        logger.info(tensor_summary(paddle_logits))

        # 2. get the logits of pytorch model
        import torch

        pytorch_model = PytorchModel.from_pretrained(self.input_dir)
        logit_hooker.register_pytorch_model_hooks(pytorch_model)

        pytorch_model.eval()
        pytorch_inputs = [torch.tensor(input_item) for input_item in inputs]
        torch_outputs = pytorch_model(*pytorch_inputs)
        # remove paddle_model and free gpu memory
        pytorch_model_state_dict = self.get_model_state_dict(pytorch_model)
        del pytorch_model

        pytorch_logits = self.resolve_pytorch_output_logits(torch_outputs)

        logger.info("===============the summary of pytorch Model logits: ===============")
        logger.info(tensor_summary(pytorch_logits))

        # 3. compare the logits
        result = allclose(paddle_logits[1:4], pytorch_logits[1:4], atol=1e-4)

        if not result:
            print("============================== compare model state dict ==============================")

            self.compare_model_state_dicts(paddle_model_state_dict, pytorch_model_state_dict, name_mappings)

            print("============================== compare model inputs & outputs ==============================")
            logit_hooker.summary()

        return result

    def on_converted(self):

        PaddleModelClass, PytorchModelClass = self.get_paddle_pytorch_model_classes()

        # 1. try to compare two loaded paddle weight file
        first_paddle_model = PaddleModelClass.from_pretrained(self.input_dir)
        second_paddle_model = PaddleModelClass.from_pretrained(self.input_dir)
        mismatched_keys = compare_model_weights(
            self.get_model_state_dict(first_paddle_model),
            self.get_model_state_dict(second_paddle_model),
        )
        for key in mismatched_keys:
            logger.error(f"the key<{key}> is not set correctly with weight")

        # 2. try to compare logits between paddle & pytorch model
        if is_torch_available() and is_transformers_available():
            result = self.compare_logits()
            if result is True:
                logger.info("the logits between pytorch model and paddle model is absolutely same")
            else:
                logger.error(
                    "the logits between pytorch model and paddle model is not same, please check it out more carefully."
                )
        else:
            logger.warning(
                "you don't install `torch` and `transformers` package, so we can't compare the logits between paddle & pytorch model"
            )


class ConversionMixin:
    @classmethod
    def support_conversion(cls, config: PretrainedConfig) -> bool:
        """check whether the model support conversion"""
        try:
            # try to get the name-mapping info
            _ = cls._get_name_mappings(config)
        except NotImplementedError:
            return False
        finally:
            return True

    @classmethod
    def convert(cls, weight_file: str, config: PretrainedConfig, cache_dir: str) -> None:
        """the entry of converting config and converting model file

        Args:
            input_dir (str | None): the input dir which contains `pytorch_model.bin` and `config.json` file
            config (PretrainedConfig): the PretrainedConfig instance of model
        """
        # FIXME(wj-Mcat): add compatibility with downstream models
        name_mappings = cls._get_name_mappings(config)
        if weight_file.endswith(".index.json"):
            if ".safetensors." in weight_file:
                files = [file for file in os.listdir(os.path.dirname(weight_file)) if file.startswith("model-")]
            else:
                files = [
                    file for file in os.listdir(os.path.dirname(weight_file)) if file.startswith("pytorch_model-")
                ]
            state_dict = {}
            for file in sorted(files):
                sub_state_dict = load_torch(os.path.join(os.path.dirname(weight_file), file))
                state_dict.update(sub_state_dict)
        else:
            state_dict = load_torch(weight_file)

        # 3. convert state_dict
        all_layer_names = set(state_dict.keys())
        for name_mapping in name_mappings:
            if name_mapping.source_name not in state_dict:
                logger.warning(f"key<{name_mapping.source_name}> not in the pytorch weight file.")
                continue

            state_dict[name_mapping.target_name] = name_mapping.run(state_dict, name_mapping.source_name)
            if name_mapping.source_name in all_layer_names:
                all_layer_names.remove(name_mapping.source_name)

        if all_layer_names:
            logger.warning(f"There are {len(all_layer_names)} tensors not initialized:")
            logger.warning(f"Keys: {all_layer_names}")

        return state_dict

    @classmethod
    def _get_name_mappings(cls, config: PretrainedConfig) -> List[StateDictNameMapping]:
        """get name mapping of PretrainedModel

        Args:
            config (PretrainedConfig): the configuration of name-mapping

        Raises:
            NotImplementedError:

        Returns:
            List[StateDictNameMapping]: the name-mappings of pretrained model
        """
        raise NotImplementedError

    @classmethod
    def get_tensor_parallel_convert_actions(
        cls,
        config: PretrainedConfig,
        loaded_state_dict_keys,
        is_split=True,
        ignore_error=False,
        base_model_prefix=None,
    ):
        name_action_mappings = cls._get_tensor_parallel_mappings(config, is_split=is_split)
        if config.quantization_config.is_weight_quantize():
            name_action_mappings = add_quant_mapping(name_action_mappings, config.quantization_config)
        state_keys_map = cls._resolve_prefix_keys(
            name_action_mappings.keys(), loaded_state_dict_keys, ignore_error, base_model_prefix=base_model_prefix
        )
        for k, v in state_keys_map.items():
            if k not in name_action_mappings:
                continue
            name_action_mappings[v] = name_action_mappings.pop(k)
        return name_action_mappings

    @classmethod
    def convert_tensor_parallel(
        cls, weight_file: str, config: PretrainedConfig, state_dict=None, ignore_error=False
    ) -> None:
        """the entry of converting config and converting model file

        Args:
            weight_file (str | None): the weight file path of `model_state.pdparams` file
            config (PretrainedConfig): the PretrainedConfig instance of model
        """

        name_action_mappings = cls._get_tensor_parallel_mappings(config)
        if config.quantization_config.is_weight_quantize():
            name_action_mappings = add_quant_mapping(name_action_mappings, config.quantization_config)
        if state_dict is None:
            with device_guard("cpu"):
                state_dict = paddle.load(weight_file, return_numpy=False)
            logger.info("Starting to convert original state_dict to tensor parallel state_dict.")

        state_keys_map = cls._resolve_prefix_keys(name_action_mappings.keys(), state_dict.keys(), ignore_error)

        for k, v in state_keys_map.items():
            name_action_mappings[v] = name_action_mappings.pop(k)

        for name, action in name_action_mappings.items():
            if name not in state_dict:
                if not ignore_error:
                    logger.warning(f"Key <{name}> not in the model state weight file.")
                continue
            tensor = state_dict.pop(name)
            new_tensor = action(tensor)
            with device_guard("cpu"):
                state_dict[name] = paddle.Tensor(new_tensor, zero_copy=True)

        return state_dict

    @classmethod
    def merge_tensor_parallel(cls, state_dict, config) -> None:
        """the entry of converting config and converting model file

        Args:
            input_dir (str | None): the input dir which contains `pytorch_model.bin` and `config.json` file
            config (PretrainedConfig): the PretrainedConfig instance of model
        """
        name_action_mappings = cls._get_tensor_parallel_mappings(config, is_split=False)
        if config.quantization_config.is_weight_quantize():
            name_action_mappings = add_quant_mapping(name_action_mappings, config.quantization_config)
        state_keys_map = cls._resolve_prefix_keys(name_action_mappings.keys(), state_dict.keys())

        for k, v in state_keys_map.items():
            name_action_mappings[v] = name_action_mappings.pop(k)

        state_dict_to_save = {}

        hcg = paddle.distributed.fleet.get_hybrid_communicate_group()
        mp_group = hcg.get_model_parallel_group()
        is_dst = paddle.distributed.get_rank(mp_group) == 0

        for key in state_dict.keys():
            tensor = state_dict[key]
            if key in name_action_mappings:
                if get_env_device() == "xpu":
                    ret = distributed_allgather(tensor, group=mp_group, offload=True)
                else:
                    ret = distributed_gather(tensor, group=mp_group, offload=True)
                action = name_action_mappings.pop(key)
                tensor = action(ret) if is_dst else None
            else:
                tensor = tensor.cpu().numpy() if is_dst else None

            # keep state dict use paddle.tensor
            if isinstance(tensor, np.ndarray):
                with device_guard("cpu"):
                    tensor = paddle.Tensor(tensor, zero_copy=True)

            state_dict_to_save[key] = tensor

        if len(name_action_mappings) > 0:
            for x in name_action_mappings.keys():
                logger.debug(f"key <{x}> need to merge tensor parallel but we can't find in model state.")

        return state_dict_to_save

    @classmethod
    def _get_tensor_parallel_mappings(cls, config: PretrainedConfig, is_split=True) -> List[StateDictNameMapping]:
        """get name mapping of PretrainedModel

        Args:
            config (PretrainedConfig): the configuration of name-mapping

        Raises:
            NotImplementedError:

        Returns:
            List[StateDictNameMapping]: the name-mappings for tensor_parallel
        """
        raise NotImplementedError

    @staticmethod
    def _resolve_prefix_keys(state_keys_base, state_keys_real, ignore_error=False, base_model_prefix=None):
        # state_keys_map base to real
        state_keys_map = {}

        if base_model_prefix:
            for k in state_keys_real:
                if k.startswith("lm_head."):
                    continue
                # remove real key name `base_model_prefix` + '.'
                state_keys_map[k[len(base_model_prefix + ".") :]] = k
            return state_keys_map

        # sorted by length，match from long to short for A.key B.key ...
        state_keys_base = sorted(state_keys_base, key=lambda x: len(x), reverse=True)
        state_keys_real = set(state_keys_real)

        for key in state_keys_base:
            for x in state_keys_real:
                if x.endswith(key):
                    state_keys_map[key] = x
                    break
            if key not in state_keys_map:
                if not ignore_error:
                    logger.debug(f"tensor parallel conversion: could not find name {key} in loaded state dict!")
            else:
                state_keys_real.remove(state_keys_map[key])

        return state_keys_map

    @classmethod
    def convert_fuse_and_split(cls, config: PretrainedConfig, state_dict, tp_actions=None):
        loaded_keys = state_dict.keys()
        # collect and convert fuse/split action
        fused_and_split_keys = []
        convert_with_same_keys = []
        fuse_actions, resume_keys = cls.get_fuse_or_split_param_convert_actions(config, loaded_keys, is_fuse=True)
        for keys, action in fuse_actions.items():
            if keys[-1] in keys[:-1]:
                assert len(keys) == 2, "only 2 keys can be converted with the same name"
                convert_with_same_keys.append(keys[-1])
            origin_states = [state_dict.pop(key) for key in keys[:-1]]
            state_dict[keys[-1]] = action(origin_states)
            fused_and_split_keys.append(keys[-1])
            logger.debug(f"Fusing parameter: {keys[:-1]} into {keys[-1]}")

        split_actions, _ = cls.get_fuse_or_split_param_convert_actions(config, loaded_keys, is_fuse=False)
        for keys, action in split_actions.items():
            if keys[-1] in keys[:-1]:
                assert len(keys) == 2, "only 2 keys can be converted with the same name"
                convert_with_same_keys.append(keys[-1])
            origin_state = state_dict.pop(keys[-1])
            split_states = action(origin_state)
            for key_idx, key in enumerate(keys[:-1]):
                state_dict[key] = split_states[key_idx]
                fused_and_split_keys.append(key)
            logger.debug(f"Splitting parameter: {keys[-1]} into {keys[:-1]}")

        if tp_actions is not None:
            for key in fused_and_split_keys:
                if key in convert_with_same_keys:
                    continue

                for name in tp_actions.keys():
                    if key.endswith(name):
                        with device_guard():
                            state_dict[key] = paddle.Tensor(tp_actions[name](state_dict.pop(key)), zero_copy=True)
                        break

        # when shard file split the weight as follows, some weights need to be resumed for next shard file
        # shard-001-file: q_weight, k_weight
        # shard_002-file: v_weight
        resume_state_dict = {k: state_dict[k] for k in resume_keys if k in state_dict}
        return state_dict, resume_state_dict

    @classmethod
    def get_fuse_or_split_param_convert_actions(
        cls,
        config: PretrainedConfig,
        loaded_state_dict_keys,
        is_fuse=True,
        ignore_error=False,
    ):
        name_action_mappings = cls._get_fuse_or_split_param_mappings(config, is_fuse)
        state_keys_map = cls._resolve_prefix_keys_for_fuse_and_split(
            name_action_mappings.keys(), loaded_state_dict_keys, ignore_error, is_fuse
        )
        for k, v in state_keys_map.items():
            name_action_mappings[v] = name_action_mappings.pop(k)

        # filter name_action_mappings with corresponding weights
        # fusing: verify all of the keys in name_action_mappings are in loaded_state_dict_keys
        # splitting: verify the last key in name_action_mappings is in loaded_state_dict_keys
        filter_name_action = {}
        resume_keys = []
        if is_fuse:
            for k, v in name_action_mappings.items():
                cond = True
                if not all(item in loaded_state_dict_keys for item in k[:-1]):
                    # resume keys for next fuse
                    resume_keys += k[:-1]
                    cond = False
                if cond:
                    filter_name_action[k] = v
        else:
            for k, v in name_action_mappings.items():
                if k[-1] in loaded_state_dict_keys:
                    filter_name_action[k] = v

        return filter_name_action, resume_keys

    @classmethod
    def _get_fuse_or_split_param_mappings(cls, config: PretrainedConfig, is_fuse=True) -> List[StateDictNameMapping]:
        """get fused parameter mapping of PretrainedModel

        Args:
            config (PretrainedConfig): the configuration of name-mapping

        Raises:
            NotImplementedError:

        Returns:
            List[StateDictNameMapping]: the name-mappings for tensor_parallel
        """
        # raise NotImplementedError(
        #     f"`_get_fuse_or_split_param_mappings` is not implemented for {cls.__name__}`. To implement it, you should "
        #     f"overwrite this method in the class {cls.__name__} in `{cls.__module__}.py`"
        # )
        return {}

    @staticmethod
    def _resolve_prefix_keys_for_fuse_and_split(state_keys_base, state_keys_real, ignore_error=False, is_fuse=True):
        state_keys_map = {}

        # use the tuple (x1,x2,x3,x4) as one key, and the prefix of x1,x2,x3 is used as a new key x4 or
        # the last key x4 is used as new keys x1,x2,x3. And, the tuple also could be (a) (x1, x1) -> convert x1 to x1;
        # (b) (x1,x2,x3) -> fuse x1 and x2 to x3; (c) (x1,x2,x3,x4) -> fuse x1, x2 and x3 to x4.

        # is_fuse: True -> fuse, False -> split
        # True: (x1,x2,x3,x4) -> [x1,x2,x3] are exist in state_keys_real, x4 is not exist in state_keys_real
        # False: (x1,x2,x3,x4) -> [x1,x2,x3] are not exist in state_keys_real, x4 is exist in state_keys_real

        for keys in state_keys_base:
            prefix = ""
            if is_fuse:
                for x in state_keys_real:
                    for base_key in keys[:-1]:
                        if x.endswith(base_key):
                            prefix = x.replace(base_key, "")
                            break
                    if prefix != "":
                        break
            else:
                base_key = keys[-1]
                for x in state_keys_real:
                    if x.endswith(base_key):
                        prefix = x.replace(base_key, "")
                        break

            new_keys = tuple([prefix + key for key in keys])
            state_keys_map[keys] = new_keys

        return state_keys_map


class Converter(ConversionMixin, LogitComparer):
    """some converters are implemented in ppdiffusers, so if remove it directly, it will make ppdiffusers down.
    TODO(wj-Mcat): this class will be removed after v2.6
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        logger.warning(
            "`paddleformers.utils.converter` module will be deprecated soon, you "
            "should change it to `paddleformers.transformers.conversion_utils`"
        )

    @classmethod
    def resolve_num_layer(cls, config_or_num_layers: Union[dict, int] = None) -> int:
        """resolve the number of transformer layer based on the key of model config, eg: `num_hidden_layers` in BertModel
        Args:
            config_or_num_layers (Union[dict, int], optional): the instance of config or num_layers. Defaults to None.
        Raises:
            ValueError: when `config_or_num_layers` is not dict/int, it will raise the error
        Returns:
            int: the number of transformer layer
        """
        from .configuration_utils import PretrainedConfig

        if isinstance(config_or_num_layers, (dict, PretrainedConfig)):
            num_layer = config_or_num_layers[cls.num_layer_key]
        elif isinstance(config_or_num_layers, int):
            num_layer = config_or_num_layers
        else:
            raise ValueError(f"the type of config_or_num_layers<{config_or_num_layers}> should be one of <dict, int>")

        return num_layer

    def convert(self, input_dir: str | None = None) -> None:
        """the entry of converting config and converting model file

        Args:
            input_dir (str | None): the input dir which contains `pytorch_model.bin` and `config.json` file
        """
        input_dir = input_dir or getattr(self, "input_dir", None)
        os.makedirs(input_dir, exist_ok=True)

        # 1. get pytorch weight file
        weight_file = os.path.join(input_dir, PYTORCH_WEIGHTS_NAME)
        if not os.path.exists(weight_file):
            raise FileNotFoundError(f"pytorch weight file<{weight_file}> not found")

        config_file = os.path.join(input_dir, CONFIG_NAME)
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"config file<{weight_file}> not found")

        # 2. construct name mapping
        # TODO(wj-Mcat): when AutoConfig is ready, construct config from AutoConfig.
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)

        state_dict = load_torch(weight_file)

        # FIXME(wj-Mcat): add compatibility with downstream models
        name_mappings = self.get_name_mapping(config)

        # 3. convert state_dict
        all_layer_names = set(state_dict.keys())
        for name_mapping in name_mappings:
            if name_mapping.source_name not in state_dict:
                logger.warning(f"key<{name_mapping.source_name}> not in the pytorch weight file.")
                continue

            state_dict[name_mapping.target_name] = name_mapping.run(state_dict.pop(name_mapping.source_name))
            all_layer_names.remove(name_mapping.source_name)

        if all_layer_names:
            logger.warning(f"There are {len(all_layer_names)} tensors not initialized:")
            logger.warning(f"Keys: {all_layer_names}")

        return state_dict
