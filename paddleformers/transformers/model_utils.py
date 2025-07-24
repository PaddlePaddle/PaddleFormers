# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import concurrent.futures
import contextlib
import copy
import gc
import inspect
import json
import os
import re
import sys
import tempfile
import warnings
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

import aistudio_sdk
import numpy as np
import paddle
import paddle.nn as nn
import six
from huggingface_hub import (
    create_repo,
    get_hf_file_metadata,
    hf_hub_url,
    repo_type_and_id_from_hf_id,
    upload_folder,
)
from huggingface_hub.utils import EntryNotFoundError
from paddle import Tensor
from paddle.distributed.fleet.meta_parallel.parallel_layers import (
    PipelineLayer,
    SharedLayerDesc,
)

try:
    from paddle.distributed.fleet.meta_parallel import LocalSharedLayerDesc
except:
    LocalSharedLayerDesc = None
from paddle.nn import Embedding, Layer

# TODO(fangzeyang) Temporary fix and replace by paddle framework downloader later
from paddle.utils.download import is_url as is_remote_url
from tqdm.auto import tqdm

from ..generation import GenerationConfig, GenerationMixin
from ..quantization.quantization_utils import (
    convert_to_quantize_state_dict,
    convert_to_weight_quantize_state_dict,
    parse_weight_quantize_algo,
    replace_with_quantization_linear,
    update_loaded_state_dict_keys,
)
from ..quantization.unified_checkpoint_quantization import dequant_unified_optimizer
from ..utils import device_guard
from ..utils.download import resolve_file_path
from ..utils.env import (
    ASYMMETRY_QUANT_SCALE_MAX,
    ASYMMETRY_QUANT_SCALE_MIN,
    CONFIG_NAME,
    PADDLE_WEIGHTS_INDEX_NAME,
    PADDLE_WEIGHTS_NAME,
    PYTORCH_WEIGHTS_INDEX_NAME,
    PYTORCH_WEIGHTS_NAME,
    SAFE_MASTER_WEIGHTS_INDEX_NAME,
    SAFE_PEFT_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    SYMMETRY_QUANT_SCALE,
)
from ..utils.log import logger
from .configuration_utils import PretrainedConfig
from .conversion_utils import ConversionMixin
from .utils import (  # convert_ndarray_dtype,
    ContextManagers,
    InitTrackerMeta,
    adapt_stale_fwd_patch,
    cached_file_for_hf_hub,
    convert_file_size_to_int,
    dtype_byte_size,
    fn_args_to_dict,
    get_checkpoint_shard_files,
    is_paddle_support_lazy_init,
    is_safetensors_available,
    paddleformers_load,
    weight_name_suffix,
)

__all__ = [
    "PretrainedModel",
    "register_base_model",
]


def dy2st_nocheck_guard_context():
    try:
        context = paddle.framework._no_check_dy2st_diff()
    except:
        context = contextlib.nullcontext()
    return context


def unwrap_optimizer(optimizer, optimizer_instances=()):
    if optimizer is None:
        return None
    while hasattr(optimizer, "_inner_opt") and not isinstance(optimizer, optimizer_instances):
        optimizer = optimizer._inner_opt
    if isinstance(optimizer, optimizer_instances):
        return optimizer
    return None


if is_safetensors_available():
    from safetensors.numpy import save_file as safe_save_file

    from ..utils.safetensors import fast_load_file as safe_load_file

    if sys.platform.startswith("win"):
        from safetensors import safe_open
    else:
        from ..utils.safetensors import fast_safe_open as safe_open


def prune_linear_layer(layer: nn.Linear, index: paddle.Tensor, dim: int = 0) -> nn.Linear:
    """
    Prune a linear layer to keep only entries in index.
    Used to remove heads.
    Args:
        layer (`paddle.nn.Linear`): The layer to prune.
        index (`paddle.Tensor`): The indices to keep in the layer.
        dim (`int`, *optional*, defaults to 0): The dimension on which to keep the indices.
    Returns:
        `paddle.nn.Linear`: The pruned layer as a new layer with `stop_gradient=False`.
    """
    index = index.to(layer.weight)
    W = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.shape)
    new_size[dim] = len(index)
    new_layer = nn.Linear(new_size[1], new_size[0], bias_attr=layer.bias is not None)
    new_layer.weight.stop_gradient = True
    new_layer.weight.copy_(W)
    new_layer.weight.stop_gradient = False
    if layer.bias is not None:
        new_layer.bias.stop_gradient = True
        new_layer.bias.copy_(b)
        new_layer.bias.stop_gradient = False
    return new_layer


def find_pruneable_heads_and_indices(
    heads: List[int], n_heads: int, head_size: int, already_pruned_heads: Set[int]
) -> Tuple[Set[int], paddle.Tensor]:
    """
    Finds the heads and their indices taking `already_pruned_heads` into account.
    Args:
        heads (`List[int]`): List of the indices of heads to prune.
        n_heads (`int`): The number of heads in the model.
        head_size (`int`): The size of each head.
        already_pruned_heads (`Set[int]`): A set of already pruned heads.
    Returns:
        `Tuple[Set[int], paddle.Tensor]`: A tuple with the remaining heads and their corresponding indices.
    """
    mask = paddle.ones([n_heads, head_size])
    heads = set(heads) - already_pruned_heads  # Convert to set and remove already pruned heads
    for head in heads:
        # Compute how many pruned heads are before the head and move the index accordingly
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.reshape([-1]).eq(1)
    index: paddle.Tensor = paddle.arange(len(mask))[mask].cast("int64")
    return heads, index


def apply_chunking_to_forward(
    forward_fn: Callable[..., paddle.Tensor], chunk_size: int, chunk_dim: int, *input_tensors
) -> paddle.Tensor:
    """
    This function chunks the `input_tensors` into smaller input tensor parts of size `chunk_size` over the dimension
    `chunk_dim`. It then applies a layer `forward_fn` to each chunk independently to save memory.
    If the `forward_fn` is independent across the `chunk_dim` this function will yield the same result as directly
    applying `forward_fn` to `input_tensors`.
    Args:
        forward_fn (`Callable[..., paddle.Tensor]`):
            The forward function of the model.
        chunk_size (`int`):
            The chunk size of a chunked tensor: `num_chunks = len(input_tensors[0]) / chunk_size`.
        chunk_dim (`int`):
            The dimension over which the `input_tensors` should be chunked.
        input_tensors (`Tuple[paddle.Tensor]`):
            The input tensors of `forward_fn` which will be chunked
    Returns:
        `paddle.Tensor`: A tensor with the same shape as the `forward_fn` would have given if applied`.
    Examples:
    ```python
    # rename the usual forward() fn to forward_chunk()
    def forward_chunk(self, hidden_states):
        hidden_states = self.decoder(hidden_states)
        return hidden_states
    # implement a chunked forward function
    def forward(self, hidden_states):
        return apply_chunking_to_forward(self.forward_chunk, self.chunk_size_lm_head, self.seq_len_dim, hidden_states)
    ```"""

    assert len(input_tensors) > 0, f"{input_tensors} has to be a tuple/list of tensors"

    # inspect.signature exist since python 3.5 and is a python method -> no problem with backward compatibility
    num_args_in_forward_chunk_fn = len(inspect.signature(forward_fn).parameters)
    if num_args_in_forward_chunk_fn != len(input_tensors):
        raise ValueError(
            f"forward_chunk_fn expects {num_args_in_forward_chunk_fn} arguments, but only {len(input_tensors)} input "
            "tensors are given"
        )

    if chunk_size > 0:
        tensor_shape = input_tensors[0].shape[chunk_dim]
        for input_tensor in input_tensors:
            if input_tensor.shape[chunk_dim] != tensor_shape:
                raise ValueError(
                    f"All input tenors have to be of the same shape: {tensor_shape}, "
                    f"found shape {input_tensor.shape[chunk_dim]}"
                )

        if input_tensors[0].shape[chunk_dim] % chunk_size != 0:
            raise ValueError(
                f"The dimension to be chunked {input_tensors[0].shape[chunk_dim]} has to be a multiple of the chunk "
                f"size {chunk_size}"
            )

        num_chunks = input_tensors[0].shape[chunk_dim] // chunk_size

        # chunk input tensor into tuples
        input_tensors_chunks = tuple(input_tensor.chunk(num_chunks, axis=chunk_dim) for input_tensor in input_tensors)
        # apply forward fn to every tuple
        output_chunks = tuple(forward_fn(*input_tensors_chunk) for input_tensors_chunk in zip(*input_tensors_chunks))
        # concatenate output at same dimension
        return paddle.concat(output_chunks, axis=chunk_dim)

    return forward_fn(*input_tensors)


def unwrap_model(model, *args, **kwargs):
    raw_model = model
    while hasattr(raw_model, "_layers") or hasattr(raw_model, "_layer"):
        if hasattr(raw_model, "_layers"):
            # Caused by issue https://github.com/PaddlePaddle/PaddleNLP/issues/5295
            # TODO: remove this after we fix the issue
            if raw_model._layers is None:
                break
            raw_model = raw_model._layers
        else:
            if raw_model._layer is None:
                break
            raw_model = raw_model._layer

    return raw_model


def _add_variant(weights_name: str, variant=None) -> str:
    if variant is not None and len(variant) > 0:
        splits = weights_name.split(".")
        splits = splits[:-1] + [variant] + splits[-1:]
        weights_name = ".".join(splits)

    return weights_name


@contextmanager
def dtype_guard(dtype="float32"):
    origin_dtype = paddle.get_default_dtype()
    paddle.set_default_dtype(dtype)
    try:
        yield
    finally:
        paddle.set_default_dtype(origin_dtype)


_init_weights = True


@contextmanager
def no_init_weights(_enable=True):
    """
    Context manager to globally disable weight initialization to speed up loading large models.

    TODO(Patrick): Delete safety argument `_enable=True` at next major version. .
    """
    global _init_weights
    old_init_weights = _init_weights
    if _enable:
        _init_weights = False
    try:
        yield
    finally:
        _init_weights = old_init_weights


def get_parameter_dtype(parameter: nn.Layer) -> paddle.dtype:
    """get dtype of parameter which should be sub-class of nn.Layer

    Args:
        parameter (nn.Layer): the instance of layer

    Returns:
        paddle.dtype: the dtype of tensor
    """

    last_dtype = None
    for t in parameter.parameters():
        last_dtype = t.dtype
        if t.is_floating_point():
            return t.dtype

    # TODO(wj-Mcat): get dtype of model when it's in DataParallel Mode.
    return last_dtype


def _split_keys_evenly(keys: list, n: int) -> list:
    """Split a list into n lists with an equal number of elements.

    Args:
        keys (list): the list to be split
        n (int): number of splits

    Returns:
        result: list of lists
    """

    total_len = len(keys)
    base_size = total_len // n
    extra = total_len % n

    result = []
    index = 0
    for _ in range(n):
        part_size = base_size + 1 if extra > 0 else base_size
        extra -= 1
        result.append(keys[index : index + part_size])
        index += part_size

    return result


def _load_part_state_dict(
    keys,
    checkpoint_file: Union[str, os.PathLike],
    tensor_parallel_split_mapping,
    fliter_dict_keys,
    device,
    quantization_linear_list=None,
    quantization_config=None,
    dtype=None,
    return_numpy=False,
):
    """load part state dict from checkpoint file.

    Args:
        keys (list): the keys of part state dict
        checkpoint_file (str): the path of checkpoint file
        tensor_parallel_split_mapping (dict): mapping from key to function
        fliter_dict_keys (list): filter keys in state dict

    Returns:
        part_state_dict (dict): the part state dict

    """
    part_state_dict = {}
    scale_dict = {}
    with safe_open(checkpoint_file, framework="np") as f:
        for key in keys:
            # 1. non-merge ckpt loading dont have filter key.
            # 2. merge ckpt will skip quant scale by `fliter_dict_keys`
            if (
                key.endswith(SYMMETRY_QUANT_SCALE)
                or key.endswith(ASYMMETRY_QUANT_SCALE_MIN)
                or key.endswith(ASYMMETRY_QUANT_SCALE_MAX)
            ):
                continue

            if fliter_dict_keys is not None and key not in fliter_dict_keys:
                continue

            py_safe_slice_ = f.get_slice(key)
            if (
                quantization_linear_list is not None
                and key.split(".weight")[0] in quantization_linear_list
                and not key.endswith("_scale")
            ):
                # numpy.array -> paddle.tensor
                weight = paddle.Tensor.__call__(py_safe_slice_[:], zero_copy=True)
                key_name = key.split(".weight")[0]
                quant_key_name = key_name + ".quant_weight"
                weight_scale_name = key_name + ".weight_scale"
                # 16bit -> 4/8bit
                quant_state_dict = convert_to_weight_quantize_state_dict(
                    state_dict={key: weight},
                    name=key_name,
                    quantization_config=quantization_config,
                    dtype=dtype,
                    weight_quantize_algo=parse_weight_quantize_algo(quantization_config, quant_key_name),
                )
                for key in list(quant_state_dict.keys()):
                    quant_state_dict[key] = quant_state_dict[key].numpy()
                if quant_key_name in tensor_parallel_split_mapping:
                    quant_state_dict[quant_key_name] = tensor_parallel_split_mapping[quant_key_name](
                        quant_state_dict[quant_key_name]
                    )
                    if weight_scale_name in tensor_parallel_split_mapping:
                        quant_state_dict[weight_scale_name] = tensor_parallel_split_mapping[weight_scale_name](
                            quant_state_dict[weight_scale_name]
                        )
                part_state_dict.update(quant_state_dict)
            else:
                if key in tensor_parallel_split_mapping:
                    if len(py_safe_slice_.shape) == 0:
                        weight = tensor_parallel_split_mapping[key](py_safe_slice_.get())
                    else:
                        weight = tensor_parallel_split_mapping[key](py_safe_slice_)
                else:
                    if len(py_safe_slice_.shape) == 0:
                        weight = py_safe_slice_.get()
                    else:
                        weight = py_safe_slice_[:]
                if not return_numpy and device == "expected":
                    with device_guard():
                        weight = paddle.Tensor.__call__(weight, zero_copy=True)
                    weight = weight._copy_to(paddle.framework._current_expected_place(), False)
                part_state_dict[key] = weight

        for key in keys:
            if (
                key.endswith(SYMMETRY_QUANT_SCALE)
                or key.endswith(ASYMMETRY_QUANT_SCALE_MIN)
                or key.endswith(ASYMMETRY_QUANT_SCALE_MAX)
            ):
                scale = f.get_tensor(key)
                if not return_numpy and device == "expected":
                    with device_guard():
                        scale = paddle.Tensor.__call__(scale, zero_copy=True)
                    scale = scale._copy_to(paddle.framework._current_expected_place(), False)
                scale_dict[key] = scale
    return part_state_dict, scale_dict


def load_state_dict(
    checkpoint_file: Union[str, os.PathLike],
    tensor_parallel_split_mapping=None,
    fliter_dict_keys=None,
    device="cpu",
    ckpt_quant_stage="O0",
    quantization_linear_list=None,
    quantization_config=None,
    dtype=None,
    return_numpy=False,
):
    """
    Reads a PaddlePaddle checkpoint file, returning properly formatted errors if they arise.
    """

    if tensor_parallel_split_mapping is None:
        tensor_parallel_split_mapping = {}

    if (
        checkpoint_file.endswith(".safetensors") or re.search(r"\.safetensors_shard_\d{4}$", checkpoint_file)
    ) and is_safetensors_available():
        # Check format of the archive
        with safe_open(checkpoint_file, framework="np") as f:
            metadata = f.metadata()
        if metadata is None:
            metadata = {"format": "np"}

        if metadata.get("format", "np") not in ["pd", "np"]:
            raise OSError(
                f"The safetensors archive passed at {checkpoint_file} does not contain the valid metadata. Make sure "
                "you save your model with the `save_pretrained` method."
            )
        if metadata.get("format", "np") == "pd":
            raise ValueError("Currently unsupport paddle weights file, use numpy instead.")
        if metadata.get("format", "np") == "np":
            thread_num = int(os.environ.get("LOAD_STATE_DICT_THREAD_NUM", "1"))
            if thread_num > 1:
                logger.info(f"Set loading state_dict thread num to {thread_num}")
            state_dict, scale_dict = {}, {}
            if thread_num <= 1:
                with safe_open(checkpoint_file, framework="np") as f:
                    state_dict, scale_dict = _load_part_state_dict(
                        list(f.keys()),
                        checkpoint_file,
                        tensor_parallel_split_mapping,
                        fliter_dict_keys,
                        device,
                        quantization_linear_list,
                        quantization_config,
                        dtype,
                        return_numpy,
                    )
            else:
                # Load state dict in multi-thread to speed up loading
                with safe_open(checkpoint_file, framework="np") as f:
                    keys_groups = _split_keys_evenly(list(f.keys()), thread_num)
                with concurrent.futures.ThreadPoolExecutor(max_workers=thread_num) as executor:
                    future_to_key = {
                        executor.submit(
                            _load_part_state_dict,
                            keys,
                            checkpoint_file,
                            tensor_parallel_split_mapping,
                            fliter_dict_keys,
                            device,
                            quantization_linear_list,
                            quantization_config,
                            dtype,
                            return_numpy,
                        ): keys
                        for keys in keys_groups
                    }
                    for future in concurrent.futures.as_completed(future_to_key):
                        res_state_dict, res_scale_dict = future.result()
                        state_dict.update(res_state_dict)
                        scale_dict.update(res_scale_dict)

            if not return_numpy:
                if device == "cpu":
                    with device_guard():
                        for k in list(state_dict.keys()):
                            state_dict[k] = paddle.Tensor.__call__(state_dict.pop(k), zero_copy=True)
                elif device == "pin_memory":
                    for k in list(state_dict.keys()):
                        state_dict[k] = paddle.to_tensor(state_dict.pop(k), place=paddle.CUDAPinnedPlace())

            if len(scale_dict) != 0:
                if ckpt_quant_stage == "O0":
                    raise ValueError('optimizer weight has quantization scales but `ckpt_quant_stage` is set to "O0"')
                state_dict = dequant_unified_optimizer(state_dict, ckpt_quant_stage, scale_dict, use_pd=True)

            return state_dict

    state_dict = paddleformers_load(checkpoint_file, map_location="cpu")
    return state_dict


def resolve_weight_file_from_hf_hub(
    repo_id: str, cache_dir: str, convert_from_torch: bool, subfolder=None, use_safetensors=False
):
    """find the suitable weight file name

    Args:
        repo_id (str): repo name of huggingface hub
        cache_dir (str): cache dir for hf
        convert_from_torch (bool): whether support converting pytorch weight file to paddle weight file
        subfolder (str, optional) An optional value corresponding to a folder inside the repo.
    """
    is_sharded = False

    if use_safetensors:
        file_name_list = [
            SAFE_WEIGHTS_INDEX_NAME,
            SAFE_WEIGHTS_NAME,
        ]
    else:
        file_name_list = [
            PYTORCH_WEIGHTS_INDEX_NAME,
            PADDLE_WEIGHTS_INDEX_NAME,
            PYTORCH_WEIGHTS_NAME,
            PADDLE_WEIGHTS_NAME,
            SAFE_WEIGHTS_NAME,  # (NOTE,lxl): 兼容极端情况
        ]
    resolved_file = None
    for fn in file_name_list:
        resolved_file = cached_file_for_hf_hub(
            repo_id, fn, cache_dir, subfolder, _raise_exceptions_for_missing_entries=False
        )
        if resolved_file is not None:
            if resolved_file.endswith(".json"):
                is_sharded = True
            break

    if resolved_file is None:
        str_name_list = ", ".join(file_name_list)
        raise EnvironmentError(
            f"{repo_id} does not appear to have a file named {str_name_list}. Checkout "
            f"'https://huggingface.co/{repo_id}' for available files."
        )

    return resolved_file, is_sharded


def register_base_model(cls):
    """
    A decorator for `PretrainedModel` class. It first retrieves the parent class
    of the class being decorated, then sets the `base_model_class` attribute
    of that parent class to be the class being decorated. In summary, the decorator registers
    the decorated class as the base model class in all derived classes under the same architecture.

    Args:
        cls (PretrainedModel): The class (inherited from PretrainedModel) to be decorated .

    Returns:
        PretrainedModel: The input class `cls` after decorating.

    Example:
        .. code-block::

            from paddleformers.transformers import BertModel, register_base_model

            BertModel = register_base_model(BertModel)
            assert BertModel.base_model_class == BertModel
    """
    base_cls = cls.__bases__[0]
    assert issubclass(
        base_cls, PretrainedModel
    ), "`register_base_model` should be used on subclasses of PretrainedModel."
    base_cls.base_model_class = cls
    return cls


class BackboneMixin:
    def forward_with_filtered_kwargs(self, *args, **kwargs):
        signature = dict(inspect.signature(self.forward).parameters)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in signature}

        return self(*args, **filtered_kwargs)


_re_layer_prefix = re.compile(r"\.(\d+)\.")


def _partion_for_pipeline_mode(keys):
    # the keys should be sort in networks order
    # TODO maybe handle tie_weight ?
    def layer_prefix(key):
        ret = _re_layer_prefix.search(key)
        if ret is not None:
            return key[0 : ret.end()]
        return ""

    keys = list(keys)
    start_idx = -1
    prefix_str = None
    partition_map = {}
    for k in keys:
        prefix = layer_prefix(k)
        if prefix != prefix_str:
            prefix_str = prefix
            start_idx += 1
        partition_map[k] = start_idx

    # if only one partition, we don't partition it
    if start_idx < 1:
        return {keys[i]: i for i in range(len(keys))}

    return partition_map


def shard_checkpoint(
    state_dict: Dict[str, paddle.Tensor],
    max_shard_size: Union[int, str] = "10GB",
    weights_name: str = PADDLE_WEIGHTS_NAME,
    shard_format="naive",
):
    """
    Splits a model state dictionary in sub-checkpoints so that the final size of each sub-checkpoint does not exceed a
    given size.

    The sub-checkpoints are determined by iterating through the `state_dict` in the order of its keys, so there is no
    optimization made to make each sub-checkpoint as close as possible to the maximum size passed. For example, if the
    limit is 10GB and we have weights of sizes [6GB, 6GB, 2GB, 6GB, 2GB, 2GB] they will get sharded as [6GB], [6+2GB],
    [6+2+2GB] and not [6+2+2GB], [6+2GB], [6GB].

    <Tip warning={true}>

    If one of the model's weight is bigger that `max_sahrd_size`, it will end up in its own sub-checkpoint which will
    have a size greater than `max_shard_size`.

    </Tip>

    Args:
        state_dict (`Dict[str, paddle.Tensor]`): The state dictionary of a model to save.
        max_shard_size (`int` or `str`, *optional*, defaults to `"10GB"`):
            The maximum size of each sub-checkpoint. If expressed as a string, needs to be digits followed by a unit
            (like `"5MB"`).
        weights_name (`str`, *optional*, defaults to `"model_state.pdparams"`):
            The name of the model save file.
        shard_format (`str`, *optional*, defaults to `"naive"`):
            support naive or pipeline.
    """
    assert shard_format in [
        "naive",
        "pipeline",
    ], f"Invalid shard_format: {shard_format}, it show be `naive` or `pipeline`."

    max_shard_size = convert_file_size_to_int(max_shard_size)

    sharded_state_dicts = []
    current_block = {}
    current_block_size = 0
    total_size = 0

    if shard_format == "naive":
        for key, weight in state_dict.items():
            # _C_ops.numel not yet support paddle.int8
            weight_size = np.prod(weight.shape) * dtype_byte_size(weight.dtype)
            # If this weight is going to tip up over the maximal size, we split.
            if current_block_size + weight_size > max_shard_size:
                # fix if the first param is large than max_shard_size
                if len(current_block) > 0:
                    sharded_state_dicts.append(current_block)
                current_block = {}
                current_block_size = 0

            current_block[key] = weight
            current_block_size += weight_size
            total_size += weight_size

        # Add the last block
        sharded_state_dicts.append(current_block)

    if shard_format == "pipeline":
        parttion_map = _partion_for_pipeline_mode(state_dict.keys())
        partition_num = max(parttion_map.values())

        for index in range(partition_num + 1):
            weight_names = [k for k, v in parttion_map.items() if v == index]
            weight_size = sum(
                state_dict[key].numel().item() * dtype_byte_size(state_dict[key].dtype) for key in weight_names
            )

            # try to add new block
            if current_block_size + weight_size > max_shard_size:
                # fix if the first param is large than max_shard_size
                if len(current_block) > 0:
                    sharded_state_dicts.append(current_block)
                current_block = {}
                current_block_size = 0
            for key in weight_names:
                current_block[key] = state_dict[key]
            current_block_size += weight_size
            total_size += weight_size

        # Add the last block
        sharded_state_dicts.append(current_block)
        logger.info(f"The average size of partition is around: {total_size//partition_num}")

    # If we only have one shard, we return it
    if len(sharded_state_dicts) == 1:
        return {weights_name: sharded_state_dicts[0]}, None

    # Otherwise, let's build the index
    weight_map = {}
    shards = {}
    weights_name_suffix = Path(weights_name).suffix
    for idx, shard in enumerate(sharded_state_dicts):
        # replace `suffix` -> `-00001-of-00002suffix`
        shard_file = weights_name.replace(
            weights_name_suffix, f"-{idx+1:05d}-of-{len(sharded_state_dicts):05d}{weights_name_suffix}"
        )
        shards[shard_file] = shard
        for key in shard.keys():
            weight_map[key] = shard_file

    # Add the metadata
    metadata = {"total_size": int(total_size)}
    index = {"metadata": metadata, "weight_map": weight_map}
    return shards, index


def load_sharded_checkpoint(model, folder, variant=None, strict=True, prefer_safe=False):
    """
    This is the same as [`paddle.nn.Layer.set_state_dict`]
    but for a sharded checkpoint.

    This load is performed efficiently: each checkpoint shard is loaded one by one in RAM and deleted after being
    loaded in the model.

    Args:
        model (`paddle.nn.Module`): The model in which to load the checkpoint.
        folder (`str` or `os.PathLike`): A path to a folder containing the sharded checkpoint.
        variant (`str`): The model variant.
        strict (`bool`, *optional`, defaults to `True`):
            Whether to strictly enforce that the keys in the model state dict match the keys in the sharded checkpoint.
        prefer_safe (`bool`, *optional*, defaults to `False`):
            If both safetensors and Paddle save files are present in checkpoint and `prefer_safe` is True, the safetensors
            files will be loaded. Otherwise, Paddle files are always loaded when possible.

    Returns:
        `NamedTuple`: A named tuple with `missing_keys` and `unexpected_keys` fields
            - `missing_keys` is a list of str containing the missing keys
            - `unexpected_keys` is a list of str containing the unexpected keys
    """
    # Load the index
    index_file = os.path.join(folder, _add_variant(PADDLE_WEIGHTS_INDEX_NAME, variant))
    safe_index_file = os.path.join(folder, _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant))

    index_present = os.path.isfile(index_file)
    safe_index_present = os.path.isfile(safe_index_file)

    if not index_present and not (safe_index_present and is_safetensors_available()):
        filenames = (
            (_add_variant(PADDLE_WEIGHTS_INDEX_NAME, variant), _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant))
            if is_safetensors_available()
            else (_add_variant(PADDLE_WEIGHTS_INDEX_NAME, variant),)
        )
        raise ValueError(f"Can't find a checkpoint index ({' or '.join(filenames)}) in {folder}.")

    load_safe = False
    if safe_index_present:
        if prefer_safe:
            if is_safetensors_available():
                load_safe = True  # load safe due to preference
            else:
                logger.warning(
                    f"Cannot load sharded checkpoint at {folder} safely since safetensors is not installed!"
                )
        elif not index_present:
            load_safe = True

    load_index = safe_index_file if load_safe else index_file

    with open(load_index, "r", encoding="utf-8") as f:
        index = json.load(f)

    shard_files = list(set(index["weight_map"].values()))

    # If strict=True, error before loading any of the state dicts.
    loaded_keys = index["weight_map"].keys()
    model_keys = model.state_dict().keys()
    missing_keys = [key for key in model_keys if key not in loaded_keys]
    unexpected_keys = [key for key in loaded_keys if key not in model_keys]
    if strict and (len(missing_keys) > 0 or len(unexpected_keys) > 0):
        error_message = f"Error(s) in loading state_dict for {model.__class__.__name__}"
        if len(missing_keys) > 0:
            str_missing_keys = ",".join([f'"{k}"' for k in missing_keys])
            error_message += f"\nMissing key(s): {str_missing_keys}."
        if len(unexpected_keys) > 0:
            str_unexpected_keys = ",".join([f'"{k}"' for k in unexpected_keys])
            error_message += f"\nMissing key(s): {str_unexpected_keys}."
        raise RuntimeError(error_message)

    loader = safe_load_file if load_safe else partial(paddleformers_load, map_location="cpu")

    for shard_file in shard_files:
        state_dict = loader(os.path.join(folder, shard_file))
        with warnings.catch_warnings():
            warnings.resetwarnings()
            warnings.filterwarnings("ignore", message=r".*is not found in the provided dict.*")
            model.set_state_dict(state_dict)

        # Make sure memory is fred before we load the next state dict.
        del state_dict
        gc.collect()

    # Return the same thing as PaddlePaddle set_state_dict function.
    return missing_keys, unexpected_keys


def faster_set_state_dict(model, state_dict, model_state_dict=None, strict_dtype=True):
    if model_state_dict is None:
        model_state_dict = model.state_dict()
    # the state_dict will be destroyed.
    unused_keys = set(state_dict.keys())
    unset_keys = set(model_state_dict.keys())
    with paddle.no_grad():
        for k, v in model_state_dict.items():
            if k in state_dict:
                v_new = state_dict.pop(k)
                if not isinstance(v_new, paddle.Tensor):
                    raise ValueError(
                        f"faster_set_state_dict need state dict with paddle.Tensor, but got {type(v_new)}"
                    )
                # 2. cast param / Tensor to dtype
                #
                if v.dtype != v_new.dtype:
                    if strict_dtype or (not v.is_floating_point() or not v_new.is_floating_point()):
                        raise ValueError(f"for key: {k}, expect dtype {v.dtype}, but got {v_new.dtype}")
                # check shape
                if list(v.shape) != list(v_new.shape):
                    raise ValueError(f"for key: {k}, expect shape {v.shape}, but got {v_new.shape}")

                dst_tensor = v.value().get_tensor()
                place = v.place

                if not v_new.place._equals(place):
                    # clear dst_tensor for save memory
                    dst_tensor._clear()
                    # v_new = v_new._copy_to(paddle.CUDAPinnedPlace(), False)
                    new_t = v_new._copy_to(place, False)
                else:
                    new_t = v_new

                if not strict_dtype and v.dtype != new_t.dtype:
                    new_t = new_t.astype(v.dtype)

                # 4. share Tensor to origin param / Tensor
                src_tensor = new_t.value().get_tensor()
                dst_tensor._share_data_with(src_tensor)
                unset_keys.remove(k)
                unused_keys.remove(k)

    error_msgs = []
    # if len(unset_keys) > 0:
    #    error_msgs.append(f"Those weight of model is not initialized: {list(unset_keys)}")
    if len(unused_keys) > 0:
        error_msgs.append(f"Those state dict keys are not using in model: {list(unused_keys)}")

    return error_msgs


def _load_state_dict_into_model(model_to_load, state_dict, start_prefix, model_to_load_state_dict=None):
    # torch will cast dtype in load_state_dict, but paddle strictly check dtype
    if model_to_load_state_dict is None:
        model_to_load_state_dict = model_to_load.state_dict()
    if len(start_prefix) > 0:
        for key in list(state_dict.keys()):
            if key.startswith(start_prefix):
                state_dict[key.replace(start_prefix, "")] = state_dict.pop(key)

    _convert_state_dict_dtype_and_shape(state_dict, model_to_load_state_dict)

    error_msgs = []

    # TODO: add return status to state_dict
    with warnings.catch_warnings(record=True) as w:
        warnings.resetwarnings()
        # paddleformers hold  missing_keys , just ignore not found warnings.
        warnings.filterwarnings("ignore", message=r".*is not found in the provided dict.*")
        warnings.filterwarnings("ignore", message=r".*paddle.to_tensor.*")
        if len(model_to_load_state_dict) > 4000 and os.getenv("DISABLE_FASTER_SET_STATE_DICT", None) is None:
            logger.warning_once(
                "The model contains an excessive number of tensors, so we utilize the faster_set_state_dict method to load tensors into the model efficiently."
                " If any issues arise during the loading process, you can disable this feature by setting the environment variable DISABLE_FASTER_SET_STATE_DICT=1."
            )
            faster_set_state_dict(model_to_load, state_dict, model_to_load_state_dict)
        else:
            model_to_load.set_state_dict(state_dict)

        error_msgs.extend([str(x.message) for x in w])

    del state_dict

    return error_msgs


def _convert_state_dict_dtype_and_shape(state_dict, model_to_load_state_dict):
    # convert the dtype of state dict
    def is_0d_or_1d(tensor):
        return len(tensor.shape) == 0 or list(tensor.shape) == [1]

    for key, value in model_to_load_state_dict.items():
        if key in list(state_dict.keys()):
            if isinstance(state_dict[key], np.ndarray):
                raise ValueError(
                    "convert_state_dict_dtype expected paddle.Tensor not numpy.ndarray, please convert numpy.ndarray to paddle.Tensor"
                )
            # confirm parameter cast is executed on the same device as model
            # TODO: cast(FP32 -> FP16) has diff on different devices, need to fix it
            if state_dict[key].is_floating_point() and state_dict[key].dtype != value.dtype:
                state_dict[key] = paddle.cast(state_dict.pop(key), value.dtype)
            # unified 0d and 1d tensor
            if is_0d_or_1d(value) and is_0d_or_1d(state_dict[key]):
                if list(value.shape) != list(state_dict[key].shape):
                    state_dict[key] = paddle.reshape(state_dict.pop(key), value.shape)


def _load_state_dict_into_meta_model(
    model,
    state_dict,
    loaded_state_dict_keys,  # left for now but could be removed, see below
    start_prefix,
    expected_keys,
    dtype=None,
    is_safetensors=False,
    keep_in_fp32_modules=None,
    model_state_dict=None,
):
    """
    This is somewhat similar to `_load_state_dict_into_model`, but deals with a model that has some or all of its
    params on a `meta` device. It replaces the model params with the data from the `state_dict`, while moving the
    params back to the normal device, but only for `loaded_state_dict_keys`.

    `start_prefix` is used for models which insert their name into model keys, e.g. `bert` in
    `bert.pooler.dense.weight`

    """
    from paddle.common_ops_import import convert_np_dtype_to_dtype_

    dtype = convert_np_dtype_to_dtype_(dtype)
    error_msgs = []
    if model_state_dict is None:
        model_state_dict = model.state_dict()
    for param_name, param in state_dict.items():
        # First part of the test is always true as loaded_state_dict_keys always contains state_dict keys.
        if param_name not in loaded_state_dict_keys or param_name not in expected_keys:
            continue

        if param_name.startswith(start_prefix):
            param_name = param_name[len(start_prefix) :]

        if param.place != paddle.framework._current_expected_place():
            param = param._copy_to(paddle.framework._current_expected_place(), False)

        # # We convert floating dtypes to the `dtype` passed. We want to keep the buffers/params
        # # in int/uint/bool and not cast them.
        if dtype is not None and paddle.is_floating_point(param):
            if (
                keep_in_fp32_modules is not None
                and any(module_to_keep_in_fp32 in param_name for module_to_keep_in_fp32 in keep_in_fp32_modules)
                and (dtype == paddle.float16 or dtype == paddle.bfloat16)
            ):
                param = param.astype(dtype=paddle.float32)
            else:
                param = param.astype(dtype=dtype)

        if dtype is None:
            old_param = model
            splits = param_name.split(".")
            for split in splits:
                old_param = getattr(old_param, split)
                if old_param is None:
                    break

            if old_param is not None:
                param = param.astype(dtype=old_param.dtype)
        with paddle.no_grad():
            model_state_dict[param_name].get_tensor()._share_data_with(param.value().get_tensor())
            param.value().get_tensor()._clear()
    return error_msgs


@six.add_metaclass(InitTrackerMeta)
class PretrainedModel(Layer, GenerationMixin, ConversionMixin):
    """
    The base class for all pretrained models. It mainly provides common methods
    for loading (construction and loading) and saving pretrained models. Loading
    and saving also rely on the following class attributes which should be overridden
    by derived classes accordingly:

    - **model_config_file** (str): Represents the file name of model configuration
      for configuration saving and loading in local file system. The value is
      `model_config.json`.
    - **resource_files_names** (dict): Name of local file where the model configuration
      can be saved and loaded locally. Currently, resources only include the model state,
      thus the dict only includes `'model_state'` as key with corresponding
      value `'model_state.pdparams'` for model weights saving and loading.
    - **pretrained_init_configuration** (dict): Provides the model configurations
      of built-in pretrained models (contrasts to models in local file system).
      It has pretrained model names as keys (such as `bert-base-uncased`), and
      the values are dict preserving corresponding configuration for model initialization.
    - **pretrained_resource_files_map** (dict): Provides resource URLs of built-in
      pretrained models (contrasts to models in local file system).
      It has the same key as resource_files_names (that is "model_state"),
      and the corresponding value is a dict with specific model name to model weights URL mapping
    - **base_model_prefix** (str): Represents the attribute associated to the
      base model in derived classes of the same architecture adding layers on
      top of the base model. Note: A base model class is pretrained model class
      decorated by `register_base_model`, such as `BertModel`; A derived model
      class is a pretrained model class adding layers on top of the base model,
      and it has a base model as attribute, such as `BertForSequenceClassification`.

    Methods common to models for text generation are defined in `GenerationMixin`
    and also inherited here.

    Besides, metaclass `InitTrackerMeta` is used to create `PretrainedModel`,
    by which subclasses can track arguments for initialization automatically.
    """

    model_config_file = CONFIG_NAME
    pretrained_init_configuration = {}
    # TODO: more flexible resource handle, namedtuple with fields as:
    # resource_name, saved_file, handle_name_for_load(None for used as __init__
    # arguments), handle_name_for_save
    resource_files_names = {"model_state": PADDLE_WEIGHTS_NAME}
    pretrained_resource_files_map = {}
    base_model_prefix = ""
    main_input_name = "input_ids"
    config_class = None
    _keep_in_fp32_modules = None

    # a list of `re` patterns of `state_dict` keys that should be removed from the list of missing
    # keys we find (keys inside the model but not in the checkpoint) and avoid unnecessary warnings.
    _keys_to_ignore_on_load_missing = None
    # a list of `re` patterns of `state_dict` keys that should be removed from the list of
    # unexpected keys we find (keys inside the checkpoint but not the model) and avoid unnecessary
    # warnings.
    _keys_to_ignore_on_load_unexpected = None
    # a list of `state_dict` keys to ignore when saving the model (useful for keys that aren't
    # trained, but which are either deterministic or tied variables)
    _keys_to_ignore_on_save = None
    _tied_weights_keys = None

    def __init__(self, *args, **kwargs):
        super(PretrainedModel, self).__init__()

        if not self.constructed_from_pretrained_config():
            return

        # extract config from args
        config = None
        for arg in args:
            if isinstance(arg, PretrainedConfig):
                config = arg
                break
        if config is not None:
            self.config: PretrainedConfig = config
            self.model_config_file = CONFIG_NAME
            self.generation_config = GenerationConfig.from_model_config(self.config) if self.can_generate() else None
            return

        # extract config from kwargs
        if "config" not in kwargs:
            raise ValueError(
                "PretrainedConfig instance not found in the arguments, you can set it as args or kwargs with config field"
            )

        config = kwargs["config"]
        if not isinstance(config, PretrainedConfig):
            raise TypeError("config parameter should be the instance of PretrainedConfig")

        self.config: PretrainedConfig = kwargs["config"]
        self.generation_config = GenerationConfig.from_model_config(self.config) if self.can_generate() else None
        self.model_config_file = CONFIG_NAME
        self.warnings_issued = {}

    def _post_init(self, original_init, *args, **kwargs):
        """
        It would be hooked after `__init__` to add a dict including arguments of
        `__init__` as a attribute named `config` of the pretrained model instance.
        """
        if not self.constructed_from_pretrained_config():
            init_dict = fn_args_to_dict(original_init, *((self,) + args), **kwargs)
            self.config = init_dict

        # only execute when it's the base method
        if (
            original_init.__module__ != "paddleformers.transformers.model_utils"
            and self.__class__.init_weights is PretrainedModel.init_weights
        ):
            self.init_weights()

        # Note:
        # 1. PipelineLayer will create parameters for each layer and
        # call `_synchronize_shared_weights()` to synchronize the shared parameters.
        # 2. When setting the model `state_dict`, `_synchronize_shared_weights` will be called to
        # synchronize the shared parameters.
        # However, `self._init_weights` will re-initialize the parameters without
        # synchronizing the shared parameters. If the following step does not load a checkpoint,
        # the shared parameters will be different.

        if isinstance(self, PipelineLayer):
            self._synchronize_shared_weights()

    def _init_weights(self, layer):
        """
        Initialize the weights. This method should be overridden by derived class.
        """
        pass

    def _initialize_weights(self, layer):
        """
        Initialize the weights if they are not already initialized.
        """
        if getattr(layer, "_is_initialized", False):
            return
        self._init_weights(layer)
        layer._is_initialized = True

    def init_weights(self):
        """
        If needed prunes and maybe initializes weights. If using a custom `PreTrainedModel`, you need to implement any
        initialization logic in `_init_weights`.
        """
        # call pure
        if _init_weights:
            # Initialize weights
            self.apply(self._initialize_weights)

            # Tie weights should be skipped when not initializing all weights
            # since from_pretrained(...) calls tie weights anyways

            # TODO(wj-Mcat): enable all tie-weights later
            # self.tie_weights()

    @classmethod
    def _from_config(cls, config, **kwargs):
        """
        All context managers that the model should be initialized under go here.

        Args:
            dtype (`paddle.dtype`, *optional*):
                Override the default `paddle.dtype` and load the model under this dtype.
        """
        dtype = kwargs.pop("dtype", None)

        if dtype is None:
            if config.dtype is not None:
                dtype = config.dtype
            else:
                dtype = paddle.get_default_dtype()

        with dtype_guard(dtype):
            model = cls(config, **kwargs)

        return model

    @classmethod
    def from_config(cls, config, **kwargs):
        """
        All context managers that the model should be initialized under go here.

        Args:
            dtype (`paddle.dtype`, *optional*):
                Override the default `paddle.dtype` and load the model under this dtype.
        """
        return cls._from_config(config, **kwargs)

    @classmethod
    def set_inference_config(cls, config, predictor_args, **kwargs):
        """
        All inference config can set here.
        Args:
            config : PretrainedConfig
                The config of the model.
            predictor_args : PredictorArgument
                The args of the predictor.
        """
        tensor_parallel_degree = kwargs.pop("tensor_parallel_degree", 1)
        tensor_parallel_rank = kwargs.pop("tensor_parallel_rank", 0)

        if predictor_args.mode == "dynamic" or predictor_args.speculate_method in ["eagle", "mtp"]:
            config.tensor_parallel_degree = tensor_parallel_degree
            config.tensor_parallel_rank = tensor_parallel_rank
            config.model_name_or_path = predictor_args.model_name_or_path
            config.quant_type = predictor_args.quant_type
            config.cachekv_int8_type = predictor_args.cachekv_int8_type
            config.use_fake_parameter = predictor_args.use_fake_parameter
            config.single_card_ptq = not predictor_args.use_fake_parameter
        config.append_attn = predictor_args.append_attn
        config.decode_strategy = predictor_args.decode_strategy
        config.mla_use_matrix_absorption = predictor_args.mla_use_matrix_absorption
        config.weightonly_group_size = predictor_args.weightonly_group_size
        config.weight_block_size = predictor_args.weight_block_size
        config.moe_quant_type = predictor_args.moe_quant_type
        config.output_via_mq = predictor_args.output_via_mq
        config.dynamic_insert = predictor_args.dynamic_insert
        if config.quantization_config.quant_method is not None:
            predictor_args.weight_block_size = config.quantization_config.weight_block_size
            config.weight_block_size = predictor_args.weight_block_size

        if config.quantization_config.quant_type is not None:
            if predictor_args.mode == "dynamic":
                predictor_args.quant_type = config.quantization_config.quant_type
                config.quant_type = config.quantization_config.quant_type
            if "c8" in config.quant_type:
                predictor_args.cachekv_int8_type = "static"
                if predictor_args.mode == "dynamic":
                    config.cachekv_int8_type = "static"

            if predictor_args.mode == "dynamic":
                ptq_multicards_num = 0
                if os.path.exists(config.model_name_or_path):
                    prefix = "activation_scales_"
                    for filename in os.listdir(config.model_name_or_path):
                        if filename.startswith(prefix):
                            ptq_multicards_num += 1

                logger.info(f"PTQ from {ptq_multicards_num} cards, so we will not split")
                if ptq_multicards_num > 1:
                    config.single_card_ptq = False

        if predictor_args.block_attn:
            config.block_size = predictor_args.block_size
            config.max_seq_len = predictor_args.total_max_length

        if predictor_args.speculate_method is not None:
            config.speculate_method = predictor_args.speculate_method
            config.speculate_max_draft_token_num = predictor_args.speculate_max_draft_token_num
            config.speculate_verify_window = predictor_args.speculate_verify_window
            config.speculate_max_candidate_len = predictor_args.speculate_max_candidate_len
            if predictor_args.speculate_method == "inference_with_reference":
                config.speculate_max_ngram_size = predictor_args.speculate_max_ngram_size
            if predictor_args.speculate_method is not None:
                if not config.get("speculate_model_type", "None") in ["eagle", "mtp"]:
                    config.decode_strategy = "speculate_decoding"
        config.return_full_hidden_states = predictor_args.return_full_hidden_states

        predictor_args.total_max_length = config.get("infer_model_max_seq_len", predictor_args.total_max_length)
        predictor_args.mla_use_matrix_absorption = config.get(
            "mla_use_matrix_absorption", predictor_args.mla_use_matrix_absorption
        )

    @classmethod
    def confirm_inference_model(cls, predictor_args, **kwargs):
        """
        Confirm the inference model whether it need to change the AVX inference Model
        Args:
            model : PretrainedModel
                The model for inference.
            predictor_args : PredictorArgument
                The args of the predictor.
        """
        return cls

    @property
    def base_model(self):
        """
        PretrainedModel: The body of the same model architecture. It is the base
            model itself for base model or the base model attribute for derived
            model.
        """
        return getattr(self, self.base_model_prefix, self)

    @property
    def model_name_list(self):
        """
        list: Contains all supported built-in pretrained model names of the
            current PretrainedModel class.
        """
        # Todo: return all model name
        return list(self.pretrained_init_configuration.keys())

    def can_generate(self) -> bool:
        """
        Returns whether this model can generate sequences with `.generate()`.
        Returns:
            `bool`: Whether this model can generate sequences with `.generate()`.
        """
        # Detects whether `prepare_inputs_for_generation` has been overwritten, which is a requirement for generation
        if "GenerationMixin" in str(self.prepare_inputs_for_generation):
            return False
        return True

    def recompute_enable(self):
        r"""
        Enable Recompute.
        All layers with the `enable_recompute` attribute will be set to `True`
        """

        def fn(layer):
            if hasattr(layer, "enable_recompute") and (layer.enable_recompute is False or layer.enable_recompute == 0):
                layer.enable_recompute = True

        self.apply(fn)

    def recompute_disable(self):
        r"""
        Disable Recompute.
        All layers with the `enable_recompute` attribute will be set to `False`
        """

        def fn(layer):
            if hasattr(layer, "enable_recompute") and (layer.enable_recompute is False or layer.enable_recompute == 0):
                layer.enable_recompute = True

        self.apply(fn)

    def get_memory_footprint(self, return_buffers=True):
        r"""
        Get the memory footprint of a model. This will return the memory footprint of the current model in bytes.
        Useful to benchmark the memory footprint of the current model and design some tests.

        Arguments:
            return_buffers (`bool`, *optional*, defaults to `True`):
                Whether to return the size of the buffer tensors in the computation of the memory footprint. Buffers
                are tensors that do not require gradients and not registered as parameters
        """
        mem = sum([param.numel().item() * param.element_size() for param in self.parameters()])
        if return_buffers:
            mem_bufs = sum([buf.numel().item() * buf.element_size() for buf in self.buffers()])
            mem = mem + mem_bufs
        return mem

    def get_model_flops(self, *args, **kwargs):
        if hasattr(self, "_get_model_flops"):
            return self._get_model_flops()

        raise NotImplementedError(f"model of {type(self)} has not implemented the `_get_model_flops`")

    def get_hardware_flops(self, *args, **kwargs):
        if hasattr(self, "_get_hardware_flops"):
            return self._get_hardware_flops()

        raise NotImplementedError(f"model of {type(self)} has not implemented the `_get_hardware_flops`")

    def get_input_embeddings(self) -> nn.Embedding:
        """get input embedding of model

        Returns:
            nn.Embedding: embedding of model
        """
        base_model = getattr(self, self.base_model_prefix, self)
        if base_model is not self:
            return base_model.get_input_embeddings()

        raise NotImplementedError(
            f"model of {type(base_model)} has not implemented the `get_input_embeddings`"
            " or `set_input_embeddings` method"
        )

    def set_input_embeddings(self, value: Embedding):
        """set new input embedding for model

        Args:
            value (Embedding): the new embedding of model

        Raises:
            NotImplementedError: Model has not implement `set_input_embeddings` method
        """
        base_model = getattr(self, self.base_model_prefix, self)
        if base_model is not self:
            return base_model.set_input_embeddings(value)
        raise NotImplementedError(
            f"model of {type(base_model)} has not implemented the `get_input_embeddings`"
            " or `set_input_embeddings` method"
        )

    def get_output_embeddings(self) -> Optional[Embedding]:
        """To be overwrited for models with output embeddings

        Returns:
            Optional[Embedding]: the otuput embedding of model
        """
        return None

    def tie_weights(self):
        """
        Tie the weights between the input embeddings and the output embeddings.
        """
        if self.config.tie_word_embeddings:
            output_embeddings = self.get_output_embeddings()
            input_embeddings = self.get_input_embeddings()
            if output_embeddings is not None and input_embeddings is not None:
                if input_embeddings.weight.shape != output_embeddings.weight.shape:
                    logger.warning(
                        f"The shape of input embeddings is {input_embeddings.weight.shape} and the shape of output embeddings is {output_embeddings.weight.shape}. "
                        "This is only expected if you are calling the `resize_token_embeddings` method"
                    )
                output_embeddings.weight = input_embeddings.weight
                if getattr(output_embeddings, "bias", None) is not None:
                    # need to pad
                    if output_embeddings.weight.shape[0] > output_embeddings.bias.shape[0]:
                        old_bias = output_embeddings.bias
                        pad_length = output_embeddings.weight.shape[0] - old_bias.shape[0]
                        output_embeddings.bias = output_embeddings.create_parameter(
                            shape=[output_embeddings.weight.shape[0]],
                            attr=output_embeddings._bias_attr,
                            dtype=output_embeddings._dtype,
                            is_bias=True,
                        )
                        new_bias = paddle.concat(
                            [old_bias, paddle.zeros([pad_length], dtype=output_embeddings.bias.dtype)]
                        )
                        output_embeddings.bias.set_value(new_bias)
                    # need to trim
                    elif output_embeddings.weight.shape[0] < output_embeddings.bias.shape[0]:
                        new_bias = output_embeddings.bias[: output_embeddings.weight.shape[0]]
                        output_embeddings.bias = output_embeddings.create_parameter(
                            shape=[output_embeddings.weight.shape[0]],
                            attr=output_embeddings._bias_attr,
                            dtype=output_embeddings._dtype,
                            is_bias=True,
                        )
                        output_embeddings.bias.set_value(new_bias)

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """resize position embedding, this method should be overrited overwrited by downstream models

        Args:
            new_num_position_embeddings (int): the new position size

        Raises:
            NotImplementedError: when called and not be implemented
        """
        raise NotImplementedError(
            f"`resize_position_embeddings` is not implemented for {self.__class__}`. To implement it, you should "
            f"overwrite this method in the class {self.__class__} in `{self.__class__.__module__}.py`"
        )

    @classmethod
    def constructed_from_pretrained_config(cls, init_func=None) -> bool:
        """check if the model is constructed from `PretrainedConfig`
        Returns:
            bool: if the model is constructed from `PretrainedConfig`
        """
        return cls.config_class is not None and issubclass(cls.config_class, PretrainedConfig)

    def save_model_config(self, save_dir: str):
        """
        Deprecated, please use `.config.save_pretrained()` instead.
        Saves model configuration to a file named "config.json" under `save_dir`.

        Args:
            save_dir (str): Directory to save model_config file into.
        """
        logger.warning("The `save_model_config` is deprecated! Please use `.config.save_pretrained()` instead.")
        self.config.save_pretrained(save_dir)

    def save_to_hf_hub(
        self,
        repo_id: str,
        private: Optional[bool] = None,
        subfolder: Optional[str] = None,
        commit_message: Optional[str] = None,
        revision: Optional[str] = None,
        create_pr: bool = False,
    ):
        """
        Uploads all elements of this model to a new HuggingFace Hub repository.
        Args:
            repo_id (str): Repository name for your model/tokenizer in the Hub.
            private (bool, optional): Whether the model/tokenizer is set to private
            subfolder (str, optional): Push to a subfolder of the repo instead of the root
            commit_message (str, optional) — The summary / title / first line of the generated commit. Defaults to: f"Upload {path_in_repo} with huggingface_hub"
            revision (str, optional) — The git revision to commit from. Defaults to the head of the "main" branch.
            create_pr (boolean, optional) — Whether or not to create a Pull Request with that commit. Defaults to False.
                If revision is not set, PR is opened against the "main" branch. If revision is set and is a branch, PR is opened against this branch.
                If revision is set and is not a branch name (example: a commit oid), an RevisionNotFoundError is returned by the server.

        Returns: The url of the commit of your model in the given repository.
        """
        repo_url = create_repo(repo_id, private=private, exist_ok=True)

        # Infer complete repo_id from repo_url
        # Can be different from the input `repo_id` if repo_owner was implicit
        _, repo_owner, repo_name = repo_type_and_id_from_hf_id(repo_url)

        repo_id = f"{repo_owner}/{repo_name}"

        # Check if README file already exist in repo
        try:
            get_hf_file_metadata(hf_hub_url(repo_id=repo_id, filename="README.md", revision=revision))
            has_readme = True
        except EntryNotFoundError:
            has_readme = False

        with tempfile.TemporaryDirectory() as root_dir:
            if subfolder is not None:
                save_dir = os.path.join(root_dir, subfolder)
            else:
                save_dir = root_dir
            # save model
            self.save_pretrained(save_dir)
            # Add readme if does not exist
            logger.info("README.md not found, adding the default README.md")
            if not has_readme:
                with open(os.path.join(root_dir, "README.md"), "w") as f:
                    f.write(f"---\nlibrary_name: paddleformers\n---\n# {repo_id}")

            # Upload model and return
            logger.info(f"Pushing to the {repo_id}. This might take a while")
            return upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=root_dir,
                commit_message=commit_message,
                revision=revision,
                create_pr=create_pr,
            )

    def save_to_aistudio(
        self,
        repo_id,
        private=True,
        license="Apache License 2.0",
        exist_ok=True,
        safe_serialization=True,
        subfolder=None,
        merge_tensor_parallel=False,
        **kwargs
    ):
        """
        Uploads all elements of this model to a new AiStudio Hub repository.
        Args:
            repo_id (str): Repository name for your model/tokenizer in the Hub.
            token (str): Your token for the Hub.
            private (bool, optional): Whether the model/tokenizer is set to private. Defaults to True.
            license (str): The license of your model/tokenizer. Defaults to: "Apache License 2.0".
            exist_ok (bool, optional): Whether to override existing repository. Defaults to: True.
            safe_serialization (bool, optional): Whether to save the model in safe serialization way. Defaults to: True.
            subfolder (str, optional): Push to a subfolder of the repo instead of the root
            merge_tensor_parallel (bool): Whether to merge the tensor parallel weights. Defaults to False.
        """

        res = aistudio_sdk.hub.create_repo(repo_id=repo_id, private=private, license=license, **kwargs)
        if "error_code" in res:
            if res["error_code"] == 10003 and exist_ok:
                logger.info(
                    f"Repo {repo_id} already exists, it will override files with the same name. To avoid this, please set exist_ok=False"
                )
            else:
                logger.error(
                    f"Failed to create repo {repo_id}, error_code: {res['error_code']}, error_msg: {res['error_msg']}"
                )
        else:
            logger.info(f"Successfully created repo {repo_id}")

        with tempfile.TemporaryDirectory() as root_dir:
            if subfolder is not None:
                save_dir = os.path.join(root_dir, subfolder)
            else:
                save_dir = root_dir
            # save model
            self.save_pretrained(
                save_dir,
                shard_format="pipeline",
                safe_serialization=(is_safetensors_available() and safe_serialization),
                max_shard_size="5GB",
                merge_tensor_parallel=merge_tensor_parallel,
            )

            # Upload model and return
            logger.info(f"Pushing to the {repo_id}. This might take a while")
            for filename in os.listdir(save_dir):
                res = aistudio_sdk.hub.upload(
                    repo_id=repo_id, path_or_fileobj=os.path.join(save_dir, filename), path_in_repo=filename, **kwargs
                )
                if "error_code" in res:
                    logger.error(
                        f"Failed to upload {filename}, error_code: {res['error_code']}, error_msg: {res['error_msg']}"
                    )
                else:
                    logger.info(f"{filename}: {res['message']}")

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None) -> nn.Embedding:
        """
        Resizes input token embeddings matrix of the model according to new_num_tokens.

        Args:
            new_num_tokens (Optional[int]):
                The number of new tokens in the embedding matrix. Increasing the size will add newly initialized
                vectors at the end. Reducing the size will remove vectors from the end. If not provided or None, just
                returns a pointer to the input tokens embedding module of the model without doing anything.

        Returns:
            paddle.nn.Embedding: The input tokens Embeddings Module of the model.
        """
        old_embeddings: nn.Embedding = self.get_input_embeddings()
        if not new_num_tokens or new_num_tokens == old_embeddings.weight.shape[0]:
            return old_embeddings

        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.set_input_embeddings(new_embeddings)

        # 2. Update vocab_size
        self.base_model.config["vocab_size"] = new_num_tokens
        self.vocab_size = new_num_tokens

        # update init_config
        self._update_init_config(self.init_config, "vocab_size", new_num_tokens)

        # Tie the weights between the input embeddings and the output embeddings if needed.
        self.tie_weights()

        return new_embeddings

    def _update_init_config(self, init_config: dict, key: str, value: Any):
        """update init_config by <key, value> pair

        Args:
            init_config (dict): the init_config instance
            key (str): the key field
            value (Any): the new value of instance
        """
        if key in init_config:
            init_config[key] = value
            return

        for arg in init_config.get("init_args", []):
            if not isinstance(arg, PretrainedModel):
                continue
            self._update_init_config(arg.init_config, key, value)

    def _get_resized_embeddings(
        self, old_embeddings: nn.Embedding, new_num_tokens: Optional[int] = None
    ) -> nn.Embedding:
        """
        Build a resized Embedding Module from a provided token Embedding Module. Increasing the size will add newly
        initialized vectors at the end. Reducing the size will remove vectors from the end

        Args:
            old_embeddings (nn.Embedding):
                Old embeddings to be resized.
            new_num_tokens (Optional[int]):
                New number of tokens in the embedding matrix.
                Increasing the size will add newly initialized vectors at the end. Reducing the size will remove
                vectors from the end.

        Returns:
            paddle.nn.Embedding: The resized Embedding Module or the old Embedding Module if new_num_tokens is None.
        """
        if new_num_tokens is None:
            return old_embeddings

        old_num_tokens, old_embedding_dim = old_embeddings.weight.shape
        if old_num_tokens == new_num_tokens:
            return old_embeddings

        if not isinstance(old_embeddings, nn.Embedding):
            raise TypeError(
                f"Old embeddings are of type {type(old_embeddings)}, which is not an instance of {nn.Embedding}. You"
                " should either use a different resize function or make sure that old_embeddings are an instance of"
                f" {nn.Embedding}."
            )

        # Build new embeddings
        new_embeddings = nn.Embedding(
            new_num_tokens,
            old_embedding_dim,
            padding_idx=old_embeddings._padding_idx,
            sparse=old_embeddings._sparse,
        )

        # make sure that new_embeddings's dtype is same as the old embeddings' dtype
        if new_embeddings.weight.dtype != old_embeddings.weight.dtype:
            new_embeddings.to(dtype=old_embeddings.weight.dtype)

        # numbers of tokens to copy
        n = min(old_num_tokens, new_num_tokens)
        with paddle.no_grad():
            new_embeddings.weight[:n, :] = old_embeddings.weight[:n, :]

        return new_embeddings

    def __setattr__(self, name, value):
        value = adapt_stale_fwd_patch(self, name, value)
        return super(PretrainedModel, self).__setattr__(name, value)

    @classmethod
    def _resolve_model_file_path(
        cls: Type[PretrainedModel],
        pretrained_model_name_or_path: str,
        from_hf_hub: bool = False,
        from_aistudio: bool = False,
        from_modelscope: bool = False,
        cache_dir: str | None = None,
        subfolder: Optional[str] = "",
        config: PretrainedConfig = None,
        convert_from_torch: bool = False,
        use_safetensors: bool | None = None,
        variant=None,
    ) -> str:
        """resolve model target file path from `` and `cache_dir`

        1. when it is file path:
            return the weight file

        2. when it is model-name:
            2.1 check default `MODEL_HOME` + `model-mame` + model_state.pdparams
            2.2 get the url from `pretrained_resource_files_map`, and set it to `pretrained_model_name_or_path`

        3. when it is local dir:
            check whether the file<local_dir + weight_file> exist

        Args:
            cls (Type[PretrainedModel]): the inherited PretrainedModel class
            pretrained_model_name_or_path (str): the model-name/url/local_dir/local_dir
            cache_dir (Optional[str], optional): cache_dir is used when name_or_path is model-name/url. Defaults to None.
            convert_from_torch (bool, optional): whether support convert pytorch model to paddle model

        Returns:
            str: the model weight file path
        """
        is_sharded = False
        sharded_metadata = None

        if pretrained_model_name_or_path is not None:
            # the following code use a lot of os.path.join, hence setting subfolder to empty str if None
            if subfolder is None:
                subfolder = ""
            pretrained_model_name_or_path = str(pretrained_model_name_or_path)
            is_local = os.path.isdir(pretrained_model_name_or_path)

            def get_file_path(pretrained_model_name_or_path, subfolder, SAFE_WEIGHTS_NAME, variant):
                return os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_NAME, variant))

            # pretrained_model_name_or_path is file
            if os.path.isfile(pretrained_model_name_or_path):
                archive_file = pretrained_model_name_or_path
                is_local = True
            # pretrained_model_name_or_path is dir
            elif is_local:
                if use_safetensors is not False and os.path.isfile(
                    get_file_path(pretrained_model_name_or_path, subfolder, SAFE_WEIGHTS_INDEX_NAME, variant)
                ):
                    # Load from a sharded safetensors checkpoint
                    archive_file = get_file_path(
                        pretrained_model_name_or_path, subfolder, SAFE_WEIGHTS_INDEX_NAME, variant
                    )
                    is_sharded = True
                elif use_safetensors is not False and os.path.isfile(
                    get_file_path(
                        pretrained_model_name_or_path, subfolder, SAFE_WEIGHTS_INDEX_NAME, weight_name_suffix()
                    )
                ):
                    # Load from a sharded safetensors checkpoint
                    archive_file = get_file_path(
                        pretrained_model_name_or_path, subfolder, SAFE_WEIGHTS_INDEX_NAME, weight_name_suffix()
                    )
                    is_sharded = True
                elif use_safetensors is not False and os.path.isfile(
                    get_file_path(pretrained_model_name_or_path, subfolder, SAFE_WEIGHTS_NAME, variant)
                ):
                    # Load from a safetensors checkpoint
                    archive_file = get_file_path(pretrained_model_name_or_path, subfolder, SAFE_WEIGHTS_NAME, variant)
                elif use_safetensors is not False and os.path.isfile(
                    get_file_path(pretrained_model_name_or_path, subfolder, SAFE_WEIGHTS_NAME, weight_name_suffix())
                ):
                    # Load from a safetensors checkpoint
                    archive_file = get_file_path(
                        pretrained_model_name_or_path, subfolder, SAFE_WEIGHTS_NAME, weight_name_suffix()
                    )
                elif os.path.isfile(
                    get_file_path(pretrained_model_name_or_path, subfolder, PADDLE_WEIGHTS_INDEX_NAME, variant)
                ):
                    # Load from a sharded PaddlePaddle checkpoint
                    archive_file = get_file_path(
                        pretrained_model_name_or_path, subfolder, PADDLE_WEIGHTS_INDEX_NAME, variant
                    )
                    is_sharded = True
                elif os.path.isfile(
                    get_file_path(
                        pretrained_model_name_or_path, subfolder, PADDLE_WEIGHTS_INDEX_NAME, weight_name_suffix()
                    )
                ):
                    # Load from a sharded PaddlePaddle checkpoint for hybrid parallel model
                    archive_file = get_file_path(
                        pretrained_model_name_or_path, subfolder, PADDLE_WEIGHTS_INDEX_NAME, weight_name_suffix()
                    )
                    is_sharded = True
                elif os.path.isfile(
                    get_file_path(pretrained_model_name_or_path, subfolder, PADDLE_WEIGHTS_NAME, variant)
                ):
                    # Load from a PaddlePaddle checkpoint
                    archive_file = get_file_path(
                        pretrained_model_name_or_path, subfolder, PADDLE_WEIGHTS_NAME, variant
                    )
                elif os.path.isfile(
                    get_file_path(
                        pretrained_model_name_or_path,
                        subfolder,
                        PADDLE_WEIGHTS_NAME,
                        weight_name_suffix(),
                    )
                ):
                    # Load from a PaddlePaddle checkpoint for hybrid parallel model
                    archive_file = get_file_path(
                        pretrained_model_name_or_path,
                        subfolder,
                        PADDLE_WEIGHTS_NAME,
                        weight_name_suffix(),
                    )
                elif os.path.isfile(
                    os.path.join(
                        pretrained_model_name_or_path, subfolder, _add_variant(PYTORCH_WEIGHTS_INDEX_NAME, variant)
                    )
                ):
                    if from_hf_hub or convert_from_torch:
                        archive_file = os.path.join(
                            pretrained_model_name_or_path, subfolder, _add_variant(PYTORCH_WEIGHTS_INDEX_NAME, variant)
                        )
                    else:
                        raise ValueError(
                            f"Found {_add_variant(PYTORCH_WEIGHTS_INDEX_NAME, variant)} in directory"
                            f" {pretrained_model_name_or_path}. Please set convert_from_torch=True in from_pretrained. eg, Model.from_pretrained(model_name, convert_from_torch=True) "
                        )
                elif os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(PYTORCH_WEIGHTS_NAME, variant))
                ):
                    if from_hf_hub or convert_from_torch:
                        archive_file = os.path.join(
                            pretrained_model_name_or_path, subfolder, _add_variant(PYTORCH_WEIGHTS_NAME, variant)
                        )
                    else:
                        raise ValueError(
                            f"Found {_add_variant(PYTORCH_WEIGHTS_NAME, variant)} in directory"
                            f" {pretrained_model_name_or_path}. Please set convert_from_torch=True in from_pretrained. eg, Model.from_pretrained(model_name, convert_from_torch=True) "
                        )
                else:
                    raise EnvironmentError(
                        f"Error no file named {_add_variant(PADDLE_WEIGHTS_NAME, variant)}, found in directory"
                        f" {pretrained_model_name_or_path}."
                    )
            elif is_remote_url(pretrained_model_name_or_path):
                resolved_archive_file = resolve_file_path(
                    pretrained_model_name_or_path,
                    pretrained_model_name_or_path,
                    subfolder,
                    cache_dir=cache_dir,
                    from_aistudio=from_aistudio,
                    from_hf_hub=from_hf_hub,
                    from_modelscope=from_modelscope,
                )

            elif pretrained_model_name_or_path in cls.pretrained_init_configuration:
                # fetch the weight url from the `pretrained_resource_files_map`
                resource_file_url = cls.pretrained_resource_files_map["model_state"][pretrained_model_name_or_path]
                resolved_archive_file = resolve_file_path(
                    pretrained_model_name_or_path,
                    [resource_file_url],
                    subfolder,
                    cache_dir=cache_dir,
                    from_aistudio=from_aistudio,
                    from_hf_hub=from_hf_hub,
                    from_modelscope=from_modelscope,
                )
            else:
                if use_safetensors is True:
                    filenames = [
                        _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant),
                        _add_variant(SAFE_WEIGHTS_NAME, variant),
                    ]
                elif use_safetensors is None:
                    filenames = [
                        _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant),
                        _add_variant(PADDLE_WEIGHTS_INDEX_NAME, variant),
                        _add_variant(SAFE_WEIGHTS_NAME, variant),
                        _add_variant(PADDLE_WEIGHTS_NAME, variant),
                        _add_variant(PYTORCH_WEIGHTS_INDEX_NAME, variant),
                        _add_variant(PYTORCH_WEIGHTS_NAME, variant),
                    ]
                else:
                    filenames = [
                        _add_variant(PADDLE_WEIGHTS_INDEX_NAME, variant),
                        _add_variant(PADDLE_WEIGHTS_NAME, variant),
                        _add_variant(PYTORCH_WEIGHTS_INDEX_NAME, variant),
                        _add_variant(PYTORCH_WEIGHTS_NAME, variant),
                    ]
                resolved_archive_file = resolve_file_path(
                    pretrained_model_name_or_path,
                    filenames,
                    subfolder,
                    cache_dir=cache_dir,
                    from_aistudio=from_aistudio,
                    from_hf_hub=from_hf_hub,
                    from_modelscope=from_modelscope,
                )
                if resolved_archive_file is None:
                    raise EnvironmentError(
                        f"Error no files {filenames} found in repo {pretrained_model_name_or_path}."
                    )
                elif "pytorch_model.bin" in str(resolved_archive_file):
                    if not from_hf_hub and not convert_from_torch:
                        raise ValueError(
                            f"Download pytorch weight in "
                            f" {resolved_archive_file}. Please set convert_from_torch=True in from_pretrained. eg, Model.from_pretrained(model_name, convert_from_torch=True) "
                        )

            if is_local:
                logger.info(f"Loading weights file {archive_file}")
                resolved_archive_file = archive_file
            else:
                logger.info(f"Loading weights file from cache at {resolved_archive_file}")
        else:
            resolved_archive_file = None

        # We'll need to download and cache each checkpoint shard if the checkpoint is sharded.
        resolved_sharded_files = None
        if str(resolved_archive_file).endswith(".json"):
            is_sharded = True
        if is_sharded:
            # resolved_archive_file becomes a list of files that point to the different checkpoint shards in this case.
            resolved_sharded_files, sharded_metadata = get_checkpoint_shard_files(
                pretrained_model_name_or_path,
                resolved_archive_file,
                from_aistudio=from_aistudio,
                from_hf_hub=from_hf_hub,
                from_modelscope=from_modelscope,
                cache_dir=cache_dir,
                subfolder=subfolder,
            )

        return resolved_archive_file, resolved_sharded_files, sharded_metadata, is_sharded

    @classmethod
    def _load_pretrained_model(
        cls,
        model: PretrainedModel,
        state_dict: Dict[str, Tensor],
        loaded_keys: List[str],
        resolved_archive_file: Union[str, List] = [],
        pretrained_model_name_or_path=None,
        config=None,
        ignore_mismatched_sizes=False,
        low_cpu_mem_usage=False,
        dtype=None,
        keep_in_fp32_modules=None,
        quantization_linear_list=None,
        sharded_metadata=None,
    ) -> Tuple[List[str]]:
        """load the state_dict into model, and do the following things:

            * check the

        Args:
            model (PretrainedModel): the pretrained model instance
            state_dict (Dict[str, Tensor]): the model state dict data
            loaded_keys (List[str]):
            ignore_mismatched_sizes (bool, optional): whether ignore error when tensor size mismatched. Defaults to False.
            dtype (_type_, optional): the dtype of model state dict. Defaults to None.

        Returns:
            Tuple[List[str]]: _description_
        """
        is_safetensors = False

        model_state_dict = model.state_dict()
        expected_keys = list(model_state_dict.keys())
        prefix = model.base_model_prefix

        if len(prefix) > 0:
            has_prefix_module = any(s.startswith(prefix) for s in loaded_keys)
            expects_prefix_module = any(s.startswith(prefix) for s in expected_keys)
        else:
            has_prefix_module = False
            expects_prefix_module = False

        # key re-naming operations are never done on the keys
        # that are loaded, but always on the keys of the newly initialized model
        remove_prefix_from_model = not has_prefix_module and expects_prefix_module
        add_prefix_to_model = has_prefix_module and not expects_prefix_module

        if remove_prefix_from_model:
            _prefix = f"{prefix}."
            expected_keys_not_prefixed = [s for s in expected_keys if not s.startswith(_prefix)]
            expected_keys = [s[len(_prefix) :] if s.startswith(_prefix) else s for s in expected_keys]
            if quantization_linear_list is not None:
                quantization_linear_list = [
                    s[len(_prefix) :] if s.startswith(_prefix) else s for s in quantization_linear_list
                ]
        elif add_prefix_to_model:
            expected_keys = [".".join([prefix, s]) for s in expected_keys]
            if quantization_linear_list is not None:
                quantization_linear_list = [".".join([prefix, s]) for s in quantization_linear_list]

        # Weight quantization if not yet quantized & update loaded_keys
        if quantization_linear_list is not None:
            if isinstance(config.quantization_config.weight_quantize_algo, str):
                post_quantize = config.quantization_config.weight_quantize_algo in [
                    "weight_only_int4",
                    "weight_only_int8",
                ]
            elif isinstance(config.quantization_config.weight_quantize_algo, dict):
                post_quantize = any(
                    key in ["weight_only_int4", "weight_only_int8"]
                    for key in config.quantization_config.weight_quantize_algo.keys()
                )
            else:
                post_quantize = False
            if post_quantize:
                origin_loaded_keys = copy.deepcopy(loaded_keys)
            else:
                origin_loaded_keys = list(model.state_dict())
            loaded_keys = update_loaded_state_dict_keys(
                loaded_keys, quantization_linear_list, config.quantization_config
            )

        missing_keys = list(set(expected_keys) - set(loaded_keys))
        unexpected_keys = list(set(loaded_keys) - set(expected_keys))

        # Optimize for skip unused shard files for supper large model
        if sharded_metadata is not None:
            assert isinstance(resolved_archive_file, list)
            new_archive_file = []
            skip_archive_file = []
            if quantization_linear_list is None:
                expected_keys_set = set(expected_keys)
            else:
                origin_expected_keys = [k.replace("quant_weight", "weight") for k in expected_keys]
                expected_keys_set = set(expected_keys + origin_expected_keys)

            for file in resolved_archive_file:
                filename = os.path.split(file)[-1]
                if not expected_keys_set.isdisjoint(set(sharded_metadata["file_map"][filename])):
                    new_archive_file.append(file)
                else:
                    skip_archive_file.append(filename)

            resolved_archive_file = new_archive_file
            if len(skip_archive_file) > 0:
                logger.info(f"Skip load files for not contains expected key, {skip_archive_file}")

        # Some models may have keys that are not in the state by design, removing them before needlessly warning
        # the user.
        if cls._keys_to_ignore_on_load_missing is not None:
            for pat in cls._keys_to_ignore_on_load_missing:
                missing_keys = [k for k in missing_keys if re.search(pat, k) is None]

        if cls._keys_to_ignore_on_load_unexpected is not None:
            for pat in cls._keys_to_ignore_on_load_unexpected:
                unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

        # Set some modules to fp32 if any
        if keep_in_fp32_modules is not None and quantization_linear_list is None:
            for name, param in model.named_parameters():
                if any(module_to_keep_in_fp32 in name for module_to_keep_in_fp32 in keep_in_fp32_modules):
                    if param.dtype != paddle.float32:
                        param_fp32 = param.cast(dtype=paddle.float32)
                        param_fp32_tensor = param_fp32.value().get_tensor()
                        param_tensor = param.value().get_tensor()
                        param_tensor._share_data_with(param_fp32_tensor)

        # Make sure we are able to load base models as well as derived models (with heads)
        start_prefix = ""
        model_to_load = model
        # (LiuTing) Non-causalLM Model dont have base_model_prefix attr, so need to remove the prefix in model state dict keyname.
        if (
            len(cls.base_model_prefix) > 0
            and not hasattr(model, cls.base_model_prefix)
            and has_prefix_module
            and not isinstance(model, PipelinePretrainedModel)
        ):
            start_prefix = cls.base_model_prefix + "."
        if len(cls.base_model_prefix) > 0 and hasattr(model, cls.base_model_prefix) and not has_prefix_module:
            model_to_load = getattr(model, cls.base_model_prefix)
            base_model_expected_keys = list(model_state_dict.keys())
            if any(key in expected_keys_not_prefixed and key not in base_model_expected_keys for key in loaded_keys):
                raise ValueError(
                    "The state dictionary of the model you are trying to load is corrupted. Are you sure it was "
                    "properly saved?"
                )
            model_to_load_state_dict = model_to_load.state_dict()
        else:
            model_to_load_state_dict = model_state_dict

        def _find_mismatched_keys(
            state_dict,
            model_state_dict,
            loaded_keys,
            add_prefix_to_model,
            remove_prefix_from_model,
            ignore_mismatched_sizes,
        ):
            mismatched_keys = []
            if ignore_mismatched_sizes:
                for checkpoint_key in loaded_keys:
                    # If the checkpoint is sharded, we may not have the key here.
                    if checkpoint_key not in state_dict:
                        continue
                    model_key = checkpoint_key
                    if remove_prefix_from_model:
                        # The model key starts with `prefix` but `checkpoint_key` doesn't so we add it.
                        model_key = f"{prefix}.{checkpoint_key}"
                    elif add_prefix_to_model:
                        # The model key doesn't start with `prefix` but `checkpoint_key` does so we remove it.
                        model_key = ".".join(checkpoint_key.split(".")[1:])

                    if (
                        model_key in model_state_dict
                        and state_dict[checkpoint_key].shape != model_state_dict[model_key].shape
                    ):
                        mismatched_keys.append(
                            (checkpoint_key, state_dict[checkpoint_key].shape, model_state_dict[model_key].shape)
                        )
                        del state_dict[checkpoint_key]
            return mismatched_keys

        def _fuse_or_split_keys(
            state_dict, config, loaded_keys, pre_tensor_parallel_split=False, resume_state_dict=None
        ):
            if resume_state_dict is not None:
                state_dict.update(resume_state_dict)

            before_fuse_keys = list(state_dict.keys())
            if pre_tensor_parallel_split:
                tp_actions = cls.get_tensor_parallel_convert_actions(
                    config, loaded_keys, ignore_error=True, base_model_prefix=prefix
                )
            else:
                tp_actions = None
            state_dict, resume_state_dict = cls.convert_fuse_and_split(config, state_dict, tp_actions)
            after_fuse_keys = list(state_dict.keys())

            fused_keys = list(set(before_fuse_keys) - set(after_fuse_keys))
            new_keys = list(set(after_fuse_keys) - set(before_fuse_keys))

            return state_dict, resume_state_dict, fused_keys, new_keys

        if quantization_linear_list is not None:
            keep_in_fp32_modules = (
                (keep_in_fp32_modules or []) + ["weight_scale"]
                if config.quantization_config.weight_quantize_algo in ["nf4", "fp4"]
                else keep_in_fp32_modules
            )
        if state_dict is not None:
            if quantization_linear_list is not None:
                # Quantize state dict
                state_dict = convert_to_quantize_state_dict(
                    state_dict,
                    quantization_linear_list,
                    config.quantization_config,
                    dtype,
                )
            else:
                # Have loaded all state_dict, no resume state_dict
                state_dict, _, fused_keys, new_keys = _fuse_or_split_keys(
                    state_dict,
                    config,
                    loaded_keys,
                    pre_tensor_parallel_split=True
                    if config is not None and config.tensor_parallel_degree > 1
                    else False,
                )
                missing_keys = list(set(missing_keys) - set(new_keys))
                unexpected_keys = list(set(unexpected_keys) - set(fused_keys))

            mismatched_keys = _find_mismatched_keys(
                state_dict,
                model_state_dict,
                loaded_keys,
                add_prefix_to_model,
                remove_prefix_from_model,
                ignore_mismatched_sizes,
            )

            if quantization_linear_list is not None:
                error_msgs = _load_state_dict_into_meta_model(
                    model_to_load,
                    state_dict,
                    loaded_keys,
                    start_prefix,
                    expected_keys,
                    dtype=dtype,
                    is_safetensors=is_safetensors,
                    keep_in_fp32_modules=keep_in_fp32_modules,
                )
            else:
                error_msgs = _load_state_dict_into_model(
                    model_to_load,
                    state_dict,
                    start_prefix,
                    model_to_load_state_dict,
                )
        else:
            # Sharded checkpoint or whole but low_cpu_mem_usage==True

            # This should always be a list but, just to be sure.
            if not isinstance(resolved_archive_file, list):
                resolved_archive_file = [resolved_archive_file]

            error_msgs = []
            mismatched_keys = []
            resume_state_dict = {}
            if len(resolved_archive_file) > 1:
                resolved_archive_file = tqdm(resolved_archive_file, desc="Loading checkpoint shards")

            for shard_file in resolved_archive_file:
                pre_tensor_parallel_split = False
                if quantization_linear_list is not None:
                    if (
                        shard_file.endswith(".safetensors")
                        and config.tensor_parallel_degree > 1
                        and "tp" not in os.path.split(shard_file)[-1]
                    ):
                        pre_tensor_parallel_split = True
                        assert origin_loaded_keys is not None, "loaded_keys is not None."
                        tp_actions = cls.get_tensor_parallel_convert_actions(
                            config,
                            origin_loaded_keys,
                            ignore_error=True,
                            base_model_prefix=prefix,
                        )
                    if post_quantize:
                        # Split -> quantize(Not support mdoel save)
                        state_dict = load_state_dict(
                            shard_file,
                            tp_actions if pre_tensor_parallel_split else None,
                            None,
                        )
                        state_dict = convert_to_quantize_state_dict(
                            state_dict,
                            quantization_linear_list,
                            config.quantization_config,
                            dtype,
                        )
                    else:
                        # quantize -> split(Support mdoel save)
                        state_dict = load_state_dict(
                            shard_file,
                            tp_actions if pre_tensor_parallel_split else None,
                            None,
                            quantization_linear_list=quantization_linear_list,
                            quantization_config=config.quantization_config,
                            dtype=dtype,
                        )
                else:
                    if (
                        shard_file.endswith(".safetensors")
                        and config.tensor_parallel_degree > 1
                        and "tp" not in os.path.split(shard_file)[-1]
                    ):
                        pre_tensor_parallel_split = True
                        assert loaded_keys is not None, "loaded_keys is not None."
                        tp_actions = cls.get_tensor_parallel_convert_actions(
                            config, loaded_keys, ignore_error=True, base_model_prefix=prefix
                        )
                    # Here we use expected_keys to optimize weights loading for pipeline model. Only works for safetensors
                    filter_dict_keys = set(expected_keys)
                    fuse_actions, _ = cls.get_fuse_or_split_param_convert_actions(config, loaded_keys, is_fuse=True)
                    split_actions, _ = cls.get_fuse_or_split_param_convert_actions(config, loaded_keys, is_fuse=False)
                    for k in list(fuse_actions.keys()):
                        need_add_except_key = k[-1] in expected_keys
                        if need_add_except_key:
                            filter_dict_keys |= set(k[:-1])
                        # remove pre_tensor_parallel_split function from tp_actions
                        if pre_tensor_parallel_split:
                            for item in k[:-1]:
                                if item in tp_actions:
                                    tp_actions.pop(item, None)

                    for k in list(split_actions.keys()):
                        need_add_except_key = False
                        for item in k[:-1]:
                            if item in expected_keys:
                                need_add_except_key = True
                                break
                        if need_add_except_key:
                            filter_dict_keys.add(k[-1])
                        # remove pre_tensor_parallel_split function from tp_actions
                        if pre_tensor_parallel_split:
                            if k[-1] in tp_actions:
                                fuse_actions.pop(k[-1], None)

                    state_dict = load_state_dict(
                        shard_file,
                        tp_actions if pre_tensor_parallel_split else None,
                        filter_dict_keys,
                    )
                    # convert for fusing or splitting weights
                    state_dict, resume_state_dict, fused_keys, new_keys = _fuse_or_split_keys(
                        state_dict,
                        config,
                        loaded_keys,
                        pre_tensor_parallel_split=pre_tensor_parallel_split,
                        resume_state_dict=resume_state_dict,
                    )
                    missing_keys = list(set(missing_keys) - set(new_keys))
                    unexpected_keys = list(set(unexpected_keys) - set(fused_keys))

                # Mismatched keys contains tuples key/shape1/shape2 of weights in the checkpoint that have a shape not
                # matching the weights in the model.
                mismatched_keys += _find_mismatched_keys(
                    state_dict,
                    model_state_dict,
                    loaded_keys,
                    add_prefix_to_model,
                    remove_prefix_from_model,
                    ignore_mismatched_sizes,
                )

                if config.tensor_parallel_degree > 1 and ".tp" not in shard_file and not pre_tensor_parallel_split:
                    logger.info("Converting state_dict to Tensor Parallel Format")
                    # ignore error for multi shard, since only parts of data
                    state_dict = cls.convert_tensor_parallel(
                        None, config, state_dict=state_dict, ignore_error=len(resolved_archive_file) > 1
                    )
                    logger.info("Converted state_dict to Tensor Parallel Format")

                if low_cpu_mem_usage or quantization_linear_list is not None:
                    new_error_msgs = _load_state_dict_into_meta_model(
                        model_to_load,
                        state_dict,
                        loaded_keys,
                        start_prefix,
                        expected_keys,
                        dtype=dtype,
                        is_safetensors=is_safetensors,
                        keep_in_fp32_modules=keep_in_fp32_modules,
                        model_state_dict=model_to_load_state_dict,
                    )
                    error_msgs += new_error_msgs
                else:
                    error_msgs += _load_state_dict_into_model(
                        model_to_load, state_dict, start_prefix, model_to_load_state_dict
                    )

                # force memory release
                del state_dict
                gc.collect()

        if len(error_msgs) > 0:
            error_msg = "\n\t".join(error_msgs)
            if " but the expected shape is" in error_msg:
                error_msg += (
                    "\n\tYou may consider adding `ignore_mismatched_sizes=True` in the model `from_pretrained` method."
                )
            raise RuntimeError(f"Error(s) in loading state_dict for {model.__class__.__name__}:\n\t{error_msg}")

        if len(unexpected_keys) > 0:
            if logger.logger.level < 20:
                logger.warning(
                    f"Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when"
                    f" initializing {model.__class__.__name__}: {sorted(unexpected_keys)}\n- This IS expected if you are"
                    f" initializing {model.__class__.__name__} from the checkpoint of a model trained on another task or"
                    " with another architecture (e.g. initializing a BertForSequenceClassification model from a"
                    " BertForPreTraining model).\n- This IS NOT expected if you are initializing"
                    f" {model.__class__.__name__} from the checkpoint of a model that you expect to be exactly identical"
                    " (initializing a BertForSequenceClassification model from a BertForSequenceClassification model)."
                )
            else:
                logger.warning(
                    f"Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when"
                    f" initializing the model, - This IS expected if you are"
                    f" initializing the model from a checkpoint of a model trained on another task or"
                    " with another architecture."
                )
        else:
            logger.info(f"All model checkpoint weights were used when initializing {model.__class__.__name__}.\n")

        if len(missing_keys) > 0:
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized: {missing_keys}\nYou should probably"
                " TRAIN this model on a down-stream task to be able to use it for predictions and inference."
            )
        elif len(mismatched_keys) == 0:
            logger.info(
                f"All the weights of {model.__class__.__name__} were initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path}.\nIf your task is similar to the task the model of the checkpoint"
                f" was trained on, you can already use {model.__class__.__name__} for predictions without further"
                " training."
            )
        if len(mismatched_keys) > 0:
            mismatched_warning = "\n".join(
                [
                    f"- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated"
                    for key, shape1, shape2 in mismatched_keys
                ]
            )
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized because the shapes did not"
                f" match:\n{mismatched_warning}\nYou should probably TRAIN this model on a down-stream task to be able"
                " to use it for predictions and inference."
            )

        return model, missing_keys, unexpected_keys, mismatched_keys

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """
        Creates an instance of `PretrainedModel`. Model weights are loaded
        by specifying name of a built-in pretrained model, a pretrained model from HF Hub, a community contributed model,
        or a local file directory path.

        Args:
            pretrained_model_name_or_path (str): Name of pretrained model or dir path
                to load from. The string can be:

                - Name of a built-in pretrained model
                - Name of a pretrained model from HF Hub
                - Name of a community-contributed pretrained model.
                - Local directory path which contains model weights file("model_state.pdparams")
                  and model config file ("model_config.json").
            from_hf_hub (bool): load model from huggingface hub. Default to `False`.
            subfolder (str, optional) An optional value corresponding to a folder inside the repo.
                Only works when loading from Huggingface Hub.
            *args (tuple): Position arguments for model `__init__`. If provided,
                use these as position argument values for model initialization.
            **kwargs (dict): Keyword arguments for model `__init__`. If provided,
                use these to update pre-defined keyword argument values for model
                initialization. If the keyword is in `__init__` argument names of
                base model, update argument values of the base model; else update
                argument values of derived model.
            load_state_as_np (bool, optional): The weights read in can be chose
                to place on CPU or GPU though the model is on the default device.
                If `True`, load the model weights as `numpy.ndarray` on CPU.
                Otherwise, weights would be loaded as tensors on the default
                device. Note that if on GPU, the latter would creates extra
                temporary tensors in addition to the model weights, which
                doubles the memory usage . Thus it is suggested to use `True`
                for big models on GPU. Default to `False`.

        Returns:
            PretrainedModel: An instance of `PretrainedModel`.

        Example:
            .. code-block::

                from paddleformers.transformers import BertForSequenceClassification

                # Name of built-in pretrained model
                model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

                # Name of pretrained model from PaddleHub
                model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

                # Name of community-contributed pretrained model
                model = BertForSequenceClassification.from_pretrained('yingyibiao/bert-base-uncased-sst-2-finetuned', num_labels=3)

                # Load from local directory path
                model = BertForSequenceClassification.from_pretrained('./my_bert/')
        """
        config = kwargs.pop("config", None)
        state_dict = kwargs.pop("state_dict", None)
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.get("force_download", False)
        ignore_mismatched_sizes = kwargs.pop("ignore_mismatched_sizes", False)
        dtype = kwargs.pop("dtype", None)
        from_hf_hub = kwargs.pop("from_hf_hub", False)
        from_aistudio = kwargs.pop("from_aistudio", False)
        from_modelscope = kwargs.pop("from_modelscope", False)
        subfolder = kwargs.pop("subfolder", None)
        if subfolder is None:
            subfolder = ""
        variant = kwargs.pop("variant", None)
        use_safetensors = kwargs.pop("use_safetensors", None if is_safetensors_available() else False)

        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", False)
        convert_from_torch = kwargs.pop("convert_from_torch", None)
        load_state_as_np = kwargs.pop("load_state_as_np", None)
        if load_state_as_np is not None:
            logger.warning("`load_state_as_np` is deprecated,  please delete it!")

        model_kwargs = kwargs

        if convert_from_torch is None and from_modelscope:
            logger.warning(
                "If you are attempting to load weights from ModelScope Hub and want to disable the default behavior of considering torch weights,"
                " you can set ·convert_from_torch=False·. By default, `convert_from_torch` is set to `True`. "
            )
            convert_from_torch = True

        # from_hf_hub default enable convert_from_torch
        if from_hf_hub and convert_from_torch is None:
            logger.warning(
                "If you are attempting to load weights from Hugging Face Hub and want to disable the default behavior of considering torch weights,"
                " you can set ·convert_from_torch=False·. By default, `convert_from_torch` is set to `True`. "
            )
            convert_from_torch = True
        # convert_from_torch default is False
        if convert_from_torch is None:
            convert_from_torch = False

        # 1. get the PretrainedConfig to init model
        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                cache_dir=cache_dir,
                from_hf_hub=from_hf_hub,
                from_aistudio=from_aistudio,
                from_modelscope=from_modelscope,
                subfolder=subfolder,
                return_unused_kwargs=True,
                **kwargs,
            )
        if "from_aistudio" in model_kwargs:
            model_kwargs.pop("from_aistudio")
        if "from_modelscope" in model_kwargs:
            model_kwargs.pop("from_modelscope")

        # if not from_hf_hub and not from_aistudio:
        #     if not os.path.exists(os.path.join(cache_dir, pretrained_model_name_or_path, subfolder, CONFIG_NAME)):
        #         config.save_pretrained(os.path.join(cache_dir, pretrained_model_name_or_path, subfolder))

        # refine options for config
        convert_from_torch = cls.support_conversion(config) and convert_from_torch
        if dtype is None:
            dtype = config.dtype

        config.dtype = dtype

        init_contexts = []
        if low_cpu_mem_usage or config.quantization_config.is_weight_quantize():
            # Instantiate model.
            init_contexts.append(no_init_weights(_enable=True))
            if is_paddle_support_lazy_init():
                init_contexts.append(paddle.LazyGuard())

        if dtype:
            init_contexts.append(dtype_guard(dtype))

        # Quantization method requires empty init to avoid unnecessary GPU allocation
        if config.quantization_config.is_weight_quantize():
            quantization_init_contexts = []
            quantization_init_contexts.append(no_init_weights(_enable=True))
            if is_paddle_support_lazy_init():
                quantization_init_contexts.append(paddle.LazyGuard())

        # Keep in fp32 modules
        keep_in_fp32_modules = None
        use_keep_in_fp32_modules = False

        # resolve model_weight file
        resolved_archive_file, resolved_sharded_files, sharded_metadata, is_sharded = cls._resolve_model_file_path(
            pretrained_model_name_or_path,
            cache_dir=cache_dir,
            subfolder=subfolder,
            from_hf_hub=from_hf_hub,
            from_aistudio=from_aistudio,
            from_modelscope=from_modelscope,
            config=config,
            convert_from_torch=convert_from_torch,
            use_safetensors=use_safetensors,
            variant=variant,
        )

        if convert_from_torch and state_dict is None:
            if (
                resolved_archive_file.endswith(PYTORCH_WEIGHTS_NAME)
                or resolved_archive_file.endswith(PYTORCH_WEIGHTS_INDEX_NAME)
                or resolved_archive_file.endswith(SAFE_WEIGHTS_NAME)
                or resolved_archive_file.endswith(SAFE_WEIGHTS_INDEX_NAME)
            ):
                # try to get the name-mapping info
                convert_dir = os.path.dirname(resolved_archive_file)
                logger.info(
                    f"Starting to convert pytorch weight file<{resolved_archive_file}> to "
                    f"paddle weight file<{convert_dir}> ..."
                )
                state_dict = cls.convert(
                    resolved_archive_file,
                    config,
                    # cache_dir=os.path.join(cache_dir, pretrained_model_name_or_path, subfolder),
                    cache_dir=convert_dir,
                )
            elif (
                resolved_archive_file.endswith(PADDLE_WEIGHTS_NAME)
                or resolved_archive_file.endswith(PADDLE_WEIGHTS_INDEX_NAME)
                or resolved_archive_file.endswith(".pdparams")
            ):
                print(f"file: {resolved_archive_file} is paddle weight.")
            else:
                raise ValueError(f"Unexpected file: {resolved_archive_file} for weight conversion.")
            # load pt weights early so that we know which dtype to init the model under
        if not is_sharded and state_dict is None:
            # 4. loading non-sharded ckpt from the state dict
            if config.tensor_parallel_degree > 1 and resolved_archive_file.endswith("model_state.pdparams"):
                state_dict = cls.convert_tensor_parallel(resolved_archive_file, config)
            elif config.tensor_parallel_degree > 1 and resolved_archive_file.endswith("model.safetensors"):
                with safe_open(resolved_archive_file, framework="np", device="cpu") as f:
                    loaded_keys = f.keys()
                tp_actions = cls.get_tensor_parallel_convert_actions(config, loaded_keys)
                state_dict = load_state_dict(resolved_archive_file, tp_actions)
            else:
                state_dict = load_state_dict(resolved_archive_file)

            logger.info("Loaded weights file from disk, setting weights to model.")

        # Check if `_keep_in_fp32_modules` is not None
        use_keep_in_fp32_modules = (cls._keep_in_fp32_modules is not None) and (
            dtype == "float16" or dtype == "bfloat16"
        )

        if state_dict is not None:
            loaded_state_dict_keys = [k for k in state_dict.keys()]
            # will only support load paddle.Tensor to model.
            for k in list(state_dict.keys()):
                if not isinstance(state_dict[k], paddle.Tensor):
                    with device_guard():
                        state_dict[k] = paddle.Tensor.__call__(state_dict.pop(k), zero_copy=True)
        else:
            if is_sharded:
                loaded_state_dict_keys = sharded_metadata["all_checkpoint_keys"]
            else:
                loaded_state_dict_keys = [k for k in state_dict.keys()]

        if low_cpu_mem_usage:  # or use_keep_in_fp32_modules:
            state_dict = None

        # will only support load paddle.Tensor to model.
        if state_dict is not None:
            for k in list(state_dict.keys()):
                if not isinstance(state_dict[k], paddle.Tensor):
                    with device_guard():
                        state_dict[k] = paddle.Tensor.__call__(state_dict.pop(k), zero_copy=True)
        # 3. init the model
        init_args = config["init_args"] or ()
        with ContextManagers(init_contexts):
            model = cls(config, *init_args, **model_kwargs)

        if use_keep_in_fp32_modules:
            # low_cpu_mem_usage = True
            keep_in_fp32_modules = model._keep_in_fp32_modules
        else:
            keep_in_fp32_modules = []

        quantization_linear_list = None
        if config.quantization_config.is_weight_quantize():
            with ContextManagers(quantization_init_contexts):
                replace_with_quantization_linear(
                    model=model,
                    quantization_config=config.quantization_config,
                    llm_int8_threshold=config.quantization_config.llm_int8_threshold,
                )
                quantization_linear_list = []
                for key in model.state_dict().keys():
                    if "quant_weight" in key:
                        quantization_linear_list.append(key[:-13])

        model, missing_keys, unexpected_keys, mismatched_keys = cls._load_pretrained_model(
            model=model,
            state_dict=state_dict,
            loaded_keys=loaded_state_dict_keys,
            resolved_archive_file=resolved_sharded_files if is_sharded else resolved_archive_file,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            config=config,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            low_cpu_mem_usage=low_cpu_mem_usage,
            dtype=dtype,
            keep_in_fp32_modules=keep_in_fp32_modules,
            quantization_linear_list=quantization_linear_list,
            sharded_metadata=sharded_metadata if is_sharded else None,
        )

        # load generation_config.json
        if model.can_generate() and pretrained_model_name_or_path is not None:
            try:
                model.generation_config = GenerationConfig.from_pretrained(
                    pretrained_model_name_or_path,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    from_hf_hub=from_hf_hub,
                    from_aistudio=from_aistudio,
                    from_modelscope=from_modelscope,
                    subfolder=subfolder,
                    **kwargs,
                )
            except:
                logger.info(
                    "Generation config file not found, using a generation config created from the model config."
                )
                pass

        # Note:
        # 1. PipelineLayer will create parameters for each layer and
        # call `_synchronize_shared_weights()` to synchronize the shared parameters.
        # 2. When setting the model `state_dict`, `_synchronize_shared_weights` will be called to
        # synchronize the shared parameters.
        # However, when state dict only contains the one piece of shared parameters, the shared parameters
        # will be different from the original shared parameters.

        if isinstance(model, PipelineLayer):
            model._synchronize_shared_weights()

        if paddle.in_dynamic_mode():
            return model

        return model, state_dict

    def save_pretrained(
        self,
        save_dir: Union[str, os.PathLike],
        is_main_process: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = paddle.save,
        max_shard_size: Union[int, str] = "10GB",
        safe_serialization: bool = False,
        variant: Optional[str] = None,
        *args,
        **kwargs,
    ):
        """
        Saves model configuration and related resources (model state) as files
        under `save_dir`. The model configuration would be saved into a file named
        "model_config.json", and model state would be saved into a file
        named "model_state.pdparams".

        The `save_dir` can be used in `from_pretrained` as argument value
        of `pretrained_model_name_or_path` to re-load the trained model.

        Args:
            save_dir (str): Directory to save files into.

        Example:
            .. code-block::

                from paddleformers.transformers import BertForSequenceClassification

                model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
                model.save_pretrained('./trained_model/')
                # reload from save_directory
                model = BertForSequenceClassification.from_pretrained('./trained_model/')
        """

        assert not os.path.isfile(save_dir), "Saving directory ({}) should be a directory, not a file".format(save_dir)
        os.makedirs(save_dir, exist_ok=True)

        merge_tensor_parallel = kwargs.get("merge_tensor_parallel", False)
        config_to_save = kwargs.get("config_to_save", None)
        shard_format = kwargs.get("shard_format", "naive")  # support naive pipeline
        # variant = kwargs.get("variant", None)
        # is_main_process = kwargs.get("is_main_process", True)

        save_directory = save_dir

        if safe_serialization and not is_safetensors_available():
            raise ImportError("`safe_serialization` requires the `safetensors library: `pip install safetensors`.")

        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        os.makedirs(save_directory, exist_ok=True)
        # Save model config

        # Only save the model in distributed training setup
        model_to_save = unwrap_model(self)

        # save the string version of dtype to the config, e.g. convert paddle.float32 => "float32"
        # we currently don't use this setting automatically, but may start to use with v5

        dtype = get_parameter_dtype(model_to_save)
        if dtype is not None:
            model_to_save.config.dtype = str(dtype).split(".")[1]
        if config_to_save is None:
            config_to_save = copy.deepcopy(model_to_save.config)

        # Save the model
        if state_dict is None:
            state_dict = model_to_save.state_dict()
            if config_to_save.tensor_parallel_degree > 1:
                if not config_to_save.quantization_config.is_support_merge_tensor_parallel() and merge_tensor_parallel:
                    logger.warning(
                        f"Quantization strategy: {config_to_save.quantization_config.weight_quantize_algo} does not support merge tensor parallel, thus we set merge_tensor_parallel to False."
                    )
                    merge_tensor_parallel = False
                if merge_tensor_parallel:
                    state_dict = model_to_save.merge_tensor_parallel(state_dict, config_to_save)
                    config_to_save.tensor_parallel_degree = 1
                    if config_to_save.tensor_parallel_rank != 0:
                        logger.info("Saving with merge_tensor_parallel, tensor_parallel_rank > 0 don't need save")
                        return
                    if variant is not None and "tp" in variant:
                        variant = "_".join([x for x in variant.split("_") if "tp" not in x])
                else:
                    variant = weight_name_suffix() if variant is None else variant

        # Attach architecture to the config
        config_to_save.architectures = [clean_model_class_name(model_to_save.__class__.__name__)]
        # Save the config
        if is_main_process:
            config_to_save.save_pretrained(save_directory)
            if self.can_generate():
                model_to_save.generation_config.save_pretrained(save_directory)

        # Handle the case where some state_dict keys shouldn't be saved
        if self._keys_to_ignore_on_save is not None:
            for ignore_key in self._keys_to_ignore_on_save:
                if ignore_key in state_dict.keys():
                    del state_dict[ignore_key]

        # Shard the model if it is too big.
        weights_name = SAFE_WEIGHTS_NAME if safe_serialization else PADDLE_WEIGHTS_NAME
        weights_name = _add_variant(weights_name, variant)

        # Save model
        shards, index = shard_checkpoint(
            state_dict, max_shard_size=max_shard_size, weights_name=weights_name, shard_format=shard_format
        )

        # Clean the folder from a previous save
        for filename in os.listdir(save_directory):
            full_filename = os.path.join(save_directory, filename)
            # If we have a shard file that is not going to be replaced, we delete it, but only from the main process
            # in distributed settings to avoid race conditions.
            weights_no_suffix = weights_name.replace(".pdparams", "").replace(".safetensors", "")

            # make sure that file to be deleted matches format of sharded file, e.g. paddle_model-00001-of-00005
            filename_no_suffix = filename.replace(".pdparams", "").replace(".safetensors", "")
            reg = re.compile(r"(.*?)-\d{5}-of-\d{5}")

            if (
                filename.startswith(weights_no_suffix)
                and os.path.isfile(full_filename)
                and filename not in shards.keys()
                and is_main_process
                and reg.fullmatch(filename_no_suffix) is not None
            ):
                os.remove(full_filename)

        # Save the model
        for shard_file, shard in shards.items():
            if safe_serialization:
                # At some point we will need to deal better with save_function (used for TPU and other distributed
                # joyfulness), but for now this enough.
                for k in list(shard.keys()):
                    if isinstance(shard[k], paddle.Tensor):
                        shard[k] = shard.pop(k).cpu().numpy()
                safe_save_file(shard, os.path.join(save_directory, shard_file), metadata={"format": "np"})
            else:
                save_function(shard, os.path.join(save_directory, shard_file))

        if index is None:
            if not safe_serialization:
                path_to_weights = os.path.join(save_directory, _add_variant(PADDLE_WEIGHTS_NAME, variant))
            else:
                path_to_weights = os.path.join(save_directory, _add_variant(SAFE_WEIGHTS_NAME, variant))
            logger.info(f"Model weights saved in {path_to_weights}")

        else:
            save_index_file = SAFE_WEIGHTS_INDEX_NAME if safe_serialization else PADDLE_WEIGHTS_INDEX_NAME
            save_index_file = os.path.join(save_directory, _add_variant(save_index_file, variant))
            # Save the index as well
            with open(save_index_file, "w", encoding="utf-8") as f:
                content = json.dumps(index, indent=2) + "\n"
                f.write(content)
            logger.info(
                f"The model is bigger than the maximum size per checkpoint ({max_shard_size}) and is going to be "
                f"split in {len(shards)} checkpoint shards. You can find where each parameters has been saved in the "
                f"index located at {save_index_file}."
            )

    def merge_auto_dist_configs(self, configs):
        """
        Merged all auto dist configs into one config.
        configs is a list of config,every config is a dict,which means a model auto_dist_config.
        [
            {
                mp_config (dict): {
                    "parallelize_plan": dict, the plan to shard the layer.
                }
                pp_config (dict): {
                    "split_spec": OrderedDict|dict|str|list(str), The pipeline parallel split point.
                    "global_spec": str|list(str), make the output tensor of specific layers on global mesh.
                }
            },{
                mp_config (dict): {
                    "parallelize_plan": dict, the plan to shard the layer.
                }
                pp_config (dict): {
                    "split_spec": OrderedDict|dict|str|list(str), The pipeline parallel split point.
                    "global_spec": str|list(str), make the output tensor of specific layers on global mesh.
                }
            },....
        ]
        """
        assert isinstance(configs, (dict, list))
        if isinstance(configs, dict):
            return configs
        final_config = {
            "mp_config": None,
            "sp_config": None,
            "pp_config": None,
        }
        for config in configs:
            if "mp_config" in config and config["mp_config"] is not None:
                if final_config["mp_config"] is None:
                    final_config["mp_config"] = config["mp_config"]
                else:
                    for k, v in config["mp_config"]["parallelize_plan"].items():
                        assert (
                            k not in final_config["mp_config"]["parallelize_plan"].keys()
                        ), f"sublayer mp_config should be a subset of model but got sublayer config {config['mp_config']} and model config {final_config['mp_config']}."
                        final_config["mp_config"]["parallelize_plan"][k] = v
            if "sp_config" in config and config["sp_config"] is not None:
                if final_config["sp_config"] is None:
                    final_config["sp_config"] = config["sp_config"]
                else:
                    for k, v in config["sp_config"]["parallelize_plan"].items():
                        assert (
                            k not in final_config["sp_config"]["parallelize_plan"].keys()
                        ), f"sublayer sp_config should be a subset of model but got sublayer config {config['sp_config']} and model config {final_config['sp_config']}."
                        final_config["sp_config"]["parallelize_plan"][k] = v
            if "pp_config" in config and config["pp_config"] is not None:
                if isinstance(config["pp_config"]["split_spec"], str):
                    config["pp_config"]["split_spec"] = [config["pp_config"]["split_spec"]]
                    if final_config["pp_config"] is None:
                        final_config["pp_config"] = config["pp_config"]
                    else:
                        final_config["pp_config"]["split_spec"] += config["pp_config"]["split_spec"]
                elif isinstance(config["pp_config"]["split_spec"], (tuple, list)):
                    if final_config["pp_config"] is None:
                        final_config["pp_config"] = config["pp_config"]
                    else:
                        final_config["pp_config"]["split_spec"] += config["pp_config"]["split_spec"]

        if final_config["pp_config"] is not None and len(final_config["pp_config"]["split_spec"]) == 1:
            final_config["pp_config"]["split_spec"] = final_config["pp_config"]["split_spec"][0]

        return final_config

    def _generate_auto_dist_config(self, auto_dist_degree):
        merged_config = {
            "sp_config": None,
            "mp_config": None,
            "pp_config": None,
        }
        for name, layer in self.named_sublayers(include_self=True):
            if hasattr(layer, "auto_dist_config"):
                if name != "":
                    prefix = name + "."
                else:
                    prefix = ""
                layer_config = layer.auto_dist_config(prefix)
                merged_config = self.merge_auto_dist_configs([merged_config, layer_config])
        final_config = {
            "dp_config": None,
            "mp_config": None,
            "pp_config": None,
        }
        if "tensor_parallel" in auto_dist_degree and auto_dist_degree["tensor_parallel"]:
            merged_config["mp_config"] is not None
            final_config["mp_config"] = merged_config["mp_config"]

        if "sequence_parallel" in auto_dist_degree and auto_dist_degree["sequence_parallel"]:
            merged_config["sp_config"] is not None
            final_config["mp_config"] = merged_config["sp_config"]

        if "pipeline_parallel" in auto_dist_degree and auto_dist_degree["pipeline_parallel"]:
            merged_config["pp_config"] is not None
            final_config["pp_config"] = merged_config["pp_config"]

        if "data_sharding_parallel" in auto_dist_degree and auto_dist_degree["data_sharding_parallel"]:
            # to avoid a circular import
            from ..trainer.trainer_utils import ShardingOption

            level = 0
            if "sharding" in auto_dist_degree and auto_dist_degree["sharding"] is not None:
                sharding = auto_dist_degree["sharding"]
                if ShardingOption.SHARD_OP in sharding:
                    level = 1
                if ShardingOption.SHARD_GRAD_OP in sharding:
                    level = 2
                if ShardingOption.FULL_SHARD in sharding:
                    level = 3
            final_config["dp_config"] = {
                "sharding_level": level,
                "sharding_mesh_dim": auto_dist_degree.get("sharding_mesh_dim", None),
            }

        return final_config


class PipelinePretrainedModel(PretrainedModel):
    def __init_hook__(self):
        if not hasattr(self, "_sequential_layers"):
            self._sequential_layers = []
            self._single_to_pp_mapping = None
            self._pp_to_single_mapping = None

    def __init__(self, config, *args, **kwargs):
        self.__init_hook__()
        super().__init__(config, *args, **kwargs)

    def add_sequential_layer(self, layer_desc, name_prefix=""):
        self.__init_hook__()
        self._sequential_layers.append({"layer": layer_desc, "name_prefix": name_prefix})

    def get_sequential_layers(self):
        self.__init_hook__()
        return [x["layer"] for x in self._sequential_layers]

    def get_sequential_name_prefixes(self):
        self.__init_hook__()
        return {str(index): x["name_prefix"] for index, x in enumerate(self._sequential_layers)}

    def _set_pipeline_name_mapping(self, mappings=None):
        if mappings is not None:
            self._single_to_pp_mapping = mappings
        else:
            single_to_pp_mapping = {}
            pp_to_single_mapping = {}

            state_dict_keys = list(super().state_dict().keys())
            first_key = ""
            for k in state_dict_keys:
                if "shared_layers" not in k:
                    first_key = k
                    break
            first_key = first_key.split(".")
            # if use virtual pp_degree, the prefix is like 0.0.xxx
            # else it will be like 0.xxx
            use_virtual_pp_degree = first_key[0].isdigit() and first_key[1].isdigit()

            prefixes = self.get_sequential_name_prefixes()
            for k in state_dict_keys:
                name_splited = k.split(".")
                if use_virtual_pp_degree:
                    if name_splited[0].isdigit():
                        if name_splited[1].isdigit():
                            idx = str(int(name_splited[0]) + int(name_splited[1]))
                            single_name = [prefixes[idx]]
                            single_name.extend(name_splited[2:])
                        else:
                            single_name = [prefixes[str(len(prefixes) - 1)]]
                            single_name.extend(name_splited[2:])
                            logger.warning(
                                f"Please check! we treat this key as last layer, get {k}, set origin name as {'.'.join(single_name)}"
                            )
                    elif name_splited[0] == "shared_layers":
                        single_name = [self.get_shardlayer_prefix(name_splited, SharedLayerDesc)]
                        single_name.extend(name_splited[2:])
                    elif name_splited[0] == "local_shared_layers":
                        single_name = [self.get_shardlayer_prefix(name_splited, LocalSharedLayerDesc)]
                        single_name.extend(name_splited[2:])
                    else:
                        raise ValueError(f"Unexpected key: {k} for pp layer.")
                else:
                    idx = name_splited[0]
                    # for normal pp layer
                    if idx.isdigit():
                        # allow empty prefix
                        single_name = [] if prefixes[idx] == "" else [prefixes[idx]]
                        single_name.extend(name_splited[1:])
                    elif idx == "shared_layers":
                        single_name = [self.get_shardlayer_prefix(name_splited, SharedLayerDesc)]
                        single_name.extend(name_splited[2:])
                    elif idx == "local_shared_layers":
                        single_name = [self.get_shardlayer_prefix(name_splited, LocalSharedLayerDesc)]
                        single_name.extend(name_splited[2:])
                    else:
                        raise ValueError(f"Unexpected key: {k} for pp layer.")

                single_to_pp_mapping[".".join(single_name)] = k
                pp_to_single_mapping[k] = ".".join(single_name)

            self._single_to_pp_mapping = single_to_pp_mapping
            self._pp_to_single_mapping = pp_to_single_mapping

        return self._single_to_pp_mapping

    def get_shardlayer_prefix(self, name_splited, shared_layer_class=SharedLayerDesc):
        """_summary_
            This function retrieves the prefix of a shared layer. The process involves:
            1. Identifying all key names of shared layers, like 'shared_weight01', 'shared_weight02', etc.
            2. For instance, given name_splited = ['shared_layers', 'shared_weight01', 'weight'],
                the 'shared_layer_key' would be name_splited[1], which is 'shared_weight01'.
            3. By traversing through all layers, the function checks if the specified
                shared_layer is present in the current stage. If found, it returns the corresponding prefix.

            Note: For retrieving all SharedLayer instances in Paddle, you can refer to the following Paddle code.
            https://github.com/PaddlePaddle/Paddle/blob/2cf724d055679a1a0e48766dfb1708b920273078/python/paddle/distributed/fleet/meta_parallel/parallel_layers/pp_layers.py#L460-L513
        Args:
            name_splited (_type_): _description_

        Returns:
            _type_: _description_
        """
        shared_layer_names = {s.layer_name for s in self._layers_desc if isinstance(s, shared_layer_class)}
        assert name_splited[1] in shared_layer_names, f"The shared layer name {name_splited[1]} must be in prefixes!"
        shared_layer_key = name_splited[1]
        for idx, layer in enumerate(self._layers_desc):
            if isinstance(layer, shared_layer_class) and layer.layer_name == shared_layer_key:
                if self.get_stage_from_index(idx) == self._stage_id:
                    return self.get_sequential_name_prefixes()[str(idx)]

        # the prefix must be in the current stage, else raise error
        raise ValueError(f"The shared layer {shared_layer_key} must be in the current stage!")

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)

        if self._single_to_pp_mapping is None:
            self._set_pipeline_name_mapping()
        assert len(self._single_to_pp_mapping) > 0, "The pipeline stage must have parameters!"

        for k in list(state_dict.keys()):
            v = state_dict.pop(k)
            state_dict[self._pp_to_single_mapping[k]] = v

        return state_dict

    def set_state_dict(self, state_dict, *args, **kwargs):
        if self._single_to_pp_mapping is None:
            self._set_pipeline_name_mapping()
        assert len(self._single_to_pp_mapping) > 0, "The pipeline stage must have parameters!"

        for k in list(state_dict.keys()):
            v = state_dict.pop(k)
            if k not in self._single_to_pp_mapping:
                continue
            state_dict[self._single_to_pp_mapping[k]] = v

        ret = super().set_state_dict(state_dict, *args, **kwargs)
        return ret


def load_sharded_checkpoint_as_one(folder, variant=None, return_numpy=False):
    """

    This load is performed efficiently: each checkpoint shard is loaded one by one in RAM and deleted after being
    loaded in the model.

    Args:
        folder (`str` or `os.PathLike`): A path to a folder containing the sharded checkpoint.
        variant (`str`): The model variant.
        return_numpy (`bool`): Whether to return numpy array instead of paddle tensor.

    """
    # Load the index
    pdparams_file = os.path.join(folder, _add_variant("model_state.pdparams", variant))
    lora_pdparams_file = os.path.join(folder, _add_variant("lora_model_state.pdparams", variant))
    safetensors_file = os.path.join(folder, _add_variant("model.safetensors", variant))
    if os.path.isfile(pdparams_file):
        return paddle.load(pdparams_file, return_numpy=return_numpy)
    if os.path.isfile(lora_pdparams_file):
        return paddle.load(lora_pdparams_file, return_numpy=return_numpy)
    if os.path.isfile(safetensors_file):
        state_dict = safe_load_file(safetensors_file)
        if not return_numpy:
            for key in list(state_dict.keys()):
                if isinstance(state_dict[key], np.ndarray):
                    state_dict[key] = paddle.Tensor.__call__(state_dict.pop(key), zero_copy=True)
        return state_dict

    index_file = os.path.join(folder, _add_variant(PADDLE_WEIGHTS_INDEX_NAME, variant))
    safe_index_file = os.path.join(folder, _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant))
    safe_master_file = os.path.join(folder, _add_variant(SAFE_MASTER_WEIGHTS_INDEX_NAME, variant))
    safe_peft_file = os.path.join(folder, _add_variant(SAFE_PEFT_WEIGHTS_INDEX_NAME, variant))

    index_present = os.path.isfile(index_file)
    safe_index_present = os.path.isfile(safe_index_file)
    safe_master_present = os.path.isfile(safe_master_file)
    safe_peft_present = os.path.isfile(safe_peft_file)

    load_safe = False
    load_index = None
    if safe_index_present:
        load_safe = True  # load safe due to preference
        load_index = safe_index_file
    elif safe_master_present:
        load_safe = True
        load_index = safe_master_file
    elif index_present:
        load_index = index_file
    elif safe_peft_present:
        load_safe = True
        load_index = safe_peft_file
    else:
        raise ValueError(f"Could not find {index_file} or {safe_index_file} or {safe_peft_file}")

    with open(load_index, "r", encoding="utf-8") as f:
        index = json.load(f)

    shard_files = list(set(index["weight_map"].values()))
    loader = safe_load_file if load_safe else partial(paddleformers_load, map_location="np" if return_numpy else "cpu")

    ret = {}
    for shard_file in tqdm(shard_files):
        state_dict = loader(os.path.join(folder, shard_file))
        ret.update(state_dict)

    if not return_numpy:
        for key in list(ret.keys()):
            if isinstance(ret[key], np.ndarray):
                ret[key] = paddle.Tensor.__call__(ret.pop(key), zero_copy=True)

    return ret


def load_tp_checkpoint(folder, cls, config, return_numpy=False):
    """

    This load is performed efficiently: Load tp checkpoint only from cpu, no need to init the model.

    Args:
        folder (`str` or `os.PathLike`): A path to a folder containing the model checkpoint.
        cls (`str`): The model class.
        config (`AutoConfig`): The model config.
        return_numpy (bool): Whether load the tp checkpoint as numpy.
    """
    if config.tensor_parallel_degree == 1 or config.tensor_parallel_degree == -1:
        return load_sharded_checkpoint_as_one(folder, return_numpy=return_numpy)
    else:
        rank_model_path = os.path.join(folder, f"model_state.tp0{config.tensor_parallel_rank}.pdparams")
        model_path = os.path.join(folder, "model_state.pdparams")
        safe_model_path = os.path.join(folder, "model.safetensors")
        if os.path.exists(rank_model_path):
            return paddle.load(rank_model_path, return_numpy=return_numpy)
        elif os.path.exists(model_path):
            state_dict = cls.convert_tensor_parallel(model_path, config)
        elif os.path.exists(safe_model_path):
            with safe_open(safe_model_path, framework="np", device="cpu") as f:
                loaded_keys = f.keys()
            tp_actions = cls.get_tensor_parallel_convert_actions(config, loaded_keys)
            state_dict = load_state_dict(safe_model_path, tp_actions, return_numpy=return_numpy)
        else:  # shard files safetensors
            resolved_archive_file, resolved_sharded_files, sharded_metadata, is_sharded = cls._resolve_model_file_path(
                pretrained_model_name_or_path=folder,
                use_safetensors=True,
            )
            if len(resolved_sharded_files) > 1:
                resolved_sharded_files = tqdm(resolved_sharded_files, desc="Loading checkpoint shards")
            loaded_state_dict_keys = sharded_metadata["all_checkpoint_keys"]
            tp_actions = cls.get_tensor_parallel_convert_actions(config, loaded_state_dict_keys, ignore_error=True)
            state_dict = {}
            for shard_file in resolved_sharded_files:
                shard_state_dict = load_state_dict(
                    shard_file,
                    tp_actions,
                    loaded_state_dict_keys,
                    return_numpy=return_numpy,
                )
                state_dict.update(shard_state_dict)
    return state_dict


def clean_model_class_name(class_name, suffixes_to_strip: Union[str, List[str]] = "Pipe"):
    """
    Returns the class name of the given model with specified suffixes removed.
    This is typically used to clean up the model name before saving it to
    config.architectures, removing implementation-specific suffixes like "Pipe".
    Args:
        class_name: The __class__.__name__ attribute.
        suffixes_to_strip (str or list of str, optional): One or more suffix strings to remove
            from the class name (e.g., 'Pipe' or ['Pipe', 'Wrapper']). If None or empty,
            no stripping is applied.
    Returns:
        str: The cleaned model class name with specified suffix removed (if present).
    """
    if not suffixes_to_strip:
        return class_name

    if isinstance(suffixes_to_strip, str):
        suffixes_to_strip = [suffixes_to_strip]

    pattern = f"({'|'.join(map(re.escape, suffixes_to_strip))})$"
    return re.sub(pattern, "", class_name)
