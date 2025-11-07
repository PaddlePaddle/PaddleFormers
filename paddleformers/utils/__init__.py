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

import contextlib
import sys
from typing import TYPE_CHECKING

from ..utils.lazy_import import _LazyModule

import_structure = {
    "nested": [
        "nested_reduce_tensor",
        "nested_empty_tensor",
        "nested_broadcast_tensor",
        "nested_broadcast_tensor_with_empty",
        "nested_copy",
        "nested_copy_place",
        "flatten_list",
        "TensorHolder",
    ],
    "import_utils": [
        "is_torch_available",
        "is_decord_available",
        "is_paddlenlp_ops_available",
        "auto_dynamic_graph_pybind",
        "is_paddle_cuda_available",
        "is_package_available",
        "is_tiktoken_available",
        "uninstall_package",
        "import_module",
        "_is_package_available",
        "is_sentencepiece_available",
        "is_paddle_available",
        "is_psutil_available",
        "is_protobuf_available",
        "is_tokenizers_available",
        "is_fast_tokenizer_available",
        "install_package",
        "is_g2p_en_available",
        "is_datasets_available",
        "is_transformers_available",
        "dynamic_graph_pybind_context",
        "custom_import",
        "direct_paddleformers_import",
    ],
    "initializer": ["to"],
    "infohub": ["infohub", "InfoHub"],
    "memory_utils": ["empty_device_cache"],
    "moe_hybrid_parallel_optimizer": ["MoEHybridParallelOptimizer"],
    "paddle_patch": ["enhance_set_value", "new_repr", "_numel", "_numpy", "enhance_init", "enhance_to_tensor"],
    "serialization": [
        "seek_by_string",
        "load_torch_inner",
        "SerializationError",
        "_element_size",
        "_rebuild_tensor_stage",
        "_maybe_decode_ascii",
        "load_torch",
        "_rebuild_parameter",
        "_rebuild_parameter_with_state",
        "UnpicklerWrapperStage",
        "read_prefix_key",
        "_storage_type_to_dtype_to_map",
        "SafeUnpickler",
        "dumpy",
        "StorageType",
    ],
    "batch_sampler": ["DistributedBatchSampler"],
    "optimizer": ["AdamWMini", "AdamWCustom", "AdamWLoRAPro"],
    "env": ["CONFIG_NAME", "GENERATION_CONFIG_NAME", "LEGACY_CONFIG_NAME"],
    "log": ["logger"],
    "masking_utils": [
        "_gen_from_sparse_attn_mask_indices",
        "masked_fill",
        "is_casual_mask",
        "_make_causal_mask",
        "_expand_2d_mask",
        "build_alibi_tensor",
        "get_use_casual_mask",
        "get_triangle_upper_mask",
    ],
    "tools": ["device_guard"],
    "downloader": ["get_weights_path_from_url"],
    "type_validators": [
        "positive_any_number",
        "positive_int",
        "padding_validator",
        "truncation_validator",
        "image_size_validator",
        "device_validator",
        "resampling_validator",
        "video_metadata_validator",
        "tensor_type_validator",
    ],
}


@contextlib.contextmanager
def device_guard(device="cpu", dev_id=0):
    origin_device = paddle.device.get_device()
    if device == "cpu":
        paddle.set_device(device)
    elif device in ["gpu", "xpu", "npu"]:
        paddle.set_device("{}:{}".format(device, dev_id))
    try:
        yield
    finally:
        paddle.set_device(origin_device)


if TYPE_CHECKING:
    import paddle

    from .batch_sampler import *
    from .env import CONFIG_NAME, GENERATION_CONFIG_NAME, LEGACY_CONFIG_NAME
    from .import_utils import *
    from .infohub import infohub
    from .initializer import to
    from .log import logger
    from .memory_utils import empty_device_cache
    from .moe_hybrid_parallel_optimizer import MoEHybridParallelOptimizer

    try:
        from .optimizer import *
    except:
        logger.info("Not support custom optimizer")

    from type_validators import *

    from .serialization import load_torch

    # hack impl for EagerParamBase to function
    # https://github.com/PaddlePaddle/Paddle/blob/fa44ea5cf2988cd28605aedfb5f2002a63018df7/python/paddle/nn/layer/layers.py#L2077
    paddle.framework.io.EagerParamBase.to = to
else:
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        import_structure,
        module_spec=__spec__,
    )
