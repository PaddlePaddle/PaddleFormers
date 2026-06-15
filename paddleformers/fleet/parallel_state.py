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

# Refer to NVIDIA Megatron-LM https://github.com/NVIDIA/Megatron-LM.git
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

from typing import TYPE_CHECKING

import paddle

if TYPE_CHECKING:
    import paddle.distributed.fleet.base.topology as topo
import warnings

from .utils import GlobalMemoryBuffer

# Intra-layer model parallel group that the current rank belongs to.
_TENSOR_MODEL_PARALLEL_GROUP = None
_TENSOR_MODEL_PARALLEL_GLOBAL_RANKS = None

### Expert-related parallel states
# Naming convention:
# _EXPERT prefix in group name means it's used for expert layer in MoE models.
# _EXPERT_MODEL denotes expert parallelism which splits number of experts across the group.
# _EXPERT_TENSOR denotes tensor parallelism of expert which splits tensor across the group.
# _EXPERT_DATA denotes data parallelism of expert which replicates weight across the group.

# Expert model parallel group that current rank belongs to.
_EXPERT_MODEL_PARALLEL_GROUP = None
# Expert tensor parallel group that current rank belongs to.
_EXPERT_TENSOR_PARALLEL_GROUP = None
# Expert tensor and model combined parallel group
_EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP = None
# Expert tensor, model, pipeline combined parallel group
_EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP = None
# Expert data parallel group
_EXPERT_DATA_PARALLEL_GROUP = None
_EXPERT_DATA_PARALLEL_GROUP_GLOO = None
_INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP = None
_INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP_GLOO = None
_INTER_PARTIAL_EXPERT_DATA_PARALLEL_GROUP = None
# Parallel state values changed on the fly
_MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_EXPERT_MODEL_PARALLEL_RANK = None
_MPU_EXPERT_TENSOR_PARALLEL_WORLD_SIZE = None
_MPU_EXPERT_TENSOR_PARALLEL_RANK = None
### End of expert related parallel states

# Inter-layer model parallel group that the current rank belongs to.
_PIPELINE_MODEL_PARALLEL_GROUP = None
_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None

_VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = None
_VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None

# Data parallel group that the current rank belongs to.
_DATA_PARALLEL_GROUP = None

# Expert model parallel group that current rank belongs to.
_EXPERT_MODEL_PARALLEL_GROUP = None

# Expert data parallel group
_EXPERT_DATA_PARALLEL_GROUP = None

# Context parallel group that the current rank belongs to
_CONTEXT_PARALLEL_GROUP = None

# Data parallel group information with context parallel combined.
_DATA_PARALLEL_GROUP_WITH_CP = None
# These values enable us to change the mpu sizes on the fly.
_MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_TENSOR_MODEL_PARALLEL_RANK = None


# Memory buffers to avoid dynamic memory allocation
_GLOBAL_MEMORY_BUFFER = None


def initialize_model_parallel(
    hcg: topo.EPHybridCommunicateGroup | topo.HybridCommunicateGroup,
    virtual_pipeline_model_parallel_size: int | None = None,
):
    global _TENSOR_MODEL_PARALLEL_GROUP
    global _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS
    _TENSOR_MODEL_PARALLEL_GROUP = hcg._mp_comm_group
    _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS = hcg._mp_group

    global _PIPELINE_MODEL_PARALLEL_GROUP
    _PIPELINE_MODEL_PARALLEL_GROUP = hcg._pp_comm_group

    global _DATA_PARALLEL_GROUP
    _DATA_PARALLEL_GROUP = hcg._sharding_comm_group

    global _EXPERT_MODEL_PARALLEL_GROUP
    global _EXPERT_DATA_PARALLEL_GROUP
    _EXPERT_MODEL_PARALLEL_GROUP = getattr(hcg, "_ep_comm_group", None)
    _EXPERT_DATA_PARALLEL_GROUP = getattr(hcg, "_moe_sharding_comm_group", None)

    global _CONTEXT_PARALLEL_GROUP
    global _DATA_PARALLEL_GROUP_WITH_CP
    _CONTEXT_PARALLEL_GROUP = getattr(hcg, "_cp_comm_group", None)
    _DATA_PARALLEL_GROUP_WITH_CP = getattr(hcg, "_cp_sharding_comm_group", None)

    if virtual_pipeline_model_parallel_size is not None:
        if not hcg._pp_comm_group.nranks > 1:
            raise RuntimeError("pipeline-model-parallel size should be greater than 1 with interleaved schedule")
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = 0
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = virtual_pipeline_model_parallel_size

    # Initialize global memory buffer
    # This isn't really "parallel state" but there isn't another good place to
    # put this. If we end up with a more generic initialization of paddleformers.fleet
    # we could stick it there
    _set_global_memory_buffer()


def get_tensor_model_parallel_group(check_initialized=True):
    """Get the tensor-model-parallel group the caller rank belongs to."""
    if check_initialized:
        assert _TENSOR_MODEL_PARALLEL_GROUP is not None, "tensor model parallel group is not initialized"
    return _TENSOR_MODEL_PARALLEL_GROUP


def get_tensor_model_parallel_world_size():
    """Return world size for the tensor-model-parallel group."""
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    if _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    return 1 if get_tensor_model_parallel_group(False) is None else get_tensor_model_parallel_group().nranks


def get_tensor_model_parallel_rank():
    """Return caller's rank for the tensor-model-parallel group."""
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    if _MPU_TENSOR_MODEL_PARALLEL_RANK is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_RANK
    if get_tensor_model_parallel_world_size() == 1:
        return 0
    return get_tensor_model_parallel_group().rank


def get_pipeline_model_parallel_group(check_initialized=True):
    """Get the pipeline-model-parallel group the caller rank belongs to."""
    if check_initialized:
        assert _PIPELINE_MODEL_PARALLEL_GROUP is not None, "pipeline_model parallel group is not initialized"
    return _PIPELINE_MODEL_PARALLEL_GROUP


def get_data_parallel_group(check_initialized=True, with_context_parallel=False):
    """Get the data-parallel group the caller rank belongs to."""
    if with_context_parallel:
        if check_initialized:
            assert (
                _DATA_PARALLEL_GROUP_WITH_CP is not None
            ), "data parallel group with context parallel combined is not initialized"
        return _DATA_PARALLEL_GROUP_WITH_CP
    else:
        if check_initialized:
            assert _DATA_PARALLEL_GROUP is not None, "data parallel group is not initialized"
        return _DATA_PARALLEL_GROUP


def get_expert_model_parallel_group(check_initialized=True):
    """Get the expert-model-parallel group the caller rank belongs to."""
    if check_initialized:
        assert _EXPERT_MODEL_PARALLEL_GROUP is not None, "expert model parallel group is not initialized"
    return _EXPERT_MODEL_PARALLEL_GROUP


def get_expert_data_parallel_group(check_initialized=True):
    """Get expert data parallel group."""
    if check_initialized:
        assert _EXPERT_DATA_PARALLEL_GROUP is not None, "Expert data parallel group is not initialized"
    return _EXPERT_DATA_PARALLEL_GROUP


def get_context_parallel_group(check_initialized=False):
    """Get the context-parallel group the caller rank belongs to."""
    if check_initialized:
        assert _CONTEXT_PARALLEL_GROUP is not None, "context parallel group is not initialized"
    return _CONTEXT_PARALLEL_GROUP


def get_context_parallel_world_size():
    """Return world size for the tensor-model-parallel group."""
    if get_context_parallel_group() is None:
        return 1
    else:
        return get_context_parallel_group().world_size


def get_pipeline_model_parallel_world_size():
    """Return world size for the pipeline-model-parallel group."""
    # TODO: Support get_pipeline_model_parallel_world_size
    return 1

    global _PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    if _PIPELINE_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    return get_pipeline_model_parallel_group().nranks


def set_pipeline_model_parallel_world_size(world_size):
    """Set the pipeline-model-parallel size"""
    global _PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _PIPELINE_MODEL_PARALLEL_WORLD_SIZE = world_size


def get_pipeline_model_parallel_rank():
    # TODO: Support get_pipeline_model_parallel_rank
    warnings.warn("get_pipeline_model_parallel_rank is not implemented yet, always returns 0 for now. ")
    return 0


def is_pipeline_first_stage(ignore_virtual=True, vp_stage=None):
    """Return True if in the first pipeline model-parallel stage, False otherwise."""
    if not ignore_virtual and get_virtual_pipeline_model_parallel_world_size() is not None:
        assert vp_stage is not None, "vp_stage must be passed if virtual pipeline is enabled"

        if vp_stage != 0:
            return False
    return get_pipeline_model_parallel_rank() == 0


def is_pipeline_last_stage(ignore_virtual=True, vp_stage=None):
    """Return True if in the last pipeline-model-parallel stage, False otherwise."""
    if not ignore_virtual and get_virtual_pipeline_model_parallel_world_size() is not None:
        assert vp_stage is not None, "vp_stage must be passed if virtual pipeline is enabled"

        if vp_stage != (get_virtual_pipeline_model_parallel_world_size() - 1):
            return False
    return get_pipeline_model_parallel_rank() == (get_pipeline_model_parallel_world_size() - 1)


def get_virtual_pipeline_model_parallel_rank():
    """Return the virtual pipeline-parallel rank."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    return _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK


def set_virtual_pipeline_model_parallel_rank(rank):
    """Set the virtual pipeline-parallel rank."""
    warnings.warn(
        "set_virtual_pipeline_model_parallel_rank in global scope is deprecated. "
        "Pass vp_stage explicitly to is_pipeline_first_stage, is_pipeline_last_stage, etc.",
        DeprecationWarning,
    )
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = rank


def set_expert_model_parallel_world_size(world_size):
    """Sets the expert-model-parallel world size."""
    global _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE
    _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE = world_size


def get_expert_model_parallel_rank():
    """Return caller's rank in the expert-model-parallel group."""
    if _MPU_EXPERT_MODEL_PARALLEL_RANK is not None:
        return _MPU_EXPERT_MODEL_PARALLEL_RANK
    if paddle.distributed.is_initialized() and get_expert_model_parallel_group(False) is not None:
        return get_expert_model_parallel_group().rank
    else:
        return 0


def set_expert_model_parallel_rank(rank):
    """Set expert-model-parallel rank."""
    global _MPU_EXPERT_MODEL_PARALLEL_RANK
    _MPU_EXPERT_MODEL_PARALLEL_RANK = rank


def get_virtual_pipeline_model_parallel_world_size():
    """Return the virtual pipeline-parallel world size."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    return _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE


def get_embedding_group(check_initialized=True):
    raise NotImplementedError("Not supported get_embedding_group yet.")
    '''
    """Get the embedding group the caller rank belongs to."""
    if check_initialized:
        assert _EMBEDDING_GROUP is not None, "embedding group is not initialized"
    return _EMBEDDING_GROUP
    '''


def get_expert_tensor_parallel_group(check_initialized=False):
    """Get the expert-tensor-parallel group the caller rank belongs to."""
    if check_initialized:
        assert _EXPERT_TENSOR_PARALLEL_GROUP is not None, "Expert tensor parallel group is not initialized"
    return _EXPERT_TENSOR_PARALLEL_GROUP


def get_expert_tensor_parallel_world_size():
    """Return world size for the expert tensor parallel group."""
    global _MPU_EXPERT_TENSOR_PARALLEL_WORLD_SIZE
    if _MPU_EXPERT_TENSOR_PARALLEL_WORLD_SIZE is not None:
        return _MPU_EXPERT_TENSOR_PARALLEL_WORLD_SIZE
    # Use tensor parallel group world size for backward compatibility otherwise
    if not _EXPERT_TENSOR_PARALLEL_GROUP:
        return _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    else:
        return get_expert_tensor_parallel_group().size()


def set_expert_tensor_parallel_world_size(world_size):
    "Set expert tensor model parallel size"
    global _MPU_EXPERT_TENSOR_PARALLEL_WORLD_SIZE
    _MPU_EXPERT_TENSOR_PARALLEL_WORLD_SIZE = world_size


def get_expert_tensor_parallel_rank():
    """Return my rank for the expert tensor parallel group."""
    global _MPU_EXPERT_TENSOR_PARALLEL_RANK
    if _MPU_EXPERT_TENSOR_PARALLEL_RANK is not None:
        return _MPU_EXPERT_TENSOR_PARALLEL_RANK
    if _EXPERT_TENSOR_PARALLEL_GROUP is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_RANK
    else:
        return 0


def set_expert_tensor_parallel_rank(rank):
    "Set expert tensor model parallel rank"
    global _MPU_EXPERT_TENSOR_PARALLEL_RANK
    _MPU_EXPERT_TENSOR_PARALLEL_RANK = rank


def get_expert_tensor_and_model_parallel_group(check_initialized=True):
    """Get the expert-tensor and expert-model group the caller rank belongs to."""
    if check_initialized:
        assert (
            _EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP is not None
        ), "Expert tensor and model parallel group is not initialized"
    return _EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP


def get_expert_tensor_and_model_parallel_world_size():
    """Return world size for the expert model parallel group times expert tensor parallel group."""
    if paddle.distributed.is_available() and paddle.distributed.is_initialized():
        world_size = get_expert_tensor_and_model_parallel_group().size()
        return world_size
    else:
        return 0


def get_expert_tensor_and_model_parallel_rank():
    """Return caller's rank in the joint tensor- and expert-model-parallel group."""
    if paddle.distributed.is_available() and paddle.distributed.is_initialized():
        return get_expert_tensor_and_model_parallel_group().rank()
    else:
        return 0


### End of expert-related functions region


def _set_global_memory_buffer():
    """Initialize global buffer."""
    global _GLOBAL_MEMORY_BUFFER
    assert _GLOBAL_MEMORY_BUFFER is None, "global memory buffer is already initialized"
    _GLOBAL_MEMORY_BUFFER = GlobalMemoryBuffer()


def have_global_memory_buffer():
    global _GLOBAL_MEMORY_BUFFER
    return _GLOBAL_MEMORY_BUFFER is not None


def get_global_memory_buffer():
    """Return the global GlobalMemoryBuffer object"""
    global _GLOBAL_MEMORY_BUFFER
    assert _GLOBAL_MEMORY_BUFFER is not None, "global memory buffer is not initialized"
    return _GLOBAL_MEMORY_BUFFER


def destroy_global_memory_buffer():
    """Sets the global memory buffer to None"""
    global _GLOBAL_MEMORY_BUFFER
    _GLOBAL_MEMORY_BUFFER = None
