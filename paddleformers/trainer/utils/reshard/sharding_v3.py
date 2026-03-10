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

"""
Reshard utilities for DygraphShardingOptimizerV3.

V3 uses a hybrid sharding scheme:
  - 1D params (bias, embedding, etc.): element-wise reduce-scatter, same as V2.
    Master weights are 1D shards; restore needs to all-gather and merge them.
  - 2D Muon params (linear weights): row-partitioned across sharding ranks.
    Each rank holds a contiguous block of rows; no cross-rank merge is needed.
  - 2D/3D MoE expert params: similar to 2D Muon params.

The `restore` function here handles both categories, unlike `sharding_v2.restore`
which assumes all tensors are 1D flat shards.
"""

import numpy as np
import paddle


def _get_v3_1d_param_names(optimizer):
    """
    Return the set of *original* parameter names that are 1D (AdamW,
    reduce-scattered) in the V3 optimizer.  These are the params in
    `optimizer._params_1d`.  All other params are 2D+ (Muon, row-partitioned).
    """
    try:
        from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer.dygraph_sharding_optimizer_v3 import (
            DygraphShardingOptimizerV3,
        )

        from ....transformers.model_utils import unwrap_optimizer

        v3_opt = unwrap_optimizer(optimizer, DygraphShardingOptimizerV3)
        if v3_opt is None:
            return set()
        return {p.name for p in v3_opt._params_1d}
    except Exception:
        return set()


def merge_tensors_1d(k, tensor_list, shape):
    """
    Merge 1D shard tensors back into a full-shape tensor.
    Identical to sharding_v2.merge_tensors but only called for 1D params.
    """
    assert len(tensor_list) > 0
    if len(tensor_list) == 1:
        t = tensor_list[0]
    else:
        assert len(tensor_list[0].shape) == 1, f"Expected 1D shard for 1D param {k}, got shape {tensor_list[0].shape}"
        t = paddle.cat(x=tensor_list, axis=0)
    tensor_size = np.prod(shape)
    padded_size = t._numel()
    assert padded_size >= tensor_size, f"{k}: padded_size {padded_size} < tensor_size {tensor_size}"
    t = t._slice(0, tensor_size)
    t.get_tensor()._set_dims(shape)
    return t


def merge_tensors_nd(k, tensor_list, shape):
    """
    Merge ND (2D/3D) Muon-param master weight shards.

    V3 row-partitions 2D weights: each rank owns a contiguous row slice.
    The tensor_list is sorted by rank (ascending) from even_distribute, so
    concatenating on axis=0 reconstructs the full param.  After concat, trim
    to `shape` elements and reshape.
    """
    assert len(tensor_list) > 0
    if len(tensor_list) == 1:
        t = tensor_list[0]
        # Flatten to 1D so _slice / _set_dims work uniformly.
        if len(t.shape) > 1:
            t = t.reshape([-1])
    else:
        # Each shard may be ND; flatten before concat.
        flat_list = [e.reshape([-1]) if len(e.shape) > 1 else e for e in tensor_list]
        t = paddle.cat(x=flat_list, axis=0)
    tensor_size = np.prod(shape)
    padded_size = t._numel()
    assert padded_size >= tensor_size, f"{k}: padded_size {padded_size} < tensor_size {tensor_size}"
    t = t._slice(0, tensor_size)
    t.get_tensor()._set_dims(shape)
    return t


def is_beta(opt_name):
    return "beta" in opt_name


def restore(node_model_state, model, optimizer):
    """
    Restore master weights from V3's mixed 1D/2D sharding layout.

    For 1D params (reduce-scattered): identical to sharding_v2.restore —
    concatenate shards from all ranks, trim padding, reshape to full shape.

    For 2D/3D Muon params (row-partitioned): concatenate row blocks from all
    ranks (or just use the single piece), trim to full numel, reshape.
    """
    # Evenly distribute params across ranks before merging.
    node_model_state.even_distribute()

    param_shapes = {k: v.shape for (k, v) in model.state_dict().items()}
    params_1d_names = _get_v3_1d_param_names(optimizer)

    def merge_func(k, v):
        structure_name = k[0]
        opt_name = k[-1]
        assert structure_name in param_shapes, f"structure_name {structure_name!r} not found in model.state_dict()"
        tensor_list = [e[1] for e in v]
        # beta accumulators are scalars — just take one copy.
        if is_beta(opt_name):
            return tensor_list[0]
        shape = param_shapes[structure_name]
        # k = (structure_name, static_name, opt_name) for master_weights
        # static_name is the raw tensor/param name (index 1).
        static_name = k[1] if len(k) > 1 else ""
        if static_name in params_1d_names:
            return merge_tensors_1d(k, tensor_list, shape)
        else:
            return merge_tensors_nd(k, tensor_list, shape)

    node_model_state.collapse_key().merge_items(merge_func)
    return node_model_state
