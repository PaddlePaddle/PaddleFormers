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
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import os
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
        )
    ),
)

import paddle
import paddle.distributed as dist
from paddle.distributed import ShardedWeight

import paddleformers.fleet.parallel_state as ps
from paddleformers.fleet.models.backends import LocalSpecProvider
from paddleformers.fleet.tensor_parallel.layers import (
    ColumnParallelLinear,
    Linear,
    RowParallelLinear,
    VocabParallelEmbedding,
    linear_with_frozen_weight,
)
from paddleformers.fleet.tensor_parallel.mappings import (
    gather_from_tensor_model_parallel_region,
    scatter_to_tensor_model_parallel_region,
)
from paddleformers.fleet.tensor_parallel.random import (
    model_parallel_cuda_manual_seed,
)
from paddleformers.fleet.transformer.transformer_config import TransformerConfig
from tests.fleet.multi_card_tests.tensor_parallel.test_utilities import Utils


def test_LinearWithFrozenWeight(tensor_parallel, allreduce_dgrad):
    size_per_partition = int(8 / tensor_parallel)

    # Input is an 8x8 identity matrix.
    input_data = paddle.eye(8).cuda()
    input_data.requires_grad = True

    # Weight is an 8x8 matrix of all ones. If tensor parallelism > 1, the weight is partitioned evenly across GPUs.
    weight = paddle.ones((8, size_per_partition)).cuda()

    # Bias is a vector of length 8 of all zeros. If tensor parallelism > 1, the bias is partitioned evenly across GPUs
    bias = paddle.zeros(size_per_partition).cuda()

    gradient_accumulation_fusion = False
    sequence_parallel = False
    grad_output_buffer = None
    wgrad_deferral_limit = None

    weight.stop_gradient = True
    bias.stop_gradient = True
    output_parallel = linear_with_frozen_weight(
        input_data,
        weight,
        bias,
        gradient_accumulation_fusion,
        allreduce_dgrad,
        sequence_parallel,
        grad_output_buffer,
        wgrad_deferral_limit,
    )
    output = gather_from_tensor_model_parallel_region(
        output_parallel
    )  # no-op if tensor_parallel == 1.
    output.sum().backward()

    expected_output = paddle.ones([8, 8]).cuda()
    expected_grad = 8 * paddle.ones([8, 8]).cuda()

    assert paddle.allclose(output, expected_output)
    assert paddle.allclose(input_data.grad, expected_grad)


def column_parallel_baseline():
    transformer_config = TransformerConfig(
        num_hidden_layers=1,
        hidden_size=12,
        num_attention_heads=4,
        use_cpu_initialization=True,
    )
    paddle.manual_seed(42)

    tp1_group = dist.new_group([dist.get_rank()])
    col_tp1 = ColumnParallelLinear(
        input_size=8,
        output_size=8,
        init_method=transformer_config.init_method,
        bias=True,
        config=transformer_config,
        skip_bias_add=False,
        gather_output=False,
        tp_group=tp1_group,
    )

    # Input is an 8x8 identity matrix.
    input_data = paddle.arange(64).reshape((8, 8)) * 0.1
    input_data.requires_grad = True

    output, _ = col_tp1(input_data)
    output.sum().backward()

    return output, input_data.grad, col_tp1.weight.grad, col_tp1.bias.grad


def test_ColumnParallelLinear(
    tensor_parallel,
    output_baseline,
    input_grad_baseline,
    weight_grad_baseline,
    bias_grad_baseline,
):
    transformer_config = TransformerConfig(
        num_hidden_layers=1,
        hidden_size=12,
        num_attention_heads=4,
        use_cpu_initialization=True,
    )

    paddle.manual_seed(42)
    model_parallel_cuda_manual_seed(42)
    size_per_partition = int(8 / tensor_parallel)
    col_tp4 = ColumnParallelLinear(
        input_size=8,
        output_size=8,
        init_method=transformer_config.init_method,
        bias=True,
        config=transformer_config,
        skip_bias_add=False,
        gather_output=True,
    )

    input_data = paddle.arange(64).reshape((8, 8)) * 0.1
    input_data.requires_grad = True

    output, _ = col_tp4(input_data)
    output.sum().backward()

    rank = ps.get_tensor_model_parallel_rank()
    assert paddle.equal_all(output, output_baseline)
    assert paddle.allclose(input_data.grad, input_grad_baseline)
    assert paddle.allclose(
        col_tp4.weight.grad, weight_grad_baseline[:, rank * 2 : (rank + 1) * 2]
    )
    assert paddle.allclose(
        col_tp4.bias.grad, bias_grad_baseline[rank * 2 : (rank + 1) * 2]
    )

    sharded_dict = col_tp4.sharded_state_dict()
    assert "bias" in sharded_dict
    bias_shard = sharded_dict["bias"]
    assert isinstance(bias_shard, ShardedWeight)
    assert "weight" in sharded_dict
    weight_shard = sharded_dict["weight"]
    assert isinstance(weight_shard, ShardedWeight)

    in_f, out_f = col_tp4.input_size, col_tp4.output_size
    assert weight_shard.global_shape == (in_f, out_f)
    assert weight_shard.local_shape == (in_f, out_f // tensor_parallel)
    assert weight_shard.global_offset == (
        0,
        rank * (out_f // tensor_parallel),
    )
    assert bias_shard.global_shape == (out_f,)
    assert bias_shard.local_shape == (out_f // tensor_parallel,)
    assert bias_shard.global_offset == (rank * (out_f // tensor_parallel),)


def row_parallel_baseline():
    transformer_config = TransformerConfig(
        num_hidden_layers=1,
        hidden_size=12,
        num_attention_heads=4,
        use_cpu_initialization=True,
    )
    paddle.manual_seed(42)

    tp1_group = dist.new_group([dist.get_rank()])
    row_tp1 = RowParallelLinear(
        input_size=8,
        output_size=8,
        init_method=transformer_config.init_method,
        bias=True,
        input_is_parallel=False,
        config=transformer_config,
        skip_bias_add=False,
        tp_group=tp1_group,
    )

    input_data = paddle.arange(64).reshape((8, 8)) * 0.1
    input_data.requires_grad = True

    output, _ = row_tp1(input_data)
    output.sum().backward()

    return output, input_data.grad, row_tp1.weight.grad, row_tp1.bias.grad


def test_RowParallelLinear(
    tensor_parallel,
    output_baseline,
    input_grad_baseline,
    weight_grad_baseline,
    bias_grad_baseline,
):
    transformer_config = TransformerConfig(
        num_hidden_layers=1,
        hidden_size=12,
        num_attention_heads=4,
        use_cpu_initialization=True,
    )

    paddle.manual_seed(42)
    model_parallel_cuda_manual_seed(42)
    size_per_partition = int(8 / tensor_parallel)
    row_tp4 = RowParallelLinear(
        input_size=8,
        output_size=8,
        init_method=transformer_config.init_method,
        bias=True,
        config=transformer_config,
        skip_bias_add=False,
        input_is_parallel=True,
    )

    input_data = paddle.arange(64).reshape((8, 8)) * 0.1
    input_data.requires_grad = True
    rank = ps.get_tensor_model_parallel_rank()
    scattered_input = scatter_to_tensor_model_parallel_region(input_data)

    output, _ = row_tp4(scattered_input)
    output.sum().backward()

    assert paddle.allclose(output, output_baseline, atol=1e-7)
    assert paddle.allclose(input_data.grad, input_grad_baseline)
    assert paddle.allclose(
        row_tp4.weight.grad, weight_grad_baseline[rank * 2 : (rank + 1) * 2, :]
    )
    assert paddle.allclose(row_tp4.bias.grad, bias_grad_baseline)

    sharded_dict = row_tp4.sharded_state_dict()
    assert "bias" in sharded_dict
    bias_shard = sharded_dict["bias"]
    assert isinstance(bias_shard, ShardedWeight)
    assert "weight" in sharded_dict
    weight_shard = sharded_dict["weight"]
    assert isinstance(weight_shard, ShardedWeight)

    in_f, out_f = row_tp4.input_size, row_tp4.output_size
    assert weight_shard.global_shape == (in_f, out_f)
    assert weight_shard.local_shape == (in_f // tensor_parallel, out_f)
    assert weight_shard.global_offset == (
        rank * (in_f // tensor_parallel),
        0,
    )
    assert bias_shard.global_shape == (out_f,)
    assert bias_shard.local_shape == bias_shard.global_shape
    assert bias_shard.global_offset == (0,)


def embedding_baseline():
    transformer_config = TransformerConfig(
        num_hidden_layers=1,
        hidden_size=12,
        num_attention_heads=4,
        use_cpu_initialization=True,
    )
    paddle.manual_seed(42)

    tp1_group = dist.new_group([dist.get_rank()])
    emb_tp1 = VocabParallelEmbedding(
        num_embeddings=16,
        embedding_dim=4,
        init_method=transformer_config.init_method,
        config=transformer_config,
        tp_group=tp1_group,
    )

    input_data = paddle.tensor(
        [[6, 3, 4, 1, 7, 13, 8, 0], [0, 5, 12, 11, 9, 2, 1, 15]]
    )
    input_data.requires_grad = True

    output = emb_tp1(input_data)
    output.sum().backward()

    return output, emb_tp1.weight.grad


def test_VocabParallelEmbedding(
    tensor_parallel, output_baseline, weight_grad_baseline
):
    transformer_config = TransformerConfig(
        num_hidden_layers=1,
        hidden_size=12,
        num_attention_heads=4,
        use_cpu_initialization=True,
    )

    paddle.manual_seed(42)
    model_parallel_cuda_manual_seed(42)
    emb_tp4 = VocabParallelEmbedding(
        num_embeddings=16,
        embedding_dim=4,
        init_method=transformer_config.init_method,
        config=transformer_config,
    )

    input_data = paddle.tensor(
        [[6, 3, 4, 1, 7, 13, 8, 0], [0, 5, 12, 11, 9, 2, 1, 15]]
    )
    input_data.requires_grad = True

    output = emb_tp4(input_data)
    output.sum().backward()

    rank = dist.get_rank()
    assert paddle.equal_all(output, output_baseline)
    assert paddle.allclose(
        emb_tp4.weight.grad, weight_grad_baseline[rank * 4 : (rank + 1) * 4, :]
    )

    sharded_dict = emb_tp4.sharded_state_dict()
    assert "bias" not in sharded_dict
    assert "weight" in sharded_dict
    weight_shard = sharded_dict["weight"]
    assert isinstance(weight_shard, ShardedWeight)
    assert weight_shard.global_shape == (
        emb_tp4.num_embeddings,
        emb_tp4.embedding_dim,
    )
    assert weight_shard.local_shape == (
        emb_tp4.num_embeddings // tensor_parallel,
        emb_tp4.embedding_dim,
    )
    assert weight_shard.global_offset == (
        rank * (emb_tp4.num_embeddings // tensor_parallel),
        0,
    )


# ---------------------------------------------------------------------------
# Tests for Linear (non-TP, weight replicated across TP ranks).
# backend.linear() in gpt_layer_specs.py resolves to this class via
# LocalSpecProvider.linear() -> Linear.
# ---------------------------------------------------------------------------


def _make_linear_config():
    return TransformerConfig(
        num_hidden_layers=1,
        hidden_size=12,
        num_attention_heads=4,
        use_cpu_initialization=True,
    )


def test_Linear_forward_basic():
    """forward(): output shape is correct and backward propagates to input."""
    config = _make_linear_config()
    paddle.manual_seed(0)
    layer = Linear(
        input_size=8,
        output_size=6,
        config=config,
        init_method=config.init_method,
        bias=True,
        skip_bias_add=False,
    )

    input_data = paddle.randn([4, 8])
    input_data.requires_grad = True

    output, output_bias = layer(input_data)

    assert output.shape == [4, 6], f"Expected [4,6], got {output.shape}"
    assert output_bias is None, (
        "skip_bias_add=False should return None as output_bias"
    )

    output.sum().backward()
    assert input_data.grad is not None
    assert input_data.grad.shape == [4, 8]


def test_Linear_skip_bias_add():
    """forward() with skip_bias_add=True returns (output, bias) instead of adding bias."""
    config = _make_linear_config()
    paddle.manual_seed(1)
    layer = Linear(
        input_size=8,
        output_size=6,
        config=config,
        init_method=config.init_method,
        bias=True,
        skip_bias_add=True,
    )

    input_data = paddle.randn([4, 8])
    output, output_bias = layer(input_data)

    assert output.shape == [4, 6]
    assert output_bias is not None, (
        "skip_bias_add=True should return the bias tensor"
    )
    assert output_bias.shape == [6]

    # Verify: output + broadcast(bias) equals the manually computed result.
    expected = paddle.matmul(input_data, layer.weight) + output_bias
    assert paddle.allclose(output + output_bias, expected, atol=1e-5)


def test_Linear_no_bias():
    """forward() with bias=False: layer.bias is None, output_bias is None."""
    config = _make_linear_config()
    paddle.manual_seed(2)
    layer = Linear(
        input_size=8,
        output_size=6,
        config=config,
        init_method=config.init_method,
        bias=False,
        skip_bias_add=False,
    )

    assert layer.bias is None

    input_data = paddle.randn([4, 8])
    output, output_bias = layer(input_data)

    assert output.shape == [4, 6]
    assert output_bias is None


def test_Linear_skip_weight_param_allocation():
    """forward() accepts an externally supplied weight when
    skip_weight_param_allocation=True."""
    config = _make_linear_config()
    paddle.manual_seed(3)
    layer = Linear(
        input_size=8,
        output_size=6,
        config=config,
        init_method=config.init_method,
        bias=False,
        skip_weight_param_allocation=True,
    )

    assert layer.weight is None

    ext_weight = paddle.randn([8, 6])
    input_data = paddle.randn([4, 8])
    output, _ = layer(input_data, weight=ext_weight)

    assert output.shape == [4, 6]
    expected = paddle.matmul(input_data, ext_weight)
    assert paddle.allclose(output, expected, atol=1e-5)


def test_Linear_forward_wrong_weight_shape():
    """forward() raises RuntimeError when supplied weight has wrong shape."""
    config = _make_linear_config()
    paddle.manual_seed(4)
    layer = Linear(
        input_size=8,
        output_size=6,
        config=config,
        init_method=config.init_method,
        bias=False,
        skip_weight_param_allocation=True,
    )

    bad_weight = paddle.randn([8, 99])  # wrong output dim
    input_data = paddle.randn([4, 8])
    try:
        layer(input_data, weight=bad_weight)
        raise AssertionError("Expected RuntimeError was not raised")
    except RuntimeError as e:
        assert "not" in str(e).lower() or "expected" in str(e).lower()


def test_Linear_frozen_weight():
    """forward() switches to linear_with_frozen_weight when weight.stop_gradient=True."""
    config = _make_linear_config()
    paddle.manual_seed(5)
    layer = Linear(
        input_size=8,
        output_size=6,
        config=config,
        init_method=config.init_method,
        bias=True,
        skip_bias_add=False,
    )
    layer.weight.stop_gradient = True

    input_data = paddle.randn([4, 8])
    input_data.requires_grad = True

    output, _ = layer(input_data)
    output.sum().backward()

    # Weight gradient must not be computed (frozen).
    assert layer.weight.grad is None
    # Input gradient must still exist.
    assert input_data.grad is not None


def test_Linear_sharded_state_dict():
    """sharded_state_dict(): weight and bias are present; weight is replicated
    so it should NOT carry per-rank shard offsets (global_offset == (0, 0))."""
    config = _make_linear_config()
    paddle.manual_seed(6)
    layer = Linear(
        input_size=8,
        output_size=6,
        config=config,
        init_method=config.init_method,
        bias=True,
    )

    sharded_dict = layer.sharded_state_dict()

    assert "weight" in sharded_dict
    assert "bias" in sharded_dict

    weight_entry = sharded_dict["weight"]
    bias_entry = sharded_dict["bias"]

    # Weight is replicated: either a plain tensor or a ShardedWeight with no
    # actual sharding (global_offset all-zeros, local == global).
    if isinstance(weight_entry, ShardedWeight):
        assert weight_entry.global_shape == (8, 6)
        assert weight_entry.local_shape == (8, 6)
        assert weight_entry.global_offset == (0, 0)
    else:
        assert list(weight_entry.shape) == [8, 6]

    if isinstance(bias_entry, ShardedWeight):
        assert bias_entry.global_shape == (6,)
        assert bias_entry.local_shape == (6,)
        assert bias_entry.global_offset == (0,)
    else:
        assert list(bias_entry.shape) == [6]


def test_Linear_extra_state():
    """get_extra_state() returns None; set_extra_state() is a no-op."""
    config = _make_linear_config()
    layer = Linear(
        input_size=8,
        output_size=6,
        config=config,
        init_method=config.init_method,
    )

    assert layer.get_extra_state() is None
    # set_extra_state must not raise.
    layer.set_extra_state({"some": "state"})
    layer.set_extra_state(None)


def test_Linear_repr():
    """__repr__() contains the expected fields."""
    config = _make_linear_config()
    layer = Linear(
        input_size=8,
        output_size=6,
        config=config,
        init_method=config.init_method,
        bias=True,
    )

    r = repr(layer)
    assert "Linear" in r
    assert "8" in r  # in_features
    assert "6" in r  # out_features
    assert "bias=True" in r  # bias is enabled
    assert "TP=1" in r


def test_Linear_via_backend_linear():
    """Verify that LocalSpecProvider().linear() returns the Linear class and
    that instances built from it behave identically to directly-constructed ones.

    This mirrors how gpt_layer_specs.py uses backend.linear() for
    q_a_proj / kv_a_proj_with_mqa in the MLA attention path."""

    backend = LocalSpecProvider()
    LinearCls = backend.linear()
    assert LinearCls is Linear, (
        f"backend.linear() should return Linear, got {LinearCls}"
    )

    config = _make_linear_config()
    paddle.manual_seed(7)
    layer = LinearCls(
        input_size=16,
        output_size=8,
        config=config,
        init_method=config.init_method,
        bias=True,
        skip_bias_add=False,
    )

    # Weight should be replicated (not distributed).
    assert layer.weight.allreduce is True
    assert layer.weight.is_distributed is False

    input_data = paddle.randn([3, 16])
    input_data.requires_grad = True
    output, output_bias = layer(input_data)

    assert output.shape == [3, 8]
    assert output_bias is None

    output.sum().backward()
    assert input_data.grad is not None
    assert layer.weight.grad is not None
    assert layer.weight.grad.shape == [16, 8]


if __name__ == "__main__":
    tensor_parallel = 4
    Utils.initialize_model_parallel(tensor_parallel, 1)
    test_LinearWithFrozenWeight(4, True)
    output_tp1, input_grad_tp1, weight_grad_tp1, bias_grad_tp1 = (
        column_parallel_baseline()
    )
    test_ColumnParallelLinear(
        tensor_parallel,
        output_tp1,
        input_grad_tp1,
        weight_grad_tp1,
        bias_grad_tp1,
    )
    output_tp1, input_grad_tp1, weight_grad_tp1, bias_grad_tp1 = (
        row_parallel_baseline()
    )
    test_RowParallelLinear(
        tensor_parallel,
        output_tp1,
        input_grad_tp1,
        weight_grad_tp1,
        bias_grad_tp1,
    )
    output_tp1, weight_grad_tp1 = embedding_baseline()
    test_VocabParallelEmbedding(4, output_tp1, weight_grad_tp1)
    test_Linear_forward_basic()
    test_Linear_skip_bias_add()
    test_Linear_no_bias()
    test_Linear_skip_weight_param_allocation()
    test_Linear_forward_wrong_weight_shape()
    test_Linear_frozen_weight()
    test_Linear_sharded_state_dict()
    test_Linear_extra_state()
    test_Linear_repr()
    test_Linear_via_backend_linear()

    # Synchronize all ranks and destroy process groups to ensure clean NCCL
    # shutdown. Without this, concurrent launches sharing the same GPUs may
    # cause SIGSEGV during process exit due to NCCL communicator teardown
    # races.
    dist.barrier()
    dist.destroy_process_group()
