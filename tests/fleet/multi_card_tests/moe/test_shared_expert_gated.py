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


import functools
import random
import unittest

import numpy as np
import paddle
from paddle.distributed import fleet

# from paddleformers.fleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddle.distributed.fleet.meta_parallel import NoPipelineParallel
from paddle.distributed.fleet.utils.sequence_parallel_utils import (
    register_sequence_parallel_allreduce_hooks,
)

import paddleformers.fleet

# from tests.unit_tests.test_utilities import Utils
import paddleformers.fleet.parallel_state as ps
from paddleformers.fleet.gpt_builders import gpt_builder
from paddleformers.fleet.models.gpt import GPTConfig
from paddleformers.fleet.tensor_parallel.mappings import (
    _gather_along_first_dim,
    _gather_along_last_dim,
)
from paddleformers.fleet.training.initialize import initialize_fleet


def _set_random_seed(
    seed_: int,
    data_parallel_random_init: bool = False,
    te_rng_tracker: bool = False,
    inference_rng_tracker: bool = False,
    use_cudagraphable_rng: bool = False,
):
    """Set random seed for reproducibility."""
    if seed_ is not None and seed_ > 0:
        # Ensure that different pipeline MP stages get different seeds.
        seed = seed_ + (
            100
            * paddleformers.fleet.parallel_state.get_pipeline_model_parallel_rank()
        )
        # Ensure different data parallel ranks get different seeds
        if data_parallel_random_init:
            seed = seed + (
                10 * paddleformers.fleet.parallel_state.get_data_parallel_rank()
            )
        random.seed(seed)
        np.random.seed(seed)
        paddle.manual_seed(seed)

        if (
            paddle.distributed.is_initialized()
            and paddle.cuda.device_count() > 0
        ):
            paddleformers.fleet.tensor_parallel.model_parallel_cuda_manual_seed(
                seed,
                te_rng_tracker,
                inference_rng_tracker,
                use_cudagraphable_rng,
            )
    else:
        raise ValueError(f"Seed ({seed_}) should be a positive integer.")


def cal_sim(a, b):
    return paddle.nn.functional.cosine_similarity(a.flatten(), b.flatten(), 0)


def check_grads(dist_model, serial_model, tp_group):
    serial_grads = {}
    for name, p in serial_model.named_parameters():
        serial_grads[name] = p.grad

    dist_grads = {}
    for name, p in dist_model.named_parameters():
        if "qkv_proj.weight" in name or "up_gate_proj.weight" in name:
            grad = _gather_along_last_dim(p.grad, tp_group)
        elif (
            "o_proj.weight" in name
            or "down_proj.weight" in name
            or "embed_tokens.weight" in name
        ):
            grad = _gather_along_first_dim(p.grad, tp_group)
        else:
            grad = p.grad
        assert (
            paddle.allclose(grad, serial_grads[name], atol=5e-8)
            and cal_sim(grad, serial_grads[name]) > 0.999
        )


def single_device_baseline(seed, batch_size, seq_len, vocab_size, config):
    seed = 46
    random.seed(seed)
    np.random.seed(seed)
    paddle.manual_seed(seed)

    gpt_model = gpt_builder(config, num_stages=1)

    paddle.manual_seed(seed)
    data = paddle.randint(
        low=0, high=vocab_size, shape=(batch_size, seq_len + 1)
    )
    input_ids = data[:, :-1]
    labels = data[:, 1:]
    position_ids = (
        paddle.arange(seq_len, dtype=paddle.int64)
        .unsqueeze(0)
        .expand([batch_size, -1])
    )

    strategy = fleet.DistributedStrategy()
    gpt_pipe_model = NoPipelineParallel(gpt_model, strategy)
    inputs = (
        {
            "input_ids": [input_ids],
            "position_ids": [position_ids],
        },
        [labels],
    )

    loss = gpt_pipe_model.forward_backward_pipeline(inputs)

    return loss, gpt_pipe_model


def run_tp_sp(
    seed,
    batch_size,
    seq_len,
    vocab_size,
    config,
    loss_baseline,
    gpt_model_baseline,
):
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": 4,
        "pp_degree": 1,
        "sharding_degree": 1,
        "sep_degree": 1,
        "cp_degree": 1,
        "ep_degree": 1,
        "moe_sharding_degree": 1,
        "order": [
            "sharding",
            "moe_sharding",
            "pp",
            "sep",
            "cp",
            "dp",
            "ep",
            "mp",
        ],
    }
    initialize_fleet(strategy)

    _set_random_seed(seed)

    gpt_model = gpt_builder(config, num_stages=1)

    register_sequence_parallel_allreduce_hooks(gpt_model, 1, False)

    paddle.manual_seed(seed)
    data = paddle.randint(
        low=0, high=vocab_size, shape=(batch_size, seq_len + 1)
    )
    input_ids = data[:, :-1]
    labels = data[:, 1:]
    position_ids = (
        paddle.arange(seq_len, dtype=paddle.int64)
        .unsqueeze(0)
        .expand([batch_size, -1])
    )

    tp_group = ps.get_tensor_model_parallel_group()

    gpt_pipe_model = NoPipelineParallel(gpt_model, strategy)
    inputs = (
        {
            "input_ids": [input_ids],
            "position_ids": [position_ids],
        },
        [labels],
    )

    loss = gpt_pipe_model.forward_backward_pipeline(inputs)

    is_within_range = lambda loss, loss_baseline: (
        False
        if loss_baseline == 0
        else abs(loss - loss_baseline) / loss_baseline <= 0.015
    )
    assert is_within_range(loss, loss_baseline), (
        f"TP-SP Loss: {loss}, Baseline Loss: {loss_baseline}, Diff: {abs(loss - loss_baseline) / loss_baseline * 100:.2f}%, The difference is out of the tolerance range 1.5%."
    )

    # check_grads(gpt_pipe_model, gpt_model_baseline, tp_group)


class TestTPSP(unittest.TestCase):
    def setUp(self):
        self.seed = 46
        self.batch_size = 2
        self.seq_len = 128
        self.vocab_size = 1024

    def test_tp_sp(self):
        config = GPTConfig(
            moe_expert_fusion=False,
            vocab_size=self.vocab_size,
            max_sequence_length=self.seq_len,
            n_routed_experts=64,
            num_experts_per_tok=2,
            moe_intermediate_size=64,
            n_shared_experts=1,
            moe_shared_expert_gate=True,
            num_hidden_layers=2,
            hidden_size=512,
            num_attention_heads=4,
            intermediate_size=1024,
            normalization="RMSNorm",
            hidden_dropout_prob=0.0,
            attention_dropout=0.0,
            use_cpu_initialization=True,
            parallel_output=True,
            tie_word_embeddings=True,
            position_embedding_type="rope",
            rotary_percent=1.0,
            rotary_base=10000,
            rope_scaling=1.0,
            init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            output_layer_init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            use_qk_norm=True,
        )
        loss, gpt_model = single_device_baseline(
            self.seed, self.batch_size, self.seq_len, self.vocab_size, config
        )

        dist_config = GPTConfig(
            moe_expert_fusion=False,
            vocab_size=self.vocab_size,
            max_sequence_length=self.seq_len,
            n_routed_experts=64,
            num_experts_per_tok=2,
            moe_intermediate_size=64,
            n_shared_experts=1,
            moe_shared_expert_gate=True,
            num_hidden_layers=2,
            hidden_size=512,
            num_attention_heads=4,
            intermediate_size=1024,
            normalization="RMSNorm",
            hidden_dropout_prob=0.0,
            attention_dropout=0.0,
            use_cpu_initialization=True,
            tensor_model_parallel_size=4,
            sequence_parallel=True,
            parallel_output=True,
            tie_word_embeddings=True,
            position_embedding_type="rope",
            rotary_percent=1.0,
            rotary_base=10000,
            rope_scaling=1.0,
            init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            output_layer_init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            use_qk_norm=True,
        )
        run_tp_sp(
            self.seed,
            self.batch_size,
            self.seq_len,
            self.vocab_size,
            dist_config,
            loss,
            gpt_model,
        )


if __name__ == "__main__":
    unittest.main()
