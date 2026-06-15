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
import os
import random
import sys

import numpy as np
import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import NoPipelineParallel

import paddleformers.fleet
from paddleformers.fleet.gpt_builders import gpt_builder
from paddleformers.fleet.models.gpt import GPTConfig
from paddleformers.fleet.training.initialize import initialize_fleet

REPO_FLAG = os.getenv("repo_flag")
SKIP_TESTS = REPO_FLAG != "paddleformers.fleet"


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
        seed = seed_ + (100 * paddleformers.fleet.parallel_state.get_pipeline_model_parallel_rank())
        # Ensure different data parallel ranks get different seeds
        if data_parallel_random_init:
            seed = seed + (10 * paddleformers.fleet.parallel_state.get_data_parallel_rank())
        random.seed(seed)
        np.random.seed(seed)
        paddle.manual_seed(seed)

        if paddle.distributed.is_initialized() and paddle.cuda.device_count() > 0:
            paddleformers.fleet.tensor_parallel.model_parallel_cuda_manual_seed(
                seed,
                te_rng_tracker,
                inference_rng_tracker,
                use_cudagraphable_rng,
            )
    else:
        raise ValueError(f"Seed ({seed_}) should be a positive integer.")


def _set_rng_flag(
    FLAGS_deterministic_rng: bool = False,
    FLAGS_deterministic_rng_grid: int = 624,
):
    """Set rng flag for weight initialization"""
    paddle.set_flags(
        {
            "FLAGS_deterministic_rng": FLAGS_deterministic_rng,
            "FLAGS_deterministic_rng_grid": FLAGS_deterministic_rng_grid,
        }
    )


def run_cp(seed, batch_size, seq_len, vocab_size, config):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": 1,
        "pp_degree": 1,
        "sharding_degree": 8,
        "sep_degree": 1,
        "cp_degree": 8,
        "ep_degree": 8,
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
    _set_rng_flag(FLAGS_deterministic_rng=True)

    gpt_model = gpt_builder(config, num_stages=1)

    paddle.manual_seed(seed)
    data = paddle.randint(low=0, high=vocab_size, shape=(batch_size, seq_len + 1)).cuda()
    input_ids = data[:, :-1]
    labels = data[:, 1:]
    position_ids = paddle.arange(seq_len, dtype=paddle.int64).unsqueeze(0).expand([batch_size, -1]).cuda()

    gpt_pipe_model = NoPipelineParallel(gpt_model, strategy)
    inputs = (
        {
            "input_ids": [input_ids],
            "position_ids": [position_ids],
        },
        [labels],
    )

    gpt_pipe_model = paddle.amp.decorate(models=gpt_pipe_model, level="O2", dtype="bfloat16")
    with paddle.amp.auto_cast(enable=True, dtype="bfloat16"):
        loss = gpt_pipe_model.forward_backward_pipeline(inputs)
        loss.backward()

    print(f"actual loss: {loss.item()}")
    loss_baseline = 7.212946
    np.testing.assert_allclose(np.array(loss), np.array(loss_baseline), rtol=1e-6, atol=1e-8)


if __name__ == "__main__":
    if SKIP_TESTS:
        print(f"Skipping tests: repo_flag={REPO_FLAG} (not 'paddleformers.fleet')")
        sys.exit(0)
    seed = 46
    batch_size = 1
    seq_len = 128
    vocab_size = 1024
    paddle.set_default_dtype("bfloat16")

    dist_config = GPTConfig(
        vocab_size=vocab_size,
        max_sequence_length=seq_len,
        num_hidden_layers=2,
        hidden_size=512,
        num_attention_heads=4,
        intermediate_size=1024,
        normalization="RMSNorm",
        apply_rope_fusion=True,
        hidden_dropout_prob=0.0,
        attention_dropout=0.0,
        use_cpu_initialization=False,
        context_parallel_size=8,
        sequence_parallel=False,
        parallel_output=True,
        tie_word_embeddings=True,
        position_embedding_type="rope",
        rotary_percent=1.0,
        rotary_base=10000,
        rope_scaling=1.0,
        bf16=True,
        autocast_dtype=paddle.bfloat16,
        params_dtype=paddle.bfloat16,
        init_method=functools.partial(paddle.nn.init.xavier_uniform_, gain=1.0),
        output_layer_init_method=functools.partial(paddle.nn.init.xavier_uniform_, gain=1.0),
    )
    run_cp(seed, batch_size, seq_len, vocab_size, dist_config)
