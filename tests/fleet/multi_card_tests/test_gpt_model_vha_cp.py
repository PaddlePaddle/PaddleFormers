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

"""
Multi-card test for VHA (SelfAttentionVHA) + Context Parallel (CP).

Exercises:
- SelfAttentionVHA: init, _apply_vha_premix, _apply_vha_postmix, _post_core_attention_hook, _get_qkv_vha, backward_dw
- DotProductAttention.forward CP path: expand_attn_mask_startend_row_indices_for_cp with experimental_dataflow (shape[-1]==2 pass), flashmask_attention_cp dispatch
- SelfAttentionVHASublayersSpec
- VHA layer spec in gpt_layer_specs.py

Run with:
    export repo_flag=paddleformers.fleet
    python -m paddle.distributed.launch --gpus=0,1,2,3,4,5,6,7 \
        tests/fleet/multi_card_tests/test_gpt_model_vha_cp.py
"""

import functools
import os
import random
import sys

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import NoPipelineParallel

import paddleformers.fleet
from paddleformers.fleet.gpt_builders import gpt_builder
from paddleformers.fleet.models.gpt import GPTConfig
from paddleformers.fleet.training.initialize import initialize_fleet

REPO_FLAG = os.getenv("repo_flag")
SKIP_TESTS = REPO_FLAG != "paddleformers.fleet"


def _set_random_seed(seed_):
    seed = seed_ + (
        100
        * paddleformers.fleet.parallel_state.get_pipeline_model_parallel_rank()
    )
    random.seed(seed)
    np.random.seed(seed)
    paddle.manual_seed(seed)
    if paddle.distributed.is_initialized() and paddle.cuda.device_count() > 0:
        paddleformers.fleet.tensor_parallel.model_parallel_cuda_manual_seed(
            seed
        )


def run_vha_cp_e2e():
    cp_degree = 8
    seed = 42
    batch_size = 1
    seq_len = 64
    vocab_size = 512

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

    config = GPTConfig(
        vocab_size=vocab_size,
        max_sequence_length=seq_len,
        num_hidden_layers=2,
        hidden_size=256,
        num_attention_heads=8,
        num_key_value_heads=2,
        intermediate_size=512,
        # VHA config
        use_vha_attention=True,
        vha_q_lora_rank=32,
        vha_postmix_rank=4,
        multi_latent_attention=False,
        # MoE config
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_shared_experts=1,
        scoring_func="sigmoid",
        topk_method="noaux_tc",
        moe_intermediate_size=128,
        moe_layer_freq=1,
        n_group=1,
        topk_group=1,
        router_aux_loss_coef=0.01,
        router_z_loss_coef=0.01,
        routed_scaling_factor=1.0,
        moe_expert_fusion=False,
        moe_token_dispatcher_type="deepep",
        gated_linear_unit=True,
        hidden_act=F.silu,
        # MTP config
        num_nextn_predict_layers=1,
        mtp_loss_scaling_factor=0.3,
        use_dense_mtp=True,
        # CP + experimental_dataflow
        context_parallel_size=8,
        expert_model_parallel_size=8,
        experimental_dataflow=True,
        gpt_model_use_experimental_version=False,
        # General
        normalization="RMSNorm",
        rms_norm_eps=1e-6,
        apply_rope_fusion=False,
        hidden_dropout_prob=0.0,
        attention_dropout=0.0,
        use_cpu_initialization=False,
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
        output_layer_init_method=functools.partial(
            paddle.nn.init.xavier_uniform_, gain=1.0
        ),
    )

    print(f"[Rank {paddle.distributed.get_rank()}] Building VHA+CP model...")
    gpt_model = gpt_builder(config, num_stages=1)

    # Prepare data
    # MTP: input_ids length = seq_len + num_nextn_predict_layers
    num_nextn = config.num_nextn_predict_layers
    total_seq = seq_len + num_nextn
    paddle.manual_seed(seed)
    data = paddle.randint(
        low=1, high=vocab_size, shape=(batch_size, total_seq + 1)
    ).cuda()
    input_ids = data[:, :-1]  # [batch, total_seq]
    labels = data[:, 1:]  # [batch, total_seq]

    # attn_mask_startend_row_indices with shape[-1]=2 for experimental_dataflow CP pass-through
    start_indices = paddle.full(
        [batch_size, 1, seq_len, 1], fill_value=seq_len, dtype=paddle.int32
    )
    end_indices = (
        paddle.arange(seq_len, dtype=paddle.int32)
        .reshape([1, 1, seq_len, 1])
        .expand([batch_size, 1, seq_len, 1])
    )
    attn_mask_startend_row_indices = paddle.concat(
        [start_indices, end_indices], axis=-1
    ).cuda()

    # MTP masks
    mtp_hidden_inputs_mask_all = paddle.ones(
        [batch_size, num_nextn, seq_len], dtype=paddle.int32
    ).cuda()
    mtp_startend_row_indices_all = paddle.full(
        [batch_size, num_nextn, seq_len, 1],
        fill_value=seq_len,
        dtype=paddle.int32,
    ).cuda()

    gpt_pipe_model = NoPipelineParallel(gpt_model, strategy)
    inputs = (
        {
            "input_ids": [input_ids],
            "labels": [labels],
            "attn_mask_startend_row_indices": [attn_mask_startend_row_indices],
            "mtp_startend_row_indices_all": [mtp_startend_row_indices_all],
            "mtp_hidden_inputs_mask_all": [mtp_hidden_inputs_mask_all],
        },
        [labels],
    )

    loss = gpt_pipe_model.forward_backward_pipeline(inputs)

    rank = paddle.distributed.get_rank()
    print(f"[Rank {rank}] Loss: {loss.item()}")

    assert paddle.isfinite(loss).item(), f"Loss is not finite: {loss.item()}"
    assert loss.item() > 0, f"Loss should be positive: {loss.item()}"

    print(f"actual loss: {loss.item()}")
    loss_baseline = 8.556518
    np.testing.assert_allclose(
        np.array(loss), np.array(loss_baseline), rtol=1e-6, atol=1e-8
    )

    # Verify VHA parameter gradients exist and are finite
    vha_param_names = ["premix_weight", "postmix_U", "postmix_V"]
    for name, param in gpt_model.named_parameters():
        if any(vn in name for vn in vha_param_names):
            assert param.grad is not None, f"VHA param {name} has no gradient"
            assert paddle.all(paddle.isfinite(param.grad)).item(), (
                f"VHA param {name} has non-finite gradient"
            )
            print(
                f"[Rank {rank}] VHA param {name}: grad norm = {paddle.norm(param.grad).item():.6f}"
            )

    print(f"[Rank {rank}] VHA+CP test PASSED")


if __name__ == "__main__":
    if SKIP_TESTS:
        print(
            f"Skipping tests: repo_flag={REPO_FLAG} (not 'paddleformers.fleet')"
        )
        sys.exit(0)
    paddle.set_default_dtype("bfloat16")
    run_vha_cp_e2e()
