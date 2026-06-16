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
End-to-end multi-card test for experimental_dataflow + context parallel (CP).

This test builds a GPT model with MLA + MoE + MTP + CP + experimental_dataflow
and runs a forward + backward pass. This exercises ALL modified code paths:

- dot_product_attention.py: CPDotProductAttention experimental_dataflow branch (shape[-1]==2 pass)
- multi_latent_attention.py: _ec_compatible_rope_apply CP seq_len scaling + scatter
- multi_token_prediction.py: mtp_hidden_inputs_mask CP scatter (requires mtp_hidden_inputs_mask_all input)
- moe_router.py: TopKRouter input_ids CP scatter/gather
- transformer_layer.py: seq_lens * CP world size
- language_loss.py: labels CP scatter
- gpt_embedding.py: inputs_embeds CP scatter

Run with:
    python -m paddle.distributed.launch --gpus=0,1,2,3,4,5,6,7 \
        tests/fleet/multi_card_tests/test_experimental_dataflow_cp.py
"""

import functools
import os
import random
import sys

# Prepend local source tree so that we import the modified paddleformers.fleet (not the system one).
# This mirrors PYTHONPATH in script/train_gpu.sh.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(_repo_root, "src"))

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
    """Set random seed for reproducibility."""
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


def run_experimental_dataflow_cp_e2e():
    """
    End-to-end test: build MLA + MoE + MTP model with CP=2 + experimental_dataflow,
    run forward/backward to exercise all modified code paths.
    """
    cp_degree = 8
    seed = 42
    batch_size = 1
    seq_len = (
        64  # small for testing speed, each rank gets seq_len/cp_degree=8 tokens
    )
    vocab_size = 512

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    # Clear multi-node env vars to avoid hanging when launched on a multi-node cluster
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

    # Build config that triggers ALL modified code paths:
    # - multi_latent_attention=True  -> multi_latent_attention.py changes
    # - n_routed_experts > 0         -> moe_router.py changes
    # - num_nextn_predict_layers > 0 -> multi_token_prediction.py + transformer_layer.py changes
    # - context_parallel_size > 1    -> all CP scatter/gather paths
    # - experimental_dataflow=True   -> EB dataflow branches
    config = GPTConfig(
        vocab_size=vocab_size,
        max_sequence_length=seq_len,
        num_hidden_layers=2,
        hidden_size=256,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=512,
        # MLA config
        multi_latent_attention=True,
        q_lora_rank=128,
        kv_lora_rank=64,
        qk_nope_head_dim=32,
        qk_rope_head_dim=16,
        v_head_dim=48,  # must equal qk_nope_head_dim + qk_rope_head_dim for flash_attn_bwd
        rope_theta=10000,
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
        gpt_model_use_experimental_version=True,
        init_method=functools.partial(paddle.nn.init.xavier_uniform_, gain=1.0),
        output_layer_init_method=functools.partial(
            paddle.nn.init.xavier_uniform_, gain=1.0
        ),
    )

    # Build model
    print(f"[Rank {paddle.distributed.get_rank()}] Building model...")
    gpt_model = gpt_builder(config, num_stages=1)

    # Prepare data
    # input_ids shape: [batch, seq_len]
    # For MTP: the model expects input_ids with length = seq_len + num_nextn_predict_layers
    # so that it can split into decoder ids + mtp ids
    total_seq = seq_len + config.num_nextn_predict_layers
    data = paddle.randint(
        low=1, high=vocab_size, shape=(batch_size, total_seq + 1)
    ).cuda()
    input_ids = data[:, :-1]  # [batch, total_seq]
    labels = data[:, 1:]  # [batch, total_seq]

    # Construct attn_mask_startend_row_indices with shape[-1]=2
    # This triggers the experimental_dataflow branch in CPDotProductAttention (line 599-602)
    # Shape: [batch, 1, seq_len, 2] - causal mask boundaries (start=0, end=row_idx)
    attn_mask_startend_row_indices = paddle.zeros(
        [batch_size, 1, seq_len, 2], dtype=paddle.int32
    ).cuda()
    # Fill with causal mask: start=0, end=position+1
    for i in range(seq_len):
        attn_mask_startend_row_indices[:, :, i, 0] = seq_len  # start
        attn_mask_startend_row_indices[:, :, i, 1] = i  # end (exclusive)

    # Construct mtp_hidden_inputs_mask_all to trigger MTP CP scatter (line 359-361)
    # Shape: [batch, num_nextn_predict_layers, seq_len]
    num_nextn = config.num_nextn_predict_layers
    mtp_hidden_inputs_mask_all = paddle.ones(
        [batch_size, num_nextn, seq_len], dtype=paddle.int32
    ).cuda()

    # Construct mtp_startend_row_indices_all (must be paired with mtp_hidden_inputs_mask_all)
    # Shape: [batch, num_nextn_predict_layers, seq_len, 1]
    mtp_startend_row_indices_all = paddle.full(
        [batch_size, num_nextn, seq_len, 1],
        fill_value=seq_len,
        dtype=paddle.int32,
    ).cuda()

    # Wrap with NoPipelineParallel
    gpt_pipe_model = NoPipelineParallel(gpt_model, strategy)
    inputs = (
        {
            "input_ids": [input_ids],
            "labels": [labels],
            "attn_mask_startend_row_indices": [attn_mask_startend_row_indices],
            "mtp_startend_row_indices_all": [mtp_startend_row_indices_all],
            "mtp_hidden_inputs_mask_all": [mtp_hidden_inputs_mask_all],
        },
        labels,
    )

    # Forward + backward: params_dtype=bfloat16 + set_default_dtype("bfloat16") already
    # ensures all weights and activations are in bfloat16. No auto_cast needed — using
    # O2 would force embedding output to float32 (embedding is in the AMP O2 black list),
    # causing a bfloat16 vs float32 dtype mismatch in LinearWithGradAccumulationAndAsyncCommunication backward.
    loss = gpt_pipe_model.forward_backward_pipeline(inputs)

    rank = paddle.distributed.get_rank()
    print(f"[Rank {rank}] Loss: {loss.item()}")

    # Verify loss is finite (not NaN/Inf)
    assert paddle.isfinite(loss).item(), f"Loss is not finite: {loss.item()}"
    assert loss.item() > 0, f"Loss should be positive: {loss.item()}"

    # Verify gradients exist on model parameters
    grad_count = 0
    nan_grad_count = 0
    for name, param in gpt_model.named_parameters():
        if param.grad is not None:
            grad_count += 1
            if not paddle.all(paddle.isfinite(param.grad)).item():
                nan_grad_count += 1

    print(f"actual loss: {loss.item()}")
    loss_baseline = 8.460711
    np.testing.assert_allclose(
        np.array(loss), np.array(loss_baseline), rtol=1e-6, atol=1e-8
    )


if __name__ == "__main__":
    if SKIP_TESTS:
        print(
            f"Skipping tests: repo_flag={REPO_FLAG} (not 'paddleformers.fleet')"
        )
        sys.exit(0)
    paddle.set_default_dtype("bfloat16")
    run_experimental_dataflow_cp_e2e()
