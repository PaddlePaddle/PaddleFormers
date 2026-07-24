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

"""Multi-card TP+SP precision alignment tests for Qwen3.5 VL model (dense & MoE).

Tests that the Qwen3.5 model produces identical (within tolerance) loss values
when running with tensor-parallel + sequence-parallel (TP=4, SP=True) compared
to a single-device baseline.

Two model variants are tested:
  1. Dense -- all layers are dense MLP (full_attention + gated_attention).
  2. MoE   -- routed experts replace the dense MLP.

Run with 4 GPUs:
    python -m paddle.distributed.launch --gpus="0,1,2,3" test_qwen3_5_model.py
"""

import functools
import os
import random
import sys
import unittest

import numpy as np
import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet.utils.sequence_parallel_utils import (
    register_sequence_parallel_allreduce_hooks,
)

import paddleformers.fleet
from paddleformers.fleet.models.gpt import GPTConfig

# Add single_card_tests to path so we can reuse test helpers
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "single_card_tests",
        "model",
    ),
)
from paddle.distributed.fleet.meta_parallel import (
    NoPipelineParallel,
    build_spec_layer,
)
from test_qwen3_5_vision_model import (
    Qwen3_5Model,
    Qwen3_5VisionProvider,
    get_qwen3_5_language_spec,
)

from paddleformers.fleet.tensor_parallel.mappings import (
    _gather_along_first_dim,
    _gather_along_last_dim,
)
from paddleformers.fleet.training.initialize import initialize_fleet

# ======================================================================
# Shared test dimensions (small for fast unit testing)
# ======================================================================

# Vision encoder dims
VIS_HIDDEN_SIZE = 64
VIS_NUM_HEADS = 4
VIS_HEAD_DIM = VIS_HIDDEN_SIZE // VIS_NUM_HEADS  # 16
VIS_NUM_LAYERS = 2
VIS_OUT_HIDDEN_SIZE = 96
VIS_INTERMEDIATE_SIZE = 128
PATCH_SIZE = 16
SPATIAL_MERGE_SIZE = 2
TEMPORAL_PATCH_SIZE = 2
IN_CHANNELS = 3
NUM_POSITION_EMBEDDINGS = 256  # 16 * 16

# Derived image dims
IMAGE_H = 64
IMAGE_W = 64
GRID_T = 1
GRID_H = IMAGE_H // PATCH_SIZE  # 4
GRID_W = IMAGE_W // PATCH_SIZE  # 4
SEQ_LEN_VIS = GRID_T * GRID_H * GRID_W  # 16
MERGED_TOKENS = SEQ_LEN_VIS // (SPATIAL_MERGE_SIZE**2)  # 4

# Language model dims
LM_HIDDEN_SIZE = VIS_OUT_HIDDEN_SIZE  # 96 — must match vision output
LM_NUM_HEADS = 4
LM_HEAD_DIM = LM_HIDDEN_SIZE // LM_NUM_HEADS  # 24
LM_NUM_LAYERS = 2
LM_INTERMEDIATE_SIZE = 128
LM_VOCAB_SIZE = 256

# VL composite dims
IMAGE_TOKEN_ID = 200
VIDEO_TOKEN_ID = 201
TEXT_BEFORE = 5
TEXT_AFTER = 3
NUM_IMAGE_TOKENS = MERGED_TOKENS  # 4
VL_SEQ_LEN = TEXT_BEFORE + NUM_IMAGE_TOKENS + TEXT_AFTER  # 12

# MRoPE section: temporal, height, width channel dims  (must sum to head_dim // 2)
MROPE_SECTION = [4, 4, 4]


# ======================================================================
# Utilities
# ======================================================================


def _set_random_seed(
    seed_: int,
    data_parallel_random_init: bool = False,
    te_rng_tracker: bool = False,
    inference_rng_tracker: bool = False,
    use_cudagraphable_rng: bool = False,
):
    """Set random seed for reproducibility."""
    if seed_ is not None and seed_ > 0:
        seed = seed_ + (
            100
            * paddleformers.fleet.parallel_state.get_pipeline_model_parallel_rank()
        )
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
        ), f"Gradient mismatch for {name}"


def is_within_range(loss, loss_baseline, tol=0.015):
    """Check that loss is within tol% of baseline."""
    if loss_baseline == 0:
        return False
    return abs(loss - loss_baseline) / abs(loss_baseline) <= tol


# ======================================================================
# Vision config builder (reusable)
# ======================================================================


def _make_vision_config():
    return Qwen3_5VisionProvider(
        num_hidden_layers=VIS_NUM_LAYERS,
        hidden_size=VIS_HIDDEN_SIZE,
        num_attention_heads=VIS_NUM_HEADS,
        head_dim=VIS_HEAD_DIM,
        out_hidden_size=VIS_OUT_HIDDEN_SIZE,
        intermediate_size=VIS_INTERMEDIATE_SIZE,
        patch_size=PATCH_SIZE,
        spatial_merge_size=SPATIAL_MERGE_SIZE,
        temporal_patch_size=TEMPORAL_PATCH_SIZE,
        in_channels=IN_CHANNELS,
        num_position_embeddings=NUM_POSITION_EMBEDDINGS,
        hidden_dropout_prob=0.0,
        attention_dropout=0.0,
        normalization="LayerNorm",
        use_qk_norm=False,
        gated_linear_unit=False,
        apply_rope_fusion=False,
        use_cpu_initialization=True,
    )


# ======================================================================
# Language config builder
# ======================================================================

LAYER_TYPE_MAP = {
    "full_attention": "self_attention",
    "linear_attention": "gated_delta_net",
}


def _make_language_config(
    *,
    tensor_model_parallel_size: int = 1,
    sequence_parallel: bool = False,
    use_moe: bool = False,
):
    """Create a GPTConfig for the Qwen3.5 language model.

    When ``use_moe=True`` the dense MLP is replaced with a routed-expert MoE
    layer.  The ``moe_layer_freq`` is set to 1 (every layer is MoE).
    """
    base_kwargs = {
        "num_hidden_layers": LM_NUM_LAYERS,
        "hidden_size": LM_HIDDEN_SIZE,
        "num_attention_heads": LM_NUM_HEADS,
        "head_dim": LM_HEAD_DIM,
        "intermediate_size": LM_INTERMEDIATE_SIZE,
        "hidden_dropout_prob": 0.0,
        "attention_dropout": 0.0,
        "normalization": "LayerNorm",
        "gated_linear_unit": False,
        "apply_rope_fusion": False,
        "vocab_size": LM_VOCAB_SIZE,
        "max_sequence_length": 1024,
        "position_embedding_type": "rope",
        "rotary_percent": 1.0,
        "rotary_base": 10000,
        "rope_scaling": False,
        "parallel_output": False,
        "tie_word_embeddings": False,
        "layer_types": ["full_attention", "linear_attention"],
        "gated_attention": True,
        "use_cpu_initialization": True,
        "tensor_model_parallel_size": tensor_model_parallel_size,
        "sequence_parallel": sequence_parallel,
        "init_method": functools.partial(
            paddle.nn.init.xavier_uniform_, gain=1.0
        ),
        "output_layer_init_method": functools.partial(
            paddle.nn.init.xavier_uniform_, gain=1.0
        ),
    }

    if use_moe:
        base_kwargs.update(
            n_routed_experts=8,
            num_experts_per_tok=2,
            moe_intermediate_size=LM_INTERMEDIATE_SIZE,
            n_shared_experts=1,
            moe_shared_expert_gate=False,
            moe_layer_freq=1,  # every layer is MoE
            moe_token_dispatcher_type="alltoall",
            moe_expert_fusion=False,
        )

    config = GPTConfig(**base_kwargs)
    # mrope_section is accessed by get_gpt_spec when position_embedding_type == "mrope"
    config.mrope_section = MROPE_SECTION
    return config


# ======================================================================
# Build Qwen3.5 VL model
# ======================================================================


def _build_qwen3_5_model(language_config, strategy):
    """Build a Qwen3.5 VL model with real vision encoder + language decoder."""
    vision_config = _make_vision_config()
    vision_model = vision_config.provide()

    language_spec = get_qwen3_5_language_spec(config=language_config)
    language_model = build_spec_layer(
        language_spec,
        seg_method="layer:TransformerLayer|EmptyLayer",
        num_stages=1,
    )

    model = Qwen3_5Model(
        config=language_config,
        vision_model=NoPipelineParallel(vision_model, strategy),
        language_model=NoPipelineParallel(language_model, strategy),
        spatial_merge_size=SPATIAL_MERGE_SIZE,
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
    )
    return model


# ======================================================================
# Prepare multimodal input data
# ======================================================================


def _make_input_data(batch_size=1):
    """Construct multimodal input: text + image tokens + pixel values."""
    input_ids = paddle.randint(0, 100, [batch_size, VL_SEQ_LEN])
    input_ids[0, TEXT_BEFORE : TEXT_BEFORE + NUM_IMAGE_TOKENS] = IMAGE_TOKEN_ID

    mm_token_type_ids = paddle.zeros([batch_size, VL_SEQ_LEN], dtype="int64")
    mm_token_type_ids[0, TEXT_BEFORE : TEXT_BEFORE + NUM_IMAGE_TOKENS] = 1

    image_grid_thw = paddle.to_tensor([[GRID_T, GRID_H, GRID_W]], dtype="int32")
    pixel_values = paddle.randn(
        [GRID_T, IN_CHANNELS, TEMPORAL_PATCH_SIZE, IMAGE_H, IMAGE_W]
    )

    return {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "mm_token_type_ids": mm_token_type_ids,
    }


# ======================================================================
# Single-device baseline
# ======================================================================


def single_device_baseline(seed, use_moe=False):
    """Run Qwen3.5 VL model on a single device and return loss + model."""
    _set_random_seed(seed)

    strategy = fleet.DistributedStrategy()
    language_config = _make_language_config(use_moe=use_moe)
    model = _build_qwen3_5_model(language_config, strategy)

    # Convert model params to bf16 (attention requires fp16/bf16 with
    # packed_seq_params, and TP reduce_scatter needs matching dtypes).
    model = paddle.amp.decorate(models=model, level="O2", dtype="bfloat16")

    dict_args = _make_input_data(batch_size=1)

    # Forward + backward under bf16 autocast
    with paddle.amp.auto_cast(level="O2", dtype="bfloat16", use_promote=False):
        output = model.forward(dict_args)
        loss = output.sum()
        loss.backward()

    return loss.item(), model


# ======================================================================
# TP + SP distributed run
# ======================================================================


def run_tp_sp(seed, loss_baseline, serial_model, use_moe=False):
    """Run Qwen3.5 VL model with TP=4 + SP and compare against baseline."""
    _set_random_seed(seed)

    strategy = fleet.DistributedStrategy()

    language_config = _make_language_config(
        tensor_model_parallel_size=4,
        sequence_parallel=True,
        use_moe=use_moe,
    )
    model = _build_qwen3_5_model(language_config, strategy)

    # Convert model params to bf16
    model = paddle.amp.decorate(models=model, level="O2", dtype="bfloat16")

    register_sequence_parallel_allreduce_hooks(model, 1, False)

    dict_args = _make_input_data(batch_size=1)

    # Forward + backward under bf16 autocast
    with paddle.amp.auto_cast(level="O2", dtype="bfloat16", use_promote=False):
        output = model.forward(dict_args)
        loss = output.sum()
        loss.backward()

    loss_val = loss.item()

    assert is_within_range(loss_val, loss_baseline), (
        f"TP-SP Loss: {loss_val}, Baseline Loss: {loss_baseline}, "
        f"Diff: {abs(loss_val - loss_baseline) / abs(loss_baseline) * 100:.2f}%, "
        f"The difference is out of the tolerance range 1.5%."
    )

    print(
        f"[PASS] Loss baseline={loss_baseline:.6f}, "
        f"TP-SP={loss_val:.6f}, "
        f"diff={abs(loss_val - loss_baseline) / abs(loss_baseline) * 100:.4f}%"
    )


# ======================================================================
# Test cases
# ======================================================================


class TestQwen3_5TPSP(unittest.TestCase):
    """TP+SP precision alignment tests for Qwen3.5 VL model."""

    @classmethod
    def setUpClass(cls):
        """Initialize distributed environment once for all tests."""
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

    def setUp(self):
        self.seed = 46

    def run_test_dense_tp_sp(self):
        """Dense model: TP=4 + SP loss matches single-device baseline."""
        loss, serial_model = single_device_baseline(self.seed, use_moe=False)
        run_tp_sp(
            self.seed,
            loss,
            serial_model,
            use_moe=False,
        )

    def run_test_moe_tp_sp(self):
        """MoE model: TP=4 + SP loss matches single-device baseline."""
        loss, serial_model = single_device_baseline(self.seed, use_moe=True)
        run_tp_sp(
            self.seed,
            loss,
            serial_model,
            use_moe=True,
        )

    def test_all(self):
        self.run_test_dense_tp_sp()
        self.run_test_moe_tp_sp()


if __name__ == "__main__":
    unittest.main()
