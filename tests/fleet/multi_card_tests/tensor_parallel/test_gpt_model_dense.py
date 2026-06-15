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
        elif "o_proj.weight" in name or "down_proj.weight" in name or "embed_tokens.weight" in name:
            grad = _gather_along_first_dim(p.grad, tp_group)
        else:
            grad = p.grad

        both_zero = grad.abs().max() == 0 and serial_grads[name].abs().max() == 0
        sim_ok = both_zero or cal_sim(grad, serial_grads[name]) > 0.999
        if not (paddle.allclose(grad, serial_grads[name], atol=5e-7, rtol=1e-6) and sim_ok):
            print(f"{name} failed, serial grad:{serial_grads[name]}, dist:{grad}")
            diff = paddle.abs(serial_grads[name] - grad)
            print(f"max diff: {diff.max()}, sim:{cal_sim(serial_grads[name], grad)}")
            raise AssertionError(f"{name} failed in check grad, serial:{serial_grads[name].shape}, dist:{grad.shape}")


def single_device_baseline(seed, batch_size, seq_len, vocab_size, config):
    seed = 46
    random.seed(seed)
    np.random.seed(seed)
    paddle.manual_seed(seed)

    gpt_model = gpt_builder(config, num_stages=1)

    paddle.manual_seed(seed)
    data = paddle.randint(low=0, high=vocab_size, shape=(batch_size, seq_len + 1))
    input_ids = data[:, :-1]
    labels = data[:, 1:]
    position_ids = paddle.arange(seq_len, dtype=paddle.int64).unsqueeze(0).expand([batch_size, -1])

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

    _set_random_seed(seed)

    gpt_model = gpt_builder(config, num_stages=1)

    register_sequence_parallel_allreduce_hooks(gpt_model, 1, False)

    paddle.manual_seed(seed)
    data = paddle.randint(low=0, high=vocab_size, shape=(batch_size, seq_len + 1))
    input_ids = data[:, :-1]
    labels = data[:, 1:]
    position_ids = paddle.arange(seq_len, dtype=paddle.int64).unsqueeze(0).expand([batch_size, -1])

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

    assert loss == loss_baseline
    check_grads(gpt_pipe_model, gpt_model_baseline, tp_group)


SEED = 46
BATCH_SIZE = 2
SEQ_LEN = 32
VOCAB_SIZE = 1024
HIDDEN_SIZE = 128
NUM_HIDDEN_LAYERS = 4
NUM_ATTENTION_HEADS = 4
INTERMEDIATE_SIZE = 256


def _make_serial_config(**extra):
    """Build a GPTConfig for the single-device baseline."""
    kwargs = {
        "vocab_size": VOCAB_SIZE,
        "max_sequence_length": SEQ_LEN,
        "num_hidden_layers": NUM_HIDDEN_LAYERS,
        "hidden_size": HIDDEN_SIZE,
        "num_attention_heads": NUM_ATTENTION_HEADS,
        "intermediate_size": INTERMEDIATE_SIZE,
        "normalization": "RMSNorm",
        "hidden_dropout_prob": 0.0,
        "attention_dropout": 0.0,
        "use_cpu_initialization": True,
        "parallel_output": True,
        "tie_word_embeddings": True,
        "position_embedding_type": "rope",
        "rotary_percent": 1.0,
        "rotary_base": 10000,
        "rope_scaling": 1.0,
        "init_method": functools.partial(paddle.nn.init.xavier_uniform_, gain=1.0),
        "output_layer_init_method": functools.partial(paddle.nn.init.xavier_uniform_, gain=1.0),
        "use_qk_norm": True,
    }
    kwargs.update(extra)
    return GPTConfig(**kwargs)


def _make_dist_config(**extra):
    """Build a GPTConfig for the distributed (TP) run."""
    kwargs = {
        "vocab_size": VOCAB_SIZE,
        "max_sequence_length": SEQ_LEN,
        "num_hidden_layers": NUM_HIDDEN_LAYERS,
        "hidden_size": HIDDEN_SIZE,
        "num_attention_heads": NUM_ATTENTION_HEADS,
        "intermediate_size": INTERMEDIATE_SIZE,
        "normalization": "RMSNorm",
        "hidden_dropout_prob": 0.0,
        "attention_dropout": 0.0,
        "use_cpu_initialization": True,
        "tensor_model_parallel_size": 4,
        "sequence_parallel": True,
        "parallel_output": True,
        "tie_word_embeddings": True,
        "position_embedding_type": "rope",
        "rotary_percent": 1.0,
        "rotary_base": 10000,
        "rope_scaling": 1.0,
        "init_method": functools.partial(paddle.nn.init.xavier_uniform_, gain=1.0),
        "output_layer_init_method": functools.partial(paddle.nn.init.xavier_uniform_, gain=1.0),
        "use_qk_norm": True,
    }
    kwargs.update(extra)
    return GPTConfig(**kwargs)


class TestTPSP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Compute all serial baselines BEFORE fleet init (TP=1),
        then initialize fleet once with TP=4."""
        # --- serial baselines (TP group not yet created) ---
        serial_cfg = _make_serial_config()
        cls.tp_sp_baseline = single_device_baseline(SEED, BATCH_SIZE, SEQ_LEN, VOCAB_SIZE, serial_cfg)

        serial_cfg_bar = _make_serial_config(
            block_attention_residuals=True,
            attn_res_block_size=2,
        )
        cls.tp_sp_bar_baseline = single_device_baseline(SEED, BATCH_SIZE, SEQ_LEN, VOCAB_SIZE, serial_cfg_bar)

        # --- initialize fleet (sets global TP=4) ---
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
        self.seed = SEED
        self.batch_size = BATCH_SIZE
        self.seq_len = SEQ_LEN
        self.vocab_size = VOCAB_SIZE

    def run_test_tp_sp(self):
        loss, gpt_model = self.tp_sp_baseline
        dist_config = _make_dist_config(sequence_parallel=True)
        run_tp_sp(
            self.seed,
            self.batch_size,
            self.seq_len,
            self.vocab_size,
            dist_config,
            loss,
            gpt_model,
        )

    def run_test_tp(self):
        loss, gpt_model = self.tp_sp_baseline
        dist_config = _make_dist_config(sequence_parallel=False)
        run_tp_sp(
            self.seed,
            self.batch_size,
            self.seq_len,
            self.vocab_size,
            dist_config,
            loss,
            gpt_model,
        )

    def run_test_tp_sp_block_attn_res(self):
        """Test block attention residuals under TP + SP."""
        loss, gpt_model = self.tp_sp_bar_baseline
        dist_config = _make_dist_config(
            block_attention_residuals=True,
            attn_res_block_size=2,
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

    def run_test_tp_block_attn_res(self):
        """Test block attention residuals under TP only (no SP)."""
        loss, gpt_model = self.tp_sp_bar_baseline
        dist_config = _make_dist_config(
            sequence_parallel=False,
            block_attention_residuals=True,
            attn_res_block_size=2,
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

    def test_all_cases(self):
        self.run_test_tp()
        self.run_test_tp_sp()
        self.run_test_tp_sp_block_attn_res()
        self.run_test_tp_block_attn_res()


if __name__ == "__main__":
    unittest.main()
