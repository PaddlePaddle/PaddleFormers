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
Multi-card unit tests for subbatch functionality.

Test scenarios:
  1. TP (Tensor Parallel) + subbatch loss: parallel_cross_entropy + subbatch
     - Verify subbatch loss equals non-subbatch loss under TP
  2. TP + SP (Sequence Parallel) + subbatch loss
     - Verify subbatch loss consistency under sequence-parallel sharding
  3. GPT full forward-backward with subbatch loss enabled
     - Compare loss and grad norm between subbatch and non-subbatch under TP
  4. subbatch() helper under distributed context

Run with:
  python -m paddle.distributed.launch --gpus 0,1,2,3 \
      tests/fleet/multi_card_tests/test_subbatch.py
"""

import os
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

import functools
import random
import unittest

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import NoPipelineParallel
from paddle.distributed.fleet.utils.sequence_parallel_utils import (
    register_sequence_parallel_allreduce_hooks,
)

import paddleformers.fleet.parallel_state as ps
from paddleformers.fleet.gpt_builders import gpt_builder
from paddleformers.fleet.models.common.language_loss.language_loss import (
    LanguageLoss,
    subbatch,
)
from paddleformers.fleet.models.gpt import GPTConfig
from paddleformers.fleet.tensor_parallel.random import (
    model_parallel_cuda_manual_seed,
)
from tests.fleet.multi_card_tests.tensor_parallel.test_utilities import Utils

# ==================== Module-level initialization (once only) ====================

TP_SIZE = None  # set in setUpModule


def setUpModule():
    """Initialize distributed environment once for all test classes.
    Uses all available GPUs as tensor parallel group."""
    global TP_SIZE
    TP_SIZE = dist.get_world_size()
    Utils.initialize_model_parallel(tensor_parallel_size=TP_SIZE)


# ==================== Helpers ====================

SEED = 42


def _set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def _base_gpt_config(**overrides):
    defaults = {
        "num_hidden_layers": 2,
        "hidden_size": 512,
        "vocab_size": 1024,
        "max_sequence_length": 64,
        "num_attention_heads": 4,
        "intermediate_size": 1024,
        "normalization": "RMSNorm",
        "hidden_dropout_prob": 0.0,
        "attention_dropout": 0.0,
        "use_bias": False,
        "rotary_percent": 1.0,
        "rotary_base": 10000,
        "rope_scaling": 1.0,
        "tie_word_embeddings": True,
        "use_qk_norm": True,
        "use_cpu_initialization": True,
        "position_embedding_type": "rope",
        "init_method": functools.partial(
            paddle.nn.init.xavier_uniform_, gain=1.0
        ),
        "output_layer_init_method": functools.partial(
            paddle.nn.init.xavier_uniform_, gain=1.0
        ),
    }
    defaults.update(overrides)
    return GPTConfig(**defaults)


def _make_inputs(batch_size, seq_len, vocab_size):
    """Build (inputs_dict, labels) for NoPipelineParallel."""
    _set_seed()
    data = paddle.randint(
        low=0, high=vocab_size, shape=(batch_size, seq_len + 1)
    )
    input_ids = data[:, :-1]
    labels = data[:, 1:]
    position_ids = (
        paddle.arange(seq_len, dtype="int64")
        .unsqueeze(0)
        .expand([batch_size, seq_len])
    )
    inputs = (
        {
            "input_ids": [input_ids],
            "position_ids": [position_ids],
        },
        [labels],
    )
    return inputs


def _make_sharded_logits(batch, seq, vocab):
    """
    Generate logits sharded along the vocab dimension, matching real TP behavior.

    Under parallel_output=True each rank holds logits[:, :, rank_start:rank_end].
    We generate the full logits with a fixed seed on all ranks, then slice by rank,
    so different ranks hold genuinely different vocab partitions.
    """
    tp_rank = ps.get_tensor_model_parallel_rank()
    tp_size = ps.get_tensor_model_parallel_world_size()
    vocab_per_rank = vocab // tp_size
    _set_seed()
    full_logits = paddle.randn([batch, seq, vocab])  # same seed → reproducible
    # slice this rank's vocab shard
    start = tp_rank * vocab_per_rank
    logits = full_logits[:, :, start : start + vocab_per_rank].clone()
    labels = paddle.randint(0, vocab, shape=[batch, seq], dtype="int64")
    return logits, labels, vocab_per_rank


# ==================== Test 1: LanguageLoss with subbatch ====================


class TestLanguageLossSubbatch(unittest.TestCase):
    """
    Verify that LanguageLoss with subbatch produces the same loss
    as without subbatch when parallel_cross_entropy is enabled.
    """

    def _run_language_loss(self, subbatch_len: int, logits, labels):
        config = _base_gpt_config(
            tensor_model_parallel_size=TP_SIZE,
            parallel_output=True,
            loss_subbatch_sequence_length=subbatch_len,
        )
        loss_module = LanguageLoss(config)
        logits_in = logits.detach().clone()
        logits_in.stop_gradient = False
        out = loss_module.forward_impl(logits_in, labels)
        out.backward()
        return out.item(), logits_in.grad.clone()

    def test_subbatch_loss_equals_full_loss(self):
        """Loss with subbatch must match loss without subbatch."""
        batch, seq, vocab = 2, 32, 1024
        logits, labels, _ = _make_sharded_logits(batch, seq, vocab)

        loss_full, grad_full = self._run_language_loss(-1, logits, labels)
        loss_sb, grad_sb = self._run_language_loss(8, logits, labels)

        np.testing.assert_allclose(
            loss_sb,
            loss_full,
            rtol=1e-5,
            err_msg=f"[TP={TP_SIZE}] subbatch loss {loss_sb} != full loss {loss_full}",
        )
        np.testing.assert_allclose(
            grad_sb.numpy(),
            grad_full.numpy(),
            rtol=1e-5,
            err_msg=f"[TP={TP_SIZE}] subbatch gradient differs from full-batch gradient",
        )

    def test_subbatch_not_triggered_when_seq_le_subbatch_len(self):
        """When seq_len <= loss_subbatch_sequence_length, should skip chunking."""
        batch, seq, vocab = 2, 8, 1024
        logits, labels, _ = _make_sharded_logits(batch, seq, vocab)

        loss_full, _ = self._run_language_loss(-1, logits, labels)
        loss_sb, _ = self._run_language_loss(64, logits, labels)

        np.testing.assert_allclose(
            loss_sb,
            loss_full,
            rtol=1e-5,
            err_msg=f"[TP={TP_SIZE}] Loss changed when subbatch not triggered",
        )

    def test_ignored_index_masking(self):
        """Subbatch with -100 ignored labels must match full-batch."""
        batch, seq, vocab = 2, 32, 1024
        logits, labels, _ = _make_sharded_logits(batch, seq, vocab)
        labels[:, : seq // 2] = -100

        loss_full, _ = self._run_language_loss(-1, logits, labels)
        loss_sb, _ = self._run_language_loss(8, logits, labels)

        np.testing.assert_allclose(
            loss_sb,
            loss_full,
            rtol=1e-5,
            err_msg=f"[TP={TP_SIZE}] ignored_index masking inconsistent under subbatch",
        )

    def test_non_divisible_seq_len(self):
        """Non-divisible seq_len must work correctly."""
        batch, seq, vocab = 2, 10, 1024
        logits, labels, _ = _make_sharded_logits(batch, seq, vocab)

        loss_full, _ = self._run_language_loss(-1, logits, labels)
        loss_sb, _ = self._run_language_loss(3, logits, labels)

        np.testing.assert_allclose(
            loss_sb,
            loss_full,
            rtol=1e-5,
            err_msg=f"[TP={TP_SIZE}] Non-divisible seq_len: subbatch loss differs",
        )


# ==================== Test 2: GPT forward-backward with TP + subbatch ====================


class TestGPTSubbatchTP(unittest.TestCase):
    """
    End-to-end test: GPT forward-backward pipeline with
    loss_subbatch_sequence_length enabled.

    Verifies:
    - loss with subbatch ≈ loss without subbatch
    - gradient norms are consistent
    """

    BATCH = 2
    SEQ = 64
    VOCAB = 1024

    def _run_gpt(self, subbatch_len: int):
        _set_seed()
        config = _base_gpt_config(
            vocab_size=self.VOCAB,
            max_sequence_length=self.SEQ,
            tensor_model_parallel_size=TP_SIZE,
            parallel_output=True,
            loss_subbatch_sequence_length=subbatch_len,
        )
        model = gpt_builder(config, num_stages=1)
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": TP_SIZE,
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
        pipe_model = NoPipelineParallel(model, strategy)
        inputs = _make_inputs(self.BATCH, self.SEQ, self.VOCAB)
        loss = pipe_model.forward_backward_pipeline(inputs)

        grad_norms = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norms[name] = param.grad.detach().norm().item()
        return loss.item(), grad_norms

    def test_gpt_loss_subbatch_equals_no_subbatch(self):
        """GPT loss with subbatch must be numerically close to loss without subbatch."""
        loss_full, grads_full = self._run_gpt(-1)
        loss_sb, grads_sb = self._run_gpt(16)

        np.testing.assert_allclose(
            loss_sb,
            loss_full,
            rtol=1e-5,
            err_msg=(
                f"[GPT TP={TP_SIZE}] loss mismatch: subbatch={loss_sb:.6f}, "
                f"full={loss_full:.6f}"
            ),
        )

        for name in grads_full:
            if name in grads_sb:
                np.testing.assert_allclose(
                    grads_sb[name],
                    grads_full[name],
                    rtol=1e-5,
                    err_msg=(
                        f"[GPT TP={TP_SIZE}] grad norm mismatch for '{name}': "
                        f"subbatch={grads_sb[name]:.6f}, full={grads_full[name]:.6f}"
                    ),
                )

    def test_gpt_subbatch_not_triggered_when_seq_short(self):
        """When subbatch_len >= seq_len, loss must be identical to no-subbatch."""
        loss_full, _ = self._run_gpt(-1)
        loss_sb, _ = self._run_gpt(self.SEQ)

        np.testing.assert_allclose(
            loss_sb,
            loss_full,
            rtol=1e-5,
            err_msg=f"[GPT TP={TP_SIZE}] Loss changed when subbatch should not trigger",
        )


# ==================== Test 3: TP + SP + subbatch ====================


class TestGPTSubbatchTPSP(unittest.TestCase):
    """
    End-to-end test: GPT forward-backward under TP + Sequence Parallel
    with loss_subbatch_sequence_length enabled.

    Verifies that subbatch produces consistent loss under SP sharding.
    """

    BATCH = 2
    SEQ = 64
    VOCAB = 1024

    def _run_gpt_sp(self, subbatch_len: int):
        _set_seed()
        model_parallel_cuda_manual_seed(SEED)
        config = _base_gpt_config(
            vocab_size=self.VOCAB,
            max_sequence_length=self.SEQ,
            tensor_model_parallel_size=TP_SIZE,
            sequence_parallel=True,
            parallel_output=True,
            loss_subbatch_sequence_length=subbatch_len,
        )
        model = gpt_builder(config, num_stages=1)
        register_sequence_parallel_allreduce_hooks(model, 1, False)

        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": TP_SIZE,
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
        pipe_model = NoPipelineParallel(model, strategy)
        inputs = _make_inputs(self.BATCH, self.SEQ, self.VOCAB)
        loss = pipe_model.forward_backward_pipeline(inputs)
        return loss.item()

    def test_gpt_sp_subbatch_loss_consistent(self):
        """GPT loss with subbatch under TP+SP must match loss without subbatch."""
        loss_full = self._run_gpt_sp(-1)
        loss_sb = self._run_gpt_sp(16)

        np.testing.assert_allclose(
            loss_sb,
            loss_full,
            rtol=1e-5,
            err_msg=(
                f"[GPT TP={TP_SIZE}+SP] loss mismatch: subbatch={loss_sb:.6f}, "
                f"full={loss_full:.6f}"
            ),
        )


# ==================== Test 4: subbatch() helper under distributed context ====================


class TestSubbatchHelperDistributed(unittest.TestCase):
    """
    Verify that the subbatch() helper function behaves correctly when called
    in a distributed context: output equals full-batch, gradients match.
    """

    def test_subbatch_output_consistent_across_ranks(self):
        """All ranks must compute the same subbatch output for identical inputs."""

        def f(x):
            return paddle.nn.functional.relu(x)

        _set_seed()
        x = paddle.randn([2, 16, 8])
        out_full = f(x)

        sb_f = subbatch(f, arg_idx=[0], axis=[1], bs=4, out_idx=1)
        out_sb = sb_f(x)

        np.testing.assert_allclose(
            out_sb.numpy(),
            out_full.numpy(),
            rtol=1e-5,
            err_msg=f"[rank {dist.get_rank()}] subbatch output inconsistent",
        )

    def test_subbatch_gradient_consistent_across_ranks(self):
        """Gradients computed via subbatch must match full-batch gradients on all ranks."""

        def f(x):
            return x * 2.0

        _set_seed()
        x_full = paddle.randn([2, 8, 4])
        x_full.stop_gradient = False
        (f(x_full)).sum().backward()
        grad_full = x_full.grad.numpy().copy()

        x_sb = x_full.detach().clone()
        x_sb.stop_gradient = False
        sb_f = subbatch(f, arg_idx=[0], axis=[1], bs=4, out_idx=1)
        sb_f(x_sb).sum().backward()
        grad_sb = x_sb.grad.numpy()

        np.testing.assert_allclose(
            grad_sb,
            grad_full,
            rtol=1e-5,
            err_msg=f"[rank {dist.get_rank()}] subbatch gradient inconsistent",
        )


if __name__ == "__main__":
    unittest.main()
