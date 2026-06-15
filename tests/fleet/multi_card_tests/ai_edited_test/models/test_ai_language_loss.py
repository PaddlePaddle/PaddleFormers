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
import sys
import unittest

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)

from paddleformers.fleet.models.common.language_loss.language_loss import LanguageLoss
from paddleformers.fleet.process_groups_config import ProcessGroupCollection
from paddleformers.fleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddleformers.fleet.training.initialize import initialize_fleet
from paddleformers.fleet.transformer.transformer_config import TransformerConfig

PP_DEGREE = 2
SEED = 66
WORLD_SIZE = None
_GPU_COMPUTE_OK = None


def _set_seed(seed_):
    np.random.seed(seed_)
    paddle.seed(seed_)
    paddle.manual_seed(seed_)


def _check_gpu_compute():
    """Check if GPU compute operations work on this environment."""
    global _GPU_COMPUTE_OK
    if _GPU_COMPUTE_OK is not None:
        return _GPU_COMPUTE_OK
    try:
        t = paddle.randn([4, 4], dtype="float32")
        _ = paddle.nn.functional.softmax(t, axis=-1)
        _GPU_COMPUTE_OK = True
    except Exception:
        _GPU_COMPUTE_OK = False
    return _GPU_COMPUTE_OK


def _init_fleet_pp2():
    """Initialize fleet with PP=2 for language loss testing."""
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": 1,
        "pp_degree": PP_DEGREE,
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
    initialize_fleet(strategy=strategy)


def setUpModule():
    """Initialize fleet once for all tests in this module."""
    global WORLD_SIZE
    WORLD_SIZE = dist.get_world_size()
    _set_seed(SEED)
    _init_fleet_pp2()
    model_parallel_cuda_manual_seed(SEED)


def _requires_gpu_compute(test_func):
    """Decorator to skip test if GPU compute is not available."""

    @unittest.skipUnless(
        _check_gpu_compute(),
        "GPU compute not available (likely CUDA driver version mismatch)",
    )
    def wrapper(*args, **kwargs):
        return test_func(*args, **kwargs)

    wrapper.__name__ = test_func.__name__
    wrapper.__doc__ = test_func.__doc__
    return wrapper


def _build_config(**overrides):
    defaults = dict(  # noqa: C408
        hidden_size=64,
        num_attention_heads=4,
        use_cpu_initialization=True,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=PP_DEGREE,
        sequence_parallel=False,
        bf16=False,
        params_dtype=paddle.float32,
        parallel_output=True,
        rms_norm_eps=1e-5,
        hidden_dropout_prob=0.0,
        attention_dropout=0.0,
        loss_subbatch_sequence_length=0,
        init_method=functools.partial(paddle.nn.init.xavier_uniform_, gain=1.0),
        output_layer_init_method=functools.partial(paddle.nn.init.xavier_uniform_, gain=1.0),
    )
    defaults.update(overrides)
    return TransformerConfig(**defaults)


class TestLanguageLoss(unittest.TestCase):
    """Test LanguageLoss module from paddleformers.fleet."""

    @classmethod
    def setUpClass(cls):
        cls.config = _build_config()
        cls.pg_collection = ProcessGroupCollection.use_mpu_process_groups()

    @_requires_gpu_compute
    def test_language_loss_creation(self):
        """Test LanguageLoss layer creation."""
        loss_fn = LanguageLoss(config=self.config)
        self.assertIsNotNone(loss_fn)

    @_requires_gpu_compute
    def test_language_loss_forward(self):
        """Test LanguageLoss forward pass with logits and labels."""
        batch_size = 2
        seq_len = 8
        vocab_size = 64

        logits = paddle.randn([batch_size, seq_len, vocab_size], dtype="float32")
        labels = paddle.randint(0, vocab_size, [batch_size, seq_len], dtype="int64")
        loss_fn = LanguageLoss(config=self.config)
        loss = loss_fn(logits, labels)
        self.assertIsNotNone(loss)

    @_requires_gpu_compute
    def test_language_loss_backward(self):
        """Test LanguageLoss backward pass, verify gradients exist."""
        batch_size = 2
        seq_len = 8
        vocab_size = 64

        logits = paddle.randn([batch_size, seq_len, vocab_size], dtype="float32")
        logits.stop_gradient = False
        labels = paddle.randint(0, vocab_size, [batch_size, seq_len], dtype="int64")
        loss_fn = LanguageLoss(config=self.config)
        loss = loss_fn(logits, labels)
        loss.sum().backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(logits.grad.abs().sum().item() > 0)


if __name__ == "__main__":
    unittest.main()
