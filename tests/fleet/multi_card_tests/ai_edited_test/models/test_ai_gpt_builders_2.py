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
from paddle.distributed.fleet import distributed_model

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)

import paddleformers.fleet
from paddleformers.fleet.gpt_builders import gpt_builder
from paddleformers.fleet.models.gpt import GPTConfig
from paddleformers.fleet.training.initialize import initialize_fleet

PP_DEGREE = 4
SEED = 46
BATCH_SIZE = 4
SEQ_LEN = 64
VOCAB_SIZE = 256


def _set_random_seed(seed_):
    if seed_ is not None and seed_ > 0:
        seed = seed_ + (100 * paddleformers.fleet.parallel_state.get_pipeline_model_parallel_rank())
        np.random.seed(seed)
        paddle.manual_seed(seed)
        if paddle.distributed.is_initialized() and paddle.cuda.device_count() > 0:
            paddleformers.fleet.tensor_parallel.model_parallel_cuda_manual_seed(seed)


def _build_encoder_config(**overrides):
    defaults = dict(  # noqa: C408
        vocab_size=VOCAB_SIZE,
        max_sequence_length=SEQ_LEN,
        num_hidden_layers=6,
        hidden_size=256,
        num_attention_heads=4,
        intermediate_size=512,
        normalization="RMSNorm",
        hidden_dropout_prob=0.0,
        attention_dropout=0.0,
        use_cpu_initialization=True,
        parallel_output=True,
        tie_word_embeddings=True,
        position_embedding_type="rope",
        rotary_percent=1.0,
        init_method=functools.partial(paddle.nn.init.xavier_uniform_, gain=1.0),
        output_layer_init_method=functools.partial(paddle.nn.init.xavier_uniform_, gain=1.0),
        pipeline_model_parallel_size=PP_DEGREE,
    )
    defaults.update(overrides)
    return GPTConfig(**defaults)


_NCCL_OK = None


def _check_nccl():
    """Check if NCCL all-reduce works."""
    global _NCCL_OK
    if _NCCL_OK is not None:
        return _NCCL_OK
    try:
        t = paddle.ones([4], dtype="float32")
        dist.all_reduce(t)
        _NCCL_OK = True
    except Exception:
        _NCCL_OK = False
    return _NCCL_OK


@unittest.skipUnless(_check_nccl(), "NCCL not available (CUDA driver version mismatch)")
class TestTransformerEncoderPP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
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
        micro_batch_size = 1
        num_acc = BATCH_SIZE // micro_batch_size
        strategy.pipeline_configs = {
            "accumulate_steps": num_acc,
            "micro_batch_size": micro_batch_size,
        }
        initialize_fleet(strategy)
        _set_random_seed(SEED)

        config = _build_encoder_config()
        cls.config = config
        gpt_model = gpt_builder(
            config,
            num_stages=config.pipeline_model_parallel_size,
            seg_method="layer:TransformerLayer|EmptyLayer",
        )
        cls.gpt_pipe_model = distributed_model(gpt_model)

    def test_encoder_pp_basic(self):
        """Test transformer encoder with basic PP."""
        micro_batch_size = 1
        num_acc = BATCH_SIZE // micro_batch_size
        data = paddle.randint(low=0, high=VOCAB_SIZE, shape=(micro_batch_size, SEQ_LEN + 1))
        input_ids = data[:, :-1]
        labels = data[:, 1:]
        position_ids = paddle.arange(0, SEQ_LEN, dtype="int64").unsqueeze(0).expand([micro_batch_size, -1])
        inputs = (
            {
                "input_ids": [input_ids] * num_acc,
                "position_ids": [position_ids] * num_acc,
            },
            [labels] * num_acc,
        )
        loss = self.gpt_pipe_model.forward_backward_pipeline(inputs, None)
        self.assertIsNotNone(loss)


if __name__ == "__main__":
    unittest.main()
