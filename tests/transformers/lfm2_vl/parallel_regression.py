# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""Opt-in real two-rank TP/SP regression; not a mock of distributed collectives.

Run from the repository root:
    python -m tests.transformers.lfm2_vl.parallel_regression
"""

import json

import numpy as np
import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet.layers.mpu.random import model_parallel_random_seed
from paddle.distributed.fleet.utils.sequence_parallel_utils import (
    register_sequence_parallel_allreduce_hooks,
)

from paddleformers.transformers.lfm2.configuration import Lfm2Config
from paddleformers.transformers.lfm2.modeling import Lfm2ForCausalLM


def worker():
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {"dp_degree": 1, "mp_degree": 2, "pp_degree": 1, "sharding_degree": 1}
    fleet.init(is_collective=True, strategy=strategy)
    rank = fleet.get_hybrid_communicate_group().get_model_parallel_rank()
    model_parallel_random_seed(26)

    for sequence_parallel in (False, True):
        for tied in (False, True):
            common = dict(
                vocab_size=128,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=3,
                num_attention_heads=4,
                num_key_value_heads=2,
                layer_types=["conv", "full_attention", "conv"],
                conv_bias=True,
                tie_word_embeddings=tied,
                max_sequence_length=8,
                tensor_parallel_output=False,
                fuse_rms_norm=False,
                _attn_implementation="eager",
            )
            reference = Lfm2ForCausalLM(Lfm2Config(**common)).train()
            parallel = Lfm2ForCausalLM(
                Lfm2Config(**common, tensor_model_parallel_size=2, sequence_parallel=sequence_parallel)
            ).train()
            state = reference.state_dict()
            with paddle.no_grad():
                for name, parameter in parallel.named_parameters():
                    source = state[name]
                    if getattr(parameter, "is_distributed", False):
                        source = source.chunk(2, axis=parameter.split_axis)[rank]
                    parameter.set_value(source)
            if sequence_parallel:
                register_sequence_parallel_allreduce_hooks(
                    parallel, accumulation_steps=2, fuse_sequence_parallel_allreduce=False
                )
            ids = paddle.to_tensor([[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]])
            mask = paddle.to_tensor([[1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1, 1, 1]])
            for _ in range(2):
                expected = reference(input_ids=ids, attention_mask=mask, use_cache=False).logits
                actual = parallel(input_ids=ids, attention_mask=mask, use_cache=False).logits
                np.testing.assert_allclose(actual.numpy(), expected.numpy(), atol=3e-5, rtol=3e-5)
                logits_diff = float((actual - expected).abs().max())
                expected.square().mean().backward()
                actual.square().mean().backward()
            reference_parameters = dict(reference.named_parameters())
            max_gradient_diff = 0.0
            for name, parameter in parallel.named_parameters():
                expected = reference_parameters[name].grad
                assert expected is not None and parameter.grad is not None, name
                if getattr(parameter, "is_distributed", False):
                    expected = expected.chunk(2, axis=parameter.split_axis)[rank]
                np.testing.assert_allclose(
                    parameter.grad.numpy(), expected.numpy(), atol=3e-5, rtol=3e-4, err_msg=name
                )
                max_gradient_diff = max(max_gradient_diff, float((parameter.grad - expected).abs().max()))
            print(
                json.dumps(
                    {
                        "rank": rank,
                        "tp": 2,
                        "sp": sequence_parallel,
                        "tied": tied,
                        "logits_max_diff": logits_diff,
                        "all_gradients_max_diff": max_gradient_diff,
                    }
                ),
                flush=True,
            )
    paddle.distributed.barrier()


if __name__ == "__main__":
    if paddle.device.cuda.device_count() < 2:
        raise RuntimeError(
            "This regression requires two CUDA devices; CPU/Gloo does not implement Fleet's calc-stream collectives."
        )
    paddle.distributed.spawn(worker, nprocs=2, backend="nccl")
