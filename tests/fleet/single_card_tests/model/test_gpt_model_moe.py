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
import subprocess
import unittest

import numpy as np
import paddle
from paddle.distributed import fleet

# from paddleformers.fleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddle.distributed.fleet.meta_parallel import NoPipelineParallel

# from tests.unit_tests.test_utilities import Utils
import paddleformers.fleet.parallel_state as ps
from paddleformers.fleet.gpt_builders import gpt_builder
from paddleformers.fleet.models.gpt import GPTConfig


def get_gpu_models_via_nvidia_smi():
    try:
        output = subprocess.check_output("nvidia-smi --query-gpu=name --format=csv,noheader", shell=True)
        models = output.decode().strip().replace("NVIDIA", "")
        return models
    except Exception as e:
        return ["Unknown"]


def judge_machine_type():
    if not paddle.is_compiled_with_cuda():
        return "No CUDA GPU"
    models = get_gpu_models_via_nvidia_smi()
    for model in models:
        name = model.upper()
        if "V" in name:
            return "V"
        elif "H" in name:
            return "H"


result = judge_machine_type()
print("你的机器类型是：", result)


class TestGPTModel(unittest.TestCase):
    def setUp(self):
        seed = 46
        random.seed(seed)
        np.random.seed(seed)
        paddle.seed(seed)
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": 1,
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
        self.strategy = strategy
        fleet.init(is_collective=True, strategy=strategy)
        hcg = fleet.get_hybrid_communicate_group()
        ps.initialize_model_parallel(hcg)

        config = GPTConfig(
            num_hidden_layers=2,
            hidden_size=512,
            vocab_size=100,
            max_sequence_length=64,
            num_attention_heads=4,
            moe_expert_fusion=False,
            intermediate_size=1024,
            normalization="RMSNorm",
            hidden_dropout_prob=0.0,
            attention_dropout=0.0,
            n_routed_experts=8,
            use_bias=False,
            rotary_percent=1.0,
            rotary_base=10000,
            rope_scaling=1.0,
            moe_intermediate_size=1024,
            moe_token_dispatcher_type="alltoall",
            n_shared_experts=1,
            init_method=functools.partial(paddle.nn.init.xavier_uniform_, gain=1.0),
            output_layer_init_method=functools.partial(paddle.nn.init.xavier_uniform_, gain=1.0),
            tie_word_embeddings=True,
            use_qk_norm=True,
        )
        self.gpt_model = gpt_builder(config, num_stages=1)
        self.config = config

    def test_forward(self) -> None:
        sequence_length = self.config.max_sequence_length
        micro_batch_size = 2

        for name, param in self.gpt_model.named_parameters():
            # 计算 L2 范数
            param_norm = param.detach().norm().item()
            param_abssum = param.detach().abs().sum().item()
            print(f"{name}: {param_norm:.6f}, {param_abssum:.6f}")

        data = list(range(sequence_length))
        input_ids = paddle.to_tensor(data, dtype=paddle.int64).repeat((micro_batch_size, 1))
        position_ids = paddle.to_tensor(data, dtype=paddle.int64).repeat((micro_batch_size, 1))
        attention_mask = paddle.ones((micro_batch_size, 1, sequence_length, sequence_length), dtype=bool)
        labels = paddle.to_tensor(list(range(1, sequence_length + 1)), dtype=paddle.int64).repeat(
            (micro_batch_size, 1)
        )

        data = (
            {
                "input_ids": [input_ids],
                "position_ids": [position_ids],
                "attention_mask": [attention_mask],
            },
            [labels],
        )

        gpt_pipe_model = NoPipelineParallel(self.gpt_model, self.strategy)
        loss = gpt_pipe_model.forward_backward_pipeline(data)

        for name, param in self.gpt_model.named_parameters():
            # 计算 L2 范数
            if param.grad is None:
                print(f"{name}: 0.000000, 0.000000")
                continue
            grad_norm = param.grad.detach().norm().item()
            grad_abssum = param.grad.detach().abs().sum().item()
            print(f"{name}: {grad_norm:.6f}, {grad_abssum:.6f}")
            if name == "0.embedding.embed_tokens.weight":
                embed_tokens_grad_norm = grad_norm

        print("loss", loss.item())

        print("embed_tokens_grad_norm", embed_tokens_grad_norm)

        repo_name = os.environ.get("repo_flag")
        if judge_machine_type() == "H":
            assert (
                loss.item() == 5.295381546020508
            ), f"loss not equal ({loss.item()} != 5.295381546020508), please check your modify"
            assert (
                embed_tokens_grad_norm == 5.6999006271362305
            ), f"grad norm of embed_tokens not equal ({embed_tokens_grad_norm} != 5.6999006271362305), please check your modify"
        elif judge_machine_type() == "V":
            assert (
                loss.item() == 5.284281253814697
            ), f"loss not equal ({loss.item()} != 5.284281253814697), please check your modify"
            assert (
                embed_tokens_grad_norm == 9.912039756774902
            ), f"grad norm of embed_tokens not equal ({embed_tokens_grad_norm} != 9.912039756774902, please check your modify"


if __name__ == "__main__":
    unittest.main()
