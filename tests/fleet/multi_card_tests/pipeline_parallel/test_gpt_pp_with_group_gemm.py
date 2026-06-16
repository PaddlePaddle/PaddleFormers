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
from paddle.distributed.fleet import distributed_model

import paddleformers.fleet
from paddleformers.fleet.gpt_builders import gpt_builder
from paddleformers.fleet.models.gpt import GPTConfig
from paddleformers.fleet.training.initialize import initialize_fleet

PP_DEGREE = 4


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
        seed = seed_ + (
            100
            * paddleformers.fleet.parallel_state.get_pipeline_model_parallel_rank()
        )
        # Ensure different data parallel ranks get different seeds
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


def run_pp(
    seed,
    batch_size,
    seq_len,
    vocab_size,
    config,
    forward_backward_overlap_scheduler=False,
):
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": config.tensor_model_parallel_size,
        "pp_degree": config.pipeline_model_parallel_size,
        "sharding_degree": 1,
        "sep_degree": 1,
        "cp_degree": 1,
        "ep_degree": config.tensor_model_parallel_size,
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
        "pp_configs": {
            "forward_backward_overlap_scheduler": forward_backward_overlap_scheduler
        },
    }
    micro_batch_size = 1
    num_acc = batch_size // micro_batch_size
    strategy.pipeline_configs = {
        "accumulate_steps": num_acc,
        "micro_batch_size": micro_batch_size,
        "enable_partial_send_recv": False,  # Must be False when sequence_parallel=True
    }
    initialize_fleet(strategy)

    _set_random_seed(seed)

    gpt_model = gpt_builder(
        config,
        num_stages=config.pipeline_model_parallel_size,
        seg_method="layer:TransformerLayer|EmptyLayer",
    )
    gpt_model = paddle.amp.decorate(
        models=gpt_model, optimizers=None, level="O2", dtype="bfloat16"
    )

    gpt_pipe_model = distributed_model(gpt_model)

    data = paddle.randint(
        low=0, high=vocab_size, shape=(micro_batch_size, seq_len + 1)
    )
    input_ids = data[:, :-1]
    labels = data[:, 1:]
    position_ids = paddle.to_tensor(data, dtype=paddle.int64).repeat(
        (micro_batch_size, 1)
    )

    inputs = (
        {
            "input_ids": [input_ids] * num_acc,
            "position_ids": [position_ids] * num_acc,
        },
        [labels] * num_acc,
    )

    loss = gpt_pipe_model.forward_backward_pipeline(inputs, None)
    return loss, gpt_pipe_model


class TestPP(unittest.TestCase):
    def setUp(self):
        self.seed = 46
        self.batch_size = 12
        self.seq_len = 128
        self.vocab_size = 1024

    def test_pp(self):
        if (
            not paddle.device.current_device_is_cpu
            and paddle.device.get_device_capability()[0] < 9
        ):
            return
        config = GPTConfig(
            vocab_size=self.vocab_size,
            max_sequence_length=self.seq_len,
            num_hidden_layers=11,
            hidden_size=512,
            num_attention_heads=4,
            intermediate_size=1024,
            normalization="RMSNorm",
            hidden_dropout_prob=0.0,
            attention_dropout=0.0,
            use_cpu_initialization=True,
            parallel_output=True,
            tie_word_embeddings=True,
            position_embedding_type="rope",
            rotary_percent=1.0,
            rotary_base=10000,
            rope_scaling=1.0,
            init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            output_layer_init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            use_qk_norm=True,
            num_empty_layers_add_in_head=2,
            num_empty_layers_add_in_tail=3,
            pipeline_model_parallel_size=PP_DEGREE,
            virtual_pipeline_model_parallel_size=2,
            tensor_model_parallel_size=2,
            expert_model_parallel_size=2,
            sequence_parallel=True,
            n_shared_experts=1,
            n_routed_experts=8,
            moe_intermediate_size=1024,
            bf16=True,
            moe_token_dispatcher_type="deepep",
            gated_linear_unit=True,
            bias_activation_fusion=True,
            norm_topk_prob=False,
        )
        config.moe_expert_fusion = True
        config.moe_deep_gemm = True

        overlap_loss, overlap_gpt_model = run_pp(
            self.seed,
            self.batch_size,
            self.seq_len,
            self.vocab_size,
            config,
            forward_backward_overlap_scheduler=True,
        )

        assert overlap_loss._md5sum() == "bc7d1057df35d56003ba91c1801a9acb"

        if paddle.distributed.get_rank() == 0:
            baseline = {
                "_layers.shared_layers.embed.embedding.embed_tokens.weight": "4d1da78d31fda37efff3ce130f49774b",
                "_layers.9.0.input_layernorm.weight": "61cf0cdd4ccdffa29bc5addde85f7a00",
                "_layers.9.0.self_attn.o_proj.weight": "e8b7ccefe45bdc4e98c34a4027d573ca",
                "_layers.9.0.self_attn.qkv_proj.weight": "c8749b42237bddc3f972b4c2a4bafcf6",
                "_layers.9.0.self_attn.q_norm.weight": "b9af1a3cff0a03e649b4d275ce8ac13c",
                "_layers.9.0.self_attn.k_norm.weight": "576bc73bc3de6d350f641efe31e8c67a",
                "_layers.9.0.post_attention_layernorm.weight": "a2988b16d7c5ba9093c680873dfbfb01",
                "_layers.9.0.mlp.gate.weight": "3f20707f0d45e4b94dd288f81350287c",
                "_layers.9.0.mlp.grouped_gemm_experts.weight1": "39449d13822de77685452d09e48f17e3",
                "_layers.9.0.mlp.grouped_gemm_experts.weight2": "d2eaeace5c65309cf1e88cfe5dd08fd6",
                "_layers.9.0.mlp.shared_experts.up_gate_proj.weight": "d14317ce681d3a801e4e7fd14ba16ac9",
                "_layers.9.0.mlp.shared_experts.down_proj.weight": "758ecd35c5e94e143c33221f6ed8d95e",
                "_layers.9.1.input_layernorm.weight": "7c3bf7bcc86076aa22f3022bd108b7cd",
                "_layers.9.1.self_attn.o_proj.weight": "623d83b63c05a984791916a46040dfdf",
                "_layers.9.1.self_attn.qkv_proj.weight": "c0ad3fc0770931e81a9600ffb288dc4f",
                "_layers.9.1.self_attn.q_norm.weight": "532f2601a39dde43d7e846c0788dc1f6",
                "_layers.9.1.self_attn.k_norm.weight": "618b318fcaf96dc314f4f73fcfdfd5b7",
                "_layers.9.1.post_attention_layernorm.weight": "91b5642254475cf7de3b2e12cab58d5e",
                "_layers.9.1.mlp.gate.weight": "7df6522a1da204d40ea895be8d72a114",
                "_layers.9.1.mlp.grouped_gemm_experts.weight1": "f7fdd1638f8095442d7b2be6f907a258",
                "_layers.9.1.mlp.grouped_gemm_experts.weight2": "42c0f502fe3a9139a41e76ebefa7701a",
                "_layers.9.1.mlp.shared_experts.up_gate_proj.weight": "9a3c7de300f34a15924746d1a0a4b29f",
                "_layers.9.1.mlp.shared_experts.down_proj.weight": "80542d9d52256dadb1f02b0e6cb1d25a",
            }

            for name, param in overlap_gpt_model.named_parameters():
                assert param.grad._md5sum() == baseline[name]


if __name__ == "__main__":
    unittest.main()
