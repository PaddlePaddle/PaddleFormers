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
            moe_expert_fusion=False,
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
        config.recompute_granularity = "full"
        config.recompute_method = "uniform"
        config.recompute_num_layers = 1

        overlap_loss, overlap_gpt_model = run_pp(
            self.seed,
            self.batch_size,
            self.seq_len,
            self.vocab_size,
            config,
            forward_backward_overlap_scheduler=True,
        )

        assert overlap_loss._md5sum() == "b9d9bab70678927c5001583312506560"

        if paddle.distributed.get_rank() == 0:
            baseline = {
                "_layers.shared_layers.embed.embedding.embed_tokens.weight": "577a358e768837f5ceb97074aad498e4",
                "_layers.9.0.input_layernorm.weight": "90fa9236e99a3253e447727366281cd5",
                "_layers.9.0.self_attn.o_proj.weight": "26e9ca47e0d34e798c78882beabd24af",
                "_layers.9.0.self_attn.qkv_proj.weight": "84c1488965be451559e734b85b0a6eb0",
                "_layers.9.0.self_attn.q_norm.weight": "6cef735e7ad0fbfef69165014404dfd3",
                "_layers.9.0.self_attn.k_norm.weight": "d4d1bcf10f1de8501fb9c42ba9d8bd70",
                "_layers.9.0.post_attention_layernorm.weight": "9a56d568ad36ae1a65825369e2b7e2b3",
                "_layers.9.0.mlp.gate.weight": "0295e0592c7397f3db52f9a551b7a7fb",
                "_layers.9.0.mlp.experts.0.up_gate_proj.weight": "59eaa4cc0dbbb89e24fecf4fccb45c72",
                "_layers.9.0.mlp.experts.0.down_proj.weight": "203ec0569d92e3e9ed6a44025d973b13",
                "_layers.9.0.mlp.experts.1.up_gate_proj.weight": "ddf2d90455142e0d0e4695996f225c17",
                "_layers.9.0.mlp.experts.1.down_proj.weight": "c9502618e5a4f8008a1fcbc31f59666a",
                "_layers.9.0.mlp.experts.2.up_gate_proj.weight": "beb41f714d435f83088e7c290fcdc2bf",
                "_layers.9.0.mlp.experts.2.down_proj.weight": "d889dfaf823bd7c4f2829268bfd12436",
                "_layers.9.0.mlp.experts.3.up_gate_proj.weight": "ed60d49ef00e4723edf4449dab7f3b22",
                "_layers.9.0.mlp.experts.3.down_proj.weight": "a41fa88f17853a14b3cee2c9baf9ebc5",
                "_layers.9.0.mlp.shared_experts.up_gate_proj.weight": "1c5c6fff18e5b548cc6b75a5880cffcc",
                "_layers.9.0.mlp.shared_experts.down_proj.weight": "e57388a64133ea8d174e481da8597647",
                "_layers.9.1.input_layernorm.weight": "e4134b04185a9514ab1e0d37137feadb",
                "_layers.9.1.self_attn.o_proj.weight": "498453442509e676c062cf509a3f9d1b",
                "_layers.9.1.self_attn.qkv_proj.weight": "2e45502db96a2efa0504b9e2c8faa15e",
                "_layers.9.1.self_attn.q_norm.weight": "8202c7dabd01dd6e19d89ab1a5a02ae0",
                "_layers.9.1.self_attn.k_norm.weight": "7b381b3e9d60a8271b92ba5c7f47020d",
                "_layers.9.1.post_attention_layernorm.weight": "5bb297b0c2dd0753e13ac1472021c79b",
                "_layers.9.1.mlp.gate.weight": "7c025be85e78dcd83f6128f97daf50e6",
                "_layers.9.1.mlp.experts.0.up_gate_proj.weight": "89a15b7c2253c6606655ae2bdd7bc3dc",
                "_layers.9.1.mlp.experts.0.down_proj.weight": "4234abb1082fa0f036b73193c4c25dcb",
                "_layers.9.1.mlp.experts.1.up_gate_proj.weight": "90049fc4b1797803e97924ebf3744bec",
                "_layers.9.1.mlp.experts.1.down_proj.weight": "8fbeff7da9c731b7a23486cd31cd3305",
                "_layers.9.1.mlp.experts.2.up_gate_proj.weight": "c67775a255c03689785fdc486ddcfd1b",
                "_layers.9.1.mlp.experts.2.down_proj.weight": "c703c6f62ad0ca2c2bbb70e820f1f962",
                "_layers.9.1.mlp.experts.3.up_gate_proj.weight": "65ccb1a90a990c9bca94388abf0bf442",
                "_layers.9.1.mlp.experts.3.down_proj.weight": "4fc647911be037f80b47202a5e4837af",
                "_layers.9.1.mlp.shared_experts.up_gate_proj.weight": "3c63f2c52c3c286cea8b7ab5ff7c2f47",
                "_layers.9.1.mlp.shared_experts.down_proj.weight": "3525150630c27471831c38b231751c20",
            }

            for name, param in overlap_gpt_model.named_parameters():
                assert param.grad._md5sum() == baseline[name]


if __name__ == "__main__":
    unittest.main()
