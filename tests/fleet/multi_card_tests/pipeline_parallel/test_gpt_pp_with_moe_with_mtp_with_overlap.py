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
MTP_DEGREE = 3

REPO_FLAG = os.getenv("repo_flag")
BRANCH = os.getenv("BRANCH")
SKIP_TESTS = (REPO_FLAG != "paddleformers.fleet") and (BRANCH == "develop")


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
            "forward_backward_overlap_scheduler": forward_backward_overlap_scheduler,
            "overlap_p2p_comm": True,
            "enable_dynamic_shape": True,
        },
    }
    micro_batch_size = 1
    num_acc = batch_size // micro_batch_size
    strategy.pipeline_configs = {
        "accumulate_steps": num_acc,
        "micro_batch_size": micro_batch_size,
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
        low=0,
        high=vocab_size,
        shape=(micro_batch_size, seq_len + MTP_DEGREE + 1),
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


@unittest.skipIf(
    SKIP_TESTS,
    f"Skipping tests: repo_flag={REPO_FLAG} (not 'paddleformers.fleet') and branch '{BRANCH}' is 'develop'",
)
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
            num_nextn_predict_layers=MTP_DEGREE,
            norm_topk_prob=False,
        )

        overlap_loss, overlap_gpt_model = run_pp(
            self.seed,
            self.batch_size,
            self.seq_len,
            self.vocab_size,
            config,
            forward_backward_overlap_scheduler=True,
        )

        print(overlap_loss._md5sum())

        rst = {}
        for name, param in overlap_gpt_model.named_parameters():
            if param.grad is not None:
                rst[name] = param.grad._md5sum()

        print(rst)

        assert overlap_loss._md5sum() == "864e194f213e7cc5e825e847c91a557d"

        if paddle.distributed.get_rank() == 0:
            baseline = {
                "_layers.shared_layers.embed.embedding.embed_tokens.weight": "d1fb19d10f5dce637467a36d7cce95bf",
                "_layers.9.0.input_layernorm.weight": "513eff5aa58d0d7f46527c3d4c8069cb",
                "_layers.9.0.self_attn.o_proj.weight": "f0f60610a0a2b1fd21410b044f620337",
                "_layers.9.0.self_attn.qkv_proj.weight": "9ad78e891dfe8fdf16efb55d6de15fd8",
                "_layers.9.0.self_attn.q_norm.weight": "56e97b9f314ef79db2506ba6f5d07072",
                "_layers.9.0.self_attn.k_norm.weight": "d15896c22708fb079cd04d17869acbae",
                "_layers.9.0.post_attention_layernorm.weight": "7880c6f6660e84c640512e5fa306fe89",
                "_layers.9.0.mlp.gate.weight": "4f0bd321ab73a64f768e4d598a9c26dd",
                "_layers.9.0.mlp.experts.0.up_gate_proj.weight": "c67a6ae2e6676f013040e9b97b160b87",
                "_layers.9.0.mlp.experts.0.down_proj.weight": "9766db124ab5c37113ae910ee893cd1b",
                "_layers.9.0.mlp.experts.1.up_gate_proj.weight": "99d6d71beed4a6a5e097daf6783269e5",
                "_layers.9.0.mlp.experts.1.down_proj.weight": "0cba801946c26bc57bbd7556d09b88b5",
                "_layers.9.0.mlp.experts.2.up_gate_proj.weight": "ff0161994b3e3dc7ec0e1b212b9923c5",
                "_layers.9.0.mlp.experts.2.down_proj.weight": "2b7c62bb7d05f4d65c76f25eb9fc23d3",
                "_layers.9.0.mlp.experts.3.up_gate_proj.weight": "b4244672d6a8d43669d926599585884a",
                "_layers.9.0.mlp.experts.3.down_proj.weight": "d91d656ed09204ddcfb090ee04599b48",
                "_layers.9.0.mlp.shared_experts.up_gate_proj.weight": "16bc652ebe54f5ac1ad13f5f14b36fe6",
                "_layers.9.0.mlp.shared_experts.down_proj.weight": "a3d9cc8d041d6d05dcaa4198e1de76ad",
                "_layers.9.1.input_layernorm.weight": "12e8413477bfb4e2f65d9888120a3171",
                "_layers.9.1.self_attn.o_proj.weight": "d4d8e76fa4c1ae72ea97406e67fabb20",
                "_layers.9.1.self_attn.qkv_proj.weight": "e6aff30c54b606f5fe046418774c2804",
                "_layers.9.1.self_attn.q_norm.weight": "b22a143f8889a9dc5509a1975db687c7",
                "_layers.9.1.self_attn.k_norm.weight": "018d67f53ee93c21cb5a3c95709877dc",
                "_layers.9.1.post_attention_layernorm.weight": "2f0ddf38c1fc77226f286befe56a106b",
                "_layers.9.1.mlp.gate.weight": "1ece9c05c781757fe0bd4905cb1a505b",
                "_layers.9.1.mlp.experts.0.up_gate_proj.weight": "cc0905f7729c09a311517858f309b082",
                "_layers.9.1.mlp.experts.0.down_proj.weight": "78f09393bd69515cdb2e79b494d5a954",
                "_layers.9.1.mlp.experts.1.up_gate_proj.weight": "5dd6b02dc579e713352dcc607323a068",
                "_layers.9.1.mlp.experts.1.down_proj.weight": "4ecb36287b0119b6ca6af7e4030f1fc3",
                "_layers.9.1.mlp.experts.2.up_gate_proj.weight": "cfa8c26637fde6a4ed796c96dd0973a1",
                "_layers.9.1.mlp.experts.2.down_proj.weight": "1699c207c9b9b1cec50141b479a56970",
                "_layers.9.1.mlp.experts.3.up_gate_proj.weight": "933c6e4711fa76cda9e70be8a370a95d",
                "_layers.9.1.mlp.experts.3.down_proj.weight": "610b9013eeeab90aa5cbd7e27108e0fa",
                "_layers.9.1.mlp.shared_experts.up_gate_proj.weight": "8af6be4d1aec9c1b8c2add665a944aca",
                "_layers.9.1.mlp.shared_experts.down_proj.weight": "1bf89913c4ddf0736f34ad28977a8cde",
            }

            for name, param in overlap_gpt_model.named_parameters():
                assert param.grad._md5sum() == baseline[name]


if __name__ == "__main__":
    unittest.main()
