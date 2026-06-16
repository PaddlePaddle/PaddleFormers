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
import pprint
import random
import subprocess
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


def get_gpu_models_via_nvidia_smi():
    try:
        output = subprocess.check_output(
            "nvidia-smi --query-gpu=name --format=csv,noheader", shell=True
        )
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
        elif "B" in name:
            return "B"


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

        overlap_loss, overlap_gpt_model = run_pp(
            self.seed,
            self.batch_size,
            self.seq_len,
            self.vocab_size,
            config,
            forward_backward_overlap_scheduler=False,
        )

        print("Overlap PP loss MD5:", overlap_loss._md5sum())
        rst = {}
        for name, param in overlap_gpt_model.named_parameters():
            if param.grad is not None:
                rst[name] = param.grad._md5sum()

        pp = pprint.PrettyPrinter(depth=None, width=200, compact=False)
        pp.pprint(rst)

        if judge_machine_type() == "H":
            actual_md5 = overlap_loss._md5sum()
            expected_md5 = "ba38c67745e4702582cf8b0004198aea"
            print(
                f"Overlap PP loss MD5 - Actual: {actual_md5}, Expected: {expected_md5}"
            )
            assert actual_md5 == expected_md5, (
                f"Overlap PP loss MD5 mismatch! Actual: {actual_md5}, Expected: {expected_md5}"
            )
            if paddle.distributed.get_rank() == 0:
                baseline = {
                    "_layers.9.0.input_layernorm.weight": "25a10e393e9c0d10015ba8a580f51579",
                    "_layers.9.0.mlp.experts.0.down_proj.weight": "e6326f1585fc7e75249d17a6869e3985",
                    "_layers.9.0.mlp.experts.0.up_gate_proj.weight": "64d7f898a1c357e76501bb1251218dfd",
                    "_layers.9.0.mlp.experts.1.down_proj.weight": "c3980dce958a06d2cd238e040e5bb97a",
                    "_layers.9.0.mlp.experts.1.up_gate_proj.weight": "96359efd26ea849525ce8940056ddb20",
                    "_layers.9.0.mlp.experts.2.down_proj.weight": "9f8a97bf23d653d4839301eab3d68017",
                    "_layers.9.0.mlp.experts.2.up_gate_proj.weight": "f7b4c3a3e3a7c98cab2c4a2acaa5efa3",
                    "_layers.9.0.mlp.experts.3.down_proj.weight": "a60c2d8f7f7706ddea3ba97b99af2c84",
                    "_layers.9.0.mlp.experts.3.up_gate_proj.weight": "0608ce0aca450109c9595f6c196d2d42",
                    "_layers.9.0.mlp.gate.weight": "4388162779c38417bc50715868270cce",
                    "_layers.9.0.mlp.shared_experts.down_proj.weight": "0eb9e5be3668b0db5b7176600ffbef06",
                    "_layers.9.0.mlp.shared_experts.up_gate_proj.weight": "6b94c2c2398f0752c6a80c63bccd274d",
                    "_layers.9.0.post_attention_layernorm.weight": "f808b23d4f82c46ceaa0662dc5ee2bb3",
                    "_layers.9.0.self_attn.k_norm.weight": "f013497b2eb6fc2fb56cf6385a8f9773",
                    "_layers.9.0.self_attn.o_proj.weight": "59131270f8b3ea003ca9ed7e875a5f57",
                    "_layers.9.0.self_attn.q_norm.weight": "a9277911431d2dfa74e01a2ba4ea8c90",
                    "_layers.9.0.self_attn.qkv_proj.weight": "0346ddfcc174671b31f7d5ae9459a44a",
                    "_layers.9.1.input_layernorm.weight": "dca7b162724df7cbbe42285ebee69279",
                    "_layers.9.1.mlp.experts.0.down_proj.weight": "a5ed075d74800dd45bfc4034c4872252",
                    "_layers.9.1.mlp.experts.0.up_gate_proj.weight": "d23b13aae3d95ec7e33ae4967c2b81cd",
                    "_layers.9.1.mlp.experts.1.down_proj.weight": "5b26f4a75588f65ce436b5e98bdaf377",
                    "_layers.9.1.mlp.experts.1.up_gate_proj.weight": "e3d8f92858750d62045d2c742a167db4",
                    "_layers.9.1.mlp.experts.2.down_proj.weight": "b2d1236c286a3c0704224fe4105eca49",
                    "_layers.9.1.mlp.experts.2.up_gate_proj.weight": "b5cfa9d6c8febd618f91ac2843d50a1c",
                    "_layers.9.1.mlp.experts.3.down_proj.weight": "c6e7e74e8b6414ec4df44eda74009772",
                    "_layers.9.1.mlp.experts.3.up_gate_proj.weight": "d9bcffaf1c65227503cec20915edb7ff",
                    "_layers.9.1.mlp.gate.weight": "a35047490b2391bb22138da0f70dadda",
                    "_layers.9.1.mlp.shared_experts.down_proj.weight": "bc25a8199b6914ea8c63acdf0b4e9bc2",
                    "_layers.9.1.mlp.shared_experts.up_gate_proj.weight": "65f8940ff749511190ba37c7ffae24ac",
                    "_layers.9.1.post_attention_layernorm.weight": "060f65020aa4a19c05a6566a281e8990",
                    "_layers.9.1.self_attn.k_norm.weight": "852233e44272c02b790f23855a2c5e3a",
                    "_layers.9.1.self_attn.o_proj.weight": "48adbd179fff65632e5a8eff6bcf6a75",
                    "_layers.9.1.self_attn.q_norm.weight": "cdf8ed164f46b098596204751d73b628",
                    "_layers.9.1.self_attn.qkv_proj.weight": "1797bbd662fbd376f3707386e574b76b",
                    "_layers.shared_layers.embed.embedding.embed_tokens.weight": "d3908eaccde26a79276ea1cf6e8bba1f",
                }
                for name, param in overlap_gpt_model.named_parameters():
                    assert param.grad._md5sum() == baseline[name], (
                        f"{name}'s grad has diff"
                    )
        elif judge_machine_type() == "B":
            assert overlap_loss._md5sum() == "cc2a9b0deaf25a56cc571465947c756a"
            if paddle.distributed.get_rank() == 0:
                baseline = {
                    "_layers.9.0.input_layernorm.weight": "7ca2f0604d4e0ca16162e0c99b673ce6",
                    "_layers.9.0.mlp.experts.0.down_proj.weight": "6709fdcfd21412de3cbbb986c8285a24",
                    "_layers.9.0.mlp.experts.0.up_gate_proj.weight": "84f7c58d939de6db2638237c6cdd5ca9",
                    "_layers.9.0.mlp.experts.1.down_proj.weight": "425cf09baee21f36506e37abdf3c153e",
                    "_layers.9.0.mlp.experts.1.up_gate_proj.weight": "122c26f8d82e064da34c1e8a3042d8b9",
                    "_layers.9.0.mlp.experts.2.down_proj.weight": "e367d0f0e9d790a56a0d5d902d3416bc",
                    "_layers.9.0.mlp.experts.2.up_gate_proj.weight": "6d2752e6cad61d7c4b27fee96fd1fa6b",
                    "_layers.9.0.mlp.experts.3.down_proj.weight": "1e3048b0d99f8078cdef1c28fc9be64d",
                    "_layers.9.0.mlp.experts.3.up_gate_proj.weight": "996c9363c8195c1815d8cf1ef04c654f",
                    "_layers.9.0.mlp.gate.weight": "0e317f2f2f3863d39f469dfaec84e5ea",
                    "_layers.9.0.mlp.shared_experts.down_proj.weight": "d4237a2ef05292fd5d848cdcfb313cd0",
                    "_layers.9.0.mlp.shared_experts.up_gate_proj.weight": "8f557c64d415117315e49fc3e881a7a8",
                    "_layers.9.0.post_attention_layernorm.weight": "c16680edcc3c601065d6b7ba6e59b886",
                    "_layers.9.0.self_attn.k_norm.weight": "ddd46106cd27497d61911d60e6ae3478",
                    "_layers.9.0.self_attn.o_proj.weight": "82363d65c8b27bdcd4b384e8827a5f93",
                    "_layers.9.0.self_attn.q_norm.weight": "5c0101ce08007baa1da7170b4ccccf8c",
                    "_layers.9.0.self_attn.qkv_proj.weight": "47bff2035d7580dac1c1ee0999172c4e",
                    "_layers.9.1.input_layernorm.weight": "90a40c97aa17ff1e71f7bd150b5c3d47",
                    "_layers.9.1.mlp.experts.0.down_proj.weight": "5084162f68d9dd375363b146d2384aa5",
                    "_layers.9.1.mlp.experts.0.up_gate_proj.weight": "26498a742537ccc7384baf7f7bfca68d",
                    "_layers.9.1.mlp.experts.1.down_proj.weight": "9e50ca48e7d835a06a86cb94ab5976b2",
                    "_layers.9.1.mlp.experts.1.up_gate_proj.weight": "ede0cb32885a8231bf3332fba5b83b24",
                    "_layers.9.1.mlp.experts.2.down_proj.weight": "b2d1236c286a3c0704224fe4105eca49",
                    "_layers.9.1.mlp.experts.2.up_gate_proj.weight": "b5cfa9d6c8febd618f91ac2843d50a1c",
                    "_layers.9.1.mlp.experts.3.down_proj.weight": "a844e2467d9162af4fc2201117105ce3",
                    "_layers.9.1.mlp.experts.3.up_gate_proj.weight": "397f7405d7f804425d378727286c3703",
                    "_layers.9.1.mlp.gate.weight": "691109eda4405398feb4327be55e1c3a",
                    "_layers.9.1.mlp.shared_experts.down_proj.weight": "747d720a6902a8609cf360c44bcb9a81",
                    "_layers.9.1.mlp.shared_experts.up_gate_proj.weight": "32ce592c21077650a9732e662334a796",
                    "_layers.9.1.post_attention_layernorm.weight": "be6685ec6294d76fc03661ef0f2ee707",
                    "_layers.9.1.self_attn.k_norm.weight": "a4da316c1d98ed68c4460fc78d8a8a68",
                    "_layers.9.1.self_attn.o_proj.weight": "265d27d2a5b58cd0b0fb7de110a30bff",
                    "_layers.9.1.self_attn.q_norm.weight": "06e3ba5983a169244127b1598b0a1a87",
                    "_layers.9.1.self_attn.qkv_proj.weight": "600a875882254ed3211f81364e354ee1",
                    "_layers.shared_layers.embed.embedding.embed_tokens.weight": "846d77005712ae32512bcedfbcce9e94",
                }
                for name, param in overlap_gpt_model.named_parameters():
                    assert param.grad._md5sum() == baseline[name], (
                        f"{name}'s grad has diff"
                    )


if __name__ == "__main__":
    unittest.main()
