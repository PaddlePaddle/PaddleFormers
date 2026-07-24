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
MTP_DEGREE = 3
# skip test for paddle pr 79368 merge
REPO_FLAG = os.getenv("repo_flag")
SKIP_TESTS = REPO_FLAG != "paddlefleet"


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


def judge_h_subtype():
    """Distinguish H800 vs H20 within the Hopper ("H") family."""
    name = "".join(get_gpu_models_via_nvidia_smi()).upper()
    if "H800" in name:
        return "H800"
    if "H20" in name:
        return "H20"
    return None


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


class TestPP(unittest.TestCase):
    def setUp(self):
        self.seed = 46
        self.batch_size = 12
        self.seq_len = 128
        self.vocab_size = 1024

    def test_pp(self):
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
            gated_linear_unit=True,
            bias_activation_fusion=True,
            num_nextn_predict_layers=MTP_DEGREE,
        )

        overlap_loss, overlap_gpt_model = run_pp(
            self.seed,
            self.batch_size,
            self.seq_len,
            self.vocab_size,
            config,
        )

        print("PP loss MD5:", overlap_loss._md5sum())

        rst = {}
        for name, param in overlap_gpt_model.named_parameters():
            if param.grad is not None:
                rst[name] = param.grad._md5sum()

        pp = pprint.PrettyPrinter(depth=None, width=200, compact=False)
        pp.pprint(rst)

        if judge_machine_type() == "H":
            actual_md5 = overlap_loss._md5sum()
            if judge_h_subtype() == "H800":
                expected_md5 = "d0cc18f8919d2968ac0ad5d577650d39"
            else:
                expected_md5 = "e5fdb6c3bc189ea3e4f2235f0e73353d"
            print(
                f"PP loss MD5 - Actual: {actual_md5}, Expected: {expected_md5}"
            )
            assert actual_md5 == expected_md5, (
                f"PP loss MD5 mismatch! Actual: {actual_md5}, Expected: {expected_md5}"
            )
            if paddle.distributed.get_rank() == 0:
                if judge_h_subtype() == "H800":
                    baseline = {
                        "_layers.shared_layers.embed.embedding.embed_tokens.weight": "60e28736b728cd43e8ced6e14129dec0",
                        "_layers.9.0.input_layernorm.weight": "e253a133c9ab652d77e2e640096ccdfa",
                        "_layers.9.0.self_attn.o_proj.weight": "8153acd0cfe6b3e7a4d8c4ca2a2bf658",
                        "_layers.9.0.self_attn.qkv_proj.weight": "fbff0aa3575ff970bcd35ddce4e51122",
                        "_layers.9.0.self_attn.q_norm.weight": "928e8002da650e48ae5f091087785667",
                        "_layers.9.0.self_attn.k_norm.weight": "10165e00c2567c1fe1f5e3a8078ab1c0",
                        "_layers.9.0.post_attention_layernorm.weight": "d9b7700033d8da3198f163c92ec0bed9",
                        "_layers.9.0.mlp.gate.weight": "7e5190c81644416bdc2689c64e102489",
                        "_layers.9.0.mlp.experts.0.up_gate_proj.weight": "aa3fe661b26badc953d9aa6066ea145c",
                        "_layers.9.0.mlp.experts.0.down_proj.weight": "91efad2ebb3e49c0e1ceaf28347bb50b",
                        "_layers.9.0.mlp.experts.1.up_gate_proj.weight": "2c2c96ce461aa28223280e749f30019b",
                        "_layers.9.0.mlp.experts.1.down_proj.weight": "e9c4f6b3e53de5f06a1eacc1a3a4dcfe",
                        "_layers.9.0.mlp.experts.2.up_gate_proj.weight": "877816e7b626d34d34f080f10b415023",
                        "_layers.9.0.mlp.experts.2.down_proj.weight": "1990ffedad8cf1acbc7f4b68a86ee379",
                        "_layers.9.0.mlp.experts.3.up_gate_proj.weight": "da2c711d3b55c5da0c40ecc4b49334a1",
                        "_layers.9.0.mlp.experts.3.down_proj.weight": "75364c05636e1d773d902fe73291254d",
                        "_layers.9.0.mlp.shared_experts.up_gate_proj.weight": "067473ff9dfcdecbcf949942a3caeba9",
                        "_layers.9.0.mlp.shared_experts.down_proj.weight": "9f47f96c26d81b1a6647d553b3b3af3f",
                        "_layers.9.1.input_layernorm.weight": "5a0ec38bbfa92c17ddf5dcb5cf4ff2ad",
                        "_layers.9.1.self_attn.o_proj.weight": "85f4cece84fb8cd7616538aee2acd57f",
                        "_layers.9.1.self_attn.qkv_proj.weight": "9c9123fd23bce9bba931b971b2eedb21",
                        "_layers.9.1.self_attn.q_norm.weight": "5ed69f04dfd86bbbf91cdfeac650b52d",
                        "_layers.9.1.self_attn.k_norm.weight": "00dc120fab3d152b284730d9b6efe8c2",
                        "_layers.9.1.post_attention_layernorm.weight": "0967ad75c77c929226e2bc1d39ce0986",
                        "_layers.9.1.mlp.gate.weight": "bf09882de3516988b785ab1ec9c65882",
                        "_layers.9.1.mlp.experts.0.up_gate_proj.weight": "1f520ba660871b094d2fb313f3568696",
                        "_layers.9.1.mlp.experts.0.down_proj.weight": "40a7ed164e23b0b62ee32dc858396212",
                        "_layers.9.1.mlp.experts.1.up_gate_proj.weight": "c2434f9506de466dab2e756678e77012",
                        "_layers.9.1.mlp.experts.1.down_proj.weight": "7dece11d41ee3ec2aa7abf287f6c7d60",
                        "_layers.9.1.mlp.experts.2.up_gate_proj.weight": "ea0ab48ef0723398fbc0743c6240772c",
                        "_layers.9.1.mlp.experts.2.down_proj.weight": "15a5fc94657b70269a9666ba85108120",
                        "_layers.9.1.mlp.experts.3.up_gate_proj.weight": "73020700f0284cc6e15a65462bafd2c5",
                        "_layers.9.1.mlp.experts.3.down_proj.weight": "bf038050b26286ea04fd86f811f13495",
                        "_layers.9.1.mlp.shared_experts.up_gate_proj.weight": "d772b98cd764a61c73966e7110edb5c9",
                        "_layers.9.1.mlp.shared_experts.down_proj.weight": "2c39784fd16c4a4dd700eb7848dc35fb",
                    }
                else:
                    baseline = {
                        "_layers.shared_layers.embed.embedding.embed_tokens.weight": "d1a6d98dcafbc3cb7a410eda68203209",
                        "_layers.9.0.input_layernorm.weight": "b191e4cfadc1463d18a842a0960aa72a",
                        "_layers.9.0.self_attn.o_proj.weight": "b0debc03067240ab78bf9e109c9216e8",
                        "_layers.9.0.self_attn.qkv_proj.weight": "57a9639ee0e5bd67a79fe1c23cf98135",
                        "_layers.9.0.self_attn.q_norm.weight": "1159364a7c7f9a5d452836e549cf170f",
                        "_layers.9.0.self_attn.k_norm.weight": "90f0a4d30437c218d875edaaaa8d0e67",
                        "_layers.9.0.post_attention_layernorm.weight": "d1dc257977317664685055f0c719e518",
                        "_layers.9.0.mlp.gate.weight": "47ffadddff01926948bb6d85282bb7a8",
                        "_layers.9.0.mlp.experts.0.up_gate_proj.weight": "abd579ad546b6df8fd865aec5e1472c2",
                        "_layers.9.0.mlp.experts.0.down_proj.weight": "8e32222a50cdba3f0736137a97091502",
                        "_layers.9.0.mlp.experts.1.up_gate_proj.weight": "ba9b869172ac0d5bdee972ff5461177b",
                        "_layers.9.0.mlp.experts.1.down_proj.weight": "f83ba2346cd26050b52498837fba06c3",
                        "_layers.9.0.mlp.experts.2.up_gate_proj.weight": "fe2186ec15ce89a37f77ba79d1b6d060",
                        "_layers.9.0.mlp.experts.2.down_proj.weight": "a98852eebdfadaa07ff75d5f6ba93e66",
                        "_layers.9.0.mlp.experts.3.up_gate_proj.weight": "ddceeb4eaf751e46bc8675d98b9d9534",
                        "_layers.9.0.mlp.experts.3.down_proj.weight": "dfabd23157dc871d69f4fc77817f91d3",
                        "_layers.9.0.mlp.shared_experts.up_gate_proj.weight": "17070eaffb13fea2d2c783295bb07c87",
                        "_layers.9.0.mlp.shared_experts.down_proj.weight": "a56b1432b2b8b091f5e7424a5b0d6984",
                        "_layers.9.1.input_layernorm.weight": "7edfbd881fe6e03e838fec63a91b0a00",
                        "_layers.9.1.self_attn.o_proj.weight": "1dcee2eeb395de72094066857a287872",
                        "_layers.9.1.self_attn.qkv_proj.weight": "47a0fda2bf7545afda9eecf6cbb2291b",
                        "_layers.9.1.self_attn.q_norm.weight": "abc2db4cac135dca12f949287b4a2a03",
                        "_layers.9.1.self_attn.k_norm.weight": "7d91efc9b561a5db6bbd329899898785",
                        "_layers.9.1.post_attention_layernorm.weight": "b58ded0b76526edfd21eaa788b37a8f1",
                        "_layers.9.1.mlp.gate.weight": "098b7e7de24b58db5c10520338a0a42b",
                        "_layers.9.1.mlp.experts.0.up_gate_proj.weight": "8de2ebb74b7c372227a21aea872ef418",
                        "_layers.9.1.mlp.experts.0.down_proj.weight": "620c622e7714559616be92a6bac5abb6",
                        "_layers.9.1.mlp.experts.1.up_gate_proj.weight": "c7377cacf94ab8f529e2a5fab633793d",
                        "_layers.9.1.mlp.experts.1.down_proj.weight": "0f75a945522a9026cbb4367bfe104e89",
                        "_layers.9.1.mlp.experts.2.up_gate_proj.weight": "b5cfa9d6c8febd618f91ac2843d50a1c",
                        "_layers.9.1.mlp.experts.2.down_proj.weight": "b2d1236c286a3c0704224fe4105eca49",
                        "_layers.9.1.mlp.experts.3.up_gate_proj.weight": "4fbaccfd91b19032984266ce2ec6c3e6",
                        "_layers.9.1.mlp.experts.3.down_proj.weight": "b46343a2c024653012af3cbdc958f71d",
                        "_layers.9.1.mlp.shared_experts.up_gate_proj.weight": "74a2298e2599f8c12fa13ddd3885b9d8",
                        "_layers.9.1.mlp.shared_experts.down_proj.weight": "72008570bb40b490c5e0954e1d51145b",
                    }
                for name, param in overlap_gpt_model.named_parameters():
                    assert param.grad._md5sum() == baseline[name], (
                        f"{name}'s grad has diff"
                    )
        elif judge_machine_type() == "B":
            assert overlap_loss._md5sum() == "1674e8e0ada05d64d7b90181cf140db0"
            if paddle.distributed.get_rank() == 0:
                baseline = {
                    "_layers.shared_layers.embed.embedding.embed_tokens.weight": "7c630094b5660eb636646f778369cdee",
                    "_layers.9.0.input_layernorm.weight": "fa36647c97c191256150157380cd6509",
                    "_layers.9.0.self_attn.o_proj.weight": "c38f6a1397f412429dc6a8f05ee9972a",
                    "_layers.9.0.self_attn.qkv_proj.weight": "b3a8f4451744dd97ed2658f93229925f",
                    "_layers.9.0.self_attn.q_norm.weight": "f16a8f03ef21cd2902c6d1aedf65dea8",
                    "_layers.9.0.self_attn.k_norm.weight": "00877c5f9c7dc10ee080b6f9a91655d2",
                    "_layers.9.0.post_attention_layernorm.weight": "aef505d1faf64f9eac8d4a25e59df94b",
                    "_layers.9.0.mlp.gate.weight": "47fd7eb38d41f46d94281b50d31e7fde",
                    "_layers.9.0.mlp.experts.0.up_gate_proj.weight": "17593839bb9781798777f79a1648e3b0",
                    "_layers.9.0.mlp.experts.0.down_proj.weight": "6de5d9ac75185dcaa1bd34e3520b1d16",
                    "_layers.9.0.mlp.experts.1.up_gate_proj.weight": "1f9cf3f2b4f3ff19943a4bb4a2164c1f",
                    "_layers.9.0.mlp.experts.1.down_proj.weight": "64b8bbd90c707b07f9bd9879472e20ae",
                    "_layers.9.0.mlp.experts.2.up_gate_proj.weight": "2eccf992ed71e5ae2c0b245ddd822b8d",
                    "_layers.9.0.mlp.experts.2.down_proj.weight": "9a8b1e7fd25023228fad6109358045e9",
                    "_layers.9.0.mlp.experts.3.up_gate_proj.weight": "bfd40aa68434ad782342a557a9dfe4cb",
                    "_layers.9.0.mlp.experts.3.down_proj.weight": "12dfe1fb29676788f61c1ea36501fcf7",
                    "_layers.9.0.mlp.shared_experts.up_gate_proj.weight": "e2f047ea2496f779f1ab8086ded3e49b",
                    "_layers.9.0.mlp.shared_experts.down_proj.weight": "fd1d66b48648ff09bc94b6ea73fbbd74",
                    "_layers.9.1.input_layernorm.weight": "6599fbf9d9398910631991bc94029209",
                    "_layers.9.1.self_attn.o_proj.weight": "28e3f0544f8bf63944f3c1581b4f8d80",
                    "_layers.9.1.self_attn.qkv_proj.weight": "b4d5f4ddf643cd052afa34e066cdf795",
                    "_layers.9.1.self_attn.q_norm.weight": "d9d8e32f6446816878c56a6861bf643d",
                    "_layers.9.1.self_attn.k_norm.weight": "077cdabd7c06ca37c8513009c66fe373",
                    "_layers.9.1.post_attention_layernorm.weight": "2873dc65f5ca2695aee0a48786906eb6",
                    "_layers.9.1.mlp.gate.weight": "0ae845f0a3d7bd3e6eb1f4846c8c01e1",
                    "_layers.9.1.mlp.experts.0.up_gate_proj.weight": "14b49f34568acdab8ffc8fe22c9a1597",
                    "_layers.9.1.mlp.experts.0.down_proj.weight": "d14faab6e743fac3c2df44ef7e42121d",
                    "_layers.9.1.mlp.experts.1.up_gate_proj.weight": "902dade986c4e2af896d3d2a3ba57cc7",
                    "_layers.9.1.mlp.experts.1.down_proj.weight": "658ddbe6ca3adda82dee064916540ff2",
                    "_layers.9.1.mlp.experts.2.up_gate_proj.weight": "21a3da1c9dc44c2494129e5c15d46374",
                    "_layers.9.1.mlp.experts.2.down_proj.weight": "946f4f30eafa2ea5bd05b56eb5d7a8b0",
                    "_layers.9.1.mlp.experts.3.up_gate_proj.weight": "70ad05902bdbdd5bdacbcd554cfab7c0",
                    "_layers.9.1.mlp.experts.3.down_proj.weight": "33fd4ff39a81bc9de8e0dc61d56436ee",
                    "_layers.9.1.mlp.shared_experts.up_gate_proj.weight": "512196041bb29930952543d48e4847e0",
                    "_layers.9.1.mlp.shared_experts.down_proj.weight": "b5b58e8860d1f3d6abc5d1601fe4b170",
                }
                mismatches = {}
                actual_all = {}
                for name, param in overlap_gpt_model.named_parameters():
                    if param.grad is None:
                        continue
                    actual_md5 = param.grad._md5sum()
                    actual_all[name] = actual_md5
                    expected = baseline.get(name)
                    if expected != actual_md5:
                        mismatches[name] = {
                            "actual": actual_md5,
                            "expected": expected,
                        }

                if mismatches:
                    print("===== MISMATCHED KEYS =====")
                    pp = pprint.PrettyPrinter(
                        depth=None, width=200, compact=False
                    )
                    pp.pprint(mismatches)

                    print("===== FULL ACTUAL DICT =====")
                    pp.pprint(actual_all)

                assert not mismatches, (
                    f"{len(mismatches)} param(s) grad mismatch: {list(mismatches.keys())}"
                )


if __name__ == "__main__":
    unittest.main()
