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
            if judge_h_subtype() == "H800":
                expected_md5 = "ac1c324951d04405f159fe60a1b02f77"
            else:
                expected_md5 = "0437752ae4c2b700b97c9249e6ca5dc3"
            print(
                f"Overlap PP loss MD5 - Actual: {actual_md5}, Expected: {expected_md5}"
            )
            assert actual_md5 == expected_md5, (
                f"Overlap PP loss MD5 mismatch! Actual: {actual_md5}, Expected: {expected_md5}"
            )
            if paddle.distributed.get_rank() == 0:
                if judge_h_subtype() == "H800":
                    baseline = {
                        "_layers.9.0.input_layernorm.weight": "0aeebb3b5ac42c1faadb299fb398a396",
                        "_layers.9.0.mlp.experts.0.down_proj.weight": "b2d1236c286a3c0704224fe4105eca49",
                        "_layers.9.0.mlp.experts.0.up_gate_proj.weight": "b5cfa9d6c8febd618f91ac2843d50a1c",
                        "_layers.9.0.mlp.experts.1.down_proj.weight": "bb2d39b853c6cde7efc6affda92e6970",
                        "_layers.9.0.mlp.experts.1.up_gate_proj.weight": "6c54f1ed3be5b09ec747f30f1c291ec1",
                        "_layers.9.0.mlp.experts.2.down_proj.weight": "46b72fbb4e114b757fe20cd8aeeed345",
                        "_layers.9.0.mlp.experts.2.up_gate_proj.weight": "cdcc7000f0ec04f286387d3250ac7cbd",
                        "_layers.9.0.mlp.experts.3.down_proj.weight": "597ccdee5c1c51a9185c4532c151f36e",
                        "_layers.9.0.mlp.experts.3.up_gate_proj.weight": "9d7a7ca9763ca2b4e367b57f6430fd7c",
                        "_layers.9.0.mlp.gate.weight": "8186b2b41857c3eabb57900cf6a7ceb5",
                        "_layers.9.0.mlp.shared_experts.down_proj.weight": "6d218d68bcb6dde5cbb1b3da6e6341f4",
                        "_layers.9.0.mlp.shared_experts.up_gate_proj.weight": "506abb1bc4d2136cfe932c56df3838b3",
                        "_layers.9.0.post_attention_layernorm.weight": "56ae029b77d9d95b6d7632d1c9a94f6d",
                        "_layers.9.0.self_attn.k_norm.weight": "4cca721a4ad6e6d5055193924c0e238e",
                        "_layers.9.0.self_attn.o_proj.weight": "563d48636a28f11ddf0f9a8640eb8731",
                        "_layers.9.0.self_attn.q_norm.weight": "e42400164ec518f2e859474a5fa6918a",
                        "_layers.9.0.self_attn.qkv_proj.weight": "6a14886ed069063fee422d0d78c31ed7",
                        "_layers.9.1.input_layernorm.weight": "cf4b7d64e2bd7e5a446bdb066038a979",
                        "_layers.9.1.mlp.experts.0.down_proj.weight": "a3d6ca99029a542ba95b965a18f62e61",
                        "_layers.9.1.mlp.experts.0.up_gate_proj.weight": "50f974c4a34d885b6ca471a5ace6b72d",
                        "_layers.9.1.mlp.experts.1.down_proj.weight": "587dd9fb61c30338fa396d8b95492c75",
                        "_layers.9.1.mlp.experts.1.up_gate_proj.weight": "4328c6ab3666d39bb00f049491de420c",
                        "_layers.9.1.mlp.experts.2.down_proj.weight": "3e5835a883f4cc758b76df5586208cd0",
                        "_layers.9.1.mlp.experts.2.up_gate_proj.weight": "98f6954c2e8b52bbc1fddcca23252de9",
                        "_layers.9.1.mlp.experts.3.down_proj.weight": "b821d3e53f5af013480662fe797c798b",
                        "_layers.9.1.mlp.experts.3.up_gate_proj.weight": "1f25c9dc954557d42d0c53a0813134f3",
                        "_layers.9.1.mlp.gate.weight": "10c7b77099afa415214b95cbf175a529",
                        "_layers.9.1.mlp.shared_experts.down_proj.weight": "7806ce29cc56e200c5c12bcc193bf2ea",
                        "_layers.9.1.mlp.shared_experts.up_gate_proj.weight": "5193abcc40d069213862054e86109972",
                        "_layers.9.1.post_attention_layernorm.weight": "5d1bff4c423a8e164169cb8960e9e332",
                        "_layers.9.1.self_attn.k_norm.weight": "6c72294d51d9a9327a681b8b9762246e",
                        "_layers.9.1.self_attn.o_proj.weight": "4df0fe86b348813e18f5b4be879b1e5e",
                        "_layers.9.1.self_attn.q_norm.weight": "bcdff6ad6a46be0452ac20ccafa6b469",
                        "_layers.9.1.self_attn.qkv_proj.weight": "4950b65afdbd50308dc190c013635586",
                        "_layers.shared_layers.embed.embedding.embed_tokens.weight": "bcdb228f9924e079397e73a58a2ce638",
                    }
                else:
                    baseline = {
                        "_layers.9.0.input_layernorm.weight": "e32c9a6da891010d14f71ade19261ff3",
                        "_layers.9.0.mlp.experts.0.down_proj.weight": "e9e37f3c3edc6707bd98736409290ef6",
                        "_layers.9.0.mlp.experts.0.up_gate_proj.weight": "e2d72ca6e2d183110a4df56564b27344",
                        "_layers.9.0.mlp.experts.1.down_proj.weight": "b1f144a17d09120440339c65e8af4127",
                        "_layers.9.0.mlp.experts.1.up_gate_proj.weight": "317e36ecfc18ac450a36c2880949767c",
                        "_layers.9.0.mlp.experts.2.down_proj.weight": "9f8a97bf23d653d4839301eab3d68017",
                        "_layers.9.0.mlp.experts.2.up_gate_proj.weight": "f7b4c3a3e3a7c98cab2c4a2acaa5efa3",
                        "_layers.9.0.mlp.experts.3.down_proj.weight": "1e8bf78ef0a1a26b7e733618d143803e",
                        "_layers.9.0.mlp.experts.3.up_gate_proj.weight": "68d31e8b2e87f214966f89095f1f70e9",
                        "_layers.9.0.mlp.gate.weight": "7a0f39e75f1613b06c18cbac2bb07435",
                        "_layers.9.0.mlp.shared_experts.down_proj.weight": "c0fcaaa402c5054f1c6df2626a116c0f",
                        "_layers.9.0.mlp.shared_experts.up_gate_proj.weight": "52c73b1656d6811fc26dfe3b5c74e297",
                        "_layers.9.0.post_attention_layernorm.weight": "e7c67654dc593e906aaeda1219b13b32",
                        "_layers.9.0.self_attn.k_norm.weight": "4df674778242d249945aeb2e5e2b7217",
                        "_layers.9.0.self_attn.o_proj.weight": "406917da377ac7330486542e696121c0",
                        "_layers.9.0.self_attn.q_norm.weight": "bfa440c62525cd8855adb7345594e72b",
                        "_layers.9.0.self_attn.qkv_proj.weight": "d1975db2bc69bf0d9c491a2d1b553fef",
                        "_layers.9.1.input_layernorm.weight": "199af42a4dec5265a2482fed05287877",
                        "_layers.9.1.mlp.experts.0.down_proj.weight": "06c2fde5fcf366735e130d3ca0f77202",
                        "_layers.9.1.mlp.experts.0.up_gate_proj.weight": "8bd6a62b277dd3113bf8563376cded02",
                        "_layers.9.1.mlp.experts.1.down_proj.weight": "eced23221ec0c668ed619c1e8c65512c",
                        "_layers.9.1.mlp.experts.1.up_gate_proj.weight": "a500d5e7cba1a74062a380fedb7c57c2",
                        "_layers.9.1.mlp.experts.2.down_proj.weight": "b2d1236c286a3c0704224fe4105eca49",
                        "_layers.9.1.mlp.experts.2.up_gate_proj.weight": "b5cfa9d6c8febd618f91ac2843d50a1c",
                        "_layers.9.1.mlp.experts.3.down_proj.weight": "e6f2e5096c2c60335189b5d39e9a285b",
                        "_layers.9.1.mlp.experts.3.up_gate_proj.weight": "676703fe62d05af10f1b96ffcc60fb89",
                        "_layers.9.1.mlp.gate.weight": "0459d360b4aa8689256908f4240cc70d",
                        "_layers.9.1.mlp.shared_experts.down_proj.weight": "c9e335e571e4d844e6a55931f66e73b6",
                        "_layers.9.1.mlp.shared_experts.up_gate_proj.weight": "df2de38186a65322b417ace2fdd8a573",
                        "_layers.9.1.post_attention_layernorm.weight": "d058a4ea6a7c8221f69ee8ccf1c54ccd",
                        "_layers.9.1.self_attn.k_norm.weight": "73522350067d7e51597a7b2eda6dd92d",
                        "_layers.9.1.self_attn.o_proj.weight": "62326976270aaa769b04351e38e0af7a",
                        "_layers.9.1.self_attn.q_norm.weight": "41b3d83530d63a57c582dafbffba90f4",
                        "_layers.9.1.self_attn.qkv_proj.weight": "50a8e5d0ff69b869e12c8e15c10474e3",
                        "_layers.shared_layers.embed.embedding.embed_tokens.weight": "42e836e61824525bdbe965a4daf06385",
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
                    "_layers.9.0.mlp.gate.weight": "fdc6f038b22df4b16d8ea63354ed05e6",
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
                    "_layers.9.1.mlp.gate.weight": "c2776b157831d7d6829f25dd1b7599c4",
                    "_layers.9.1.mlp.shared_experts.down_proj.weight": "747d720a6902a8609cf360c44bcb9a81",
                    "_layers.9.1.mlp.shared_experts.up_gate_proj.weight": "32ce592c21077650a9732e662334a796",
                    "_layers.9.1.post_attention_layernorm.weight": "be6685ec6294d76fc03661ef0f2ee707",
                    "_layers.9.1.self_attn.k_norm.weight": "a4da316c1d98ed68c4460fc78d8a8a68",
                    "_layers.9.1.self_attn.o_proj.weight": "265d27d2a5b58cd0b0fb7de110a30bff",
                    "_layers.9.1.self_attn.q_norm.weight": "06e3ba5983a169244127b1598b0a1a87",
                    "_layers.9.1.self_attn.qkv_proj.weight": "600a875882254ed3211f81364e354ee1",
                    "_layers.shared_layers.embed.embedding.embed_tokens.weight": "846d77005712ae32512bcedfbcce9e94",
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
