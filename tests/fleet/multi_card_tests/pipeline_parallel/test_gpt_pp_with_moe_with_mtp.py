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
MTP_DEGREE = 3


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
            expected_md5 = "e5fdb6c3bc189ea3e4f2235f0e73353d"
            print(
                f"PP loss MD5 - Actual: {actual_md5}, Expected: {expected_md5}"
            )
            assert actual_md5 == expected_md5, (
                f"PP loss MD5 mismatch! Actual: {actual_md5}, Expected: {expected_md5}"
            )
            if paddle.distributed.get_rank() == 0:
                baseline = {
                    "_layers.shared_layers.embed.embedding.embed_tokens.weight": "235347d268241e782b3cdeb491818326",
                    "_layers.9.0.input_layernorm.weight": "7db8e991113db53389a9a9ca842c0c65",
                    "_layers.9.0.self_attn.o_proj.weight": "3f18521b6eaf7a20bb7404454148f538",
                    "_layers.9.0.self_attn.qkv_proj.weight": "9a899e80b9102e62da30ef9230ef5c89",
                    "_layers.9.0.self_attn.q_norm.weight": "34977b12cbf965c17472cd78a1d961c9",
                    "_layers.9.0.self_attn.k_norm.weight": "c08c1bfb813bb602d2278cef41e98830",
                    "_layers.9.0.post_attention_layernorm.weight": "03831b1aa61a8e9d3e39b761f4075bc7",
                    "_layers.9.0.mlp.gate.weight": "28399a4c07757354559cd1a9fb2d41e5",
                    "_layers.9.0.mlp.experts.0.up_gate_proj.weight": "7531d94ff8941d490b4e2ca5e1204cf8",
                    "_layers.9.0.mlp.experts.0.down_proj.weight": "5d4031bfdb0854a43c1775e623b99e26",
                    "_layers.9.0.mlp.experts.1.up_gate_proj.weight": "76faced9b6cee11d1ff00af28c305e5a",
                    "_layers.9.0.mlp.experts.1.down_proj.weight": "405c43d65b6a0c54e05adc94ccabe6fd",
                    "_layers.9.0.mlp.experts.2.up_gate_proj.weight": "c694b35d3cc7aa9c9cc5d529c6200881",
                    "_layers.9.0.mlp.experts.2.down_proj.weight": "e3f27f0c0dbbc01c55ec19679833bc93",
                    "_layers.9.0.mlp.experts.3.up_gate_proj.weight": "0258a9351b3f93a02773d264a1d23b09",
                    "_layers.9.0.mlp.experts.3.down_proj.weight": "a6b97d8954114a7e5c2bddad19ed21ce",
                    "_layers.9.0.mlp.shared_experts.up_gate_proj.weight": "53d843db60ee1ba3b7d92d40468222e1",
                    "_layers.9.0.mlp.shared_experts.down_proj.weight": "e8629c22f979a0bd9bec36c31eead626",
                    "_layers.9.1.input_layernorm.weight": "acda88995c7a87e86437926558a416b9",
                    "_layers.9.1.self_attn.o_proj.weight": "4ecb9f9e04c98d3631dc4f2973c08ed1",
                    "_layers.9.1.self_attn.qkv_proj.weight": "3175f69967fc47a394200c165716f840",
                    "_layers.9.1.self_attn.q_norm.weight": "ed060896721aa0e80639fe2cfbed0343",
                    "_layers.9.1.self_attn.k_norm.weight": "16a5b6f503d3e970035011dac4cc86f1",
                    "_layers.9.1.post_attention_layernorm.weight": "645992e8bbb7b22686c97d84df1ad895",
                    "_layers.9.1.mlp.gate.weight": "516f1f10115bed1f162c473430d5dbfa",
                    "_layers.9.1.mlp.experts.0.up_gate_proj.weight": "5a26639d7b89a5f4d8733a52f74449e5",
                    "_layers.9.1.mlp.experts.0.down_proj.weight": "e18498896d33cf03d51ba8b227cc06f8",
                    "_layers.9.1.mlp.experts.1.up_gate_proj.weight": "2bcfd342acf23a751b00d01e8fcd9709",
                    "_layers.9.1.mlp.experts.1.down_proj.weight": "9ff972b3eba01a890420293751243309",
                    "_layers.9.1.mlp.experts.2.up_gate_proj.weight": "b5cfa9d6c8febd618f91ac2843d50a1c",
                    "_layers.9.1.mlp.experts.2.down_proj.weight": "b2d1236c286a3c0704224fe4105eca49",
                    "_layers.9.1.mlp.experts.3.up_gate_proj.weight": "a18bebe9e528a9d444181cca5787719c",
                    "_layers.9.1.mlp.experts.3.down_proj.weight": "bbc042016d0d781377c17b77f17903d9",
                    "_layers.9.1.mlp.shared_experts.up_gate_proj.weight": "52e0befaca5f633922bf45b5c64515c5",
                    "_layers.9.1.mlp.shared_experts.down_proj.weight": "8615e16eb658ea9ae102cbc0d8f64d18",
                }
                for name, param in overlap_gpt_model.named_parameters():
                    assert param.grad._md5sum() == baseline[name], (
                        f"{name}'s grad has diff"
                    )
        elif judge_machine_type() == "B":
            assert overlap_loss._md5sum() == "a7d554835b295e80ec1211e740cfa188"
            if paddle.distributed.get_rank() == 0:
                baseline = {
                    "_layers.shared_layers.embed.embedding.embed_tokens.weight": "7c630094b5660eb636646f778369cdee",
                    "_layers.9.0.input_layernorm.weight": "fa36647c97c191256150157380cd6509",
                    "_layers.9.0.self_attn.o_proj.weight": "c38f6a1397f412429dc6a8f05ee9972a",
                    "_layers.9.0.self_attn.qkv_proj.weight": "b3a8f4451744dd97ed2658f93229925f",
                    "_layers.9.0.self_attn.q_norm.weight": "f16a8f03ef21cd2902c6d1aedf65dea8",
                    "_layers.9.0.self_attn.k_norm.weight": "00877c5f9c7dc10ee080b6f9a91655d2",
                    "_layers.9.0.post_attention_layernorm.weight": "aef505d1faf64f9eac8d4a25e59df94b",
                    "_layers.9.0.mlp.gate.weight": "fb3e70a532f926752085039010b9149c",
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
                    "_layers.9.1.mlp.gate.weight": "6196fb17bc772474adda8d4d6f054b1e",
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
                for name, param in overlap_gpt_model.named_parameters():
                    assert param.grad._md5sum() == baseline[name], (
                        f"{name}'s grad has diff"
                    )


if __name__ == "__main__":
    unittest.main()
