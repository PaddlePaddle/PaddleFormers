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

import unittest

from paddleformers.fleet.models.gpt.utils import GPTModelEstimator


class TestEstimatorForGLM45Air(unittest.TestCase):
    def setUp(self):
        self.seq_length = 4096
        self.create_estimator()

    def create_estimator(self):
        """
        Reference: https://huggingface.co/zai-org/GLM-4.5-Air/blob/main/config.json
        """
        self.model_name = "GLM-4.5-Air"
        self.estimator = GPTModelEstimator(
            seq_length=self.seq_length,
            vocab_size=151552,
            untie_embeddings_and_output_weights=True,  # tie_word_embeddings: false
            num_hidden_layers=46,
            hidden_size=4096,
            intermediate_size=10944,
            gated_linear_unit=True,  # hidden_act: "silu"
            num_attention_heads=96,
            head_dim=128,  # head_dim
            num_kv_heads=8,  # num_key_value_heads
            moe_layer_freq=[0] + [1] * 45,  # first_k_dense_replace: 1
            moe_num_experts=128,
            moe_intermediate_size=1408,
            moe_shared_expert_intermediate_size=1408,
            moe_topk=8,
            num_nextn_predict_layers=1,  # num_nextn_predict_layers: 1
            bf16=True,
        )

    def test_estimator(self):
        total_params, activated_params = self.estimator.estimate_num_parameters()
        flops_per_token = self.estimator.estimate_flops_per_token()

        gbs = 1
        flops_per_step = self.estimator.estimate_flops_per_step(gbs)

        tokens_per_second_per_gpu = 1000
        mfu = self.estimator.estimate_mfu(tokens_per_second_per_gpu)

        print(f"\n{'=' * 150}")
        print(
            f"{self.model_name:<14} | "
            f"Params: {total_params / 1e9:7.2f}B | "
            f"Act: {activated_params / 1e9:6.2f}B | "
            f"{flops_per_token / 1e9:8.2f} GFLOPs/Token | "
            f"{flops_per_step / 1e12:8.2f} TFLOPs/Step (gbs={gbs}) | "
            f"MFU: {mfu * 100:.2f}% (tokens_per_second_per_gpu={tokens_per_second_per_gpu})"
        )
        print(f"{'=' * 150}")


class TestEstimatorForDeepSeekV3(TestEstimatorForGLM45Air):
    def create_estimator(self):
        """
        Reference: https://huggingface.co/deepseek-ai/DeepSeek-V3-Base/blob/main/config.json
        """
        self.model_name = "DeepSeek-V3"
        self.estimator = GPTModelEstimator(
            seq_length=self.seq_length,
            vocab_size=129280,
            untie_embeddings_and_output_weights=True,  # tie_word_embeddings: false
            num_hidden_layers=61,
            hidden_size=7168,
            intermediate_size=18432,
            gated_linear_unit=True,  # hidden_act: "silu"
            num_attention_heads=128,
            causal_mask=True,
            multi_latent_attention=True,
            q_lora_rank=1536,
            kv_lora_rank=512,
            qk_head_dim=128,  # qk_nope_head_dim
            qk_pos_emb_head_dim=64,  # qk_rope_head_dim
            v_head_dim=128,
            moe_layer_freq=[0] * 3 + [1] * 58,  # first_k_dense_replace: 3
            moe_num_experts=256,
            moe_intermediate_size=2048,
            moe_shared_expert_intermediate_size=2048,
            moe_topk=8,
            num_nextn_predict_layers=1,  # num_nextn_predict_layers: 1
            fp8=True,
        )


class TestEstimatorForQwen3_30BA3B(TestEstimatorForGLM45Air):
    def create_estimator(self):
        """
        Reference: https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json
        """
        self.model_name = "Qwen3-30B-A3B"
        self.estimator = GPTModelEstimator(
            seq_length=4096,
            vocab_size=151936,
            untie_embeddings_and_output_weights=True,  # tie_word_embeddings: false
            num_hidden_layers=48,
            hidden_size=2048,
            intermediate_size=6144,  # intermediate_size
            gated_linear_unit=True,  # hidden_act: "silu"
            num_attention_heads=32,
            head_dim=128,  # head_dim
            num_kv_heads=4,  # num_key_value_heads
            moe_layer_freq=[1] * 48,  # decoder_sparse_step: 1
            moe_num_experts=128,
            moe_intermediate_size=768,
            moe_shared_expert_intermediate_size=0,
            moe_topk=8,
            num_nextn_predict_layers=None,
            fp16=True,
        )


if __name__ == "__main__":
    unittest.main()
