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
from unittest.mock import MagicMock

from paddleformers.transformers.deepseek_v3.mfu_utils import DeepSeekProjection


class TestDeepSeekProjection(unittest.TestCase):
    """Tests for DeepSeekProjection."""

    def _make_model_config(self, q_lora_rank=448, n_routed_experts=64, n_shared_experts=2):
        config = MagicMock()
        config.vocab_size = 102400
        config.seq_length = 4096
        config.hidden_size = 2048
        config.intermediate_size = 1408
        config.moe_intermediate_size = 1408
        config.num_hidden_layers = 28
        config.first_k_dense_replace = 1
        config.num_attention_heads = 16
        config.qk_nope_head_dim = 128
        config.q_lora_rank = q_lora_rank
        config.kv_lora_rank = 512
        config.qk_rope_head_dim = 64
        config.n_shared_experts = n_shared_experts
        config.n_routed_experts = n_routed_experts
        config.num_experts_per_tok = 6
        return config

    def test_init(self):
        config = self._make_model_config()
        proj = DeepSeekProjection(config)
        self.assertEqual(proj._vocab_size, 102400)
        self.assertEqual(proj._dim, 2048)
        self.assertEqual(proj._n_layers, 28)

    def test_init_with_train_options(self):
        config = self._make_model_config()
        train_options = MagicMock()
        train_options.causal_mask = True
        train_options.fused_atten = True
        proj = DeepSeekProjection(config, train_options)
        self.assertTrue(proj._causal_mask)
        self.assertTrue(proj._fused_atten)

    def test_init_without_train_options(self):
        config = self._make_model_config()
        proj = DeepSeekProjection(config, train_options=None)
        self.assertTrue(proj._causal_mask)
        self.assertTrue(proj._fused_atten)

    def test_get_num_params_with_embedding(self):
        config = self._make_model_config(q_lora_rank=448, n_routed_experts=64, n_shared_experts=2)
        proj = DeepSeekProjection(config)
        num_params, num_activated = proj.get_num_params(include_embedding=True)
        self.assertGreater(num_params, 0)
        self.assertGreater(num_activated, 0)

    def test_get_num_params_without_embedding(self):
        config = self._make_model_config()
        proj = DeepSeekProjection(config)
        num_params, num_activated = proj.get_num_params(include_embedding=False)
        self.assertGreater(num_params, 0)
        self.assertGreater(num_activated, 0)

    def test_get_num_params_no_lora_q(self):
        """Test param count when q_lora_rank is None."""
        config = self._make_model_config(q_lora_rank=None)
        proj = DeepSeekProjection(config)
        num_params, num_activated = proj.get_num_params(include_embedding=True)
        self.assertGreater(num_params, 0)

    def test_get_num_params_no_experts(self):
        """Test param count with single expert."""
        config = self._make_model_config(n_routed_experts=1, n_shared_experts=0)
        proj = DeepSeekProjection(config)
        num_params, num_activated = proj.get_num_params(include_embedding=True)
        self.assertGreater(num_params, 0)

    def test_get_num_flop_fwd(self):
        config = self._make_model_config()
        proj = DeepSeekProjection(config)
        flops = proj.get_num_flop_fwd(batch_size=1)
        self.assertGreater(flops, 0)

    def test_get_num_flop_bwd(self):
        config = self._make_model_config()
        proj = DeepSeekProjection(config)
        flops = proj.get_num_flop_bwd(batch_size=1)
        self.assertGreater(flops, 0)

    def test_get_num_flop_bwd_no_fused(self):
        config = self._make_model_config()
        train_options = MagicMock()
        train_options.causal_mask = True
        train_options.fused_atten = False
        proj = DeepSeekProjection(config, train_options)
        flops = proj.get_num_flop_bwd(batch_size=1)
        self.assertGreater(flops, 0)

    def test_get_num_flop_per_token(self):
        config = self._make_model_config()
        proj = DeepSeekProjection(config)
        flops = proj.get_num_flop_per_token()
        self.assertGreater(flops, 0)

    def test_get_num_flop_qk_fwd(self):
        config = self._make_model_config()
        proj = DeepSeekProjection(config)
        flops = proj._get_num_flop_QK_fwd(batch_size=1)
        self.assertGreater(flops, 0)


if __name__ == "__main__":
    unittest.main()
