# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
import os
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)


import unittest
from unittest.mock import patch


class TestGPTModelEstimatorDefaults(unittest.TestCase):
    """Test GPTModelEstimator default values."""

    def test_default_fields(self):
        from paddleformers.fleet.models.gpt.utils import GPTModelEstimator

        est = GPTModelEstimator()
        self.assertEqual(est.seq_length, 0)
        self.assertEqual(est.vocab_size, 0)
        self.assertFalse(est.untie_embeddings_and_output_weights)
        self.assertEqual(est.num_hidden_layers, 0)
        self.assertEqual(est.hidden_size, 0)
        self.assertEqual(est.intermediate_size, 0)
        self.assertFalse(est.gated_linear_unit)
        self.assertEqual(est.num_attention_heads, 0)
        self.assertEqual(est.head_dim, 0)
        self.assertEqual(est.num_kv_heads, 0)
        self.assertFalse(est.causal_mask)
        self.assertFalse(est.multi_latent_attention)
        self.assertIsNone(est.q_lora_rank)
        self.assertFalse(est.bf16)
        self.assertFalse(est.fp16)
        self.assertFalse(est.fp8)


class TestGPTModelEstimatorDenseParams(unittest.TestCase):
    """Test GPTModelEstimator parameter estimation for dense models."""

    def test_estimate_num_parameters_dense_no_glu(self):
        from paddleformers.fleet.models.gpt.utils import GPTModelEstimator

        est = GPTModelEstimator(
            vocab_size=32000,
            hidden_size=4096,
            num_hidden_layers=32,
            intermediate_size=11008,
            num_attention_heads=32,
            head_dim=128,
            num_kv_heads=32,
            moe_num_experts=0,
            moe_intermediate_size=0,
            moe_topk=0,
        )
        total, activated = est.estimate_num_parameters()
        self.assertGreater(total, 0)
        self.assertEqual(total, activated)

    def test_estimate_num_parameters_dense_with_glu(self):
        from paddleformers.fleet.models.gpt.utils import GPTModelEstimator

        est = GPTModelEstimator(
            vocab_size=32000,
            hidden_size=4096,
            num_hidden_layers=32,
            intermediate_size=11008,
            num_attention_heads=32,
            head_dim=128,
            num_kv_heads=32,
            gated_linear_unit=True,
            moe_num_experts=0,
            moe_intermediate_size=0,
            moe_topk=0,
        )
        total, activated = est.estimate_num_parameters()
        self.assertGreater(total, 0)

    def test_estimate_num_parameters_untied(self):
        from paddleformers.fleet.models.gpt.utils import GPTModelEstimator

        est_tied = GPTModelEstimator(
            vocab_size=32000,
            hidden_size=4096,
            untie_embeddings_and_output_weights=False,
            moe_num_experts=0,
            moe_intermediate_size=0,
            moe_topk=0,
        )
        est_untied = GPTModelEstimator(
            vocab_size=32000,
            hidden_size=4096,
            untie_embeddings_and_output_weights=True,
            moe_num_experts=0,
            moe_intermediate_size=0,
            moe_topk=0,
        )
        total_tied, _ = est_tied.estimate_num_parameters()
        total_untied, _ = est_untied.estimate_num_parameters()
        self.assertEqual(total_untied, total_tied * 2)


class TestGPTModelEstimatorMoEParams(unittest.TestCase):
    """Test GPTModelEstimator parameter estimation for MoE models."""

    def test_estimate_num_parameters_moe(self):
        from paddleformers.fleet.models.gpt.utils import GPTModelEstimator

        est = GPTModelEstimator(
            vocab_size=32000,
            hidden_size=4096,
            num_hidden_layers=4,
            intermediate_size=11008,
            num_attention_heads=32,
            head_dim=128,
            num_kv_heads=32,
            moe_layer_freq=[1, 0, 1, 0],
            moe_num_experts=8,
            moe_intermediate_size=11008,
            moe_topk=2,
        )
        total, activated = est.estimate_num_parameters()
        self.assertGreater(total, 0)
        self.assertGreater(total, activated)

    def test_estimate_num_parameters_moe_with_shared_expert(self):
        from paddleformers.fleet.models.gpt.utils import GPTModelEstimator

        est = GPTModelEstimator(
            vocab_size=32000,
            hidden_size=4096,
            num_hidden_layers=4,
            intermediate_size=11008,
            num_attention_heads=32,
            head_dim=128,
            num_kv_heads=32,
            moe_layer_freq=[1, 0, 1, 0],
            moe_num_experts=8,
            moe_intermediate_size=11008,
            moe_topk=2,
            moe_shared_expert_intermediate_size=4096,
        )
        total, activated = est.estimate_num_parameters()
        self.assertGreater(total, activated)


class TestGPTModelEstimatorMLAParams(unittest.TestCase):
    """Test GPTModelEstimator parameter estimation for MLA models."""

    def test_estimate_num_parameters_mla(self):
        from paddleformers.fleet.models.gpt.utils import GPTModelEstimator

        est = GPTModelEstimator(
            vocab_size=32000,
            hidden_size=4096,
            num_hidden_layers=2,
            intermediate_size=11008,
            num_attention_heads=32,
            head_dim=128,
            num_kv_heads=32,
            multi_latent_attention=True,
            q_lora_rank=512,
            kv_lora_rank=512,
            qk_head_dim=128,
            qk_pos_emb_head_dim=64,
            v_head_dim=128,
            moe_num_experts=0,
            moe_intermediate_size=0,
            moe_topk=0,
        )
        total, activated = est.estimate_num_parameters()
        self.assertGreater(total, 0)

    def test_estimate_num_parameters_mla_no_lora(self):
        from paddleformers.fleet.models.gpt.utils import GPTModelEstimator

        est = GPTModelEstimator(
            vocab_size=32000,
            hidden_size=4096,
            num_hidden_layers=2,
            intermediate_size=11008,
            num_attention_heads=32,
            head_dim=128,
            num_kv_heads=32,
            multi_latent_attention=True,
            q_lora_rank=None,
            kv_lora_rank=512,
            qk_head_dim=128,
            qk_pos_emb_head_dim=64,
            v_head_dim=128,
            moe_num_experts=0,
            moe_intermediate_size=0,
            moe_topk=0,
        )
        total, activated = est.estimate_num_parameters()
        self.assertGreater(total, 0)


class TestGPTModelEstimatorFLOPs(unittest.TestCase):
    """Test GPTModelEstimator FLOPs estimation."""

    def test_estimate_flops_per_token_dense(self):
        from paddleformers.fleet.models.gpt.utils import GPTModelEstimator

        est = GPTModelEstimator(
            seq_length=2048,
            vocab_size=32000,
            hidden_size=4096,
            num_hidden_layers=2,
            intermediate_size=11008,
            num_attention_heads=32,
            head_dim=128,
            num_kv_heads=32,
            moe_num_experts=0,
            moe_intermediate_size=0,
            moe_topk=0,
        )
        flops = est.estimate_flops_per_token()
        self.assertGreater(flops, 0)

    def test_estimate_flops_per_step(self):
        from paddleformers.fleet.models.gpt.utils import GPTModelEstimator

        est = GPTModelEstimator(
            seq_length=2048,
            vocab_size=32000,
            hidden_size=4096,
            num_hidden_layers=2,
            intermediate_size=11008,
            num_attention_heads=32,
            head_dim=128,
            num_kv_heads=32,
            moe_num_experts=0,
            moe_intermediate_size=0,
            moe_topk=0,
        )
        flops_step = est.estimate_flops_per_step(batch_size=4)
        self.assertGreater(flops_step, 0)

    def test_estimate_flops_with_mtp(self):
        from paddleformers.fleet.models.gpt.utils import GPTModelEstimator

        est = GPTModelEstimator(
            seq_length=2048,
            vocab_size=32000,
            hidden_size=4096,
            num_hidden_layers=2,
            intermediate_size=11008,
            num_attention_heads=32,
            head_dim=128,
            num_kv_heads=32,
            num_nextn_predict_layers=2,
            moe_layer_freq=[0, 0],
            moe_num_experts=0,
            moe_intermediate_size=0,
            moe_topk=0,
        )
        flops = est.estimate_flops_per_token()
        self.assertGreater(flops, 0)


class TestGPTModelEstimatorMFU(unittest.TestCase):
    """Test GPTModelEstimator MFU estimation."""

    @patch("paddle.device.is_compiled_with_cuda", return_value=False)
    def test_estimate_mfu_no_cuda(self, mock_cuda):
        from paddleformers.fleet.models.gpt.utils import GPTModelEstimator

        est = GPTModelEstimator()
        mfu = est.estimate_mfu(tokens_per_second_per_gpu=1000.0)
        self.assertEqual(mfu, 0)

    @patch("paddle.device.cuda.get_device_name", return_value="A100")
    @patch("paddle.device.is_compiled_with_cuda", return_value=True)
    def test_estimate_mfu_bf16_a100(self, mock_cuda, mock_name):
        from paddleformers.fleet.models.gpt.utils import GPTModelEstimator

        est = GPTModelEstimator(
            bf16=True,
            seq_length=2048,
            vocab_size=32000,
            hidden_size=4096,
            num_hidden_layers=2,
            intermediate_size=11008,
            num_attention_heads=32,
            head_dim=128,
            num_kv_heads=32,
            moe_num_experts=0,
            moe_intermediate_size=0,
            moe_topk=0,
        )
        mfu = est.estimate_mfu(tokens_per_second_per_gpu=1000.0)
        self.assertGreater(mfu, 0)


class TestGPUSpecifications(unittest.TestCase):
    """Test GPUSpecifications dataclass."""

    def test_a100_specs(self):
        from paddleformers.fleet.models.gpt.utils import (
            GPU_SPECIFICATIONS_REGISTRATION,
        )

        a100 = next(
            s for s in GPU_SPECIFICATIONS_REGISTRATION if "A100" in s.names
        )
        self.assertEqual(a100.FP32_TFLOPS, 19.5)
        self.assertEqual(a100.BF16_TFLOPS, 312)
        self.assertIsNone(a100.FP8_TFLOPS)

    def test_h100_specs(self):
        from paddleformers.fleet.models.gpt.utils import GPU_SPECIFICATIONS_REGISTRATION

        h100 = next(
            s for s in GPU_SPECIFICATIONS_REGISTRATION if "H100" in s.names
        )
        self.assertEqual(h100.FP32_TFLOPS, 67)
        self.assertEqual(h100.FP8_TFLOPS, 1979)

    def test_b200_specs(self):
        from paddleformers.fleet.models.gpt.utils import GPU_SPECIFICATIONS_REGISTRATION

        b200 = next(
            s for s in GPU_SPECIFICATIONS_REGISTRATION if "B200" in s.names
        )
        self.assertEqual(b200.BF16_TFLOPS, 2200)
        self.assertEqual(b200.FP8_TFLOPS, 4500)

    def test_gb200_specs(self):
        from paddleformers.fleet.models.gpt.utils import GPU_SPECIFICATIONS_REGISTRATION

        gb200 = next(
            s for s in GPU_SPECIFICATIONS_REGISTRATION if "GB200" in s.names
        )
        self.assertEqual(gb200.FP32_TFLOPS, 80)
        self.assertEqual(gb200.BF16_TFLOPS, 2500)


class TestFillFeature(unittest.TestCase):
    """Test fill_feature function."""

    def test_fill_target_positions(self):
        """Fill marked positions with value, leave others unchanged."""
        import paddle

        from paddleformers.fleet.models.gpt.utils import fill_feature

        # [B, S, D] = [1, 4, 3]
        input_embeds = paddle.ones([1, 4, 3], dtype="float32") * 2.0
        # positions 0,2 are padding (True means fill)
        target_index = paddle.to_tensor([[True, False, True, False]])

        result = fill_feature(input_embeds, target_index, 0.0)

        self.assertEqual(result.shape, [1, 4, 3])
        # filled positions should be 0
        self.assertAlmostEqual(result[0, 0, 0].item(), 0.0)
        self.assertAlmostEqual(result[0, 2, 0].item(), 0.0)
        # untouched positions should remain 2
        self.assertAlmostEqual(result[0, 1, 0].item(), 2.0)
        self.assertAlmostEqual(result[0, 3, 0].item(), 2.0)

    def test_no_target_positions(self):
        """When target_index is all False, input_embeds should be unchanged."""
        import paddle

        from paddleformers.fleet.models.gpt.utils import fill_feature

        input_embeds = paddle.ones([2, 3, 4], dtype="float32") * 5.0
        target_index = paddle.zeros([2, 3], dtype="bool")

        result = fill_feature(input_embeds, target_index, 0.0)
        self.assertAlmostEqual(result.sum().item(), 5.0 * 2 * 3 * 4)


class TestGetDevicePeakTFLOPS(unittest.TestCase):
    """Test _get_device_peak_tflops method."""

    @patch("paddle.device.is_compiled_with_cuda", return_value=False)
    def test_no_cuda_returns_none(self, mock_cuda):
        from paddleformers.fleet.models.gpt.utils import GPTModelEstimator

        est = GPTModelEstimator()
        result = est._get_device_peak_tflops()
        self.assertIsNone(result)

    @patch("paddle.device.cuda.get_device_name", return_value="UNKNOWN_GPU")
    @patch("paddle.device.is_compiled_with_cuda", return_value=True)
    def test_unknown_gpu_returns_none(self, mock_cuda, mock_name):
        import logging

        from paddleformers.fleet.models.gpt.utils import GPTModelEstimator

        est = GPTModelEstimator()
        # Patch logger.warning to avoid issues with corrupted handler state
        # from other tests in the suite
        with patch.object(
            logging.getLogger("paddleformers.fleet.models.gpt.utils"), "warning"
        ):
            result = est._get_device_peak_tflops()
        self.assertIsNone(result)

    @patch("paddle.device.cuda.get_device_name", return_value="A800-SXM4-80GB")
    @patch("paddle.device.is_compiled_with_cuda", return_value=True)
    def test_a800_matched(self, mock_cuda, mock_name):
        from paddleformers.fleet.models.gpt.utils import GPTModelEstimator

        est = GPTModelEstimator(bf16=True)
        result = est._get_device_peak_tflops()
        self.assertEqual(result, 312)


if __name__ == "__main__":
    unittest.main()
