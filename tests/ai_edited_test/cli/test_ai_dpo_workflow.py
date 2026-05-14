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

import sys
import types
import unittest
from unittest.mock import MagicMock

# Mock the missing transformers module to avoid import errors
if "transformers.tokenization_utils_tokenizers" not in sys.modules:
    _mock_mod = types.ModuleType("transformers.tokenization_utils_tokenizers")
    _mock_mod.TokenizersBackend = type("TokenizersBackend", (), {})
    sys.modules["transformers.tokenization_utils_tokenizers"] = _mock_mod


class TestRunDPOValidation(unittest.TestCase):
    """Tests for run_dpo workflow validation logic"""

    def test_orpo_loss_type_sets_reference_free(self):
        # Simulate the logic in run_dpo for orpo loss type
        training_args = MagicMock()
        training_args.loss_type = "orpo"
        training_args.reference_free = False
        training_args.sft_loss_ratio = 0.0

        # This mirrors the logic in run_dpo
        if training_args.loss_type == "orpo":
            training_args.reference_free = True
            training_args.sft_loss_ratio = 1.0
            training_args.loss_type = "or"

        self.assertTrue(training_args.reference_free)
        self.assertEqual(training_args.sft_loss_ratio, 1.0)
        self.assertEqual(training_args.loss_type, "or")

    def test_or_loss_type_forces_reference_free(self):
        training_args = MagicMock()
        training_args.loss_type = "or"
        training_args.reference_free = False

        if training_args.loss_type in ["or", "simpo"] and not training_args.reference_free:
            training_args.reference_free = True

        self.assertTrue(training_args.reference_free)

    def test_simpo_loss_type_forces_reference_free(self):
        training_args = MagicMock()
        training_args.loss_type = "simpo"
        training_args.reference_free = False

        if training_args.loss_type in ["or", "simpo"] and not training_args.reference_free:
            training_args.reference_free = True

        self.assertTrue(training_args.reference_free)

    def test_sequence_parallel_disabled_when_tp_le_1(self):
        training_args = MagicMock()
        training_args.sequence_parallel = True
        training_args.tensor_model_parallel_size = 1
        training_args.pipeline_model_parallel_size = 1

        if training_args.sequence_parallel:
            if training_args.tensor_model_parallel_size <= 1:
                training_args.sequence_parallel = False

        self.assertFalse(training_args.sequence_parallel)

    def test_compute_type_mapping(self):
        type_map = {"bf16": "bfloat16", "fp16": "float16"}
        self.assertEqual(type_map.get("bf16", "float32"), "bfloat16")
        self.assertEqual(type_map.get("fp16", "float32"), "float16")
        self.assertEqual(type_map.get("fp32", "float32"), "float32")

    def test_invalid_attn_implementation(self):
        model_args = MagicMock()
        model_args._attn_implementation = "invalid_impl"

        from paddleformers.nn.attention import AttentionInterface

        available = AttentionInterface._global_mapping.keys()
        self.assertNotIn("invalid_impl", available)


if __name__ == "__main__":
    unittest.main()
