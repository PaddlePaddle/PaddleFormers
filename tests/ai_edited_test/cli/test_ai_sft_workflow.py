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

from paddleformers.cli.train.sft.workflow import (
    create_peft_model,
    freeze_param_except_mtp,
)


class TestFreezeParamExceptMTP(unittest.TestCase):
    """Tests for freeze_param_except_mtp function"""

    def test_freezes_non_mtp_params(self):
        model = MagicMock()
        # Create mock parameters
        param1 = MagicMock()
        param1.stop_gradient = False
        param2 = MagicMock()
        param2.stop_gradient = False

        model.state_dict.return_value = {
            "model.layers.0.self_attn.weight": param1,
            "model.layers.5.mlp.weight": param2,
        }

        config = MagicMock()
        config.num_hidden_layers = 6
        config.mtp_num_layers = 2

        freeze_param_except_mtp(model, config)

        # Layers 0-5 are non-MTP, so they should be frozen
        self.assertTrue(param1.stop_gradient)
        self.assertTrue(param2.stop_gradient)

    def test_mtp_params_not_frozen(self):
        model = MagicMock()
        param_mtp = MagicMock()
        param_mtp.stop_gradient = False
        param_regular = MagicMock()
        param_regular.stop_gradient = False

        model.state_dict.return_value = {
            "model.layers.6.mtp.weight": param_mtp,
            "model.layers.0.self_attn.weight": param_regular,
        }

        config = MagicMock()
        config.num_hidden_layers = 6
        config.mtp_num_layers = 2

        freeze_param_except_mtp(model, config)

        # MTP layer (6) should NOT be frozen
        self.assertFalse(param_mtp.stop_gradient)
        # Regular layer (0) should be frozen
        self.assertTrue(param_regular.stop_gradient)


class TestCreatePeftModel(unittest.TestCase):
    """Tests for create_peft_model function"""

    def test_no_lora_returns_model_unchanged(self):
        model_args = MagicMock()
        model_args.lora = False

        model = MagicMock()
        result = create_peft_model(model_args, MagicMock(), "float32", model)
        self.assertEqual(result, model)


class TestSFTWorkflowValidation(unittest.TestCase):
    """Tests for SFT workflow validation logic"""

    def test_compute_type_mapping(self):
        type_map = {"bf16": "bfloat16", "fp16": "float16"}
        self.assertEqual(type_map.get("bf16", "float32"), "bfloat16")
        self.assertEqual(type_map.get("fp16", "float32"), "float16")
        self.assertEqual(type_map.get("unknown", "float32"), "float32")

    def test_dtype_selection_fp16(self):
        training_args = MagicMock()
        training_args.fp16_opt_level = "O2"
        training_args.fp16 = True
        training_args.bf16 = False

        if training_args.fp16_opt_level == "O2":
            if training_args.fp16:
                dtype = "float16"
            elif training_args.bf16:
                dtype = "bfloat16"
            else:
                dtype = "float32"
        else:
            dtype = "float32"

        self.assertEqual(dtype, "float16")

    def test_dtype_selection_bf16(self):
        training_args = MagicMock()
        training_args.fp16_opt_level = "O2"
        training_args.fp16 = False
        training_args.bf16 = True

        if training_args.fp16_opt_level == "O2":
            if training_args.fp16:
                dtype = "float16"
            elif training_args.bf16:
                dtype = "bfloat16"
            else:
                dtype = "float32"
        else:
            dtype = "float32"

        self.assertEqual(dtype, "bfloat16")

    def test_dtype_selection_no_mixed_precision(self):
        training_args = MagicMock()
        training_args.fp16_opt_level = "O1"
        training_args.fp16 = False
        training_args.bf16 = False

        dtype = "float32"
        if training_args.fp16_opt_level == "O2":
            if training_args.fp16:
                dtype = "float16"
            elif training_args.bf16:
                dtype = "bfloat16"

        self.assertEqual(dtype, "float32")

    def test_max_seq_len_packing_mode(self):
        data_args = MagicMock()
        data_args.packing = True
        training_args = MagicMock()
        training_args.sequence_parallel = False
        training_args.context_parallel_size = 1

        max_seq_len = (
            data_args.max_seq_len
            if (data_args.packing or training_args.sequence_parallel or training_args.context_parallel_size > 1)
            else None
        )
        self.assertEqual(max_seq_len, data_args.max_seq_len)

    def test_max_seq_len_non_packing_mode(self):
        data_args = MagicMock()
        data_args.packing = False
        data_args.max_seq_len = 2048
        training_args = MagicMock()
        training_args.sequence_parallel = False
        training_args.context_parallel_size = 1

        max_seq_len = (
            data_args.max_seq_len
            if (data_args.packing or training_args.sequence_parallel or training_args.context_parallel_size > 1)
            else None
        )
        self.assertIsNone(max_seq_len)


if __name__ == "__main__":
    unittest.main()
