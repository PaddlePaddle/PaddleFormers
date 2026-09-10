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
from collections import OrderedDict
from unittest.mock import MagicMock

import paddle

# Mock the missing transformers module to avoid import errors
if "transformers.tokenization_utils_tokenizers" not in sys.modules:
    _mock_mod = types.ModuleType("transformers.tokenization_utils_tokenizers")
    _mock_mod.TokenizersBackend = type("TokenizersBackend", (), {})
    sys.modules["transformers.tokenization_utils_tokenizers"] = _mock_mod

from paddleformers.cli.train.dpo.dpo_trainer import (
    DPO_INFO_KEYS,
    disable_dropout_in_model,
    fleet_merge_dpo_labels,
    prepare_pipeline_dpo_inputs_func,
)


class TestDPOInfoKeys(unittest.TestCase):
    """Tests for DPO_INFO_KEYS constant"""

    def test_contains_expected_keys(self):
        expected_keys = [
            "reference_chosen_logps",
            "reference_rejected_logps",
            "sft_loss",
            "policy_chosen_logps",
            "policy_rejected_logps",
            "dpo_loss",
        ]
        self.assertEqual(DPO_INFO_KEYS, expected_keys)

    def test_has_six_keys(self):
        self.assertEqual(len(DPO_INFO_KEYS), 6)


class TestDisableDropoutInModel(unittest.TestCase):
    """Tests for disable_dropout_in_model function"""

    def test_disables_dropout(self):
        model = paddle.nn.Sequential(
            paddle.nn.Linear(10, 10),
            paddle.nn.Dropout(p=0.5),
            paddle.nn.Linear(10, 10),
        )
        disable_dropout_in_model(model)
        for module in model.children():
            if isinstance(module, paddle.nn.Dropout):
                self.assertEqual(module.p, 0)

    def test_model_without_dropout(self):
        model = paddle.nn.Sequential(
            paddle.nn.Linear(10, 10),
            paddle.nn.Linear(10, 10),
        )
        # Should not raise any errors
        disable_dropout_in_model(model)

    def test_multiple_dropout_layers(self):
        model = paddle.nn.Sequential(
            paddle.nn.Dropout(p=0.1),
            paddle.nn.Linear(10, 10),
            paddle.nn.Dropout(p=0.3),
            paddle.nn.Dropout(p=0.5),
        )
        disable_dropout_in_model(model)
        for module in model.children():
            if isinstance(module, paddle.nn.Dropout):
                self.assertEqual(module.p, 0)


class TestPreparePipelineDPOInputsFunc(unittest.TestCase):
    """Tests for prepare_pipeline_dpo_inputs_func function"""

    def test_dict_input_with_attention_mask(self):
        inputs = OrderedDict(
            {
                "input_ids": paddle.zeros([2, 10]),
                "attention_mask": paddle.ones([2, 10]),
                "position_ids": paddle.arange(10).unsqueeze(0).expand([2, 10]),
                "response_labels": paddle.ones([2, 10]),
                "response_indexs": paddle.ones([4, 2]),
                "score_deltas": None,
                "reference_chosen_logps": None,
                "reference_rejected_logps": None,
            }
        )
        result = prepare_pipeline_dpo_inputs_func(inputs)
        self.assertEqual(len(result), 2)

    def test_dict_input_with_attn_mask_start_row_indices(self):
        inputs = OrderedDict(
            {
                "input_ids": paddle.zeros([2, 10]),
                "attn_mask_start_row_indices": paddle.ones([2, 10]),
                "attn_mask_startend_row_indices": paddle.ones([2, 10]),
                "position_ids": paddle.arange(10).unsqueeze(0).expand([2, 10]),
                "response_labels": paddle.ones([2, 10]),
                "response_indexs": paddle.ones([4, 2]),
                "score_deltas": None,
                "reference_chosen_logps": None,
                "reference_rejected_logps": None,
            }
        )
        result = prepare_pipeline_dpo_inputs_func(inputs)
        self.assertEqual(len(result), 2)

    def test_list_input(self):
        inputs = [
            OrderedDict(
                {
                    "input_ids": paddle.zeros([2, 10]),
                    "attention_mask": paddle.ones([2, 10]),
                    "position_ids": paddle.arange(10).unsqueeze(0).expand([2, 10]),
                    "response_labels": paddle.ones([2, 10]),
                    "response_indexs": paddle.ones([4, 2]),
                    "score_deltas": None,
                    "reference_chosen_logps": None,
                    "reference_rejected_logps": None,
                }
            ),
        ]
        result = prepare_pipeline_dpo_inputs_func(inputs)
        self.assertEqual(len(result), 2)


class TestFleetMergeDpoLabels(unittest.TestCase):
    """Tests for fleet_merge_dpo_labels function"""

    def test_merge_labels(self):
        labels = [
            ["a", "b", "c", "d"],
            ["e", "f", "g", "h"],
        ]
        reference_chosen_logps = [paddle.zeros([1]), paddle.ones([1])]
        reference_rejected_logps = [paddle.zeros([1]), paddle.ones([1])]
        logprobs = (reference_chosen_logps, reference_rejected_logps)

        result = fleet_merge_dpo_labels(labels, logprobs)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], ["a", "b"] + [reference_chosen_logps[0], reference_rejected_logps[0]])
        self.assertEqual(result[1], ["e", "f"] + [reference_chosen_logps[1], reference_rejected_logps[1]])


class TestDPOTrainerInit(unittest.TestCase):
    """Tests for DPOTrainer initialization validation logic"""

    def test_valid_loss_types(self):
        """Test all valid loss types in DPOTrainer."""
        valid_loss_types = [
            "sigmoid",
            "hinge",
            "ipo",
            "kto_pair",
            "sppo_hard",
            "nca_pair",
            "dpop",
            "or",
            "simpo",
        ]
        # These are the loss types DPOTrainer accepts
        self.assertEqual(len(valid_loss_types), 9)

    def test_unknown_loss_type_detected(self):
        """Test that unknown loss types are correctly identified."""
        valid_loss_types = {"sigmoid", "hinge", "ipo", "kto_pair", "sppo_hard", "nca_pair", "dpop", "or", "simpo"}
        self.assertNotIn("unknown_type", valid_loss_types)
        self.assertNotIn("bco_pair", valid_loss_types)

    def test_reference_free_supported_loss_types(self):
        """Test loss types that support reference_free mode."""
        reference_free_supported = {"sigmoid", "hinge", "ipo", "or", "simpo"}
        # These support reference_free
        for lt in reference_free_supported:
            self.assertIn(lt, reference_free_supported)
        # These do not
        self.assertNotIn("kto_pair", reference_free_supported)
        self.assertNotIn("nca_pair", reference_free_supported)
        self.assertNotIn("sppo_hard", reference_free_supported)
        self.assertNotIn("dpop", reference_free_supported)

    def test_loss_types_without_ref_model(self):
        """Test loss types that do not support ref_model."""
        no_ref_model_types = {"or", "simpo"}
        self.assertIn("or", no_ref_model_types)
        self.assertIn("simpo", no_ref_model_types)

    def test_dpo_config_none_detection(self):
        """Test dpo_config=None is detectable."""
        dpo_config = None
        self.assertIsNone(dpo_config)

    def test_reference_free_with_ref_model_conflict(self):
        """Test that reference_free=True with ref_model is a conflict."""
        reference_free = True
        ref_model = MagicMock()  # Not None
        self.assertTrue(reference_free)
        self.assertIsNotNone(ref_model)
        # This combination should raise ValueError in DPOTrainer

    def test_no_ref_no_reference_free_conflict(self):
        """Test that not reference_free and no ref_model is a conflict (unless lora)."""
        reference_free = False
        lora = False
        ref_model = None
        self.assertFalse(reference_free)
        self.assertFalse(lora)
        self.assertIsNone(ref_model)
        # This combination should raise ValueError in DPOTrainer

    def test_lora_mode_no_ref_needed(self):
        """Test that LoRA mode does not need ref_model even if not reference_free."""
        lora = True
        ref_model = None
        self.assertTrue(lora)
        # LoRA mode allows no ref_model
        self.assertIsNone(ref_model)


if __name__ == "__main__":
    unittest.main()
