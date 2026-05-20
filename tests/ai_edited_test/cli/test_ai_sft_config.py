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
from dataclasses import fields

# Mock the missing transformers module to avoid import errors
if "transformers.tokenization_utils_tokenizers" not in sys.modules:
    _mock_mod = types.ModuleType("transformers.tokenization_utils_tokenizers")
    _mock_mod.TokenizersBackend = type("TokenizersBackend", (), {})
    sys.modules["transformers.tokenization_utils_tokenizers"] = _mock_mod

from paddleformers.cli.train.sft.sft_config import SFTConfig


class TestSFTConfig(unittest.TestCase):
    """Tests for SFTConfig dataclass"""

    def test_default_values(self):
        config = SFTConfig(output_dir="/tmp/test_sft_output", amp_master_grad=False)
        self.assertFalse(config.autotuner_benchmark)
        self.assertEqual(config.eval_iters, -1)
        self.assertEqual(config.decay_steps, 0)
        self.assertFalse(config.tensor_parallel_output)
        self.assertEqual(config.max_estimate_samples, 1e5)
        self.assertFalse(config.unified_checkpoint)
        self.assertEqual(config.unified_checkpoint_config, "")
        self.assertEqual(config.dataset_text_field, "text")
        self.assertEqual(config.learning_rate, 2.0e-5)
        self.assertEqual(config.max_seq_len, 2048)
        self.assertIsNone(config.dataset_num_proc)
        self.assertEqual(config.dataset_batch_size, 1000)
        self.assertIsNone(config.model_init_kwargs)
        self.assertIsNone(config.dataset_kwargs)
        self.assertIsNone(config.eval_packing)
        self.assertFalse(config.use_ssa)
        self.assertEqual(config.ssa_group_size_ratio, 0.25)

    def test_autotuner_benchmark_mode(self):
        config = SFTConfig(output_dir="/tmp/test_sft_output", autotuner_benchmark=True, amp_master_grad=False)
        self.assertEqual(config.max_steps, 5)
        self.assertTrue(config.do_train)
        self.assertFalse(config.do_export)
        self.assertFalse(config.do_predict)
        self.assertFalse(config.do_eval)
        self.assertTrue(config.overwrite_output_dir)
        self.assertFalse(config.load_best_model_at_end)

    def test_custom_values(self):
        config = SFTConfig(
            output_dir="/tmp/custom_output",
            eval_iters=100,
            decay_steps=500,
            tensor_parallel_output=True,
            max_seq_len=4096,
            learning_rate=1e-4,
            dataset_num_proc=4,
            dataset_batch_size=500,
            use_ssa=True,
            ssa_group_size_ratio=0.5,
            amp_master_grad=False,
        )
        self.assertEqual(config.eval_iters, 100)
        self.assertEqual(config.decay_steps, 500)
        self.assertTrue(config.tensor_parallel_output)
        self.assertEqual(config.max_seq_len, 4096)
        self.assertEqual(config.learning_rate, 1e-4)
        self.assertEqual(config.dataset_num_proc, 4)
        self.assertEqual(config.dataset_batch_size, 500)
        self.assertTrue(config.use_ssa)
        self.assertEqual(config.ssa_group_size_ratio, 0.5)

    def test_fields_count(self):
        all_fields = [f.name for f in fields(SFTConfig)]
        self.assertGreaterEqual(len(all_fields), 20)


if __name__ == "__main__":
    unittest.main()
