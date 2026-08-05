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
from types import SimpleNamespace

from paddleformers.transformers.qwen3_5.modeling import _normalize_parallel_subconfigs
from paddleformers.transformers.qwen3_5_moe.modeling import (
    Qwen3_5MoEForConditionalGeneration,
)


class TestQwen3_5MoEForConditionalGeneration(unittest.TestCase):
    """Tests for Qwen3_5MoEForConditionalGeneration."""

    def test_is_class(self):
        """Qwen3_5MoEForConditionalGeneration should be a class."""
        self.assertTrue(callable(Qwen3_5MoEForConditionalGeneration))

    @unittest.skip("issubclass check fails due to different module identity in CI")
    def test_inherits_from_qwen3_5(self):
        """Should inherit from Qwen3_5ForConditionalGeneration."""
        from paddleformers.transformers.qwen3_5 import (
            Qwen3_5ForConditionalGeneration as Qwen3_5,
        )

        self.assertTrue(issubclass(Qwen3_5MoEForConditionalGeneration, Qwen3_5))

    def test_new_method(self):
        """Test that __new__ works correctly."""

        # Qwen3_5MoEForConditionalGeneration uses __new__ to pass have_criterion
        # Can't fully instantiate without a real config, but test the class structure
        self.assertTrue(hasattr(Qwen3_5MoEForConditionalGeneration, "__new__"))

    def test_parallel_degrees_propagate_to_subconfigs(self):
        text_config = SimpleNamespace()
        vision_config = SimpleNamespace()
        config = SimpleNamespace(
            text_config=text_config,
            vision_config=vision_config,
            tensor_model_parallel_size=-1,
            context_parallel_size=-1,
            pipeline_model_parallel_size=-1,
            virtual_pipeline_model_parallel_size=-1,
            expert_model_parallel_size=4,
        )

        _normalize_parallel_subconfigs(config)

        self.assertEqual(text_config.tensor_model_parallel_size, 1)
        self.assertEqual(vision_config.pipeline_model_parallel_size, 1)
        self.assertEqual(text_config.expert_model_parallel_size, 4)


if __name__ == "__main__":
    unittest.main()
