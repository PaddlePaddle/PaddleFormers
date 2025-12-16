# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
import unittest
from tempfile import TemporaryDirectory

from parameterized import parameterized

from paddleformers.mergekit import MergeConfig, MergeModel
from paddleformers.transformers import AutoModelForCausalLM
from tests.testing_utils import require_package


class TestMergeModel(unittest.TestCase):
    @parameterized.expand([("slerp",), ("della",), ("dare_linear",), ("ties",)])
    def test_merge_model_np(self, merge_method):
        with TemporaryDirectory() as tempdir:
            model = AutoModelForCausalLM.from_pretrained(
                "Paddleformers/tiny-random-qwen3", convert_from_hf=True, dtype="bfloat16"
            )
            pd_path = os.path.join(tempdir, "pd_model")
            model.save_pretrained(pd_path)
            safe_path = os.path.join(tempdir, "safe_model")
            model.save_pretrained(safe_path, safe_serialization="safetensors")

            # test mix
            merge_config = MergeConfig(
                merge_method=merge_method, model_path_list=[safe_path, pd_path], output_path=tempdir
            )
            mergekit = MergeModel(merge_config)
            mergekit.merge_model()

            # test mix with base model
            merge_config = MergeConfig(
                merge_method=merge_method,
                model_path_list=[safe_path, pd_path],
                output_path=tempdir,
                base_model_path=safe_path,
            )
            mergekit = MergeModel(merge_config)
            mergekit.merge_model()

            # test safetensor only
            merge_config = MergeConfig(
                merge_method=merge_method, model_path_list=[safe_path, safe_path], output_path=tempdir, n_process=2
            )
            mergekit = MergeModel(merge_config)
            mergekit.merge_model()

            # test safetensor only with base model
            merge_config = MergeConfig(
                merge_method=merge_method,
                model_path_list=[safe_path, safe_path],
                output_path=tempdir,
                n_process=2,
                base_model_path=safe_path,
            )
            mergekit = MergeModel(merge_config)
            mergekit.merge_model()

    @parameterized.expand([("slerp",), ("della",), ("dare_linear",), ("ties",)])
    def test_merge_model_pd(self, merge_method):
        with TemporaryDirectory() as tempdir:
            model = AutoModelForCausalLM.from_pretrained(
                "Paddleformers/tiny-random-qwen3", convert_from_hf=True, dtype="bfloat16"
            )
            pd_path = os.path.join(tempdir, "pd_model")
            model.save_pretrained(pd_path)
            safe_path = os.path.join(tempdir, "safe_model")
            model.save_pretrained(safe_path, safe_serialization="safetensors")

            # test mix
            merge_config = MergeConfig(
                merge_method=merge_method, model_path_list=[safe_path, pd_path], output_path=tempdir, tensor_type="pd"
            )
            mergekit = MergeModel(merge_config)
            mergekit.merge_model()

            # test mix with base model
            merge_config = MergeConfig(
                merge_method=merge_method,
                model_path_list=[safe_path, pd_path],
                output_path=tempdir,
                base_model_path=safe_path,
                tensor_type="pd",
            )
            mergekit = MergeModel(merge_config)
            mergekit.merge_model()

            # test safetensor only
            merge_config = MergeConfig(
                merge_method=merge_method,
                model_path_list=[safe_path, safe_path],
                output_path=tempdir,
                tensor_type="pd",
            )
            mergekit = MergeModel(merge_config)
            mergekit.merge_model()

            # test safetensor only with base model
            merge_config = MergeConfig(
                merge_method=merge_method,
                model_path_list=[safe_path, safe_path],
                output_path=tempdir,
                tensor_type="pd",
                base_model_path=safe_path,
            )
            mergekit = MergeModel(merge_config)
            mergekit.merge_model()

    @require_package("transformers", "torch")
    def test_fuse_qkv_lora_merge_torch(self):
        with TemporaryDirectory() as tempdir:
            # create torch model
            from transformers import Qwen3Config, Qwen3ForCausalLM

            torch_model_path = os.path.join(tempdir, "torch_model")
            config = Qwen3Config(
                hidden_size=16,
                intermediate_size=1120,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
            )
            model = Qwen3ForCausalLM(config)
            model.save_pretrained(torch_model_path)

            # load torch base model with fc(fused qkv/ffn)
            from paddleformers.transformers import Qwen3Config, Qwen3ForCausalLM

            model_config = Qwen3Config.from_pretrained(torch_model_path)
            model_config.fuse_attention_qkv = True
            model_config.fuse_attention_ffn = True
            fused_base_model = Qwen3ForCausalLM.from_pretrained(
                torch_model_path,
                config=model_config,
                convert_from_hf=True,
                dtype="float32",
                load_checkpoint_format="flex_checkpoint",
            )

            # create lora model
            from paddleformers.cli.utils import get_lora_target_modules
            from paddleformers.peft import LoRAConfig, LoRAModel

            target_modules = get_lora_target_modules(fused_base_model)
            lora_config = LoRAConfig(
                target_modules=target_modules,
                r=8,
                lora_alpha=4,
            )
            lora_model = LoRAModel(fused_base_model, lora_config)
            lora_model_path = os.path.join(tempdir, "lora_model")
            lora_model.save_pretrained(lora_model_path, save_checkpoint_format="flex_checkpoint")

            # merge fused lora model
            from paddleformers.mergekit import MergeConfig, MergeModel

            output_path = os.path.join(tempdir, "merged_model")
            merge_config = MergeConfig(
                base_model_path=torch_model_path,
                lora_model_path=lora_model_path,
                output_path=output_path,
                convert_from_hf=True,
                save_to_hf=True,
            )
            mergekit = MergeModel(merge_config)
            mergekit.merge_model()
