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

from paddlenlp.transformers.aistudio_utils import aistudio_download
from paddlenlp.utils.download import resolve_file_path


class TestAistudioDownload(unittest.TestCase):
    def test_aistudio_download(self):
        # 设置测试数据
        repo_id = "PaddleNLP/DeepSeek-R1-Distill-Qwen-1.5B"
        filename = "model.safetensors"
        revision = "master"
        local_dir = "./local/model"

        # 调用待测试的函数
        result = resolve_file_path(
            repo_id=repo_id,
            filenames=filename,
            revision=revision,
            from_aistudio=True,
            local_dir=local_dir,
        )

        # 验证结果
        print(result)
        self.assertEqual(result, f"{local_dir}/{filename}")

    def test_aistudio_download_transformer(self):
        # 设置测试数据
        repo_id = "PaddleNLP/DeepSeek-R1-Distill-Qwen-1.5B"
        filename = "model.safetensors"
        revision = "master"
        cache_dir = "./local/model"

        # 调用待测试的函数
        result = aistudio_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            cache_dir=cache_dir,
        )

        # 验证结果
        print(result)
        self.assertEqual(result, f"{cache_dir}/{filename}")


if __name__ == "__main__":
    unittest.main()
