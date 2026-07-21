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

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from paddleformers.transformers.feature_extraction_utils import FEATURE_EXTRACTOR_NAME
from paddleformers.transformers.phi4_multimodal.processor import Phi4MultimodalProcessor


class Phi4MultimodalProcessorTest(unittest.TestCase):
    def test_get_arguments_resolves_remote_subfolder_config(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            config_path = Path(tmpdirname) / FEATURE_EXTRACTOR_NAME
            config_path.write_text(
                json.dumps(
                    {
                        "size": {"height": 336, "width": 336},
                        "patch_size": 16,
                        "feature_size": 40,
                        "sampling_rate": 8000,
                    }
                ),
                encoding="utf-8",
            )
            tokenizer = MagicMock()

            with (
                patch(
                    "paddleformers.transformers.phi4_multimodal.processor.resolve_file_path",
                    return_value=str(config_path),
                ) as resolve_mock,
                patch(
                    "paddleformers.transformers.auto.tokenizer.AutoTokenizer.from_pretrained",
                    return_value=tokenizer,
                ) as tokenizer_mock,
            ):
                (
                    image_processor,
                    feature_extractor,
                    loaded_tokenizer,
                ) = Phi4MultimodalProcessor._get_arguments_from_pretrained(
                    "microsoft/Phi-4-multimodal-instruct",
                    subfolder="processor",
                    cache_dir="/tmp/phi4-cache",
                    download_hub="huggingface",
                    local_files_only=True,
                )

        resolve_mock.assert_called_once_with(
            "microsoft/Phi-4-multimodal-instruct",
            FEATURE_EXTRACTOR_NAME,
            subfolder="processor",
            cache_dir="/tmp/phi4-cache",
            download_hub="huggingface",
            local_files_only=True,
            force_return=True,
        )
        tokenizer_mock.assert_called_once_with(
            "microsoft/Phi-4-multimodal-instruct",
            subfolder="processor",
            cache_dir="/tmp/phi4-cache",
            download_hub="huggingface",
            local_files_only=True,
        )
        self.assertEqual(image_processor.size, {"height": 336, "width": 336})
        self.assertEqual(image_processor.patch_size, 16)
        self.assertEqual(feature_extractor.feature_size, 40)
        self.assertEqual(feature_extractor.sampling_rate, 8000)
        self.assertIs(loaded_tokenizer, tokenizer)


if __name__ == "__main__":
    unittest.main()
