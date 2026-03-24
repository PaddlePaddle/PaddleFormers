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

import unittest

import paddle

from paddleformers.transformers import AutoFeatureExtractor
from paddleformers.transformers.audio_processing_utils import process_audio_info
from tests.testing_utils import skip_for_none_ce_case


class TestHFMultiSourceAudioProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "audio": "https://paddlenlp.bj.bcebos.com/models/community/paddlemix/audio-files/wave.wav",
                    },
                ],
            },
        ]
        cls.audio = process_audio_info(conversation, use_audio_in_video=True)

    def preprocess(self, feature_extractor):
        inputs = feature_extractor(self.audio, return_tensors="pd")
        self.assertIsInstance(inputs["input_features"], paddle.Tensor)

    # TODO: Temporarily use repo_id oftiny model, replace later.
    def test_aistudio(self):
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            "PaddleFormers/tiny-random-qwen3omni", download_hub="aistudio"
        )
        self.preprocess(feature_extractor)

    @skip_for_none_ce_case
    def test_model_scope(self):
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            "Qwen/Qwen3-Omni-30B-A3B-Instruct", download_hub="modelscope"
        )
        self.preprocess(feature_extractor)

    @skip_for_none_ce_case
    def test_hf_hub(self):
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            "Qwen/Qwen3-Omni-30B-A3B-Instruct", download_hub="modelscope"
        )
        self.preprocess(feature_extractor)
