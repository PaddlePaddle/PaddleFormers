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
        import requests

        AUDIO_URL = "https://paddlenlp.bj.bcebos.com/models/community/paddlemix/audio-files/wave.wav"
        audio_response = requests.get(AUDIO_URL)
        with open("./cough.wav", "wb") as f:
            f.write(audio_response.content)
        cls.audio = process_audio_info("./wave.wav")

    def preprocess(self, feature_extractor):
        inputs = feature_extractor(self.audio, return_tensors="pd")
        self.assertIsInstance(inputs["pixel_values"], paddle.Tensor)

    @skip_for_none_ce_case
    def test_model_scope(self):
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            "Qwen/Qwen3-Omni-30B-A3B-Instruct", download_hub="modelscope"
        )
        self.preprocess(feature_extractor)

    @skip_for_none_ce_case
    def test_hf_hub(self):
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            "Qwen/Qwen3-Omni-30B-A3B-Instruct", download_hub="huggingface"
        )
        self.preprocess(feature_extractor)
