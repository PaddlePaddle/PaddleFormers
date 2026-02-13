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

import requests
from qwen_omni_utils import process_mm_info

from paddleformers.transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-Omni-30B-A3B-Instruct", download_hub="modelscope")
text = "What can you see and hear? Answer in one short sentence."
image_url = "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example1.jpg"
audio_url = "https://paddlenlp.bj.bcebos.com/models/community/paddlemix/audio-files/wave.wav"
image_response = requests.get(image_url)
audio_response = requests.get(audio_url)
with open("./example1.jpg", "wb") as f:
    f.write(image_response.content)
with open("./wave.wav", "wb") as f:
    f.write(audio_response.content)
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "./example1.jpg"},
            {"type": "audio", "audio": "./wave.wav"},
            {"type": "text", "text": "What can you see and hear? Answer in one short sentence."},
        ],
    },
]
USE_AUDIO_IN_VIDEO = True

text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
inputs = processor(
    text=text,
    audio=audios,
    images=images,
    videos=videos,
    return_tensors="pd",
    padding=True,
    use_audio_in_video=USE_AUDIO_IN_VIDEO,
)
