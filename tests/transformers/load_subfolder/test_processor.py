# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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


# import unittest

# from paddleformers.transformers import AutoProcessor, CLIPProcessor
# from paddleformers.utils.log import logger

# from tests.testing_utils import slow


# @unittest.skip("skipping due to connection error!")
# class ProcessorLoadTester(unittest.TestCase):
#     @slow
#     def test_clip_load(self):
#         logger.info("Download model from PaddleFormers BOS")
#         clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", from_hf_hub=False)
#         clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32", from_hf_hub=False)

#         logger.info("Download model from local")
#         clip_processor.save_pretrained("./paddleformers-test-model/clip-vit-base-patch32")
#         clip_processor = CLIPProcessor.from_pretrained("./paddleformers-test-model/clip-vit-base-patch32")
#         clip_processor = AutoProcessor.from_pretrained("./paddleformers-test-model/clip-vit-base-patch32")
#         logger.info("Download model from PaddleFormers BOS with subfolder")
#         clip_processor = CLIPProcessor.from_pretrained("./paddleformers-test-model/", subfolder="clip-vit-base-patch32")
#         clip_processor = AutoProcessor.from_pretrained("./paddleformers-test-model/", subfolder="clip-vit-base-patch32")

#         logger.info("Download model from PaddleFormers BOS with subfolder")
#         clip_processor = CLIPProcessor.from_pretrained(
#             "baicai/paddleformers-test-model", subfolder="clip-vit-base-patch32", from_hf_hub=False
#         )
#         clip_processor = AutoProcessor.from_pretrained(
#             "baicai/paddleformers-test-model", subfolder="clip-vit-base-patch32", from_hf_hub=False
#         )

#         logger.info("Download model from aistudio")
#         clip_processor = CLIPProcessor.from_pretrained("aistudio/clip-vit-base-patch32", from_aistudio=True)
#         clip_processor = AutoProcessor.from_pretrained("aistudio/clip-vit-base-patch32", from_aistudio=True)

#         logger.info("Download model from aistudio with subfolder")
#         clip_processor = CLIPProcessor.from_pretrained(
#             "aistudio/paddleformers-test-model", subfolder="clip-vit-base-patch32", from_aistudio=True
#         )
#         clip_processor = AutoProcessor.from_pretrained(
#             "aistudio/paddleformers-test-model", subfolder="clip-vit-base-patch32", from_aistudio=True
#         )
