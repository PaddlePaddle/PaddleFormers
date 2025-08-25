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

# from paddleformers.transformers import AutoImageProcessor, CLIPImageProcessor
# from paddleformers.utils.log import logger

# from tests.testing_utils import slow


# @unittest.skip("skipping due to connection error!")
# class ImageProcessorLoadTester(unittest.TestCase):
#     @slow
#     def test_clip_load(self):
#         logger.info("Download model from PaddleFormers BOS")
#         clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32", from_hf_hub=False)
#         clip_processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32", from_hf_hub=False)

#         logger.info("Download model from local")
#         clip_processor.save_pretrained("./paddleformers-test-model/clip-vit-base-patch32")
#         clip_processor = CLIPImageProcessor.from_pretrained("./paddleformers-test-model/clip-vit-base-patch32")
#         clip_processor = AutoImageProcessor.from_pretrained("./paddleformers-test-model/clip-vit-base-patch32")
#         logger.info("Download model from PaddleFormers BOS with subfolder")
#         clip_processor = CLIPImageProcessor.from_pretrained(
#             "./paddleformers-test-model/", subfolder="clip-vit-base-patch32"
#         )
#         clip_processor = AutoImageProcessor.from_pretrained(
#             "./paddleformers-test-model/", subfolder="clip-vit-base-patch32"
#         )

#         logger.info("Download model from PaddleFormers BOS with subfolder")
#         clip_processor = CLIPImageProcessor.from_pretrained(
#             "baicai/paddleformers-test-model", subfolder="clip-vit-base-patch32", from_hf_hub=False
#         )
#         clip_processor = AutoImageProcessor.from_pretrained(
#             "baicai/paddleformers-test-model", subfolder="clip-vit-base-patch32", from_hf_hub=False
#         )

#         logger.info("Download model from aistudio")
#         clip_processor = CLIPImageProcessor.from_pretrained("aistudio/clip-vit-base-patch32", download_hub="aistudio")
#         clip_processor = AutoImageProcessor.from_pretrained("aistudio/clip-vit-base-patch32", download_hub="aistudio")

#         logger.info("Download model from aistudio with subfolder")
#         clip_processor = CLIPImageProcessor.from_pretrained(
#             "aistudio/paddleformers-test-model", subfolder="clip-vit-base-patch32", download_hub="aistudio"
#         )
#         clip_processor = AutoImageProcessor.from_pretrained(
#             "aistudio/paddleformers-test-model", subfolder="clip-vit-base-patch32", download_hub="aistudio"
#         )


# class ImageProcessorSubfolderLoadTester(unittest.TestCase):
#     def test_clip_subfolder_load(self):
#         logger.info("Download model with subfolder")
#         clip_processor = CLIPImageProcessor.from_pretrained(  # noqa: F841
#             "runwayml/stable-diffusion-v1-5", subfolder="feature_extractor"
#         )
