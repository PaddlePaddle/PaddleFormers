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

import json
import os
import tempfile

import numpy as np
import paddle
from PIL import Image

from ..transformers.test_utils import check_json_file_has_correct_format


def prepare_image_inputs(image_processor_tester, equal_resolution=False, numpify=False, paddlefy=False):
    """This function prepares a list of PIL images, or a list of numpy arrays if one specifies numpify=True,
    or a list of PaddlePaddle tensors if one specifies paddlefy=True.
    One can specify whether the images are of the same resolution or not.
    """

    assert not (numpify and paddlefy), "You cannot specify both numpy and PaddlePaddle tensors at the same time"

    image_inputs = []
    for i in range(image_processor_tester.batch_size):
        if equal_resolution:
            width = height = image_processor_tester.max_resolution
        else:
            # To avoid getting image width/height 0
            min_resolution = image_processor_tester.min_resolution
            if getattr(image_processor_tester, "size_divisor", None):
                # If `size_divisor` is defined, the image needs to have width/size >= `size_divisor`
                min_resolution = max(image_processor_tester.size_divisor, min_resolution)
            width, height = np.random.choice(np.arange(min_resolution, image_processor_tester.max_resolution), 2)
        image_inputs.append(
            np.random.randint(255, size=(image_processor_tester.num_channels, width, height), dtype=np.uint8)
        )

    if not numpify and not paddlefy:
        # PIL expects the channel dimension as last dimension
        image_inputs = [Image.fromarray(np.moveaxis(image, 0, -1)) for image in image_inputs]

    if paddlefy:
        image_inputs = [paddle.to_tensor(image) for image in image_inputs]

    return image_inputs


class ImageProcessingSavingTestMixin:
    test_cast_dtype = None

    def test_image_processor_to_json_string(self):
        image_processor = self.image_processing_class(**self.image_processor_dict)
        obj = json.loads(image_processor.to_json_string())
        for key, value in self.image_processor_dict.items():
            self.assertEqual(obj[key], value)

    def test_image_processor_to_json_file(self):
        image_processor_first = self.image_processing_class(**self.image_processor_dict)

        with tempfile.TemporaryDirectory() as tmpdirname:
            json_file_path = os.path.join(tmpdirname, "image_processor.json")
            image_processor_first.to_json_file(json_file_path)
            image_processor_second = self.image_processing_class.from_json_file(json_file_path)

        self.assertEqual(image_processor_second.to_dict(), image_processor_first.to_dict())

    def test_image_processor_from_and_save_pretrained(self):
        image_processor_first = self.image_processing_class(**self.image_processor_dict)

        with tempfile.TemporaryDirectory() as tmpdirname:
            saved_file = image_processor_first.save_pretrained(tmpdirname)[0]
            check_json_file_has_correct_format(saved_file)
            image_processor_second = self.image_processing_class.from_pretrained(tmpdirname)

        self.assertEqual(image_processor_second.to_dict(), image_processor_first.to_dict())
