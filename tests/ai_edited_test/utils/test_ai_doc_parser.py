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

import base64
import os
import tempfile
import unittest
from io import BytesIO
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from paddleformers.utils.doc_parser import DocParser


class TestDocParserInit(unittest.TestCase):
    def test_default_init(self):
        parser = DocParser()
        self.assertEqual(parser.ocr_lang, "ch")
        self.assertFalse(parser.use_angle_cls)
        self.assertFalse(parser.layout_analysis)
        self.assertIsNone(parser.ocr_infer_model)

    def test_custom_init(self):
        parser = DocParser(ocr_lang="en", layout_analysis=True, use_gpu=True, device_id=0)
        self.assertEqual(parser.ocr_lang, "en")
        self.assertTrue(parser.layout_analysis)
        self.assertTrue(parser.use_gpu)
        self.assertEqual(parser.device_id, 0)


class TestDocParserGetBuffer(unittest.TestCase):
    def test_file_path_short(self):
        # Create a temp file with content < 1024 chars
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            img = Image.new("RGB", (10, 10), color=(255, 0, 0))
            img.save(f, format="JPEG")
            path = f.name
        try:
            buff = DocParser._get_buffer(path)
            self.assertIsInstance(buff, bytes)
            self.assertTrue(len(buff) > 0)
        finally:
            os.unlink(path)

    def test_base64_input_long(self):
        # Create base64 string longer than 1024 chars
        img = Image.new("RGB", (100, 100), color=(0, 255, 0))
        buf = BytesIO()
        img.save(buf, format="JPEG")
        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        # Ensure it is long enough
        if len(b64_str) < 1024:
            b64_str = b64_str + "A" * (1024 - len(b64_str) + 1)
        buff = DocParser._get_buffer(b64_str)
        self.assertIsInstance(buff, bytes)

    def test_file_like_output(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            img = Image.new("RGB", (10, 10), color=(0, 0, 255))
            img.save(f, format="JPEG")
            path = f.name
        try:
            result = DocParser._get_buffer(path, file_like=True)
            self.assertIsInstance(result, BytesIO)
        finally:
            os.unlink(path)

    def test_nonexistent_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            DocParser._get_buffer("/nonexistent/path/to/file.jpg")


class TestDocParserReadImage(unittest.TestCase):
    def test_read_image_from_path(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            img = Image.new("RGB", (32, 32), color=(128, 128, 128))
            img.save(f, format="JPEG")
            path = f.name
        try:
            result = DocParser.read_image(path)
            self.assertIsInstance(result, np.ndarray)
            self.assertEqual(result.shape[2], 3)
        finally:
            os.unlink(path)


class TestDocParserNormalizeBox(unittest.TestCase):
    def test_normalize_box(self):
        box = [10, 20, 30, 40]
        old_size = (100, 200)
        new_size = (50, 100)
        result = DocParser._normalize_box(box, old_size, new_size)
        self.assertEqual(len(result), 4)
        # All values should be integers
        for v in result:
            self.assertIsInstance(v, int)

    def test_normalize_box_with_offset(self):
        box = [10, 20, 30, 40]
        old_size = (100, 200)
        new_size = (50, 100)
        result = DocParser._normalize_box(box, old_size, new_size, offset_x=5, offset_y=10)
        self.assertEqual(len(result), 4)


class TestDocParserExpandImageToA4Size(unittest.TestCase):
    def test_expand_tall_image(self):
        # Tall image: h/w >= 1.42
        image = np.ones((500, 200, 3), dtype=np.uint8) * 255
        result, offset_x, offset_y = DocParser.expand_image_to_a4_size(image, center=True)
        self.assertEqual(result.shape[0], 500)
        # Width should have expanded
        self.assertGreaterEqual(result.shape[1], 200)

    def test_expand_wide_image(self):
        # Wide image: h/w <= 1.40
        image = np.ones((200, 500, 3), dtype=np.uint8) * 255
        result, offset_x, offset_y = DocParser.expand_image_to_a4_size(image, center=True)
        self.assertEqual(result.shape[1], 500)
        # Height should have expanded
        self.assertGreaterEqual(result.shape[0], 200)

    def test_expand_tall_no_center(self):
        image = np.ones((500, 200, 3), dtype=np.uint8) * 255
        result, offset_x, offset_y = DocParser.expand_image_to_a4_size(image, center=False)
        self.assertEqual(offset_x, 0)
        self.assertEqual(offset_y, 0)

    def test_expand_wide_no_center(self):
        image = np.ones((200, 500, 3), dtype=np.uint8) * 255
        result, offset_x, offset_y = DocParser.expand_image_to_a4_size(image, center=False)
        self.assertEqual(offset_x, 0)
        self.assertEqual(offset_y, 0)

    def test_a4_ratio_no_change(self):
        # Image already close to A4 ratio (1.414)
        image = np.ones((424, 300, 3), dtype=np.uint8) * 255
        result, offset_x, offset_y = DocParser.expand_image_to_a4_size(image)
        # Should be unchanged since ratio is between 1.40 and 1.42
        self.assertEqual(result.shape[0], 424)
        self.assertEqual(result.shape[1], 300)


class TestDocParserCall(unittest.TestCase):
    def test_call_delegates_to_parse(self):
        parser = DocParser()
        with patch.object(parser, "parse", return_value={"doc": "test"}) as mock_parse:
            parser(doc={"doc": "test"})
            mock_parse.assert_called_once_with(doc={"doc": "test"})


class TestDocParserInitOcrInference(unittest.TestCase):
    def test_already_initialized_returns_early(self):
        parser = DocParser()
        mock_model = MagicMock()
        parser.ocr_infer_model = mock_model
        # Should return without reinitializing - model should stay the same
        parser.init_ocr_inference()
        # The ocr_infer_model should not have been replaced
        self.assertEqual(parser.ocr_infer_model, mock_model)

    @patch("paddleformers.utils.doc_parser.paddleocr", create=True)
    def test_import_error_raises(self, mock_paddleocr):
        parser = DocParser()
        parser.ocr_infer_model = None
        with patch.dict("sys.modules", {"paddleocr": None}):
            with self.assertRaises(RuntimeError):
                parser.init_ocr_inference()


if __name__ == "__main__":
    unittest.main()
