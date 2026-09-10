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

import io
import unittest
from unittest.mock import MagicMock, patch

from paddleformers.datasets.template.mm_plugin import (
    MMPluginMixin,
    _check_video_is_nested_images,
    _make_batched_images,
    get_mm_plugin,
    register_mm_plugin,
)


class TestCheckVideoIsNestedImages(unittest.TestCase):
    """Tests for _check_video_is_nested function."""

    def test_nested_images(self):
        from PIL import Image

        frames = [Image.new("RGB", (10, 10)), Image.new("RGB", (10, 10))]
        self.assertTrue(_check_video_is_nested_images(frames))

    def test_nested_strings(self):
        frames = ["frame1.jpg", "frame2.jpg"]
        self.assertTrue(_check_video_is_nested_images(frames))

    def test_not_nested(self):
        self.assertFalse(_check_video_is_nested_images("video.mp4"))
        self.assertFalse(_check_video_is_nested_images(123))


class TestMakeBatchedImages(unittest.TestCase):
    """Tests for _make_batched_images function."""

    def test_basic_batching(self):
        images = [1, 2, 3, 4, 5]
        imglens = [2, 3]
        result = _make_batched_images(images, imglens)
        self.assertEqual(result, [[1, 2], [3, 4, 5]])

    def test_single_batch(self):
        images = [1, 2, 3]
        imglens = [3]
        result = _make_batched_images(images, imglens)
        self.assertEqual(result, [[1, 2, 3]])


class TestMMPluginMixin(unittest.TestCase):
    """Tests for MMPluginMixin class."""

    def _make_plugin(self, image_token="<image>", video_token="<video>", audio_token="<audio>"):
        return MMPluginMixin(image_token=image_token, video_token=video_token, audio_token=audio_token)

    def test_validate_input_no_images_with_image_token(self):
        """Test validation when images provided but image_token is None."""
        plugin = self._make_plugin(image_token=None, video_token=None, audio_token=None)
        with self.assertRaises(ValueError):
            plugin._validate_input(MagicMock(), [MagicMock()], [], [])

    def test_validate_input_no_videos_with_video_token(self):
        """Test validation when videos provided but video_token is None."""
        plugin = self._make_plugin(image_token=None, video_token=None, audio_token=None)
        with self.assertRaises(ValueError):
            plugin._validate_input(MagicMock(), [], [MagicMock()], [])

    def test_validate_input_no_audios_with_audio_token(self):
        """Test validation when audios provided but audio_token is None."""
        plugin = self._make_plugin(image_token=None, video_token=None, audio_token=None)
        with self.assertRaises(ValueError):
            plugin._validate_input(MagicMock(), [], [], [MagicMock()])

    def test_validate_input_image_token_no_processor(self):
        """Test validation when image_token set but no processor."""
        plugin = self._make_plugin()
        with self.assertRaises(ValueError):
            plugin._validate_input(None, [MagicMock()], [], [])

    def test_validate_input_image_token_no_image_processor(self):
        """Test validation when image_token set but no image_processor."""
        plugin = self._make_plugin()
        processor = MagicMock()
        processor.image_processor = None
        with self.assertRaises(ValueError):
            plugin._validate_input(processor, [MagicMock()], [], [])

    def test_validate_messages_mismatch_image(self):
        """Test validation when number of images doesn't match placeholders."""
        plugin = self._make_plugin()
        messages = [{"content": "hello <image> <image>"}]
        with self.assertRaises(ValueError):
            plugin._validate_messages(messages, [MagicMock()], [], [])

    def test_validate_messages_mismatch_video(self):
        """Test validation when number of videos doesn't match placeholders."""
        plugin = self._make_plugin()
        messages = [{"content": "hello <video> <video>"}]
        with self.assertRaises(ValueError):
            plugin._validate_messages(messages, [], [MagicMock()], [])

    def test_validate_messages_mismatch_audio(self):
        """Test validation when number of audios doesn't match placeholders."""
        plugin = self._make_plugin()
        messages = [{"content": "hello <audio> <audio>"}]
        with self.assertRaises(ValueError):
            plugin._validate_messages(messages, [], [], [MagicMock()])

    def test_validate_messages_match(self):
        """Test validation passes when numbers match."""
        plugin = self._make_plugin()
        messages = [{"content": "hello <image>"}]
        # Should not raise
        plugin._validate_messages(messages, [MagicMock()], [], [])

    def test_preprocess_image_resize_large(self):
        """Test image preprocessing with large image."""
        from PIL import Image

        plugin = self._make_plugin()
        large_image = Image.new("RGB", (2000, 2000))
        result = plugin._preprocess_image(large_image, image_max_pixels=1000 * 1000, image_min_pixels=32 * 32)
        self.assertLessEqual(result.width * result.height, 1000 * 1000)

    def test_preprocess_image_resize_small(self):
        """Test image preprocessing with small image."""
        from PIL import Image

        plugin = self._make_plugin()
        small_image = Image.new("RGB", (10, 10))
        result = plugin._preprocess_image(small_image, image_max_pixels=768 * 768, image_min_pixels=32 * 32)
        self.assertGreaterEqual(result.width * result.height, 32 * 32)

    def test_preprocess_image_convert_mode(self):
        """Test image preprocessing converts to RGB."""
        from PIL import Image

        plugin = self._make_plugin()
        rgba_image = Image.new("RGBA", (100, 100))
        result = plugin._preprocess_image(rgba_image, image_max_pixels=768 * 768, image_min_pixels=32 * 32)
        self.assertEqual(result.mode, "RGB")

    def test_file_download_invalid(self):
        """Test _file_download with invalid URL/path."""
        plugin = self._make_plugin()
        with self.assertRaises(ValueError):
            plugin._file_download("not_a_valid_url_or_path")

    @patch("paddleformers.datasets.template.mm_plugin.requests.get")
    def test_file_download_http(self, mock_get):
        """Test _file_download with HTTP URL."""
        mock_response = MagicMock()
        mock_response.content = b"fake_data"
        mock_get.return_value = mock_response

        plugin = self._make_plugin()
        result = plugin._file_download("http://example.com/file")
        self.assertIsInstance(result, io.BytesIO)


class TestGetMMPlugin(unittest.TestCase):
    """Tests for get_mm_plugin and register_mm_plugin."""

    def test_get_existing_plugin(self):
        """Test getting a registered plugin."""
        plugin = get_mm_plugin(name="base")
        self.assertIsNotNone(plugin)

    def test_get_nonexistent_plugin(self):
        """Test getting a non-registered plugin raises ValueError."""
        with self.assertRaises(ValueError):
            get_mm_plugin(name="nonexistent_plugin_xyz")

    def test_register_and_get_plugin(self):
        """Test registering and getting a custom plugin."""
        # Create a new plugin class
        class CustomPlugin(MMPluginMixin):
            pass

        register_mm_plugin("test_custom_mm", CustomPlugin)
        plugin = get_mm_plugin(name="test_custom_mm", image_token="<img>")
        self.assertIsInstance(plugin, CustomPlugin)

    def test_register_duplicate_plugin(self):
        """Test registering a duplicate plugin name raises ValueError."""
        with self.assertRaises(ValueError):
            register_mm_plugin("base", MMPluginMixin)


if __name__ == "__main__":
    unittest.main()
