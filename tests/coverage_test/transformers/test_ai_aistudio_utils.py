# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from unittest.mock import patch

from requests import HTTPError


class TestAistudioUtils(unittest.TestCase):
    """Tests for paddleformers.transformers.aistudio_utils"""

    def test_unauthorized_error(self):
        """Test UnauthorizedError can be raised and caught."""
        from paddleformers.transformers.aistudio_utils import UnauthorizedError

        with self.assertRaises(UnauthorizedError):
            raise UnauthorizedError("test")

    def test_entry_not_found_error(self):
        """Test EntryNotFoundError can be raised and caught."""
        from paddleformers.transformers.aistudio_utils import EntryNotFoundError

        with self.assertRaises(EntryNotFoundError):
            raise EntryNotFoundError("test")

    def test_add_subfolder_with_subfolder(self):
        """Test _add_subfolder when subfolder is provided and non-empty."""
        from paddleformers.transformers.aistudio_utils import _add_subfolder

        result = _add_subfolder("model.safetensors", "subfolder")
        self.assertEqual(result, "subfolder/model.safetensors")

    def test_add_subfolder_with_none_subfolder(self):
        """Test _add_subfolder when subfolder is None."""
        from paddleformers.transformers.aistudio_utils import _add_subfolder

        result = _add_subfolder("model.safetensors", None)
        self.assertEqual(result, "model.safetensors")

    def test_add_subfolder_with_empty_subfolder(self):
        """Test _add_subfolder when subfolder is empty string."""
        from paddleformers.transformers.aistudio_utils import _add_subfolder

        result = _add_subfolder("model.safetensors", "")
        self.assertEqual(result, "model.safetensors")

    @patch("paddleformers.transformers.aistudio_utils.download")
    def test_aistudio_download_success(self, mock_download):
        """Test aistudio_download when download succeeds."""
        from paddleformers.transformers.aistudio_utils import aistudio_download

        mock_download.return_value = "/path/to/file"
        result = aistudio_download("repo_id", filename="model.safetensors")
        self.assertEqual(result, "/path/to/file")
        mock_download.assert_called_once_with(
            repo_id="repo_id",
            file_path="model.safetensors",
            revision="master",
        )

    @patch("paddleformers.transformers.aistudio_utils.download")
    def test_aistudio_download_with_revision(self, mock_download):
        """Test aistudio_download with a specific revision."""
        from paddleformers.transformers.aistudio_utils import aistudio_download

        mock_download.return_value = "/path/to/file"
        result = aistudio_download("repo_id", filename="model.safetensors", revision="v1.0")
        self.assertEqual(result, "/path/to/file")
        mock_download.assert_called_once_with(
            repo_id="repo_id",
            file_path="model.safetensors",
            revision="v1.0",
        )

    @patch("paddleformers.transformers.aistudio_utils.download")
    def test_aistudio_download_with_cache_dir(self, mock_download):
        """Test aistudio_download with a cache_dir parameter."""
        from paddleformers.transformers.aistudio_utils import aistudio_download

        mock_download.return_value = "/path/to/file"
        result = aistudio_download("repo_id", filename="model.safetensors", cache_dir="/tmp/cache")
        self.assertEqual(result, "/path/to/file")
        mock_download.assert_called_once_with(
            repo_id="repo_id",
            file_path="model.safetensors",
            revision="master",
            local_dir="/tmp/cache",
        )

    @patch("paddleformers.transformers.aistudio_utils.download")
    def test_aistudio_download_with_subfolder(self, mock_download):
        """Test aistudio_download with a subfolder parameter."""
        from paddleformers.transformers.aistudio_utils import aistudio_download

        mock_download.return_value = "/path/to/file"
        result = aistudio_download("repo_id", filename="model.safetensors", subfolder="checkpoint")
        self.assertEqual(result, "/path/to/file")
        mock_download.assert_called_once_with(
            repo_id="repo_id",
            file_path="checkpoint/model.safetensors",
            revision="master",
        )

    @patch("paddleformers.transformers.aistudio_utils.download")
    def test_aistudio_download_with_cache_dir_and_revision(self, mock_download):
        """Test aistudio_download with both cache_dir and revision."""
        from paddleformers.transformers.aistudio_utils import aistudio_download

        mock_download.return_value = "/path/to/file"
        result = aistudio_download(
            "repo_id",
            filename="model.safetensors",
            cache_dir="/tmp/cache",
            revision="v2.0",
        )
        self.assertEqual(result, "/path/to/file")
        mock_download.assert_called_once_with(
            repo_id="repo_id",
            file_path="model.safetensors",
            revision="v2.0",
            local_dir="/tmp/cache",
        )

    @patch("paddleformers.transformers.aistudio_utils.download")
    def test_aistudio_download_value_error(self, mock_download):
        """Test aistudio_download when download raises ValueError."""
        from paddleformers.transformers.aistudio_utils import aistudio_download

        mock_download.side_effect = ValueError("file not found")
        with self.assertRaises(EnvironmentError) as ctx:
            aistudio_download("repo_id", filename="model.safetensors")
        self.assertIn("Cannot find", str(ctx.exception))
        self.assertIn("model.safetensors", str(ctx.exception))
        self.assertIn("repo_id", str(ctx.exception))

    @patch("paddleformers.transformers.aistudio_utils.download")
    def test_aistudio_download_entry_not_found_error(self, mock_download):
        """Test aistudio_download when download raises EntryNotFoundError."""
        from paddleformers.transformers.aistudio_utils import (
            EntryNotFoundError,
            aistudio_download,
        )

        mock_download.side_effect = EntryNotFoundError("entry not found")
        with self.assertRaises(EnvironmentError) as ctx:
            aistudio_download("repo_id", filename="model.safetensors")
        self.assertIn("Cannot find the requested file", str(ctx.exception))
        self.assertIn("model.safetensors", str(ctx.exception))
        self.assertIn("repo_id", str(ctx.exception))

    @patch("paddleformers.transformers.aistudio_utils.download")
    def test_aistudio_download_http_error(self, mock_download):
        """Test aistudio_download when download raises HTTPError."""
        from paddleformers.transformers.aistudio_utils import aistudio_download

        mock_http_error = HTTPError("connection error")
        mock_download.side_effect = mock_http_error
        with self.assertRaises(EnvironmentError) as ctx:
            aistudio_download("repo_id", filename="model.safetensors")
        self.assertIn("specific connection error", str(ctx.exception))
        self.assertIn("repo_id", str(ctx.exception))

    @patch("paddleformers.transformers.aistudio_utils.download")
    def test_aistudio_download_generic_exception(self, mock_download):
        """Test aistudio_download when download raises a generic exception."""
        from paddleformers.transformers.aistudio_utils import aistudio_download

        mock_download.side_effect = RuntimeError("unexpected error")
        with self.assertRaises(EnvironmentError) as ctx:
            aistudio_download("repo_id", filename="model.safetensors")
        self.assertIn("Please make sure the", str(ctx.exception))
        self.assertIn("model.safetensors", str(ctx.exception))
        self.assertIn("repo_id", str(ctx.exception))

    @patch("paddleformers.transformers.aistudio_utils.download")
    def test_aistudio_download_no_filename(self, mock_download):
        """Test aistudio_download when filename is None (passed as None)."""
        from paddleformers.transformers.aistudio_utils import aistudio_download

        mock_download.return_value = "/path/to/file"
        result = aistudio_download("repo_id", filename=None)
        self.assertEqual(result, "/path/to/file")
        # filename should be None since subfolder default is ""
        mock_download.assert_called_once_with(
            repo_id="repo_id",
            file_path=None,
            revision="master",
        )

    @patch("paddleformers.transformers.aistudio_utils.download")
    def test_aistudio_download_with_extra_kwargs(self, mock_download):
        """Test aistudio_download passes through extra kwargs via **kwargs."""
        from paddleformers.transformers.aistudio_utils import aistudio_download

        mock_download.return_value = "/path/to/file"
        aistudio_download("repo_id", filename="model.bin", force_download=True)
        # extra kwargs that are not revision or cache_dir are not added to download_kwargs
        mock_download.assert_called_once_with(
            repo_id="repo_id",
            file_path="model.bin",
            revision="master",
        )
