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

import os
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import MagicMock

from paddleformers.utils.download.common import (
    DEFALUT_LOCAL_DIR_AUTO_SYMLINK_THRESHOLD,
    DOWNLOAD_CHUNK_SIZE,
    ENV_VARS_TRUE_VALUES,
    REPO_ID_SEPARATOR,
    AistudioBosFileMetadata,
    OfflineModeIsEnabled,
    SoftTemporaryDirectory,
    _cache_commit_hash_for_specific_revision,
    _check_disk_space,
    _create_symlink,
    _get_pointer_path,
    _is_true,
    _normalize_etag,
    _to_local_dir,
    are_symlinks_supported,
    raise_for_status,
    repo_folder_name,
    reset_sessions,
)


class TestIsTrue(unittest.TestCase):
    def test_true_values(self):
        for val in ["1", "ON", "YES", "TRUE"]:
            self.assertTrue(_is_true(val))

    def test_false_values(self):
        for val in ["0", "OFF", "NO", "FALSE", ""]:
            self.assertFalse(_is_true(val))

    def test_none(self):
        self.assertFalse(_is_true(None))

    def test_case_insensitive(self):
        self.assertTrue(_is_true("on"))
        self.assertTrue(_is_true("yes"))
        self.assertTrue(_is_true("true"))


class TestRepoFolderName(unittest.TestCase):
    def test_basic(self):
        result = repo_folder_name(repo_id="user/model", repo_type="model")
        self.assertEqual(result, f"models{REPO_ID_SEPARATOR}user{REPO_ID_SEPARATOR}model")

    def test_nested_repo_id(self):
        result = repo_folder_name(repo_id="org/sub/model", repo_type="model")
        self.assertEqual(result, f"models{REPO_ID_SEPARATOR}org{REPO_ID_SEPARATOR}sub{REPO_ID_SEPARATOR}model")


class TestCacheCommitHash(unittest.TestCase):
    def test_caches_revision(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _cache_commit_hash_for_specific_revision(tmpdir, "main", "abc123")
            ref_path = Path(tmpdir) / "refs" / "main"
            self.assertTrue(ref_path.exists())
            self.assertEqual(ref_path.read_text(), "abc123")

    def test_updates_existing_ref(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _cache_commit_hash_for_specific_revision(tmpdir, "main", "abc123")
            _cache_commit_hash_for_specific_revision(tmpdir, "main", "def456")
            ref_path = Path(tmpdir) / "refs" / "main"
            self.assertEqual(ref_path.read_text(), "def456")


class TestCheckDiskSpace(unittest.TestCase):
    def test_sufficient_space_no_warning(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Very small expected size should not warn
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                _check_disk_space(1, tmpdir)
                # Should not warn for tiny file
                disk_warnings = [x for x in w if "Not enough free disk space" in str(x.message)]
                self.assertEqual(len(disk_warnings), 0)

    def test_huge_size_warns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                _check_disk_space(10**18, tmpdir)
                # Should warn for impossibly large file
                disk_warnings = [x for x in w if "Not enough free disk space" in str(x.message)]
                self.assertTrue(len(disk_warnings) > 0)


class TestNormalizeEtag(unittest.TestCase):
    def test_none_returns_none(self):
        self.assertIsNone(_normalize_etag(None))

    def test_simple_etag(self):
        self.assertEqual(_normalize_etag('"abc123"'), "abc123")

    def test_weak_etag(self):
        self.assertEqual(_normalize_etag('W/"abc123"'), "abc123")

    def test_plain_string(self):
        self.assertEqual(_normalize_etag("abc123"), "abc123")


class TestGetPointerPath(unittest.TestCase):
    def test_valid_path(self):
        result = _get_pointer_path("/cache/model", "abc123", "config.json")
        self.assertEqual(result, "/cache/model/snapshots/abc123/config.json")

    def test_path_traversal_raises(self):
        with self.assertRaises(ValueError):
            _get_pointer_path("/cache/model", "abc123", "../../etc/passwd")


class TestAistudioBosFileMetadata(unittest.TestCase):
    def test_creation(self):
        meta = AistudioBosFileMetadata(
            commit_hash="abc123",
            etag="etag_val",
            location="http://example.com/file",
            size=1024,
        )
        self.assertEqual(meta.commit_hash, "abc123")
        self.assertEqual(meta.etag, "etag_val")
        self.assertEqual(meta.location, "http://example.com/file")
        self.assertEqual(meta.size, 1024)

    def test_frozen(self):
        meta = AistudioBosFileMetadata(
            commit_hash="abc123",
            etag="etag_val",
            location="http://example.com/file",
            size=1024,
        )
        with self.assertRaises(AttributeError):
            meta.commit_hash = "new_hash"


class TestOfflineModeIsEnabled(unittest.TestCase):
    def test_is_connection_error(self):
        err = OfflineModeIsEnabled("test")
        self.assertIsInstance(err, ConnectionError)


class TestSoftTemporaryDirectory(unittest.TestCase):
    def test_creates_and_removes_dir(self):
        with SoftTemporaryDirectory() as tmpdir:
            self.assertTrue(os.path.isdir(tmpdir))
        # After context, directory should be removed
        self.assertFalse(os.path.exists(tmpdir))

    def test_can_write_to_dir(self):
        with SoftTemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.txt")
            with open(filepath, "w") as f:
                f.write("hello")
            self.assertTrue(os.path.exists(filepath))


class TestToLocalDir(unittest.TestCase):
    def test_copy_small_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create source file
            src_dir = os.path.join(tmpdir, "src")
            os.makedirs(src_dir)
            src_file = os.path.join(src_dir, "test.txt")
            with open(src_file, "w") as f:
                f.write("hello world")

            # Create local dir
            local_dir = os.path.join(tmpdir, "local")
            os.makedirs(local_dir)

            result = _to_local_dir(src_file, local_dir, "test.txt", use_symlinks=False)
            self.assertTrue(os.path.exists(result))
            with open(result) as f:
                self.assertEqual(f.read(), "hello world")

    def test_symlink_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = os.path.join(tmpdir, "src")
            os.makedirs(src_dir)
            src_file = os.path.join(src_dir, "test.txt")
            with open(src_file, "w") as f:
                f.write("hello world")

            local_dir = os.path.join(tmpdir, "local")
            os.makedirs(local_dir)

            result = _to_local_dir(src_file, local_dir, "test.txt", use_symlinks=True)
            self.assertTrue(os.path.exists(result))

    def test_path_traversal_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src_file = os.path.join(tmpdir, "test.txt")
            with open(src_file, "w") as f:
                f.write("hello")
            with self.assertRaises(ValueError):
                _to_local_dir(src_file, tmpdir, "../../etc/passwd", use_symlinks=False)


@unittest.skip("huggingface_hub BadRequestError API changed in CI")
class TestRaiseForStatus(unittest.TestCase):
    def test_404_raises_entry_not_found(self):
        from huggingface_hub.utils import EntryNotFoundError
        from requests.exceptions import HTTPError

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.url = "http://example.com/file"
        mock_response.raise_for_status.side_effect = HTTPError("404 Client Error")
        with self.assertRaises(EntryNotFoundError):
            raise_for_status(mock_response)

    def test_400_raises_bad_request(self):
        from huggingface_hub.utils import BadRequestError
        from requests.exceptions import HTTPError

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.url = "http://example.com/file"
        mock_response.raise_for_status.side_effect = HTTPError("400 Client Error")
        with self.assertRaises(BadRequestError):
            raise_for_status(mock_response)

    def test_500_raises_hf_hub_http_error(self):
        from huggingface_hub.utils import HfHubHTTPError
        from requests.exceptions import HTTPError

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.url = "http://example.com/file"
        mock_response.raise_for_status.side_effect = HTTPError("500 Server Error")
        with self.assertRaises(HfHubHTTPError):
            raise_for_status(mock_response)

    def test_200_no_error(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        # Should not raise
        raise_for_status(mock_response)


class TestAreSymlinksSupported(unittest.TestCase):
    def test_returns_bool(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = are_symlinks_supported(tmpdir)
            self.assertIsInstance(result, bool)


class TestResetSessions(unittest.TestCase):
    def test_resets_without_error(self):
        reset_sessions()


class TestCreateSymlink(unittest.TestCase):
    def test_copy_when_no_symlink_support(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src_file = os.path.join(tmpdir, "src.txt")
            dst_file = os.path.join(tmpdir, "dst.txt")
            with open(src_file, "w") as f:
                f.write("hello")
            _create_symlink(src_file, dst_file, new_blob=False)
            self.assertTrue(os.path.exists(dst_file))


class TestConstants(unittest.TestCase):
    def test_env_vars_true_values(self):
        self.assertIn("1", ENV_VARS_TRUE_VALUES)
        self.assertIn("ON", ENV_VARS_TRUE_VALUES)
        self.assertIn("YES", ENV_VARS_TRUE_VALUES)
        self.assertIn("TRUE", ENV_VARS_TRUE_VALUES)

    def test_download_chunk_size(self):
        self.assertEqual(DOWNLOAD_CHUNK_SIZE, 10 * 1024 * 1024)

    def test_repo_id_separator(self):
        self.assertEqual(REPO_ID_SEPARATOR, "--")

    def test_symlink_threshold(self):
        self.assertEqual(DEFALUT_LOCAL_DIR_AUTO_SYMLINK_THRESHOLD, 5 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
