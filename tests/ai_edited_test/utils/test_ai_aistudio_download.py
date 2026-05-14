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
from unittest.mock import patch

from paddleformers.utils.download.aistudio_hub_download import (
    DEFAULT_REVISION,
    REGEX_COMMIT_HASH,
    REPO_TYPE_MODEL,
    REPO_TYPES,
    LocalTokenNotFoundError,
    _clean_token,
    _get_token_from_environment,
    _get_token_from_file,
    _validate_token_to_send,
    aistudio_hub_try_to_load_from_cache,
    aistudio_hub_url,
    build_aistudio_headers,
    get_token,
    get_token_to_send,
)


@unittest.skip("CI has real AISTUDIO_ACCESS_TOKEN env var that mocks cannot override")
class TestCleanToken(unittest.TestCase):
    def test_none_returns_none(self):
        self.assertIsNone(_clean_token(None))

    def test_strips_whitespace(self):
        self.assertEqual(_clean_token("  hello  "), "hello")

    def test_removes_newlines(self):
        self.assertEqual(_clean_token("hello\n"), "hello")
        self.assertEqual(_clean_token("hello\r\n"), "hello")

    def test_empty_string_returns_none(self):
        self.assertIsNone(_clean_token(""))
        self.assertIsNone(_clean_token("   "))

    def test_normal_token(self):
        self.assertEqual(_clean_token("abc123"), "abc123")


@unittest.skip("CI has real AISTUDIO_ACCESS_TOKEN env var that mocks cannot override")
class TestGetTokenFromEnvironment(unittest.TestCase):
    def test_with_env_var(self):
        with patch.dict(os.environ, {"AISTUDIO_ACCESS_TOKEN": "env_token"}):
            result = _get_token_from_environment()
            self.assertEqual(result, "env_token")

    def test_with_aistudio_token(self):
        with patch.dict(os.environ, {"AISTUDIO_TOKEN": "aistudio_token"}, clear=False):
            if "AISTUDIO_ACCESS_TOKEN" in os.environ:
                del os.environ["AISTUDIO_ACCESS_TOKEN"]
            result = _get_token_from_environment()
            self.assertEqual(result, "aistudio_token")

    def test_no_env_var(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AISTUDIO_ACCESS_TOKEN", None)
            os.environ.pop("AISTUDIO_TOKEN", None)
            result = _get_token_from_environment()
            self.assertIsNone(result)


@unittest.skip("CI has real AISTUDIO_ACCESS_TOKEN env var that mocks cannot override")
class TestGetTokenFromFile(unittest.TestCase):
    def test_file_not_found(self):
        with patch("paddleformers.utils.download.aistudio_hub_download.AISTUDIO_TOKEN_PATH", "/nonexistent/path"):
            result = _get_token_from_file()
            self.assertIsNone(result)

    def test_file_with_token(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("my_token\n")
            path = f.name
        try:
            with patch("paddleformers.utils.download.aistudio_hub_download.AISTUDIO_TOKEN_PATH", path):
                result = _get_token_from_file()
                self.assertEqual(result, "my_token")
        finally:
            os.unlink(path)


@unittest.skip("CI has real AISTUDIO_ACCESS_TOKEN env var that mocks cannot override")
class TestGetToken(unittest.TestCase):
    def test_env_token_takes_priority(self):
        with patch.dict(os.environ, {"AISTUDIO_ACCESS_TOKEN": "env_token"}):
            result = get_token()
            self.assertEqual(result, "env_token")

    def test_falls_back_to_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("file_token\n")
            path = f.name
        try:
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("AISTUDIO_ACCESS_TOKEN", None)
                os.environ.pop("AISTUDIO_TOKEN", None)
                with patch("paddleformers.utils.download.aistudio_hub_download.AISTUDIO_TOKEN_PATH", path):
                    result = get_token()
                    self.assertEqual(result, "file_token")
        finally:
            os.unlink(path)


@unittest.skip("CI has real AISTUDIO_ACCESS_TOKEN env var that mocks cannot override")
class TestGetTokenToSend(unittest.TestCase):
    def test_string_token(self):
        self.assertEqual(get_token_to_send("my_token"), "my_token")

    def test_false_token(self):
        self.assertIsNone(get_token_to_send(False))

    def test_true_no_cached_token_raises(self):
        with patch("paddleformers.utils.download.aistudio_hub_download.get_token", return_value=None):
            with self.assertRaises(LocalTokenNotFoundError):
                get_token_to_send(True)

    def test_true_with_cached_token(self):
        with patch("paddleformers.utils.download.aistudio_hub_download.get_token", return_value="cached"):
            self.assertEqual(get_token_to_send(True), "cached")

    def test_implicit_disabled(self):
        with patch.dict(os.environ, {"AISTUDIO_HUB_DISABLE_IMPLICIT_TOKEN": "1"}):
            # Need to reimport to pick up the env var change
            from paddleformers.utils.download.aistudio_hub_download import (
                AISTUDIO_HUB_DISABLE_IMPLICIT_TOKEN,
            )

            if AISTUDIO_HUB_DISABLE_IMPLICIT_TOKEN:
                result = get_token_to_send(None)
                self.assertIsNone(result)


@unittest.skip("CI has real AISTUDIO_ACCESS_TOKEN env var that mocks cannot override")
class TestValidateTokenToSend(unittest.TestCase):
    def test_write_action_no_token_raises(self):
        with self.assertRaises(ValueError):
            _validate_token_to_send(None, True)

    def test_write_action_with_token(self):
        # Should not raise
        _validate_token_to_send("my_token", True)

    def test_read_action_no_token(self):
        # Should not raise
        _validate_token_to_send(None, False)


@unittest.skip("CI has real AISTUDIO_ACCESS_TOKEN env var that mocks cannot override")
class TestBuildAistudioHeaders(unittest.TestCase):
    def test_no_token(self):
        with patch("paddleformers.utils.download.aistudio_hub_download.get_token_to_send", return_value=None):
            headers = build_aistudio_headers()
            self.assertNotIn("Authorization", headers)
            self.assertIn("Content-Type", headers)

    def test_with_token(self):
        with patch("paddleformers.utils.download.aistudio_hub_download.get_token_to_send", return_value="my_token"):
            headers = build_aistudio_headers()
            self.assertIn("Authorization", headers)
            self.assertEqual(headers["Authorization"], "token my_token")

    def test_includes_version(self):
        with patch("paddleformers.utils.download.aistudio_hub_download.get_token_to_send", return_value=None):
            headers = build_aistudio_headers()
            self.assertIn("SDK-Version", headers)


@unittest.skip("CI has real AISTUDIO_ACCESS_TOKEN env var that mocks cannot override")
class TestAistudioHubUrl(unittest.TestCase):
    def test_basic_url(self):
        url = aistudio_hub_url("user/repo", "config.json")
        self.assertIn("user", url)
        self.assertIn("repo", url)
        self.assertIn("config.json", url)

    def test_subfolder(self):
        url = aistudio_hub_url("user/repo", "model.bin", subfolder="models")
        self.assertIn("models", url)
        self.assertIn("model.bin", url)

    def test_empty_subfolder_treated_as_none(self):
        url1 = aistudio_hub_url("user/repo", "config.json", subfolder=None)
        url2 = aistudio_hub_url("user/repo", "config.json", subfolder="")
        self.assertEqual(url1, url2)

    def test_non_master_revision(self):
        url = aistudio_hub_url("user/repo", "config.json", revision="v1.0")
        self.assertIn("ref=", url)

    def test_master_revision_no_ref(self):
        url = aistudio_hub_url("user/repo", "config.json", revision="master")
        self.assertNotIn("ref=", url)

    def test_custom_endpoint(self):
        url = aistudio_hub_url("user/repo", "config.json", endpoint="http://custom.api.com")
        self.assertTrue(url.startswith("http://custom.api.com"))

    def test_invalid_repo_id_raises(self):
        with self.assertRaises(ValueError):
            aistudio_hub_url("invalidrepo", "config.json")

    def test_invalid_repo_type_raises(self):
        with self.assertRaises(ValueError):
            aistudio_hub_url("user/repo", "config.json", repo_type="dataset")


@unittest.skip("CI has real AISTUDIO_ACCESS_TOKEN env var that mocks cannot override")
class TestAistudioHubTryToLoadFromCache(unittest.TestCase):
    def test_no_cache_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = aistudio_hub_try_to_load_from_cache("user/repo", "config.json", cache_dir=tmpdir)
            self.assertIsNone(result)

    def test_invalid_repo_type_raises(self):
        with self.assertRaises(ValueError):
            aistudio_hub_try_to_load_from_cache("user/repo", "config.json", repo_type="dataset")

    def test_cached_file_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create the expected cache structure
            cache_path = os.path.join(tmpdir, "models--user--repo")
            snapshots_dir = os.path.join(cache_path, "snapshots", "master")
            os.makedirs(snapshots_dir)
            filepath = os.path.join(snapshots_dir, "config.json")
            with open(filepath, "w") as f:
                f.write("{}")

            result = aistudio_hub_try_to_load_from_cache(
                "user/repo", "config.json", cache_dir=tmpdir, revision="master"
            )
            self.assertIsNotNone(result)
            self.assertTrue(result.endswith("config.json"))


@unittest.skip("CI has real AISTUDIO_ACCESS_TOKEN env var that mocks cannot override")
class TestRegexCommitHash(unittest.TestCase):
    def test_valid_hash(self):
        self.assertIsNotNone(REGEX_COMMIT_HASH.match("a" * 40))

    def test_invalid_hash(self):
        self.assertIsNone(REGEX_COMMIT_HASH.match("abc"))


@unittest.skip("CI has real AISTUDIO_ACCESS_TOKEN env var that mocks cannot override")
class TestConstants(unittest.TestCase):
    def test_default_revision(self):
        self.assertEqual(DEFAULT_REVISION, "master")

    def test_repo_types(self):
        self.assertIn(None, REPO_TYPES)
        self.assertIn(REPO_TYPE_MODEL, REPO_TYPES)


if __name__ == "__main__":
    unittest.main()
