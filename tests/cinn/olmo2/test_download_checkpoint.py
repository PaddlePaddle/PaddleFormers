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

"""Small offline tests; no Paddle, network, GPU, or large file needed."""
import hashlib
import io
import tempfile
import unittest
from pathlib import Path

from tests.cinn.olmo2.download_checkpoint import fetch


class Response(io.BytesIO):
    status = 200


class DownloadTest(unittest.TestCase):
    def setUp(self):
        self.data = b"initial checkpoint bytes"
        self.spec = dict(
            repo_id="owner/repo",
            revision="a" * 40,
            remote_path="cinn/initial.pdparams",
            bytes=len(self.data),
            sha256=hashlib.sha256(self.data).hexdigest(),
        )
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "initial.pdparams"

    def opener(self, request, timeout):
        self.assertIn("ref=" + "a" * 40, request.full_url)
        self.assertNotIn("Authorization", request.headers)
        return Response(self.data)

    def test_stream_all_bytes_without_disk(self):
        result = fetch(self.spec, opener=self.opener)
        self.assertTrue(result["anonymous"])
        self.assertEqual(result["sha256"], self.spec["sha256"])
        self.assertEqual(list(self.path.parent.iterdir()), [])

    def test_download_and_verify_existing_without_network(self):
        fetch(self.spec, self.path, self.opener)
        self.assertEqual(self.path.read_bytes(), self.data)

        def forbidden(*args, **kwargs):
            self.fail("Existing valid file should not require network")

        self.assertEqual(fetch(self.spec, self.path, forbidden)["operation"], "existing_local_file_verified")

    def test_existing_wrong_file_preserved(self):
        self.path.write_bytes(b"user file")
        with self.assertRaises(ValueError):
            fetch(self.spec, self.path, self.opener)
        self.assertEqual(self.path.read_bytes(), b"user file")

    def test_corrupt_or_partial_download_cleans_temporary(self):
        for data in (self.data[:-1], self.data + b"x", b"x" * len(self.data)):
            with self.subTest(data=data):
                with self.assertRaises(ValueError):
                    fetch(self.spec, self.path, lambda *a, **kw: Response(data))
                self.assertEqual(list(self.path.parent.iterdir()), [])

    def test_network_failure_cleanup(self):
        def fail(*args, **kwargs):
            raise TimeoutError()

        with self.assertRaises(TimeoutError):
            fetch(self.spec, self.path, fail)
        self.assertEqual(list(self.path.parent.iterdir()), [])

    def test_symlink_rejected(self):
        self.path.symlink_to("missing")
        with self.assertRaises(ValueError):
            fetch(self.spec, self.path, self.opener)
        self.assertTrue(self.path.is_symlink())

    def test_non_full_http_response_rejected(self):
        response = Response(self.data)
        response.status = 206
        with self.assertRaises(ValueError):
            fetch(self.spec, self.path, lambda *a, **kw: response)
        self.assertEqual(list(self.path.parent.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
