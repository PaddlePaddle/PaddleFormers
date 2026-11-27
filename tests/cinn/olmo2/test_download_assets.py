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

import hashlib
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from tests.cinn.olmo2.download_assets import member_bytes, unpack_assets


class AssetDownloadTest(unittest.TestCase):
    def make_archive(self, path, members):
        with tarfile.open(path, "w:gz") as archive:
            for name, content, kind in members:
                info = tarfile.TarInfo(name)
                info.type = kind
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))

    def test_reject_link_duplicate_missing_and_oversize_members(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.tar.gz"
            for members in (
                [("assets/config.json", b"", tarfile.SYMTYPE)],
                [("assets/config.json", b"x", tarfile.REGTYPE)] * 2,
                [("../config.json", b"x", tarfile.REGTYPE)],
                [("assets/config.json", b"1234", tarfile.REGTYPE)],
            ):
                with self.subTest(members=members):
                    self.make_archive(path, members)
                    with tarfile.open(path) as archive, self.assertRaises(ValueError):
                        member_bytes(archive, "assets/config.json", 3)

    def test_prepare_exact_depth_only_config_and_refuse_overwrite(self):
        original = dict(num_hidden_layers=16, hidden_size=2048)
        expected = {
            "original_config.json": (json.dumps(original, indent=2) + "\n").encode(),
            "config.json": (json.dumps(dict(original, num_hidden_layers=2), indent=2) + "\n").encode(),
            "tokenizer.json": b'{"tokenizer":"fixture"}',
        }
        manifest = [
            dict(root="assets", file=name, bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
            for name, content in expected.items()
        ]
        members = [
            ("assets/config.json", expected["original_config.json"], tarfile.REGTYPE),
            ("assets/tokenizer.json", expected["tokenizer.json"], tarfile.REGTYPE),
            ("licenses/GSM8K-MIT.txt", b"fixture license", tarfile.REGTYPE),
            ("licenses/OLMo-Apache-2.0.txt", b"fixture license", tarfile.REGTYPE),
            ("../../outside", b"must not be extracted", tarfile.REGTYPE),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "test.tar.gz"
            self.make_archive(archive, members)
            output = root / "assets"
            unpack_assets(archive, output, manifest)
            for name, content in expected.items():
                self.assertEqual((output / name).read_bytes(), content)
            self.assertEqual({p.name for p in root.iterdir()}, {"test.tar.gz", "assets"})
            with self.assertRaises(FileExistsError):
                unpack_assets(archive, output, manifest)

    def test_manifest_mismatch_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            members = [("assets/config.json", b'{"num_hidden_layers":16}', tarfile.REGTYPE)]
            for name in ("GSM8K-MIT.txt", "OLMo-Apache-2.0.txt"):
                members.append((f"licenses/{name}", b"license", tarfile.REGTYPE))
            self.make_archive(root / "test.tar.gz", members)
            manifest = [dict(root="assets", file="config.json", bytes=1, sha256="0" * 64)]
            with self.assertRaisesRegex(ValueError, "Asset mismatch"):
                unpack_assets(root / "test.tar.gz", root / "assets", manifest)


if __name__ == "__main__":
    unittest.main()
