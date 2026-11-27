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
import json
import os
import sys
from pathlib import Path


def verify_assets(manifest, roots):
    for entry in manifest:
        if entry["root"] not in roots:
            raise ValueError(f"Unknown asset root: {entry['root']}")
        relative = Path(entry["file"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Asset path must stay inside its declared root: {relative}")
        path = roots[entry["root"]] / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != entry["sha256"] or path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Asset mismatch: {path}")


def main():
    root = Path(__file__).parent
    roots = {"checkpoint": Path(sys.argv[1]), "assets": Path(os.environ["CINN_ASSET_DIR"]), "repo": root}
    manifest = json.loads((root / "asset_manifest.json").read_text())
    verify_assets(manifest, roots)
    print("All checkpoint/data/config SHA256 checks passed.")


if __name__ == "__main__":
    main()
