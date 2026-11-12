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

root = Path(__file__).parent
weights = Path(sys.argv[1])
manifest = json.loads((root / "asset_manifest.json").read_text())
for entry in manifest:
    path = (weights if entry["kind"] == "weight" else Path(os.environ["CINN_RESOURCE_DIR"])) / entry["file"]
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != entry["sha256"] or path.stat().st_size != entry["bytes"]:
        raise ValueError(f"Asset mismatch: {path}")
print("All checkpoint/data SHA256 checks passed; model config is defined in run_model.py.")
