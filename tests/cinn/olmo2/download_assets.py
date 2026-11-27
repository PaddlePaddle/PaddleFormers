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

"""Prepare publicly downloadable, hash-pinned tokenizer/config assets; no GPU required."""
import argparse
import json
import sys
import tarfile
import tempfile
from pathlib import Path
from urllib.request import ProxyHandler, build_opener, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tests.cinn.olmo2.download_checkpoint import check_stream, fetch
from tests.cinn.olmo2.verify_assets import verify_assets

RESOURCE = {
    "repo_id": "PaddleClas/olmo2",
    "remote_path": "cinn/olmo2/resources.tar.gz",
    "revision": "823ba1a3427a27fed6d71824325475e7f6fc8b4e",
    "bytes": 2535210,
    "sha256": "1a51eeea0892f7ca10faf7317277b38c797bfa08a51fd81dc9ea4e5bd182d5d4",
}


def member_bytes(archive, name, limit):
    matches = [item for item in archive.getmembers() if item.name == name]
    if len(matches) != 1 or not matches[0].isfile() or matches[0].size > limit:
        raise ValueError(f"Expected one bounded regular archive member: {name}")
    with archive.extractfile(matches[0]) as stream:
        return stream.read(limit + 1)


def unpack_assets(archive_path, output, manifest):
    """Never extract arbitrary archive paths; copy only the declared regular files."""
    output.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive_path, "r:gz") as archive:
        original = member_bytes(archive, "assets/config.json", 1024 * 1024)
        raw = json.loads(original)
        if raw.get("num_hidden_layers") != 16:
            raise ValueError("Expected original 16-layer configuration")
        raw["num_hidden_layers"] = 2
        (output / "original_config.json").write_bytes(original)
        (output / "config.json").write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
        for entry in manifest:
            if entry["root"] != "assets" or entry["file"] in ("config.json", "original_config.json"):
                continue
            name = entry["file"]
            if Path(name).name != name:
                raise ValueError("Asset filename must not contain a directory")
            content = member_bytes(archive, f"assets/{name}", entry["bytes"])
            (output / name).write_bytes(content)
        notices = output / "licenses"
        notices.mkdir()
        for name in ("GSM8K-MIT.txt", "OLMo-Apache-2.0.txt"):
            (notices / name).write_bytes(member_bytes(archive, f"licenses/{name}", 64 * 1024))
    verify_assets([item for item in manifest if item["root"] == "assets"], {"assets": output})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New asset directory; refuses to overwrite")
    parser.add_argument("--no-proxy", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    entry = Path(__file__).resolve().parent
    manifest = json.loads((entry / "asset_manifest.json").read_text())
    opener = build_opener(ProxyHandler({})).open if args.no_proxy else urlopen
    with tempfile.TemporaryDirectory(prefix="olmo2-assets-", dir=args.output.parent) as temporary:
        archive_path = Path(temporary) / "resources.tar.gz"
        fetch(RESOURCE, archive_path, opener)
        with archive_path.open("rb") as stream:
            check_stream(stream, RESOURCE)
        prepared = Path(temporary) / "assets"
        unpack_assets(archive_path, prepared, manifest)
        # Reserve the target rather than replacing any existing directory.
        args.output.mkdir(exist_ok=False)
        for path in prepared.iterdir():
            path.rename(args.output / path.name)
    print(f"Config/tokenizer SHA256 verified: {args.output}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Redirected object-storage URLs may carry temporary signatures.
        print(f"Asset download failed: {type(error).__name__}", file=sys.stderr)
        raise SystemExit(1)
