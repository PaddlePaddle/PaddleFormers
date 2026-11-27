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

"""Download the exact initial checkpoint, or stream-verify it without disk use.

Standard library only; anonymous HTTPS; no AI Studio token required.
Existing files are never overwritten. Nothing is deserialized or executed.
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urlencode
from urllib.request import ProxyHandler, Request, build_opener, urlopen

CHUNK = 8 * 1024 * 1024


def check_stream(stream, spec, output=None, progress=False):
    digest = hashlib.sha256()
    count = 0
    last_report = time.monotonic()
    while True:
        block = stream.read(CHUNK)
        if not block:
            break
        count += len(block)
        if count > spec["bytes"]:
            raise ValueError("File exceeds expected size")
        digest.update(block)
        if output is not None:
            output.write(block)
        if progress and time.monotonic() - last_report >= 20:
            print(f"Verified {count:,} / {spec['bytes']:,} bytes", flush=True)
            last_report = time.monotonic()
    if count != spec["bytes"] or digest.hexdigest() != spec["sha256"]:
        raise ValueError("Checkpoint size or SHA256 mismatch")
    return {"bytes": count, "sha256": digest.hexdigest()}


def checkpoint_url(spec):
    path = quote(spec["remote_path"], safe="")
    query = urlencode({"ref": spec["revision"]})
    return f"https://git.aistudio.baidu.com/api/v1/repos/{spec['repo_id']}/media/{path}?{query}"


def fetch(spec, destination=None, opener=urlopen):
    if destination is not None:
        destination = Path(destination)
        if destination.is_symlink():
            raise ValueError("Refusing a symlink destination")
        if destination.exists():
            with destination.open("rb") as stream:
                result = check_stream(stream, spec)
            return {"operation": "existing_local_file_verified", **result}
        destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    output = None
    try:
        if destination is not None:
            fd, temporary = tempfile.mkstemp(prefix=".initial-download-", dir=destination.parent)
            output = os.fdopen(fd, "wb")
        request = Request(checkpoint_url(spec), headers={"User-Agent": "OLMo2-CINN-checkpoint-verifier/1"})
        with opener(request, timeout=60) as response:
            if response.status != 200:
                raise ValueError("Expected a full HTTP 200 response, not a partial download")
            result = check_stream(response, spec, output, progress=True)
        if output is not None:
            output.flush()
            os.fsync(output.fileno())
            output.close()
            output = None
            # Atomic publication without clobbering a file created concurrently.
            os.link(temporary, destination)
        return {
            "operation": "downloaded_and_verified" if destination else "remote_stream_verified",
            "anonymous": True,
            **result,
        }
    finally:
        if output is not None:
            output.close()
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=root / "checkpoint_source.json")
    parser.add_argument("--output", type=Path, default=root / "checkpoint/initial.pdparams")
    parser.add_argument(
        "--verify-remote-only", action="store_true", help="Read and hash every remote byte; do not save a second copy"
    )
    parser.add_argument("--no-proxy", action="store_true", help="Connect directly instead of using HTTP(S)_PROXY")
    parser.add_argument("--receipt", type=Path, help="Write a new verification receipt; never overwrite")
    args = parser.parse_args()
    if args.receipt and args.receipt.exists():
        parser.error("Receipt already exists")
    spec = json.loads(args.manifest.read_text())
    started = time.monotonic()
    opener = build_opener(ProxyHandler({})).open if args.no_proxy else urlopen
    result = fetch(spec, None if args.verify_remote_only else args.output, opener)
    result.update(
        {
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "proxy_environment_bypassed": args.no_proxy,
            "source": spec,
        }
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        with args.receipt.open("x") as stream:
            stream.write(rendered)
    print(rendered, end="", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Redirected storage URLs may be signed: never print exception URLs.
        print(f"Checkpoint verification failed: {type(error).__name__}", file=sys.stderr)
        raise SystemExit(1)
