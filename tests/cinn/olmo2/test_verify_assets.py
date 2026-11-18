# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import hashlib
import tempfile
import unittest
from pathlib import Path

from tests.cinn.olmo2.verify_assets import verify_assets


class AssetRootsTest(unittest.TestCase):
    def test_roots_are_explicit_even_for_identical_filenames(self):
        with tempfile.TemporaryDirectory() as directory:
            roots = {key: Path(directory) / key for key in ("checkpoint", "assets", "repo")}
            manifest = []
            for key, root in roots.items():
                root.mkdir()
                content = key.encode()
                (root / "config.json").write_bytes(content)
                manifest.append(
                    dict(root=key, file="config.json", bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
                )
            verify_assets(manifest, roots)
            (roots["assets"] / "config.json").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "Asset mismatch"):
                verify_assets(manifest, roots)

    def test_unknown_root_missing_file_and_escape_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            roots = {"assets": Path(directory)}
            entry = dict(root="assets", file="missing", bytes=0, sha256=hashlib.sha256(b"").hexdigest())
            with self.assertRaises(FileNotFoundError):
                verify_assets([entry], roots)
            with self.assertRaisesRegex(ValueError, "Unknown asset root"):
                verify_assets([{**entry, "root": "typo"}], roots)
            for name in ("../outside", "/absolute"):
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "declared root"):
                    verify_assets([{**entry, "file": name}], roots)
