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

import argparse
import hashlib
import io
import json
import runpy
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

helpers = runpy.run_path(str(Path(__file__).with_name("prepare_data.py")))


class PrepareDataTest(unittest.TestCase):
    def test_npz_roundtrip_and_historical_headers(self):
        original_header = zipfile.ZipInfo.FileHeader
        arrays = dict(images=np.arange(24, dtype="uint8").reshape(2, 3, 4), labels=np.array([1, 2], dtype="int64"))
        for compressed in (False, True):
            with self.subTest(compressed=compressed):
                output = io.BytesIO()
                helpers["write_npz"](output, arrays, compressed)
                data = output.getvalue()
                self.assertEqual(struct.unpack_from("<H", data, 4)[0], 45)
                self.assertEqual(struct.unpack_from("<II", data, 18), (0xFFFFFFFF, 0xFFFFFFFF))
                with zipfile.ZipFile(io.BytesIO(data)) as archive:
                    self.assertEqual([member.create_version for member in archive.infolist()], [45, 45])
                with np.load(io.BytesIO(data), allow_pickle=False) as result:
                    for name, expected in arrays.items():
                        np.testing.assert_array_equal(result[name], expected)
        self.assertIs(zipfile.ZipInfo.FileHeader, original_header)

    def test_preprocessing_rgb_center_crop(self):
        blob = io.BytesIO()
        Image.new("RGB", (400, 200), (17, 23, 42)).save(blob, format="PNG")
        image = helpers["preprocess"](blob.getvalue())
        self.assertEqual(image.shape, (224, 224, 3))
        self.assertEqual(image.dtype, np.uint8)
        np.testing.assert_array_equal(image[0, 0], [17, 23, 42])

    def test_invalid_selection_and_existing_output_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection = root / "selection.json"
            selection.write_text("{}\n")
            metadata = root / "assets.json"
            entry = dict(file="data/manifest.json", bytes=3, sha256=hashlib.sha256(b"{}\n").hexdigest())
            metadata.write_text(json.dumps([entry]))
            args = argparse.Namespace(
                layout="classification", selection=selection, asset_manifest=metadata, output=root / "output"
            )
            selection.write_text('{"changed": true}\n')
            with self.assertRaisesRegex(ValueError, "Selection manifest"):
                helpers["rebuild"](args)
            self.assertFalse(args.output.exists())
            selection.write_text("{}\n")
            args.output.mkdir()
            with self.assertRaises(FileExistsError):
                helpers["rebuild"](args)

    def test_object_arrays_not_serialized(self):
        with self.assertRaises(ValueError):
            helpers["write_npz"](io.BytesIO(), {"bad": np.array([object()], dtype=object)}, False)
