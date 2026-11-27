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

"""Rebuild pinned test inputs from user-provided, licensed ImageNet archives.

No download, random sample selection, label remapping, or image redistribution.
The resulting NPZ must match the original test manifest byte for byte.
"""

import argparse
import hashlib
import io
import json
import struct
import tarfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat


def preprocess(blob):
    with Image.open(io.BytesIO(blob)) as image:
        image = image.convert("RGB")
        width, height = image.size
        size = (224, int(height * 224 / width)) if width < height else (int(width * 224 / height), 224)
        image = image.resize(size, Image.Resampling.BILINEAR)
        left, top = (size[0] - 224) // 2, (size[1] - 224) // 2
        return np.asarray(image.crop((left, top, left + 224, top + 224)), dtype="uint8")


def labels_from_devkit(path):
    with tarfile.open(path) as archive:
        meta = next(member for member in archive if member.name.endswith("/meta.mat"))
        synsets = loadmat(
            io.BytesIO(archive.extractfile(meta).read()),
            squeeze_me=True,
            struct_as_record=False,
        )["synsets"]
        leaves = [synset for synset in synsets if int(synset.num_children) == 0]
        labels = {name: index for index, name in enumerate(sorted(str(s.WNID) for s in leaves))}
        if len(labels) != 1000:
            raise ValueError("Expected the original 1000-class ILSVRC2012 devkit")
        ids = {int(s.ILSVRC2012_ID): str(s.WNID) for s in leaves}
        truth = next(
            member for member in archive.getmembers() if member.name.endswith("ILSVRC2012_validation_ground_truth.txt")
        )
        validation = [labels[ids[int(value)]] for value in archive.extractfile(truth).read().split()]
    return labels, validation


class HistoricalZipInfo(zipfile.ZipInfo):
    def FileHeader(self, zip64=None):
        # Newer Python may emit 32-bit sizes for small force_zip64 members.
        # The pinned archives use ZIP64 sentinels and version 4.5 throughout.
        # Scope compatibility to these entries; never patch global zipfile.
        if zip64:
            self.create_version = max(self.create_version, 45)
            self.extract_version = max(self.extract_version, 45)
        header = super().FileHeader(zip64)
        if zip64:
            header = bytearray(header)
            struct.pack_into("<II", header, 18, 0xFFFFFFFF, 0xFFFFFFFF)
            header = bytes(header)
        return header


def write_npz(buffer, arrays, compressed):
    # Preserve historical NumPy NPZ ZIP64 headers. Different NumPy versions
    # may otherwise store identical NPY payloads with different ZIP headers.
    compression = zipfile.ZIP_DEFLATED if compressed else zipfile.ZIP_STORED
    with zipfile.ZipFile(buffer, mode="w", compression=compression, allowZip64=True) as archive:
        for name, array in arrays.items():
            info = HistoricalZipInfo(name + ".npy")
            info.compress_type = compression
            with archive.open(info, mode="w", force_zip64=True) as member:
                np.lib.format.write_array(member, array, allow_pickle=False)


def rebuild(args):
    manifest = json.loads(args.asset_manifest.read_text())
    entries = {entry["file"]: entry for entry in manifest}
    selection_name = "images_manifest.json" if args.layout == "multimodal" else "manifest.json"
    data_name = "images24.npz" if args.layout == "multimodal" else "data.npz"
    # Restore the historical JSON serialization before checking its digest:
    # classification had a final newline; the multimodal selection did not.
    selection_bytes = json.dumps(json.loads(args.selection.read_text()), indent=2).encode("utf-8")
    if args.layout == "classification":
        selection_bytes += b"\n"
    selection_entry = entries[f"data/{selection_name}"]
    if (
        len(selection_bytes) != selection_entry["bytes"]
        or hashlib.sha256(selection_bytes).hexdigest() != selection_entry["sha256"]
    ):
        raise ValueError("Selection manifest does not match the model entry")
    if args.output.exists():
        raise FileExistsError(f"Use a new output directory: {args.output}")
    selected = json.loads(selection_bytes)
    train_names = selected["train"]
    val_names = selected["validation" if args.layout == "multimodal" else "val"]
    labels, validation = labels_from_devkit(args.devkit)
    with zipfile.ZipFile(args.train_archive) as archive:
        train_x = np.stack([preprocess(archive.read(name)) for name in train_names])
    train_y = np.array([labels[Path(name).parent.name] for name in train_names], dtype="int64")
    wanted = set(val_names)
    val_images = {}
    with tarfile.open(args.val_archive, mode="r|") as archive:
        for member in archive:
            name = Path(member.name).name
            if name in wanted:
                val_images[name] = preprocess(archive.extractfile(member).read())
                if len(val_images) == len(wanted):
                    break
    missing = wanted - val_images.keys()
    if missing:
        raise ValueError(f"Missing validation images: {sorted(missing)}")
    val_x = np.stack([val_images[name] for name in val_names])
    val_y = np.array([validation[int(Path(name).stem.rsplit("_", 1)[1]) - 1] for name in val_names], dtype="int64")
    buffer = io.BytesIO()
    if args.layout == "multimodal":
        write_npz(
            buffer, dict(images=np.concatenate([train_x, val_x]), labels=np.concatenate([train_y, val_y])), False
        )
    else:
        write_npz(buffer, dict(train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y), True)
    data = buffer.getvalue()
    expected = entries[f"data/{data_name}"]
    digest = hashlib.sha256(data).hexdigest()
    if len(data) != expected["bytes"] or digest != expected["sha256"]:
        raise ValueError(
            f"Rebuilt input differs from reference: {digest}. Check archive versions and Pillow/NumPy versions; "
            "do not update the expected hash to bypass this check."
        )
    destination = args.output / "data"
    destination.mkdir(parents=True, exist_ok=False)
    (destination / data_name).write_bytes(data)
    (destination / selection_name).write_bytes(selection_bytes)
    print(json.dumps(dict(output=str(args.output), sha256=digest, bytes=len(data))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-archive", required=True, type=Path, help="Licensed imagenet100.zip")
    parser.add_argument("--val-archive", required=True, type=Path, help="Licensed ILSVRC2012_img_val.tar")
    parser.add_argument("--devkit", required=True, type=Path, help="ILSVRC2012_devkit_t12.tar.gz")
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--asset-manifest", required=True, type=Path)
    parser.add_argument("--layout", required=True, choices=("classification", "multimodal"))
    parser.add_argument("--output", required=True, type=Path)
    rebuild(parser.parse_args())


if __name__ == "__main__":
    main()
