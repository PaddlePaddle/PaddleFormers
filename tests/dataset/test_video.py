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

import sys
from io import BytesIO

import numpy as np
import paddle

# First, delete torchcodec from sys.modules if it exists
if "torchcodec" in sys.modules:
    del sys.modules["torchcodec"]

# Enable torch proxy
paddle.compat.enable_torch_proxy(scope={"torchcodec"})

# Now try to import torchcodec
from torchcodec.decoders import VideoDecoder

video_path = "../utils/Shufflers/dataset_videos/draw.mp4"

with open(video_path, "rb") as f:
    res = BytesIO(f.read())

decoder = VideoDecoder(res, num_ffmpeg_threads=0)
decoder.metadata.average_fps_from_header = 30.0
# Output decoder metadata
print("=== Decoder Metadata ===")
print(f"average_fps: {decoder.metadata.average_fps}")

video_fps = decoder.metadata.average_fps
idx = [0, 18, 36, 53, 71, 89, 107, 125, 143, 160, 178, 196]
video = decoder.get_frames_at(indices=idx).data
sw = np.load("../tmp/swft_video.npy")
print("\n=== Video Result ===")
print(f"video shape: {video.shape}")
print(f"video dtype: {video.dtype}")
print(f"video device: {video.device}")
print(f"diff: {np.argwhere(sw!=video).T}")
# print(f'video: {video}')

paddle.compat.disable_torch_proxy()
print("\nTest completed successfully!")
