# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional


@dataclass
class DataArguments:
    r"""Arguments pertaining to what data we are going to input our model for training and evaluation."""
    
    # dataset
    max_seq_len: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length."},
    )

    # model
    patch_size: int = field(default=14, metadata={"help": "Patch size"})
    merge_size: int = field(default=2, metadata={"help": "Merge size"})
    spatial_conv_size: int = field(
        default=2,
        metadata={"help": "spatial conv size"},
    )
    temporal_conv_size: int = field(
        default=2,
        metadata={"help": "temporal conv size"},
    )

    # processor
    video_fps: int = field(default=2, metadata={"help": "fps for sampling frames"})
    video_min_frames: int = field(
        default=16, metadata={"help": "fps for sampling frames with min"}
    )
    video_max_frames: int = field(
        default=480, metadata={"help": "fps for sampling frames with max"}
    )
    video_target_frames: int = field(
        default=-1, metadata={"help": "fps for sampling frames with target"}
    )
    video_frames_sample: str = field(
        default="middle", metadata={"help": " middle, rand, leading"}
    )
    max_pixels: int = field(default=28 * 28 * 1280, metadata={"help": "adaptive use max-pixels"})
    min_pixels: int = field(default=56 * 56, metadata={"help": "adaptiveuse min-pixels"})
    video_max_pixels: int = field(
        default=28 * 28 * 1280, metadata={"help": "video adaptive use max-pixels"}
    )
    video_min_pixels: int = field(
        default=56 * 56, metadata={"help": "video adaptiveuse min-pixels"}
    )
    render_timestamp: bool = field(default=False, metadata={"help": "render timestamp"})
    do_resize: bool = field(default=True, metadata={"help": "whether to resize"})