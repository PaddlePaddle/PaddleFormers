# !/usr/bin/env python3

# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

"""
End2EndProcessorArgumentsHelper

"""
from dataclasses import dataclass, field


@dataclass
class BasePreprocessArguments:
    def __post_init__(self):
        pass


@dataclass
class UtteranceProcessorArguments(BasePreprocessArguments):
    """
    args for UtteranceArguments
    """

    tokenizer: str = field(default=None, metadata={"help": "path of tokenizer"})
    tokenizer_name: str = field(default=None, metadata={"help": "path of tokenizer"})

    def __post_init__(self):
        if self.tokenizer_name and not self.tokenizer:
            self.tokenizer = self.tokenizer_name
        super().__post_init__()


@dataclass
class CoarseProcessorArguments(BasePreprocessArguments):
    """
    args for CoarseProcessor
    """

    video_fps: int = field(default=2, metadata={"help": "fps for sampling frames"})
    video_min_frames: int = field(default=16, metadata={"help": "fps for sampling frames with min"})
    video_max_frames: int = field(default=480, metadata={"help": "fps for sampling frames with max"})
    video_target_frames: int = field(default=-1, metadata={"help": "fps for sampling frames with target"})
    video_frames_sample: str = field(default="middle", metadata={"help": " middle, rand, leading"})

    def __post_init__(self):
        self.video_frames_sample = self.video_frames_sample.lower()
        assert self.video_frames_sample in ["middle", "rand", "leading"]
        super().__post_init__()


@dataclass
class InputIdsMassageArguments(BasePreprocessArguments):
    """
    args for InputIdsMassageProcessor
    """

    corpus_name: str = field(default=None, metadata={"help": "corpus name"})
    im_prefix_length: int = field(default=64, metadata={"help": "number of image placeholder"})

    use_pic_id: bool = field(default=True, metadata={"help": "add Picture Id"})

    prompt_dir: str = field(default="./", metadata={"help": "prompt path"})

    serialize_output: bool = field(default=True, metadata={"help": "serialize output"})
    one_sample_in_one_seq: bool = field(default=False, metadata={"help": "one sample in one seq"})
    variable_resolution: bool = field(default=False, metadata={"help": "use variable resolution"})
    spatial_conv_size: int = field(
        default=2,
        metadata={"help": "spatial conv size"},
    )
    adaptive_max_imgtoken_option: str = field(default=None, metadata={"help": "adaptive max image token"})
    adaptive_max_imgtoken_rate: str = field(default=None, metadata={"help": "adaptive max image token rate"})
    max_pixels: int = field(default=None, metadata={"help": "adaptive use max-pixels"})
    min_pixels: int = field(default=None, metadata={"help": "adaptiveuse min-pixels"})
    video_max_pixels: int = field(default=None, metadata={"help": "video adaptive use max-pixels"})
    video_min_pixels: int = field(default=None, metadata={"help": "video adaptiveuse min-pixels"})
    drop_untrainble_sample: bool = field(default=False, metadata={"help": "drop untrainable samples"})
    chat_template: str = field(default="ernie_vl", metadata={"help": "chat template"})

    def __post_init__(self):
        if self.adaptive_max_imgtoken_option is not None and self.adaptive_max_imgtoken_rate is not None:
            self.adaptive_max_imgtoken_option = [
                int(op) for op in self.adaptive_max_imgtoken_option.strip().split(",")
            ]
            self.adaptive_max_imgtoken_rate = [float(op) for op in self.adaptive_max_imgtoken_rate.strip().split(",")]
        super().__post_init__()


@dataclass
class ImageModificationProcessorArguments(BasePreprocessArguments):
    """
    args for ImageModificationProcessor
    """

    image_token_len: int = field(default=64, metadata={"help": "image placeholder num per frame"})
    image_dtype: str = field(default="uint8", metadata={"help": "image dtype"})
    render_timestamp: bool = field(default=False, metadata={"help": "render timestamp"})
    sft_shift_by_one: bool = field(default=False, metadata={"help": "SFT data_processor shift-by-one"})

    def __post_init__(self):
        super().__post_init__()


@dataclass
class End2EndProcessorArgumentsHelper(BasePreprocessArguments):
    """
    args for End2EndProcessorArgumentsHelper
    """

    load_args_from_api: bool = field(default=False, metadata={"help": "load arguments from api"})

    def __post_init__(self):
        super().__post_init__()


@dataclass
class End2EndProcessorArguments(
    UtteranceProcessorArguments,
    CoarseProcessorArguments,
    InputIdsMassageArguments,
    ImageModificationProcessorArguments,
    End2EndProcessorArgumentsHelper,
):
    def __post_init__(self):
        super().__post_init__()
