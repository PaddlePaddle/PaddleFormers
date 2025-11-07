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

"""MMSpecialTokensConfig class"""

import logging

logger = logging.getLogger(__name__)

__all__ = ("MMSpecialTokensConfig", "DATATYPE_2_ID", "IDTYPES_2_ID", "IMAGETYPES_2_ID")

DATATYPE_2_ID = {"mm": 0, "lm": 1, "audio": 2}
IDTYPES_2_ID = {"text": 0, "image": 1, "video": 2, "audio": 3}
IMAGETYPES_2_ID = {"image": 0, "video": 1, "padded_image": 2}


class MMSpecialTokensConfig:
    """_summary_"""

    use_ocr_specialtoken = True
    coor_num = 1001
    image_placeholder = "<|IMAGE_PLACEHOLDER|>"
    audio_placeholder = "<|AUDIO_PLACEHOLDER|>"
    ocr_coor = [f"<|LOC_{i}|>" for i in range(coor_num)]
    ocr_begin_end = ["<|LOC_BEGIN|>", "<|LOC_END|>", "<|LOC_SEP|>"]
    mm_begin_end = ["<|BOI|>", "<|EOI|>", "<|BOA|>", "<|EOA|>", "<|BOV|>", "<|EOV|>"]

    @classmethod
    def get_special_tokens_info(cls):
        """_summary_

        Returns:
            _type_: _description_
        """
        return {
            k: getattr(cls, k)
            for k in [
                "image_placeholder",
                "audio_placeholder",
                "ocr_coor",
                "ocr_begin_end",
                "mm_begin_end",
            ]
        }
