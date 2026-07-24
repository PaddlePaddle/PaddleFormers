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

"""MTP Magic Send: MagicInstance.

MagicInstance provides index-based addressing for input_ids in the
multi-layer MTP magic send pipeline.
"""

from __future__ import annotations


class MagicInstance:
    """Index-based addressing storage for input_ids.

    Lifecycle:
    - set_data: DistDataLoader writes the full accumulation-period list
    - get: MTP layer forward indexes by magic_idx
    - get/set_magic_count: per MTP-layer-instance counter
    - clear_count_dict: reset at optimizer / eval step boundary
    """

    def __init__(self):
        self.magic_send_dict = {}
        self.magic_cnt_dict = {}

    def set_data(self, new_dict):
        for k in new_dict:
            self.magic_send_dict[k] = new_dict[k]

    def get(self, key):
        assert key in self.magic_send_dict, f"{key} not in magic_send_dict"
        return self.magic_send_dict[key]

    def get_magic_count(self, key):
        assert key in self.magic_cnt_dict, f"{key} not in magic_cnt_dict"
        return self.magic_cnt_dict[key]

    def set_magic_count(self, key, value):
        self.magic_cnt_dict[key] = value

    def clear_count_dict(self):
        for key in self.magic_cnt_dict:
            self.magic_cnt_dict[key] = -1


mtp_magic_instance = MagicInstance()
