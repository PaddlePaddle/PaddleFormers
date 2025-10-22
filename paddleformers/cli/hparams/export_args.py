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

from dataclasses import dataclass, field


@dataclass
class ExportArguments:
    """Export Argument"""

    # export parameter
    copy_tokenizer: bool = field(default=True, metadata={"help": "Copy tokenizer file"})
    mergekit_task_config: str = field(default=None, metadata={"help": "The merge config path."})

    # split parameter
    max_shard_size: int = field(default=5, metadata={"help": "The maximum size (GB) per checkpoint."})

    # hf parameter
    hf_hub_id: str = field(default=None, metadata={"help": "The hugging face id that needs to be uploaded."})
