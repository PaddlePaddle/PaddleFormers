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

import copy

from ..configuration_utils import PretrainedConfig
from ..llama.configuration import LlamaConfig
from ..qwen2.configuration import Qwen2Config
from .configuration_intern_vit import InternVisionConfig


class InternVLChatConfig(PretrainedConfig):
    model_type = "internvl_chat"
    is_composition = True
    sub_configs = {"vision_config": InternVisionConfig}
    processor_class = "InternVL3Processor"

    def __init__(
        self,
        vision_config=None,
        llm_config=None,
        use_backbone_lora=0,
        use_llm_lora=0,
        select_layer=-1,
        force_image_size=None,
        downsample_ratio=0.5,
        template=None,
        dynamic_image_size=False,
        use_thumbnail=False,
        ps_version="v1",
        min_dynamic_patch=1,
        max_dynamic_patch=6,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if vision_config is None:
            vision_config = {"architectures": ["InternVisionModel"]}
        if llm_config is None:
            llm_config = {"architectures": ["Qwen2ForCausalLM"]}

        self.vision_config = InternVisionConfig(**vision_config)
        architecture = llm_config.get("architectures", ["Qwen2ForCausalLM"])[0]
        if architecture == "LlamaForCausalLM":
            self.llm_config = LlamaConfig(**llm_config)
        elif architecture == "Qwen2ForCausalLM":
            self.llm_config = Qwen2Config(**llm_config)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        self.use_backbone_lora = use_backbone_lora
        self.use_llm_lora = use_llm_lora
        self.select_layer = select_layer
        self.force_image_size = force_image_size
        self.downsample_ratio = downsample_ratio
        self.template = template
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.ps_version = ps_version
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.tie_word_embeddings = self.llm_config.tie_word_embeddings

    def to_dict(self, saving_file=False):
        output = super().to_dict(saving_file=saving_file)
        output["vision_config"] = self.vision_config.to_dict(saving_file=saving_file)
        output["llm_config"] = self.llm_config.to_dict(saving_file=saving_file)
        output["model_type"] = self.__class__.model_type
        return output


__all__ = ["InternVLChatConfig"]
