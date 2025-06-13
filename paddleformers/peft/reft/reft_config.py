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

import json
import os

from .modeling_utils import get_type_from_string


class ReFTConfig:
    def __init__(
        self,
        representations,
        intervention_params=None,
        position=None,
        intervention_types=None,
        sorted_keys=None,
        intervention_dimensions=None,
        **kwargs,
    ):
        if not isinstance(representations, list):
            representations = [representations]

        self.representations = representations
        self.intervention_types = intervention_types
        overwrite_intervention_types = []
        for reprs in self.representations:
            if reprs["intervention"] is not None:
                overwrite_intervention_types += [type(reprs["intervention"])]

        self.intervention_types = overwrite_intervention_types
        self.sorted_keys = sorted_keys
        self.intervention_dimensions = intervention_dimensions
        self.intervention_params = intervention_params
        self.position = position

    def to_dict(self):
        return {
            "representations": self.representations,
            "intervention_types": self.intervention_types,
            "sorted_keys": self.sorted_keys,
        }

    @staticmethod
    def from_pretrained(load_directory):
        saved_config = json.load(open(os.path.join(load_directory, "config.json"), "r"))
        for representation, intervention_type in zip(
            saved_config["representations"], saved_config["intervention_types"]
        ):
            representation["intervention"] = get_type_from_string(intervention_type)(
                **saved_config["intervention_params"]
            )
        reft_config = ReFTConfig(
            representations=saved_config["representations"],
            intervention_params=saved_config["intervention_params"],
        )
        return reft_config

    def save_pretrained(self, save_directory):
        config_dict = {}
        config_dict["representations"] = [
            {
                "layer": repr["layer"],
                "component": repr["component"],
                "low_rank_dimension": repr["low_rank_dimension"],
            }
            for repr in self.representations
        ]

        config_dict["intervention_params"] = self.intervention_params
        config_dict["intervention_types"] = [repr(intervention_type) for intervention_type in self.intervention_types]
        config_dict["position"] = self.position
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=4)
