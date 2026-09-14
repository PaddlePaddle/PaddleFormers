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

"""Assembly of HyperEncoder, a multimodal encoder, on top of PaddleFormers.

## Split of responsibilities against the PaddleFleet side

| Here (PaddleFormers) | PaddleFleet `models/hyperencoder/` |
|---|---|
| `configuration.py` -- holds only the config class | `layer_specs.py` -- spec builders |
| `modeling_fleet.py` -- `HyperEncoderProvider` / `TransformerBlock` subclass / top-level Model | `modality_encoders.py` / `norm.py` / `attn_backend.py` |

The transfer from `PretrainedConfig` to the Fleet config uses the framework's
generic mechanism (`TransformerConfig.from_config`: `object.__new__` +
`register_attributes` + `__post_init__`). Non-default architecture values are
carried as the field defaults of `HyperEncoderProvider`, and the few that need
to be derived live in its `__post_init__`.

Module layout:

* `HyperEncoderBlock(TransformerBlock)` -- a thin `TransformerBlock` subclass;
* the top-level Model;
* the Provider that turns a config into the network.

## Why `_LazyModule` is not used here

A lazy module setup requires simultaneous registration in
`transformers/__init__.py` before `AutoModel` / CLI can discover it. This
module currently only needs plain explicit imports; it can be switched to a
lazy module and registered when full CLI integration is added.
"""

from .configuration import HyperEncoderConfig
from .modeling_fleet import (
    HyperEncoderBlock,
    HyperEncoderModel,
    HyperEncoderModelFleet,
    HyperEncoderProvider,
)

__all__ = [
    "HyperEncoderConfig",
    "HyperEncoderProvider",
    "HyperEncoderBlock",
    "HyperEncoderModel",  # The FleetLayer network itself (returned by Provider.provide())
    "HyperEncoderModelFleet",  # PretrainedModel wrapper, the AutoModel entry point
]
