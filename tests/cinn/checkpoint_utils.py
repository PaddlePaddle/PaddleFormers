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

"""Strict checkpoint loading for reproducible CINN comparisons."""
from collections.abc import Mapping


def strict_set_state_dict(model, state_dict):
    """Reject incompatible checkpoints before changing any model parameter."""
    if not isinstance(state_dict, Mapping):
        raise ValueError("Checkpoint model state must be a parameter mapping")
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(expected))
    mismatched = {
        name: (tuple(expected[name].shape), getattr(state_dict[name], "shape", None))
        for name in expected.keys() & state_dict.keys()
        if not hasattr(state_dict[name], "shape") or tuple(expected[name].shape) != tuple(state_dict[name].shape)
    }
    if missing or unexpected or mismatched:
        raise ValueError(
            f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}, "
            f"shapes(expected, actual)={mismatched}"
        )
    missing, unexpected = model.set_state_dict(state_dict)
    if missing or unexpected:
        raise ValueError(f"Checkpoint load skipped parameters: missing={missing}, unexpected={unexpected}")
