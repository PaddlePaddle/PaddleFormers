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
from unittest.mock import patch

import pytest

from paddleformers.transformers import InternLMConfig


@pytest.mark.parametrize("rotary_type,rope_type", [("origin", "default"), ("dynamic", "dynamic")])
def test_rotary_configuration_validation(rotary_type, rope_type):
    config = InternLMConfig(rotary={"base": 10000, "type": rotary_type, "scaling_factor": 2.0})
    assert config.rope_parameters["rope_type"] == rope_type
    assert config.rope_parameters["rope_theta"] == 10000


@pytest.mark.parametrize("factor", [2, 0.5])
def test_invalid_scaling_factor_warning(factor):
    with patch("paddleformers.transformers.modeling_rope_utils.logger.warning") as warning:
        InternLMConfig(rotary={"base": 10000, "type": "dynamic", "scaling_factor": factor})
    assert any("factor field must be a float >= 1" in str(call) for call in warning.call_args_list)
