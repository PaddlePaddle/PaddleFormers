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

import tempfile
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

from paddleformers.transformers.llama4.modeling import Llama4ForCausalLM

from .test_modeling import Llama4ModelTester


def test_tied_flex_export_discards_output_head_alias():
    config = Llama4ModelTester(None).get_config()
    config.tie_word_embeddings = True
    config.fuse_rms_norm = False
    model = Llama4ForCausalLM(config).eval()
    assert "lm_head.weight -> _" in model._gen_inv_aoa_config(config)["aoa_statements"]
    with tempfile.TemporaryDirectory() as directory:
        model.save_pretrained(directory, save_checkpoint_format="flex_checkpoint")
        exported = {}
        for shard in Path(directory).glob("*.safetensors"):
            exported.update(load_file(shard))
        assert exported
        assert not any("lm_head" in name for name in exported)
        restored = Llama4ForCausalLM.from_pretrained(directory, load_checkpoint_format="flex_checkpoint").eval()
        assert set(restored.state_dict()) == set(model.state_dict())
        for name, value in model.state_dict().items():
            np.testing.assert_array_equal(restored.state_dict()[name].numpy(), value.numpy(), err_msg=name)
