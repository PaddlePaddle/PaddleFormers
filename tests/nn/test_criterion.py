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

import copy
import unittest
from types import SimpleNamespace

import paddle

from paddleformers.nn.criterion import CriterionLayer
from paddleformers.transformers import LlamaConfig


class TestCriterionLayer(unittest.TestCase):
    def setUp(self):
        self.config = LlamaConfig()
        self.batch_size = 2
        self.seq_len = 4
        self.vocab_size = self.config.vocab_size
        self.logits = paddle.randn([self.batch_size, self.seq_len, self.vocab_size], dtype="float32")
        self.labels = paddle.randint(0, self.vocab_size, shape=[self.batch_size, self.seq_len], dtype="int64")

    def test_forward_default_sft(self):
        layer = CriterionLayer(config=self.config)
        layer(self.logits, self.labels)

    def test_forward_non_fuse_subbatch_sft(self):
        config = copy.deepcopy(self.config)
        config.loss_subbatch_sequence_length = 2
        config.use_fused_head_and_loss_fn = False
        layer = CriterionLayer(config=config)
        layer(self.logits, self.labels)

    def test_forward_with_loss_mask(self):
        layer = CriterionLayer(config=self.config)
        loss_mask = paddle.randint(0, 2, shape=[self.batch_size, self.seq_len])
        layer(self.logits, self.labels, loss_mask=loss_mask)

    def test_loss_type_selection(self):
        # 测试不同配置下的 loss_type
        config_dpo = copy.deepcopy(self.config)
        config_dpo.dpo_config = SimpleNamespace(
            beta=0.1,
            offset_alpha=0.0,
            simpo_gamma=0.5,
            normalize_logps=True,
            label_smoothing=0.0,
            loss_type="sigmoid",
            pref_loss_ratio=1.0,
            sft_loss_ratio=0.0,
            dpop_lambda=50,
            ref_model_update_steps=-1,
            reference_free=False,
            lora=False,
        )
        layer = CriterionLayer(config=config_dpo)
        self.assertEqual(layer.loss_type, "dpo")

        config_kto = copy.deepcopy(self.config)
        config_kto.kto_config = SimpleNamespace(beta=0.1, desirable_weight=1.0, undesirable_weight=1.0, lora=False)
        layer = CriterionLayer(config=config_kto)
        self.assertEqual(layer.loss_type, "kto")
        layer = CriterionLayer(config=self.config)
        self.assertEqual(layer.loss_type, "sft")


if __name__ == "__main__":
    unittest.main()
