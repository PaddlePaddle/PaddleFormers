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
import os
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)


import unittest


class TestModelType(unittest.TestCase):
    """Tests for ModelType enum."""

    def test_encoder_or_decoder_value(self):
        from paddleformers.fleet.transformer.enums import ModelType

        self.assertEqual(ModelType.encoder_or_decoder.value, 1)

    def test_encoder_or_decoder_is_instance(self):
        from paddleformers.fleet.transformer.enums import ModelType

        self.assertIsInstance(ModelType.encoder_or_decoder, ModelType)

    def test_encoder_and_decoder_deprecated(self):
        from paddleformers.fleet.transformer.enums import ModelType

        with self.assertRaises(ValueError):
            _ = ModelType.encoder_or_decoder.encoder_and_decoder

    def test_model_type_members(self):
        from paddleformers.fleet.transformer.enums import ModelType

        members = list(ModelType)
        self.assertEqual(len(members), 1)
        self.assertEqual(members[0], ModelType.encoder_or_decoder)


class TestLayerType(unittest.TestCase):
    """Tests for LayerType enum."""

    def test_embedding_value(self):
        from paddleformers.fleet.transformer.enums import LayerType

        self.assertEqual(LayerType.embedding.value, 1)

    def test_loss_value(self):
        from paddleformers.fleet.transformer.enums import LayerType

        self.assertEqual(LayerType.loss.value, 2)

    def test_encoder_value(self):
        from paddleformers.fleet.transformer.enums import LayerType

        self.assertEqual(LayerType.encoder.value, 3)

    def test_decoder_value(self):
        from paddleformers.fleet.transformer.enums import LayerType

        self.assertEqual(LayerType.decoder.value, 4)

    def test_mtp_value(self):
        from paddleformers.fleet.transformer.enums import LayerType

        self.assertEqual(LayerType.mtp.value, 5)

    def test_all_members_count(self):
        from paddleformers.fleet.transformer.enums import LayerType

        self.assertEqual(len(list(LayerType)), 5)

    def test_comparison(self):
        from paddleformers.fleet.transformer.enums import LayerType

        self.assertEqual(LayerType.embedding, LayerType.embedding)
        self.assertNotEqual(LayerType.embedding, LayerType.decoder)


class TestAttnType(unittest.TestCase):
    """Tests for AttnType enum."""

    def test_self_attn_value(self):
        from paddleformers.fleet.transformer.enums import AttnType

        self.assertEqual(AttnType.self_attn.value, 1)

    def test_cross_attn_value(self):
        from paddleformers.fleet.transformer.enums import AttnType

        self.assertEqual(AttnType.cross_attn.value, 2)

    def test_members_count(self):
        from paddleformers.fleet.transformer.enums import AttnType

        self.assertEqual(len(list(AttnType)), 2)


class TestAttnMaskType(unittest.TestCase):
    """Tests for AttnMaskType enum."""

    def test_padding_value(self):
        from paddleformers.fleet.transformer.enums import AttnMaskType

        self.assertEqual(AttnMaskType.padding.value, 1)

    def test_causal_value(self):
        from paddleformers.fleet.transformer.enums import AttnMaskType

        self.assertEqual(AttnMaskType.causal.value, 2)

    def test_no_mask_value(self):
        from paddleformers.fleet.transformer.enums import AttnMaskType

        self.assertEqual(AttnMaskType.no_mask.value, 3)

    def test_padding_causal_value(self):
        from paddleformers.fleet.transformer.enums import AttnMaskType

        self.assertEqual(AttnMaskType.padding_causal.value, 4)

    def test_arbitrary_value(self):
        from paddleformers.fleet.transformer.enums import AttnMaskType

        self.assertEqual(AttnMaskType.arbitrary.value, 5)

    def test_causal_bottom_right_value(self):
        from paddleformers.fleet.transformer.enums import AttnMaskType

        self.assertEqual(AttnMaskType.causal_bottom_right.value, 6)

    def test_members_count(self):
        from paddleformers.fleet.transformer.enums import AttnMaskType

        self.assertEqual(len(list(AttnMaskType)), 6)

    def test_iteration(self):
        from paddleformers.fleet.transformer.enums import AttnMaskType

        values = [m.value for m in AttnMaskType]
        self.assertEqual(values, [1, 2, 3, 4, 5, 6])


class TestAttnBackend(unittest.TestCase):
    """Tests for AttnBackend enum."""

    def test_flash_value(self):
        from paddleformers.fleet.transformer.enums import AttnBackend

        self.assertEqual(AttnBackend.flash.value, 1)

    def test_fused_value(self):
        from paddleformers.fleet.transformer.enums import AttnBackend

        self.assertEqual(AttnBackend.fused.value, 2)

    def test_unfused_value(self):
        from paddleformers.fleet.transformer.enums import AttnBackend

        self.assertEqual(AttnBackend.unfused.value, 3)

    def test_local_value(self):
        from paddleformers.fleet.transformer.enums import AttnBackend

        self.assertEqual(AttnBackend.local.value, 4)

    def test_auto_value(self):
        from paddleformers.fleet.transformer.enums import AttnBackend

        self.assertEqual(AttnBackend.auto.value, 5)

    def test_members_count(self):
        from paddleformers.fleet.transformer.enums import AttnBackend

        self.assertEqual(len(list(AttnBackend)), 5)

    def test_name_access(self):
        from paddleformers.fleet.transformer.enums import AttnBackend

        self.assertEqual(AttnBackend.flash.name, "flash")
        self.assertEqual(AttnBackend.auto.name, "auto")


class TestEnumImports(unittest.TestCase):
    """Test that all enums can be imported from the enums module."""

    def test_import_all_enums(self):
        from paddleformers.fleet.transformer.enums import (
            AttnBackend,
            AttnMaskType,
            AttnType,
            LayerType,
            ModelType,
        )

        self.assertIsNotNone(ModelType)
        self.assertIsNotNone(LayerType)
        self.assertIsNotNone(AttnType)
        self.assertIsNotNone(AttnMaskType)
        self.assertIsNotNone(AttnBackend)


if __name__ == "__main__":
    unittest.main()
