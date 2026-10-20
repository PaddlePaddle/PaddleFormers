# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.
import unittest

from paddleformers.transformers import FalconPerceptionConfig
from tests.transformers.test_configuration_common import ConfigTester


class FalconPerceptionConfigTest(unittest.TestCase):
    def setUp(self):
        self.config_tester = ConfigTester(
            self,
            config_class=FalconPerceptionConfig,
            common_properties=["dim", "n_heads", "n_layers"],
            dim=32,
            n_heads=4,
            n_layers=2,
            head_dim=8,
            vocab_size=300,
            spatial_patch_size=2,
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_perception_fields_roundtrip(self):
        config = FalconPerceptionConfig(
            dim=32,
            coord_enc_dim=8,
            coord_dec_dim=24,
            coord_out_dim=16,
            size_enc_dim=8,
            size_dec_dim=24,
            size_out_dim=16,
            segm_out_dim=16,
            num_segm_layers=2,
            pad_token_id=7,
            architectures=["FalconPerceptionForSegmentation"],
        )
        restored = FalconPerceptionConfig.from_dict(config.to_dict())
        self.assertEqual(restored.to_dict(), config.to_dict())
        self.assertEqual(restored.model_type, "falcon_perception")
        self.assertEqual(restored.pad_token_id, 7)


if __name__ == "__main__":
    unittest.main()
