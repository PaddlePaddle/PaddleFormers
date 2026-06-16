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

# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import random
import unittest
from copy import deepcopy

import numpy as np
import paddle
from paddle.distributed import fleet

import paddleformers.fleet.parallel_state as ps
from paddleformers.fleet.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
)
from paddleformers.fleet.models.multimodal.llava_model import LLaVAModel
from paddleformers.fleet.transformer.transformer_config import TransformerConfig


class TestLLaVAModel(unittest.TestCase):
    def setUp(self):
        if not ps.have_global_memory_buffer():
            seed = 46
            random.seed(seed)
            np.random.seed(seed)
            paddle.manual_seed(seed)
            strategy = fleet.DistributedStrategy()
            strategy.hybrid_configs = {
                "dp_degree": 1,
                "mp_degree": 1,
                "pp_degree": 1,
                "sharding_degree": 1,
                "sep_degree": 1,
                "cp_degree": 1,
                "ep_degree": 1,
                "moe_sharding_degree": 1,
                "order": [
                    "sharding",
                    "moe_sharding",
                    "pp",
                    "sep",
                    "cp",
                    "dp",
                    "ep",
                    "mp",
                ],
            }
            fleet.init(is_collective=True, strategy=strategy)
            hcg = fleet.get_hybrid_communicate_group()
            ps.initialize_model_parallel(hcg)

        self.language_hidden_size = 64
        self.language_num_attention_heads = 4

        language_config = TransformerConfig(
            num_hidden_layers=3,
            hidden_size=self.language_hidden_size,
            num_attention_heads=self.language_num_attention_heads,
            use_cpu_initialization=False,
        )
        vision_config = TransformerConfig(
            num_hidden_layers=2,
            hidden_size=16,
            num_attention_heads=2,
            use_cpu_initialization=False,
        )
        vision_projection_config = TransformerConfig(
            num_hidden_layers=2,
            hidden_size=self.language_hidden_size,
            intermediate_size=32,
            num_attention_heads=1,
            use_cpu_initialization=False,
        )

        language_layer_spec = get_gpt_layer_local_spec()
        vision_layer_spec = deepcopy(language_layer_spec)
        vision_projection_spec = deepcopy(
            language_layer_spec.sublayers_spec.mlp.sublayers_spec
        )

        language_config.language_model_type = "dummy"
        vision_config.vision_model_type = "clip"
        self.model = LLaVAModel(
            language_transformer_config=language_config,
            language_transformer_layer_spec=language_layer_spec,
            language_vocab_size=8192,
            language_max_sequence_length=4096,
            vision_transformer_config=vision_config,
            vision_transformer_layer_spec=vision_layer_spec,
            drop_vision_class_token=False,
            vision_projection_config=vision_projection_config,
            vision_projection_layer_spec=vision_projection_spec,
            img_h=336,
            img_w=336,
            patch_dim=14,
        )

    def tearDown(self):
        pass

    def test_constructor(self):
        assert isinstance(self.model, LLaVAModel)

        num_weights = sum([p.numel() for p in self.model.parameters()])

        assert num_weights == 1486080

    def test_set_input_tensor(self):
        expected_shape = (1, 2, 3, 4)
        input_tensor = paddle.zeros(expected_shape)
        self.model.set_input_tensor(input_tensor)
        assert self.model.vision_model.decoder.input_tensor.shape == list(
            expected_shape
        ), f"input_shape {input_tensor.shape}, expected_shape {expected_shape}"

    def test_preprocess_data(self):
        hidden_size = 72

        # 3 images with 1 tile and 2 image with 2 tiles = 7 tiles.
        image_embeddings = paddle.arange(
            577 * 7 * hidden_size, dtype=paddle.float
        ).reshape(577, 7, hidden_size)

        image_token_index = self.model.image_token_index
        input_ids = paddle.arange(1024).expand(5, 1024).contiguous()
        input_ids[0, 0] = image_token_index  # image before text
        input_ids[1, 100] = image_token_index  # image in between
        input_ids[2, -1] = image_token_index  # image at the end
        # input_ids[3] - no image
        input_ids[4, 50] = image_token_index  # two images in between
        input_ids[4, 150] = image_token_index

        # Using negative sign to distinguish from image embeddings.
        language_embeddings = -paddle.arange(
            5 * 1024 * hidden_size, dtype=paddle.float
        ).reshape(5, 1024, hidden_size)

        # Labels are input_ids shifted to left by one.
        labels = (
            paddle.arange(1, 1025, dtype=paddle.int)
            .expand(5, 1024)
            .contiguous()
        )
        # labels[0] - image token got dropped by shift to left by one.
        labels[1, 99] = image_token_index
        labels[2, -2] = image_token_index
        # labels[3] - no image.
        labels[4, 49] = image_token_index
        labels[4, 149] = image_token_index

        loss_mask = paddle.ones((5, 1024), dtype=paddle.float)
        # Mask some text inputs (the text mask should carry over)
        loss_mask[:2, :10] = 0.0
        loss_mask[:2, 110:120] = 0.0

        # Number of tiles for each image in the batch.
        num_image_tiles = paddle.tensor([1, 2, 1, 2, 1], dtype=paddle.int)

        use_inference_kv_cache = False
        inference_context = None

        embeddings, labels, loss_mask = self.model._preprocess_data(
            image_embeddings,
            language_embeddings,
            input_ids,
            loss_mask,
            labels,
            use_inference_kv_cache,
            inference_context,
            image_token_index,
            num_image_tiles,
        )

        img_seq_len = 577
        # The fifth sample has 2 images with 3 tiles and 1024 text tokens.
        max_seq_len = 3 * img_seq_len - 2 + 1024

        assert embeddings.shape == [max_seq_len, 5, hidden_size], (
            f"{embeddings.shape} "
        )
        assert labels.shape == [5, max_seq_len]
        assert loss_mask.shape == labels.shape

        # First sample where image is before text (index 0).
        expected_embeddings = paddle.empty(max_seq_len, hidden_size)
        expected_embeddings[:577] = image_embeddings[:, 0]
        expected_embeddings[577:1600] = language_embeddings[0, 1:]
        expected_embeddings[1600:] = 0  # padding

        expected_labels = paddle.empty(max_seq_len, dtype=paddle.int)
        expected_labels[:576] = -100  # image
        expected_labels[576:1600] = paddle.arange(1, 1025, dtype=paddle.int)
        expected_labels[1600:] = -100  # padding

        expected_loss_mask = paddle.empty(max_seq_len, dtype=paddle.float)
        expected_loss_mask[:577] = 0
        expected_loss_mask[577:586] = 0
        expected_loss_mask[586:686] = 1
        expected_loss_mask[686:696] = 0
        expected_loss_mask[696:1600] = 1
        expected_loss_mask[1600:] = 0

        assert paddle.allclose(embeddings[:, 0], expected_embeddings)
        assert paddle.allclose(labels[0], expected_labels)
        assert paddle.allclose(loss_mask[0], expected_loss_mask)

        # Second sample where image is in between (index 100). The image has 2 tiles.
        expected_embeddings = paddle.empty(max_seq_len, hidden_size)
        expected_embeddings[:100] = language_embeddings[1, :100]
        expected_embeddings[100:677] = image_embeddings[:, 1]
        expected_embeddings[677:1254] = image_embeddings[:, 2]
        expected_embeddings[1254:2177] = language_embeddings[1, 101:]
        expected_embeddings[2177:] = 0  # padding

        expected_labels = paddle.empty(max_seq_len, dtype=paddle.int)
        expected_labels[:99] = paddle.arange(1, 100)
        expected_labels[99:1253] = -100  # image
        expected_labels[1253:2177] = paddle.arange(101, 1025)
        expected_labels[2177:] = -100  # padding

        expected_loss_mask = paddle.empty(max_seq_len, dtype=paddle.float)
        expected_loss_mask[:10] = 0
        expected_loss_mask[10:99] = 1
        # Last text position before the image is not required to predict the first image embedding.
        expected_loss_mask[99] = 0
        expected_loss_mask[100:1254] = 0
        expected_loss_mask[1254:1263] = 1
        expected_loss_mask[1263:1273] = 0
        expected_loss_mask[1273:2177] = 1
        expected_loss_mask[2177:] = 0  # padding

        assert paddle.allclose(embeddings[:, 1], expected_embeddings)
        assert paddle.allclose(labels[1], expected_labels)
        assert paddle.allclose(loss_mask[1], expected_loss_mask)

        # Third sample where image is at the end.
        expected_embeddings = paddle.empty(max_seq_len, hidden_size)
        expected_embeddings[:1023] = language_embeddings[2, :1023]
        expected_embeddings[1023:1600] = image_embeddings[:, 3]
        expected_embeddings[1600:] = 0  # padding

        expected_labels = paddle.empty(max_seq_len, dtype=paddle.int)
        expected_labels[:1022] = paddle.arange(1, 1023)
        expected_labels[1022:1599] = -100
        expected_labels[1599] = 1024
        expected_labels[1600:] = -100  # padding

        expected_loss_mask = paddle.empty(max_seq_len, dtype=paddle.float)
        expected_loss_mask[:1022] = 1
        # Last text position before the image is not required to predict the first image embedding.
        expected_loss_mask[1022] = 0
        expected_loss_mask[1023:1600] = 0
        expected_loss_mask[1600:] = 0  # padding

        assert paddle.allclose(embeddings[:, 2], expected_embeddings)
        assert paddle.allclose(labels[2], expected_labels)
        assert paddle.allclose(loss_mask[2], expected_loss_mask)

        # Fourth sample where there is no image.
        expected_embeddings = paddle.empty(max_seq_len, hidden_size)
        expected_embeddings[:1024] = language_embeddings[3]
        expected_embeddings[1024:] = 0  # padding

        expected_labels = paddle.empty(max_seq_len, dtype=paddle.int)
        expected_labels[:1024] = paddle.arange(1, 1025)
        expected_labels[1024:] = -100  # padding

        expected_loss_mask = paddle.empty(max_seq_len, dtype=paddle.float)
        expected_loss_mask[:1024] = 1
        expected_loss_mask[1024:] = 0  # padding

        assert paddle.allclose(embeddings[:, 3], expected_embeddings)
        assert paddle.allclose(labels[3], expected_labels)
        assert paddle.allclose(loss_mask[3], expected_loss_mask)

        # Fifth sample has two images in between (indices 50 and 150). The first image has two tiles.
        expected_embeddings = paddle.empty(max_seq_len, hidden_size)
        expected_embeddings[:50] = language_embeddings[4, :50]
        expected_embeddings[50:627] = image_embeddings[:, 4]  # two tiles
        expected_embeddings[627:1204] = image_embeddings[:, 5]
        expected_embeddings[1204:1303] = language_embeddings[4, 51:150]
        expected_embeddings[1303:1880] = image_embeddings[:, 6]
        expected_embeddings[1880:] = language_embeddings[4, 151:]

        expected_labels = paddle.empty(max_seq_len, dtype=paddle.int)
        expected_labels[:49] = paddle.arange(1, 50)
        expected_labels[49:1203] = -100  # image
        expected_labels[1203:1302] = paddle.arange(51, 150)
        expected_labels[1302:1879] = -100  # image
        expected_labels[1879:] = paddle.arange(151, 1025)

        expected_loss_mask = paddle.empty(max_seq_len, dtype=paddle.float)
        expected_loss_mask[:49] = 1
        expected_loss_mask[49:1204] = 0
        expected_loss_mask[1204:1302] = 1
        expected_loss_mask[1302:1880] = 0
        expected_loss_mask[1880:] = 1

        assert paddle.allclose(embeddings[:, 4], expected_embeddings)
        assert paddle.allclose(labels[4], expected_labels)
        assert paddle.allclose(loss_mask[4], expected_loss_mask)

    def test_forward(self):
        # 3 images with 1 tile and 2 images with 2 tiles.
        img = paddle.randn((7, 3, 336, 336))

        image_token_index = self.model.image_token_index
        input_ids = paddle.randint(0, 2048, (5, 1024))
        input_ids[0, 0] = image_token_index  # image before text
        input_ids[1, 100] = image_token_index  # image in between
        input_ids[2, -1] = image_token_index  # image at the end
        # input_ids[3] - no image
        input_ids[4, 50] = image_token_index
        input_ids[4, 150] = image_token_index

        position_ids = (
            paddle.arange(0, 1024, dtype=paddle.int)
            .expand(5, 1024)
            .contiguous()
        )

        loss_mask = paddle.ones((5, 1024))

        attention_mask = None  # Causal.

        labels = paddle.randint(0, 2048, (5, 1024))
        labels[1, 99] = image_token_index
        labels[2, -2] = image_token_index

        num_image_tiles = paddle.tensor([1, 2, 1, 2, 1], dtype=paddle.int)

        # Try with labels.
        output, new_loss_mask = self.model.forward(
            img,
            input_ids,
            position_ids,
            attention_mask,
            labels,
            loss_mask,
            num_image_tiles=num_image_tiles,
        )
        # The maximum sequence length is given by the sample with 2 images in 3 tiles, minus two image token indices, plus other text tokens.
        img_seq_len = 577
        max_seq_len = img_seq_len * 3 - 2 + 1024

        assert new_loss_mask.shape == [5, max_seq_len]

    def test_freeze(self):
        self.model.freeze(
            freeze_language_model=True,
            freeze_vision_model=True,
            freeze_vision_projection=False,
        )

        for module in [self.model.language_model, self.model.vision_model]:
            for param in module.parameters():
                assert not param.requires_grad

        for param in self.model.vision_projection.parameters():
            assert param.requires_grad


class TestLLaVAModelVisionEncoders(unittest.TestCase):
    num_weights_by_encoder = {"siglip": 1826368, "radio-g": 2838336}

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def _create_model(self, vision_model_type):
        """创建模型的辅助方法"""
        language_config = TransformerConfig(
            num_hidden_layers=3,
            hidden_size=128,
            num_attention_heads=8,
            use_cpu_initialization=False,
        )
        vision_config = TransformerConfig(
            num_hidden_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            use_cpu_initialization=False,
        )
        vision_projection_config = TransformerConfig(
            num_hidden_layers=2,
            hidden_size=128,
            intermediate_size=72,
            num_attention_heads=1,
            use_cpu_initialization=False,
        )

        language_layer_spec = get_gpt_layer_local_spec()
        vision_layer_spec = deepcopy(language_layer_spec)
        vision_projection_spec = deepcopy(
            language_layer_spec.sublayers_spec.mlp.sublayers_spec
        )

        language_config.language_model_type = "dummy"
        vision_config.vision_model_type = vision_model_type
        model = LLaVAModel(
            language_transformer_config=language_config,
            language_transformer_layer_spec=language_layer_spec,
            language_vocab_size=2048,
            language_max_sequence_length=4096,
            vision_transformer_config=vision_config,
            vision_transformer_layer_spec=vision_layer_spec,
            drop_vision_class_token=False,
            vision_projection_config=vision_projection_config,
            vision_projection_layer_spec=vision_projection_spec,
            img_h=336,
            img_w=336,
            patch_dim=14,
        )

        return model, vision_model_type

    def test_constructor_siglip(self):
        """测试 SigLIP 视觉编码器的构造函数"""
        model, vision_model_type = self._create_model("siglip")
        self.assertIsInstance(model, LLaVAModel)

        num_weights = sum([p.numel() for p in model.parameters()])
        assert num_weights == self.num_weights_by_encoder[vision_model_type], (
            f"num_weights {num_weights} {self.num_weights_by_encoder[vision_model_type]}"
        )

    def test_constructor_radio_g(self):
        """测试 Radio-G 视觉编码器的构造函数"""
        model, vision_model_type = self._create_model("radio-g")
        self.assertIsInstance(model, LLaVAModel)

        num_weights = sum([p.numel() for p in model.parameters()])
        assert num_weights == self.num_weights_by_encoder[vision_model_type], (
            f"num_weights {num_weights} {self.num_weights_by_encoder[vision_model_type]}"
        )

    def test_set_input_tensor_siglip(self):
        """测试 SigLIP 视觉编码器的 set_input_tensor 方法"""
        model, _ = self._create_model("siglip")
        expected_shape = [1, 2, 3, 4]
        input_tensor = paddle.zeros(expected_shape)
        model.set_input_tensor(input_tensor)
        assert model.vision_model.decoder.input_tensor.shape == expected_shape

    def test_set_input_tensor_radio_g(self):
        """测试 Radio-G 视觉编码器的 set_input_tensor 方法"""
        model, _ = self._create_model("radio-g")
        expected_shape = [1, 2, 3, 4]
        input_tensor = paddle.zeros(expected_shape)
        model.set_input_tensor(input_tensor)
        assert model.vision_model.decoder.input_tensor.shape == expected_shape


if __name__ == "__main__":
    unittest.main()
