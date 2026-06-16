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

import contextlib
import random
import unittest
from functools import partial

import numpy as np
import paddle
from paddle.distributed import fleet

import paddleformers.fleet.parallel_state as ps
import paddleformers.fleet.tensor_parallel
from paddleformers.fleet.models.kimi_k25.kimi_k25_builders import (
    kimi_k25_vision_builder,
)
from paddleformers.fleet.models.kimi_k25.kimi_k25_model import (
    KimiK25VisionModel,
)
from paddleformers.fleet.transformer.transformer_config import TransformerConfig


def _set_random_seed(
    seed_: int,
    data_parallel_random_init: bool = False,
    te_rng_tracker: bool = False,
    inference_rng_tracker: bool = False,
    use_cudagraphable_rng: bool = False,
):
    """Set random seed for reproducibility."""
    if seed_ is not None and seed_ > 0:
        # Ensure that different pipeline MP stages get different seeds.
        seed = seed_ + (100 * ps.get_pipeline_model_parallel_rank())
        # Ensure different data parallel ranks get different seeds
        if data_parallel_random_init:
            seed = seed + (10 * ps.get_data_parallel_rank())
        random.seed(seed)
        np.random.seed(seed)
        paddle.manual_seed(seed)

        if (
            paddle.distributed.is_initialized()
            and paddle.cuda.device_count() > 0
        ):
            paddleformers.fleet.tensor_parallel.model_parallel_cuda_manual_seed(
                seed,
                te_rng_tracker,
                inference_rng_tracker,
                use_cudagraphable_rng,
            )
    else:
        raise ValueError(f"Seed ({seed_}) should be a positive integer.")


def create_kimi_k25_vision_config(**kwargs):
    """Create a KimiK25 vision config with proper settings."""
    # Default base config for TransformerConfig
    base_config = {
        "num_hidden_layers": 2,
        "hidden_size": 64,
        "num_attention_heads": 4,
        "use_cpu_initialization": False,
    }

    # KimiK25 specific attributes that should not be passed to TransformerConfig.__init__
    kimi_specific_keys = {
        "patch_size",
        "init_pos_emb_height",
        "init_pos_emb_width",
        "init_pos_emb_time",
        "pos_emb_type",
        "merge_kernel_size",
        "mm_hidden_size",
        "text_hidden_size",
        "intermediate_size",
        "projector_ln_eps",
        "max_height",
        "max_width",
    }

    # Separate kimi-specific kwargs from base config
    kimi_kwargs = {}
    for key in kimi_specific_keys:
        if key in kwargs:
            kimi_kwargs[key] = kwargs.pop(key)

    base_config.update(kwargs)
    config = TransformerConfig(**base_config)

    # Set KimiK25 specific attributes (these are class-level attributes in provider)
    config.patch_size = kimi_kwargs.get("patch_size", 14)
    config.init_pos_emb_height = kimi_kwargs.get("init_pos_emb_height", 4)
    config.init_pos_emb_width = kimi_kwargs.get("init_pos_emb_width", 4)
    config.init_pos_emb_time = kimi_kwargs.get("init_pos_emb_time", 2)
    config.pos_emb_type = kimi_kwargs.get("pos_emb_type", "divided_fixed")
    config.merge_kernel_size = kimi_kwargs.get("merge_kernel_size", (2, 2))
    config.mm_hidden_size = kimi_kwargs.get("mm_hidden_size", 64)
    config.text_hidden_size = kimi_kwargs.get("text_hidden_size", 128)
    config.intermediate_size = kimi_kwargs.get("intermediate_size", 256)
    config.projector_ln_eps = kimi_kwargs.get("projector_ln_eps", 1e-5)
    config.max_height = kimi_kwargs.get("max_height", 64)
    config.max_width = kimi_kwargs.get("max_width", 64)
    config.normalization = kwargs.get("normalization", "LayerNorm")

    # Vision model specific
    config.gated_linear_unit = False
    config.high_precision_rope = True

    return config


def build_kimi_k25_vision_model(config):
    """Build KimiK25 vision model from config."""
    pp_size = config.pipeline_model_parallel_size

    model_init_device_context = contextlib.nullcontext
    if config.init_model_with_meta_device:
        model_init_device_context = partial(paddle.device, device="meta")

    with model_init_device_context():
        model = kimi_k25_vision_builder(
            config,
            seg_method="layer:TransformerLayer|EmptyLayer",
            num_stages=pp_size,
        )
    return model


class TestKimiK25VisionModelFleet(unittest.TestCase):
    """Test KimiK25VisionModel with complete model."""

    @classmethod
    def setUpClass(cls):
        """Set up distributed environment once for all tests."""
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

    def _create_model(self, **config_kwargs):
        """Create a KimiK25VisionModel for testing."""
        _set_random_seed(46)
        config = create_kimi_k25_vision_config(**config_kwargs)
        model = build_kimi_k25_vision_model(config)
        return model, config

    def _get_attn_mask_startend_row_indices(self, grid_thws: paddle.Tensor):
        """Compute attention mask start/end row indices."""
        lengths = paddle.cat(
            (
                paddle.zeros([1], dtype=grid_thws.dtype),
                grid_thws[:, 0] * grid_thws[:, 1] * grid_thws[:, 2],
            )
        )
        cu_seqlens = lengths.cumsum(axis=0).cast(paddle.int32)
        cu_seqlens_rm_first = cu_seqlens[1:]
        cu_seqlens_rm_last = cu_seqlens[:-1]
        repeats = cu_seqlens_rm_first - cu_seqlens_rm_last

        startend_row_indices_lts = paddle.repeat_interleave(
            cu_seqlens_rm_first, repeats
        ).reshape([1, 1, -1, 1])
        startend_row_indices_ute = paddle.repeat_interleave(
            cu_seqlens_rm_last, repeats
        ).reshape([1, 1, -1, 1])
        startend_row_indices = paddle.concat(
            [startend_row_indices_lts, startend_row_indices_ute], axis=-1
        )
        return startend_row_indices

    def test_model_construction(self):
        """Test that model can be constructed."""
        model, config = self._create_model()
        self.assertIsInstance(model, KimiK25VisionModel)

        # Check parameter count
        num_weights = sum([p.numel() for p in model.parameters()])
        self.assertGreater(num_weights, 0)
        print(f"Model has {num_weights} parameters")

    def test_model_construction_with_different_layers(self):
        """Test model construction with different number of layers."""
        for num_layers in [1, 2, 4]:
            model, config = self._create_model(num_hidden_layers=num_layers)
            self.assertIsInstance(model, KimiK25VisionModel)

    def test_model_construction_with_different_hidden_size(self):
        """Test model construction with different hidden sizes."""
        for hidden_size in [32, 64, 128]:
            model, config = self._create_model(
                hidden_size=hidden_size,
                num_attention_heads=hidden_size // 16,  # head_dim = 16
            )
            self.assertIsInstance(model, KimiK25VisionModel)

    def test_forward_single_image(self):
        """Test forward pass with a single image."""
        model, config = self._create_model()

        # Single image: 1 frame, 4x4 patches (56x56 image with patch_size=14)
        num_patches = 4 * 4  # 16 patches
        pixel_values = paddle.randn([num_patches, 3, 14, 14])
        grid_thws = paddle.to_tensor([[1, 4, 4]])
        attn_mask_startend_row_indices = (
            self._get_attn_mask_startend_row_indices(grid_thws)
        )

        input_dict = {
            "pixel_values": pixel_values,
            "grid_thws": grid_thws,
            "attn_mask_startend_row_indices": attn_mask_startend_row_indices,
        }

        output = model(input_dict)
        self.assertIn("hidden_states", output)
        print(
            f"Single image output hidden_states type: {type(output['hidden_states'])}"
        )

    def test_forward_multi_frame_video(self):
        """Test forward pass with multiple frames (video)."""
        model, config = self._create_model()

        # Video: 2 frames, each 4x4 patches
        num_patches = 2 * 4 * 4  # 32 patches
        pixel_values = paddle.randn([num_patches, 3, 14, 14])
        grid_thws = paddle.to_tensor([[2, 4, 4]])
        attn_mask_startend_row_indices = (
            self._get_attn_mask_startend_row_indices(grid_thws)
        )

        input_dict = {
            "pixel_values": pixel_values,
            "grid_thws": grid_thws,
            "attn_mask_startend_row_indices": attn_mask_startend_row_indices,
        }

        output = model(input_dict)
        self.assertIn("hidden_states", output)
        print(
            f"Multi-frame output hidden_states type: {type(output['hidden_states'])}"
        )

    def test_forward_batch_images(self):
        """Test forward pass with a batch of images."""
        model, config = self._create_model()

        # Batch of 2 images with different sizes
        # Image 1: 1 frame, 4x4 patches = 16 patches
        # Image 2: 1 frame, 4x4 patches = 16 patches
        total_patches = 32
        pixel_values = paddle.randn([total_patches, 3, 14, 14])
        grid_thws = paddle.to_tensor([[1, 4, 4], [1, 4, 4]])
        attn_mask_startend_row_indices = (
            self._get_attn_mask_startend_row_indices(grid_thws)
        )

        input_dict = {
            "pixel_values": pixel_values,
            "grid_thws": grid_thws,
            "attn_mask_startend_row_indices": attn_mask_startend_row_indices,
        }

        output = model(input_dict)
        self.assertIn("hidden_states", output)

    def test_forward_variable_size_images(self):
        """Test forward pass with variable size images."""
        model, config = self._create_model()

        # Two images with different sizes
        # Image 1: 1 frame, 4x4 patches = 16 patches
        # Image 2: 1 frame, 6x6 patches = 36 patches
        total_patches = 16 + 36
        pixel_values = paddle.randn([total_patches, 3, 14, 14])
        grid_thws = paddle.to_tensor([[1, 4, 4], [1, 6, 6]])
        attn_mask_startend_row_indices = (
            self._get_attn_mask_startend_row_indices(grid_thws)
        )

        input_dict = {
            "pixel_values": pixel_values,
            "grid_thws": grid_thws,
            "attn_mask_startend_row_indices": attn_mask_startend_row_indices,
        }

        output = model(input_dict)
        self.assertIn("hidden_states", output)

    def test_forward_with_attention_mask(self):
        """Test forward pass with custom attention mask."""
        model, config = self._create_model()

        num_patches = 16
        pixel_values = paddle.randn([num_patches, 3, 14, 14])
        grid_thws = paddle.to_tensor([[1, 4, 4]])
        attn_mask_startend_row_indices = (
            self._get_attn_mask_startend_row_indices(grid_thws)
        )

        # Create attention mask
        attention_mask = paddle.ones([1, num_patches])

        input_dict = {
            "pixel_values": pixel_values,
            "grid_thws": grid_thws,
            "attention_mask": attention_mask,
            "attn_mask_startend_row_indices": attn_mask_startend_row_indices,
        }

        output = model(input_dict)
        self.assertIn("hidden_states", output)

    def test_model_dtype_float32(self):
        """Test model with float32 dtype."""
        model, config = self._create_model(params_dtype=paddle.float32)

        num_patches = 16
        pixel_values = paddle.randn(
            [num_patches, 3, 14, 14], dtype=paddle.float32
        )
        grid_thws = paddle.to_tensor([[1, 4, 4]])
        attn_mask_startend_row_indices = (
            self._get_attn_mask_startend_row_indices(grid_thws)
        )

        input_dict = {
            "pixel_values": pixel_values,
            "grid_thws": grid_thws,
            "attn_mask_startend_row_indices": attn_mask_startend_row_indices,
        }

        output = model(input_dict)
        self.assertIn("hidden_states", output)

    @unittest.skip(
        "float16 dtype has issue with GPU weight initialization in paddleformers.fleet"
    )
    def test_model_dtype_float16(self):
        """Test model with float16 dtype."""
        model, config = self._create_model(params_dtype=paddle.float16)

        num_patches = 16
        pixel_values = paddle.randn(
            [num_patches, 3, 14, 14], dtype=paddle.float16
        )
        grid_thws = paddle.to_tensor([[1, 4, 4]])
        attn_mask_startend_row_indices = (
            self._get_attn_mask_startend_row_indices(grid_thws)
        )

        input_dict = {
            "pixel_values": pixel_values,
            "grid_thws": grid_thws,
            "attn_mask_startend_row_indices": attn_mask_startend_row_indices,
        }

        output = model(input_dict)
        self.assertIn("hidden_states", output)

    def test_model_different_merge_kernel_size(self):
        """Test model with different merge kernel sizes."""
        for kernel_size in [(1, 1), (2, 2), (2, 4)]:
            model, config = self._create_model(merge_kernel_size=kernel_size)

            num_patches = 16
            pixel_values = paddle.randn([num_patches, 3, 14, 14])
            grid_thws = paddle.to_tensor([[1, 4, 4]])
            attn_mask_startend_row_indices = (
                self._get_attn_mask_startend_row_indices(grid_thws)
            )

            input_dict = {
                "pixel_values": pixel_values,
                "grid_thws": grid_thws,
                "attn_mask_startend_row_indices": attn_mask_startend_row_indices,
            }

            output = model(input_dict)
            self.assertIn("hidden_states", output)
            print(f"Merge kernel {kernel_size} test passed")

    def test_model_gradient_computation(self):
        """Test that gradients can be computed."""
        model, config = self._create_model()

        num_patches = 16
        pixel_values = paddle.randn([num_patches, 3, 14, 14])
        pixel_values.stop_gradient = False
        grid_thws = paddle.to_tensor([[1, 4, 4]])
        attn_mask_startend_row_indices = (
            self._get_attn_mask_startend_row_indices(grid_thws)
        )

        input_dict = {
            "pixel_values": pixel_values,
            "grid_thws": grid_thws,
            "attn_mask_startend_row_indices": attn_mask_startend_row_indices,
        }

        output = model(input_dict)
        hidden_states = output["hidden_states"]

        # Compute loss and backward
        if isinstance(hidden_states, list):
            loss = sum(h.sum() for h in hidden_states)
        else:
            loss = hidden_states.sum()

        loss.backward()

        # Check that gradients exist
        has_grad = False
        for param in model.parameters():
            if param.grad is not None:
                has_grad = True
                break
        self.assertTrue(has_grad, "Model should have gradients after backward")

    def test_model_eval_mode(self):
        """Test model in evaluation mode."""
        model, config = self._create_model()
        model.eval()

        num_patches = 16
        pixel_values = paddle.randn([num_patches, 3, 14, 14])
        grid_thws = paddle.to_tensor([[1, 4, 4]])
        attn_mask_startend_row_indices = (
            self._get_attn_mask_startend_row_indices(grid_thws)
        )

        input_dict = {
            "pixel_values": pixel_values,
            "grid_thws": grid_thws,
            "attn_mask_startend_row_indices": attn_mask_startend_row_indices,
        }

        output = model(input_dict)
        self.assertIn("hidden_states", output)

    def test_model_train_mode(self):
        """Test model in training mode."""
        model, config = self._create_model()
        model.train()

        num_patches = 16
        pixel_values = paddle.randn([num_patches, 3, 14, 14])
        grid_thws = paddle.to_tensor([[1, 4, 4]])
        attn_mask_startend_row_indices = (
            self._get_attn_mask_startend_row_indices(grid_thws)
        )

        input_dict = {
            "pixel_values": pixel_values,
            "grid_thws": grid_thws,
            "attn_mask_startend_row_indices": attn_mask_startend_row_indices,
        }

        output = model(input_dict)
        self.assertIn("hidden_states", output)


class TestKimiK25VisionModelConfigVariations(unittest.TestCase):
    """Test KimiK25VisionModel with various config variations."""

    @classmethod
    def setUpClass(cls):
        """Set up distributed environment once for all tests."""
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

    def test_config_with_empty_layers(self):
        """Test config with empty layers in head and tail."""
        config = create_kimi_k25_vision_config(
            num_hidden_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            use_cpu_initialization=True,
            num_empty_layers_add_in_head=1,
            num_empty_layers_add_in_tail=1,
        )
        model = build_kimi_k25_vision_model(config)
        self.assertIsInstance(model, KimiK25VisionModel)

    def test_config_with_qk_norm(self):
        """Test config with QK normalization enabled."""
        config = create_kimi_k25_vision_config(
            num_hidden_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            use_cpu_initialization=True,
            use_qk_norm=True,
        )
        model = build_kimi_k25_vision_model(config)
        self.assertIsInstance(model, KimiK25VisionModel)


if __name__ == "__main__":
    unittest.main()
