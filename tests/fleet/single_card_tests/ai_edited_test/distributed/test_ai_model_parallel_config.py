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
import os
import sys

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)


# Tests for src/paddleformers.fleet/model_parallel_config.py

import unittest


class TestModelParallelConfig(unittest.TestCase):
    """Tests for ModelParallelConfig dataclass."""

    def test_default_values(self):
        """Test default config values."""
        import paddle

        from paddleformers.fleet.model_parallel_config import ModelParallelConfig

        config = ModelParallelConfig()
        self.assertEqual(config.tensor_model_parallel_size, 1)
        self.assertTrue(config.parallel_output)
        self.assertFalse(config.sequence_parallel)
        self.assertEqual(config.pipeline_model_parallel_size, 1)
        self.assertIsNone(config.virtual_pipeline_model_parallel_size)
        self.assertEqual(config.context_parallel_size, 1)
        self.assertEqual(config.expert_model_parallel_size, 1)
        self.assertFalse(config.fp16)
        self.assertFalse(config.bf16)
        self.assertEqual(config.params_dtype, paddle.float32)
        self.assertEqual(config.fa_version, 2)
        self.assertFalse(config.deterministic_mode)
        self.assertFalse(config.gradient_accumulation_fusion)

    def test_sequence_parallel_false_when_tp_is_one(self):
        """Test that sequence_parallel is set to False when tp_size <= 1."""
        from paddleformers.fleet.model_parallel_config import ModelParallelConfig

        config = ModelParallelConfig(sequence_parallel=True, tensor_model_parallel_size=1)
        self.assertFalse(config.sequence_parallel)

    def test_sequence_parallel_allowed_when_tp_gt_one(self):
        """Test sequence_parallel allowed when tp_size > 1."""
        from paddleformers.fleet.model_parallel_config import ModelParallelConfig

        config = ModelParallelConfig(sequence_parallel=True, tensor_model_parallel_size=4)
        self.assertTrue(config.sequence_parallel)

    def test_expert_tensor_parallel_size_default(self):
        """Test expert_tensor_parallel_size defaults to tensor_model_parallel_size."""
        from paddleformers.fleet.model_parallel_config import ModelParallelConfig

        config = ModelParallelConfig(tensor_model_parallel_size=4)
        self.assertEqual(config.expert_tensor_parallel_size, 4)

    def test_autocast_dtype_default(self):
        """Test autocast_dtype defaults to params_dtype."""
        import paddle

        from paddleformers.fleet.model_parallel_config import ModelParallelConfig

        config = ModelParallelConfig(params_dtype=paddle.float16)
        self.assertEqual(config.autocast_dtype, paddle.float16)

    def test_microbatch_group_size_per_vp_stage_default(self):
        """Test microbatch_group_size_per_vp_stage defaults to pp_size."""
        from paddleformers.fleet.model_parallel_config import ModelParallelConfig

        config = ModelParallelConfig(pipeline_model_parallel_size=4)
        self.assertEqual(config.microbatch_group_size_per_vp_stage, 4)

    def test_sequence_parallel_without_tp_raises(self):
        """Test that sequence_parallel without tensor parallelism raises."""
        from paddleformers.fleet.model_parallel_config import ModelParallelConfig

        # tensor_model_parallel_size=1 triggers auto-disable, but if we set it
        # to >1 and sequence_parallel, it should work.
        config = ModelParallelConfig(
            tensor_model_parallel_size=2,
            sequence_parallel=True,
        )
        self.assertTrue(config.sequence_parallel)

    def test_defer_embedding_wgrad_without_pp_raises(self):
        """Test defer_embedding_wgrad with pp=1 raises ValueError."""
        from paddleformers.fleet.model_parallel_config import ModelParallelConfig

        with self.assertRaises(ValueError) as ctx:
            ModelParallelConfig(
                defer_embedding_wgrad_compute=True,
                pipeline_model_parallel_size=1,
                gradient_accumulation_fusion=True,
            )
        self.assertIn("pipeline model parallel", str(ctx.exception).lower())

    def test_defer_embedding_wgrad_without_gaf_raises(self):
        """Test defer_embedding_wgrad without gradient_accumulation_fusion raises."""
        from paddleformers.fleet.model_parallel_config import ModelParallelConfig

        with self.assertRaises(ValueError) as ctx:
            ModelParallelConfig(
                defer_embedding_wgrad_compute=True,
                pipeline_model_parallel_size=2,
                gradient_accumulation_fusion=False,
            )
        self.assertIn("gradient accumulation fusion", str(ctx.exception).lower())

    def test_defer_embedding_wgrad_negative_limit_raises(self):
        """Test defer_embedding_wgrad with negative limit raises ValueError."""
        from paddleformers.fleet.model_parallel_config import ModelParallelConfig

        with self.assertRaises(ValueError) as ctx:
            ModelParallelConfig(
                defer_embedding_wgrad_compute=True,
                pipeline_model_parallel_size=2,
                gradient_accumulation_fusion=True,
                wgrad_deferral_limit=-1,
            )
        self.assertIn("wgrad deferral limit", str(ctx.exception).lower())

    def test_defer_embedding_wgrad_valid(self):
        """Test valid defer_embedding_wgrad config."""
        from paddleformers.fleet.model_parallel_config import ModelParallelConfig

        config = ModelParallelConfig(
            defer_embedding_wgrad_compute=True,
            pipeline_model_parallel_size=2,
            gradient_accumulation_fusion=True,
            wgrad_deferral_limit=5,
        )
        self.assertTrue(config.defer_embedding_wgrad_compute)
        self.assertEqual(config.wgrad_deferral_limit, 5)

    def test_expert_and_tensor_parallel_requires_sequence_parallel(self):
        """Test that expert + tensor parallel requires sequence parallel."""
        from paddleformers.fleet.model_parallel_config import ModelParallelConfig

        with self.assertRaises(ValueError) as ctx:
            ModelParallelConfig(
                expert_model_parallel_size=2,
                tensor_model_parallel_size=2,
                sequence_parallel=False,
            )
        self.assertIn("sequence parallelism must be used", str(ctx.exception))

    def test_expert_and_tensor_parallel_with_sequence_parallel_ok(self):
        """Test that expert + tensor + sequence parallel is valid."""
        from paddleformers.fleet.model_parallel_config import ModelParallelConfig

        config = ModelParallelConfig(
            expert_model_parallel_size=2,
            tensor_model_parallel_size=2,
            sequence_parallel=True,
        )
        self.assertTrue(config.sequence_parallel)

    def test_overlap_p2p_comm_warmup_flush_valid(self):
        """Test overlap_p2p_comm_warmup_flush with valid config."""
        from paddleformers.fleet.model_parallel_config import ModelParallelConfig

        config = ModelParallelConfig(
            overlap_p2p_comm=True,
            batch_p2p_comm=False,
            overlap_p2p_comm_warmup_flush=True,
        )
        self.assertTrue(config.overlap_p2p_comm_warmup_flush)

    def test_overlap_p2p_comm_warmup_flush_with_batch_raises(self):
        """Test overlap_p2p_comm_warmup_flush with batch_p2p_comm raises."""
        from paddleformers.fleet.model_parallel_config import ModelParallelConfig

        with self.assertRaises(ValueError):
            ModelParallelConfig(
                overlap_p2p_comm=True,
                batch_p2p_comm=True,
                overlap_p2p_comm_warmup_flush=True,
            )

    def test_overlap_p2p_comm_warmup_flush_without_overlap_raises(self):
        """Test overlap_p2p_comm_warmup_flush without overlap_p2p_comm raises."""
        from paddleformers.fleet.model_parallel_config import ModelParallelConfig

        with self.assertRaises(ValueError):
            ModelParallelConfig(
                overlap_p2p_comm=False,
                batch_p2p_comm=False,
                overlap_p2p_comm_warmup_flush=True,
            )

    def test_custom_values(self):
        """Test custom config values."""
        from paddleformers.fleet.model_parallel_config import ModelParallelConfig

        config = ModelParallelConfig(
            tensor_model_parallel_size=8,
            pipeline_model_parallel_size=4,
            virtual_pipeline_model_parallel_size=2,
            context_parallel_size=2,
            expert_model_parallel_size=2,
            fp16=True,
            perform_initialization=False,
            use_cpu_initialization=True,
            variable_seq_lengths=True,
            overlap_p2p_comm=True,
            batch_p2p_comm=False,
            cpu_offloading=True,
            cpu_offloading_num_layers=5,
            tp_comm_overlap=True,
            cross_entropy_loss_fusion=True,
            sequence_parallel=True,
        )
        self.assertEqual(config.tensor_model_parallel_size, 8)
        self.assertEqual(config.pipeline_model_parallel_size, 4)
        self.assertEqual(config.virtual_pipeline_model_parallel_size, 2)
        self.assertEqual(config.context_parallel_size, 2)
        self.assertEqual(config.expert_model_parallel_size, 2)
        self.assertTrue(config.fp16)
        self.assertFalse(config.perform_initialization)
        self.assertTrue(config.use_cpu_initialization)
        self.assertTrue(config.variable_seq_lengths)
        self.assertTrue(config.overlap_p2p_comm)
        self.assertFalse(config.batch_p2p_comm)
        self.assertTrue(config.cpu_offloading)
        self.assertEqual(config.cpu_offloading_num_layers, 5)
        self.assertTrue(config.tp_comm_overlap)
        self.assertTrue(config.cross_entropy_loss_fusion)


if __name__ == "__main__":
    unittest.main()
