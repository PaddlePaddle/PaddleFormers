# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

"""
HFFormatFullParamSaver TP4+Sharding2 Distributed Strategy Test
"""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import paddle
import paddle.nn as nn
from paddle.distributed import fleet

from paddleformers.transformers.linear_utils import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from paddleformers.transformers.model_utils import HFFormatFullParamSaver
from tests.parallel_launch import TestMultipleGpus
from tests.testing_utils import require_paddle_at_least_8_gpu, skip_for_none_ce_case

# Add path
sys.path.append(str(Path(__file__).parent.parent))


class SimpleModelConfig:
    """Simple model configuration"""

    def __init__(self, vocab_size=1000, hidden_size=256, num_layers=4):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers


class SimpleEmbedding(nn.Layer):
    """Simplified embedding layer"""

    def __init__(self, vocab_size, hidden_size):
        super(SimpleEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)

    def forward(self, input_ids):
        return self.embedding(input_ids)


class ParallelTransformerLayer(nn.Layer):
    """Transformer layer with parallel linear layers"""

    def __init__(self, hidden_size):
        super(ParallelTransformerLayer, self).__init__()
        self.hidden_size = hidden_size

        # Self-attention layers with parallel linear layers
        self.q_proj = ColumnParallelLinear(hidden_size, hidden_size, gather_output=False, has_bias=True)
        self.k_proj = ColumnParallelLinear(hidden_size, hidden_size, gather_output=False, has_bias=True)
        self.v_proj = ColumnParallelLinear(hidden_size, hidden_size, gather_output=False, has_bias=True)
        self.attn_out_proj = RowParallelLinear(hidden_size, hidden_size, input_is_parallel=True, has_bias=True)

        # Feed-forward network with parallel linear layers
        self.ffn_proj1 = ColumnParallelLinear(hidden_size, hidden_size * 4, gather_output=False, has_bias=True)
        self.ffn_proj2 = RowParallelLinear(hidden_size * 4, hidden_size, input_is_parallel=True, has_bias=True)

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        # Self-attention with parallel layers
        residual = x

        # Query, Key, Value projections with communication operators
        q = self.q_proj(self.norm1(x))
        k = self.k_proj(self.norm1(x))
        v = self.v_proj(self.norm1(x))

        # Simplified attention computation
        attention_scores = paddle.matmul(q, k.transpose([0, 2, 1])) / (self.hidden_size**0.5)
        attention_weights = nn.functional.softmax(attention_scores, axis=-1)
        attended = paddle.matmul(attention_weights, v)

        # Output projection with communication operators
        attn_output = self.attn_out_proj(attended)
        x = residual + attn_output

        # Feed-forward network with parallel layers
        residual = x
        ff_output = self.ffn_proj1(self.norm2(x))
        ff_output = self.activation(ff_output)
        ff_output = self.ffn_proj2(ff_output)
        x = residual + ff_output

        return x


class ParallelModel(nn.Layer):
    """Model with parallel linear layers"""

    def __init__(self, config):
        super(ParallelModel, self).__init__()
        self.config = config
        self.embedding = SimpleEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.LayerList([ParallelTransformerLayer(config.hidden_size) for _ in range(config.num_layers)])

        # Output layer with parallel linear
        self.output_layer = ColumnParallelLinear(
            config.hidden_size, config.vocab_size, gather_output=True, has_bias=True  # Gather output for final result
        )
        self.norm = nn.LayerNorm(config.hidden_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.output_layer(x)


def initialize_tp4_sharding2_distributed():
    """Initialize TP4+Sharding2 distributed environment"""
    # Get world size
    world_size = paddle.distributed.get_world_size()

    # Configure TP4 + Sharding2 strategy (8 GPU = TP4 * Sharding2)
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,  # Data parallelism degree
        "mp_degree": 4,  # Model parallelism degree (TP4)
        "pp_degree": 1,  # Pipeline parallelism degree
        "sharding_degree": 2,  # Sharding parallelism degree (Sharding2)
    }

    fleet.init(is_collective=True, strategy=strategy)

    # Get hybrid communication group
    hcg = fleet.get_hybrid_communicate_group()

    print("Distributed environment initialized:")
    print(f"  - World size: {world_size}")
    print(f"  - TP degree: {strategy.hybrid_configs['mp_degree']}")
    print(f"  - Sharding degree: {strategy.hybrid_configs['sharding_degree']}")
    print(f"  - Current Rank: {paddle.distributed.get_rank()}")

    return hcg


class TestHFFormatSaverTP4Sharding2(TestMultipleGpus):
    """HFFormatFullParamSaver TP4+Sharding2 Test"""

    def setUp(self):
        """Test setup"""
        self.temp_dir = tempfile.mkdtemp()

        # Set environment variables
        os.environ.update(
            {
                "NCCL_ALGO": "Tree",
                "NVIDIA_TF32_OVERRIDE": "0",
                "NCCL_DEBUG": "INFO",
                "NCCL_IB_TIMEOUT": "22",
                "FLAGS_embedding_deterministic": "1",
                "FLAGS_cudnn_deterministic": "1",
                "Flags_mp_aysnc_allreduce": "1",
                "Flags_skip_mp_c_identity": "1",
                "FLAGS_shard_norm_align_dp": "0",
                "FLAGS_shard_use_reduce": "1",
                "FLAGS_eager_communication_connection": "1",
            }
        )

        # Initialize distributed environment
        self.hcg = initialize_tp4_sharding2_distributed()

    def tearDown(self):
        """Test teardown"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @skip_for_none_ce_case
    @require_paddle_at_least_8_gpu
    def test_save_checkpoint_tp4_sharding2(self):
        """Test checkpoint saving with TP4+Sharding2 distributed strategy"""

        rank = paddle.distributed.get_rank()

        print(f"Rank {rank}: Starting TP4+Sharding2 test")

        # Create model on all ranks for distributed training
        config = SimpleModelConfig(vocab_size=1000, hidden_size=256, num_layers=4)
        model = ParallelModel(config)
        print(f"Rank {rank}: Parallel model created with ColumnParallelLinear layers")

        # Wait for all ranks to synchronize
        paddle.distributed.barrier()

        if rank == 0:
            print(f"Rank {rank}: Creating HFFormatFullParamSaver")

        # Create HFFormatFullParamSaver
        saver = HFFormatFullParamSaver(
            model=model,
            aoa_config=None,
            saved_in_one_node=True,
            memory_growth_threshold=2 * 1024**3,  # 2GB
        )

        # Create save directory
        save_dir = os.path.join(self.temp_dir, "hf_checkpoint_tp4_sharding2")
        os.makedirs(save_dir, exist_ok=True)

        if rank == 0:
            print(f"Rank {rank}: Starting checkpoint save to {save_dir}")

        # Execute save operation
        total_saved_size = saver.save_checkpoint(save_dir, max_shard_size="50MB")

        # Verify save results
        print(f"Rank {rank}: Save completed, file size: {total_saved_size}")

        # Check if files were created
        if rank < saver.num_saver_ranks:  # Only save ranks should create files
            files = os.listdir(save_dir)
            safetensor_files = [f for f in files if f.endswith(".safetensors")]

            self.assertTrue(len(safetensor_files) > 0, f"Rank {rank}: No safetensors files created")
            print(f"Rank {rank}: Created {len(safetensor_files)} safetensors files")

            # Verify file format
            for file in safetensor_files:
                file_path = os.path.join(save_dir, file)
                self.assertTrue(os.path.isfile(file_path), f"Rank {rank}: File {file} should be a regular file")
                file_size = os.path.getsize(file_path)
                self.assertTrue(file_size > 0, f"Rank {rank}: File {file} should have positive size")

        paddle.distributed.barrier()

        # Final verification on rank 0
        if rank == 0:
            all_files = []
            for root, dirs, files in os.walk(save_dir):
                for file in files:
                    all_files.append(os.path.join(root, file))

            print(f"Rank {rank}: Total files created: {len(all_files)}")
            print(f"Rank {rank}: File list: {all_files}")

            # Verify required files exist
            expected_extensions = [".safetensors", ".json"]
            for ext in expected_extensions:
                matching_files = [f for f in all_files if f.endswith(ext)]
                self.assertTrue(len(matching_files) > 0, f"Rank {rank}: No {ext} files found")

        paddle.distributed.barrier()
        print(f"Rank {rank}: TP4+Sharding2 test completed")

    @skip_for_none_ce_case
    @require_paddle_at_least_8_gpu
    def test_save_with_different_shard_sizes(self):
        """Test the impact of different shard sizes on saving"""
        rank = paddle.distributed.get_rank()

        # Create model on all ranks for distributed training
        config = SimpleModelConfig(vocab_size=500, hidden_size=128, num_layers=3)
        model = ParallelModel(config)
        print(f"Rank {rank}: Parallel model created for shard size testing")

        paddle.distributed.barrier()

        # Test different shard sizes
        shard_sizes = ["10MB", "50MB", "100MB"]

        for shard_size in shard_sizes:
            saver = HFFormatFullParamSaver(
                model=model,
                aoa_config=None,
                saved_in_one_node=True,
                memory_growth_threshold=1 * 1024**3,
            )

            save_dir = os.path.join(self.temp_dir, f"test_shard_{shard_size}")
            os.makedirs(save_dir, exist_ok=True)

            saver.save_checkpoint(save_dir, max_shard_size=shard_size)

            if rank < saver.num_saver_ranks:
                files = [f for f in os.listdir(save_dir) if f.endswith(".safetensors")]
                print(f"Rank {rank}: {shard_size} shard size created {len(files)} files")

            paddle.distributed.barrier()


if __name__ == "__main__":
    unittest.main()
