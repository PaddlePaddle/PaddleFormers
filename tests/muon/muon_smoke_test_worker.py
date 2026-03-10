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

"""
Muon optimizer smoke test — distributed worker script.

Launched by test_muon_smoke.py via TestMultipleGpus.run_2gpu().
Creates a tiny model with 2D weights, wraps with fleet (sharding V2 or V3),
and runs a few training steps with the Muon optimizer.

Success = completes without error and loss is not NaN.
"""

import math
import os

import numpy as np
import paddle
import paddle.nn as nn
from paddle.distributed import fleet
from paddle.nn import ClipGradByGlobalNorm

# Seed for reproducibility
seed = 42
np.random.seed(seed)
paddle.seed(seed)


class SimpleNet(nn.Layer):
    """Tiny model with 2D weights (for Muon) and 1D biases (for AdamW)."""

    def __init__(self, vocab_size=32, hidden_size=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias_attr=False)

    def forward(self, input_ids, labels=None):
        x = self.embedding(input_ids)
        x = self.linear1(x)
        x = paddle.nn.functional.gelu(x)
        x = self.linear2(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits.reshape([-1, logits.shape[-1]]), labels.reshape([-1]))
        return loss, logits


def main():
    # Fleet init
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "sharding_degree": 2,
        "dp_degree": 1,
        "mp_degree": 1,
        "pp_degree": 1,
    }
    strategy.hybrid_configs["sharding_configs"].split_param = True
    fleet.init(is_collective=True, strategy=strategy)

    rank = paddle.distributed.get_rank()
    is_v3 = os.environ.get("FLAGS_sharding_v3", "0") == "1"
    version = "V3" if is_v3 else "V2"

    # Create model
    model = SimpleNet(vocab_size=32, hidden_size=64)

    # Create Muon optimizer
    optimizer = paddle.optimizer.Muon(
        parameters=model.parameters(),
        learning_rate=0.001,
        weight_decay=0.00001,
        grad_clip=ClipGradByGlobalNorm(0.5),
    )

    # AMP O2 wrapping
    from paddle.distributed.fleet.utils.mix_precision_utils import (
        MixPrecisionLayer,
        MixPrecisionOptimizer,
    )

    model = MixPrecisionLayer(model, dtype="bfloat16")
    optimizer = MixPrecisionOptimizer(optimizer)

    # Fleet wrapping
    model = fleet.distributed_model(model)
    optimizer = fleet.distributed_optimizer(optimizer)

    # Training loop
    steps = 3
    losses = []
    for step in range(steps):
        input_ids = paddle.randint(0, 32, [4, 8])
        labels = paddle.randint(0, 32, [4, 8])
        loss, _ = model(input_ids, labels=labels)
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        loss_val = loss.item()
        losses.append(loss_val)
        if rank == 0:
            print(f"[Sharding {version}] Step {step}: loss={loss_val:.4f}")

    # Verify no NaN
    for i, l in enumerate(losses):
        assert not math.isnan(l), f"Step {i}: loss is NaN"
        assert not math.isinf(l), f"Step {i}: loss is Inf"

    if rank == 0:
        print(f"[Sharding {version}] Muon smoke test PASSED ({steps} steps)")


if __name__ == "__main__":
    main()
