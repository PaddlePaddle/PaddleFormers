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
"""Two-card load of an fp8_block HF checkpoint through HFDequantLoadTransform.

The "local" read mode is unreachable with a single card, because a whole-tensor
shard always plans a global read.  This launches two ranks so the block-aligned
slice arithmetic in checkpoint_dequant.py is actually exercised.
"""

import inspect
import os
import subprocess
import sys
import tempfile
import unittest

import paddle
import paddle.distributed as dist

WORKER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_dequant_load_dist_logic.py")


def _load_transform_is_supported():
    """Whether the installed Paddle describes distributed transform targets by shard.

    The wheel that first shipped ``load_transform`` described a distributed
    target by its global shape, so ``read_plan()`` never saw a local shard and
    the local path could not run.  ``_apply_load_transform`` gained a
    ``read_plans`` argument in the same change that fixed this, so it is the
    marker for a Paddle new enough to run this test.
    """
    if "load_transform" not in inspect.signature(dist.load_state_dict).parameters:
        return False
    from paddle.distributed.flex_checkpoint.dcp import load_state_dict as dcp

    return "read_plans" in inspect.signature(dcp._apply_load_transform).parameters


@unittest.skipUnless(paddle.device.cuda.device_count() > 1, "test requires multiple GPUs")
@unittest.skipUnless(_load_transform_is_supported(), "paddle is too old to shard a load_transform target")
class TestHFDequantLoadDist(unittest.TestCase):
    def test_block_aligned_shard_loads_its_own_rows(self):
        with tempfile.TemporaryDirectory() as ckpt_dir:
            env = dict(os.environ, ckpt_path=ckpt_dir)
            command = [
                sys.executable,
                "-m",
                "paddle.distributed.launch",
                "--devices",
                "0,1",
                "--log_dir",
                os.path.join(ckpt_dir, "log"),
                WORKER_SCRIPT,
            ]
            process = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertEqual(
                process.returncode,
                0,
                f"two-card load failed:\n{process.stdout}\n{process.stderr}",
            )


if __name__ == "__main__":
    unittest.main()
