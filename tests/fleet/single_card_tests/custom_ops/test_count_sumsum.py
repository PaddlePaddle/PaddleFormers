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

import unittest

import numpy as np
import paddle

from paddlefleet_ops import count_cumsum


class TestCountCumsumOp(unittest.TestCase):
    def setUp(self):
        if not paddle.is_compiled_with_cuda():
            self.skipTest("CUDA is not available")
        np.random.seed(2026)
        paddle.seed(2026)

    def ref_count_cumsum(self, x_np, E, do_cumsum):
        counts = np.zeros(E, dtype=np.int32)
        for v in x_np:
            if 0 <= v < E:
                counts[v] += 1
        if do_cumsum:
            cumsum = np.cumsum(counts)
        else:
            cumsum = np.zeros(0, dtype=np.int32)
        return counts, cumsum

    def run_case(self, dtype, N, E, do_cumsum):
        assert E % 4 == 0
        x_np = np.random.randint(0, E, size=N).astype(dtype)
        if N > 10:
            x_np[:3] = [-1, E, E + 1]
        x = paddle.to_tensor(x_np, place="gpu")
        count_out, cumsum_out = count_cumsum(x, E, do_cumsum)
        paddle.device.cuda.synchronize()
        count_out_np = count_out.cpu().numpy()
        cumsum_out_np = cumsum_out.cpu().numpy()
        ref_counts, ref_cumsum = self.ref_count_cumsum(x_np, E, do_cumsum)

        np.testing.assert_array_equal(
            count_out_np, ref_counts, err_msg="CountOutput mismatch"
        )
        np.testing.assert_array_equal(
            cumsum_out_np, ref_cumsum, err_msg="CumsumOutput mismatch"
        )
        self.assertEqual(count_out_np.shape, (E,))
        if do_cumsum:
            self.assertEqual(cumsum_out_np.shape, (E,))
        else:
            self.assertEqual(cumsum_out_np.shape, (0,))

    def test_count_cumsum_int32(self):
        for do_cumsum in [True]:
            self.run_case(dtype="int32", N=37, E=16, do_cumsum=do_cumsum)
            self.run_case(dtype="int32", N=1000, E=32, do_cumsum=do_cumsum)
            self.run_case(dtype="int32", N=256, E=64, do_cumsum=do_cumsum)

    def test_count_cumsum_int64(self):
        for do_cumsum in [True]:
            self.run_case(dtype="int64", N=123, E=16, do_cumsum=do_cumsum)
            self.run_case(dtype="int64", N=2048, E=32, do_cumsum=do_cumsum)
            self.run_case(dtype="int64", N=512, E=64, do_cumsum=do_cumsum)

    def test_count_cumsum_largeE(self):
        E = 4096
        N = 10000
        self.run_case(dtype="int32", N=N, E=E, do_cumsum=True)
        self.run_case(dtype="int64", N=N, E=E, do_cumsum=True)


if __name__ == "__main__":
    unittest.main()
