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
from unittest.mock import MagicMock, patch

import paddle

from paddleformers.transformers.auto_utils import einsum, get_mesh

IS_SINGLE_CARD = paddle.distributed.get_world_size() <= 1


class TestEinsum(unittest.TestCase):
    """Tests for the einsum replacement function."""

    def test_rule_s_se_to_se(self):
        a = paddle.randn([4, 3])
        b = paddle.randn([4, 3])
        result = einsum("s,se->se", a, b)
        expected = a.reshape([a.shape[0], -1]) * b
        self.assertTrue(paddle.allclose(result, expected, atol=1e-5))

    def test_rule_se_sc_to_sec(self):
        a = paddle.randn([4, 3])
        b = paddle.randn([4, 5])
        result = einsum("se,sc->sec", a, b)
        expected = a.unsqueeze(2) * b.unsqueeze(1)
        self.assertTrue(paddle.allclose(result, expected, atol=1e-5))

    def test_rule_se_se_to_s(self):
        a = paddle.randn([4, 3])
        b = paddle.randn([4, 3])
        result = einsum("se,se->s", a, b)
        expected = paddle.bmm(a.unsqueeze(1), b.unsqueeze(2)).reshape(-1)
        self.assertTrue(paddle.allclose(result, expected, atol=1e-5))

    def test_rule_se_sec_to_sec(self):
        a = paddle.randn([4, 3])
        b = paddle.randn([4, 3, 5])
        result = einsum("se,sec->sec", a, b)
        expected = paddle.unsqueeze(a, axis=2) * b
        self.assertTrue(paddle.allclose(result, expected, atol=1e-5))

    def test_rule_sec_sm_to_ecm(self):
        s, e, c, m = 4, 3, 5, 6
        a = paddle.randn([s, e, c])
        b = paddle.randn([s, m])
        result = einsum("sec,sm->ecm", a, b)
        expected = paddle.matmul(a.reshape([a.shape[0], -1]).t(), b).reshape([e, -1, m])
        self.assertTrue(paddle.allclose(result, expected, atol=1e-4))

    def test_rule_sec_ecm_to_sm(self):
        s, e, c, m = 4, 3, 5, 6
        a = paddle.randn([s, e, c])
        b = paddle.randn([e, c, m])
        result = einsum("sec,ecm->sm", a, b)
        expected = paddle.matmul(a.reshape([a.shape[0], -1]), b.reshape([-1, b.shape[-1]]))
        self.assertTrue(paddle.allclose(result, expected, atol=1e-4))

    def test_rule_ks_ksm_to_sm(self):
        k, s, m = 3, 4, 5
        a = paddle.randn([k, s])
        b = paddle.randn([k, s, m])
        result = einsum("ks,ksm->sm", a, b)
        a_t = a.t().unsqueeze(1)
        b_reshaped = b.reshape([k, -1]).t().reshape([s, m, k])
        expected = paddle.bmm(a_t, b_reshaped.transpose(1, 2)).squeeze(2)
        self.assertTrue(paddle.allclose(result, expected, atol=1e-4))

    def test_fallback_to_paddle_einsum(self):
        a = paddle.randn([3, 4])
        b = paddle.randn([4, 5])
        result = einsum("ij,jk->ik", a, b)
        expected = paddle.matmul(a, b)
        self.assertTrue(paddle.allclose(result, expected, atol=1e-4))


@unittest.skipIf(
    IS_SINGLE_CARD,
    "get_mesh requires distributed environment with fleet initialized",
)
class TestGetMesh(unittest.TestCase):
    """Tests for the get_mesh function."""

    @patch("paddleformers.transformers.auto_utils.dist")
    def test_get_mesh_no_pp_idx(self, mock_dist):
        mock_mesh = MagicMock()
        mock_dist.fleet.auto.get_mesh.return_value = mock_mesh
        result = get_mesh()
        mock_dist.fleet.auto.get_mesh.assert_called_once()
        self.assertEqual(result, mock_mesh)

    @patch("paddleformers.transformers.auto_utils.dist")
    def test_get_mesh_with_pp_idx_and_pp_in_dim_names(self, mock_dist):
        mock_mesh = MagicMock()
        mock_mesh.dim_names = ["pp", "dp"]
        mock_sub_mesh = MagicMock()
        mock_mesh.get_mesh_with_dim.return_value = mock_sub_mesh
        mock_dist.fleet.auto.get_mesh.return_value = mock_mesh
        result = get_mesh(pp_idx=0)
        mock_mesh.get_mesh_with_dim.assert_called_once_with("pp", 0)
        self.assertEqual(result, mock_sub_mesh)

    @patch("paddleformers.transformers.auto_utils.dist")
    def test_get_mesh_with_pp_idx_but_no_pp_in_dim_names(self, mock_dist):
        mock_mesh = MagicMock()
        mock_mesh.dim_names = ["dp"]
        mock_dist.fleet.auto.get_mesh.return_value = mock_mesh
        result = get_mesh(pp_idx=0)
        self.assertEqual(result, mock_mesh)


if __name__ == "__main__":
    unittest.main()
