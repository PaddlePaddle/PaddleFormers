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

import paddle
import paddle.distributed as dist


def get_mesh(pp_idx=None):
    """
    获得pp_idx的mesh
    """
    mesh = dist.fleet.auto.get_mesh()
    if pp_idx is not None and "pp" in mesh.dim_names:
        mesh = mesh.get_mesh_with_dim("pp", pp_idx)
    return mesh


def einsum(rule, a, b):
    """
    Use other ops to replace einsum. The implementation
    is from https://github.com/deepspeedai/DeepSpeed.
    """
    if rule == "s,se->se":
        return a.reshape([a.shape[0], -1]) * b
    elif rule == "se,sc->sec":
        return a.unsqueeze(2) * b.unsqueeze(1)
    elif rule == "se,se->s":
        return paddle.bmm(a.unsqueeze(1), b.unsqueeze(2)).reshape(-1)
    elif rule == "se,sec->sec":
        return paddle.unsqueeze(a, axis=2) * b
    elif rule == "sec,sm->ecm":
        s, e, c = a.shape
        m = b.shape[1]
        return paddle.matmul(a.reshape([a.shape[0], -1]).t(), b).reshape([e, -1, m])
    elif rule == "sec,ecm->sm":
        return paddle.matmul(a.reshape([a.shape[0], -1]), b.reshape([-1, b.shape[-1]]))
    elif rule == "ks,ksm->sm":
        k = b.shape[0]
        s = b.shape[1]
        m = b.shape[2]
        # [k, s] -> [s, k] -> [s, 1, k]
        a = a.t().unsqueeze(1)
        # [k,s,m] -> [k, sm] -> [sm, k] -> [s, m, k]
        b = b.reshape([k, -1]).t().reshape([s, m, k])
        # bmm([s, 1, k], [s, m, k]^t) -> [s, m, 1]
        return paddle.bmm(a, b.transpose(1, 2)).squeeze(2)
    else:
        return paddle.einsum(rule, a, b)
