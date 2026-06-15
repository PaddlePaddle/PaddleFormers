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

paddle.enable_compat()
import enum
import random
from collections.abc import Generator

import torch
from paddlefleet_ops.deep_gemm.testing import get_arch_major
from paddlefleet_ops.deep_gemm.utils import (
    align,
    ceil_div,
    get_mk_alignment_for_contiguous_layout,
    per_block_cast_to_fp8,
    per_channel_cast_to_fp8,
    per_token_cast_to_fp8,
)


class KernelType(enum.Enum):
    Kernel1D1D = 0
    Kernel1D2D = 1
    KernelNoSF = 2

    def is_1d1d(self):
        return self.value == 0

    def is_1d2d(self):
        return self.value == 1

    def is_nosf(self):
        return self.value == 2


class MajorTypeAB(enum.Enum):
    KMajor = 0
    MNMajor = 1

    def is_k_major(self):
        return self.value == 0

    def is_mn_major(self):
        return self.value == 1


def get_ue8m0_usage(kernel_type: KernelType) -> bool:
    if get_arch_major() == 9:
        return False
    return kernel_type.is_1d1d()


def get_kernel_types(dtype: torch.dtype) -> tuple:
    if dtype == torch.bfloat16:
        return (KernelType.KernelNoSF,)

    return (KernelType.Kernel1D2D,) if get_arch_major() == 9 else (KernelType.Kernel1D1D,)


def get_major_ab(allow_a_mn_major: bool, allow_b_mn_major: bool) -> Generator:
    for major_a in (MajorTypeAB.KMajor, MajorTypeAB.MNMajor):
        for major_b in (MajorTypeAB.KMajor, MajorTypeAB.MNMajor):
            if major_a.is_mn_major() and not allow_a_mn_major:
                continue
            if major_b.is_mn_major() and not allow_b_mn_major:
                continue
            yield major_a, major_b


def enumerate_normal(dtype: torch.dtype) -> Generator:
    assert dtype in (torch.float8_e4m3fn, torch.bfloat16)

    fp32_output_nk = [(256, 7168), (129280, 7168)]
    bf16_output_nk = [
        (2112, 7168),
        (576, 7168),
        (24576, 1536),
        (32768, 512),
        (7168, 16384),
        (4096, 7168),
        (7168, 2048),
    ]
    m_fwd_list, m_bwd_list = (
        [128, 4096],
        [
            4096,
        ],
    )
    nk_list = list(bf16_output_nk)

    # Only BF16 GEMM needs FP32 outputs
    if dtype == torch.bfloat16:
        nk_list += fp32_output_nk

    for kernel_type in get_kernel_types(dtype):
        # Forward
        for m in m_fwd_list:
            for i in range(len(nk_list)):
                n, k = nk_list[i]
                out_dtype = torch.bfloat16 if i < len(bf16_output_nk) else torch.float
                yield (
                    kernel_type,
                    m,
                    n,
                    k,
                    MajorTypeAB.KMajor,
                    MajorTypeAB.KMajor,
                    False,
                    out_dtype,
                )

        # Backward
        for m in m_bwd_list:
            for n, k in nk_list:
                override_major = MajorTypeAB.MNMajor
                override_kernel_type = kernel_type
                if get_arch_major() == 9 and dtype == torch.float8_e4m3fn:
                    override_major = MajorTypeAB.KMajor
                    override_kernel_type = KernelType.Kernel1D1D
                yield (
                    kernel_type,
                    m,
                    k,
                    n,
                    MajorTypeAB.KMajor,
                    override_major,
                    False,
                    torch.bfloat16,
                )  # Dgrad
                yield (
                    override_kernel_type,
                    n,
                    m,
                    k,
                    override_major,
                    override_major,
                    True,
                    torch.float,
                )  # Wgrad
                yield (
                    override_kernel_type,
                    n,
                    m,
                    k,
                    override_major,
                    override_major,
                    False,
                    torch.bfloat16,
                )  # Wgrad


def enumerate_m_grouped_contiguous(dtype: torch.dtype) -> Generator:
    for kernel_type in get_kernel_types(dtype):
        for num_groups, expected_m_per_group, n, k in (
            (4, 8192, 4096, 7168),
            (4, 8192, 7168, 2048),
            (8, 4096, 4096, 7168),
            (8, 4096, 7168, 2048),
        ):
            for major_a, major_b in get_major_ab(False, get_arch_major() != 9 or dtype != torch.float8_e4m3fn):
                yield (
                    kernel_type,
                    num_groups,
                    expected_m_per_group,
                    n,
                    k,
                    major_a,
                    major_b,
                )


def enumerate_m_grouped_masked(dtype: torch.dtype) -> Generator:
    max_m = 4096
    for kernel_type in get_kernel_types(dtype):
        for num_groups, m in ((1, 1024), (2, 512), (4, 256)):
            for n, k in (
                (4096, 7168),
                (7168, 2048),
            ):
                yield kernel_type, num_groups, max_m, m, n, k


def enumerate_k_grouped_contiguous(dtype: torch.dtype):
    # Only K-major is supported for SM90 FP8
    major_a, major_b = (
        (MajorTypeAB.KMajor, MajorTypeAB.KMajor)
        if get_arch_major() == 9 and dtype == torch.float8_e4m3fn
        else (MajorTypeAB.MNMajor, MajorTypeAB.MNMajor)
    )
    # Must with FP32 accumulation and 1D1D kernels
    for num_groups, m, n, expected_k_per_group in (
        (4, 4096, 7168, 8192),
        (4, 7168, 2048, 8192),  # EP64
        (8, 4096, 7168, 4096),
        (8, 7168, 2048, 4096),  # EP32
        (16, 4096, 7168, 2048),
        (16, 7168, 2048, 2048),
    ):  # EP16
        ks = [
            align(
                int(expected_k_per_group * random.uniform(0.7, 1.3)),
                get_mk_alignment_for_contiguous_layout(),
            )
            for _ in range(num_groups)
        ]
        yield num_groups, m, n, major_a, major_b, ks, expected_k_per_group


def enumerate_sf_layout():
    for use_ue8m0 in (False, True):
        for with_transpose in (True, False):
            for mn in (4096, 4097, 8192):
                for k in (128, 7168, 7296):
                    for num_groups in (1, 2, 4):
                        yield mn, k, with_transpose, use_ue8m0, num_groups


def enumerate_k_grouped_sf_layout():
    alignment = get_mk_alignment_for_contiguous_layout()
    assert alignment % 128 == 0
    for mn in (4096, 7168):
        for num_groups, avg_k in ((16, 2048), (8, 4096), (72, 384), (128, 256)):
            ks = [align(int(random.uniform(0.7, 1.3) * avg_k), alignment) for _ in range(num_groups)]
            yield mn, ks, num_groups


def enumerate_transpose():
    for mn in (64, 4096, 16384):
        for delta in (0, 101, 202, 303):
            for k in (128, 1024, 4096, 9984, 16384):
                yield mn + delta, k


def generate_normal(
    m: int,
    n: int,
    k: int,
    major_a: MajorTypeAB,
    major_b: MajorTypeAB,
    accumulate: bool,
    out_dtype: torch.dtype,
    kernel_type: KernelType,
    use_ue8m0: bool = False,
    use_bf16: bool = False,
):
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    d = (
        torch.randn((m, n), device="cuda", dtype=out_dtype) * 32
        if accumulate
        else torch.empty((m, n), device="cuda", dtype=out_dtype)
    )
    c = d if accumulate else None
    ref_d = (a.float() @ b.float().t() + (c if accumulate else 0)).to(out_dtype)

    if use_bf16:
        a = a if major_a.is_k_major() else a.T.contiguous().T
        b = b if major_b.is_k_major() else b.T.contiguous().T
        return a, b, c, d, ref_d

    a_fp8 = per_token_cast_to_fp8(a, use_ue8m0=use_ue8m0)
    b_fp8 = (
        per_token_cast_to_fp8(b, use_ue8m0=use_ue8m0)
        if kernel_type.is_1d1d() and accumulate
        else per_block_cast_to_fp8(b, use_ue8m0=use_ue8m0)
    )
    a_fp8 = a_fp8 if major_a.is_k_major() else (a_fp8[0].T.contiguous().T, a_fp8[1])
    b_fp8 = b_fp8 if major_b.is_k_major() else (b_fp8[0].T.contiguous().T, b_fp8[1])
    return a_fp8, b_fp8, c, d, ref_d


def generate_m_grouped_contiguous(
    num_groups: int,
    expected_m_per_group: int,
    n: int,
    k: int,
    major_a: MajorTypeAB,
    major_b: MajorTypeAB,
    use_ue8m0: bool = False,
    use_bf16: bool = False,
):
    actual_ms = [int(expected_m_per_group * random.uniform(0.7, 1.3)) for _ in range(num_groups)]
    aligned_ms = [align(actual_m, get_mk_alignment_for_contiguous_layout()) for actual_m in actual_ms]
    m = sum(aligned_ms)

    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((num_groups, n, k), device="cuda", dtype=torch.bfloat16)
    m_indices = torch.empty(m, device="cuda", dtype=torch.int32)
    d = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    ref_d = torch.randn((m, n), device="cuda", dtype=torch.bfloat16)

    start = 0
    for i, (actual_m, aligned_m) in enumerate(zip(actual_ms, aligned_ms)):
        actual_end = start + actual_m
        aligned_end = start + aligned_m
        m_indices[start:actual_end] = i
        m_indices[actual_end:aligned_end] = -1
        ref_d[start:aligned_end] = a[start:aligned_end] @ b[i].t()
        start = aligned_end
    ref_d = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(ref_d), ref_d)

    if use_bf16:
        b = b if major_b.is_k_major() else b.mT.contiguous().mT
        return m, a, b, m_indices, d, ref_d

    assert major_a.is_k_major()
    a_fp8 = per_token_cast_to_fp8(a, use_ue8m0=use_ue8m0)
    b_fp8_list = []
    b_fp8_scale = []
    for i in range(num_groups):
        out, scale = per_block_cast_to_fp8(b[i], use_ue8m0=use_ue8m0)
        b_fp8_list.append(out)
        b_fp8_scale.append(scale)
    b_fp8 = (paddle.stack(b_fp8_list, dim=0), paddle.stack(b_fp8_scale, dim=0))
    b_fp8 = b_fp8 if major_b.is_k_major() else (b_fp8[0].mT.contiguous().mT, b_fp8[1])
    return m, a_fp8, b_fp8, m_indices, d, ref_d


def generate_m_grouped_masked(
    num_groups: int,
    max_m: int,
    expected_m_per_group: int,
    n: int,
    k: int,
    use_ue8m0: bool = False,
    use_bf16: bool = False,
):
    a = torch.randn((num_groups, max_m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((num_groups, n, k), device="cuda", dtype=torch.bfloat16)
    d = torch.empty((num_groups, max_m, n), device="cuda", dtype=torch.bfloat16)
    ref_d = torch.einsum("gmk,gnk->gmn", a, b)

    masked_m = torch.empty((num_groups,), device="cuda", dtype=torch.int)
    for j in range(num_groups):
        masked_m[j] = int(expected_m_per_group * random.uniform(0.7, 1.3))
    assert masked_m.amax().item() <= max_m

    if use_bf16:
        return a, b, masked_m, d, ref_d

    a_fp8 = (
        torch.empty_like(a, dtype=torch.bfloat16),
        torch.empty(
            (num_groups, max_m, ceil_div(k, 128)),
            device="cuda",
            dtype=torch.float,
        ),
    )
    b_fp8 = (
        torch.empty_like(b, dtype=torch.bfloat16),
        torch.empty(
            (num_groups, ceil_div(n, 128), ceil_div(k, 128)),
            device="cuda",
            dtype=torch.float,
        ),
    )
    for i in range(num_groups):
        a_fp8[0][i], a_fp8[1][i] = per_token_cast_to_fp8(a[i].to(torch.bfloat16), use_ue8m0=use_ue8m0)
        b_fp8[0][i], b_fp8[1][i] = per_block_cast_to_fp8(b[i].to(torch.bfloat16), use_ue8m0=use_ue8m0)
    a_fp8 = (a_fp8[0].to(torch.float8_e4m3fn), a_fp8[1])
    b_fp8 = (b_fp8[0].to(torch.float8_e4m3fn), b_fp8[1])

    return a_fp8, b_fp8, masked_m, d, ref_d


def generate_k_grouped_contiguous(
    num_groups: int,
    m: int,
    n: int,
    major_a: MajorTypeAB,
    major_b: MajorTypeAB,
    ks: list[int],
    use_ue8m0: bool = False,
    use_bf16: bool = False,
):
    assert get_mk_alignment_for_contiguous_layout() % 128 == 0
    k = sum(ks)

    a = torch.randn((k, m), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((k, n), device="cuda", dtype=torch.bfloat16)
    c = torch.randn((num_groups, m, n), device="cuda", dtype=torch.float) * 32
    d = c
    ref_d = torch.empty_like(c)

    start = 0
    for i, group_k in enumerate(ks):
        end = start + group_k
        ref_d[i] = c[i] + (a[start:end].T @ b[start:end])
        start = end

    if use_bf16:
        assert (major_a, major_b) == (MajorTypeAB.MNMajor, MajorTypeAB.MNMajor)
        return k, a, b, c, d, ref_d

    a_fp8 = per_channel_cast_to_fp8(a, use_ue8m0=use_ue8m0)
    b_fp8 = per_channel_cast_to_fp8(b, use_ue8m0=use_ue8m0)

    # Transpose for K Major A/B
    if (major_a, major_b) == (MajorTypeAB.KMajor, MajorTypeAB.KMajor):
        a, sfa = a_fp8
        b, sfb = b_fp8
        a_chunks = []
        b_chunks = []
        prefix = 0
        for K in ks:
            a_chunks.append(
                a[
                    prefix : prefix + K,
                ].T.flatten()
            )
            b_chunks.append(
                b[
                    prefix : prefix + K,
                ].T.flatten()
            )
            prefix += K
        new_a = torch.concat(a_chunks)
        new_b = torch.concat(b_chunks)
        a_fp8, b_fp8 = (new_a, sfa.T), (new_b, sfb.T)
    else:
        assert (major_a, major_b) == (MajorTypeAB.MNMajor, MajorTypeAB.MNMajor)

    return k, a_fp8, b_fp8, c, d, ref_d
