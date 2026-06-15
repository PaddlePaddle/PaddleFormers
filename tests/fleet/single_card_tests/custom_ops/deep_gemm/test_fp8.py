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
import copy
import random

import torch
from paddlefleet_ops import deep_gemm
from paddlefleet_ops.deep_gemm.testing import calc_diff, get_arch_major, ignore_env

from .generators import (
    KernelType,
    enumerate_k_grouped_contiguous,
    enumerate_m_grouped_contiguous,
    enumerate_m_grouped_masked,
    enumerate_normal,
    generate_k_grouped_contiguous,
    generate_m_grouped_contiguous,
    generate_m_grouped_masked,
    generate_normal,
    get_ue8m0_usage,
)


@ignore_env("DG_JIT_PTXAS_CHECK", lambda: get_arch_major() == 9)
def test_gemm() -> None:
    print("Testing GEMM:")
    scores = []
    for (
        kernel_type,
        m,
        n,
        k,
        major_a,
        major_b,
        accumulate,
        out_dtype,
    ) in enumerate_normal(torch.float8_e4m3fn):
        major_opt = "N" if major_a.is_k_major() else "T"
        major_opt += "T" if major_b.is_k_major() else "N"
        out_opt = "FP32" if out_dtype == torch.float else "BF16"
        acc_opt = f"acc={int(accumulate)}"
        kernel_opt = "1D1D" if kernel_type.is_1d1d() else "1D2D"
        use_ue8m0 = get_ue8m0_usage(kernel_type)
        disable_ue8m0_cast = not use_ue8m0
        recipe = (1, 1, 128) if kernel_type.is_1d1d() and accumulate else None

        for test_alias in (False, True):
            a, b, c, d, ref_d = generate_normal(
                m,
                n,
                k,
                major_a,
                major_b,
                accumulate,
                out_dtype,
                kernel_type,
                use_ue8m0=use_ue8m0,
            )
            func_name = f"fp8_gemm_{major_opt.lower() if test_alias else 'nt'}"
            if test_alias:
                a = a if major_a.is_k_major() else (a[0].T, a[1].T)
                b = b if major_b.is_k_major() else (b[0].T, b[1].T)
                assert a[0].is_contiguous() and b[0].is_contiguous()
            getattr(deep_gemm, func_name)(
                a,
                b,
                d,
                c=c,
                disable_ue8m0_cast=disable_ue8m0_cast,
                recipe=recipe,
            )
            diff = calc_diff(d, ref_d)
            assert diff < 0.001, (
                f"{m=}, {n=}, {k=}, {kernel_opt}, {major_opt=}, {accumulate=}, {out_dtype=}, "
                f"{diff:.5f}, alias={test_alias}"
            )


def test_m_grouped_gemm_contiguous() -> None:
    print("Testing m-grouped contiguous GEMM:")

    for (
        kernel_type,
        num_groups,
        expected_m_per_group,
        n,
        k,
        major_a,
        major_b,
    ) in enumerate_m_grouped_contiguous(dtype=torch.float8_e4m3fn):
        major_opt = "N" if major_a.is_k_major() else "T"
        major_opt += "T" if major_b.is_k_major() else "N"
        kernel_opt = "1D1D" if kernel_type.is_1d1d() else "1D2D"
        use_ue8m0 = get_ue8m0_usage(kernel_type)
        disable_ue8m0_cast = not use_ue8m0

        for test_alias in (False, True):
            m, a, b, m_indices, d, ref_d = generate_m_grouped_contiguous(
                num_groups,
                expected_m_per_group,
                n,
                k,
                major_a,
                major_b,
                use_ue8m0=use_ue8m0,
            )
            func_name = f"m_grouped_fp8_gemm_{(major_opt.lower() if test_alias else 'nt')}_contiguous"
            if test_alias:
                assert major_a.is_k_major()
                b = b if major_b.is_k_major() else (b[0].mT, b[1].mT)
                assert a[0].is_contiguous() and b[0].is_contiguous()
            getattr(deep_gemm, func_name)(a, b, d, m_indices, disable_ue8m0_cast=disable_ue8m0_cast)
            d = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(d), d)
            diff = calc_diff(d, ref_d)
            assert diff < 0.001, f"{m=}, {n=}, {k=}, {major_opt}, {kernel_opt}, {diff:.5f}, alias={test_alias}"
    print()


def test_m_grouped_gemm_masked() -> None:
    print("Testing m-grouped masked GEMM:")

    # TODO: when the actual `m` is greater than `expected_m_per_group`, efficiency may significantly decrease.
    for (
        kernel_type,
        num_groups,
        max_m,
        expected_m_per_group,
        n,
        k,
    ) in enumerate_m_grouped_masked(torch.float8_e4m3fn):
        kernel_opt = "1D1D" if kernel_type.is_1d1d() else "1D2D"
        use_ue8m0 = get_ue8m0_usage(kernel_type)
        disable_ue8m0_cast = not use_ue8m0

        # Test correctness
        for i in range(10):
            a, b, masked_m, d, ref_d = generate_m_grouped_masked(
                num_groups,
                max_m,
                expected_m_per_group,
                n,
                k,
                use_ue8m0=use_ue8m0,
            )
            deep_gemm.m_grouped_fp8_gemm_nt_masked(
                a,
                b,
                d,
                masked_m,
                expected_m_per_group,
                disable_ue8m0_cast=disable_ue8m0_cast,
            )
            for j in range(num_groups):
                if masked_m[j].item() == 0:
                    continue
                diff = calc_diff(d[j, : masked_m[j].item()], ref_d[j, : masked_m[j].item()])
                assert (
                    diff < 0.001
                ), f"{max_m=}, {n=}, {k=}, {j=}, masked_m={masked_m[j]}, {kernel_opt}, {num_groups=}, {diff:.5f}"
    print()


def test_k_grouped_gemm_contiguous() -> None:
    print("Testing k-grouped contiguous GEMM:")

    k_grouped_fp8_gemm_contiguous = (
        deep_gemm.k_grouped_fp8_gemm_nt_contiguous
        if get_arch_major() == 9
        else deep_gemm.k_grouped_fp8_gemm_tn_contiguous
    )
    for (
        num_groups,
        m,
        n,
        major_a,
        major_b,
        ks,
        expected_k_per_group,
    ) in enumerate_k_grouped_contiguous(torch.float8_e4m3fn):
        use_ue8m0 = get_ue8m0_usage(KernelType.Kernel1D1D)

        for test_empty_groups in (False, True):
            new_ks = copy.deepcopy(ks)
            if test_empty_groups and len(ks) > 1:
                new_ks[random.randint(0, num_groups - 1)] = 0
            k, a, b, c, d, ref_d = generate_k_grouped_contiguous(
                num_groups, m, n, major_a, major_b, new_ks, use_ue8m0=use_ue8m0
            )
            new_ks_tensor = torch.tensor(new_ks, dtype=torch.int, device="cuda")
            k_grouped_fp8_gemm_contiguous(a, b, d, new_ks, new_ks_tensor, c)

            diff = calc_diff(d, ref_d)
            assert diff < 0.001, f"{m=}, {n=}, {k=}, {ks=}, {diff:.5f}"

    print()


if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)

    print("Running DeepGEMM bf16 tests")

    test_gemm()
    test_m_grouped_gemm_contiguous()
    test_m_grouped_gemm_masked()
    test_k_grouped_gemm_contiguous()
