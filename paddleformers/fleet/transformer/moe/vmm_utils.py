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

"""VMM (Virtual Memory Management) utility functions for auto subbatch."""

import paddle
from paddle.device.cuda.memory_analyzer import GB, MemoryAnalysisTool

import paddlefleet_ops


def vmm_free_and_growable_block_info() -> list[tuple[int, int]]:
    """
    获取当前堆中所有 free block 的信息，堆顶可增长的空间也被视为一个 free block。
    返回列表按照 block size 从小到大排序。

    堆大小的上限由 FLAGS_max_reserved_threshold_in_gb 配置。
    """
    all_heaps = MemoryAnalysisTool.vmm_all_block_info()
    assert all_heaps, (
        "vmm_all_block_info() returned empty, is FLAGS_use_virtual_memory_auto_growth=True?"
    )
    all_blocks = all_heaps[-1]  # sorted by addr
    (max_reserved,) = paddle.framework.get_flags(
        "FLAGS_max_reserved_threshold_in_gb"
    ).values()
    max_reserved = max_reserved * GB

    free_blocks = [(size, addr) for size, addr, free in all_blocks if free]

    # 堆顶可增长空间即是: 配置的reserved上限 - 当前已reserved的总量
    growable = max_reserved - paddle.cuda.memory_reserved()

    if not all_blocks:
        return [(growable, 0)] if growable > 0 else []

    heap_top = all_blocks[-1][1] + all_blocks[-1][0]
    heap_limit = heap_top + growable

    # 如果最后一个 block 是空闲的，将其与 growable 合并；否则将 growable 单独作为一个 block
    if growable > 0:
        if all_blocks[-1][2]:
            size, addr = free_blocks[-1]
            free_blocks[-1] = (size + growable, addr)
        else:
            size, addr, _ = all_blocks[-1]
            free_blocks.append((growable, size + addr))

    # 移除超过堆大小限制的 block，即使能分配也不使用
    while free_blocks:
        size, addr = free_blocks[-1]
        if addr >= heap_limit:
            free_blocks.pop()
            continue
        if addr + size > heap_limit:
            free_blocks[-1] = (heap_limit - addr, addr)
        break

    free_blocks.sort()
    return free_blocks


def find_max_concurrent_subbatch_size(
    feature_sizes: list[int],
    upper: int | None = None,
) -> int:
    """
    找到最大的 subbatch_size，使得对于所有 feature_sizes[i] * subbatch_size 的 Tensor 都能同时分配。
    如果无法分配，返回 0。

    如果指定了 upper，则搜索 subbatch_size 时搜索上限不超过 upper。

    注：这个函数是启发式搜索，总是优先把大的 tensor 分配到大的碎片，虽然这样可能得不到理论最优 subbatch_size，
    但 VMM 分配器也是这样的策略，如果本函数给出一个过于极限的大小，可能导致 VMM 分配不出来。
    """
    feature_sizes = sorted(feature_sizes, reverse=True)  # largest first
    if not feature_sizes or feature_sizes[0] == 0:
        return 1

    free_blocks = vmm_free_and_growable_block_info()  # smallest first
    if not free_blocks:
        return 0

    # 如果只有1个tensor，只要检查最大的碎片
    if len(feature_sizes) == 1:
        max_subbatch_size = free_blocks[-1][0] // feature_sizes[0]
        return max_subbatch_size

    # 如果有2个tensor，只要检查最大的1或2个碎片
    if len(feature_sizes) == 2:
        # 如果都放在同一个碎片里
        max_subbatch_size = free_blocks[-1][0] // sum(feature_sizes)

        # 如果分别放在两个碎片里
        if len(free_blocks) > 1:
            subbatch_size = min(
                free_blocks[-1][0] // feature_sizes[0],
                free_blocks[-2][0] // feature_sizes[1],
            )
            max_subbatch_size = max(max_subbatch_size, subbatch_size)

        return max_subbatch_size

    def can_pack(subbatch_size):
        i = 0  # 下一个要分配的 tensor

        for size, _ in reversed(free_blocks):
            # 如果当前碎片连一个 tensor 都分配不了，那后面的碎片更分配不了，肯定失败
            if size < feature_sizes[i] * subbatch_size:
                return False

            # 在当前碎片中分配尽可能多的 tensor
            consumed = 0
            while consumed + feature_sizes[i] * subbatch_size <= size:
                consumed += feature_sizes[i] * subbatch_size
                i += 1
                if i == len(feature_sizes):
                    return True

        # 碎片用完了 tensor 还没分配完，也是失败
        return False

    # 有3个或以上tensor，使用二分搜索
    left = 0
    right = free_blocks[-1][0] // feature_sizes[0] if upper is None else upper

    while left < right:
        mid = (left + right + 1) // 2
        if can_pack(mid):
            left = mid
        else:
            right = mid - 1

    return left


def find_max_sequence_subbatch_size(feature_size: int, length: int = 1) -> int:
    """
    找到最大的 subbatch_size，使得可以将 Tensor [length, feature_size] 在 length 维
    按照 subbatch_size 切分后能够分配。如果无法分配，返回 0。

    如果不指定 length，相当于分析大小为 feature_size 的 Tensor 能否分配。
    """
    free_blocks = vmm_free_and_growable_block_info()  # smallest first

    def can_pack(subbatch_size):
        num_subbatches = (length + subbatch_size - 1) // subbatch_size

        for size, _ in reversed(free_blocks):
            # 如果当前碎片连一个 subbatch 都分配不了，那后面的碎片更分配不了，肯定失败
            if size < feature_size * subbatch_size:
                return False

            # 在当前碎片中分配尽可能多的 subbatch
            num_subbatches -= size // (feature_size * subbatch_size)
            if num_subbatches <= 0:
                return True

        # 碎片用完了 subbatch 还没分配完，也是失败
        return False

    # 使用二分搜索
    left, right = 0, length

    while left < right:
        mid = (left + right + 1) // 2
        if can_pack(mid):
            left = mid
        else:
            right = mid - 1

    return left


def tokens_zip_unique_add_with_subbatch(
    zipped, unzipped, index_unzipped, zipped_rows, subbatch_rows=None
):
    """
    tokens_zip_unique_add_with_subbatch
    """
    if subbatch_rows is None or subbatch_rows <= 0 or zipped_rows <= 0:
        return paddlefleet_ops.tokens_zip_unique_add(
            zipped, unzipped, index_unzipped, zipped_rows
        )
    else:
        if isinstance(zipped, paddle.Tensor):
            num_split = (zipped_rows + subbatch_rows - 1) // subbatch_rows
            remainder = zipped_rows % subbatch_rows
            if remainder == 0:
                rows = [subbatch_rows] * num_split
            else:
                rows = [subbatch_rows] * (num_split - 1) + [remainder]

            if zipped.shape[0] == 0:
                dtype = zipped.dtype
                hidden_size = zipped.shape[1]
                zipped = [
                    paddle.zeros([r, hidden_size], dtype=dtype) for r in rows
                ]
            else:
                zipped = paddle.split(zipped, rows, axis=0)
        return paddlefleet_ops.tokens_zip_unique_add_subbatch(
            zipped, unzipped, index_unzipped, zipped_rows, subbatch_rows
        )


def merge_subbatch_cast(x, dtype):
    """
    将 zip_unzip_fusion=False 时的 float32 分块累加器合并为连续 tensor 并 cast 到目标 dtype。

    zip_unzip_fusion=False 时，显存不足以分配完整 [S, H] 的输出 buffer，
    因此用 list[Tensor] 分块累加（float32 保精度）。所有专家算完后调用此函数：
      [chunk0_f32, chunk1_f32, ...] → concat + cast → result_bf16 [S, H]

    如果 x 已经是单个 Tensor（zip_unzip_fusion=True 或只有一个分块），直接 cast。
    """
    if isinstance(x, (list, tuple)):
        if len(x) == 1:
            x = x[0]
            return x.cast(dtype) if x.dtype != dtype else x
        else:
            return paddlefleet_ops.merge_subbatch_cast(x, dtype)
    else:
        return x.cast(dtype) if x.dtype != dtype else x
