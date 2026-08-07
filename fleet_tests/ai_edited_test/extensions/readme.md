# Extensions Tests / 扩展模块测试

Unit tests for PaddleFleet custom CUDA extensions, flashmask, triton operators, and index utilities.
PaddleFleet 自定义 CUDA 扩展、FlashMask、Triton 算子和索引工具的单元测试。

## Test Files

| File | Description / 描述 |
|------|-------------------|
| `test_ai_block_mask_bitonic.py` | Tests for bitonic_argsort_device and _compare_and_swap triton kernels / 测试 bitonic 排序 Triton 核函数 |
| `test_ai_block_mask_utils.py` | Tests for find_blocks_topp function and triton kernel existence / 测试 top-p block 查找函数 |
| `test_ai_block_mask_utils_extra.py` | Tests for find_blocks_topp reshape, shape handling, check_fully_masked / 测试 top-p block 查找的形状处理与全 mask 检查 |
| `test_ai_block_mask_utils_extra2.py` | Tests for block_mask_utils module structure and _extract_raw_ptrs / 测试 block_mask_utils 模块结构与原始指针提取 |
| `test_ai_block_mask_utils_extra3.py` | Tests for block_mask_utils triton kernel definitions / 测试 block_mask_utils Triton 核函数定义 |
| `test_ai_block_mask_utils_extra4.py` | Tests for prepare_maxmin and scan_maxmin_chunked kernel / 测试最大最小值准备与分块扫描核函数 |
| `test_ai_flashmask_ext_structure.py` | Tests for flashmask extensions module structure / 测试 FlashMask 扩展模块结构 |
| `test_ai_flashmask_kernels.py` | Tests for check_dense_contains_partial_stride and gemm_fuse_softmax_causal / 测试稠密包含部分 stride 检查与 softmax 融合核函数 |
| `test_ai_index_utils.py` | Tests for prepare_maxmin with various chunk sizes and dtypes / 测试不同分块大小和数据类型的最大最小值准备 |
| `test_ai_index_utils_extra.py` | Tests for prepare_maxmin output shapes and chunk divisibility / 测试 prepare_maxmin 输出形状与分块整除性 |
| `test_ai_index_utils_extra2.py` | Tests for index_utils module structure and scan_maxmin_chunked / 测试索引工具模块结构与分块扫描 |
| `test_ai_index_utils_extra3.py` | Tests for find_blocks_topp in index_utils / 测试索引工具中的 top-p block 查找 |
| `test_ai_ops_extra.py` | Tests for _extensions.ops module functions existence / 测试扩展算子模块函数存在性 |
| `test_ai_ops_extra_2.py` | Tests for _extensions.ops function signatures / 测试扩展算子函数签名 |
| `test_ai_rr_attn_estimate_extra.py` | Tests for _require helper, RawPtrs dataclass, and constants / 测试辅助函数、原始指针数据类与常量 |
| `test_ai_rr_attn_estimate_extra2.py` | Tests for _prepare_stride_maxmin_ptrs with various modes / 测试不同模式下的 stride 最大最小指针准备 |
| `test_ai_rr_attn_estimate_triton_op.py` | Tests for _require, _extract_raw_ptrs, RawPtrs, StrideMaxMinPtrs, rr_attn_estimate / 测试旋转率注意力估算 Triton 算子 |
