# FP8 Tests / FP8 模块测试

Unit tests for PaddleFleet FP8 quantization, linear layers, and related utilities.
PaddleFleet FP8 量化、线性层及相关工具的单元测试。

## Test Files

| File | Description / 描述 |
|------|-------------------|
| `test_ai_fp8_module_structure.py` | Tests for fp8 module structure via direct source loading / 测试 FP8 模块结构 |
| `test_ai_fp8_utils.py` | Tests for is_fp8_tensor function / 测试 FP8 张量判断函数 |
| `test_ai_linear.py` | Tests for FP8Linear class and _FP8Gemm static forward method / 测试 FP8 线性层与 FP8 GEMM 前向方法 |
| `test_ai_quantization.py` | Tests for get_quant_func with blockwise recipe / 测试 blockwise 量化函数获取 |
| `test_ai_quantization_extra.py` | Tests for get_quant_func blockwise recipe quant methods / 测试 blockwise 量化方法与输入转置 |
| `test_ai_quantization_extra2.py` | Additional tests for get_quant_func partial objects / 测试量化函数的部分对象与量化方法 |
| `test_ue8m0.py` | Tests for use_ue8m0 code paths in fused_stack_quant and MoELayer fp8 / 测试 UE8M0 量化路径 |
