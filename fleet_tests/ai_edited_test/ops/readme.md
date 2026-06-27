# Ops Tests / 算子模块测试

Unit tests for PaddleFleet ops module including cross entropy, MoE topk fusion, RMS norm, sigmoid gate, and triton utils.
PaddleFleet 算子模块的单元测试，包括交叉熵、MoE TopK 融合、RMS 归一化和 Triton 工具。

## Test Files

| File | Description / 描述 |
|------|-------------------|
| `test_ai_cross_entropy_extra3.py` | Tests for cross_entropy kernel attributes and structure / 测试交叉熵核函数属性与结构 |
| `test_ai_fused_linear_ce_backward.py` | Tests for LigerFusedLinearCrossEntropyFunction backward / 测试融合线性交叉熵反向传播 |
| `test_ai_fused_linear_ce_cross_entropy.py` | Tests for liger_cross_entropy_kernel definition / 测试 Liger 交叉熵核函数定义 |
| `test_ai_fused_linear_ce_main.py` | Tests for fused_linear_cross_entropy_forward / 测试融合线性交叉熵前向传播 |
| `test_ai_fused_linear_ce_utils.py` | Tests for element_mul_kernel function / 测试逐元素乘法核函数 |
| `test_ai_moe_topk_fusion.py` | Tests for MoETopkFusion PyLayer class definition / 测试 MoE TopK 融合 PyLayer 定义 |
| `test_ai_moe_topk_fusion_extra.py` | Tests for MoETopkFusion forward method / 测试 MoE TopK 融合前向传播 |
| `test_ai_moe_topk_fusion_extra2.py` | Tests for MoE TopK fusion module structure / 测试 MoE TopK 融合模块结构 |
| `test_ai_ops_utils.py` | Tests for import_custom_ops function / 测试自定义算子导入函数 |
| `test_ai_ops_utils_extra.py` | Tests for is_torch_compat_available function / 测试 Torch 兼容性判断函数 |
| `test_ai_rms_norm_fusion.py` | Tests for RMSNormFusionTriton PyLayer definition / 测试 RMS 归一化融合 Triton PyLayer |
| `test_ai_rms_norm_fusion_extra.py` | Tests for RMSNormFusionTriton forward parameter handling / RMS 归一化融合前向参数处理测试 |
| `test_ai_rms_norm_fusion_extra2.py` | Tests for RMS norm fusion module structure / RMS 归一化融合模块结构测试 |
| `test_ai_sigmoid_gate_fusion.py` | Tests for SigmoidGateFusionTriton PyLayer definition / 测试 Sigmoid 门控融合 Triton PyLayer |
| `test_ai_sigmoid_gate_fusion_extra.py` | Comprehensive tests for sigmoid gate computation / Sigmoid 门控计算综合测试 |
| `test_ai_sigmoid_gate_fusion_extra2.py` | Tests for sigmoid gate fusion module structure / Sigmoid 门控融合模块结构测试 |
| `test_ai_triton_compat.py` | Tests for _is_package_installed function / 测试包安装判断函数 |
| `test_ai_triton_compat_extra.py` | Tests for ops triton_compat module structure / Triton 兼容模块结构测试 |
| `test_ai_triton_utils_extra.py` | Tests for triton_ops/utils.py functions / Triton 工具函数测试 |
