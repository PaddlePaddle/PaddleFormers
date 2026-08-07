# MoE Tests / 混合专家模块测试

Unit tests for PaddleFleet Mixture of Experts layer, router, expert, token dispatcher, and related utilities.
PaddleFleet 混合专家层、路由器、专家、Token 调度器及相关工具的单元测试。

## Test Files

| File | Description / 描述 |
|------|-------------------|
| `test_ai_block_attn_res.py` | Unit tests for block_attn_res module / 测试 block attention residual 模块 |
| `test_ai_fused_a2a.py` | Unit tests for fused_a2a module / 测试融合 All-to-All 通信模块 |
| `test_ai_fusion_layer_utils.py` | Unit tests for fusion_layer_utils module / 测试融合层工具模块 |
| `test_ai_moe_expert.py` | Unit tests for moe_expert module / 测试 MoE 专家模块 |
| `test_ai_moe_fp8_utils.py` | Unit tests for fp8_utils module / 测试 MoE FP8 工具模块 |
| `test_ai_moe_layer.py` | Unit tests for moe_layer module / 测试 MoE 层模块 |
| `test_ai_moe_layer_extra.py` | Extra tests for MoELayer expert parallel initialization / MoE 层专家并行初始化额外测试 |
| `test_ai_moe_router.py` | Unit tests for moe_router module / 测试 MoE 路由器模块 |
| `test_ai_moe_utils.py` | Unit tests for moe_utils module / 测试 MoE 工具模块 |
| `test_ai_multi_token_prediction.py` | Unit tests for multi_token_prediction module / 测试多 Token 预测模块 |
| `test_ai_token_dispatcher.py` | Unit tests for token_dispatcher module / 测试 Token 调度器模块 |
| `test_kgroupgemm.py` | Tests for k-grouped gemm code paths including fp8+deep_gemm / 测试 k-grouped GEMM 代码路径 |
| `test_latent_moe.py` | Tests for latent MoE config field defaults and validation / 测试潜在 MoE 配置默认值与校验 |
