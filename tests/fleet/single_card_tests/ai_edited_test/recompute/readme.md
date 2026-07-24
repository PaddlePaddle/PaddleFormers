# Recompute Tests / 重计算模块测试

Unit tests for PaddleFleet refined recompute module including flash attention, queue check, and flash mask. / PaddleFleet 精细化重计算模块的单元测试，包括 Flash Attention、队列检查和 FlashMask。

## Test Files

| File | Description / 描述 |
|------|-------------------|
| `test_ai_flash_attn.py` | Tests for flashattn_auto_cast function / 测试 FlashAttention 自动类型转换 |
| `test_ai_flash_attn_extra.py` | Tests for _get_fa_version function / 测试 FA 版本获取函数 |
| `test_ai_flash_attn_extra2.py` | Tests for _get_fa_version function / FA 版本获取函数额外测试 |
| `test_ai_flash_attn_extra3.py` | Tests for FlashMaskCpAttention query sequence length validation / 测试 FlashMask CP 注意力序列长度校验 |
| `test_ai_flash_attn_extra4.py` | Tests for flashattn_auto_cast function / FlashAttention 自动类型转换额外测试 |
| `test_ai_flash_attn_extra5.py` | Tests for RefinedRcomputeFlashAttention initialization / 测试精细化重计算 FlashAttention 初始化 |
| `test_ai_flashattn_auto_cast.py` | Tests for _get_fa_version with XPU device / XPU 设备下 FA 版本获取测试 |
| `test_ai_queue_check.py` | Tests for RefinedRcomputeQueue class / 测试精细化重计算队列类 |
| `test_ai_queue_check_extra.py` | Tests for RefinedRcomputeQueue initialization / 重计算队列初始化额外测试 |
| `test_ai_queue_check_extra2.py` | Tests for RefinedRcomputeQueue in queue_check module / queue_check 模块中的重计算队列测试 |
| `test_ai_refined_recompute_flash_mask.py` | Tests for RefinedRcomputeFlashMaskAttention initialization / 测试精细化重计算 FlashMask 注意力初始化 |
| `test_ai_refined_recompute_flash_mask_cp.py` | Tests for RefinedRcomputeFlashMaskCpAttention initialization / 测试精细化重计算 FlashMask CP 注意力初始化 |
