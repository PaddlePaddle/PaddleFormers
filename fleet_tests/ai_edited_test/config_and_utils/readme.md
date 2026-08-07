# Config and Utils Tests / 配置与工具模块测试

Unit tests for PaddleFleet configuration, training arguments, timers, and utility functions.
PaddleFleet 配置、训练参数、计时器和工具函数的单元测试。

## Test Files

| File | Description / 描述 |
|------|-------------------|
| `test_ai_arguments.py` | Tests for parse_args and core_transformer_config_from_args / 测试命令行参数解析与核心 Transformer 配置构建 |
| `test_ai_config_logger.py` | Tests for config_logger module path and enabled state / 测试配置日志器模块路径与启用状态 |
| `test_ai_config_logger_extra.py` | Tests for get_config_logger_path and has_config_logger_enabled / 测试配置日志器路径获取与启用状态查询 |
| `test_ai_context_parallel_utils_extra2.py` | Tests for mark_context_parallel_parameter_disable_scale_grad / 测试上下文并行参数禁用缩放梯度标记 |
| `test_ai_global_vars.py` | Tests for global_vars get_args/set_args and get_timers/set_timers / 测试全局变量的 get/set 访问器 |
| `test_ai_gpt_builders.py` | Tests for gpt_builder and _get_transformer_layer_spec_func / 测试 GPT 构建器与 Transformer 层规格获取 |
| `test_ai_jit.py` | Tests for jit module jit_fuser function / 测试 JIT 编译器的 fuser 函数 |
| `test_ai_package_info.py` | Tests for package_info module metadata fields / 测试包信息模块元数据字段 |
| `test_ai_parallel_state_extra.py` | Tests for expert model parallel state getter/setter / 测试专家模型并行状态的读写函数 |
| `test_ai_spec_utils.py` | Tests for LayerSpec, import_spec_layer, get_layer, and build_spec_layer / 测试层级规格定义、导入与构建 |
| `test_ai_timers.py` | Tests for Timer, RuntimeTimer, and Timers classes / 测试计时器相关类 |
| `test_ai_timers_extra.py` | Tests for _Timer basic operations including start/stop/error handling / 测试基础计时器的启停与错误处理 |
| `test_ai_training_global_vars_extra.py` | Tests for get_args, set_args, and destroy_global_vars / 测试训练全局变量的获取、设置与销毁 |
| `test_ai_training_initialize.py` | Tests for set_logging function with various logging levels / 测试日志设置函数 |
| `test_ai_utils.py` | Tests for WrappedTensor, GlobalMemoryBuffer, ensure_divisibility, divide / 测试张量封装、全局内存缓冲区、整除检查等工具函数 |
| `test_ai_utils_extra.py` | Additional tests for GlobalMemoryBuffer, ensure_divisibility, divide / 工具函数的额外测试 |
| `test_ai_utils_extra2.py` | Tests for ensure_divisibility and divide utility functions / 整除检查与除法工具函数测试 |
| `test_ai_utils_extra3.py` | Additional tests for GlobalMemoryBuffer and make_viewless_tensor / 全局内存缓冲区与无视图张量测试 |
| `test_ai_utils_extra4.py` | Tests for get_batch_on_this_cp_rank and nvtx_decorator / 测试 CP 排序批次获取与 NVTX 装饰器 |
| `test_ai_yaml_arguments.py` | Tests for _flatten_configs function with various dict structures / 测试配置字典扁平化函数 |
| `test_ai_yaml_arguments_extra.py` | Edge case tests for _flatten_configs with mixed types / 混合类型配置扁平化的边界测试 |
| `test_ai_yaml_arguments_extra2.py` | Tests for _flatten_configs with None values and load_yaml / 测试含 None 值的配置扁平化与 YAML 加载 |
| `test_ai_yaml_arguments_extra3.py` | Tests for load_yaml and _flatten_configs integration / YAML 加载与配置扁平化集成测试 |
| `test_ai_yaml_arguments_extra4.py` | Edge case tests for load_yaml with empty, single-value, and list YAML / 空值、单值与列表 YAML 加载的边界测试 |
