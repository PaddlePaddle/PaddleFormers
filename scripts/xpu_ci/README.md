# XPU CI 测试框架

这是一个用于验证 PaddleFormers 模型训练 loss 值的自动化测试框架。

## 目录结构

```
xpu_ci/
├── base_value/              # 存放基准 loss 值的 JSON 文件
│   └── ernie_21b_sft_loss.json
├── config/                  # 存放训练配置文件
│   └── ernie_21b_sft.yaml
├── utils/                   # 公共工具函数
│   ├── __init__.py
│   └── log_analyzer.py      # 日志解析和比较函数
├── conftest.py              # pytest 配置和公共 fixtures
├── test_ernie_21b_sft.py    # 具体测试用例
├── test_example_template.py.template  # 新测试用例模板
└── README.md                # 本文档
```

## 核心改进

与原始版本相比，新框架具有以下优势：

### ✅ 代码复用
- 提取了 `TrainingTestRunner` 基类，封装通用逻辑
- 新测试用例只需 10 行代码

### ✅ 配置灵活
- 容差值可按测试配置（默认 1e-6）
- 超时时间可自定义（默认 1 小时）
- 支持不同配置文件和基准值

### ✅ 错误处理
- 基准文件不存在时给出友好提示
- 命令超时自动终止并报错
- 详细的失败信息（包含 step、diff 等）

### ✅ pytest 最佳实践
- 使用 fixtures 管理路径和资源
- 自动生成唯一日志文件（避免并行冲突）
- 利用 tmp_path 管理临时文件

### ✅ 易于扩展
- 提供模板文件快速添加新测试
- 统一的目录和命名规范

## 快速开始

### 运行现有测试

```bash
# 运行所有 XPU CI 测试
cd /paddle/sjx_cuda12.6_py310/formers_test/PaddleFormers
pytest tests/xpu_ci/ -v

# 运行特定测试
pytest tests/xpu_ci/test_ernie_21b_sft.py -v

# 显示详细输出（推荐：可以看到执行的命令和实时进度）
pytest tests/xpu_ci/test_ernie_21b_sft.py -v -s
```

### 测试输出示例

运行测试时会看到详细的执行信息：

```
================================================================================
EXECUTING TRAINING COMMAND
================================================================================
Command: paddleformers-cli train tests/xpu_ci/config/ernie_21b_sft.yaml
Working Directory: /paddle/sjx_cuda12.6_py310/formers_test/PaddleFormers
Config File: tests/xpu_ci/config/ernie_21b_sft.yaml
Baseline File: tests/xpu_ci/base_value/ernie_21b_sft_loss.json
Tolerance: 1e-06
Timeout: 3600s
Log File: /tmp/pytest-xxx/test_ernie_21b_sft_training0/test_ernie_21b_sft_training.log
================================================================================

Command finished with return code: 0
✓ Command succeeded. Log saved to: /tmp/pytest-xxx/test_ernie_21b_sft_training0/test_ernie_21b_sft_training.log

✓ Loss validation PASSED - all values within tolerance
```

### 日志文件内容

每次测试都会生成包含完整元数据的日志文件：

```
================================================================================
TRAINING TEST EXECUTION LOG
================================================================================
Executed Command: paddleformers-cli train tests/xpu_ci/config/ernie_21b_sft.yaml
Working Directory: /paddle/sjx_cuda12.6_py310/formers_test/PaddleFormers
Config File: tests/xpu_ci/config/ernie_21b_sft.yaml
Baseline File: tests/xpu_ci/base_value/ernie_21b_sft_loss.json
Tolerance: 1e-06
Timeout: 3600s
Return Code: 0
================================================================================

[训练输出日志内容...]
```

### 添加新测试用例

#### 核心理念：命令直接可见

新框架的设计理念是：**每个测试用例中直接写完整的 shell 命令**，不进行过度抽象。
这样做的好处：
- ✅ 命令一目了然，便于理解和调试
- ✅ 支持任意 shell 命令（不限于 paddleformers-cli）
- ✅ 可以使用环境变量、管道等 shell 特性
- ✅ 每个测试可以完全独立定制

#### 步骤

1. **复制模板文件**
   ```bash
   cd tests/xpu_ci
   cp test_example_template.py.template test_your_test.py
   ```

2. **编写测试函数**
   ```python
   import pytest
   from conftest import run_command_and_validate

   def test_your_model(project_root, base_value_dir, log_file):
       """Test description.

       This test runs the following shell command:
           <写清楚你要执行的完整命令>
       """
       # 直接写你要执行的 shell 命令
       cmd = "paddleformers-cli train tests/xpu_ci/config/your_config.yaml"

       passed, error_msg = run_command_and_validate(
           cmd=cmd,
           baseline_path=base_value_dir / "your_model_loss.json",
           log_file=log_file,
           working_dir=project_root,
           tolerance=1e-6,
           timeout=3600
       )

       if not passed:
           pytest.fail(error_msg)
   ```

3. **生成基准值**

   首先运行一次训练，获取 loss 值：
   ```bash
   # 手动运行你的命令
   cd /paddle/sjx_cuda12.6_py310/formers_test/PaddleFormers
   paddleformers-cli train tests/xpu_ci/config/your_config.yaml | tee output.log

   # 提取 loss 值
   grep "loss:" output.log
   ```

4. **创建基准文件**

   在 `base_value/` 目录创建 `your_model_loss.json`：
   ```json
   {
       "1": 7.12345678,
       "2": 7.08901234,
       ...
   }
   ```

5. **运行测试验证**
   ```bash
   pytest tests/xpu_ci/test_your_test.py -v -s
   ```

#### 示例：不同类型的命令

**示例 1：标准 paddleformers-cli 命令**
```python
def test_llama_training(project_root, base_value_dir, log_file):
    cmd = "paddleformers-cli train tests/xpu_ci/config/llama_7b.yaml"

    passed, error_msg = run_command_and_validate(
        cmd=cmd,
        baseline_path=base_value_dir / "llama_7b_loss.json",
        log_file=log_file,
        working_dir=project_root,
        tolerance=1e-6,
        timeout=3600
    )

    if not passed:
        pytest.fail(error_msg)
```

**示例 2：使用自定义 Python 脚本**
```python
def test_custom_script(project_root, base_value_dir, log_file):
    cmd = "python scripts/train_custom.py --epochs 10 --lr 1e-5"

    passed, error_msg = run_command_and_validate(
        cmd=cmd,
        baseline_path=base_value_dir / "custom_script_loss.json",
        log_file=log_file,
        working_dir=project_root,
        tolerance=1e-5,
        timeout=7200
    )

    if not passed:
        pytest.fail(error_msg)
```

**示例 3：带环境变量的命令**
```python
def test_with_env_vars(project_root, base_value_dir, log_file):
    cmd = "CUDA_VISIBLE_DEVICES=0,1,2,3 XPU_VISIBLE_DEVICES=0,1 paddleformers-cli train config.yaml"

    passed, error_msg = run_command_and_validate(
        cmd=cmd,
        baseline_path=base_value_dir / "multi_device_loss.json",
        log_file=log_file,
        working_dir=project_root,
        tolerance=1e-6,
        timeout=3600
    )

    if not passed:
        pytest.fail(error_msg)
```

**示例 4：复杂的 shell 命令组合**
```python
def test_with_preprocessing(project_root, base_value_dir, log_file):
    cmd = """
    export DATA_DIR=/path/to/data && \
    python preprocess.py --input $DATA_DIR --output /tmp/processed && \
    paddleformers-cli train tests/xpu_ci/config/model.yaml --data /tmp/processed
    """

    passed, error_msg = run_command_and_validate(
        cmd=cmd,
        baseline_path=base_value_dir / "with_preprocessing_loss.json",
        log_file=log_file,
        working_dir=project_root,
        tolerance=1e-6,
        timeout=3600
    )

    if not passed:
        pytest.fail(error_msg)
```

## 参数说明

### TrainingTestRunner 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `config_path` | Path/str | 必需 | 训练配置文件路径 |
| `baseline_path` | Path/str | 必需 | 基准 loss JSON 文件路径 |
| `tolerance` | float | 1e-6 | 允许的绝对误差 |
| `timeout` | int | 3600 | 命令超时时间（秒） |

### 容差建议

- **严格测试**（回归测试）：`1e-6` 或 `1e-7`
- **一般测试**：`1e-5`
- **宽松测试**（大模型、长序列）：`1e-4`

## 工具函数

### 日志解析

`tests/ci_utils/log_analyzer.py` 提供两个核心函数：

```python
from ci_utils.log_analyzer import parse_loss_values, compare_with_baseline

# 从日志中提取 loss 值
losses = parse_loss_values(log_content)
# 返回: {1: 7.11594725, 2: 7.08606052, ...}

# 与基准值比较
passed, details = compare_with_baseline(losses, baseline_path, tolerance=1e-6)
# 返回: (True/False, {step: {current, baseline, diff, passed}})
```

## 常见问题

### Q: 测试失败提示 "No loss values found"
**A:** 检查日志格式是否匹配。当前支持的格式：
```
loss: 7.11594725 learning_rate: 5e-07 global_step: 1 ...
```

如果格式不同，需要修改 `tests/ci_utils/log_analyzer.py` 中的正则表达式。

### Q: 如何调整容差？
**A:** 在创建 runner 时传入 `tolerance` 参数：
```python
runner = training_runner(..., tolerance=1e-5)  # 更宽松
```

### Q: 测试运行时间过长
**A:** 调整 `timeout` 参数或在配置文件中减少训练步数。

### Q: 并行运行测试会冲突吗？
**A:** 不会。每个测试使用独立的临时日志文件（由 pytest 的 `tmp_path` 管理）。

### Q: 如何只验证部分 step 的 loss？
**A:** 在基准 JSON 文件中只保留需要验证的 step。比对时会自动跳过基准文件中不存在的 step。

## 最佳实践

1. **命名规范**
   - 测试文件：`test_<model_name>.py`
   - 配置文件：`<model_name>.yaml`
   - 基准文件：`<model_name>_loss.json`

2. **基准值管理**
   - 使用版本控制跟踪基准值变化
   - 注释说明基准值的生成环境和时间

3. **测试组织**
   - 一个模型可以有多个测试（不同配置、不同场景）
   - 相关测试可以放在同一个文件中

4. **调试技巧**
   - 使用 `-v` 查看详细输出
   - 使用 `-s` 查看 print 输出
   - 检查生成的日志文件（路径会在失败信息中显示）

## 维护和扩展

如果需要支持新的日志格式或比较逻辑：

1. 修改 `tests/ci_utils/log_analyzer.py` 中的解析函数
2. 扩展 `TrainingTestRunner` 类添加新功能
3. 更新本文档

## 示例

参考 `test_ernie_21b_sft.py` 查看完整的工作示例。
