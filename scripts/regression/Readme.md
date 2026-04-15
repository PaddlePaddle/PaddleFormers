# PaddleFormers CI/CE

## 支持的模型列表

| 模型名称 | Repo ID | 模型类型 | Template |
|---------|---------|---------|----------|
| qwen3moe | PaddleFormers/tiny-random-qwen3moev2 | text | qwen3 |
| qwen2_moe | PaddleFormers/tiny-random-qwen2moev2 | text | qwen |
| qwen3 | PaddleFormers/tiny-random-qwen3v2 | text | qwen3 |
| glm_moe | PaddleFormers/tiny-random-glm4moe-bf16 | text | glm4_moe |
| qwen2 | PaddleFormers/tiny-random-qwen2v2 | text | qwen |
| llama | PaddleFormers/tiny-random-llama3 | text | llama3 |
| qwen3_next | PaddleFormers/tiny-random-qwen3next | text | qwen3 |
| phi3 | PaddleFormers/tiny-random-phi4-bf16 | text | phi4 |
| gemma3_text | PaddleFormers/tiny-random-gemma3 | text | gemma |
| deepseek_v3 | PaddleFormers/tiny-random-deepseek-v3 | text | deepseek3 |
| qwen3_vl | PaddleFormers/tiny-random-qwen3vlv2 | vl | qwen3_vl |
| qwen3_vl_moe | PaddleFormers/tiny-random-qwen3vlmoev2 | vl | qwen3_vl |
| qwen2_5_vl | PaddleFormers/tiny-random-qwen25vlv2 | vl | qwen2_vl |
| paddleocr_vl | PaddleFormers/tiny-random-paddleocr-vl-bf16 | vl | ernie_vl_nothink |

## 新增模型

### 模型存储路径

模型提前上传到 Aistudio。

在 config.yaml 中新增模型配置即可,可复制其他模型配置，修改 repo_id 等参数，如下
```bash
repo_id: PaddleFormers/tiny-random-qwen3moev2
    model_type: text # text纯文模型，vl 多模模型
    cli_args:
        template: qwen3
        save_checkpoint_format: flex_checkpoint
        load_checkpoint_format: flex_checkpoint
    base_loss:
        dpo_full_loss: 0.69314718
        dpo_full_resume_loss: 0.69314718
    base_result:
        pt_full_excepted_result:
        - [94529, 130950, 94529, 138785, 11615, 90320, 84803, 138785, 791, 104475]
    ...
```

本地自测：

模型存储到 `PaddleFormers/` 当前路径，或者设置环境变量指定路径：

```bash
export PF_HOME="/xx/repo_id"
```

### 运行测试

```bash
# 运行单个模型测试
python -m pytest -s -v --models=qwen2_moe scripts/regression/test_models.py

# 运行多个模型测试
python -m pytest -s -v --models=qwen3,glm_moe scripts/regression/test_models.py

# 运行所有模型测试
python -m pytest -s -v --models=all scripts/regression/test_models.py
```

### 更新 Baseline

```bash
# 更新单个模型的 baseline
python -m pytest -s -v --models=qwen3vl scripts/regression/test_models.py::TestTrain::test_full_tp_vl --update-baseline=glm_moe

# 更新多个模型的 baseline
python -m pytest -s -v --models=deepseek_v3,qwen3_next scripts/regression/test_models.py --update-baseline=deepseek_v3,qwen3_next

# 更新所有模型的 baseline
python -m pytest -s -v --models=all scripts/regression/test_models.py --update-baseline=all

# 在PR 中更新baseline,描述中添加
[update-baseline: llama]

```
### CI 触发规则

```bash
1、自动回归对应模型

修改PaddleFormers/paddleformers/transformers/*.py
修改PaddleFormers/tests/transformers/*.py

2、回归glm_moe模型

只改动了.py文件，没有捕获到修改模型文件
```
