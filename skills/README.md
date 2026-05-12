# README

目录中是使用 PaddleFormers 复用模型的 skills，可直接加载到 Claude Code 中使用，或可作为参考信息。

## 加载方法

在 PaddleFormers 根目录中建立 claude 目录`.claude/skills`，将对应的skill复制到`.claude/skills`中，在根目录中启动 Claude Code，即可加载。

使用时需要将目标模型的权重文件下载到根目录，使用模型名命名：如：

```bash
qwen3-model/
├── config.json
├── generation_config.json
├── model.safetensors
├── tokenizer.json
├── tokenizer_config.json
├── vocab.json
├── merges.txt
├── special_tokens_map.json
├── added_tokens.json
├── chat_template.jinja
└── tokenizer.model
```



， HuggingFace 的 transformers 代码库下载到根目录，或将对应的组网（`modeling_qwen3.py`）、配置（`configuration_qwen3.py`）等实现，放入权重目录下，运行 skills 即可：

```bash
/hf-to-paddle-convert qwen3
```

## 各 skill 功能

| skill 名字                     | 功能                                                         |
| ------------------------------ | ------------------------------------------------------------ |
| `/hf-to-paddle-code-convert`   | 组网转换，将 HuggingFace 组网转换为 PaddleFormers 组网，并实现 `auto` 注册，跑通组网 |
| `/hf-to-paddle-weight-aligner` | 权重对齐，测试 PaddleFormers 能否正确加载权重。需要运行环境中同时装有 transformers 和PaddleFormers，会对齐二者加载权重是否一致 |
| `/hf-to-paddle-logits-aligner` | 前向推理对齐，测试 PaddleFormers 和 transformers 加载权重后，使用同一输入得到的 logits 是否一致，如不一致，找到原因 |

