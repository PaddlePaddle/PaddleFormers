# Llama Guard4 图文 tiny 动态图/CINN

本目录是随机初始化的框架测试模型（25997120参数），不是完整 Llama Guard4，也不评价安全分类能力。仅图文，不含音频。FP32、关闭 TF32、batch1、seq64、28×28图像；文本为图像类别 ID 描述，20训练/4留出真实图片。实际网络由当前源码 `tests/transformers/llama4/test_multimodal.py:get_tiny_config` 构建，将文本词表设为202048、image token 设为200092；assets/config.json 来自上游缩层资产，不用它构建本次 tiny，实际配置保存在运行产物 config/中。

## 环境和源码

Python3.10.20、Paddle3.5.0.dev20260722，commit `08de6be508389d5277b02e78fbc8742f1ed78160`，编译 CUDA12.6，V100S 32GB，驱动580.105.08。依赖 transformers5.3.0、tokenizers0.22.2、numpy1.26.4、Pillow12.3.0；按 PaddleFormers requirements 准备其余依赖。单卡无 AMP、无 DP/PP/MP。

```bash
git clone https://github.com/PaddlePaddle/PaddleFormers.git /path/to/PaddleFormers
# Use a checkout containing tests/cinn; run the commands below from its root.
```


## 权重与数据

权重不上传、不自动下载；CHECKPOINT_DIR 应包含 initial.pdparams（共同随机起点）、inference.pdparams（历史动态图300步权重，两端推理共用）。大小和 SHA256见 asset_manifest.json。缺少权重直接报错，不自动创建。恢复状态只加载可信来源文件。

data/images24.npz 与 images_manifest.json 是历史测试的小图像集与来源记录；沿用 ImageNet 子集类别 ID，RGB 通过 PIL resize 到28×28，再/127.5-1。文本 `Image class ID: <id>.`，前插图像 token，不加额外特殊 token，最多65token 构造64token 输入和下一 tokenlabels，padding labels=-100。assets 中的 tokenizer 继承上游 Llama Guard4资产，使用时遵守 Meta Llama4许可证及上游条款：https://huggingface.co/meta-llama/Llama-Guard-4-12B 。不将本仓解释为对上游资产重新授权。

## 运行

```bash
cd /path/to/PaddleFormers
export CINN_RESOURCE_DIR=/path/to/llama_guard4-resources
export CHECKPOINT_DIR=/path/to/local-guard4-weights
export PYTHON_BIN=/path/to/python
bash tests/cinn/llama_guard4/run_model.sh
bash tests/cinn/llama_guard4/run_model_cinn.sh
```

默认快速入口分别执行20步训练及独立进程推理（预热10/计时30），训练是 AdamW lr=2e-5真实参数更新，不是零学习率。GPU_ID 默认0；重复运行需新的 RESULTS_DIR，拒绝覆盖。正式训练/独立推理：

```bash
MODE=dynamic STEPS=10000 RESULTS_DIR=results/full bash tests/cinn/llama_guard4/train.sh
MODE=cinn STEPS=10000 RESULTS_DIR=results/full bash tests/cinn/llama_guard4/train.sh
MODE=dynamic RESULTS_DIR=results/infer1 bash tests/cinn/llama_guard4/infer.sh
MODE=cinn RESULTS_DIR=results/infer1 bash tests/cinn/llama_guard4/infer.sh
MODE=dynamic STEPS=10000 RESUME=/path/to/training_state.pdstate \
  RESULTS_DIR=results/resumed bash tests/cinn/llama_guard4/train.sh
```

每500步及结束保存模型/优化器/RNG；每100步评估4条留出数据。20步只证明入口可用，不能替代已有10000步子集拟合证据。本次交付回归不重新执行完整长跑。

## 结果

results/<mode>-train 下有逐步 loss.jsonl、validation.jsonl、grads.npz、final.pdparams 及 training_state.pdstate；<mode>-infer 下有 logits.npy 和 inference.json（逐次耗时、均值、中位数）；run.json 记录权重参数哈希与版本，inputs.npz 记录输入。结果与权重不入 Git。

推理是固定图文前向，非 generate，计时不含预处理/H2D，每次前后 GPU 同步，排除预热及首次编译。训练首步含诊断保存，不计入稳态。吞吐提升=(dynamic_ms/cinn_ms-1)×100%；延迟下降=(1-cinn_ms/dynamic_ms)×100%。该 tiny 结果不代表完整模型部署性能。

2026-10-07新入口本地回归：动态/CINN 各20步训练及独立推理正常退出，CINN 两任务均有编译标志。20步 loss 最大差1.19e-7；对齐视觉 wrapper 参数名前缀后，63个首步梯度最大差5.25e-6，无非零梯度丢失。推理最大差5.36e-6；动态图输入、权重哈希和 logits 与历史验收一致。本轮前向中位8.750/4.065ms，仅单次入口回归。动态断点 step20恢复到22通过，未重跑10000步。

## Official repository entry

2026-11-06 local regression on develop f7fa77f: 20 training steps and independent inference passed in both modes. Loss max difference 1.19e-7; 63 gradients retained after mapping the CINN vision wrapper prefix back to the original names, gradient max difference 5.25e-6; same-checkpoint logits max difference 5.36e-6. This is an entry regression, not a new convergence run.

Import PaddleFormers from this checkout. External resources are not included: set CINN_RESOURCE_DIR to the bundle containing assets/ and data/ listed in asset_manifest.json, and CHECKPOINT_DIR to the matching weights. Obtain the bundles from the contributor; public download links will be added later. Respect the source dataset license. Historical results above are not a new long-run acceptance of this checkout.
# 权重与配置校验

本入口使用随脚本提交的 tiny_config.json，不再导入模型单测内部的配置函数。该配置固化自历史入口的实际运行配置；已用两份交付 checkpoint 验证 63 项 state_dict 条目逐项完整加载。外部 assets/config.json 是上游缩层配置，不应直接替代本入口的 tiny 配置。

初始加载及续跑都检查 checkpoint 的 key、shape 与 set_state_dict 返回值；不匹配即报错，不允许跳过参数后继续使用随机初始化。续跑只支持本入口生成的状态，不修改 Paddle 的私有反序列化白名单。tests/cinn/test_checkpoint_utils.py 包含缺失 key、多余 key、错 shape 等负例。
