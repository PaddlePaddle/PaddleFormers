# fastvlm 图文 CINN 复现

用于框架测试的随机 tiny、FP32、关闭 TF32、单卡 batch1；20张训练/4张留出真实图片，类别 ID 数字 token 任务，不是原模型 caption/OCR/安全质量评测。模型结构和预处理固定在 run_model.py，运行生成 config/；不更换成完整模型。视觉 SEBlock 使用原生 shape/reshape；视觉塔和文本 decoder 分图编译，图文组装动态。图像64×64。

## 环境与资产

本地历史环境：V100S 32GB、Python3.10.20、Paddle3.5.0.dev20260722，commit 08de6be508389d5277b02e78fbc8742f1ed78160，编译 CUDA12.6，numpy1.26.4、Pillow12.3.0。安装相应 Paddle GPU 包及 PaddleFormers requirements。无 AMP、无分布式。


权重不上传、不下载，CHECKPOINT_DIR 中放 initial.pdparams 和 inference.pdparams，分别为历史随机初始权重和动态图3000步最终权重；两端训推各共用对应权重。asset_manifest.json 记录大小/SHA256，不匹配即失败。data/images24.npz 是固定真实 ImageNet 子集，来源列表见 images_manifest.json；仅用于此框架实验，使用须遵守原数据集使用条款。

## 运行

```bash
export CINN_RESOURCE_DIR=/path/to/fastvlm-resources
export CHECKPOINT_DIR=/path/to/local-weights
export PYTHON_BIN=/path/to/python
bash tests/cinn/fastvlm/run_model.sh
bash tests/cinn/fastvlm/run_model_cinn.sh
MODE=dynamic STEPS=3000 RESULTS_DIR=results/full bash tests/cinn/fastvlm/train.sh
MODE=cinn STEPS=3000 RESULTS_DIR=results/full bash tests/cinn/fastvlm/train.sh
MODE=dynamic RESULTS_DIR=results/infer1 bash tests/cinn/fastvlm/infer.sh
MODE=cinn RESULTS_DIR=results/infer1 bash tests/cinn/fastvlm/infer.sh
```

两个快速入口分别用独立进程执行20步训练和固定前向推理。AdamW 学习率2e-5，其他参数按原脚本，不是零学习率；完整训练目标3000步。推理预热10/计时30，前后 GPU 同步，排除数据预处理/H2D 及首次编译，不是 generate。GPU_ID 默认0。重复运行指定新 RESULTS_DIR，拒绝覆盖。

## 验证边界

CINN 显式 to_static，日志缺少 Compiling subgraph with CINN backend 则入口失败。训练 loss.jsonl 逐步记录，validation.jsonl 记录留出结果，gradients.npz 保存首步参数梯度；推理保存 logits.npy 及逐次计时 inference.json。原始子集拟合与泛化效果分开，不因短测跑通就宣称收敛。权重、日志和生成结果不提交 Git。

本目录是历史已完成长跑脚本的可移植入口；交付回归只做20步及独立推理，不重新宣称执行全部长跑。吞吐提升=(动态 ms/CINNms-1)×100%，延迟下降=(1-CINNms/动态 ms)×100%。性能只代表当前 tiny 及编译范围。

2026-10-07 本地交付回归通过：两端各20步，loss 最大绝对差4.77e-7；共同权重独立推理 logits 最大差1.91e-6；86项首步梯度均保留，最大绝对差4.89e-4。推理单次短测中位耗时动态10.17ms/CINN5.16ms，非多轮稳定性能结论。

## Official repository entry

2026-11-06 local entry regression on model source c636d0d: both modes completed 20 updates and independent inference. Loss max difference 4.77e-7; 86 gradients retained, gradient max difference 4.88e-4; same-checkpoint logits max difference 1.91e-6. This does not replace historical convergence evidence.

Import PaddleFormers from this checkout. External resources are not included: set CINN_RESOURCE_DIR to the bundle containing assets/ and data/ listed in asset_manifest.json, and CHECKPOINT_DIR to the matching weights. Obtain the bundles from the contributor; public download links will be added later. Respect the source dataset license. Historical results above are not a new long-run acceptance of this checkout.
