# mistral3_pixtral 图文 CINN 复现

用于框架测试的随机 tiny、FP32、关闭 TF32、单卡 batch1；20张训练/4张留出真实图片，类别 ID 数字 token 任务，不是原模型 caption/OCR/安全质量评测。模型结构和预处理固定在 run_model.py，运行生成 config/；不更换成完整模型。视觉 transformer 层和文本 decoder 层编译，外层 shape 处理/projector/图文组装动态，非全图。图像24×24。

## 环境与资产

本地历史环境：V100S 32GB、Python3.10.20、Paddle3.5.0.dev20260722，commit 08de6be508389d5277b02e78fbc8742f1ed78160，编译 CUDA12.6，numpy1.26.4、Pillow12.3.0。安装相应 Paddle GPU 包及 PaddleFormers requirements。无 AMP、无分布式。


权重不随代码仓库提交，按下方链接下载后在 CHECKPOINT_DIR 中放 initial.pdparams 和 inference.pdparams，分别为历史随机初始权重和动态图10000步最终权重；两端训推各共用对应权重。asset_manifest.json 记录大小/SHA256，不匹配即失败。data/images24.npz 是固定真实 ImageNet 子集，来源列表见 images_manifest.json；仅用于此框架实验，使用须遵守原数据集使用条款。

## 运行

```bash
export CINN_RESOURCE_DIR=/path/to/mistral3_pixtral-resources
export CHECKPOINT_DIR=/path/to/local-weights
export PYTHON_BIN=/path/to/python
bash tests/cinn/mistral3_pixtral/run_model.sh
bash tests/cinn/mistral3_pixtral/run_model_cinn.sh
MODE=dynamic STEPS=10000 RESULTS_DIR=results/full bash tests/cinn/mistral3_pixtral/train.sh
MODE=cinn STEPS=10000 RESULTS_DIR=results/full bash tests/cinn/mistral3_pixtral/train.sh
MODE=dynamic RESULTS_DIR=results/infer1 bash tests/cinn/mistral3_pixtral/infer.sh
MODE=cinn RESULTS_DIR=results/infer1 bash tests/cinn/mistral3_pixtral/infer.sh
```

两个快速入口分别用独立进程执行20步训练和固定前向推理。AdamW 学习率2e-5，其他参数按原脚本，不是零学习率；完整训练目标10000步。推理预热10/计时30，前后 GPU 同步，排除数据预处理/H2D 及首次编译，不是 generate。GPU_ID 默认0。重复运行指定新 RESULTS_DIR，拒绝覆盖。

## 验证边界

CINN 显式 to_static，日志缺少 Compiling subgraph with CINN backend 则入口失败。训练 loss.jsonl 逐步记录，validation.jsonl 记录留出结果，gradients.npz 保存首步参数梯度；推理保存 logits.npy 及逐次计时 inference.json。原始子集拟合与泛化效果分开，不因短测跑通就宣称收敛。权重、日志和生成结果不提交 Git。

本目录是历史已完成长跑脚本的可移植入口；交付回归只做20步及独立推理，不重新宣称执行全部长跑。吞吐提升=(动态 ms/CINNms-1)×100%，延迟下降=(1-CINNms/动态 ms)×100%。性能只代表当前 tiny 及编译范围。

2026-10-07 本地交付回归通过：两端各20步，loss 最大绝对差2.38e-7；共同权重独立推理 logits 最大差1.67e-6；44项首步梯度均保留，最大绝对差2.01e-7。本次仅为短程入口验证。

## Official repository entry

2026-11-06 local entry regression on model source 2b3802f: both modes completed 20 updates and independent inference. Loss max difference 2.38e-7; 44 gradients retained, gradient max difference 2.01e-7; same-checkpoint logits max difference 1.67e-6. This does not replace historical convergence evidence.

Import PaddleFormers from this checkout. External resources are not included: set CINN_RESOURCE_DIR to the bundle containing data/ listed in asset_manifest.json, and CHECKPOINT_DIR to the matching weights. This entry defines the tiny model configuration in run_model.py; it does not load external config or tokenizer files. Download checkpoints using the links below; rebuild the image data from licensed archives using the instructions below. Respect the source dataset license. Historical results above are not a new long-run acceptance of this checkout.

## Rebuild the exact image test data

The image subset comes from ImageNet/ILSVRC2012. Obtain authorized access under the
[ImageNet terms](https://www.image-net.org/download.php); this PR does not redistribute
image pixels. Checkpoints are public, and the sample filenames are included in this repository.
No private contributor bundle is required if you have the matching source archives.

The preparation script uses the original 1000-class labels and RGB short-edge-224 bilinear
resize plus center crop. It reconstructs the historical NPZ serialization and checks the
**original** size/SHA256 before saving; it never changes the reference hash or silently
substitutes random images. Tested with NumPy 1.26.4, Pillow 12.3.0 and SciPy 1.15.3.

```bash
# Use a new output directory. Obtain these three archives through authorized access.
export CINN_RESOURCE_DIR=/path/to/new-mistral3_pixtral-resources
python tests/cinn/mistral3_pixtral/prepare_data.py \\
  --train-archive /path/to/imagenet100.zip \\
  --val-archive /path/to/ILSVRC2012_img_val.tar \\
  --devkit /path/to/ILSVRC2012_devkit_t12.tar.gz \\
  --selection tests/cinn/mistral3_pixtral/image_selection.json \\
  --asset-manifest tests/cinn/mistral3_pixtral/asset_manifest.json \\
  --layout multimodal --output "$CINN_RESOURCE_DIR"
# CHECKPOINT_DIR contains the public initial.pdparams and inference.pdparams.
python tests/cinn/mistral3_pixtral/verify_assets.py "$CHECKPOINT_DIR"
```

Archive requirements: the training ZIP contains the `imagenet100/<WNID>/<filename>`
paths listed in the selection JSON; validation uses the original ILSVRC2012 filenames.
Missing or different source images fail explicitly. This preparation needs CPU only and
is not a new convergence/performance evaluation.

## CINN checkpoint downloads

Model library: [PaddleClas/Mistral-small-3.1](https://aistudio.baidu.com/modelsdetail/53385/space).

- [initial.pdparams](https://git.aistudio.baidu.com/api/v1/repos/PaddleClas/Mistral-small-3.1/media/cinn%2Fmistral3_pixtral%2Fcheckpoint%2Finitial.pdparams): common training initialization for dynamic and CINN runs.
- [inference.pdparams](https://git.aistudio.baidu.com/api/v1/repos/PaddleClas/Mistral-small-3.1/media/cinn%2Fmistral3_pixtral%2Fcheckpoint%2Finference.pdparams): the same checkpoint must be used for both independent inference runs.
- [checkpoint_manifest.json](https://git.aistudio.baidu.com/api/v1/repos/PaddleClas/Mistral-small-3.1/media/cinn%2Fmistral3_pixtral%2Fcheckpoint_manifest.json): file sizes, SHA256 and test scope.

Save the two weight files under `CHECKPOINT_DIR`. The existing `asset_manifest.json` pins their hashes; do not substitute other weights from the model library. These are framework-test checkpoints, not a new task-quality evaluation. The image data are separate and are not supplied by these weight downloads.
