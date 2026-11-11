# Llama4 text tiny 动态图/CINN 复现

## 范围

这是框架测试用随机 tiny，不是完整 Llama4 或原始预训练权重缩层。152992 参数、2 层、hidden32、4 个本地专家、FP32、batch1、seq64、eager attention、TF32 关闭。训练使用 GSM8K 固定20条训练/4条留出样本；目标是训练子集拟合，不是回答质量。Llama Guard4 不包含在本目录。

本目录保留历史实验计算路径。CINN 编译整个文本 forward wrapper；loss、optimizer 和数据循环在 wrapper 外执行，反向穿过已编译 forward。唯一配置适配是用 `not_to_static` 包裹实例的 `config.get`，避免 AST 追踪配置对象；没有修改 PaddleFormers 源文件。CINN 生效以日志中的 `Compiling subgraph with CINN backend` 为准，启动脚本缺少标志即失败。

## 环境和源码

验证环境：V100S 32GB，Python3.10.20，Paddle3.5.0.dev20260722，Paddle commit `08de6be508389d5277b02e78fbc8742f1ed78160`，编译 CUDA12.6；驱动580.105.08。FP32，无 AMP、无分布式。依赖包括 transformers5.3.0、tokenizers0.22.2、numpy1.26.4、Pillow12.3.0；PaddleFormers 使用源码而不是 pip 安装包。

使用与历史验收一致的提交，不自动切换最新 develop：

```bash
git clone https://github.com/PaddlePaddle/PaddleFormers.git /path/to/PaddleFormers
# Use a checkout containing tests/cinn; run the commands below from its root.
```


## 本地权重（本次不上传）

设置 `CHECKPOINT_DIR` 指向包含以下文件的目录：

- `initial.pdparams`：原随机初始化，训练两端共同起点。
- `inference.pdparams`：原300步动态图训练后的 checkpoint，两端推理共同使用；这也是543.33%复核的权重，不是30000步最终权重。

配置和小型测试 tokenizer 位于 assets；所有文件大小与 SHA256位于 `asset_manifest.json`。脚本不在缺少权重时重新随机初始化，不自动下载、不上传权重。第三方仍需由维护者单独提供这两个文件。

## 入口

两个 Llama 入口委托 `tests/cinn/llama_runner.py` 共享训练、推理和断点恢复逻辑；本入口固定为文本模型，不再包含另一模型的不可达分支。导入入口不会启动 GPU。`tests/cinn/test_llama_runner.py` 覆盖参数校验、计时契约和断点恢复等价性；续训分段也保存首步 logits 和梯度探针。

```bash
cd /path/to/PaddleFormers
export CINN_RESOURCE_DIR=/path/to/llama4-resources
export CHECKPOINT_DIR=/path/to/llama4-local-weights
export PYTHON_BIN=/path/to/python
bash tests/cinn/llama4/run_model.sh
bash tests/cinn/llama4/run_model_cinn.sh
```

每个入口依次启动独立的训练和推理进程。默认训练20步，AdamW lr=2e-5（真实更新，沿用原默认 weight_decay）；默认推理预热50次、计时200次。`GPU_ID` 默认0。重复运行必须指定新的 `RESULTS_DIR`，不会覆盖旧结果。

单任务或完整训练：

```bash
MODE=dynamic STEPS=30000 RESULTS_DIR=results/full bash tests/cinn/llama4/train.sh
MODE=cinn STEPS=30000 RESULTS_DIR=results/full bash tests/cinn/llama4/train.sh
MODE=dynamic RESULTS_DIR=results/repeat1 bash tests/cinn/llama4/infer.sh
MODE=cinn RESULTS_DIR=results/repeat1 bash tests/cinn/llama4/infer.sh
# 从本入口生成的完整训练状态续跑，仍提供同一 initial.pdparams
MODE=dynamic STEPS=30000 RESUME=/path/to/training_state.pdstate \
  RESULTS_DIR=results/resumed bash tests/cinn/llama4/train.sh
```

checkpoint/optimizer/RNG 每500步及结束保存；RESUME 只能加载可信来源文件。20步只是快速验证，不能替代30000步的历史子集拟合证据。数据预处理：question + 换行 + answer，tokenizer 不添加特殊 token，最多65token，构造成64token 输入及下一 token 标签，padding label=-100；固定顺序轮转20条，每100步评估4条留出样本。GSM8K 来源：https://github.com/openai/grade-school-math （MIT）；外部资源包包含24条测试样本，tokenizer 是测试资产，不是官方完整 Llama4 tokenizer。

## 产物与性能定义

- `results/<mode>-train/loss.jsonl`：每步 loss 及耗时；另有 validation.jsonl、完整恢复状态和 final.pdparams。
- `results/<mode>-infer/inference.json`：预热后每次耗时、平均值、中位数；logits.npy 用于数值比较。
- `run.json`：参数哈希、源码/Paddle 版本；inputs.npz 记录实际输入。
- 同级 `.log`：完整运行日志；结果和训练产生的权重默认不入 Git。

推理是同一固定输入的完整 logits 前向，不是自回归生成。不包含数据读取、预处理及 H2D；每次计时前后同步 GPU，排除预热/首次编译。训练首步还包含诊断输出保存，不能计入稳态。吞吐提升=(动态耗时/CINN 耗时-1)×100%；延迟下降=(1-CINN 耗时/动态耗时)×100%，两者不可混用。

历史独占 GPU 三组复测：动态6.593/6.542/6.598ms，CINN1.013/1.025/1.026ms，跨组中位耗时对应吞吐+543.33%、延迟-84.46%，logits 最大差3.58e-6。仅说明该 tiny 固定前向的收益，不推广到完整模型。正式复核建议交替动态/CINN 顺序，运行三组。

本次交付入口只做短程训推及续训接口回归，不宣称重新执行30000步；已有长跑证据与新入口回归分开记录。

2026-10-07 本地入口回归通过：动态/CINN 均20步训练且均完成独立推理；共同初始权重哈希一致，20步 loss 最大差9.54e-7、首步梯度最大差9.54e-7，无非零梯度消失。推理输入/参数哈希与历史实验一致，logits 最大差3.58e-6；本轮推理中位6.719/1.046ms，仅作入口回归，不替换三组正式复核。动态恢复从 step20正确继续执行 step20、21。CINN 训练与推理日志均有编译标志，进程正常退出。原始本地日志未提交；历史三组复核摘要由贡献者单独保留。

## Official repository entry

2026-11-06 local regression on develop f7fa77f: 20 training steps and independent inference passed in both modes. Loss max difference 9.54e-7; 27 gradients retained, gradient max difference 9.54e-7; same-checkpoint logits max difference 3.58e-6. This is an entry regression, not a new convergence run.

Import PaddleFormers from this checkout. External resources are not included: set CINN_RESOURCE_DIR to the bundle containing assets/ and data/ listed in asset_manifest.json, and CHECKPOINT_DIR to the matching weights. Obtain the bundles from the contributor; public download links will be added later. Respect the source dataset license. Historical results above are not a new long-run acceptance of this checkout.
