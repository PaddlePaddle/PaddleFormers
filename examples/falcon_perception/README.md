# Falcon Perception 单卡 Paddle 推理

本实现将本地 Falcon Perception 的前向、解码状态机、AnyUp 和分割后处理迁移为原生 Paddle 动态图。输入一张图片和一句 query，输出归一化的 `xy` / `hw`、COCO RLE、二值掩码和可视化图片。

默认使用 **FP32 + greedy decoding**。推理验收覆盖：单卡前向 logits diff 为 `1e-2` 量级；输出正常，前 10 个生成 token 与 Transformers 完全一致。脚本暂按保守的最大绝对误差 `≤ 0.01` 判断前向。BF16 尚未通过这套验收，保留为实验选项。

## 环境

以下复现使用独立的 `paddle_falcon` 环境与原始 PyTorch 基线 `falcon` 环境。两端必须分进程运行，避免 PaddleFormers 对 Transformers 后端检测的设置影响 PyTorch。

| 项目 | 已验证版本 |
| --- | --- |
| Python | 3.10 |
| Paddle | 3.4.0.post20260907+4d8f64b7049，CUDA 12.6 |
| Paddle 端 Transformers | 5.3.0 |
| Paddle 端 NumPy | 1.26.4 |
| 原始模型基线 | PyTorch 2.11.0+cu128，Transformers 5.4.0 |
| GPU | RTX 3090，物理卡 6 / 7 |

在已安装目标仓库依赖的 Paddle 环境中，额外依赖见 `requirements.txt`。本阶段依赖当前 Paddle 的 Tensor 兼容接口，未验证旧版 Paddle。显式模型类的推理不依赖 PyTorch 或 Triton；框架导入路径可能需要其 PaddleFleet 运行环境。`einops` 是基础依赖；`pycocotools` 仅在导出 COCO RLE 或运行分割对齐时需要，缺失时给出安装提示。

## 推理

在目标 PaddleFormers 仓库根目录执行：

```bash
conda activate paddle_falcon
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
MODEL=/data/Falcon_Perception
IMAGE=/sda/ggbond/falcon-perception-run/sample_cats.png

CUDA_VISIBLE_DEVICES=6 python examples/falcon_perception/infer.py \
  --model "$MODEL" --image "$IMAGE" --query cat \
  --dtype float32 --min-dimension 256 --max-dimension 1024 \
  --output /sda/ggbond/falcon-perception-run/paddle_inference
```

也可将物理卡改为 7。进程内设备编号为 `gpu:0`。默认配置即上面的 FP32、256 / 1024 和 `--max-new-tokens 64`。输出包括：

- `predictions.json`：每个实例的归一化中心坐标、尺寸与 RLE。
- `mask_000.png` 等：恢复到输入原图尺寸的二值掩码。
- `overlay.png`：掩码、框和实例编号。
- `manifest.json`：版本、参数、加载统计、结果和耗时。

使用新的输出目录保存每次实验。加 `--trace` 会导出完整 prefill logits、实际生成 token、逐步最后位置的 logits、28 层 hidden states、预处理输入、AnyUp 特征采样和原始 mask logits，文件较大，仅在对齐时开启。计时包含首次执行和可选张量导出，不代表稳定态性能。

也可显式调用模型类：

```python
import os
os.environ["FLAGS_allow_tf32_cublas"] = "0"
os.environ["FLAGS_allow_tf32_cudnn"] = "0"
import paddle
from PIL import Image
from paddleformers.transformers.falcon_perception import FalconPerceptionForSegmentation

paddle.set_device("gpu:0")
model = FalconPerceptionForSegmentation.from_pretrained(model_path, dtype="float32")
results = model.generate(Image.open(image_path).convert("RGB"), "cat", max_new_tokens=64)
print(results[0])
```

## 权重与实现

`from_pretrained` 直接读取原目录的 `config.json`、tokenizer 和 safetensors，不需要中间权重副本。加载前严格检查全部 key 和 shape：当前权重加载 231 个张量，其中 125 个实际 `nn.Linear` 权重转置；Embedding、卷积、频率及 AnyUp 的自定义 packed 权重保持各自布局。缺失、额外或形状不符的权重会报错。

保留本地原实现的 interleaved FFN、时空 RoPE、图像双向 / 文本因果 / document mask、attention sink、坐标去重、mask NMS 和 RLE 行为。AnyUp 分块收集窗口内 K/V，不创建完整的百万像素 Q×K 注意力矩阵。

当前模型继承 `paddle.nn.Layer`，保留专用的图像分割 `generate(images, queries, ...)` 和严格本地 safetensors 加载接口。`AutoConfig` 支持读取配置；模型未注册到 `AutoModel`、`AutoModelForCausalLM` 或 `AutoModelForConditionalGeneration`。请使用上面的显式模型类。当前不提供 Hub 模型下载或 `save_pretrained` 接口。

当前迁移范围为单卡推理，未提供训练 loss、标签处理、recompute、训练 checkpoint 保存恢复或分布式训练接口。前向沿用原始推理约定：传入 `tokens`、原生 attention mask、KV cache 及所需图像/位置输入，返回 `(logits, final_hidden_state)`。

单次仅支持一张图和一个 query；`compile=False`。解码预算沿用原代码的 128-token 向上取整分配和停止条件，`max_new_tokens` 不是严格逐 token 的截断上限。当前端到端验收覆盖 greedy 路径。

## 复现精度对齐

四猫图由 Transformers 的 COCO 双猫 fixture 水平拼接得到（原图 640×480，拼接后 1280×480）。可准备相同输入：

```bash
curl --fail -L https://raw.githubusercontent.com/huggingface/transformers/main/tests/fixtures/tests_samples/COCO/000000039769.png -o sample_cats.png
python - <<'PY'
from PIL import Image
image = Image.open("sample_cats.png").convert("RGB")
four = Image.new("RGB", (image.width * 2, image.height))
four.paste(image, (0, 0))
four.paste(image, (image.width, 0))
four.save("sample_four_cats.png")
PY
```

在仓库根目录运行（按实际路径设置模型、图片和两个 Python 环境）：

```bash
python examples/falcon_perception/run_alignment.py \
  --torch-python /sda/ggbond/miniconda3/envs/falcon/bin/python \
  --paddle-python /sda/ggbond/miniconda3/envs/paddle_falcon/bin/python \
  --model "$MODEL" \
  --image "$PWD/sample_four_cats.png" --query cat \
  --min-dimension 256 --max-dimension 512 \
  --torch-gpu 7 --paddle-gpu 6 \
  --output /sda/ggbond/falcon-perception-run/alignment/reproduce_four_cats_512
```

脚本分别启动原始 PyTorch 与 Paddle 进程，保存各自日志与 trace，最后运行 `compare_reference.py`。未通过会返回非零退出码。原始 PyTorch 在 RTX 3090 上将主干 FlexAttention 的 tile 设为 32×32，以适配 FP32 共享内存；不改原模型文件。

也可分别使用 `export_torch_reference.py --trace`、`infer.py --trace` 导出，再运行：

```bash
python examples/falcon_perception/compare_reference.py \
  --reference /path/to/torch_trace --actual /path/to/paddle_trace \
  --report /path/to/report.json
```

比较脚本默认仅以两条验收标准及相同配置 / 输入判定 `passed`：

| 验收项 | 脚本检查 |
| --- | --- |
| 单卡前向 logits diff 为 1e-2 量级 | 全部 prompt 位置、全部词表的最大绝对误差 ≤ 0.01；同时报告均值 |
| 模型生成验证 | 输出结构、几何和 RLE 有效；实际生成的前 10 个 token ID 完全相同 |

完整前向保存为 `prefill_logits.npy`。`acceptance.forward.first_20_positions` 单独统计输入序列前 20 个位置的完整词表 logits（序列不足 20 时如实报告实际位置数），并提供逐位置均值及最大值。`mean_abs` / `log_diff` 定义为两端 logits 绝对差的算术均值，误差归约使用 NumPy float64；两端模型计算仍为 FP32。生成 token 取自实际 decode 输入，并写入 `manifest.json` 的 `generated_tokens`，不将最后一次 forward 中未被采样的 argmax 算作生成结果。单个用例不足 10 个 token 时返回 `sufficient_tokens=false`，不能按“前 10 个一致”判为通过；应换用自然产生足够 token 的测试输入。

原先的逐层 hidden states、每步 logits、AnyUp、几何和掩码 IoU 检查保留在 `diagnostics_passed`。默认它们作为额外诊断；传入 `--strict-diagnostics` 时才要求诊断指标也全部通过。报告的 `max_logits_abs` 为每步最后位置的差异，正式完整前向指标见 `acceptance.forward`，两者不可混用。

2026-10-19 正式 FP32 用例采用双猫图水平拼接得到的四猫测试图、query=`cat`、min/max=256/512：两端均输出 4 个实例、自然生成 14 个 token，前 10 个一致，完整前向 logits 最大差异低于 0.01。完整数值、路径和结论见 `validation_results.json` 中的 `acceptance`。

前 10 个实际生成 token ID 为：

```text
[268, 240, 241, 262, 240, 241, 262, 240, 241, 262]
```

历史双猫 / 遥控器 / 无目标 / 竖图回归通过了原诊断检查，但均不足 10 个生成 token，不能据此宣称完成这条生成验收。补查双猫图 query=`cat and remote control`、max=1024 的完整前向最大差异约 `0.036`，在当前保守的 `≤ 0.01` 判定下未通过；其最后位置差异较小不代表完整前向通过。该未通过用例一并记录。

同一模型连续调用 `cat → elephant → cat` 的实例数为 `2 → 0 → 2`，两次 `cat` 的几何与 RLE 完全一致。BF16 旧诊断中 logits 差异可达 `1.0`，因此本阶段仍使用 FP32。以上是数值回归结果，未作数据集标注精度评估。

## 2026-11-28 FP32 推理复核

修复后的 Paddle 代码与原始 PyTorch checkpoint 在两张 RTX 3090 上分别运行，使用同一张四猫图、同一份权重，query=`cat`，min/max dimension=`256/512`，greedy decoding，关闭 TF32。原始 PyTorch 主干使用 FlexAttention，Paddle 使用原生分块 attention；两端未启用 BF16/FP16。逐项核对 token、像素、pixel mask、时空 RoPE 位置和空坐标/尺寸输入一致。

| 指标 | 实测结果 |
| --- | --- |
| 输入序列前 20 个位置 × 全词表 | `[1, 20, 65536]` |
| 前 20 个位置 `log_diff = mean(abs(Paddle - PyTorch))` | `3.6378768621e-5` |
| 前 20 个位置最大绝对误差 | `2.0217895508e-4` |
| 完整 prefill logits | `[1, 400, 65536]` |
| 完整 prefill 平均 / 最大绝对误差 | `4.2741028752e-5` / `0.0034523010` |
| 两端实际生成 token 数 | 均为 `14`，前 `10` 个完全一致 |
| 实例数 | 两端均为 `4` |
| 配对分割掩码 IoU 范围 | `0.9998707653–0.9999826891` |
| 归一化中心 / 尺寸最大差异 | `0` |

前 20 个位置为 prefill 的 `[:, :20, :]`，包含图像起始/注册 token 和图像 patch token，不是前 20 个生成 token。生成一致性从实际 decode 输入读取 ID，比较前 10 个实际被选择的 token，而非最后一次前向的未采样 argmax。首次预填充生成的下一个 token 同样计入该列表。

机器可读结果见 `alignment_results_20261128.json`，包含逐位置误差、环境、输入形状、图像和权重 SHA256；完整 trace 与逐层报告保留在验证机器的 `/sda/ggbond/falcon-pr5026-review-validation-20261128/`。按上面的 `run_alignment.py` 命令重新运行即可生成完整报告。当前结论覆盖该单图、单 query、单分辨率用例，不代表任意图像、query、分辨率或 BF16 均已通过。

## 单元测试

```bash
CUDA_VISIBLE_DEVICES=6 python -m pytest -o addopts='' \
  tests/transformers/falcon_perception -q
```

覆盖混合 attention mask 与 decode offset、sink、AnyUp 窗口边界及 QK/V 不同宽度、RoPE、交错 FFN、严格权重映射、异常权重、KV cache 多层位置 / reset、mask NMS 和 RLE，以及完整前向检查、真实生成 token 提取、短序列拒绝和验收 / 诊断判定。

## 移除训练扩展后的回归（2026-10-19）

保留的推理、加载和 Auto 测试共 29 项通过，无跳过。GPU 7 通过当时的 `AutoModelForCausalLM` 入口加载原始完整权重（该入口现已移除，请使用显式模型类），对 512 尺度四猫样例回归：输出 4 个有效实例，完整前向 logits 最大差异为 `0.0034523010`，前 10 个实际生成 token 与已有 Transformers 参考一致。该结果只对应上述样例与尺度。

测试和精度报告保存在 `/tmp/falcon_remove_training_validation/tests.xml` 与 `/tmp/falcon_remove_training_validation/alignment_report.json`。
