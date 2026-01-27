# 1. 任务简介

Visual Grounding（视觉定位）是一种让多模态大模型能够将自然语言描述精确映射到图像具体区域（Bounding Box）的机制，通过文本指令与像素坐标的语义对齐，提升模型对物理世界的感知与交互能力。这种机制使得大模型不再局限于全局的图像描述，而是能够根据指令精准锁定图像中的特定目标。

多模态大模型的 Visual Grounding 能力通常应用于需要高精度空间定位或细粒度交互的场景，例如基于文本的图像编辑、自动驾驶中的长尾目标检测等。在这些场景中，Grounding 机制能够充分发挥大模型的跨模态理解能力，同时输出结构化的坐标信息，实现更智能、更精准的视觉任务处理。

本文通过一组基于[COCO 数据集](https://huggingface.co/datasets/detection-datasets/coco)采样处理得到的 Grounding 数据集和[Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)模型，演示如何使用[PaddleFormers](https://github.com/PaddlePaddle/PaddleFormers)训练一个支持 Grounding 能力的 LoRA 模型全流程，并验证训练好的模型在定位模式下的性能和效果。整个实验使用1张80G A100显卡(约占61GB)进行训练，总耗时约5小时。

# 2. 任务准备

## 2.1. 模型准备

PaddleFormers 通过在训练配置文件中指定字段 `model_name_or_path`来设置所用的模型。启动训练时如果本地没有该模型的缓存，那么 PaddleFormers 会自动下载模型并加载使用。

您也可以将对应的字段指定成您的本地路径，来加载已经下载好的模型。

## 2.2. 数据准备

这里使用从[COCO 数据集](https://huggingface.co/datasets/detection-datasets/coco)中随机筛选出的15,000条样本作为示例，数据中每一行为一条标准的 `messages` 格式数据，同时包含图像路径信息。

**数据格式示例**

如果需要使用自定义数据，需要将数据整理成如下格式：

```json
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "<image>请找出图中的水果并描述它们的位置"}, {"role": "assistant", "content": "桌子上放着一个<ref-object><bbox>，旁边还有一串<ref-object><bbox>"}], "images": ["xxx.jpg"], "objects": {"ref": ["红苹果", "香蕉"], "bbox": [[245.0, 310.5, 380.0, 450.0], [400.5, 320.0, 650.0, 580.5]]}}
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "<image>Please detect the object in this image."}, {"role": "assistant", "content": "<ref-object><bbox>, <ref-object><bbox></box>"}], "images": ["xxx.jpg"}, "objects": {"ref": ["person", "person"], "bbox": [[200, 300, 500, 600], [324, 557, 409, 683]]}}
```

**字段说明**

数据包含三个核心字段，分别为`messages`、`images`和`objects`。

* 字段`messages`是一个由一系列对话内容构成的列表，列表中的每条对话内容均包含两个子字段，分别为`role`和`content`。在`role`字段中，填入`system`、`user` 或 `assistant`，分别表示“系统设定”、“用户输入”和“模型回复”，需特别说明的是，`system`这一角色标识仅能出现在对话的首轮。而`content`字段用于填入具体的对话内容，特别地，在`user`对应的内容中需包含`<image>`标识以引入图片；在`assistant`对应的内容中，需使用`<ref-object>`和`<bbox>`作为动态占位符，分别指代“目标物体名称”和“目标检测框”。
* 字段`images`是一个由图片路径构成的列表，对应于对话中输入的图像文件/路径。
* 字段`objects`是一个包含具体标注信息的字典，包含`ref`和`bbox`两个子字段。`ref`存储具体的物体名称文本，`bbox`存储对应的原始坐标数据（如 `[xmin, ymin, xmax, ymax]`）。这两个列表中的数据需严格按照顺序，与`messages`字段中出现的`<ref-object>`和`<bbox>` 占位符一一对应，训练框架会自动将其解析并转换为模型所需的坐标格式。

**数据集转换脚本**

为了方便快速上手，本实验提供了一个自动化脚本，支持 **一键完成数据准备** 。该脚本会自动从 huggingface 或 modelscope 下载 COCO 数据集，并处理好 Qwen2.5-VL 模型特有的 Grounding 任务格式，生成可用于训练的`train.jsonl`和`val.jsonl`文件。

<details>
  <summary><b>转换脚本（点击展开/收起）</b></summary>

```python
import os
import io
import json
import glob
import math
import random
import argparse
import pyarrow.parquet as pq

from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

from paddleformers.utils.log import logger

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def parse_args():
    parser = argparse.ArgumentParser(description="COCO Dataset Preparation for Qwen2.5-VL Grounding")
    parser.add_argument("--dataset_repo", type=str, default="detection-datasets/coco", help="dataset repository ID")
    parser.add_argument(
        "--output_dir", type=str, default="./data/coco_grounding", help="Output directory for processed data"
    )
    parser.add_argument("--total_samples", type=int, default=15000, help="Total number of samples to process")
    parser.add_argument("--val_ratio", type=float, default=0.01, help="Validation set ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


# Target resolution following Qwen2.5-VL dynamic resolution strategy
def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280
) -> Tuple[int, int]:
    """Rescales the image so that the following conditions are met:
    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


# Mapping bbox in absolute pixel values (Only for Qwen2.5-VL)
def convert_to_qwen25vl_format(bbox: List[float], orig_height: int, orig_width: int) -> List[int]:
    new_height, new_width = smart_resize(orig_height, orig_width)
    scale_w = new_width / orig_width
    scale_h = new_height / orig_height

    x1, y1, x2, y2 = bbox
    x1_new = round(x1 * scale_w)
    y1_new = round(y1 * scale_h)
    x2_new = round(x2 * scale_w)
    y2_new = round(y2 * scale_h)

    x1_new = max(0, min(x1_new, new_width - 1))
    y1_new = max(0, min(y1_new, new_height - 1))
    x2_new = max(0, min(x2_new, new_width - 1))
    y2_new = max(0, min(y2_new, new_height - 1))

    return [x1_new, y1_new, x2_new, y2_new]


def get_data_path(dataset_repo: str) -> str:
    download_hub = os.environ.get("DOWNLOAD_SOURCE", "huggingface")
    try:
        if download_hub == "huggingface":
            logger.info(f"Checking dataset {dataset_repo} (HuggingFace)...")
            from huggingface_hub import snapshot_download

            local_dir = snapshot_download(repo_id=dataset_repo, repo_type="dataset", allow_patterns="data/*.parquet")

        elif download_hub == "modelscope":
            from modelscope.msdatasets import MsDataset

            dataset_repo_ms = dataset_repo.replace("detection-datasets", "AI-ModelScope")
            logger.info(f"Checking dataset {dataset_repo_ms} (ModelScope)...")
            local_dir = MsDataset.load(dataset_repo_ms, subset_name="detection-datasets--coco", use_streaming=True)
        else:
            raise ValueError(f"Invalid download hub: {download_hub}")

    except Exception as e:
        if download_hub == "huggingface":
            download_cmd = f"hf download {dataset_repo} --repo-type dataset"
        elif download_hub == "modelscope":
            repo_ms_name = dataset_repo.replace("detection-datasets", "AI-ModelScope")
            download_cmd = f"modelscope download --dataset {repo_ms_name}"
        else:
            download_cmd = "N/A（Unexpected download hub）"
        logger.error(f"DOWNLOAD FAILED. Please try downloading manually using these commands: {download_cmd}")

        raise RuntimeError(f"Failed to download from {download_hub}") from e

    data_path = os.path.join(local_dir, "data")
    if not os.path.exists(data_path) and os.path.exists(local_dir):
        return local_dir
    return data_path


def scan_dataset_metadata(files: List[str], desc: str) -> List[dict]:
    candidates = []
    for f in tqdm(files, desc=desc):
        try:
            df = pq.read_table(f, columns=["objects"]).to_pandas()
            for idx, row in df.iterrows():
                cats = row["objects"].get("category", [])
                if any(0 <= c < len(COCO_CLASSES) for c in cats):
                    candidates.append({"file": f, "idx": idx})
        except Exception as e:
            logger.warning(f"Skipping corrupt file {f}: {e}")
    return candidates


def process_row(row, img_save_dir: str) -> Optional[Dict]:
    img_id = row["image_id"]
    fname = f"{img_id:012d}.jpg"
    save_path = os.path.join(img_save_dir, fname)

    try:
        if os.path.exists(save_path):
            img = Image.open(save_path).convert("RGB")
        else:
            image_bytes = row["image"]["bytes"]
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img.save(save_path)
    except Exception as e:
        logger.error(f"Error processing image {img_id}: {e}")
        return None

    objects = row["objects"]
    refs, bboxes = [], []

    category_list = objects.get("category", [])
    bbox_list = objects.get("bbox", [])

    if len(category_list) != len(bbox_list):
        return None

    for cat, bbox in zip(category_list, bbox_list):
        if 0 <= cat < len(COCO_CLASSES):
            refs.append(COCO_CLASSES[cat])
            new_bbox = convert_to_qwen25vl_format(bbox, img.height, img.width)
            bboxes.append(new_bbox)

    if not refs:
        return None

    text_label = ", ".join(["<ref-object><bbox>"] * len(refs))

    return {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "<image>Task: Object Detection"},
            {"role": "assistant", "content": text_label},
        ],
        "images": [os.path.join("images", fname)],
        "objects": {"ref": refs, "bbox": bboxes},
    }


def main():
    args = parse_args()

    img_dir = os.path.join(args.output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    logger.info(f"Starting processing, Output Dir: {args.output_dir}")

    data_path = get_data_path(args.dataset_repo)
    all_files = glob.glob(os.path.join(data_path, "*.parquet"))

    train_files = [f for f in all_files if "train" in os.path.basename(f)]
    val_files = [f for f in all_files if "val" in os.path.basename(f)]

    if not train_files:
        logger.error(f"No parquet training files found in {data_path}")
        return

    logger.info("Scanning metadata (Phase 1)...")
    train_pool = scan_dataset_metadata(train_files, "Scanning Train")
    val_pool = scan_dataset_metadata(val_files, "Scanning Val")

    n_val = int(args.total_samples * args.val_ratio)
    n_train = args.total_samples - n_val

    if len(train_pool) < n_train:
        logger.warning(f"Requested {n_train} train samples, but only found {len(train_pool)}. Using all available.")
        n_train = len(train_pool)

    if len(val_pool) < n_val:
        logger.warning(f"Requested {n_val} val samples, but only found {len(val_pool)}. Using all available.")
        n_val = len(val_pool)

    logger.info(f"Sampling Plan: Train={n_train}, Val={n_val} (Target Total={args.total_samples})")

    random.seed(args.seed)
    random.shuffle(train_pool)
    random.shuffle(val_pool)

    selected_train = train_pool[:n_train]
    selected_val = val_pool[:n_val]

    tasks = defaultdict(list)
    for item in selected_train:
        tasks[item["file"]].append((item["idx"], "train"))
    for item in selected_val:
        tasks[item["file"]].append((item["idx"], "val"))

    logger.info(f"Processing images (Phase 2) - Reading from {len(tasks)} parquet files...")

    train_path = os.path.join(args.output_dir, "train.jsonl")
    val_path = os.path.join(args.output_dir, "val.jsonl")

    with open(train_path, "w", encoding="utf-8") as train_f, open(val_path, "w", encoding="utf-8") as val_f:

        for p_file, task_list in tqdm(tasks.items(), desc="Processing Parquet"):
            try:
                df = pq.read_table(p_file).to_pandas()
                for row_idx, split in task_list:
                    entry = process_row(df.iloc[row_idx], img_dir)
                    if entry:
                        line = json.dumps(entry, ensure_ascii=False) + "\n"
                        if split == "train":
                            train_f.write(line)
                        else:
                            val_f.write(line)
            except Exception as e:
                logger.error(f"Failed to process file {p_file}: {e}")
                continue

    logger.info(f"Output images and jsonl saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
```

</details>

```log
[2025-12-30 15:05:18,936] [    INFO] - Starting processing, Output Dir: ./data/coco_grounding
[2025-12-30 15:05:18,937] [    INFO] - Checking dataset detection-datasets/coco (HuggingFace)...
Fetching 42 files: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 42/42 [00:00<00:00, 322048.94it/s]
[2025-12-30 15:05:19,934] [    INFO] - Scanning metadata (Phase 1)...
Scanning Train: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:04<00:00,  8.30it/s]
Scanning Val: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 10.42it/s]
[2025-12-30 15:05:24,946] [    INFO] - Sampling Plan: Train=14850, Val=150 (Target Total=15000)
[2025-12-30 15:05:25,003] [    INFO] - Processing images (Phase 2) - Reading from 42 parquet files...
Processing Parquet: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 42/42 [01:11<00:00,  1.71s/it]
[2025-12-30 15:06:36,813] [    INFO] - Output images and jsonl saved to: ./data/coco_grounding
```

## 2.3. 训练前效果测试

在正式开始模型训练前，本实验从验证集数据`val.jsonl`中随机挑选一条样本，测试模型的初始表现。

本实验所使用的数据集是一个 Grounding（视觉定位）数据集，它旨在教会模型将文本描述精确映射到图像中的具体坐标区域，从而满足用户 Grounding 任务的需求

以这条数据为例：

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "<image>Task: Object Detection"
    },
    {
      "role": "assistant",
      "content": "<ref-object><bbox>, <ref-object><bbox>, <ref-object><bbox>, <ref-object><bbox>"
    }
  ],
  "images": [
    "images/000000299887.jpg"
  ],
  "objects": {
    "ref": [
      "motorcycle",
      "person",
      "person",
      "truck"
    ],
    "bbox": [
      [5, 129, 322, 471],
      [212, 105, 447, 475],
      [373, 121, 522, 475],
      [0, 178, 48, 213]
    ]
  }
}
```

在训练前，对模型进行了推理测试，可以通过下面的推理脚本来实现：

```python
from paddleformers.transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, process_vision_info
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", convert_from_hf=True,
).eval()

# change the implementation of attention(default is "eager")
model.language_model.config._attn_implementation = "flashmask"
model.visual.config._attn_implementation = "flashmask"

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "./data/coco_grounding/images/000000299887.jpg",
            },
            {"type": "text", "text": "Task: Object Detection"},
        ],
    }
]
# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pd",
)

generated_ids = model.generate(**inputs, max_new_tokens=256)
output_text = processor.batch_decode(
    generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(f"Model Output: \n{output_text[0]}")
```

模型回复如下：

```text
The image contains the following objects:

1. A motorcycle with a windshield and red taillight.
2. Two people standing next to the motorcycle, one wearing a blue shirt and suspenders, and the other in a white shirt and red scarf.
3. A building structure on the right side of the image.
4. Trees and grass in the background.
5. A vehicle partially visible behind the trees.

If you need further analysis or detection of specific parts of the image, please let me know!
```

显然，模型未能遵循目标定位的指令要求。它仅输出了通用的图像描述，而未生成关键的边界框坐标，无法满足用户对结构化 Grounding 任务的输出需求。

# 3. 模型训练

本次实验采用 LoRA 训练策略，超参数设置参考示例 yaml [sft-vl/lora.yaml](../../config/sft-vl/lora.yaml)，具体训练配置文件如下：

```yaml
### data
train_dataset_type: messages
eval_dataset_type: messages
train_dataset_path: ./data/coco_grounding/train.jsonl
train_dataset_prob: "1.0"
eval_dataset_path: ./data/coco_grounding/val.jsonl
eval_dataset_prob: "1.0"
max_seq_len: 8192
packing: true
mix_strategy: concat
template_backend: custom
template: qwen2_vl

### model
model_name_or_path: Qwen/Qwen2.5-VL-7B-Instruct
attn_impl: flashmask
lora: true
lora_rank: 8
lora_alpha: 32

### finetuning
# base
stage: VL-SFT
fine_tuning: lora
seed: 23
do_train: true
do_eval: true
per_device_eval_batch_size: 1
per_device_train_batch_size: 1
num_train_epochs: 5
max_steps: -1
eval_steps: 100
evaluation_strategy: steps
save_steps: 100
save_strategy: steps
logging_steps: 1
gradient_accumulation_steps: 16
logging_dir: ./vdl_log_sft_lora_coco_grounding_15k_gbs16
output_dir: ./checkpoints/qwen2.5-vl-sft-lora-coco-grounding-15k-gbs16
disable_tqdm: true
eval_accumulation_steps: 16

# train
warmup_ratio: 0.05
learning_rate: 1.0e-4

# performance
tensor_model_parallel_size: 1
pipeline_model_parallel_size: 1
sharding: stage1
recompute_granularity: full
recompute_method: uniform
recompute_num_layers: 1
bf16: true
fp16_opt_level: O2
unified_checkpoint: false
save_checkpoint_format: "flex_checkpoint"
load_checkpoint_format: "flex_checkpoint"
freeze_config: freeze_vision freeze_aligner
```

其中有几个比较重要的参数，在此简单说明：

* max_seq_len：最大数据长度，决定模型处理输入数据，需要保证数据长度小于此值，以防止数据被截断。建议在训练前估计数据集中数据长度的范围，防止大部分数据被截断从而影响训练效果，建议设置为8192、32768；
* lora_rank：LoRA 矩阵秩，影响 LoRA 层参数规模，秩越大，模型拟合复杂任务的能力越强，但会增加计算开销。建议设置为8或16；
* lora_alpha：LoRA 缩放系数，影响权重的缩放因子，影响 LoRA 层对原始模型权重的干预程度，建议设置成2或4倍 lora_rank；
* num_train_epochs：训练的 epoch 数。推荐设置为2-3，本次实验由于训练集数据规模大，任务具有一定难度，选择训练5个 epoch 以确保训练效果。
* gradient_accumulation_steps：梯度累积的步数。推荐参考以下算式设置：`每次梯度更新时的批大小 = 训练总卡数 * per_device_train_batch_size * gradient_accumulation_steps / tensor_model_parallel_size / pipeline_model_parallel_size = 16 或 32`
* warmup_raito：学习率的预热步数比例，一般设置为总步数的5%~10%。也可以通过 `warmup_steps` 直接指定具体的预热步数。
* freeze_config：参数冻结配置，通过设置 freeze_vit 和 freeze_aligner 冻结 ViT 和 Projector 模块参数(也不进行 LoRA 训练)，只对 LLM 部分进行 LoRA 训练

## 3.1. 命令行训练脚本

仅需一行代码即可开始模型训练：

```shell
CUDA_VISIBLE_DEVICES=0 paddleformers-cli train qwen25vl_grounding_lora.yaml
```

注：使用 LoRA 方式训练模型后，为了方便推理，需要将 LoRA 参数合并到模型主权重中，参考[LoRA 合参](../../README.md#4-lora-%E5%8F%82%E6%95%B0%E5%90%88%E5%B9%B6)部分。

## 3.2. 训练曲线观察

在训练过程中，您可以利用 VisualDL 工具对模型的 Loss 变化及收敛情况进行观察：

```shell
# visualdl依赖安装: pip install visualdl
# 指定端口和地址，以及日志目录（需要和前面训练配置文件中的目录对应上）
visualdl --logdir ./vdl_log_sft_lora_coco_grounding_15k_gbs16/ --port 8080 --host 0.0.0.0
```

本次 LoRA 训练策略的 Loss 曲线如下：

<div align="center">
  <img width="856" height="602" alt="grounding_vdl_log" src="https://github.com/user-attachments/assets/b13bfaf7-0f64-4734-9883-a14c8915f72c" />
</div>

从曲线图中可以看出，LoRA 微调的 Loss 在训练初期均快速下降，表明模型正在快速适应新的指令格式（由描述性文本转变为坐标输出），随着训练步数增加，曲线逐渐趋于平稳并基本收敛。

# 4. 效果评估

模型训练完成后，再次以该样本为例进行测试。以下是转换为推理输入格式的 Message：

```json
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "data/coco_grounding/images/000000299887.jpg",
            },
            {"type": "text", "text": "Task: Object Detection."},
        ],
    }
]
```

模型输出：

```text
motorcycle(0,132),(379,475), person(212,106),(442,475), person(362,118),(522,475), truck(0,188),(48,213)
```

对比训练前的描述性文本，可以看出模型已经学习到了 Grounding 任务的输出范式，模型不再输出泛化的图像内容描述，而是精准遵循指令，输出包含类别标签与边界框坐标的结构化数据，将视觉感知能力转化为像素级的定位能力，符合预期的训练效果。

为了直观评估模型的实际表现，本实验提供了配套的可视化脚本，将该样本的预测结果和标签进行对比展示。


<details>
  <summary><b>评估脚本（点击展开/收起）</b></summary>

```python
import json
import math
import os
import re

from PIL import Image, ImageDraw, ImageFont


MODEL_RESULT = {
    "image_path": "images/000000299887.jpg",
    "ground_truth": {
        "ref": [
            "motorcycle",
            "person",
            "person",
            "truck"
        ],
        "bbox": [
            [5, 129, 322, 471],
            [212, 105, 447, 475],
            [373, 121, 522, 475],
            [0, 178, 48, 213]
        ]
    },
    "prediction": "motorcycle(0,132),(379,475), person(212,106),(442,475), person(362,118),(522,475), truck(0,189),(48,213)"
}
ROOT_DIR = "data/coco_grounding"

# Target resolution following Qwen2.5-VL dynamic resolution strategy.
def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280
):
    """Rescales the image so that the following conditions are met:
    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError("absolute aspect ratio must be smaller than 200")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def parse_prediction_string(pred_str):
    if not pred_str:
        return []
    pattern = r"([a-zA-Z0-9_ ]+)\s*\(\s*([\d\.]+)\s*,\s*([\d\.]+)\s*\)\s*,\s*\(\s*([\d\.]+)\s*,\s*([\d\.]+)\s*\)"
    matches = re.findall(pattern, pred_str)
    results = []
    for m in matches:
        label = m[0].strip()
        bbox = [float(x) for x in m[1:]]
        results.append({"label": label, "bbox": bbox})
    return results


def get_color_by_label(label):
    palette = [
        "#FF0000",
        "#00AA00",
        "#0000FF",
        "#FF00FF",
        "#800080",
        "#008080",
        "#FFA500",
        "#8B4513",
        "#DC143C",
        "#2E8B57",
        "#4B0082",
        "#FF4500",
        "#2F4F4F",
        "#8B0000",
        "#191970",
    ]

    color_index = hash(label) % len(palette)
    return palette[color_index]


def visualize_sample(
    json_data,
    root_image_dir="",
    output_path="output_vis.jpg",
    show_gt=True,
    show_pred=True,
    use_random_color=True,
):
    if isinstance(json_data, str):
        item = json.loads(json_data)
    else:
        item = json_data

    rel_path = item.get("image_path", "")
    gt_data = item.get("ground_truth", {})
    pred_str = item.get("prediction", "")

    full_image_path = os.path.join(root_image_dir, rel_path)

    try:
        img = Image.open(full_image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image not found at: {full_image_path}")
        img = Image.new("RGB", (640, 640), color=(200, 200, 200))

    orig_w, orig_h = img.size

    try:
        new_h, new_w = smart_resize(orig_h, orig_w)
        resized_img = img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
        scale_x = new_w / orig_w
        scale_y = new_h / orig_h
    except Exception as e:
        print(f"Resize Error: {e}")
        return

    draw = ImageDraw.Draw(resized_img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
    except:
        try:
            font = ImageFont.truetype("arialbd.ttf", 16)
        except:
            font = ImageFont.load_default()

    def draw_single_box(bbox, label, color, line_style="solid", offset_y=0):
        x1, y1, x2, y2 = bbox
        nx1, ny1 = x1 * scale_x, y1 * scale_y
        nx2, ny2 = x2 * scale_x, y2 * scale_y

        draw.rectangle([nx1, ny1, nx2, ny2], outline=color, width=3)
        display_text = label
        text_bbox = draw.textbbox((0, 0), display_text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        text_bg_x1 = nx1
        text_bg_y1 = ny1 - text_h - 4 + offset_y

        if text_bg_y1 < 0:
            text_bg_y1 = ny1 + 4

        text_bg_x2 = text_bg_x1 + text_w + 8
        text_bg_y2 = text_bg_y1 + text_h + 4

        draw.rectangle([text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2], fill=color)
        draw.text((text_bg_x1 + 4, text_bg_y1 + 2), display_text, font=font, fill=(255, 255, 255))

    if show_gt and gt_data:
        refs = gt_data.get("ref", [])
        bboxes = gt_data.get("bbox", [])

        for label, bbox in zip(refs, bboxes):
            label_text = f"GT: {label}"
            color = get_color_by_label(label) if use_random_color else "#00AA00"
            draw_single_box(bbox, label_text, color=color, offset_y=0)

    if show_pred and pred_str:
        preds = parse_prediction_string(pred_str)
        for p in preds:
            label = p["label"]
            label_text = f"Pred: {label}"
            color = get_color_by_label(label) if use_random_color else "#FF0000"
            offset = 25 if (show_gt and not use_random_color) else 0
            draw_single_box(p["bbox"], label_text, color=color, offset_y=offset)

    resized_img.save(output_path, quality=95)
    print(f"Visualization saved to: {output_path}")


if __name__ == "__main__":
    visualize_sample(
        MODEL_RESULT,
        root_image_dir=ROOT_DIR,
        output_path="vis_gt.jpg",
        show_gt=True,
        show_pred=False,
        use_random_color=True,
    )
    visualize_sample(
        MODEL_RESULT,
        root_image_dir=ROOT_DIR,
        output_path="vis_pred.jpg",
        show_gt=False,
        show_pred=True,
        use_random_color=True,
    )

```

</details>

<div align="center">
  <div style="display: flex; gap: 15px; align-items: center; max-width: 90%;">
    <img alt="grounding_demo_gt" src="https://github.com/user-attachments/assets/ad2dd8fc-e34c-4385-b85a-b004343a286e" style="width: 300px; height: auto;" />
    <img alt="grounding_demo_pred" src="https://github.com/user-attachments/assets/36821793-94a7-427f-8629-5711faa83199" style="width: 300px; height: auto;" />
  </div>
</div>

*左图为 GT，右图为预测结果*

# 5. 模型部署
您可以参考[基于 FastDeploy / vLLM 部署模型](../../../docs/zh/deployment_guide.md)，将训练完成的模型使用 vLLM / FastDeploy 等工具进行部署使用
