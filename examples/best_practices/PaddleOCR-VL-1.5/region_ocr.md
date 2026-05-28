# 基于 PaddleOCR-VL-1.5微调实现区域识别能力

## 任务简介
PaddleOCR-VL-1.5 是 PaddleOCR-VL 的全新升级版本，作为一款 0.9B 参数量的超轻量级视觉语言模型 (VLM)，它在 OmniDocBench v1.5 上取得了 94.5% 的 SOTA 准确率，刷新了文档解析领域的性能标杆。该模型不仅延续了前代的高效特性，更在表格、公式及文本识别方面实现了显著提升。

**PaddleOCR-VL-1.5 的核心突破：**

* **极致鲁棒性**：针对真实世界的物理干扰进行了深度优化，在扫描伪影、倾斜、卷曲、屏幕翻拍及光照不均等五大复杂场景下，表现出优于主流开源及闭源模型的抗干扰能力。
* **多任务扩展**：新增了**印章识别（Seal Recognition）**与 **端到端文本定位（Text Spotting）**能力，支持不规则形状的精准定位与多边形检测，有效解决了倾斜或形变文档的解析难题。
* **长文档与多语言支持**：支持跨页表格合并与跨页段落标题识别，解决了长文档解析中的内容碎片化问题；同时扩展了对藏文、孟加拉文等语言的支持，并在生僻字、古籍、多语言表格等场景下表现优异。

**区域识别（RegionOCR）任务**

尽管 PaddleOCR-VL-1.5 具备强大的全文解析能力，但在许多垂直领域的业务流中，我们往往只需要关注文档中特定区域（Region of Interest, ROI）的信息。**区域识别**任务旨在给定图像及目标区域的情况下，精准识别该区域内的文字内容。

这一任务在以下场景中尤为关键：

* **票据与表单处理**：从发票、快递单或申请表中精准提取“金额”、“单号”、“日期”等特定字段，而无需解析无关文字。
* **复杂版面分析**：在密集排版的报表或说明书中，针对性地识别特定单元格或段落。
* **印章与签名提取**：利用模型新增的印章识别能力，对合同或公文中的印章内容进行专项提取与核验。

本教程旨在提供基于 PaddleFormers 微调 PaddleOCR-VL-1.5 模型适配区域识别任务的微调教程，使其能够更好地适应特定领域的文档结构与内容识别需求，资源需求和运行耗时见下方表格：

|硬件|SFT|显存|用时|
|-|-|-|-|
|8*A800|全参|36|1h 08min|
|8*A800|LoRA|33|1h 13min|



## 任务准备
### 模型准备
PaddleFormers 通过在训练配置文件中指定字段`model_name_or_path`来设置所用的模型。启动训练时如果本地没有该模型的缓存，那么 PaddleFormers 会自动下载模型并加载使用。

您也可以将对应的字段指定成您的本地路径，来加载已经下载好的模型。

### 数据集准备
**Demo 数据**

为了方便起见，我们提供了一个快速上手的 RegionOCR 数据集，可用于微调 PaddleOCR-VL-1.5-0.9B 对区域文本进行识别，使用以下命令下载并解压到 `./region_visual`目录（数据集大小约 12 G）：

```bash
wget https://paddleformers.bj.bcebos.com/datasets/ocr-vl/region_visual_dataset.tar
mkdir -p ./region_visual
tar -xvf region_visual_dataset.tar -C ./region_visual
```
其中包含训练集 `region_visual_train.jsonl`、验证集 `region_visual_val.jsonl` 和测试集 `region_visual_test.jsonl`，对应含有 7 万、1 万和 2 万的数据。示例如下：

<div align="center">
  <img width="500" alt="region_train_example" src="https://github.com/forBlank/PaddleFormers/blob/paddleocr_vl_v15_doc/examples/best_practices/PaddleOCR-VL-1.5/assets/region_train_example.jpg" />
</div>

```json
{
    "messages": [
        {"role": "user", "content": "<image>Recognize the text inside the red box"},
        {"role": "assistant", "content": "a way that fulfills the intentions of a user"}
    ],
    "images": ["images/RegionOCR_train.jpg"]
}
```

一个 SFT 数据样本中需包含以下字段：

* `messages`：文本数据列表，记录了用户与模型之间的交互过程，其中每个元素包含一个 `role` 和一个 `content`。
    * `role`：代表消息发送者的身份。
        * `"user"`：用户，代表输入端。
        * `"assistant"`：助手/模型，代表输出端。

    * `content`：消息的具体内容。
        * 输入端包含指令和图片占位符。
            * 提示指令 `Prompt`：根据微调任务自定义提示。
                * 此处使用 `Recognize the text inside the red box` 引导模型识别图片中红色框内的文本。

            * 图片占位符 `<image>`：在文本数据中标记图片插入的位置。

        * 输出端包含模型预期生成的正确答案
            * 此处为图片中红色框内的字符。



* `images`：图像数据列表，存储了对话中涉及到的图片路径（本地路径或 URL）。

**自行准备数据**

如果您想要基于自己的数据集进行训练，请参考 [PaddleFormers - 数据集格式文档](https://github.com/PaddlePaddle/PaddleFormers/blob/develop/docs/zh/dataset_format.md#24-%E5%A4%9A%E6%A8%A1%E6%80%81%E6%8C%87%E4%BB%A4%E5%BE%AE%E8%B0%83sft%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F) 准备数据。

RegionOCR 数据格式可以通过 Spotting 数据格式的数据集生成，我们提供了一个基于开源数据集 [laion-400M](https://laion.ai/blog/laion-400-open-dataset/) 构建的 laion_spotting 数据集，使用以下命令下载：

```bash
wget https://paddleformers.bj.bcebos.com/datasets/ocr-vl/laion_spotting.jsonl
```
Spotting 数据格式说明：

```json
{
    "messages": [
        {"role": "user", "content": "<image>Spotting:"},
        {"role": "assistant", "content": "Jodi Sta.Maria<|LOC_446|><|LOC_45|><|LOC_579|><|LOC_45|><|LOC_579|><|LOC_99|><|LOC_446|><|LOC_99|>\nRichard Yap<|LOC_625|><|LOC_45|><|LOC_731|><|LOC_48|><|LOC_731|><|LOC_107|><|LOC_625|><|LOC_107|>\nBe<|LOC_551|><|LOC_109|><|LOC_618|><|LOC_109|><|LOC_618|><|LOC_256|><|LOC_551|><|LOC_256|>\nCareful<|LOC_431|><|LOC_176|><|LOC_733|><|LOC_176|><|LOC_733|><|LOC_432|><|LOC_431|><|LOC_432|>\nwith<|LOC_511|><|LOC_424|><|LOC_606|><|LOC_424|><|LOC_606|><|LOC_539|><|LOC_511|><|LOC_539|>\nmy<|LOC_576|><|LOC_528|><|LOC_651|><|LOC_528|><|LOC_651|><|LOC_637|><|LOC_576|><|LOC_637|>\nHeart<|LOC_482|><|LOC_571|><|LOC_717|><|LOC_573|><|LOC_717|><|LOC_819|><|LOC_482|><|LOC_819|>\nABSOCBN<|LOC_529|><|LOC_848|><|LOC_668|><|LOC_848|><|LOC_668|><|LOC_885|><|LOC_529|><|LOC_885|>"}
    ],
    "images": ["https://paddle-model-ecology.bj.bcebos.com/PPOCRVL/dataset/laion/batch_0110/aa2f1b3003a398c26f49826dfe0d90a1.jpg"]
}
```
结构与上述内容一致，其中输出端包含模型预期生成的正确答案，格式为文本内容和归一化至 0-1000 刻度的相对坐标标签的结构，实现了识别文字与其在图中空间位置（四边形顶点）的端到端结构化表达。

我们提供数据转换脚本用于将 Spotting 数据格式转为 RegionOCR 数据格式，关键配置项如下：

* `MAX_SAMPLES_PER_IMG`：每张图片最多生成多少个框，默认为`1`个，`None`表示不限制；
* `MAX_TOTAL_SAMPLES`：生成的样本数目，默认为`100000`；
* `BOX_COLOR`：定位框的颜色，默认为红色 `(0, 0, 255)`；
* `BOX_THICKNESS`：定位框的粗细，默认为`3`像素；

<details>
  <summary><b> Spotting2RegionOCR 转换脚本（点击展开/收起）</b></summary>

```python
import json
import re
import os
import cv2
import numpy as np
import requests
import hashlib
import random
random.seed(42)
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# ================= 配置项 =================
INPUT_FILE = './laion_spotting.jsonl'   # 输入的jsonl文件
OUTPUT_FILE = './region_visual.jsonl'   # 输出的jsonl文件
SAVE_DIR = 'images'                     # 图片保存目录

MAX_SAMPLES_PER_IMG = 1              # 每张图片最多生成多少个框 (None表示不限制)
MAX_TOTAL_SAMPLES = 100000           # 生成达到这个总数量后，立即停止程序
PROCESSES = cpu_count()              # 进程数

BOX_COLOR = (0, 0, 255)              # 红框
BOX_THICKNESS = 3                    # 框粗细
# ==========================================

# 正则预编译
loc_pattern = re.compile(r'^(.*?)((?:<\|LOC_\d+\|>)+)$')
num_pattern = re.compile(r'(\d+)')

def download_image_cv2(url):
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            image_array = np.asarray(bytearray(resp.content), dtype=np.uint8)
            img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            return img
    except Exception:
        pass
    return None

def get_bbox_from_quad(coords_list, width, height):
    if len(coords_list) < 2: return None
    xs = [x / 1000.0 * width for x in coords_list[0::2]]
    ys = [y / 1000.0 * height for y in coords_list[1::2]]
    x_min, y_min = int(min(xs)), int(min(ys))
    x_max, y_max = int(max(xs)), int(max(ys))
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(width - 1, x_max), min(height - 1, y_max)
    return (x_min, y_min, x_max, y_max)

def worker_process_line(args):
    """子进程逻辑保持不变"""
    line_idx, line_content = args
    results = []

    line_content = line_content.strip()
    if not line_content: return results

    try:
        data = json.loads(line_content)
        image_url = data['images'][0]

        assistant_content = ""
        for msg in data['messages']:
            if msg['role'] == 'assistant':
                assistant_content = msg['content']
                break

        if not assistant_content: return results

        lines = assistant_content.strip().split('\n')
        valid_items = []
        for text_line in lines:
            match = loc_pattern.match(text_line)
            if match:
                text_content = match.group(1).strip()
                loc_tags = match.group(2)
                coords = [int(x) for x in num_pattern.findall(loc_tags)]
                valid_items.append({"text": text_content, "coords": coords})

        if not valid_items: return results

        if MAX_SAMPLES_PER_IMG is not None and len(valid_items) > MAX_SAMPLES_PER_IMG:
            selected_items = random.sample(valid_items, MAX_SAMPLES_PER_IMG)
        else:
            selected_items = valid_items

        original_img = download_image_cv2(image_url)
        if original_img is None: return results

        img_h, img_w = original_img.shape[:2]

        for idx, item in enumerate(selected_items):
            bbox = get_bbox_from_quad(item['coords'], img_w, img_h)
            if not bbox: continue

            vis_img = original_img.copy()
            cv2.rectangle(vis_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), BOX_COLOR, BOX_THICKNESS)

            file_hash = hashlib.md5(f"{image_url}_{item['text']}_{idx}".encode()).hexdigest()[:12]
            img_filename = f"{line_idx}_{file_hash}.jpg"
            local_img_path = os.path.join(SAVE_DIR, img_filename)

            cv2.imwrite(local_img_path, vis_img)

            new_entry = {
                "messages": [
                    {"role": "user", "content": "<image>Recognize the text inside the red box"},
                    {"role": "assistant", "content": item['text']}
                ],
                "images": [local_img_path]
            }
            results.append(json.dumps(new_entry, ensure_ascii=False))

    except Exception:
        pass

    return results

def main():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    print("正在读取原始文件...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        # 这里依然读取所有行，但后面我们会提前退出
        lines = list(enumerate(f))

    total_lines = len(lines)
    print(f"读取完毕，共 {total_lines} 行源数据。目标生成量: {MAX_TOTAL_SAMPLES}")
    print(f"启动 {PROCESSES} 个进程并行处理...")

    total_generated = 0

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        # 1. 创建进程池
        pool = Pool(processes=PROCESSES)

        try:
            # 2. 迭代处理结果
            iterator = pool.imap_unordered(worker_process_line, lines, chunksize=10)

            for result_list in tqdm(iterator, total=total_lines):
                if not result_list:
                    continue

                # --- 核心控制逻辑 ---

                # 计算还需要多少条
                needed = MAX_TOTAL_SAMPLES - total_generated

                # 如果当前这一批加上去会超标，就切片，只取需要的部分
                if len(result_list) > needed:
                    result_list = result_list[:needed]

                # 写入文件
                for json_str in result_list:
                    f_out.write(json_str + '\n')
                    total_generated += 1

                # 检查是否达标
                if total_generated >= MAX_TOTAL_SAMPLES:
                    print(f"\n已达到目标数量 {MAX_TOTAL_SAMPLES}，正在停止所有进程...")
                    pool.terminate() # 强制杀死所有子进程
                    break

        except KeyboardInterrupt:
            print("用户中断，正在停止...")
            pool.terminate()
        finally:
            pool.join() # 等待进程完全清理

    print(f"\n======================================")
    print(f"处理结束")
    print(f"最终生成数据量: {total_generated}")
    print(f"输出文件: {OUTPUT_FILE}")
    print(f"======================================")

if __name__ == "__main__":
    main()
```
</details>

转换脚本将根据 Spotting 数据中的相对坐标来在绘制图片中绘制对应的定位框，并将对应的文本内容作为 RegionOCR 数据的正确答案。生成 RegionOCR 格式数据集后，可按照需要切分成训练集、验证集和测试集。

## 训练配置
我们针对区域识别示例数据集提供了配置文件，其中的关键训练超参数如下：

* `num_train_epochs=2`：训练的 epoch 数。
* `warmup_ratio=0.1`：线性预热步数，根据任务困难程度调整，此处建议设置成训练步数的 10%。
* `per_device_train_batch_size=16`：每张卡的 batch size 大小，建议根据显存占用情况调整。
* `max_seq_len=16384`：最大序列长度，超出该长度的数据将被截断或者丢弃。建议在训练前估计数据集中数据长度的范围，防止大部分数据被截断从而影响训练效果。
* `gradient_accumulation_steps=1`：梯度累积步数。
    * 每达到该步数整数倍更新一次模型参数。
    * 当显存不足时，可以减小 `per_device_train_batch_size` 并增大 `gradient_accumulation_steps`。
    * 用时间换空间策略，可以减少显存占用，但会延长训练时间。

* `learning_rate`：学习率，即每次参数更新的幅度。
    * 全参训练 `learning_rate=5e-6`
    * LoRA 训练 `learning_rate=5e-4`

* 自定义模板和多模态数据处理插件
    * 由于 PaddleOCR-VL-1.5 的 `chat_template`相较于 PaddleOCR-VL 有更新，我们可以自定义新模板并通过外接的方式注册和使用。
    * `template=paddleocr_vl_v15`：指定数据预处理的模板。
    * `custom_register_path=./paddleocr_vl_v15_template.py`：指定自定义模板的文件路径。


请将以下模板文件保存至本地：

<details>
  <summary><b> PaddleOCR-VL-1.5 模板文件（点击展开/收起）</b></summary>

```python
from paddleformers.datasets.template.template import *
from paddleformers.datasets.template.mm_plugin import *
from paddleformers.datasets.template.augment_utils import *

# ==========================================
# MMPlugin
# ==========================================

@dataclass
class PaddleOCRVLV15Plugin(BasePlugin):
    image_bos_token: str = "<|IMAGE_START|>"
    image_eos_token: str = "<|IMAGE_END|>"

    def __init__(self, image_token, video_token, audio_token, **kwargs):
        super().__init__(image_token, video_token, audio_token, **kwargs)

        # here, we don't use image augmentation to simplify the training
        # you can customize the image augmentation as you like
        self.image_augmentation = self.get_ocr_augmentations(
            rotation_p=0.0,
            jpeg_p=0.0,
            scale_p=0.0,
            padding_p=0.0,
            color_jitter_p=0.0,
        )

    def get_ocr_augmentations(
        self,
        scale_range=(0.8, 1.2),
        scale_p=0.5,
        padding_range=(0, 15),
        padding_p=0.5,
        rotation_degrees=[0],
        rotation_p=0.5,
        color_jitter_p=0.5,
        jpeg_quality_range=(40, 90),
        jpeg_p=0.5,
    ):

        augmentations = []

        if scale_p > 0:
            scale_transform = RandomScale(scale_range=scale_range)
            augmentations.append(RandomApply([scale_transform], p=scale_p))

        if padding_p > 0:
            padding_transform = RandomSingleSidePadding(padding_range=padding_range, fill="white")
            augmentations.append(RandomApply([padding_transform], p=padding_p))

        if rotation_p > 0 and rotation_degrees:
            rotation_transform = RandomDiscreteRotation(degrees=rotation_degrees, interpolation="nearest", expand=True)
            augmentations.append(RandomApply([rotation_transform], p=rotation_p))

        if color_jitter_p > 0:
            color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1)
            augmentations.append(RandomApply([color_jitter], p=color_jitter_p))

        if jpeg_p > 0:
            jpeg_transform = JpegCompression(quality_range=jpeg_quality_range)
            augmentations.append(RandomApply([jpeg_transform], p=jpeg_p))

        return transforms.Compose(augmentations)

    @override
    def _preprocess_image(self, image, **kwargs):

        width, height = image.size
        image_max_pixels = kwargs["image_max_pixels"]
        image_min_pixels = kwargs["image_min_pixels"]
        image_processor = kwargs["image_processor"]

        # pre-resize before augmentation
        resized_height, resized_width = image_processor.get_smarted_resize(
            height,
            width,
            min_pixels=image_min_pixels,
            max_pixels=image_max_pixels,
        )[0]

        image = image.resize((resized_width, resized_height))

        if image and hasattr(self, "image_augmentation"):
            image = self.image_augmentation(image)

        return image

    @override
    def _get_mm_inputs(
        self,
        images,
        videos,
        audios,
        processor,
        **kwargs,
    ):
        image_processor = getattr(processor, "image_processor", None)
        mm_inputs = {}
        if len(images) != 0:
            images = self._regularize_images(
                images,
                image_max_pixels=getattr(image_processor, "max_pixels", 1003520),
                image_min_pixels=getattr(image_processor, "min_pixels", 112896),
                image_processor=image_processor,
            )["images"]
            mm_inputs.update(image_processor(images, return_tensors="pd"))

        return mm_inputs

    @override
    def process_messages(
        self,
        messages,
        images,
        videos,
        audios,
        mm_inputs,
        processor,
    ):
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens = 0
        messages = deepcopy(messages)
        image_processor = getattr(processor, "image_processor")

        merge_length = getattr(image_processor, "merge_size") ** 2
        if self.expand_mm_tokens:
            image_grid_thw = mm_inputs.get("image_grid_thw", [])
        else:
            image_grid_thw = [None] * len(images)

        # here, we replace the IMAGE_PLACEHOLDER with the corresponding image tokens
        # you can customize the way of inserting image tokens as you like
        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                image_seqlen = (
                    image_grid_thw[num_image_tokens].prod().item() // merge_length if self.expand_mm_tokens else 1
                )
                content = content.replace(
                    IMAGE_PLACEHOLDER,
                    f"{self.image_bos_token}{self.image_token * image_seqlen}{self.image_eos_token}",
                    1,
                )
                num_image_tokens += 1

            message["content"] = content

        return messages

register_mm_plugin(
    name = "paddleocr_vl_v15",
    plugin_class = PaddleOCRVLV15Plugin,
)

# ==========================================
# Template
# ==========================================

register_template(
    name="paddleocr_vl_v15",
    format_user=StringFormatter(slots=["User: {{content}}\nAssistant:\n"]), # "/n" after "Assistant:"
    format_assistant=StringFormatter(slots=["{{content}}"]),
    format_system=StringFormatter(slots=["{{content}}\n"]),
    format_prefix=EmptyFormatter(slots=["<|begin_of_sentence|>"]),
    chat_sep="<|end_of_sentence|>",
    mm_plugin=get_mm_plugin(name="paddleocr_vl_v15", image_token="<|IMAGE_PLACEHOLDER|>"),
)
```
</details>

除了自定义模板的格式，PaddleFormers 还支持自定义模板的多模态数据处理插件，包括：自定义数据增强、自定义多模态 Token 以及替换形式，更多模板和多模态数据处理插件的自定义和使用请参考 [PaddleFormers - 注册 template&mm_plugin](https://github.com/PaddlePaddle/PaddleFormers/blob/develop/docs/zh/template_zh.md)。

更多相关参数可在配置文件中查看。

<details>
  <summary><b> 全参配置（点击展开/收起）</b></summary>

```yaml
### data
train_dataset_type: messages
eval_dataset_type: messages
train_dataset_path: ./region_visual/region_visual_train.jsonl
train_dataset_prob: "1.0"
eval_dataset_path: ./region_visual/region_visual_val.jsonl
eval_dataset_prob: "1.0"
max_seq_len: 16384
padding_free: True
truncate_packing: False
dataloader_num_workers: 8
mix_strategy: concat
template_backend: custom
template: paddleocr_vl_v15
custom_register_path: ./paddleocr_vl_v15_template.py


### model
model_name_or_path: PaddlePaddle/PaddleOCR-VL-1.5
_attn_implementation: flashmask
copy_custom_file_list: "configuration_paddleocr_vl.py image_processing_paddleocr_vl.py modeling_paddleocr_vl.py processing_paddleocr_vl.py inference.yml"

### finetuning
# base
stage: VL-SFT
fine_tuning: full
seed: 23
do_train: true
do_eval: true
per_device_eval_batch_size: 16
per_device_train_batch_size: 16
num_train_epochs: 2
max_steps: -1
max_estimate_samples: 500
eval_steps: 400
evaluation_strategy: steps
save_steps: 400
save_strategy: steps
logging_steps: 1
gradient_accumulation_steps: 1
logging_dir: ./PaddleOCR-VL-1.5-SFT-RegionOCR/visualdl_logs/
output_dir: ./PaddleOCR-VL-1.5-SFT-RegionOCR
disable_tqdm: true
eval_accumulation_steps: 16

# train
lr_scheduler_type: cosine
warmup_ratio: 0.1
learning_rate: 5.0e-6
min_lr: 5.0e-7

# optimizer
weight_decay: 0.1
adam_epsilon: 1.0e-8
adam_beta1: 0.9
adam_beta2: 0.95

# performance
tensor_model_parallel_size: 1
pipeline_model_parallel_size: 1
sharding: stage1
recompute_granularity: full
recompute_method: uniform
recompute_num_layers: 1
bf16: true
fp16_opt_level: O2

# save
unified_checkpoint: False
save_checkpoint_format: "flex_checkpoint"
load_checkpoint_format: "flex_checkpoint"
```
</details>

<details>
  <summary><b> LoRA 配置（点击展开/收起）</b></summary>

```yaml
### data
train_dataset_type: messages
eval_dataset_type: messages
train_dataset_path: ./region_visual/region_visual_train.jsonl
train_dataset_prob: "1.0"
eval_dataset_path: ./region_visual/region_visual_val.jsonl
eval_dataset_prob: "1.0"
max_seq_len: 16384
padding_free: True
truncate_packing: False
dataloader_num_workers: 8
mix_strategy: concat
template_backend: custom
template: paddleocr_vl_v15
custom_register_path: ./paddleocr_vl_v15_template.py


### model
model_name_or_path: PaddlePaddle/PaddleOCR-VL-1.5
_attn_implementation: flashmask
lora: true
lora_rank: 8
copy_custom_file_list: "configuration_paddleocr_vl.py image_processing_paddleocr_vl.py modeling_paddleocr_vl.py processing_paddleocr_vl.py inference.yml"

### finetuning
# base
stage: VL-SFT
fine_tuning: lora
seed: 23
do_train: true
do_eval: true
per_device_eval_batch_size: 16
per_device_train_batch_size: 16
num_train_epochs: 2
max_steps: -1
max_estimate_samples: 500
eval_steps: 400
evaluation_strategy: steps
save_steps: 400
save_strategy: steps
logging_steps: 1
gradient_accumulation_steps: 1
logging_dir: ./PaddleOCR-VL-1.5-SFT-RegionOCR-lora/visualdl_logs/
output_dir: ./PaddleOCR-VL-1.5-SFT-RegionOCR-lora
disable_tqdm: true
eval_accumulation_steps: 16

# train
lr_scheduler_type: cosine
warmup_ratio: 0.1
learning_rate: 5.0e-4
min_lr: 5.0e-5

# optimizer
weight_decay: 0.1
adam_epsilon: 1.0e-8
adam_beta1: 0.9
adam_beta2: 0.95

# performance
tensor_model_parallel_size: 1
pipeline_model_parallel_size: 1
sharding: stage1
recompute_granularity: full
recompute_method: uniform
recompute_num_layers: 1
bf16: true
fp16_opt_level: O2

# save
unified_checkpoint: False
save_checkpoint_format: "flex_checkpoint"
load_checkpoint_format: "flex_checkpoint"
```
</details>

## SFT 训练
### SFT 全参训练
使用以下命令行即可启动全参训练：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
paddleformers-cli train examples/best_practices/PaddleOCR-VL-1.5/paddleocr-vl-v15_full_16k_region_config.yaml \
                        model_name_or_path=PaddlePaddle/PaddleOCR-VL-1.5 \
                        train_dataset_path=./region_visual/region_visual_train.jsonl \
                        eval_dataset_path=./region_visual/region_visual_val.jsonl \
                        pre_alloc_memory=29
```
设置 `pre_alloc_memory` 预分配显存从而减少显存碎片，根据序列长度、批大小和硬件显存调整。

PaddleFormers 默认使用机器上的全部 GPU，可以通过环境变量 `CUDA_VISIBLE_DEVICES` 设置 PaddleFormers 能够使用的 GPU。

可以通过 `visualdl` 对训练过程可视化，使用以下命令行即可启动（下方命令将端口 port 设置为 `8084`，需要根据实际情况设置可用端口）：

```bash
visualdl --logdir ./PaddleOCR-VL-1.5-SFT-RegionOCR/visualdl_logs/ --port 8084
```

成功启动后该服务后，在浏览器输入 `ip:port` ，则可以看到训练日志（通过 `hostname -i` 命令可以查看机器的 ip 地址）。

损失曲线如下：

<div align="center">
  <img width="500" alt="regionocr_train_loss" src="https://github.com/forBlank/PaddleFormers/blob/paddleocr_vl_v15_doc/examples/best_practices/PaddleOCR-VL-1.5/assets/regionocr_train_loss.png" />
</div>


### SFT LoRA 训练
使用以下命令行即可启动 LoRA 训练：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
paddleformers-cli train examples/best_practices/PaddleOCR-VL-1.5/paddleocr-vl-v15_lora_16k_region_config.yaml \
                        model_name_or_path=PaddlePaddle/PaddleOCR-VL-1.5 \
                        train_dataset_path=./region_visual/region_visual_train.jsonl \
                        eval_dataset_path=./region_visual/region_visual_val.jsonl \
                        pre_alloc_memory=26
```


## 模型结构说明
### SFT 全参
全参训练结束后，模型会保存在 `output_dir=./PaddleOCR-VL-1.5-SFT-RegionOCR` 指定路径下，其中包含：

* config.json：模型配置文件
* model-0000X-of-0000Y.safetensors：模型权重文件
* model.safetensors.index.json：模型权重索引文件
* tokenizer.model & tokenizer_config.json & special_tokens_map.json & added_tokens.json：分词器文件
* train_args.bin：训练参数文件，记录训练使用的参数等
* train_state.json：训练状态文件，记录训练步数和最优指标等
* train_results.json & all_results.json：训练结果文件，记录训练进度&用时&每步耗时&每样本耗时等
* generation.json：生成配置文件
* checkpoint-[save_steps*n]：检查点文件夹，在 `save_steps` 整数倍保存训练状态，除以上文件外，还会保存 master-weight & optimizer-state & scheduler-state 等，可用于训练中断后恢复训练

### SFT LoRA
LoRA 训练结束后，模型会保存在 `output_dir=./PaddleOCR-VL-1.5-SFT-RegionOCR-lora` 指定路径下。相较于 SFT 全参，SFT LoRA 的模型结构会有所不同，其中包含：

* lora_config.json：LoRA 模型配置文件
* peft_model-0000X-of-0000Y.safetensors：LoRA 模型权重文件
* peft_model.safetensors.index.json：LoRA 权重索引文件

使用以下命令行即可合并 LoRA 权重：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
paddleformers-cli export examples/best_practices/PaddleOCR-VL-1.5/paddleocr-vl-v15_lora_export_region.yaml \
    model_name_or_path=PaddlePaddle/PaddleOCR-VL-1.5 \
    output_dir=./PaddleOCR-VL-1.5-SFT-RegionOCR-lora
```
合并后的完整模型权重保存在 `output_dir=./PaddleOCR-VL-1.5-SFT-RegionOCR-lora/export` 路径下。

## 推理
### 单样本推理
RegionOCR 测试图像：

<div align="center">
  <img width="500" alt="region_test_example" src="https://github.com/forBlank/PaddleFormers/blob/paddleocr_vl_v15_doc/examples/best_practices/PaddleOCR-VL-1.5/assets/region_test_example.jpg" />
</div>

使用以下命令行进行单样本推理：

```bash
python generate.py
```

<details>
  <summary><b> 单样本推理脚本（点击展开/收起）</b></summary>


```python
import requests
from io import BytesIO

import paddle
from PIL import Image
from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor
from paddleformers.generation import GenerationConfig

model_path = "./PaddleOCR-VL-1.5-SFT-RegionOCR"

model = AutoModelForConditionalGeneration.from_pretrained(
    model_path, convert_from_hf=True,
).eval()

# change the implementation of attention(default is "eager")
model.config._attn_implementation = "flashmask"
model.visual.config._attn_implementation = "flashmask"

processor = AutoProcessor.from_pretrained(model_path)

image_path = "./region_visual/images/52091_4821c014cb46.jpg"
image = Image.open(image_path).convert("RGB")

PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
    "region": "Recognize the text inside the red box",
}
task = "region" # Options: 'ocr' | 'table' | 'chart' | 'formula' | 'region'

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image
            },
            {"type": "text", "text": PROMPTS[task]},
        ],
    }
]

# Preparation for inference
inputs = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pd",
)

generation_config = GenerationConfig(
    do_sample=False, # greedy_search
    bos_token_id=1,
    eos_token_id=2,
    pad_token_id=0,
    use_cache=True
)

with paddle.no_grad():
    generated_ids = model.generate(**inputs, generation_config=generation_config, max_new_tokens=1024)
    generated_ids = generated_ids[0].tolist()[0]
    output_text = processor.decode(generated_ids, skip_special_tokens=True)

print(output_text)

# GT = eccentric, reclusive writer, William Forrester. Jamal is a gifted student and
# Excepted Answer = eccentric, reclusive writer, William Forrester. Jamal is a gifted student and
```

</details>

预期输出为测试图像中红色定位框内的文本：eccentric, reclusive writer, William Forrester. Jamal is a gifted student and

### 测试集评估
使用归一化 Levenshtein 编辑距离作为评估指标：

* Levenshtein 编辑距离：从预测字符串 A 变为真实字符串 B 最少需要的操作次数（插入/删除/修改一个字符）。
* 归一化 Levenshtein 编辑距离：将编辑距离除以 max(A, B) 来进行归一化。

使用以下命令行安装 Levenshtein 库：

```bash
pip install Levenshtein
```
使用以下命令行进行测试集评估：

```bash
model_path="./PaddleOCR-VL-1.5-SFT-RegionOCR"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m paddle.distributed.launch --log_dir ./log \
    eval.py \
    --model_name_or_path "${model_path}" \
    --data_path ./region_visual/region_visual_test.jsonl \
    --output_path "${model_name}_eval_result.jsonl"
```

<details>
  <summary><b> 测试集评估脚本（点击展开/收起）</b></summary>

```python
import argparse
import json
import os
import sys
import time
import requests
from io import BytesIO

from PIL import Image
import paddle
import paddle.distributed as dist
from tqdm import tqdm
import Levenshtein  # Requires python-Levenshtein

from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor, AutoConfig
from paddleformers.generation import GenerationConfig

def parse_args():
    parser = argparse.ArgumentParser(description="PaddleFormers & PaddleOCR-VL Model Evaluation Script")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Model path or name")
    parser.add_argument("--data_path", type=str, required=True, help="Test data path (jsonl format)")
    parser.add_argument("--output_path", type=str, default="eval_results.jsonl", help="Result save path")
    parser.add_argument("--max_length", type=int, default=1024, help="Max generation length")
    parser.add_argument("--device", type=str, default="gpu", help="Device: gpu / cpu / xpu / iluvatar_gpu")
    return parser.parse_args()


def load_model_and_processor(model_path, device):
    print(f"Loading model: {model_path} ...")
    paddle.set_device(device)

    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForConditionalGeneration.from_pretrained(model_path, convert_from_hf=True,)
    # change the implementation of attention(default is "eager")
    model.config._attn_implementation = "flashmask"
    model.visual.config._attn_implementation = "flashmask"
    model.eval()
    print("Model loaded successfully!")
    return model, processor


def compute_metrics(predictions, references):
    """
    Compute evaluation metrics: Normalized Edit Distance
    """
    total_ned = 0
    num_samples = len(predictions)

    if num_samples == 0:
        return 0.0

    for pred, ref in zip(predictions, references):
        # Compute NED
        dist = Levenshtein.distance(pred, ref)
        max_len = max(len(pred), len(ref))
        if max_len > 0:
            total_ned += dist / max_len

    avg_ned = total_ned / num_samples
    return avg_ned


def generate_response(model, processor, messages, max_length=1024):

    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pd",
    )

    generation_config = GenerationConfig(
        do_sample=False, # greedy_search
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        use_cache=True
    )

    with paddle.no_grad():
        generated_ids = model.generate(**inputs, generation_config=generation_config, max_new_tokens=1024)
        generated_ids = generated_ids[0].tolist()[0]
        output_text = processor.decode(generated_ids, skip_special_tokens=True)

    return output_text


def main():
    start_time = time.time()
    args = parse_args()

    # Initialize distributed environment
    try:
        dist.init_parallel_env()
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    except Exception:
        rank = 0
        world_size = 1
        print("Distributed environment not detected, using single card mode.")

    # 1. Load Model
    model, processor = load_model_and_processor(args.model_name_or_path, args.device)

    # 2. Read Data
    if rank == 0:
        print(f"Reading data: {args.data_path}")
    samples = []
    with open(args.data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    # Data Splitting
    total_samples = len(samples)
    samples = samples[rank::world_size]

    if rank == 0:
        print(f"Total test samples loaded: {total_samples}")
    print(f"[Rank {rank}] Assigned {len(samples)} samples")

    # 3. Inference Loop
    results = []
    # Predictions and references will be aggregated by rank 0 at the end

    for sample in tqdm(samples, desc=f"[Rank {rank}] Inferencing", position=rank):

        query = sample["messages"][0]["content"]
        image_path = sample["images"][0]
        image = Image.open(image_path).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image
                    },
                    {"type": "text", "text": query.replace('<image>', '')},
                ],
            }
        ]
        output = generate_response(model, processor, messages, args.max_length)
        sample["answer"] = output
        sample["label"] = sample["messages"][1]["content"]

        results.append(sample)

    # 4. Save partial results
    part_file = f"{args.output_path}.part{rank}"
    with open(part_file, 'w', encoding='utf-8') as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
    print(f"[Rank {rank}] Results saved to temporary file: {part_file}")

    # Wait for all processes to complete
    if world_size > 1:
        dist.barrier()

    # 5. Rank 0 Aggregation and Evaluation
    if rank == 0:
        all_results = []
        print("Aggregating results from all Ranks...")
        # Collect all partial results
        for r in range(world_size):
            part_file_r = f"{args.output_path}.part{r}"
            if os.path.exists(part_file_r):
                with open(part_file_r, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            all_results.append(json.loads(line))
                # Remove temporary file
                try:
                    os.remove(part_file_r)
                except OSError as e:
                    print(f"Warning: Unable to remove temporary file {part_file_r}: {e}")
            else:
                print(f"Warning: Result file {part_file_r} for Rank {r} not found")

        # Extract predictions and labels
        predictions = [res.get("answer", "") for res in all_results]
        references = [res.get("label", "") for res in all_results]

        # Compute metrics
        print("Computing evaluation metrics...")
        avg_ned = compute_metrics(predictions, references)

        # Output results
        print("\n" + "="*40)
        print("        Evaluation Report")
        print("="*40)
        print(f"Model: {args.model_name_or_path}")
        print(f"Total Samples: {len(all_results)}")
        print("-" * 40)
        print(f"Avg. NED: {avg_ned:.4f} (Lower is better)")
        print("="*40)

        # Save detailed results
        with open(args.output_path, 'w', encoding='utf-8') as f:
            for res in all_results:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
        print(f"\nDetailed results saved to: {args.output_path}")

        end_time = time.time()
        print(f"Total execution time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
```
</details>

评估结果保存在 `output_path=./PaddleOCR-VL-SFT-RegionOCR_eval_result.jsonl` 文件中。

上述直接运行的评估脚本基于 PaddleFormers 原生推理，测试集较大的情况下，推理速度可能较慢。推荐使用 FastDeploy 等高性能推理引擎将微调后的模型部署为 API 服务，通过高并发请求进行批量推理评估，详见[部署推理](#部署推理)。

微调前后的模型测试集评估结果如下：

|Model|Avg. NED|
|-|-|
|PaddleOCR-VL-1.5|0.9312|
|PaddleOCR-VL-1.5RegionOCR-SFT (Full)|0.2061|
|PaddleOCR-VL-1.5RegionOCR-SFT (LoRA)|0.2031|

### 部署推理
部署 PaddleOCR-VL-1.5 模型，请参考 [PaddleFormers - 模型部署文档](https://github.com/PaddlePaddle/PaddleFormers/blob/develop/docs/zh/deployment_guide.md) 和 [FastDeploy - PaddleOCR-VL-0.9B Best Practices](https://paddlepaddle.github.io/FastDeploy/zh/best_practices/PaddleOCR-VL-0.9B/)。



## 注意事项

### 更多硬件上的使用说明
PaddleOCR-VL-1.5 支持基于昆仑芯 P800 和天数智芯 150s 进行微调。本教程选用了规模较大的数据集，在部分硬件上运行时间可能较长，如果希望**快速跑通流程**，或仅需验证**国产硬件环境的兼容性**，建议优先参考我们的 [PaddleFormers - 基于 PaddleOCR-VL 微调实现孟加拉语识别能力](https://github.com/PaddlePaddle/PaddleFormers/tree/develop/examples/best_practices/PaddleOCR-VL)，该教程使用精简数据集，可在短时间内完成微调全链路，助力迅速掌握多硬件适配技巧。

### RoPE Triton Kernel 加速

使用英伟达 GPU 训练的过程中，RoPE 可以通过使用 Triton Kernel 实现来加速。

可以使用以下命令安装 triton 和 use-triton-in-paddle 依赖包：

```shell
pip install triton==3.6.0
pip install use-triton-in-paddle==0.1.0
```

请注意，Triton Kernel 对于硬件环境要求较高，如果硬件环境不支持，请使用以下命令卸载 triton 和 use-triton-in-paddle 依赖包：

```shell
pip uninstall triton -y
pip uninstall use-triton-in-paddle -y
```
