# 基于 PaddleOCR-VL-1.5微调表格数据

## 任务简介
PaddleOCR-VL-1.5 是 PaddleOCR-VL 的全新升级版本，作为一款 0.9B 参数量的超轻量级视觉语言模型 (VLM)，它在 OmniDocBench v1.5 上取得了 94.5% 的 SOTA 准确率，刷新了文档解析领域的性能标杆。该模型不仅延续了前代的高效特性，更在表格、公式及文本识别方面实现了显著提升。

**PaddleOCR-VL-1.5 的核心突破：**

* **极致鲁棒性**：针对真实世界的物理干扰进行了深度优化，在扫描伪影、倾斜、卷曲、屏幕翻拍及光照不均等五大复杂场景下，表现出优于主流开源及闭源模型的抗干扰能力。
* **多任务扩展**：新增了**印章识别（Seal Recognition）**与 **端到端文本定位（Text Spotting）**能力，支持不规则形状的精准定位与多边形检测，有效解决了倾斜或形变文档的解析难题。
* **长文档与多语言支持**：支持跨页表格合并与跨页段落标题识别，解决了长文档解析中的内容碎片化问题；同时扩展了对藏文、孟加拉文等语言的支持，并在生僻字、古籍、多语言表格等场景下表现优异。

**表格识别任务**

在真实业务流中，**表格（Table）**具有极高的信息密度和复杂的结构，不仅仅是文本的排列，更是逻辑关系的二维映射。

在本次微调教程中，我们将聚焦于**复杂表格识别（Complex Table Recognition）**。这不仅仅是识别单元格内的文字，更是要让模型精准理解并还原表格的结构。我们关注的复杂表格通常具有以下特征：

* **多重合并单元格**：存在大量的跨行（Rowspan）或跨列（Colspan）操作，导致视觉对齐关系与逻辑归属关系不一致。
* **嵌套与多层表头**：表头具有层级结构，需要模型理解父子层级。
* **空白表格元素**：表格具有大量的空白元素，需要模型正确处理数据稀疏场景。

本教程旨在提供基于 PaddleFormers 微调 PaddleOCR-VL-1.5 模型适配复杂表格识别任务的微调教程，值得一提的是 PaddleOCR-VL-1.5 已经具有很强的复杂表格识别能力（可通过 [PaddleOCR 官网](https://aistudio.baidu.com/paddleocr) 在线体验），资源需求和运行耗时见下方表格：

|硬件|SFT|显存|用时|
|-|-|-|-|
|8*A800|全参|46|4h 26min|
|8*A800|LoRA|42|4h 30min|



## 任务准备
### 模型准备
PaddleFormers 通过在训练配置文件中指定字段`model_name_or_path`来设置所用的模型。启动训练时如果本地没有该模型的缓存，那么 PaddleFormers 会自动下载模型并加载使用。

您也可以将对应的字段指定成您的本地路径，来加载已经下载好的模型。

### 数据集准备
**Demo 数据**

为了方便起见，我们提供了一个快速上手的复杂表格 Table 数据集，可用于微调 PaddleOCR-VL-1.5-0.9B 对复杂表格进行识别，该数据集为程序生成的复杂表格结构，实际内容不具备现实意义，使用以下命令下载并解压到 `./complex_table` 目录（数据集大小约 33 G）：

```bash
wget https://paddleformers.bj.bcebos.com/datasets/ocr-vl/complex_table_dataset.tar
mkdir -p ./complex_table
tar -xvf complex_table_dataset.tar -C ./complex_table
```
其中包含训练集 `complex_table_train.jsonl`、验证集 `complex_table_val.jsonl` 和测试集 `complex_table_test.jsonl`，对应含有 12 万、1 万和 2 万的数据。示例如下：

<div align="center">
  <img width="500" alt="table_train_example" src="https://github.com/forBlank/PaddleFormers/blob/paddleocr_vl_v15_doc/examples/best_practices/PaddleOCR-VL-1.5/assets/table_train_example.png" />
</div>

```json
{
    "messages": [
        {"role": "user", "content": "<image>Table Recognition:"},
        {"role": "assistant", "content": "<fcel>E-commerce Sales Data<fcel>颁奖机构<fcel>Bureau of Industry and Security for ExportRegulations<fcel>原材料实际质量<fcel>Union Representation Details<lcel><lcel><lcel><fcel>目前影像学检查结果<fcel>Art Exhibition Theme<fcel>Alternative Medicine<fcel>期货交割记录<nl><fcel>环保项目公众参与度<fcel>女性<fcel>Immediately<fcel>普通话二级甲等<fcel>昂歌信息科技有限公司学院<fcel>西藏自治区市<fcel>785-42-0107<fcel>Other<ecel><ecel><ucel><ecel><nl><fcel>股利分配决策分析<ecel><fcel>Dec 2022-Jul 2025<fcel>改革, 坚持, 比重<ecel><fcel>English TOEFL 100<fcel>Kevin Villegas 278.788.0670<lcel><lcel><lcel><ucel><fcel>2023.09-2025.04<nl><fcel>Economic Forecasting Models<fcel>739.316.3006<fcel>20k-15k<fcel>15915523931<fcel>accessories, sitting, bids<fcel>吸纳, 拓宽, 打牢<fcel>Within 1 month<ecel><fcel>2025-12<fcel>PMP<ucel><ecel><nl><fcel>Employee Wellness Program Date<ecel><ecel><fcel>20k-15k<fcel>随时<ecel><fcel>Immediately<fcel>1969-06-24<fcel>博士<fcel>产品经理<ucel><ecel><nl><fcel>Sales Training Programs<fcel>迪摩科技有限公司职业技术学院<fcel>项目经理<fcel>$80k-$70k per year<fcel>2023.08-2025.12<fcel>Brianna Carroll (715)258-3235<fcel>Democrat<fcel>产品总监<fcel>2005-05-19<fcel>Master<ucel><ecel><nl><fcel>收藏情况<fcel>2021-08<ecel><fcel>跨领域从业者<fcel>Do you come here often? Initially composing light-hearted and irreverentworks, he also wrote serious, sombre and religious pieces beginningin the 1930s.<fcel>Jones, Crawford and Freeman<fcel>2021.08-2025.12<ecel><fcel>Senior Developer<lcel><ucel><ecel><nl><fcel>Metz Standards<fcel>义渠是西北黄土高原上的强国，自春秋至战国，与秦抗衡百余年 匈奴、东胡等游牧民族更是军事素质高，作战能力强<fcel>$80k-$80k per year<fcel>随着各国之间政治、经济关系的加强，诸夏文化与秦、楚、吴、越文化的交流与融合，统一的趋向日益强烈 军队逐渐改变成步兵和骑兵，并以军功论赏和升迁，因此军队的战斗力增强，所向无敌<fcel>Chilean<ecel><fcel>Japanese JLPT N1<fcel>群众<fcel>Within 3 months<fcel>海创网络有限公司大学<ucel><fcel>Business Administration<nl><fcel>特殊检查宣教内容<ecel><ecel><ecel><fcel>15287 Alexis Isle Suite 568, Banksshire, Idaho<fcel>Associate<fcel>中共党员<ecel><fcel>Schmidt-Martinez<fcel>Colorado, Guatemala<fcel>核心开发<fcel>高级工程师<nl><fcel>HL7 Interfaces<ecel><fcel>泰国<fcel>男性<ecel><fcel>$60k-$70k per year<fcel>671-94-2297<ecel><fcel>Minnesota, Bangladesh<fcel>Sep 2024-Apr 2025<fcel>howephilip@example.net<ecel><nl>"}
    ],
    "images": ["images/9c1e1573af7c_border_211_CSQ3XM27W0EXMQI0T8N5_0.png"]}
```
一个 SFT 数据样本中需包含以下字段：

* `messages`：文本数据列表，记录了用户与模型之间的交互过程，其中每个元素包含一个 `role` 和一个 `content`。
    * `role`：代表消息发送者的身份。
        * `"user"`：用户，代表输入端。
        * `"assistant"`：助手/模型，代表输出端。

    * `content`：消息的具体内容。
        * 输入端包含指令和图片占位符。
            * 提示指令 `Prompt`：PaddleOCR-VL-1.5 支持以下提示指令，可根据识别任务设置
                * 文字识别 `"OCR:"`（最通用）
                * 表格识别 `"Table Recognition:"`
                * 公式识别 `"Formula Recognition:"`
                * 图表识别 `"Chart Recognition:"`
                * spotting 识别 `"Spotting:"`
                * 印章识别 `"Seal Recognition:"`
                * 或者根据微调任务自定义提示

            * 图片占位符 `<image>`：在文本数据中标记图片插入的位置。

        * 输出端包含模型预期生成的正确答案，即图片中的表格内容和结构。


* `images`：图像数据列表，存储了对话中涉及到的图片路径（本地路径或 URL）。

值得注意的是，表格结构使用 OTSL 格式表示，相关结构控制符以及具体意义如下所示：

1. `<ecel>`: 结束当前单元格（End Cell）。用于标记单元格的结束。
2. `<fcel>`: 开始一个新的单元格（First Cell）。通常用于表格中的第一个单元格。
3. `<xcel>`: 开始一个新的单元格（eXtended Cell）。用于表格中除第一个单元格外的其他单元格。
4. `<lcel>`: 结束当前行并开始新行（Last Cell）。用于标记一行的结束。
5. `<ucel>`: 合并单元格（Union Cell）。用于表示跨多行或多列的合并单元格。
6. `<nl>`: 换行（New Line）。用于文本中的换行操作。



**自行准备数据**

如果您想要基于自己的数据集进行训练，请参考 [PaddleFormers - 数据集格式文档](https://github.com/PaddlePaddle/PaddleFormers/blob/develop/docs/zh/dataset_format.md#24-%E5%A4%9A%E6%A8%A1%E6%80%81%E6%8C%87%E4%BB%A4%E5%BE%AE%E8%B0%83sft%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F) 准备数据。



## 训练配置
我们针对区域识别示例数据集提供了配置文件，其中的关键训练超参数如下：

* `num_train_epochs=1`：训练的 epoch 数。
* `warmup_ratio=0.01`：线性预热步数，根据任务困难程度调整，此处建议设置成训练步数的 1%。
* `per_device_train_batch_size=8`：每张卡的 batch size 大小，建议根据显存占用情况调整。
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

除了自定义模板的格式，PaddleFormers 还支持自定义模板的多模态数据处理插件，包括：自定义数据增强、自定义多模态 Token 以及替换形式，更多模板和多模态数据处理插件的自定义和使用请参考 [PaddleFormers - template&mm_plugin](https://github.com/PaddlePaddle/PaddleFormers/blob/develop/docs/zh/template_zh.md)。

更多相关参数可在配置文件中查看。

<details>
  <summary><b> 全参配置（点击展开/收起）</b></summary>

```yaml
### data
train_dataset_type: messages
eval_dataset_type: messages
train_dataset_path: ./complex_table/complex_table_train.jsonl
train_dataset_prob: "1.0"
eval_dataset_path: ./complex_table/complex_table_val.jsonl
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
per_device_eval_batch_size: 8
per_device_train_batch_size: 8
num_train_epochs: 1
max_steps: -1
max_estimate_samples: 500
eval_steps: 400
evaluation_strategy: steps
save_steps: 400
save_strategy: steps
logging_steps: 1
gradient_accumulation_steps: 1
logging_dir: ./PaddleOCR-VL-1.5-SFT-Table/visualdl_logs/
output_dir: ./PaddleOCR-VL-1.5-SFT-Table
disable_tqdm: true
eval_accumulation_steps: 16

# train
lr_scheduler_type: cosine
warmup_ratio: 0.01
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
train_dataset_path: ./complex_table/complex_table_train.jsonl
train_dataset_prob: "1.0"
eval_dataset_path: ./complex_table/complex_table_val.jsonl
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
per_device_eval_batch_size: 8
per_device_train_batch_size: 8
num_train_epochs: 1
max_steps: -1
max_estimate_samples: 500
eval_steps: 400
evaluation_strategy: steps
save_steps: 400
save_strategy: steps
logging_steps: 1
gradient_accumulation_steps: 1
logging_dir: ./PaddleOCR-VL-1.5-SFT-Table-lora/visualdl_logs/
output_dir: ./PaddleOCR-VL-1.5-SFT-Table-lora
disable_tqdm: true
eval_accumulation_steps: 16

# train
lr_scheduler_type: cosine
warmup_ratio: 0.01
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
paddleformers-cli train examples/best_practices/PaddleOCR-VL-1.5/paddleocr-vl_full_16k_table_config.yaml \
                        model_name_or_path=PaddlePaddle/PaddleOCR-VL-1.5 \
                        train_dataset_path=./complex_table/complex_table_train.jsonl \
                        eval_dataset_path=./complex_table/complex_table_val.jsonl \
                        pre_alloc_memory=39
```
设置 `pre_alloc_memory` 预分配显存从而减少显存碎片，根据序列长度、批大小和硬件显存调整。

PaddleFormers 默认使用机器上的全部 GPU，可以通过环境变量 `CUDA_VISIBLE_DEVICES` 设置 PaddleFormers 能够使用的 GPU。

可以通过 `visualdl` 对训练过程可视化，使用以下命令行即可启动（下方命令将端口 port 设置为 `8084`，需要根据实际情况设置可用端口）：

```bash
visualdl --logdir ./PaddleOCR-VL-1.5-SFT-Table/visualdl_logs/ --port 8084
```
成功启动后该服务后，在浏览器输入 `ip:port` ，则可以看到训练日志（通过 `hostname -i` 命令可以查看机器的 ip 地址）。

损失曲线如下：

<div align="center">
  <img width="500" alt="table_train_loss" src="https://github.com/forBlank/PaddleFormers/blob/paddleocr_vl_v15_doc/examples/best_practices/PaddleOCR-VL-1.5/assets/table_train_loss.png" />
</div>


从损失曲线中可以看到，在训练起始阶段损失已经处于一个较低的数值（约 0.03），表明 PaddleOCR-VL-1.5 已经具有很强的复杂表格识别能力；随着训练进行，损失并没有显著下降，而是在一定范围内波动，说明模型通过微调获得的性能提升并不会很大。

### SFT LoRA 训练
使用以下命令行即可启动 LoRA 训练：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
paddleformers-cli train examples/best_practices/PaddleOCR-VL-1.5/paddleocr-vl_lora_16k_table_config.yaml \
                        model_name_or_path=PaddlePaddle/PaddleOCR-VL-1.5 \
                        train_dataset_path=./complex_table/complex_table_train.jsonl \
                        eval_dataset_path=./complex_table/complex_table_val.jsonl \
                        pre_alloc_memory=35
```


## 模型结构说明
### SFT 全参
全参训练结束后，模型会保存在 `output_dir=./PaddleOCR-VL-1.5-SFT-Table` 指定路径下，其中包含：

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
LoRA 训练结束后，模型会保存在 `output_dir=./PaddleOCR-VL-1.5-SFT-Table-lora` 指定路径下。相较于 SFT 全参，SFT LoRA 的模型结构会有所不同，其中包含：

* lora_config.json：LoRA 模型配置文件
* peft_model-0000X-of-0000Y.safetensors：LoRA 模型权重文件
* peft_model.safetensors.index.json：LoRA 权重索引文件

使用以下命令行即可合并 LoRA 权重：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
paddleformers-cli export ./examples/config/run_export.yaml \
    model_name_or_path=PaddlePaddle/PaddleOCR-VL-1.5 \
    output_dir=./PaddleOCR-VL-1.5-SFT-Table-lora
```
合并后的完整模型权重保存在 `output_dir=./PaddleOCR-VL-1.5-SFT-Table-lora/export` 路径下。

## 推理
### 单样本推理
Table 测试图像：

<div align="center">
  <img width="500" alt="table_test_example" src="https://github.com/forBlank/PaddleFormers/blob/paddleocr_vl_v15_doc/examples/best_practices/PaddleOCR-VL-1.5/assets/table_test_example.png" />
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

model_path = "./PaddleOCR-VL-1.5-SFT-Table"

model = AutoModelForConditionalGeneration.from_pretrained(
    model_path, convert_from_hf=True,
).eval()

# change the implementation of attention(default is "eager")
model.config._attn_implementation = "flashmask"
model.visual.config._attn_implementation = "flashmask"

processor = AutoProcessor.from_pretrained(model_path)

image_path = "./complex_table/images/d9315261d816_border_2476_2N2P46P1TUIH2B9FQSHF_0.png"
image = Image.open(image_path).convert("RGB")

PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
}
task = "table" # Options: 'ocr' | 'table' | 'chart' | 'formula'

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
# GT = <fcel>Position<fcel>Phone Number<fcel>项目经验<fcel>兴趣爱好<fcel>Ethnicity<fcel>毕业院校<fcel>到岗时间<fcel>自我评价<fcel>Job Description<fcel>语言能力<fcel>Phone Number<fcel>Availability<nl><fcel>项目角色<fcel>兴趣爱好<fcel>Registered Residence<fcel>毕业院校<fcel>语言能力<fcel>职位<fcel>技能专长<fcel>Employment Period<fcel>Language Proficiency<fcel>工作经历<fcel>公司名称<fcel>Job Description<nl><fcel>姓名<fcel>电子邮箱<fcel>Widowed<fcel>$80k-$70k per year<fcel>Project premier<fcel>75 years old<ucel><fcel>Product Manager<fcel>Master<fcel>yong24@example.org<fcel>Male<fcel>813 Samantha Branch, Port Edward, Virginia<nl><fcel>Project Role<fcel>项目描述<fcel>惠派国际公司传媒有限公司<ecel><ecel><ecel><ucel><fcel>2020-01<fcel>630101195903312489<fcel>Master<fcel>昭王时，秦开质于东胡，他智勇双全，东胡王甚信之，因此行动自由，得以了解东胡南部的山川险要、布防情况与军队的活动规律 自此兵力遂强<fcel>natian@example.net<nl><fcel>兴趣爱好<fcel>教育背景<ecel><fcel>Product Manager<fcel>530926194912044335<ecel><ucel><ecel><ecel><ecel><ecel><fcel>PhD<nl><fcel>兴趣爱好<fcel>Major<fcel>887-47-5598<ecel><fcel>532530194409137547<fcel>Master<ucel><fcel>turnerchristopher@example.net<fcel>10k-25k<fcel>项目经理<ecel><fcel>主体, 惠及, 保护, 途径, 监督<nl><fcel>教育背景<fcel>性别<ecel><fcel>吴淑英 18859020112<ecel><fcel>Haskell features a type system with type inference and lazyevaluation. Haskell is a standardized, general-purpose purely functional programming language,with non-strict semantics and strong static typing.<ucel><fcel>Japanese JLPT N1<fcel>1983-12-17<ecel><fcel>$60k-$100k per year<fcel>Immediately<nl><fcel>Major<fcel>民族<fcel>Salvadorian<ecel><fcel>130527194508081214<fcel>5 years experience<ucel><fcel>Ariana Berry (509)652-3851x6189<fcel>130125198909285959<ecel><fcel>Democrat<fcel>法语流利<nl><fcel>身份证号<fcel>Company<fcel>2017-10<fcel>PhD<fcel>内蒙古自治区哈尔滨县南京路<fcel>许芳<fcel>随时<ecel><fcel>数字100信息有限公司职业技术学院<ecel><fcel>Cross-domain professional<fcel>丧偶<nl><fcel>Skills<fcel>政治面貌<ecel><ecel><fcel>西藏自治区佳县马鞍山街T座<fcel>25k-25k<fcel>CPA<fcel>北马里亚纳群岛<fcel>Immediately<fcel>2024.02-2025.12<fcel>20k-35k<fcel>Fluent French<nl>
# Expected Answer = <fcel>Position<fcel>Phone Number<fcel>项目经验<fcel>兴趣爱好<fcel>Ethnicity<fcel>毕业院校<fcel>到岗时间<fcel>自我评价<fcel>Job Description<fcel>语言能力<fcel>Phone Number<fcel>Availability<nl><fcel>项目角色<fcel>兴趣爱好<fcel>Registered Residence<fcel>毕业院校<fcel>语言能力<fcel>职位<fcel>技能专长<fcel>Employment Period<fcel>Language Proficiency<fcel>工作经历<fcel>公司名称<fcel>Job Description<nl><fcel>姓名<fcel>电子邮箱<fcel>Widowed<fcel>$80k-$70k per year<fcel>Project premier<fcel>75 years old<ucel><fcel>Product Manager<fcel>Master<fcel>yong24@example.org<fcel>Male<fcel>813 Samantha Branch, Port Edward, Virginia<nl><fcel>Project Role<fcel>项目描述<fcel>惠派国际公司传媒有限公司<ecel><ecel><ecel><ucel><fcel>2020-01<fcel>630101195903312489<fcel>Master<fcel>昭王时，秦开质于东胡，他智勇双全，东胡王甚信之，因此行动自由，得以了解东胡南部的山川险要、布防情况与军队的活动规律 自此兵力遂强<fcel>natian@example.net<nl><fcel>兴趣爱好<fcel>教育背景<ecel><fcel>Product Manager<fcel>530926194912044335<ecel><ucel><ecel><ecel><ecel><ecel><fcel>PhD<nl><fcel>兴趣爱好<fcel>Major<fcel>887-47-5598<ecel><fcel>532530194409137547<fcel>Master<ucel><fcel>turnerchristopher@example.net<fcel>10k-25k<fcel>项目经理<ecel><fcel>主体, 惠及, 保护, 途径, 监督<nl><fcel>教育背景<fcel>性别<ecel><fcel>吴淑英 18859020112<ecel><fcel>Haskell features a type system with type inference and lazyevaluation. Haskell is a standardized, general-purpose purely functional programming language,with non-strict semantics and strong static typing.<ucel><fcel>Japanese JLPT N1<fcel>1983-12-17<ecel><fcel>$60k-$100k per year<fcel>Immediately<nl><fcel>Major<fcel>民族<fcel>Salvadorian<ecel><fcel>130527194508081214<fcel>5 years experience<ucel><fcel>Ariana Berry (509)652-3851x6189<fcel>130125198909285959<ecel><fcel>Democrat<fcel>法语流利<nl><fcel>身份证号<fcel>Company<fcel>2017-10<fcel>PhD<fcel>内蒙古自治区哈尔滨县南京路<fcel>许芳<fcel>随时<ecel><fcel>数字100信息有限公司职业技术学院<ecel><fcel>Cross-domain professional<fcel>丧偶<nl><fcel>Skills<fcel>政治面貌<ecel><ecel><fcel>西藏自治区佳县马鞍山街T座<fcel>25k-25k<fcel>CPA<fcel>北马里亚纳群岛<fcel>Immediately<fcel>2024.02-2025.12<fcel>20k-35k<fcel>Fluent French<nl>
```
</details>

预期输出为测试图像中的表格内容和结构。

为了直观展示表格结构，我们可以将 OTSL 格式转为 HTML 格式，该功能依赖 PaddleX，使用以下命令行安装 PaddleX 库：

```bash
pip install "paddlex[ocr]"
```
更多安装方式请参考 [PaddleX 安装教程](https://paddlepaddle.github.io/PaddleX/latest/installation/installation.html)。使用以下命令行进行格式转换：

```bash
python otsl2html.py
```

<details>
  <summary><b> OTSL2HTML 转换脚本（点击展开/收起）</b></summary>

```python
from paddlex.inference.pipelines.paddleocr_vl.uilts import convert_otsl_to_html

table_otsl = "<fcel>Position<fcel>Phone Number<fcel>项目经验<fcel>兴趣爱好<fcel>Ethnicity<fcel>毕业院校<fcel>到岗时间<fcel>自我评价<fcel>Job Description<fcel>语言能力<fcel>Phone Number<fcel>Availability<nl><fcel>项目角色<fcel>兴趣爱好<fcel>Registered Residence<fcel>毕业院校<fcel>语言能力<fcel>职位<fcel>技能专长<fcel>Employment Period<fcel>Language Proficiency<fcel>工作经历<fcel>公司名称<fcel>Job Description<nl><fcel>姓名<fcel>电子邮箱<fcel>Widowed<fcel>$80k-$70k per year<fcel>Project premier<fcel>75 years old<ucel><fcel>Product Manager<fcel>Master<fcel>yong24@example.org<fcel>Male<fcel>813 Samantha Branch, Port Edward, Virginia<nl><fcel>Project Role<fcel>项目描述<fcel>惠派国际公司传媒有限公司<ecel><ecel><ecel><ucel><fcel>2020-01<fcel>630101195903312489<fcel>Master<fcel>昭王时，秦开质于东胡，他智勇双全，东胡王甚信之，因此行动自由，得以了解东胡南部的山川险要、布防情况与军队的活动规律 自此兵力遂强<fcel>natian@example.net<nl><fcel>兴趣爱好<fcel>教育背景<ecel><fcel>Product Manager<fcel>530926194912044335<ecel><ucel><ecel><ecel><ecel><ecel><fcel>PhD<nl><fcel>兴趣爱好<fcel>Major<fcel>887-47-5598<ecel><fcel>532530194409137547<fcel>Master<ucel><fcel>turnerchristopher@example.net<fcel>10k-25k<fcel>项目经理<ecel><fcel>主体, 惠及, 保护, 途径, 监督<nl><fcel>教育背景<fcel>性别<ecel><fcel>吴淑英 18859020112<ecel><fcel>Haskell features a type system with type inference and lazyevaluation. Haskell is a standardized, general-purpose purely functional programming language,with non-strict semantics and strong static typing.<ucel><fcel>Japanese JLPT N1<fcel>1983-12-17<ecel><fcel>$60k-$100k per year<fcel>Immediately<nl><fcel>Major<fcel>民族<fcel>Salvadorian<ecel><fcel>130527194508081214<fcel>5 years experience<ucel><fcel>Ariana Berry (509)652-3851x6189<fcel>130125198909285959<ecel><fcel>Democrat<fcel>法语流利<nl><fcel>身份证号<fcel>Company<fcel>2017-10<fcel>PhD<fcel>内蒙古自治区哈尔滨县南京路<fcel>许芳<fcel>随时<ecel><fcel>数字100信息有限公司职业技术学院<ecel><fcel>Cross-domain professional<fcel>丧偶<nl><fcel>Skills<fcel>政治面貌<ecel><ecel><fcel>西藏自治区佳县马鞍山街T座<fcel>25k-25k<fcel>CPA<fcel>北马里亚纳群岛<fcel>Immediately<fcel>2024.02-2025.12<fcel>20k-35k<fcel>Fluent French<nl>"

table_html = convert_otsl_to_html(table_otsl)

style = """
<style>
table {
  border-collapse: collapse;
  width: 100%;
}
td, th {
  border: 1px solid black;
  padding: 8px;
}
</style>
"""
full_html = f"<html><head>{style}</head><body>{table_html}</body></html>"

with open("table_test_html.html", "w", encoding="utf-8") as f:
    f.write(full_html)
print("Save to table_test_html.html")

```
</details>

得到的 HTML 格式表格如下：

<div align="center">
  <img width="500" alt="table_test_html" src="https://github.com/forBlank/PaddleFormers/blob/paddleocr_vl_v15_doc/examples/best_practices/PaddleOCR-VL-1.5/assets/table_test_html.png" />
</div>

<details>
  <summary><b> HTML 源文件（点击展开/收起）</b></summary>

```html
<html><head>
<style>
table {
  border-collapse: collapse;
  width: 100%;
}
td, th {
  border: 1px solid black;
  padding: 8px;
}
</style>
</head><body><table><tr><td>Position</td><td>Phone Number</td><td>项目经验</td><td>兴趣爱好</td><td>Ethnicity</td><td>毕业院校</td><td>到岗时间</td><td>自我评价</td><td>Job Description</td><td>语言能力</td><td>Phone Number</td><td>Availability</td></tr><tr><td>项目角色</td><td>兴趣爱好</td><td>Registered Residence</td><td>毕业院校</td><td>语言能力</td><td>职位</td><td rowspan="7">技能专长</td><td>Employment Period</td><td>Language Proficiency</td><td>工作经历</td><td>公司名称</td><td>Job Description</td></tr><tr><td>姓名</td><td>电子邮箱</td><td>Widowed</td><td>$80k-$70k per year</td><td>Project premier</td><td>75 years old</td><td>Product Manager</td><td>Master</td><td>yong24@example.org</td><td>Male</td><td>813 Samantha Branch, Port Edward, Virginia</td></tr><tr><td>Project Role</td><td>项目描述</td><td>惠派国际公司传媒有限公司</td><td></td><td></td><td></td><td>2020-01</td><td>630101195903312489</td><td>Master</td><td>昭王时，秦开质于东胡，他智勇双全，东胡王甚信之，因此行动自由，得以了解东胡南部的山川险要、布防情况与军队的活动规律 自此兵力遂强</td><td>natian@example.net</td></tr><tr><td>兴趣爱好</td><td>教育背景</td><td></td><td>Product Manager</td><td>530926194912044335</td><td></td><td></td><td></td><td></td><td></td><td>PhD</td></tr><tr><td>兴趣爱好</td><td>Major</td><td>887-47-5598</td><td></td><td>532530194409137547</td><td>Master</td><td>turnerchristopher@example.net</td><td>10k-25k</td><td>项目经理</td><td></td><td>主体, 惠及, 保护, 途径, 监督</td></tr><tr><td>教育背景</td><td>性别</td><td></td><td>吴淑英 18859020112</td><td></td><td>Haskell features a type system with type inference and lazyevaluation. Haskell is a standardized, general-purpose purely functional programming language,with non-strict semantics and strong static typing.</td><td>Japanese JLPT N1</td><td>1983-12-17</td><td></td><td>$60k-$100k per year</td><td>Immediately</td></tr><tr><td>Major</td><td>民族</td><td>Salvadorian</td><td></td><td>130527194508081214</td><td>5 years experience</td><td>Ariana Berry (509)652-3851x6189</td><td>130125198909285959</td><td></td><td>Democrat</td><td>法语流利</td></tr><tr><td>身份证号</td><td>Company</td><td>2017-10</td><td>PhD</td><td>内蒙古自治区哈尔滨县南京路</td><td>许芳</td><td>随时</td><td></td><td>数字100信息有限公司职业技术学院</td><td></td><td>Cross-domain professional</td><td>丧偶</td></tr><tr><td>Skills</td><td>政治面貌</td><td></td><td></td><td>西藏自治区佳县马鞍山街T座</td><td>25k-25k</td><td>CPA</td><td>北马里亚纳群岛</td><td>Immediately</td><td>2024.02-2025.12</td><td>20k-35k</td><td>Fluent French</td></tr></table></body></html>
```
</details>

### 测试集评估
微调前后的模型测试集评估结果如下：

|Model|Avg. NED<br>(structure)|Avg. NED<br>(overall)|Avg. TEDS<br>(structure)|Avg. TEDS<br>(overall)|
|-|-|-|-|-|
|PaddleOCR-VL-1.5|0.9906|0.9683|0.9867|0.9661|
|PaddleOCR-VL-1.5-<br>Table-SFT (Full)|0.9925|0.9749|0.9899|0.96740|
|PaddleOCR-VL-1.5-<br>Table-SFT (LoRA)|0.9909|0.9703|0.9872|0.9687|

### 部署推理
部署 PaddleOCR-VL-1.5 模型，请参考 [PaddleFormers - 模型部署文档](https://github.com/PaddlePaddle/PaddleFormers/blob/develop/docs/zh/deployment_guide.md) 和 [FastDeploy - PaddleOCR-VL-0.9B Best Practices](https://paddlepaddle.github.io/FastDeploy/zh/best_practices/PaddleOCR-VL-0.9B/)



## 注意事项
### 更多硬件上的使用说明
PaddleOCR-VL-1.5 支持基于昆仑芯 P800 和天数智芯 150s 进行微调。本教程选用了规模较大的数据集，在部分硬件上运行时间可能较长，如果希望**快速跑通流程**，或仅需验证**国产硬件环境的兼容性**，建议优先参考我们的 [PaddleFormers - 基于 PaddleOCR-VL 微调实现孟加拉语识别能力](https://github.com/PaddlePaddle/PaddleFormers/tree/develop/examples/best_practices/PaddleOCR-VL)，该教程使用精简数据集，可在短时间内完成微调全链路，助力迅速掌握多硬件适配技巧。
