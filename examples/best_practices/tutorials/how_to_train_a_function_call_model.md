# 1. 任务简介

Function Calling 是一种让大模型能够调用外部函数的机制，当模型遇到自身知识外的内容，需要通过工具查询时，会输出结构化的调用信息，引导使用者调用工具进行上下文的补充。这种机制使得大模型不再局限于自身的知识库和计算能力，而是能够借助外部函数实现更广泛、更复杂的业务功能。

<div align="center">
  <img width="1013" height="293" alt="function_call_demo" src="https://github.com/user-attachments/assets/dd6e9c15-5d16-4de5-b24f-a985ddd7ed81" />
</div>

大模型的 Function Calling 能力通常用于需要外部数据支持或复杂操作的场景，例如查询数据库以获取实时数据、调用 API 以接入第三方服务或执行多步骤任务以完成复杂的业务流程。在这些场景中，Function Calling 机制能够充分发挥大模型的语言理解和生成能力，同时借助外部函数的强大功能，实现更高效、更准确的业务处理。

本文通过使用一组 Function Call 数据集和[Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)模型演示如何使用[PaddleFormers](https://github.com/PaddlePaddle/PaddleFormers)训练一个支持 Function Call 能力的模型。

本次实验旨在说明如何基于 PaddleFormers 训练一个支持 Function Calling 能力的模型，验证模型 Function Calling 能力的性能和效果。整个实验使用1张40G A100显卡进行训练，耗时约5分钟。

# 2. 任务准备

## 2.1. 模型准备

PaddleFormers 通过在训练配置文件中指定字段`model_name_or_path`来设置所用的模型。启动训练时如果本地没有该模型的缓存，那么 PaddleFormers 会自动下载模型并加载使用。

您也可以将对应的字段指定成您的本地路径，来加载已经下载好的模型。

需特别注意的是，若要训练一个具备 Function Call 能力的模型，必须确保所准备的模型的 tokenizer 和对话模板均支持 Function Call 功能。否则，训练效果很可能无法达到预期。（您可以通过模型的`tokenizer_config.json`文件进行查看，如果您不确定哪些模型可用，我们推荐使用 qwen3系列的模型进行训练。）

## 2.2. 数据准备

我们这里使用工具调用数据（[new_tool_call_dataset.csv](https://huggingface.co/datasets/trishonc/tool-call/blob/main/new_tool_call_dataset.csv)）来演示。这是一个工具调用数据集，它会教模型在有需要时，调用合适的工具来满足用户的需求。

为了快速演示，我们使用如下脚本处理数据集，取其中前200条样本作为示例，构建一个200条的 demo 数据集（如果想要构建自己的数据集，请参考以下文档：[数据集格式说明](../../../docs/zh/dataset_format.md)）：

```python
import csv, json

with open("new_tool_call_dataset.csv", "r", encoding="utf-8") as f:
    datas = csv.DictReader(f)

    demo_datas = []
    for data in datas:
        messages = []
        messages.append(
            {
                "role": "user",
                "content": data["user"],
            }
        )
        if data["tool_call"] == "yes":
            messages.append(
                {
                    "role": "assistant",
                    "content": "<think>\n</think>",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": json.loads(data["response"]),
                        }
                    ]
                }
            )
        else:
            messages.append(
                {
                    "role": "assistant",
                    "content": "<think>\n</think>" + data["response"],
                }
            )
        if int(data["tools_count"]) > 0:
            tools = json.loads(data["tools"])
            new_tools = []
            for tool in tools:
                new_tools.append(
                    {
                        "type": "function",
                        "function": tool,
                    }
                )
            demo_datas.append(
            {
                "messages": messages,
                "tools": new_tools
            }
        )
        else:
            demo_datas.append(
                {
                    "messages": messages
                }
            )
with open("new_tool_call_dataset_demo.jsonl", "w") as f:
    for data in demo_datas[:200]:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")
```

处理后的数据中每一行为一条 messages 格式的数据，数据示例如下：

```json
{"messages": [{"role": "user", "content": "I need to generate an invoice for a customer named John Doe. He bought 2 apples for $1 each and 3 oranges for $2 each."}, {"role": "assistant", "content": "<think>\n</think>", "tool_calls": [{"type": "function", "function": {"name": "generate_invoice", "arguments": {"customer_name": "John Doe", "items": [{"name": "apple", "quantity": 2, "price": 1}, {"name": "orange", "quantity": 3, "price": 2}]}}}]}], "tools": [{"type": "function", "function": {"name": "generate_invoice", "description": "Generate an invoice for a customer", "parameters": {"type": "object", "properties": {"customer_name": {"type": "string", "description": "Name of the customer"}, "items": {"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string", "description": "Name of the item"}, "quantity": {"type": "integer", "description": "Quantity of the item"}, "price": {"type": "number", "description": "Price of the item"}}, "required": ["name", "quantity", "price"]}, "description": "List of items in the invoice"}}, "required": ["customer_name", "items"]}}}]}
```

数据仅包含一个名为`messages`的字段和一个名为`tools`的字段。

字段`messages`是一个由一系列对话内容构成的列表，列表中的每条对话内容均包含三个子字段，分别为`role`、`content`和`tool_calls`。在`role`字段中，可以填入`system`、`user`、`assistant`或`tool`，分别表示“系统设定”、“用户输入”、“模型回复”和“工具调用结果”，需特别说明的是，`system`这一角色标识仅能出现在对话的首轮。而`content`字段则用于填入具体的对话内容，特别的，在“模型回复”中，可以在`tool_calls`字段中增加工具调用的内容。

字段`tools`是一个由一系列可调用的工具构成的列表，每个工具可以根据需要定义类型、名称、参数等信息。

## 2.3. 训练前效果测试

在正式开始模型训练前，我们从数据集（[new_tool_call_dataset.csv](https://huggingface.co/datasets/trishonc/tool-call/blob/main/new_tool_call_dataset.csv)）中未使用的数据中随机挑选一条负样本，测试模型的回复。

使用如下脚本进行模型的预测，我们在这条数据中提供了7种不同的工具，其中一种名为`search_recipe`的工具可以完成用户的需求：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-0.6B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto"
)

# prepare the model input
prompt = "I have some chicken, broccoli, and cheese. Can you find me a recipe?"
messages = [
    {"role": "user", "content": prompt}
]
tools = [
    {"type": "function", "function": {"name": "calculate_distance", "description": "Calculate the distance between two locations", "parameters": {"type": "object", "properties": {"origin": {"type": "string", "description": "The origin location"}, "destination": {"type": "string", "description": "The destination location"}}, "required": ["origin", "destination"]}}},
    {"type": "function", "function": {"name": "search_movies", "description": "Search for movies based on title, genre, or release year", "parameters": {"type": "object", "properties": {"title": {"type": "string", "description": "The title of the movie"}, "genre": {"type": "string", "description": "The genre of the movie"}, "release_year": {"type": "integer", "description": "The release year of the movie"}}}}},
    {"type": "function", "function": {"name": "get_movie_details", "description": "Get details of a movie", "parameters": {"type": "object", "properties": {"movie_title": {"type": "string", "description": "The title of the movie"}, "year": {"type": "integer", "description": "The year the movie was released"}}, "required": ["movie_title"]}}},
    {"type": "function", "function": {"name": "search_recipe", "description": "Search for a recipe based on ingredients", "parameters": {"type": "object", "properties": {"ingredients": {"type": "array", "items": {"type": "string"}, "description": "The list of ingredients to search for"}, "cuisine": {"type": "string", "description": "The cuisine type to filter the recipes"}}, "required": ["ingredients"]}}},
    {"type": "function", "function": {"name": "calculate_area", "description": "Calculate the area of a shape", "parameters": {"type": "object", "properties": {"shape": {"type": "string", "description": "The type of shape (e.g., rectangle, circle)"}, "dimensions": {"type": "object", "properties": {"length": {"type": "number", "description": "The length of the shape"}, "width": {"type": "number", "description": "The width of the shape"}}, "required": ["length", "width"]}}, "required": ["shape", "dimensions"]}}},
    {"type": "function", "function": {"name": "calculate_age", "description": "Calculate the age based on birthdate", "parameters": {"type": "object", "properties": {"birthdate": {"type": "string", "format": "date", "description": "The birthdate in yyyy-mm-dd format"}}, "required": ["birthdate"]}}},
    {"type": "function", "function": {"name": "calculate_tip", "description": "Calculate the tip amount", "parameters": {"type": "object", "properties": {"bill_amount": {"type": "number", "description": "The total bill amount"}, "tip_percentage": {"type": "number", "description": "The percentage tip to calculate"}}, "required": ["bill_amount", "tip_percentage"]}}},
    ]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    tools=tools,
    enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    do_sample=False,
    max_new_tokens=32768
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")

print("content:", content)
```

以这条数据为例：

```python
prompt = "I have some chicken, broccoli, and cheese. Can you find me a recipe?"
tools = [
    {"type": "function", "function": {"name": "calculate_distance", "description": "Calculate the distance between two locations", "parameters": {"type": "object", "properties": {"origin": {"type": "string", "description": "The origin location"}, "destination": {"type": "string", "description": "The destination location"}}, "required": ["origin", "destination"]}}},
    {"type": "function", "function": {"name": "search_movies", "description": "Search for movies based on title, genre, or release year", "parameters": {"type": "object", "properties": {"title": {"type": "string", "description": "The title of the movie"}, "genre": {"type": "string", "description": "The genre of the movie"}, "release_year": {"type": "integer", "description": "The release year of the movie"}}}}},
    {"type": "function", "function": {"name": "get_movie_details", "description": "Get details of a movie", "parameters": {"type": "object", "properties": {"movie_title": {"type": "string", "description": "The title of the movie"}, "year": {"type": "integer", "description": "The year the movie was released"}}, "required": ["movie_title"]}}},
    {"type": "function", "function": {"name": "search_recipe", "description": "Search for a recipe based on ingredients", "parameters": {"type": "object", "properties": {"ingredients": {"type": "array", "items": {"type": "string"}, "description": "The list of ingredients to search for"}, "cuisine": {"type": "string", "description": "The cuisine type to filter the recipes"}}, "required": ["ingredients"]}}},
    {"type": "function", "function": {"name": "calculate_area", "description": "Calculate the area of a shape", "parameters": {"type": "object", "properties": {"shape": {"type": "string", "description": "The type of shape (e.g., rectangle, circle)"}, "dimensions": {"type": "object", "properties": {"length": {"type": "number", "description": "The length of the shape"}, "width": {"type": "number", "description": "The width of the shape"}}, "required": ["length", "width"]}}, "required": ["shape", "dimensions"]}}},
    {"type": "function", "function": {"name": "calculate_age", "description": "Calculate the age based on birthdate", "parameters": {"type": "object", "properties": {"birthdate": {"type": "string", "format": "date", "description": "The birthdate in yyyy-mm-dd format"}}, "required": ["birthdate"]}}},
    {"type": "function", "function": {"name": "calculate_tip", "description": "Calculate the tip amount", "parameters": {"type": "object", "properties": {"bill_amount": {"type": "number", "description": "The total bill amount"}, "tip_percentage": {"type": "number", "description": "The percentage tip to calculate"}}, "required": ["bill_amount", "tip_percentage"]}}}
    ]
```

在训练前，模型回复如下：

```text
content: Sure! Here's a recipe using chicken, broccoli, and cheese:

**Cheese and Broccoli Chicken Recipe**

- **Ingredients**:
  - Chicken breast
  - Broccoli
  - Cheese

- **Instructions**:
  1. Preheat your oven to 375°F (190°C).
  2. Cook the chicken breast until it's tender.
  3. Add the broccoli and cheese to the oven and cook for 15-20 minutes.
  4. Serve warm.

```

模型并没有调用工具，而是产生了幻觉，直接回答了用户的问题。

显然这个回答并不符合我们的需求，因此我们的目标是通过微调，使得模型可以调用我们提供的工具获取相关信息后进行回复。

# 3. 模型训练

## 3.1. 训练超参数配置

训练超参数可参考[full.yaml](../../config/sft/full.yaml)配置，以下是本次实验使用的 yaml：

```yaml
### data
train_dataset_type: messages
eval_dataset_type: messages
train_dataset_path: new_tool_call_dataset_demo.jsonl
train_dataset_prob: "1.0"
eval_dataset_path: new_tool_call_dataset_demo.jsonl
eval_dataset_prob: "1.0"
max_seq_len: 8192
packing: false
mix_strategy: concat
template_backend: custom
template: qwen3

### model
model_name_or_path: Qwen/Qwen3-0.6B
_attn_implementation: flashmask

### finetuning
# base
stage: SFT
fine_tuning: full
seed: 23
do_train: true
do_eval: true
per_device_eval_batch_size: 1
per_device_train_batch_size: 1
num_train_epochs: 5
max_steps: -1
eval_steps: 1000
evaluation_strategy: steps
save_steps: 1000
save_strategy: steps
logging_steps: 1
gradient_accumulation_steps: 16
logging_dir: ./vdl_log
output_dir: ./checkpoints/qwen0.6-sft-full
disable_tqdm: true
eval_accumulation_steps: 16

# train
warmup_steps: 6
learning_rate: 1.0e-5

# performance
tensor_model_parallel_size: 1
pipeline_model_parallel_size: 1
sharding: stage2
recompute: true
bf16: true
fp16_opt_level: O2
```

其中有几个比较重要的参数，在此简单说明：

* max_seq_len：最大数据长度，我们需要保证数据长度小于此值，以防止数据被截断。建议在训练前估计数据集中数据长度的范围，防止大部分数据被截断从而影响训练效果。
* num_train_epochs：训练的 epoch 数。推荐设置为2-3，本次实验由于训练集数据较少，选择训练5个 epoch 以确保训练效果。
* per_device_train_batch_size：每张卡的 batch size 大小。推荐设置为1，可根据显存占用情况增加。
* gradient_accumulation_steps：梯度累积的步数。推荐参考以下算式设置：`每次梯度更新时的批大小 = 训练总卡数 * per_device_train_batch_size * gradient_accumulation_steps = 16 或 32`
* warmup_steps：学习率的预热步数，一般设置为总步数的10%。

## 3.2. 通过命令行训练

仅需一行代码即可完成模型训练：

```shell
paddleformers-cli train function_call.yaml
```

## 3.3. 训练曲线观察

在训练过程中，我们可以利用 visualdl 工具来查看模型的 loss 曲线：

```shell
# 指定端口和地址，以及日志目录（需要和前面训练配置文件中的目录对应上）
visualdl --logdir ./vdl_log/ --port 8080 --host 0.0.0.0
```

本次训练的 loss 曲线如下：

<div align="center">
  <img width="431" height="337" alt="function_call_vdl_log" src="https://github.com/user-attachments/assets/53705e66-8d65-4c06-95ac-4c520c8d957c" />
</div>

可以看到，模型的训练 loss 在不断稳定下降。

# 4. 效果评估

完成模型训练后，我们再次以这条数据为例，测试模型的回复：

```python
prompt = "I have some chicken, broccoli, and cheese. Can you find me a recipe?"
tools = [
    {"type": "function", "function": {"name": "calculate_distance", "description": "Calculate the distance between two locations", "parameters": {"type": "object", "properties": {"origin": {"type": "string", "description": "The origin location"}, "destination": {"type": "string", "description": "The destination location"}}, "required": ["origin", "destination"]}}},
    {"type": "function", "function": {"name": "search_movies", "description": "Search for movies based on title, genre, or release year", "parameters": {"type": "object", "properties": {"title": {"type": "string", "description": "The title of the movie"}, "genre": {"type": "string", "description": "The genre of the movie"}, "release_year": {"type": "integer", "description": "The release year of the movie"}}}}},
    {"type": "function", "function": {"name": "get_movie_details", "description": "Get details of a movie", "parameters": {"type": "object", "properties": {"movie_title": {"type": "string", "description": "The title of the movie"}, "year": {"type": "integer", "description": "The year the movie was released"}}, "required": ["movie_title"]}}},
    {"type": "function", "function": {"name": "search_recipe", "description": "Search for a recipe based on ingredients", "parameters": {"type": "object", "properties": {"ingredients": {"type": "array", "items": {"type": "string"}, "description": "The list of ingredients to search for"}, "cuisine": {"type": "string", "description": "The cuisine type to filter the recipes"}}, "required": ["ingredients"]}}},
    {"type": "function", "function": {"name": "calculate_area", "description": "Calculate the area of a shape", "parameters": {"type": "object", "properties": {"shape": {"type": "string", "description": "The type of shape (e.g., rectangle, circle)"}, "dimensions": {"type": "object", "properties": {"length": {"type": "number", "description": "The length of the shape"}, "width": {"type": "number", "description": "The width of the shape"}}, "required": ["length", "width"]}}, "required": ["shape", "dimensions"]}}},
    {"type": "function", "function": {"name": "calculate_age", "description": "Calculate the age based on birthdate", "parameters": {"type": "object", "properties": {"birthdate": {"type": "string", "format": "date", "description": "The birthdate in yyyy-mm-dd format"}}, "required": ["birthdate"]}}},
    {"type": "function", "function": {"name": "calculate_tip", "description": "Calculate the tip amount", "parameters": {"type": "object", "properties": {"bill_amount": {"type": "number", "description": "The total bill amount"}, "tip_percentage": {"type": "number", "description": "The percentage tip to calculate"}}, "required": ["bill_amount", "tip_percentage"]}}}
    ]
```

模型回复如下：

```text
content: <tool_call>
{"name": "search_recipe", "arguments": {"ingredients": ["chicken", "broccoli", "cheese"]}}
</tool_call>
```

从模型回复的部分可以看到，相比于训练前的模型没有调用合适的工具，训练后的模型调用了正确的工具`search_recipe`来完成用户的需求。（注意：模型只能给出需要调用的工具名称和调用参数，工具的具体执行步骤需要自行设计对应工具完成，此处仅提供模型的输出示例，对于工具的执行部分不做展开说明。）

# 5. 模型部署

您可以参考[基于 FastDeploy / vLLM 部署模型](../../../docs/zh/deployment_guide.md)，将训练完成的模型使用 vLLM / FastDeploy 等工具进行部署使用
