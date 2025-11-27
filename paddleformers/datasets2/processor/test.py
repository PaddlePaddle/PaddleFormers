# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if __name__ == "__main__":
    import json
    import os
    from pprint import pprint

    import numpy as np
    import paddle

    from paddleformers.datasets2.processor import SupervisedDatasetProcessor
    from paddleformers.datasets2.processor.encoder import Ernie45VLEncoder
    from paddleformers.datasets2.processor.vision_loader import VisionLoader
    from paddleformers.hparams.data_args import DataArguments
    from paddleformers.transformers import AutoTokenizer, Ernie4_5_VLTokenizer

    # from paddleformers.transformers.ernie4_5_moe_vl import Ernie4_5_VLMoeProcessor

    data_args = DataArguments(
        max_seq_len=16384,
        min_pixels=3136,
        max_pixels=4816896,
        video_frames_sample="middle",
        render_timestamp=True,
    )
    print(data_args)

    encoder = Ernie45VLEncoder(data_args=data_args)

    tokenizer = AutoTokenizer.from_pretrained(
        "/root/paddlejob/workspace/env_run/peiziliang/baidu/ernie/ERNIE_4p5_VL_28B_A3B_Thinking",
    )

    # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")

    vision_loader = VisionLoader(data_args=data_args)
    processor = SupervisedDatasetProcessor(
        encoder=encoder,
        processor=tokenizer,
        vision_loader=vision_loader,
        data_args=data_args,
    )

    dataset = []
    data1 = {
        "messages": [
            {"role": "system", "content": "这是system中的内容。"},
            {"role": "user", "content": "<image>你好呀！"},
            {"role": "assistant", "content": "您好，很高兴为您服务！"},
            {"role": "user", "content": "<video>今天天气怎么样？"},
            {"role": "assistant", "content": "<think>\n这个问题我不会\n</think>\n\n不知道啊"},
        ],
        "images": ["/root/paddlejob/workspace/env_run/peiziliang/ERNIE/examples/data/DoclingMatix/44/0.png"],
        "videos": ["/root/paddlejob/workspace/env_run/peiziliang/ERNIE/examples/data/NExTVideo/0008/2403134475.mp4"],
    }
    dataset.append(data1)
    data2 = {
        "messages": [
            {"role": "system", "content": "这是system中的内容。"},
            {"role": "user", "content": "<image>你好呀！"},
            {"role": "assistant", "content": "您好，很高兴为您服务！"},
            {"role": "user", "content": "<video>今天天气怎么样？"},
            {"role": "assistant", "content": "不知道啊"},
        ],
        "images": ["/root/paddlejob/workspace/env_run/peiziliang/ERNIE/examples/data/DoclingMatix/44/0.png"],
        "videos": ["/root/paddlejob/workspace/env_run/peiziliang/ERNIE/examples/data/NExTVideo/0008/2403134475.mp4"],
    }
    dataset.append(data2)
    print("Input:")
    pprint(dataset)
    print("\nOutput:")
    dataset = processor.preprocess_dataset(dataset[0])
    # pprint(dataset)
    print(tokenizer.encode(tokenizer.cls_token))
    print(dataset["input_ids"][:8])
    print(tokenizer.decode(dataset["input_ids"][:8]))

    with open("/root/paddlejob/workspace/env_run/peiziliang/sft_vl-train_process_messages_format1.jsonl", "r") as f:
        for input_data, line in zip([data1, data2], f):
            output_data = json.loads(line)
            data = processor.preprocess_dataset(input_data)
            for key in data.keys():
                if isinstance(data[key], np.ndarray):
                    data[key] = data[key].tolist()
                elif isinstance(data[key], paddle.Tensor):
                    data[key] = data[key].tolist()
                print("key: {}\tis equal: {}".format(key, data[key] == output_data[key]))
    exit()
    # with open("/root/paddlejob/workspace/env_run/peiziliang/sft_vl-train_process_messages_format1.jsonl", "w") as f:
    #     for input_data in [data1, data2]:
    #         data = processor.preprocess_dataset(input_data)
    #         for key in data.keys():
    #             if isinstance(data[key], np.ndarray):
    #                 data[key] = data[key].tolist()
    #             elif isinstance(data[key], paddle.Tensor):
    #                 data[key] = data[key].tolist()
    #         f.write(json.dumps(data, ensure_ascii=False) + '\n')
    with open(
        "/root/paddlejob/workspace/env/output/peiziliang/ERNIE/examples/data/sft_vl-train_process_messages_format.jsonl",
        "r",
    ) as f:
        input_datas = f.readlines()

    with open("/root/paddlejob/workspace/env/output/peiziliang/processed_data.jsonl", "r") as f:
        for line in f:
            try:
                data = json.loads(line)
                input_ids = data["input_ids"]
                text = tokenizer.decode(input_ids[-3:-1]).lstrip(">")
                print(text)
                ids = int(text) - 1
                # if ids != 16:
                #     continue
                model_input = processor.preprocess_dataset([json.loads(input_datas[ids])])[0]

                # print("model_input['input_ids'][:20]: ", model_input["input_ids"][:20])
                # print("model_input['token_type_ids'][:20]: ", model_input["token_type_ids"][:20])
                # print("data['input_ids'][:20]: ", data["input_ids"][:20])
                # print("data['token_type_ids'][:20]: ", data["token_type_ids"][:20])
                for key, value in model_input.items():
                    # print(key, " is eqaul: ", value[:200] == data[key][:200])
                    print(key, " is eqaul: ", np.array_equal(value[:200], np.array(data[key][:200])))

                #     # print(len(value))
                #     # print(value[2740:2750])
                #     print(tokenizer.decode(value[-2000:]))
                #     print("-------------------")
                #     # print(len(data[key]))
                #     # print(data[key][-2000:])
                #     print(tokenizer.decode(data[key][-2000:]))
                #     # for i in range(len(value)):
                #     #     if value[i] != data[key][i]:
                #     #         print(f'{key} index {i}: {value[i]}')
                #     #         break
                #     break
                # print("model_input['grid_thw']:", model_input["grid_thw"])
                # print("data['grid_thw']:", data["grid_thw"])
                # print("model_input['images']:", model_input["images"].shape)
                # images = np.array(data["images"])
                # print("data['images']:", images.shape)
                # print("equal: ", np.array_equal(model_input["images"], images))
                # print("data['labels'] == model_input['labels']:", data['labels'][-100:] == model_input['labels'][-100:])
                # print("data['position_ids']: ", data["position_ids"][2000:])
                # print("model_input['position_ids']: ", model_input["position_ids"].tolist()[2000:])
                print("data['image_type_ids]: ", data["image_type_ids"])
                print(
                    "len(model_input['token_type_ids']) == len(model_input['input_ids']): ",
                    len(model_input["token_type_ids"]) == len(model_input["input_ids"]),
                )
                print("-------------------")
                break
            except Exception as e:
                print(e)
                # break
