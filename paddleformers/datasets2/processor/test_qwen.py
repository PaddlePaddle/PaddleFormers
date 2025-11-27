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
    from pprint import pprint

    import numpy as np

    from paddleformers.datasets2.processor import SupervisedDatasetProcessor
    from paddleformers.datasets2.processor.encoder import Qwen2VLEncoder
    from paddleformers.datasets2.processor.vision_loader import VisionLoader
    from paddleformers.hparams.data_args import DataArguments
    from paddleformers.transformers import AutoProcessor

    data_args = DataArguments(
        max_seq_len=16384,
        min_pixels=3136,
        max_pixels=4816896,
        video_min_frames=4,
        video_max_frames=768,
        render_timestamp=True,
    )
    print(data_args)

    processor = AutoProcessor.from_pretrained("/root/paddlejob/workspace/env_run/peiziliang/Qwen2.5-VL-3B-Instruct")

    encoder = Qwen2VLEncoder(data_args=data_args)

    vision_loader = VisionLoader(data_args=data_args)
    processor = SupervisedDatasetProcessor(
        encoder=encoder,
        processor=processor,
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
        # "videos": ["https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_video/example_video.mp4"],
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
    print("done")
