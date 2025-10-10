if __name__ == "__main__":
    import json
    import numpy as np
    from pprint import pprint
    from paddleformers.datasets2.processor.vision_processor import Qwen2VLVisionProcessor
    from paddleformers.datasets2.processor.auto_processor import Qwen2VLProcessor
    from paddleformers.transformers import AutoTokenizer
    from paddleformers.hparams.data_args import DataArguments
    from paddleformers.datasets2.processor import SupervisedDatasetProcessor

    data_args = DataArguments(
        max_seq_len=16384,
        min_pixels=3136,
        max_pixels=4816896,
        video_min_frames=4,
        video_max_frames=768,
        render_timestamp=True,
    )
    print(data_args)
    
    auto_processor = Qwen2VLProcessor(data_args=data_args)
    # tokenizer = AutoTokenizer.from_pretrained(
    #     "/root/paddlejob/workspace/env/output/peiziliang/baidu/paddle_internal/ernie-4_5-vl-28b-a3b-bf16-tokenizer",
    #     trust_remote_code=True,    
    # )
    # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-2B")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-235B-A22B-Thinking")

    vision_processor = Qwen2VLVisionProcessor(data_args=data_args)
    processor = SupervisedDatasetProcessor(
        auto_processor=auto_processor,
        tokenizer=tokenizer,
        vision_processor=vision_processor,
        data_args=data_args,
    )

    dataset = []
    data1 = {
        "messages": [{"role": "system", "content": "这是system中的内容。"}, 
                    {"role": "user", "content": "<image>你好呀！"}, 
                    {"role": "assistant", "content": "您好，很高兴为您服务！"}, 
                    {"role": "user", "content": "<video>今天天气怎么样？"}, 
                    {"role": "assistant", "content": "<think>\n这个问题我不会\n</think>\n\n不知道啊"}],
        "images": ["/root/paddlejob/workspace/env/output/peiziliang/ERNIE/examples/data/DoclingMatix/44/0.png"],
        "videos": ["/root/paddlejob/workspace/env/output/peiziliang/ERNIE/examples/data/NExTVideo/0008/2403134475.mp4"],
    }
    dataset.append(data1)
    data2 = {
        "messages": [{"role": "system", "content": "这是system中的内容。"}, 
                    {"role": "user", "content": "<image>你好呀！"}, 
                    {"role": "assistant", "content": "您好，很高兴为您服务！"}, 
                    {"role": "user", "content": "<video>今天天气怎么样？"}, 
                    {"role": "assistant", "content": "不知道啊"}],
        "images": ["/root/paddlejob/workspace/env/output/peiziliang/ERNIE/examples/data/DoclingMatix/44/0.png"],
        "videos": ["/root/paddlejob/workspace/env/output/peiziliang/ERNIE/examples/data/NExTVideo/0008/2403134475.mp4"],
    }
    dataset.append(data2)
    print("Input:")
    pprint(dataset)
    print("\nOutput:")
    dataset = processor.preprocess_dataset(dataset)
    pprint(dataset)