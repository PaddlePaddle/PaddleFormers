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

import numpy as np
from paddle.io import IterableDataset

from paddleformers.datasets2.processor import SupervisedDatasetProcessor
from paddleformers.datasets2.processor.auto_processor import Qwen2VLProcessor
from paddleformers.datasets2.processor.vision_processor import Qwen2VLVisionProcessor
from paddleformers.datasets2.reader.mix_datasets import (
    MultiSourceDataset,
    create_dataset_instance,
)
from paddleformers.hparams.data_args import DataArguments
from paddleformers.transformers import AutoTokenizer

from paddleformers.datasets2.template.template import get_template_and_fix_tokenizer

from transformers import AutoProcessor

class SFTDataSet(IterableDataset):
    def __init__(self, **dataset_config):

        self.tokenizer = dataset_config["tokenizer"]
        # self.reader = dataset_config["data_reader"]
        # self.processor = dataset_config["data_processor"]

        # data loader
        multi_source_dataset = MultiSourceDataset(**dataset_config)

        # data mix
        self.mix_datasets = create_dataset_instance(
            dataset_config["mix_strategy"],
            multi_source_dataset,
            **dataset_config,
        )

        # debug
        for item in self.mix_datasets:
            print(item)
            break

        data_args = DataArguments(
            template="ernie",
            train_on_prompt=False,
            tool_format=None,
            default_system=None,
            enable_thinking=True,
        )

        # 得到register的template
        self.template = get_template_and_fix_tokenizer(dataset_config["tokenizer"], data_args)

        # data_args = DataArguments(
        #     max_seq_len=16384,
        #     min_pixels=3136,
        #     max_pixels=4816896,
        #     video_min_frames=4,
        #     video_max_frames=768,
        #     render_timestamp=True,
        # )
        # auto_processor = Qwen2VLProcessor(data_args=data_args)
        # vision_processor = Qwen2VLVisionProcessor(data_args=data_args)
        # self.processor = SupervisedDatasetProcessor(
        #     auto_processor=auto_processor,
        #     tokenizer=dataset_config["tokenizer"],
        #     vision_processor=vision_processor,
        #     data_args=data_args,
        # )

    def __len__(self):
        return self.mix_datasets.__len__()

    def __iter__(self):

        for item in self.mix_datasets:
            # res = self.processor.preprocess_dataset(item)

            # import pdb
            # pdb.set_trace()


            # 使用self.processor处理多模输入，得到拼接后的结果
            images = item['images']
            videos = []
            audios = []
            try:
                self.processor = AutoProcessor.from_pretrained(
                    '/root/paddlejob/workspace/env/output/lrl/PaddleFormers/models/Qwen2.5-VL-3B-Instruct',
                    use_fast=True,
                )
            except ValueError:  # try another one
                self.processor = AutoProcessor.from_pretrained(
                    '/root/paddlejob/workspace/env/output/lrl/PaddleFormers/models/Qwen2.5-VL-3B-Instruct',
                    use_fast=False,
                )
            except Exception as e:
                logger.info(f"Failed to load processor: {e}.")
                processor = None
            
            messages = self.template.mm_plugin.process_messages(item["messages"], images, videos, audios, self.processor)

            input_ids, labels = self.template.mm_plugin.process_token_ids(
                [], [], images, videos, audios, self.tokenizer, self.processor
            )
            
            # 套template，转ids
            system = None
            tools = None
            encoded_pairs = self.template.encode_multiturn(self.tokenizer, messages, system, tools)

            import pdb
            pdb.set_trace()
            
            total_length = len(input_ids) + (1 if self.template.efficient_eos else 0)

            # debug
            print(res)

            yield res


class SFTPackingDataset(IterableDataset):
    def __init__(self, processed_dataset, **dataset_config):
        self.processed_dataset = processed_dataset
        self.packing = dataset_config["packing"]
        self.greedy_intokens = dataset_config["greedy_intokens"]

        self.estimate = False
        # The number of valid samples and skipped samples in estimation
        self.unused_samples = 0
        self.used_samples = 0
        # If used_estimate_samples exceeds max_estimate_samples,stop estimating.
        self.used_estimate_samples = 0
        self.max_estimate_samples = 0
        # set max estimate samples
        # if not self.is_valid:
        #     self.max_estimate_samples = len(self.mix_datasets)

        self.max_seq_len = dataset_config["max_seq_len"]

    def __iter__(self):

        dataset_iterator = iter(self.processed_dataset)

        if not self.packing:
            for _ in range(len(self.processed_dataset)):
                example = next(dataset_iterator)
                actual_example_num = 1
                sequence = example
                # unused_samples and used_samples are used to calculate skip_samples and actual_train_samples
                if sequence is None:
                    if self.estimate:
                        self.unused_samples += actual_example_num
                    continue
                if self.estimate:
                    self.used_samples += actual_example_num
                batch_sequence, cur_len = [sequence], len(sequence["input_ids"])
                yield batch_sequence

                if self.estimate:
                    self.used_estimate_samples += actual_example_num
                    if self.used_estimate_samples >= self.max_estimate_samples:
                        self.used_estimate_samples = 0
                        # Set flag to False and yield empty list to signal the end of estimation
                        self.estimate = False
                        yield []
            if len(batch_sequence) > 0:
                yield batch_sequence
        else:
            if not self.greedy_intokens:
                # base
                for _ in range(len(self.processed_dataset)):
                    example = next(dataset_iterator)
                    actual_example_num = 1
                    sequence = example
                    if sequence is None:
                        if self.estimate:
                            self.unused_samples += actual_example_num
                        continue
                    if self.estimate:
                        self.used_samples += actual_example_num
                    if cur_len + len(sequence["input_ids"]) <= self.max_seq_len:
                        batch_sequence.append(sequence)
                        cur_len += len(sequence["input_ids"])
                    else:
                        yield batch_sequence
                        batch_sequence, cur_len = [sequence], len(sequence["input_ids"])

                    if self.estimate:
                        self.used_estimate_samples += actual_example_num
                        if self.used_estimate_samples >= self.max_estimate_samples:
                            # Yield left batch sequence before estimation ends
                            if len(batch_sequence) > 0:
                                yield batch_sequence
                            self.used_estimate_samples = 0
                            # Set flag to False and yield empty list to signal the end of estimation
                            self.estimate = False
                            yield []
                if len(batch_sequence) > 0:
                    yield batch_sequence
            else:
                # Pseudo multiple rounds + group greedy intokens.
                buffer_size = 500
                examples = []
                actual_example_num_list = []
                i = 0
                for _ in range(len(self.processed_dataset)):
                    example = next(dataset_iterator)
                    actual_example_num = 1
                    if i < buffer_size:
                        examples.append(example)
                        actual_example_num_list.append(actual_example_num)
                        i += 1
                    else:
                        # Running greedy strategy in examples.
                        generate_packs = self._generate_greedy_packs(examples, actual_example_num_list)
                        for pack in generate_packs:
                            if len(pack) > 0:
                                yield pack
                        examples = [example]
                        i = 1

                    if self.estimate:
                        self.used_estimate_samples += actual_example_num
                        # Stop estimation if the number of samples used in estimation is larger than max_estimate_samples
                        if self.used_estimate_samples >= self.max_estimate_samples:
                            # Yield left packs before estimation ends
                            if len(examples) > 0:
                                generate_packs = self._generate_greedy_packs(examples, actual_example_num_list)
                                for pack in generate_packs:
                                    if len(pack) > 0:
                                        yield pack
                            # Set flag to False and yield empty list to signal the end of estimation
                            self.estimate = False
                            yield []

                if len(examples) > 0:
                    generate_packs = self._generate_greedy_packs(examples, actual_example_num_list)
                    for pack in generate_packs:
                        if len(pack) > 0:
                            yield pack

    def _generate_greedy_packs(self, examples, actual_example_num_list):
        """Generate packed sequences using greedy strategy.

        Args:
            examples: List of examples to pack.
            actual_example_num_list: List of example counts.

        Returns:
            list: List of packed sequences.
        """

        left_len = np.zeros([len(examples)]) - 1
        left_len[0] = self.max_seq_len  # At the beginning, only the first pack is valid.
        generate_packs = [[] for i in range(len(examples))]
        index = 0
        left_index = 0

        while index < len(examples):
            sequence = examples[index]
            if sequence is None:
                if self.estimate:
                    self.unused_samples += actual_example_num_list[index]
                index += 1
                continue

            max_left_index = left_len.argmax()
            # Put the current sequence into the largest left space valid pack.
            if len(sequence["input_ids"]) <= left_len[max_left_index]:
                generate_packs[max_left_index].append(sequence)
                left_len[max_left_index] -= len(sequence["input_ids"])
                if self.estimate:
                    self.used_samples += actual_example_num_list[index]
                index += 1
            else:
                left_index += 1
                left_len[left_index] = self.max_seq_len

        return generate_packs


if __name__ == "__main__":
    # Load tokenizer & dataset
    tokenizer = AutoTokenizer.from_pretrained("/root/paddlejob/workspace/env/output/lrl/PaddleFormers/models/Qwen2.5-VL-3B-Instruct")

    dataset_config = {
        "tokenizer": tokenizer,
        "max_seq_len": 8192,
        "random_seed": 42,
        "num_replicas": 1,
        "rank": 0,
        "num_samples_each_epoch": 6000000,
        "random_shuffle": True,
        "greedy_intokens": True,
        "packing": True,
        "mix_strategy": "concat",
        "encode_one_turn": True,
        "use_template": True,
        "reverse": True,
        "task_group": "/root/paddlejob/workspace/env/output/lrl/PaddleFormers/data/vl/sft_vl-train_demo1.jsonl",
        "task_group_prob": "1.0",
        "sub_dataset_type": "erniekit",
    }

    train_dataset = SFTDataSet(**dataset_config)
    train_packing_dataset = SFTPackingDataset(train_dataset, **dataset_config)

    for item in train_packing_dataset:
        print(item)
        break
