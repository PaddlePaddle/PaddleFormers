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

from dataclasses import dataclass
from typing import List

import numpy as np
from paddle.io import IterableDataset

from paddleformers.datasets2.reader.mix_datasets import create_dataset_instance
from paddleformers.datasets2.reader.multi_source_datasets import MultiSourceDataset
from paddleformers.datasets2.data_utils import postprocess_fc_sequence
from paddleformers.transformers import AutoProcessor, AutoTokenizer
from paddleformers.transformers.tokenizer_utils import PretrainedTokenizer
from paddleformers.utils.log import logger


@dataclass
class Sequence:
    """Encapsulated sequence class."""

    token_ids: List[int]
    position_ids: List[int]
    labels: List[int]
    loss_mask: List[int]
    num_examples: int
    images: List[str]
    videos: List[str]
    audios: List[str]


class SFTDataSet(IterableDataset):
    def __init__(self, **dataset_config):

        # parameter init
        self.tokenizer = dataset_config["tokenizer"]
        self.processor = dataset_config["processor"]
        self.max_seq_len = dataset_config["max_seq_len"]
        self.template = dataset_config["template_instance"]
        self.template_backend = dataset_config["template_backend"]
        self.use_template = dataset_config["use_template"]
        self.split_multi_turn = dataset_config["split_multi_turn"]
        self.encode_one_turn = dataset_config["encode_one_turn"]

        # special token
        self.end_of_response = getattr(self.tokenizer.special_tokens_map, "sep_token", "<|end_of_sentence|>")
        self.begin_token = getattr(self.tokenizer.special_tokens_map, "cls_token", "<|begin_of_sentence|>")
        self.newline_token = self.tokenizer.tokenize("\n")
        if isinstance(self.tokenizer, PretrainedTokenizer):
            self.end_of_response_id = self.tokenizer._convert_token_to_id([self.end_of_response])[0]
            self.begin_token_id = self.tokenizer._convert_token_to_id([self.begin_token])[0]
        else:
            self.end_of_response_id = self.tokenizer.convert_tokens_to_ids([self.end_of_response])[0]
            self.begin_token_id = self.tokenizer.convert_tokens_to_ids([self.begin_token])[0]

        # data loader + multisource dataset mix
        multi_source_dataset = MultiSourceDataset(**dataset_config)
        self.mix_datasets = create_dataset_instance(
            dataset_config["mix_strategy"],
            multi_source_dataset,
            **dataset_config,
        )

    def __len__(self):
        return len(self.mix_datasets)

    def __iter__(self):
        for example in self.mix_datasets:
            system = example.get("system", None)
            tools = example.get("tools", None)
            images = example.get("images", [])
            videos = example.get("videos", [])
            audios = example.get("audios", [])
            if self.use_template:
                if self.template_backend == "jinja":
                    if not self.tokenizer.chat_template:
                        self.tokenizer.chat_template = NONE_CHAT_TEMPLATE
                    if self.split_multi_turn:
                        encoded_pairs = postprocess_fc_sequence(self.tokenizer, example)
                    else:
                        encoded_pairs = self.tokenizer.encode_chat_inputs(
                            example, encode_one_turn=self.encode_one_turn
                        )
                else:
                    # 对多模信息做处理，将messages里面的占位符替换
                    messages = self.template.mm_plugin.process_messages(
                        example, images, videos, audios, self.processor
                    )
                    # 套template，转ids
                    encoded_pairs = self.template.encode_multiturn(self.tokenizer, messages, system, tools)
            else:
                encoded_pairs = self.tokenizer.encode_chat_inputs_with_no_template(
                    example, encode_one_turn=self.encode_one_turn
                )
            # 转input_ids, labels
            num_reserved_tokens_for_each_dialog = 1
            num_reserved_tokens_for_each_turn = 8

            cur_len = num_reserved_tokens_for_each_dialog

            turn_index = len(encoded_pairs) - 1

            tokens = []
            loss_mask = []
            while turn_index >= 0:
                tokens_src, tokens_target = encoded_pairs[turn_index]
                if len(tokens_src) + len(tokens_target) > (
                    self.max_seq_len + 1 - cur_len - num_reserved_tokens_for_each_turn
                ):
                    # If the source (src) exceeds length limit, discard this round of conversation data
                    # If the target (tgt) exceeds length limit, truncate it
                    if len(tokens_src) > self.max_seq_len + 1 - cur_len - num_reserved_tokens_for_each_turn:
                        break
                    else:
                        reverse_len = (
                            self.max_seq_len + 1 - cur_len - num_reserved_tokens_for_each_turn - len(tokens_src)
                        )
                        tokens_target = tokens_target[:reverse_len]

                tokens = tokens_src + tokens_target + tokens

                loss_mask = (
                    [0] * (len(tokens_src) - 1) + [example["label"][turn_index]] * (len(tokens_target) + 1) + loss_mask
                )
                assert len(tokens) == len(loss_mask), f"{len(tokens)}-{len(loss_mask)}"

                cur_len = len(tokens)

                turn_index -= 1

            # Not even one turn can be added, so need to do warning and skip this example
            if len(tokens) <= num_reserved_tokens_for_each_dialog + num_reserved_tokens_for_each_turn:
                try:
                    # For print log
                    sub_src = example["messages"][0]["content"].strip()[:50]
                    sub_tgt = example["messages"][-1]["content"].strip()[-50:]
                    if len(tokens) > 0:
                        logger.warning(f"This data is too short: '{{'src':[{sub_src}, ……],'tgt':[……{sub_tgt}]}}'")
                    else:
                        logger.warning(f"This data is too long: '{{'src':[{sub_src}, ……],'tgt':[……{sub_tgt}]}}'")
                except Exception:
                    logger.warning("[SKIP] wrong example")

            if self.begin_token_id is not None and self.end_of_response_id is not None:
                if tokens[0] != self.begin_token_id:
                    tokens = [self.begin_token_id] + tokens
                    loss_mask = [0] + loss_mask

                if len(tokens) > self.max_seq_len:
                    raise RuntimeError(f"token_ids is too long: {len(tokens)}")

                # Add EOS token at the end
                del tokens[-1]
                del loss_mask[-1]
                labels = tokens[1:] + [self.tokenizer.eos_token_id]

                # end_of_response is a special token that indicates the end of the turn.
                # end_token is a special token that indicates the end of the answer.
                labels = [
                    label if label != self.end_of_response_id else self.tokenizer.eos_token_id for label in labels
                ]
            else:
                # labels = tokens[1:] + [self.tokenizer.eos_token_id]
                # tokens = tokens[:-1] + [self.tokenizer.eos_token_id]
                labels = tokens[1:] + [-100]
                if len(tokens) > self.max_seq_len:
                    raise RuntimeError(f"token_ids is too long: {len(tokens)}")

            pos_ids = list(range(len(tokens)))

            if sum(loss_mask) == 0:
                logger.warning(f"[SKIP] all labels set to 0: {example}")
                return None

            assert len(tokens) == len(loss_mask), f"{len(tokens)}-{len(loss_mask)}"
            assert len(tokens) == len(labels), f"{len(tokens)}-{len(labels)}"

            yield Sequence(
                token_ids=tokens,
                position_ids=pos_ids,
                labels=labels,
                loss_mask=loss_mask,
                num_examples=1,
                images=images,
                videos=videos,
                audios=audios,
            )


class SFTPackingDataset(IterableDataset):
    def __init__(self, processed_dataset, **dataset_config):
        self.processed_dataset = processed_dataset
        self.packing = dataset_config["packing"]
        self.greedy_intokens = dataset_config["greedy_intokens"]
        self.max_seq_len = dataset_config["max_seq_len"]
        self.is_valid = dataset_config["is_valid"]

        self.estimate = False
        # The number of valid samples and skipped samples in estimation
        self.unused_samples = 0
        self.used_samples = 0
        # If used_estimate_samples exceeds max_estimate_samples,stop estimating.
        self.used_estimate_samples = 0
        self.max_estimate_samples = 0
        # set max estimate samples
        if not self.is_valid:
            self.max_estimate_samples = len(self.processed_dataset)

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
                batch_sequence, cur_len = [sequence], len(sequence.token_ids)
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
                    if cur_len + len(sequence.token_ids) <= self.max_seq_len:
                        batch_sequence.append(sequence)
                        cur_len += len(sequence.token_ids)
                    else:
                        yield batch_sequence
                        batch_sequence, cur_len = [sequence], len(sequence.token_ids)

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
            if len(sequence.token_ids) <= left_len[max_left_index]:
                generate_packs[max_left_index].append(sequence)
                left_len[max_left_index] -= len(sequence.token_ids)
                if self.estimate:
                    self.used_samples += actual_example_num_list[index]
                index += 1
            else:
                left_index += 1
                left_len[left_index] = self.max_seq_len

        return generate_packs


if __name__ == "__main__":

    model_path = "/root/paddlejob/workspace/env/output/lrl/PaddleFormers/models/Qwen2.5-VL-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path)
    dataset_config = {
        "tokenizer": tokenizer,
        "processor": processor,
        "random_seed": 42,
        "random_shuffle": True,
        "num_replicas": 1,
        "rank": 0,
        "mix_strategy": "concat",
        "num_samples_each_epoch": 6000000,
        "packing": True,
        "greedy_intokens": True,
        "max_seq_len": 8192,
        "encode_one_turn": True,
        "use_template": True,
        "reverse": True,
        "is_valid": False,
    }
    dataset_config.update(
        {
            "task_group": "/root/paddlejob/workspace/env/output/lrl/PaddleFormers/data/vl/experiment.jsonl",
            "task_group_prob": "1.0",
            "sub_dataset_type": "erniekit",
        }
    )

    train_dataset = SFTDataSet(**dataset_config)
    train_packing_dataset = SFTPackingDataset(train_dataset, **dataset_config)

    for item in train_packing_dataset:
        print(item)
        break
