# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import annotations

import copy
import json
import os
import sys
import time
from abc import abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Thread
from typing import List

import numpy as np
import paddle
import paddle.incubate.multiprocessing as mp
from paddle.base.framework import in_cinn_mode, in_pir_executor_mode
from paddle.distributed import fleet

from paddlenlp.generation import GenerationConfig, TextIteratorStreamer
from paddlenlp.trainer import PdArgumentParser

from paddleformers.transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PretrainedConfig,
    PretrainedModel,
    PretrainedTokenizer,
)
from paddlenlp.trl import llm_utils
from paddlenlp.utils.import_utils import (
    auto_dynamic_graph_pybind,
    is_paddlenlp_ops_available,
)
from paddlenlp.utils.log import logger
import gc

from gradual_block_quant import apply_block_gptq
from quant_utils import load_quant_model


@dataclass
class PredictorArgument:
    model_name_or_path: str = field(default=None, metadata={"help": "The directory of model."})
    model_prefix: str = field(default="model", metadata={"help": "the prefix name of static model"})
    save_path: str = field(default="./gbq_model", metadata={"help": "the path to save model"})
    src_length: int = field(default=None, metadata={"help": "The max length of source text."})
    min_length: int = field(default=1, metadata={"help": "the min length for decoding."})
    max_length: int = field(default=1024, metadata={"help": "the max length for decoding."})
    top_k: int = field(default=0, metadata={"help": "top_k parameter for generation"})
    top_p: float = field(default=0.7, metadata={"help": "top_p parameter for generation"})
    temperature: float = field(default=0.95, metadata={"help": "temperature parameter for generation"})
    repetition_penalty: float = field(default=1.0, metadata={"help": "repetition penalty parameter for generation"})
    device: str = field(default="gpu", metadata={"help": "Device"})
    dtype: str = field(default=None, metadata={"help": "Model dtype"})
    lora_path: str = field(default=None, metadata={"help": "The directory of LoRA parameters. Default to None"})
    export_precache: bool = field(default=False, metadata={"help": "whether use prefix weight to do infer"})
    prefix_path: str = field(
        default=None, metadata={"help": "The directory of Prefix Tuning parameters. Default to None"}
    )
    decode_strategy: str = field(
        default="sampling",
        metadata={
            "help": "the decoding strategy of generation, which should be one of ['sampling', 'greedy_search', 'beam_search']. Default to sampling"
        },
    )
    use_flash_attention: bool = field(
        default=False,
        metadata={"help": "Whether to use flash attention"},
    )

    mode: str = field(
        default="dynamic", metadata={"help": "the type of predictor, it should be one of [dynamic, static]"}
    )
    inference_model: bool = field(default=False, metadata={"help": "whether use InferenceModel to do generation"})
    quant_type: str = field(
        default="",
        metadata={
            "help": "Quantization type. Supported values: a8w8, a8w8c8, a8w8_fp8, a8w8c8_fp8, weight_only_int4, weight_only_int8"
        },
    )
    avx_model: bool = field(
        default=False, metadata={"help": "whether use AvxModel to do generation when using cpu inference"}
    )
    avx_type: str = field(
        default=None,
        metadata={
            "help": "avx compute type. Supported values: fp16, bf16,fp16_int8\
        fp16: first_token and next_token run in fp16\
        fp16_int8 : first_token run in fp16, next token run in int8"
        },
    )
    avx_cachekv_type: str = field(
        default="fp16",
        metadata={"help": "avx cachekv type. Supported values: fp16,int8"},
    )
    batch_size: int = field(default=1, metadata={"help": "The batch size of data."})
    benchmark: bool = field(
        default=False,
        metadata={
            "help": "If benchmark set as `True`, we will force model decode to max_length, which is helpful to compute throughput. "
        },
    )
    use_fake_parameter: bool = field(default=False, metadata={"help": "use fake parameter, for ptq scales now."})
    block_attn: bool = field(default=False, metadata={"help": "whether use block attention"})
    block_size: int = field(default=64, metadata={"help": "the block size for cache_kvs."})
    cachekv_int8_type: str = field(
        default=None,
        metadata={
            "help": "If cachekv_int8_type set as `dynamic`, cache kv would be quantized to int8 dynamically. If cachekv_int8_type set as `static`, cache kv would be quantized to int8 Statically."
        },
    )

    append_attn: bool = field(default=False, metadata={"help": "whether use append attention"})

    chat_template: str = field(
        default=None,
        metadata={
            "help": "the path of `chat_template.json` file to handle multi-rounds conversation. "
            "If is None(do not set --chat_template argument), it will use the default `chat_template.json`;"
            "If is equal with `model_name_or_path`, it will use the default loading; "
            "If is directory, it will find the `chat_template.json` under the directory; If is file, it will load it."
            "If is none string, it will not use chat_template.json."
        },
    )

    total_max_length: int = field(
        default=4096, metadata={"help": "Super parameter. Maximum sequence length(encoder+decoder)."}
    )
    speculate_method: str = field(
        default=None,
        metadata={
            "help": "speculate method, it should be one of ['None', 'inference_with_reference', 'eagle', 'mtp']"
        },
    )
    speculate_max_draft_token_num: int = field(
        default=1,
        metadata={"help": "the max length of draft tokens for speculate method."},
    )
    speculate_max_ngram_size: int = field(default=1, metadata={"help": "the max ngram size of speculate method."})
    speculate_verify_window: int = field(
        default=2, metadata={"help": "the max length of verify window for speculate method."}
    )
    speculate_max_candidate_len: int = field(default=5, metadata={"help": "the max length of candidate tokens."})
    draft_model_name_or_path: str = field(default=None, metadata={"help": "The directory of eagle or draft model"})
    draft_model_quant_type: str = field(
        default="",
        metadata={"help": "Draft model quantization type. Reserved for future"},
    )
    return_full_hidden_states: bool = field(default=False, metadata={"help": "whether return full hidden_states"})

    mla_use_matrix_absorption: bool = field(default=False, metadata={"help": "implement mla with matrix-absorption."})
    weightonly_group_size: int = field(default=-1, metadata={"help": "the max length of candidate tokens."})
    weight_block_size: List[int] = field(
        default_factory=lambda: [128, 128],
        metadata={"help": "Quantitative granularity of weights. Supported values: [128 128]"},
    )
    moe_quant_type: str = field(
        default="",
        metadata={"help": "Quantization type of moe. Supported values: weight_only_int4, weight_only_int8"},
    )
    output_via_mq: bool = field(
        default=True,
        metadata={"help": "Controls whether the message queue is enabled for output"},
    )
    dynamic_insert: bool = field(default=False, metadata={"help": "whether use dynamic insert"})
    total_request_num: int = field(default=None, metadata={"help": "The total number of request data"})
    lazy_load: bool = field(
        default=False,
        metadata={"help": "Whether to use lazy load"},
    )
    gptq: bool = field(
        default=False,
        metadata={"help": "Whether to use gptq"},
    )
    iq: bool = field(
        default=False,
        metadata={"help": "Whether to use iq"},
    )
    ptq_samples: int = field(
        default=-1,
        metadata={"help": ""},
    )
    offload_data: bool = field(
        default=False,
        metadata={"help": "Whether to offload data"},
    )
    debug: bool = field(
        default=0,
        metadata={"help": "Whether to offload data"},
    )
    use_hessian: bool = field(
        default=0,
        metadata={"help": "Whether to offload data"},
    )
    use_tq: bool = field(
        default=0,
        metadata={"help": "Whether to offload data"},
    )
    use_wint4: bool = field(
        default=0,
        metadata={"help": "Whether to offload data"},
    )
    wint4_all: bool = field(
        default=0,
        metadata={"help": "Whether to offload data"},
    )
    pp_id: bool = field(
        default=0,
        metadata={"help": "Whether to offload data"},
    )
    group_size: int = field(
        default=128,
        metadata={"help": "Whether to offload data"},
    )
    load_quant_path: str = field(
        default=None,
        metadata={"help": "Whether to offload data"},
    )

    def __post_init__(self):
        if self.speculate_method is not None:
            self.append_attn = True
        if self.append_attn:
            self.block_attn = True
        if self.block_attn:
            self.inference_model = True
        assert self.max_length < self.total_max_length, "max_length should smaller than total_max_length."
        if self.src_length is None:
            self.src_length = self.total_max_length - self.max_length
        # update config parameter for inference predictor
        if self.decode_strategy == "greedy_search":
            self.top_p = 0.0
            self.temperature = 1.0
        if self.total_request_num is None:
            self.total_request_num = self.batch_size


@dataclass
class ModelArgument:
    model_type: str = field(
        default=None,
        metadata={"help": "the type of the model, which can be one of ['gpt-3', 'ernie-3.5-se', 'llama-img2txt']"},
    )
    data_file: str = field(default=None, metadata={"help": "data file directory"})
    output_file: str = field(default="output.json", metadata={"help": "predict result file directory"})


def batchfy_text(texts, batch_size):
    batch_texts = []
    batch_start = 0
    while batch_start < len(texts):
        batch_texts += [texts[batch_start : min(batch_start + batch_size, len(texts))]]
        batch_start += batch_size
    return batch_texts


class BasePredictor:
    def __init__(
        self, config: PredictorArgument, tokenizer: PretrainedTokenizer = None, model: PretrainedModel = None
    ):
        if model is not None and hasattr(model, "config"):
            self.model_config = model.config
        else:
            self.model_config = AutoConfig.from_pretrained(config.model_name_or_path)

        self.config: PredictorArgument = config
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, padding_side="left")

        self.tokenizer = tokenizer

        self.return_tensors = "pd"
        self.tensor_parallel_rank, self.tensor_parallel_degree = llm_utils.init_dist_env()
        self.model_config.tensor_parallel_rank, self.model_config.tensor_parallel_degree = (
            self.tensor_parallel_rank,
            self.tensor_parallel_degree,
        )

        try:
            self.generation_config = GenerationConfig.from_pretrained(config.model_name_or_path)
        except:
            logger.warning(
                "Can't find generation config, so it will not use generation_config field in the model config"
            )
            self.generation_config = None

    def _preprocess(self, source, tgt=None):
        # if self.tokenizer.chat_template is not None:
        #     # for str -> List[str] eg. "hello"
        #     # for List[str] -> List[str]  eg. ["hello", "hello new"]
        #     # for List[List[str]] -> List[List[List[str]]]  eg. 历史对话形式,一轮
        #     #             [ [ "Hello, how are you?", "I'm doing great. How can I help you today?"],
        #     #                ["I'd like to show off how chat templating works!"], ]
        #     # for List[Dict] -> List[List[Dict]]  [{'role': 'user', 'content': 'hello'}, {'role': 'assistant', 'content': 'nice'}]
        #     #                                 ->  [[{'role': 'user', 'content': 'hello'}, {'role': 'assistant', 'content': 'nice'}]]
        #     if not isinstance(source, list) or not isinstance(source[0], str):
        #         source = [source]
        #     source = [self.tokenizer.apply_chat_template(sentence, tokenize=False) for sentence in source]
        #     if tgt is not None:
        #         source = [source[0] + tgt[0]]

        tokenized_source = self.tokenizer(
            source,
            max_length=self.config.src_length,
            truncation=True,
            return_attention_mask=True,
            return_tensors=self.return_tensors,
            padding=True,
            # when use chat_template, it should not add special tokens
            # chatglm2 prefix-tokens can not be tokenized into ids
            add_special_tokens=self.tokenizer.chat_template is None,
        )
        return tokenized_source

    @abstractmethod
    def _infer(self, inputs):
        raise NotImplementedError

    def _postprocess(self, predictions, return_tokens=False):
        decoded_predictions = self.tokenizer.batch_decode(
            predictions, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        if return_tokens:
            return decoded_predictions, predictions
        else:
            return decoded_predictions

    def predict(self, input_texts: str | list[str], return_tokens=False):
        tokenized_source = self._preprocess(input_texts)
        # Synchronize the HPU device for the static graph predictor
        # Ensure that configuration data read from the CPU is updated to the HPU device
        paddle.device.synchronize()
        predictions = self._infer(tokenized_source)
        decoded_predictions = self._postprocess(predictions, return_tokens=return_tokens)
        return decoded_predictions


class DygraphPredictor(BasePredictor):
    def __init__(
        self, config: PredictorArgument, tokenizer: PretrainedTokenizer = None, model: PretrainedModel = None, **kwargs
    ):
        super().__init__(config, tokenizer, model)
        self.model = model
        if config.dtype is not None:
            dtype = config.dtype
        else:
            raise ValueError("Please specific the model dtype.")

        if self.model is None:
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_name_or_path,
                use_flash_attention=config.use_flash_attention,
                dtype=dtype,
                convert_from_hf=True,
                tensor_parallel_degree=self.tensor_parallel_degree,
                tensor_parallel_rank=self.tensor_parallel_rank,
            )
        self.model.eval()

    @paddle.no_grad()
    def _infer(self, inputs: dict[str, paddle.Tensor]):
        result = self.model.generate(
            **inputs,
            # max_new_tokens=self.config.max_length,
            # bos_token_id=self.tokenizer.bos_token_id,
            # eos_token_id=llm_utils.get_eos_token_id(self.tokenizer, self.generation_config),
            # pad_token_id=self.tokenizer.pad_token_id,
            # decode_strategy=self.config.decode_strategy,
            # temperature=self.config.temperature,
            # top_k=self.config.top_k,
            # top_p=self.config.top_p,
            # repetition_penalty=self.config.repetition_penalty,
            max_new_tokens=1024,
            temperature=0.1,
            top_p=0.7,
            repetition_penalty=1,
        )
        result = result[0]
        return result


class AutoPredictor:
    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path).`"
        )

    @classmethod
    def create_predictor(
        cls,
        predictor_args: PredictorArgument,
        config: PretrainedConfig,
        model_args: ModelArgument,
        tokenizer: PretrainedTokenizer = None,
        model: PretrainedModel = None,
        **kwargs,
    ):
        """
        Create a predictor

        Args:
            predictor_args (PredictorArgument): The predictor arguments.
            config (PretrainedConfig): The model configuration.
            model_args (ModelArgument): The model arguments.
            tokenizer (PretrainedTokenizer): The tokenizer.
            **kwargs: Additional keyword arguments.
        Returns:
            Predictor: The predictor.
        """
        cache_kvs_shape = None  # used for not block_attn/append_attn
        cache_k_shapes = None  # used for block_attn/append_attn
        cache_v_shapes = None  # used for block_attn/append_attn

        # static or dynamic
        execute_mode = "Dygraph" if predictor_args.mode == "dynamic" else "StaticGraph"

        # infer/ no infer
        inference_mode = ""

        predictor_class_name = execute_mode + inference_mode + "Predictor"

        import_class = sys.modules[__name__]

        # import class
        predictor_class = getattr(import_class, predictor_class_name)

        # instance
        predictor = predictor_class(
            predictor_args,
            tokenizer=tokenizer,
            model=model,
            cache_k_shapes=cache_k_shapes,
            cache_v_shapes=cache_v_shapes,
            cache_kvs_shape=cache_kvs_shape,
            model_args=model_args,
            **kwargs,
        )
        return predictor


def create_predictor(
    predictor_args: PredictorArgument,
    model_args: ModelArgument,
    **kwargs,
):
    paddle.set_device(predictor_args.device)
    paddle.set_default_dtype(predictor_args.dtype)

    from paddlenlp.utils.env import USE_FAST_TOKENIZER

    tokenizer = AutoTokenizer.from_pretrained(
        predictor_args.model_name_or_path
    )

    # init chat_template for tokenizer
    llm_utils.init_chat_template(tokenizer, predictor_args.model_name_or_path, predictor_args.chat_template)

    # # TODO(wj-Mcat): fix llama tokenzier pad_token bug

    config = AutoConfig.from_pretrained(predictor_args.model_name_or_path)

    tensor_parallel_rank, tensor_parallel_degree = llm_utils.init_dist_env()

    model = None

    # model loading
    if False: # predictor_args.inference_model: #---#
        pass #wangna
        # model = AutoInferenceModelForCausalLM.from_pretrained(
        #     predictor_args.model_name_or_path,
        #     config=config,
        #     predictor_args=predictor_args,
        #     model_args=model_args,
        #     dtype=predictor_args.dtype,
        #     tensor_parallel_degree=tensor_parallel_degree,
        #     tensor_parallel_rank=tensor_parallel_rank,
        # )
    else:
        if predictor_args.mode == "dynamic":
            # model import (gpt-3,ernie) or AutoModel
            if model_args.model_type == "gpt-3":
                sys.path.append("./gpt-3")
                from modeling import GPTForCausalLM

                model = GPTForCausalLM.from_pretrained(
                    predictor_args.model_name_or_path,
                    dtype=predictor_args.dtype,
                    tensor_parallel_degree=tensor_parallel_degree,
                    tensor_parallel_rank=tensor_parallel_rank,
                    tensor_parallel_output=False,
                )
            elif model_args.model_type == "ernie-3.5-se":
                sys.path.append("./ernie-3.5-se")
                from modeling import Ernie35ForCausalLM

                tensor_parallel_degree = paddle.distributed.get_world_size()
                tensor_parallel_rank = paddle.distributed.get_rank()
                model = Ernie35ForCausalLM.from_pretrained(
                    predictor_args.model_name_or_path,
                    dtype=predictor_args.dtype,
                    tensor_parallel_degree=tensor_parallel_degree,
                    tensor_parallel_rank=tensor_parallel_rank,
                    tensor_parallel_output=False,
                )
            else:
                with paddle.LazyGuard():
                    with paddle.no_grad():
                        model = AutoModelForCausalLM.from_pretrained(
                            predictor_args.model_name_or_path,
                            dtype=predictor_args.dtype, 
                            convert_from_hf=True, 
                            use_flash_attention=predictor_args.use_flash_attention,
                            tensor_parallel_degree=tensor_parallel_degree,
                            tensor_parallel_rank=tensor_parallel_rank,
                            tensor_parallel_output=False,
                        )
    predictor = AutoPredictor.create_predictor(predictor_args, config, model_args, tokenizer, model=model, **kwargs)

    return predictor


def predict():
    parser = PdArgumentParser((PredictorArgument, ModelArgument))
    predictor_args, model_args = parser.parse_args_into_dataclasses()

    llm_utils.set_triton_cache(predictor_args.model_name_or_path, predictor_args.mode)
    try:
        from paddle.utils import try_import

        try_import("paddlenlp_ops")
    except ImportError:
        logger.warning("paddlenlp_ops does not exist, please install paddlenlp_ops.")
        return
    tensor_parallel_degree = paddle.distributed.get_world_size()
    if tensor_parallel_degree > 1:
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": tensor_parallel_degree,
            "pp_degree": 1,
            "sharding_degree": 1,
        }
        fleet.init(is_collective=True, strategy=strategy)

    predictor = create_predictor(predictor_args, model_args)

    source_texts = []
    target_texts = []
    if model_args.data_file:
        with open(model_args.data_file, "r", encoding="utf-8") as f:
            for line in f:
                example = json.loads(line)
                # src tgt
                if isinstance(example["instruction"], str) or predictor.tokenizer.chat_template is None:
                    if isinstance(example["instruction"], str):
                        source_texts.append(example["instruction"])
                        target_texts.append(example["output"])
                    else:
                        # load multi-rounds dataset
                        source_texts.append(example["instruction"][0])
                        target_texts.append(example["output"][0])
                else:
                    source_texts.append(list(zip(example["instruction"], example["output"])))
                    target_texts.append("")

    else:
        source_texts = [
            "济南燃气结清费用需要带什么资料", "中通快递从辽宁到新疆要多久", "度小满逾期如何协商停催", "顺丰快递是昼夜不停的运吗?"
        ]
        target_texts = ["", "", "", ""]

    batch_source_texts = batchfy_text(source_texts, predictor_args.batch_size)
    batch_target_texts = batchfy_text(target_texts, predictor_args.batch_size)

    apply_block_gptq(predictor.model, predictor, batch_source_texts, batch_target_texts, predictor_args)
    # exit()

    if predictor_args.load_quant_path:
        load_quant_model(predictor.model, predictor_args, None, [])

    if predictor_args.benchmark:
        benchmark(predictor)



def benchmark(predictor, predictor_args, model_args):
    # Just construct a simple benchmark input. We pad input to the src_length.
    test_texts = "你是百度AI，请**参考公开资料提供的信息，回答用户问题**，做到**时效性高，专业权威，客观无偏见**。\n\n### 公开资料说明\n1. 如果不同的公开资料出现矛盾且都符合正常逻辑，务必参考权威性更高的公开资料。如果根据权威性无法区分，请给用户提供多种说法。\n2. 如果用户问题对时间信息比较敏感，结合当前时间和公开资料的发布时间选择合适公开资料。\n\n### 引用说明\n1. 将相关公开资料索引用方括号包裹，置于相关内容后，例如：\"这是相关内容。[1][2]\"。\n\n### 回答要求\n1. 优先满足用户的主要需求，并且从用户问题和公开资料中挖掘用户可能的潜在需求进行满足。\n2. 优先参考公开资料中提供的信息，如果公开资料确实没有用户需求的相关信息，请你说明公开资料没有提及相关内容，并基于自身知识回答。\n3. 如果用户包含负向情绪，如焦虑/不安/困惑/气愤/孤独无助等，请你用更有人情味的风格回答。\n4. 如果用户问题涉及网站访问、平台查询、资源获取、工具使用等需求，并且公开资料中提供了对应的准确链接时,请以\"[网址名称](URL)\"的格式给出。\n\n### 背景信息\n当前时间：2025年09月12日星期五\n当前所在地：山东省济南市\n用户画像：性别: 女;年龄: 中年;手机型号: vivo XFold3;\n用户检索历史: [济南济华燃气王官庄服务站电话: 2025-09-12 16:30:28; 王官庄济华燃气营业厅: 2025-09-12 16:30:21; 燃气结清费用去哪: 2025-09-12 16:28:09; 2024年济南高中录取分数线: 2025-09-12 13:29:08; 2024年高中录取分数线: 2025-09-12 13:28:54; 2024年高中寒假放假时间表: 2025-09-12 13:28:49; 济南385分能上高中吗: 2025-09-12 13:27:40; 385分能上高中吗: 2025-09-12 13:27:26; ABS材质对人体有害吗: 2025-09-12 11:30:55; abs是什么材质?: 2025-09-12 11:30:16; 公租房小区儿童娱乐区域属于配套建设吗: 2025-09-12 09:32:17; 公租房小区没有娱乐设施吗为什么: 2025-09-12 09:30:32; 2026年取消公租房最新通知: 2025-09-12 09:30:07; 公租房小区没有娱乐设施吗: 2025-09-12 09:29:53; 公租房小区没有娱乐设施吗: 2025-09-12 09:28:02; 公租房小区没有娱乐设施合法吗: 2025-09-12 09:26:40; 小区娱乐设施谁安装: 2025-09-12 09:17:26; 监控用漏电保护器还是空气开关: 2025-09-12 08:48:27; 漏电保护器和空气开关有什么区别: 2025-09-12 08:45:56; 秋天吃什么食物最好: 2025-09-12 08:25:17; 红花如意丸的功效与作用: 2025-09-11 10:57:15; 雪莲果是凉性的还是热性的: 2025-09-11 09:09:30; 狗狗不吃饭但精神很好: 2025-09-10 20:26:16; 狗狗不吃饭是怎么回事: 2025-09-10 20:25:04; 狗用脚挠痒痒怎么回事: 2025-09-10 20:24:23; 百香果的功效和作用: 2025-09-10 13:02:05; 劳务合同属于什么合同类型: 2025-09-09 16:28:58; 劳务合同属于行政合同吗: 2025-09-09 16:28:35; 自来水地埋管是什么材料做的: 2025-09-09 14:38:29; 2025年1月灭火器更换新规定: 2025-09-09 09:27:37; 新灭火器第一次换粉是什么时候: 2025-09-09 09:26:43; 口臭是胃火还是肝火: 2025-09-08 19:42:29; ]\n\n### 公开资料\n[1] 标题: 奥德集团有限公司费县分公司办事服务 \n参考特征:内容权威性非常高， 作者权威性非常高， 时效性较高， 发布于2025-05-27\n正文: (一)报装资料、申请受理: 1.开发商及村委集体用户安装:单体楼的建设工程规划许可证复印件、单体楼建筑施工图、小区整体平面图电子版一份。 2.零散居民户安装:即前期社区整体安装燃气时未安装的用户,出示房产证或者购房合同原件。 3.非居民用户:工商用户及小微用户等非居民用户燃气报装资料需提供用气地址的产权资料、营业执照、有效身份证明。\n\n\n[2] 标题: 燃气缴费、维修及相关服务办理程序、线上线下办理渠道、时限、网点设置、服务标准、服务承诺和便民措施 \n参考特征:内容权威性较高， 作者权威性非常高， 时效性较高， 发布于2025-02-19\n正文: 1、用户充值流程: (1)营业厅充值流程: 用户携带燃气卡、本到营业网点柜台→递交燃气卡、本→用户付款→营业员核对信息→系统充值→打印收款收据→递还燃气卡、本、收据→充值完成。 (2)政务大厅充值流程: 用户携带身份证、燃气卡、本到政务大厅→自助叫号机叫号→柜台钱等候叫号→递交燃气卡、本→用户付款→营业员核对信息→系统充值→打印收款收据→递还燃气卡、本、收据→充值完成。 (3)线上充值流程: 微信关注“奥德悦生活”微信公众号→首次登陆绑定用户编号→选择购气量→在线支付→到就近自助写卡机(实时更新,可在公众号内查询)写卡。\n\n\n[3] 标题: 燃气民用销户办事指南 \n参考特征:内容权威性非常高， 作者权威性非常高， 时效性较高， 发布于2025-06-06\n正文: 用户携带户主身份证原件及复印件、天然气用户卡、银行卡复印件(退还预存燃气费),并提交销户申请。\n\n\n[4] 标题: 【办事服务】2025年山东长乐集团民生燃气有限公司用气申请、过户、销户等项目办事服务指南\n参考特征:内容权威性较高， 作者权威性非常高， 时效性较高， 发布于2025-06-03\n正文: 一、申请 单位或是小区统一安装由单位、小区物业办公室或开发公司向燃气公司提出安装申请,填写申请单,并提供小区平面图;散户安装(1)持有效身份证件及现金(如开发商代收,需带天然气配套设施收费票据或证明)(2)签订居民燃气供用气合同。\n\n### 用户问题\n济南燃气结清费用需要带什么资料", 
    
    benchmark_texts = [
        test_texts
    ]

    batch_benchmark_texts = batchfy_text(benchmark_texts, 1)
    print("***********Start Benchmark**********")

    warmup_time = 5
    test_time = 20

    print("***********Start Warmup**********")
    for _ in range(warmup_time):
        for bs, batch_source_text in enumerate(batch_benchmark_texts):
            predictor.predict(batch_source_text)

    print("***********Start Speed Test**********")
    start = time.perf_counter()
    output_tokens = 0
    for _ in range(test_time):
        for bs, batch_source_text in enumerate(batch_benchmark_texts):
            results = predictor.predict(batch_source_text, return_tokens=True)
            if predictor.tensor_parallel_rank == 0:
                output_tokens += sum([len(tokens) for tokens in results[-1]])
    end = time.perf_counter()
    if predictor.tensor_parallel_rank == 0:
        print("Avg Elapse time is: ", (end - start) / test_time)
        print("Output tokens is: ", output_tokens)
        print(
            "Input length is: {}, Output length is: {}, bs is: {}, IPS: {:.3f} tokens/s, QPS: {:.3f} requests/s. ".format(
                16384,
                1024,
                1,
                (output_tokens / (end - start)),
                (1 * test_time / (end - start)),
            )
        )


if __name__ == "__main__":
    predict()
