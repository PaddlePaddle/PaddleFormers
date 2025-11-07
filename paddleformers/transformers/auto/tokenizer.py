# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 Google AI, Google Brain and the HuggingFace Inc. team.
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
import json
import os
import warnings
from typing import Dict, Optional, Union

import transformers as hf
from transformers import AutoConfig, PretrainedConfig
from transformers.dynamic_module_utils import (
    get_class_from_dynamic_module,
    resolve_trust_remote_code,
)
from transformers.modeling_gguf_pytorch_utils import load_gguf_checkpoint
from transformers.models.auto.configuration_auto import (
    config_class_to_model_type,
    replace_list_option_in_docstrings,
)
from transformers.models.auto.tokenization_auto import (
    TOKENIZER_MAPPING,
    TOKENIZER_MAPPING_NAMES,
    get_tokenizer_config,
    tokenizer_class_from_name,
)
from transformers.models.encoder_decoder.configuration_encoder_decoder import (
    EncoderDecoderConfig,
)
from transformers.tokenization_utils_base import TOKENIZER_CONFIG_FILE
from transformers.utils import cached_file

from ...utils.download import DownloadSource, resolve_file_path
from ...utils.log import logger
from ..tokenizer_utils import PaddleTokenizerMixin


def get_paddleformers_tokenizer_config(
    pretrained_model_name_or_path: Union[str, os.PathLike],
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    force_download: bool = False,
    resume_download: Optional[bool] = None,
    proxies: Optional[Dict[str, str]] = None,
    token: Optional[Union[bool, str]] = None,
    revision: Optional[str] = None,
    local_files_only: bool = False,
    subfolder: str = "",
    **kwargs,
):
    """
    Loads the tokenizer configuration from a pretrained model tokenizer configuration.

    Args:
        pretrained_model_name_or_path (`str` or `os.PathLike`):
            This can be either:

            - a string, the *model id* of a pretrained model configuration hosted inside a model repo on
              huggingface.co.
            - a path to a *directory* containing a configuration file saved using the
              [`~PreTrainedTokenizer.save_pretrained`] method, e.g., `./my_model_directory/`.

        cache_dir (`str` or `os.PathLike`, *optional*):
            Path to a directory in which a downloaded pretrained model configuration should be cached if the standard
            cache should not be used.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether or not to force to (re-)download the configuration files and override the cached versions if they
            exist.
        resume_download:
            Deprecated and ignored. All downloads are now resumed by default when possible.
            Will be removed in v5 of Transformers.
        proxies (`Dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
        token (`str` or *bool*, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `huggingface-cli login` (stored in `~/.huggingface`).
        revision (`str`, *optional*, defaults to `"main"`):
            The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
            git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
            identifier allowed by git.
        local_files_only (`bool`, *optional*, defaults to `False`):
            If `True`, will only try to load the tokenizer configuration from local files.
        subfolder (`str`, *optional*, defaults to `""`):
            In case the tokenizer config is located inside a subfolder of the model repo on huggingface.co, you can
            specify the folder name here.

    <Tip>

    Passing `token=True` is required when you want to use a private model.

    </Tip>

    Returns:
        `Dict`: The configuration of the tokenizer.

    Examples:

    ```python
    # Download configuration from huggingface.co and cache.
    tokenizer_config = get_tokenizer_config("google-bert/bert-base-uncased")
    # This model does not have a tokenizer config so the result will be an empty dict.
    tokenizer_config = get_tokenizer_config("FacebookAI/xlm-roberta-base")

    # Save a pretrained tokenizer locally and you can reload its config
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
    tokenizer.save_pretrained("tokenizer-test")
    tokenizer_config = get_tokenizer_config("tokenizer-test")
    ```"""
    download_hub = kwargs.get("download_hub", None)

    resolved_config_file = resolve_file_path(
        pretrained_model_name_or_path,
        TOKENIZER_CONFIG_FILE,
        cache_dir=cache_dir,
        force_download=force_download,
        resume_download=resume_download,
        proxies=proxies,
        token=token,
        revision=revision,
        local_files_only=local_files_only,
        subfolder=subfolder,
        download_hub=download_hub,
    )
    if resolved_config_file is None:
        logger.info("Could not locate the tokenizer configuration file, will try to use the model config instead.")
        return {}
    with open(resolved_config_file, encoding="utf-8") as reader:
        result = json.load(reader)

    return result


def _bind_paddle_mixin_if_available(tokenizer_class):
    """
    Bind the PaddleTokenizerMixin if Paddle is available; otherwise, return the original class.

    Args:
        tokenizer_class: The original tokenizer class.

    Returns:
        The tokenizer class bound with PaddleTokenizerMixin, or the original class.
    """
    return type(tokenizer_class.__name__, (PaddleTokenizerMixin, tokenizer_class), {})


class AutoTokenizer(hf.AutoTokenizer):
    """
    Smart AutoTokenizer that automatically adapts based on available dependencies:

    1. **Multi-source support**: Supports HuggingFace, PaddleFormers, and other download sources
    2. **Conditional Paddle integration**: Automatically detects PaddlePaddle availability
    3. **Fallback compatibility**: Works seamlessly with or without Paddle dependencies
    4. **Enhanced functionality**: Extends HuggingFace's standard tokenizer loading logic

    Features:
    - Automatically binds PaddleTokenizerMixin when PaddlePaddle is available
    - Falls back to pure Transformers mode when PaddlePaddle is not available
    - Maintains full compatibility with all HuggingFace tokenizers
    - Supports custom download sources through environment variables
    """

    @classmethod
    @replace_list_option_in_docstrings(TOKENIZER_MAPPING_NAMES)
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        download_hub = kwargs.get("download_hub", None)
        if download_hub is None:
            download_hub = os.environ.get("DOWNLOAD_SOURCE", "huggingface")
        use_auth_token = kwargs.pop("use_auth_token", None)
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if kwargs.get("token") is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            kwargs["token"] = use_auth_token

        config = kwargs.pop("config", None)
        kwargs["_from_auto"] = True

        use_fast = kwargs.pop("use_fast", True)
        tokenizer_type = kwargs.pop("tokenizer_type", None)
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        gguf_file = kwargs.get("gguf_file")

        # First, let's see whether the tokenizer_type is passed so that we can leverage it
        if tokenizer_type is not None:
            tokenizer_class = None
            tokenizer_class_tuple = TOKENIZER_MAPPING_NAMES.get(tokenizer_type, None)

            if tokenizer_class_tuple is None:
                raise ValueError(
                    f"Passed `tokenizer_type` {tokenizer_type} does not exist. `tokenizer_type` should be one of "
                    f"{', '.join(c for c in TOKENIZER_MAPPING_NAMES)}."
                )

            tokenizer_class_name, tokenizer_fast_class_name = tokenizer_class_tuple

            if use_fast:
                if tokenizer_fast_class_name is not None:
                    tokenizer_class = tokenizer_class_from_name(tokenizer_fast_class_name)
                else:
                    logger.warning(
                        "`use_fast` is set to `True` but the tokenizer class does not have a fast version. "
                        " Falling back to the slow version."
                    )
            if tokenizer_class is None:
                tokenizer_class = tokenizer_class_from_name(tokenizer_class_name)

            if tokenizer_class is None:
                raise ValueError(f"Tokenizer class {tokenizer_class_name} is not currently imported.")

            # Bind PaddleTokenizerMixin
            tokenizer_class = _bind_paddle_mixin_if_available(tokenizer_class)
            return tokenizer_class.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)

        # Next, let's try to use the tokenizer_config file to get the tokenizer class.
        # download tokenizer_config.json file to get tokenizer class name
        if download_hub == DownloadSource.HUGGINGFACE:
            tokenizer_config = get_tokenizer_config(pretrained_model_name_or_path, **kwargs)
            if "_commit_hash" in tokenizer_config:
                kwargs["_commit_hash"] = tokenizer_config["_commit_hash"]
            config_tokenizer_class = tokenizer_config.get("tokenizer_class")
        else:
            try:
                tokenizer_config = get_paddleformers_tokenizer_config(pretrained_model_name_or_path, **kwargs)
            except Exception as e:
                if any(
                    keyword in str(e).lower()
                    for keyword in ["not exist", "not found", "entrynotfound", "notexist", "does not appear"]
                ):
                    hf_link = f"https://huggingface.co/{pretrained_model_name_or_path}"
                    modelscope_link = f"https://modelscope.cn/models/{pretrained_model_name_or_path}"
                    encoded_model_name = pretrained_model_name_or_path.replace("/", "%2F")
                    aistudio_link = f"https://aistudio.baidu.com/modelsoverview?sortBy=weight&q={encoded_model_name}"

                    raise ValueError(
                        f"Unable to find {TOKENIZER_CONFIG_FILE} in the model repository '{pretrained_model_name_or_path}'. Please check:\n"
                        f"The model repository ID is correct for your chosen source:\n"
                        f"   - Hugging Face Hub: {hf_link}\n"
                        f"   - ModelScope: {modelscope_link}\n"
                        f"   - AI Studio: {aistudio_link}\n"
                        f"Note: The repository ID may differ between ModelScope, AI Studio, and Hugging Face Hub.\n"
                        f"You are currently using the download source: {download_hub}. Please check the repository ID on the official website."
                    ) from None
                else:
                    raise
            config_tokenizer_class = tokenizer_config.get("tokenizer_class")

        tokenizer_auto_map = None
        if "auto_map" in tokenizer_config:
            if isinstance(tokenizer_config["auto_map"], (tuple, list)):
                # Legacy format for dynamic tokenizers
                tokenizer_auto_map = tokenizer_config["auto_map"]
            else:
                tokenizer_auto_map = tokenizer_config["auto_map"].get("AutoTokenizer", None)

        # If that did not work, let's try to use the config.
        if config_tokenizer_class is None:
            if not isinstance(config, PretrainedConfig):
                if gguf_file:
                    gguf_path = cached_file(pretrained_model_name_or_path, gguf_file, **kwargs)
                    config_dict = load_gguf_checkpoint(gguf_path, return_tensors=False)["config"]
                    config = AutoConfig.for_model(**config_dict)
                else:
                    config = AutoConfig.from_pretrained(
                        pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
                    )
            config_tokenizer_class = config.tokenizer_class
            if hasattr(config, "auto_map") and "AutoTokenizer" in config.auto_map:
                tokenizer_auto_map = config.auto_map["AutoTokenizer"]

        has_remote_code = tokenizer_auto_map is not None
        has_local_code = type(config) in TOKENIZER_MAPPING or (
            config_tokenizer_class is not None
            and (
                tokenizer_class_from_name(config_tokenizer_class) is not None
                or tokenizer_class_from_name(config_tokenizer_class + "Fast") is not None
            )
        )

        if has_remote_code:
            if use_fast and tokenizer_auto_map[1] is not None:
                class_ref = tokenizer_auto_map[1]
            else:
                class_ref = tokenizer_auto_map[0]
            if "--" in class_ref:
                upstream_repo = class_ref.split("--")[0]
            else:
                upstream_repo = None
            trust_remote_code = resolve_trust_remote_code(
                trust_remote_code, pretrained_model_name_or_path, has_local_code, has_remote_code, upstream_repo
            )

        if has_remote_code and trust_remote_code:
            tokenizer_class = get_class_from_dynamic_module(class_ref, pretrained_model_name_or_path, **kwargs)
            _ = kwargs.pop("code_revision", None)
            tokenizer_class.register_for_auto_class()

            # Bind PaddleTokenizerMixin
            tokenizer_class = _bind_paddle_mixin_if_available(tokenizer_class)
            return tokenizer_class.from_pretrained(
                pretrained_model_name_or_path, *inputs, trust_remote_code=trust_remote_code, **kwargs
            )
        elif config_tokenizer_class is not None:

            tokenizer_class = None
            if use_fast and not config_tokenizer_class.endswith("Fast"):
                tokenizer_class_candidate = f"{config_tokenizer_class}Fast"
                tokenizer_class = tokenizer_class_from_name(tokenizer_class_candidate)
            if tokenizer_class is None:
                tokenizer_class_candidate = config_tokenizer_class
                tokenizer_class = tokenizer_class_from_name(tokenizer_class_candidate)
            if tokenizer_class is None:
                raise ValueError(
                    f"Tokenizer class {tokenizer_class_candidate} does not exist or is not currently imported."
                )

            # Bind PaddleTokenizerMixin
            tokenizer_class = _bind_paddle_mixin_if_available(tokenizer_class)
            return tokenizer_class.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)

        # Otherwise we have to be creative.
        # if model is an encoder decoder, the encoder tokenizer class is used by default
        if isinstance(config, EncoderDecoderConfig):
            if type(config.decoder) is not type(config.encoder):  # noqa: E721
                logger.warning(
                    f"The encoder model config class: {config.encoder.__class__} is different from the decoder model "
                    f"config class: {config.decoder.__class__}. It is not recommended to use the "
                    "`AutoTokenizer.from_pretrained()` method in this case. Please use the encoder and decoder "
                    "specific tokenizer classes."
                )
            config = config.encoder

        model_type = config_class_to_model_type(type(config).__name__)
        if model_type is not None:
            tokenizer_class_py, tokenizer_class_fast = TOKENIZER_MAPPING[type(config)]

            if tokenizer_class_fast and (use_fast or tokenizer_class_py is None):
                # Bind PaddleTokenizerMixin
                tokenizer_class_fast = _bind_paddle_mixin_if_available(tokenizer_class_fast)
                return tokenizer_class_fast.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
            else:
                if tokenizer_class_py is not None:
                    # Bind PaddleTokenizerMixin
                    tokenizer_class_py = _bind_paddle_mixin_if_available(tokenizer_class_py)
                    return tokenizer_class_py.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
                else:
                    raise ValueError(
                        "This tokenizer cannot be instantiated. Please make sure you have `sentencepiece` installed "
                        "in order to use this tokenizer."
                    )

        raise ValueError(
            f"Unrecognized configuration class {config.__class__} to build an AutoTokenizer.\n"
            f"Model type should be one of {', '.join(c.__name__ for c in TOKENIZER_MAPPING)}."
        )


__all__ = ["AutoTokenizer", "TOKENIZER_MAPPING"]
