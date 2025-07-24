# copyright (c) 2023 paddlepaddle authors. all rights reserved.
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
""" Generation configuration class and utilities."""

import copy
import json
import os
import warnings
from typing import Any, Dict, Optional, Union

from huggingface_hub import hf_hub_download
from paddle.common_ops_import import convert_dtype

from paddleformers import __version__

from ..transformers.configuration_utils import PretrainedConfig
from ..utils import GENERATION_CONFIG_NAME
from ..utils.download import resolve_file_path
from ..utils.downloader import hf_file_exists
from ..utils.log import logger

DEFAULT_MAX_NEW_TOKENS = 20


def resolve_hf_generation_config_path(repo_id: str, cache_dir: str, subfolder=None) -> str:
    """resolve config file from hf hub

    Args:
        repo_id (str): the repo name from huggingface hub
        cache_dir (str): the cachedir
        subfolder (str, optional) An optional value corresponding to a folder inside the repo.

    Returns:
        str: the downloaded config file
    """
    if hf_file_exists(repo_id=repo_id, filename=GENERATION_CONFIG_NAME, subfolder=subfolder):
        file_name = GENERATION_CONFIG_NAME
    else:
        raise ValueError(f"can not find the paddle/pytorch config file from: https://huggingface.co/{repo_id}")

    return hf_hub_download(
        repo_id=repo_id,
        filename=file_name,
        cache_dir=cache_dir,
        subfolder=subfolder,
        library_name="PaddleNLP",
        library_version=__version__,
    )


class GenerationConfig:
    r"""
    Arg:
        > Parameters that control the length of the output
            max_length (int, optional): The maximum length of the sequence to
                be generated. Default to 20.
            min_length (int, optional): The minimum length of the sequence to
                be generated. Default to 0.
            decode_strategy (str, optional): The decoding strategy in generation.
                Currently, there are three decoding strategies supported:
                "greedy_search", "sampling" and "beam_search". Default to
                "greedy_search".
            temperature (float, optional): The value used to module the next
                token probabilities in the "sampling" strategy. Default to 1.0,
                which means no effect.
            top_k (int, optional): The number of highest probability tokens to
                keep for top-k-filtering in the "sampling" strategy. Default to
                0, which means no effect.
            top_p (float, optional): The cumulative probability for
                top-p-filtering in the "sampling" strategy. The value should
                satisfy :math:`0 <= top\_p < 1`. Default to 1.0, which means no
                effect.
            repetition_penalty (float, optional):
                The parameter for repetition penalty. 1.0 means no penalty. See `this paper
                <https://arxiv.org/pdf/1909.05858.pdf>`__ for more details. Defaults to 1.0.
            num_beams (int, optional): The number of beams in the "beam_search"
                strategy. Default to 1.
            num_beam_groups (int, optional):
                Number of groups to divide `num_beams` into in order to use DIVERSE
                BEAM SEARCH. See `this paper <https://arxiv.org/pdf/1610.02424.pdf>`__
                for more details. Default to 1.
            length_penalty (float, optional): The exponential penalty to the
                sequence length in the "beam_search" strategy. The larger this
                param is, the more that the model would generate shorter
                sequences. Default to 0.0, which means no penalty.
            early_stopping (bool, optional): Whether to stop searching in the
                "beam_search" strategy when at least `num_beams` sentences are
                finished per batch or not. Default to False.
            bos_token_id (int, optional): The id of the `bos_token`. Default to
                None.
            eos_token_id (int, optional): The id of the `eos_token`. Default to
                None.
            pad_token_id (int, optional): The id of the `pad_token`. Default to
                None.
            decoder_start_token_id (int, optional): The start token id for
                encoder-decoder models. Default to None.
            forced_bos_token_id (int, optional): The id of the token to force as
                the first generated token. Usually use for multilingual models.
                Default to None.
            forced_eos_token_id (int, optional): The id of the token to force as
                the last generated token. Default to None.
            num_return_sequences (int, optional): The number of returned
                sequences for each sequence in the batch. Default to 1.
            diversity_rate (float, optional): If num_beam_groups is 1, this is the
                diversity_rate for Diverse Siblings Search. See
                `this paper https://arxiv.org/abs/1611.08562`__ for more details.
                If not, this is the diversity_rate for DIVERSE BEAM SEARCH.
            use_cache: (bool, optional): Whether to use the model cache to
                speed up decoding. Default to True.
            use_fast: (bool, optional): Whether to use fast entry of model
                for FastGeneration. Default to False.
            use_fp16_decoding: (bool, optional): Whether to use fp16 for decoding.
                Only works when fast entry is available. Default to False.
            trunc_input: (bool, optional): Whether to truncate the inputs from
                output sequences . Default to True.
            model_kwargs (dict): It can be used to specify additional kwargs
                passed to the model.
    """

    def _get_generation_mode(self):
        if hasattr(self, "num_beams") and self.num_beams == 1:
            if hasattr(self, "do_sample") and self.do_sample is True:
                generation_mode = "sampling"
            else:
                generation_mode = "greedy_search"
        else:
            generation_mode = "beam_search"

        return generation_mode

    def __init__(self, **kwargs):
        # Parameters that control the length of the output
        self.max_new_tokens = kwargs.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)

        if "min_new_token" in kwargs:
            logger.warning("<min_new_token> field is deprecated. Please use <min_new_tokens> instead.")
            kwargs["min_new_tokens"] = kwargs.pop("min_new_token")

        self.min_new_tokens = kwargs.pop("min_new_tokens", 0)
        self.max_length = kwargs.pop("max_length", 0)
        self.min_length = kwargs.pop("min_length", 0)
        self.early_stopping = kwargs.pop("early_stopping", False)
        self.trunc_input = kwargs.pop("trunc_input", True)

        # Parameters for manipulation of the model output logits
        self.diversity_rate = kwargs.pop("diversity_rate", 0.0)
        self.temperature = kwargs.pop("temperature", 1.0)
        self.top_k = kwargs.pop("top_k", 50)
        self.top_p = kwargs.pop("top_p", 1.0)
        self.repetition_penalty = kwargs.pop("repetition_penalty", 1.0)
        self.length_penalty = kwargs.pop("length_penalty", 1.0)
        self.no_repeat_ngram_size = kwargs.pop("no_repeat_ngram_size", None)
        self.forced_bos_token_id = kwargs.pop("forced_bos_token_id", None)
        self.forced_eos_token_id = kwargs.pop("forced_eos_token_id", None)
        self.num_beams = kwargs.pop("num_beams", 1)
        self.num_beam_groups = kwargs.pop("num_beam_groups", 1)
        self.use_cache = kwargs.pop("use_cache", True)

        # Parameters that define the output variables of `generate`
        self.num_return_sequences = kwargs.pop("num_return_sequences", 1)

        # Special tokens that can be used at generation time
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)

        # Generation parameters exclusive to encoder-decoder models
        self.use_fast = kwargs.pop("use_fast", False)
        self.use_fp16_decoding = kwargs.pop("use_fp16_decoding", False)
        self.fast_ptq_sampling = kwargs.pop("fast_ptq_sampling", False)
        self.decoder_start_token_id = kwargs.pop("decoder_start_token_id", None)
        self._from_model_config = kwargs.pop("_from_model_config", False)
        self.paddleformers_version = kwargs.pop("paddleformers_version", __version__)

        # Additional attributes without default values
        if not self._from_model_config:
            # we don't want to copy values from the model config if we're initializing a `GenerationConfig` from a
            # model's default configuration file
            for key, value in kwargs.items():
                try:
                    setattr(self, key, value)
                except AttributeError as err:
                    logger.error(f"Can't set {key} with value {value} for {self}")
                    raise err

        # Parameters that control the generation strategy used
        if "decode_strategy" in kwargs:
            self.decode_strategy = kwargs.pop("decode_strategy")
        else:
            self.decode_strategy = self._get_generation_mode()

        # Validate the values of the attributes
        self.validate(is_init=True)

    def __eq__(self, other):
        if not isinstance(other, GenerationConfig):
            return False

        self_dict = self.__dict__.copy()
        other_dict = other.__dict__.copy()
        # ignore metadata
        for metadata_field in ["_from_model_config", "paddleformers_version"]:
            self_dict.pop(metadata_field, None)
            other_dict.pop(metadata_field, None)
        return self_dict == other_dict

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

    def validate(self, is_init=False):
        """
        Validates the values of the attributes of the [`GenerationConfig`] instance. Raises exceptions in the presence
        of parameterization that can be detected as incorrect from the configuration instance alone.

        Note that some parameters are best validated at generate runtime, as they may depend on other inputs and/or the
        model, such as parameters related to the generation length.
        """

        # Validation of individual attributes
        if self.early_stopping not in {True, False, "never"}:
            raise ValueError(f"`early_stopping` must be a boolean or 'never', but is {self.early_stopping}.")

        # Validation of attribute relations:
        fix_location = ""
        if is_init:
            fix_location = (
                " This was detected when initializing the generation config instance, which means the corresponding "
                "file may hold incorrect parameterization and should be fixed."
            )

        # 1. detect sampling-only parameterization when not in sampling mode
        if self.decode_strategy == "greedy_search":
            greedy_wrong_parameter_msg = (
                "using greedy search strategy. However, `{flag_name}` is set to `{flag_value}` -- this flag is only "
                'used in sample-based generation modes. You should set `decode_strategy="greedy_search" ` or unset `{flag_name}`.'
                + fix_location
            )
            if self.temperature != 1.0:
                warnings.warn(
                    greedy_wrong_parameter_msg.format(flag_name="temperature", flag_value=self.temperature),
                    UserWarning,
                )
            if self.top_p != 1.0:
                warnings.warn(
                    greedy_wrong_parameter_msg.format(flag_name="top_p", flag_value=self.top_p),
                    UserWarning,
                )

        # 2. detect beam-only parameterization when not in beam mode
        if self.decode_strategy != "beam_search":
            single_beam_wrong_parameter_msg = (
                "`num_beams` is set to 1. However, `{flag_name}` is set to `{flag_value}` -- this flag is only used "
                "in beam-based generation modes. You should set `num_beams>1` or unset `{flag_name}`." + fix_location
            )
            if self.early_stopping is not False:
                warnings.warn(
                    single_beam_wrong_parameter_msg.format(flag_name="early_stopping", flag_value=self.early_stopping),
                    UserWarning,
                )
            if self.num_beam_groups != 1:
                warnings.warn(
                    single_beam_wrong_parameter_msg.format(
                        flag_name="num_beam_groups", flag_value=self.num_beam_groups
                    ),
                    UserWarning,
                )
            if self.length_penalty != 1.0:
                warnings.warn(
                    single_beam_wrong_parameter_msg.format(flag_name="length_penalty", flag_value=self.length_penalty),
                    UserWarning,
                )

        # 4. check `num_return_sequences`
        if self.num_return_sequences != 1:
            if self.decode_strategy == "greedy_search":
                raise ValueError(
                    "Greedy methods without beam search do not support `num_return_sequences` different than 1 "
                    f"(got {self.num_return_sequences})."
                )

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        config_file_name: Optional[Union[str, os.PathLike]] = None,
        **kwargs,
    ):
        r"""
        Save a generation configuration object to the directory `save_directory`, so that it can be re-loaded using the
        [`~GenerationConfig.from_pretrained`] class method.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the configuration JSON file will be saved (will be created if it does not exist).
            config_file_name (`str` or `os.PathLike`, *optional*, defaults to `"generation_config.json"`):
                Name of the generation configuration JSON file to be saved in `save_directory`.
        """

        # At save time, validate the instance -- if any warning/exception is thrown, we refuse to save the instance
        try:
            with warnings.catch_warnings(record=True) as caught_warnings:
                self.validate()
            for w in caught_warnings:
                raise ValueError(w.message)
        except ValueError as exc:
            warnings.warn(
                "The generation config instance is invalid -- `.validate()` throws warnings and/or exceptions. "
                "Fix these issues to save the configuration. This warning will be raised to an exception."
                "\n\nThrown during validation:\n" + str(exc),
                UserWarning,
            )
            return

        config_file_name = config_file_name if config_file_name is not None else GENERATION_CONFIG_NAME

        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        os.makedirs(save_directory, exist_ok=True)

        output_config_file = os.path.join(save_directory, config_file_name)

        self.to_json_file(output_config_file, use_diff=True)
        logger.info(f"Configuration saved in {output_config_file}")

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        from_hf_hub: bool = False,
        from_aistudio: bool = False,
        from_modelscope: bool = False,
        config_file_name: Optional[Union[str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        force_download: bool = False,
        **kwargs,
    ) -> "GenerationConfig":
        r"""
        Instantiate a [`GenerationConfig`] from a generation configuration file.

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                This can be either:

                - a string, the *model id* of a pretrained model configuration hosted inside a model repo on
                   bos server. Valid model ids can be located at the root-level, like `bert-base-uncased`, or
                  namespaced under a user or organization name, like `dbmdz/bert-base-german-cased`.
                - a path to a *directory* containing a configuration file saved using the
                  [`~PretrainedConfig.save_pretrained`] method, e.g., `./my_model_directory/`.
                - a path or url to a saved configuration JSON *file*, e.g., `./my_model_directory/configuration.json`.
            from_hf_hub (bool, *optional*):
                load config from huggingface hub: https://huggingface.co/models
            cache_dir (`str` or `os.PathLike`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force to (re-)download the configuration files and override the cached versions if
                they exist.
            return_unused_kwargs (`bool`, *optional*, defaults to `False`):
                If `False`, then this function returns just the final configuration object.

                If `True`, then this functions returns a `Tuple(config, unused_kwargs)` where *unused_kwargs* is a
                dictionary consisting of the key/value pairs whose keys are not configuration attributes: i.e., the
                part of `kwargs` which has not been used to update `config` and is otherwise ignored.
            kwargs (`Dict[str, Any]`, *optional*):
                The values in kwargs of any keys which are configuration attributes will be used to override the loaded
                values. Behavior concerning key/value pairs whose keys are *not* configuration attributes is controlled
                by the `return_unused_kwargs` keyword parameter.

        Returns:
            [`GenerationConfig`]: The configuration object instantiated from this pretrained model.

        Examples:

        ```python
        >>> from paddleformers.transformers import GenerationConfig

        >>> generation_config = GenerationConfig.from_pretrained("gpt2")

        >>> # E.g. config was saved using *save_pretrained('./test/saved_model/')*
        >>> generation_config.save_pretrained("./test/saved_model/")
        >>> generation_config = GenerationConfig.from_pretrained("./test/saved_model/")

        >>> # You can also specify configuration names to your generation configuration file
        >>> generation_config.save_pretrained("./test/saved_model/", config_file_name="my_configuration.json")
        >>> generation_config = GenerationConfig.from_pretrained("./test/saved_model/", "my_configuration.json")

        >>> # If you'd like to try a minor variation to an existing configuration, you can also pass generation
        >>> # arguments to `.from_pretrained()`. Be mindful that typos and unused arguments will be ignored
        >>> generation_config, unused_kwargs = GenerationConfig.from_pretrained(
        ...     "gpt2", top_k=1, foo=False, do_sample=True, return_unused_kwargs=True
        ... )
        >>> generation_config.top_k
        1

        >>> unused_kwargs
        {'foo': False}
        ```"""
        config_file_name = config_file_name if config_file_name is not None else GENERATION_CONFIG_NAME

        subfolder = kwargs.pop("subfolder", "")
        if subfolder is None:
            subfolder = ""

        resolved_config_file = resolve_file_path(
            pretrained_model_name_or_path,
            [config_file_name],
            subfolder,
            cache_dir=cache_dir,
            force_download=force_download,
            from_aistudio=from_aistudio,
            from_hf_hub=from_hf_hub,
            from_modelscope=from_modelscope,
        )
        assert (
            resolved_config_file is not None
        ), f"please make sure {config_file_name} under {pretrained_model_name_or_path}"
        try:
            logger.info(f"Loading configuration file {resolved_config_file}")
            # Load config dict
            config_dict = cls._dict_from_json_file(resolved_config_file)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise EnvironmentError(f"Config file<'{resolved_config_file}'> is not a valid JSON file.")

        return cls.from_dict(config_dict, **kwargs)

    @classmethod
    def _dict_from_json_file(cls, json_file: Union[str, os.PathLike]):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)

    def dict_paddle_dtype_to_str(self, d: Dict[str, Any]) -> None:
        """
        Checks whether the passed dictionary and its nested dicts have a *paddle_dtype* key and if it's not None,
        converts paddle.dtype to a string of just the type. For example, `paddle.float32` get converted into *"float32"*
        string, which can then be stored in the json format.
        """
        if d.get("dtype", None) is not None and not isinstance(d["dtype"], str):
            d["dtype"] = convert_dtype(d["dtype"])
        for value in d.values():
            if isinstance(value, dict):
                self.dict_paddle_dtype_to_str(value)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs) -> "GenerationConfig":
        """
        Instantiates a [`GenerationConfig`] from a Python dictionary of parameters.

        Args:
            config_dict (`Dict[str, Any]`):
                Dictionary that will be used to instantiate the configuration object.
            kwargs (`Dict[str, Any]`):
                Additional parameters from which to initialize the configuration object.

        Returns:
            [`GenerationConfig`]: The configuration object instantiated from those parameters.
        """
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

        config = cls(**{**config_dict, **kwargs})
        unused_kwargs = config.update(**kwargs)

        # logger.info(f"Generate config {config}")
        if return_unused_kwargs:
            return config, unused_kwargs
        else:
            return config

    def to_diff_dict(self) -> Dict[str, Any]:
        """
        Removes all attributes from config which correspond to the default config attributes for better readability and
        serializes to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        config_dict = self.to_dict()

        # get the default config dict
        default_config_dict = GenerationConfig().to_dict()

        serializable_config_dict = {}

        # only serialize values that differ from the default config
        for key, value in config_dict.items():
            if key not in default_config_dict or key == "transformers_version" or value != default_config_dict[key]:
                serializable_config_dict[key] = value

        self.dict_paddle_dtype_to_str(serializable_config_dict)
        return serializable_config_dict

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        output = copy.deepcopy(self.__dict__)

        # PaddleFormers version when serializing this file
        output["paddleformers_version"] = __version__

        self.dict_paddle_dtype_to_str(output)
        return output

    def to_json_string(self, use_diff: bool = True) -> str:
        """
        Serializes this instance to a JSON string.

        Args:
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default `GenerationConfig()`
                is serialized to JSON string.

        Returns:
            `str`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        if use_diff is True:
            config_dict = self.to_diff_dict()
        else:
            config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: Union[str, os.PathLike], use_diff: bool = True):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default `GenerationConfig()`
                is serialized to JSON file.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string(use_diff=use_diff))

    @classmethod
    def from_model_config(cls, model_config: PretrainedConfig) -> "GenerationConfig":
        """
        Instantiates a [`GenerationConfig`] from a [`PretrainedConfig`]. This function is useful to convert legacy
        [`PretrainedConfig`] objects, which may contain generation parameters, into a stand-alone [`GenerationConfig`].

        Args:
            model_config (`PretrainedConfig`):
                The model config that will be used to instantiate the generation config.

        Returns:
            [`GenerationConfig`]: The configuration object instantiated from those parameters.
        """
        config_dict = model_config.to_dict()
        config_dict.pop("_from_model_config", None)
        config = cls.from_dict(config_dict, return_unused_kwargs=False, _from_model_config=True)

        # Special case: some models have generation attributes set in the decoder. Use them if still unset in the
        # generation config.
        for decoder_name in ("decoder", "generator", "text_config"):
            if decoder_name in config_dict:
                default_generation_config = GenerationConfig()
                decoder_config = config_dict[decoder_name]
                for attr in config.to_dict().keys():
                    if attr in decoder_config and getattr(config, attr) == getattr(default_generation_config, attr):
                        setattr(config, attr, decoder_config[attr])

        return config

    def update(self, **kwargs):
        """
        Updates attributes of this class instance with attributes from `kwargs` if they match existing atributtes,
        returning all the unused kwargs.

        Args:
            kwargs (`Dict[str, Any]`):
                Dictionary of attributes to tentatively update this class.

        Returns:
            `Dict[str, Any]`: Dictionary containing all the key-value pairs that were not used to update the instance.
        """
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                to_remove.append(key)

        # remove all the attributes that were updated, without modifying the input dict
        unused_kwargs = {key: value for key, value in kwargs.items() if key not in to_remove}
        return unused_kwargs
