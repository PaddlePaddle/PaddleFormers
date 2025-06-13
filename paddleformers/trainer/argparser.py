# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# # Copyright 2020 The HuggingFace Team. All rights reserved.
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

# This file is modified from
#  https://github.com/huggingface/transformers/blob/main/src/transformers/hf_argparser.py

import dataclasses
import json
import os
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, ArgumentTypeError
from copy import copy
from enum import Enum
from inspect import isclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    NewType,
    Optional,
    Tuple,
    Union,
    get_args,
    get_type_hints,
)

from omegaconf import DictConfig, OmegaConf

from ..utils.log import logger

DataClass = NewType("DataClass", Any)
DataClassType = NewType("DataClassType", Any)

__all__ = [
    "PdArgumentParser",
    "strtobool",
]


# From https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def strtobool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError(
            f"Truthy value expected: got {v} but expected one of yes/no, true/false, t/f, y/n, 1/0 (case insensitive)."
        )


class PdArgumentParser(ArgumentParser):
    """
    This subclass of `argparse.ArgumentParser` uses type hints on dataclasses to generate arguments.

    The class is designed to play well with the native argparse. In particular, you can add more (non-dataclass backed)
    arguments to the parser after initialization and you'll get the output back after parsing as an additional
    namespace. Optional: To create sub argument groups use the `_argument_group_name` attribute in the dataclass.
    """

    dataclass_types: Iterable[DataClassType]

    def __init__(self, dataclass_types: Union[DataClassType, Iterable[DataClassType]], **kwargs):
        """
        Args:
            dataclass_types:
                Dataclass type, or list of dataclass types for which we will "fill" instances with the parsed args.
            kwargs:
                (Optional) Passed to `argparse.ArgumentParser()` in the regular way.
        """
        # To make the default appear when using --help
        if "formatter_class" not in kwargs:
            kwargs["formatter_class"] = ArgumentDefaultsHelpFormatter
        super().__init__(**kwargs)
        if dataclasses.is_dataclass(dataclass_types):
            dataclass_types = [dataclass_types]
        self.dataclass_types = list(dataclass_types)
        for dtype in self.dataclass_types:
            self._add_dataclass_arguments(dtype)

    @staticmethod
    def _parse_dataclass_field(parser: ArgumentParser, field: dataclasses.Field):
        field_name = f"--{field.name}"
        kwargs = field.metadata.copy()
        # field.metadata is not used at all by Data Classes,
        # it is provided as a third-party extension mechanism.
        if isinstance(field.type, str):
            raise RuntimeError(
                "Unresolved type detected, which should have been done with the help of "
                "`typing.get_type_hints` method by default"
            )

        origin_type = getattr(field.type, "__origin__", field.type)
        if origin_type is Union:
            if len(field.type.__args__) != 2 or type(None) not in field.type.__args__:
                raise ValueError("Only `Union[X, NoneType]` (i.e., `Optional[X]`) is allowed for `Union`")
            if bool not in field.type.__args__:
                # filter `NoneType` in Union (except for `Union[bool, NoneType]`)
                field.type = (
                    field.type.__args__[0] if isinstance(None, field.type.__args__[1]) else field.type.__args__[1]
                )
                origin_type = getattr(field.type, "__origin__", field.type)

        # A variable to store kwargs for a boolean field, if needed
        # so that we can init a `no_*` complement argument (see below)
        bool_kwargs = {}
        if isinstance(field.type, type) and issubclass(field.type, Enum):
            kwargs["choices"] = [x.value for x in field.type]
            kwargs["type"] = type(kwargs["choices"][0])
            if field.default is not dataclasses.MISSING:
                kwargs["default"] = field.default
            else:
                kwargs["required"] = True
        # fix https://github.com/huggingface/transformers/pull/16946
        elif field.type is bool or field.type == Optional[bool]:
            # Copy the current kwargs to use to instantiate a `no_*` complement argument below.
            # We do not initialize it here because the `no_*` alternative must be instantiated after the real argument
            bool_kwargs = copy(kwargs)

            # Hack because type=bool in argparse does not behave as we want.
            kwargs["type"] = strtobool
            if field.type is bool or (field.default is not None and field.default is not dataclasses.MISSING):
                # Default value is False if we have no default when of type bool.
                default = False if field.default is dataclasses.MISSING else field.default
                # This is the value that will get picked if we don't include --field_name in any way
                kwargs["default"] = default
                # This tells argparse we accept 0 or 1 value after --field_name
                kwargs["nargs"] = "?"
                # This is the value that will get picked if we do --field_name (without value)
                kwargs["const"] = True
        elif isclass(origin_type) and issubclass(origin_type, list):
            # support one dimension list and two dimension list
            if hasattr(get_args(field.type)[0], "__args__"):
                kwargs["type"] = field.type.__args__[0].__args__[0]
                kwargs["action"] = "append"
            else:
                kwargs["type"] = field.type.__args__[0]

            kwargs["nargs"] = "+"
            if field.default_factory is not dataclasses.MISSING:
                kwargs["default"] = field.default_factory()
            elif field.default is dataclasses.MISSING:
                kwargs["required"] = True
        else:
            kwargs["type"] = json.loads if field.type is dict else field.type
            if field.default is not dataclasses.MISSING:
                kwargs["default"] = field.default
            elif field.default_factory is not dataclasses.MISSING:
                kwargs["default"] = field.default_factory()
            else:
                kwargs["required"] = True
        parser.add_argument(field_name, **kwargs)

        # Add a complement `no_*` argument for a boolean field AFTER the initial field has already been added.
        # Order is important for arguments with the same destination!
        # We use a copy of earlier kwargs because the original kwargs have changed a lot before reaching down
        # here and we do not need those changes/additional keys.
        if field.default is True and (field.type is bool or field.type == Optional[bool]):
            bool_kwargs["default"] = False
            parser.add_argument(f"--no_{field.name}", action="store_false", dest=field.name, **bool_kwargs)

    def _add_dataclass_arguments(self, dtype: DataClassType):
        if hasattr(dtype, "_argument_group_name"):
            parser = self.add_argument_group(dtype._argument_group_name)
        else:
            parser = self

        try:
            type_hints: Dict[str, type] = get_type_hints(dtype)
        except NameError:
            raise RuntimeError(
                f"Type resolution failed for f{dtype}. Try declaring the class in global scope or "
                f"removing line of `from __future__ import annotations` which opts in Postponed "
                f"Evaluation of Annotations (PEP 563)"
            )

        for field in dataclasses.fields(dtype):
            if not field.init:
                continue
            field.type = type_hints[field.name]
            self._parse_dataclass_field(parser, field)

    def parse_args_into_dataclasses(
        self, args=None, return_remaining_strings=False, look_for_args_file=True, args_filename=None
    ) -> Tuple[DataClass, ...]:
        """
        Parse command-line args into instances of the specified dataclass types.

        This relies on argparse's `ArgumentParser.parse_known_args`. See the doc at:
        docs.python.org/3.7/library/argparse.html#argparse.ArgumentParser.parse_args

        Args:
            args:
                List of strings to parse. The default is taken from sys.argv. (same as argparse.ArgumentParser)
            return_remaining_strings:
                If true, also return a list of remaining argument strings.
            look_for_args_file:
                If true, will look for a ".args" file with the same base name as the entry point script for this
                process, and will append its potential content to the command line args.
            args_filename:
                If not None, will uses this file instead of the ".args" file specified in the previous argument.

        Returns:
            Tuple consisting of:

                - the dataclass instances in the same order as they were passed to the initializer.abspath
                - if applicable, an additional namespace for more (non-dataclass backed) arguments added to the parser
                  after initialization.
                - The potential list of remaining argument strings. (same as argparse.ArgumentParser.parse_known_args)
        """
        if args_filename or (look_for_args_file and len(sys.argv)):
            if args_filename:
                args_file = Path(args_filename)
            else:
                args_file = Path(sys.argv[0]).with_suffix(".args")

            if args_file.exists():
                fargs = args_file.read_text().split()
                args = fargs + args if args is not None else fargs + sys.argv[1:]
                # in case of duplicate arguments the first one has precedence
                # so we append rather than prepend.

        return self.common_parse(args, return_remaining_strings)

    def common_parse(self, args, return_remaining_strings) -> Tuple[DataClass, ...]:
        namespace, remaining_args = self.parse_known_args(args=args)
        outputs = []
        for dtype in self.dataclass_types:
            keys = {f.name for f in dataclasses.fields(dtype) if f.init}
            inputs = {k: v for k, v in vars(namespace).items() if k in keys}
            for k in keys:
                delattr(namespace, k)
            obj = dtype(**inputs)
            outputs.append(obj)
        if len(namespace.__dict__) > 0:
            # additional namespace.
            outputs.append(namespace)
        if return_remaining_strings:
            return (*outputs, remaining_args)
        else:
            if remaining_args:
                raise ValueError(f"Some specified arguments are not used by the PdArgumentParser: {remaining_args}")

            return (*outputs,)

    def read_json(self, json_file: str) -> list:
        json_file = Path(json_file)
        if json_file.exists():
            with open(json_file, "r") as file:
                data = json.load(file)
            json_args = []
            for key, value in data.items():
                if isinstance(value, list):
                    json_args.extend([f"--{key}", *[str(v) for v in value]])
                elif isinstance(value, dict):
                    json_args.extend([f"--{key}", json.dumps(value)])
                else:
                    json_args.extend([f"--{key}", str(value)])
            return json_args
        else:
            raise FileNotFoundError(f"The argument file {json_file} does not exist.")

    def read_yaml(self, yaml_file: str) -> list:
        import yaml

        yaml_file = Path(yaml_file)
        if yaml_file.exists():
            with open(yaml_file, "r") as file:
                data = yaml.safe_load(file)
            yaml_args = []
            for key, value in data.items():
                if isinstance(value, list):
                    yaml_args.extend([f"--{key}", *[str(v) for v in value]])
                elif isinstance(value, dict):
                    yaml_args.extend([f"--{key}", json.dumps(value)])
                else:
                    yaml_args.extend([f"--{key}", str(value)])
            return yaml_args
        else:
            raise FileNotFoundError(f"The argument file {yaml_file} does not exist.")

    def parse_json_file(self, json_file: str, return_remaining_strings=False) -> Tuple[DataClass, ...]:
        """
        Alternative helper method that does not use `argparse` at all, instead loading a json file and populating the
        dataclass types.
        """
        json_args = self.read_json(json_file)
        return self.common_parse(json_args, return_remaining_strings)

    def parse_json_file_and_cmd_lines(self, return_remaining_strings=False) -> Tuple[DataClass, ...]:
        """
        Extend the functionality of `parse_json_file` to handle command line arguments in addition to loading a JSON
        file.

        When there is a conflict between the command line arguments and the JSON file configuration,
        the command line arguments will take precedence.

        Returns:
            Tuple consisting of:

                - the dataclass instances in the same order as they were passed to the initializer.abspath
        """
        if not sys.argv[1].endswith(".json"):
            raise ValueError(f"The first argument should be a JSON file, but it is {sys.argv[1]}")
        json_args = self.read_json(sys.argv[1])
        # In case of conflict, command line arguments take precedence
        args = json_args + sys.argv[2:]
        return self.common_parse(args, return_remaining_strings)

    def parse_yaml_file_and_cmd_lines(self, return_remaining_strings=False) -> Tuple[DataClass, ...]:
        """
        Extend the functionality of `parse_yaml_file` to handle command line arguments in addition to loading a YAML
        file.

        When there is a conflict between the command line arguments and the YAML file configuration,
        the command line arguments will take precedence.

        Returns:
            Tuple consisting of:

                - the dataclass instances in the same order as they were passed to the initializer.abspath
        """
        if not sys.argv[1].endswith(".yaml"):
            raise ValueError(f"The first argument should be a YAML file, but it is {sys.argv[1]}")
        yaml_args = self.read_yaml(sys.argv[1])
        # In case of conflict, command line arguments take precedence
        args = yaml_args + sys.argv[2:]
        return self.common_parse(args, return_remaining_strings)

    def read_python(self, python_file: str) -> list:

        python_file = Path(python_file)

        def get_variables_exec(file_path):
            def flatten(config):
                ret = {}
                for k, v in config.items():
                    if type(v) is dict:
                        sub = flatten(v)
                        for sk, sv in sub.items():
                            ret[sk] = sv
                    else:
                        ret[k] = v
                return ret

            with open(file_path, "r", encoding="utf-8") as f:
                code = compile(f.read(), file_path, "exec")
                globals_dict = {}
                exec(code, globals_dict)
                ret_dict = {k: globals_dict[k] for k in globals_dict if not k.startswith("__")}
                return flatten(ret_dict)

        if python_file.exists():
            data = get_variables_exec(python_file)

            python_args = []
            for key, value in data.items():
                if isinstance(value, list):
                    python_args.extend([f"--{key}", *[str(v) for v in value]])
                elif isinstance(value, dict):
                    python_args.extend([f"--{key}", json.dumps(value)])
                else:
                    python_args.extend([f"--{key}", str(value)])
            return python_args
        else:
            raise FileNotFoundError(f"The argument file {python_file} does not exist.")

    def parse_python_file_and_cmd_lines(self, return_remaining_strings=False) -> Tuple[DataClass, ...]:
        """
        Extend the functionality of `parse_python_file` to handle command line arguments in addition to loading a python
        file.

        When there is a conflict between the command line arguments and the YAML file configuration,
        the command line arguments will take precedence.

        Returns:
            Tuple consisting of:

                - the dataclass instances in the same order as they were passed to the initializer.abspath
        """
        if not sys.argv[1].endswith(".py"):
            raise ValueError(f"The first argument should be a PYTHON file, but it is {sys.argv[1]}")
        python_args = self.read_python(sys.argv[1])
        # In case of conflict, command line arguments take precedence
        args = python_args + sys.argv[2:]
        return self.common_parse(args, return_remaining_strings)

    def parse_dict(self, args: dict) -> Tuple[DataClass, ...]:
        """
        Alternative helper method that does not use `argparse` at all, instead uses a dict and populating the dataclass
        types.
        """

        def to_regular_dict(obj):
            if isinstance(obj, DictConfig):
                obj = OmegaConf.to_container(obj, resolve=True)
            if isinstance(obj, dict):
                return {k: to_regular_dict(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_regular_dict(v) for v in obj]
            return obj

        def get_resume_checkpoint_path(args):
            """
            get resume checkpoint path from mpirun env
            """
            pdc_init_step = os.getenv("PDC_INIT_STEP")
            # user defined resume_from_checkpoint
            user_defined_resume_from_checkpoint = args.get("resume_from_checkpoint", None)
            if pdc_init_step is None:
                logger.info(f"user has defined resume_from_checkpoint: {user_defined_resume_from_checkpoint}")
                return user_defined_resume_from_checkpoint
            else:
                if pdc_init_step == "0":
                    # from_scratch train process launched by pdc longjob
                    if user_defined_resume_from_checkpoint is None:
                        logger.info("resume training process from scratch (step 0)")
                        return None
                    else:
                        # Launching the sft_base training process using an initial checkpoint with the starting step set to 0.
                        # For instance, resume training from the checkpoint located at ‘./output/eb/checkpoint-init’.
                        logger.info(
                            f"init_step == 0 and user has defined resume_from_checkpoint: {user_defined_resume_from_checkpoint}"
                        )
                        return user_defined_resume_from_checkpoint
                else:
                    # pdc_init_step > 0
                    logger.info(f"resume training process by pdc longjob with resume step: {pdc_init_step}")
                    resume_checkpoint = os.path.join(args.get("output_dir", None), f"checkpoint-{pdc_init_step}")
                    if user_defined_resume_from_checkpoint is not None:
                        logger.warning(
                            f"pdc_init_step:{pdc_init_step} and resume_ckpt:{user_defined_resume_from_checkpoint} exist together, use resume_checkpoint:{resume_checkpoint}"
                        )
                    return resume_checkpoint

        args["resume_from_checkpoint"] = get_resume_checkpoint_path(args)
        args_for_json = to_regular_dict(args)

        json_filename = args_for_json.get("args_output_to_local")
        if json_filename:
            try:
                with open(json_filename, "w") as json_file:
                    json.dump(args_for_json, json_file, indent=4)
            except Exception as e:
                logger.error(f"Failed to write args output JSON file: {e}")
                # Optionally handle the error or log it, then continue

        outputs = []
        for dtype in self.dataclass_types:
            keys = {f.name for f in dataclasses.fields(dtype) if f.init}
            inputs = {k: v for k, v in args.items() if k in keys}
            obj = dtype(**inputs)
            outputs.append(obj)
        return (*outputs,)
