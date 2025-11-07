# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import os
from argparse import ArgumentTypeError
from enum import Enum
from pathlib import Path
from typing import Dict, Literal, Optional, Union

from huggingface_hub import _CACHED_NO_EXIST
from huggingface_hub import file_exists as hf_hub_file_exists
from huggingface_hub import hf_hub_download
from huggingface_hub import try_to_load_from_cache as hf_hub_try_to_load_from_cache
from huggingface_hub.utils import (
    EntryNotFoundError,
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

try:
    from paddle import __version__
except ImportError:
    __version__ = ""

from requests import HTTPError

from ..log import logger


class DownloadSource(str, Enum):
    DEFAULT = ""
    HUGGINGFACE = "huggingface"
    AISTUDIO = "aistudio"
    MODELSCOPE = "modelscope"


MODEL_MAPPINGS = {}
HF_MODEL_MAPPINGS = {}


def register_model_group(models: dict[str, dict[DownloadSource, str]]) -> None:
    for name, sources in models.items():
        MODEL_MAPPINGS[name] = sources
        if DownloadSource.HUGGINGFACE in sources:
            HF_MODEL_MAPPINGS[sources[DownloadSource.HUGGINGFACE]] = name


def check_repo(model_name_or_path, download_hub):
    if "PF_HOME" in os.environ:
        home_path = os.environ["PF_HOME"]
        home_model_path = os.path.join(home_path, model_name_or_path)
        if os.path.isfile(home_model_path) or os.path.isdir(home_model_path):
            model_name_or_path = home_model_path
    is_local = os.path.isfile(model_name_or_path) or os.path.isdir(model_name_or_path)
    if not is_local:
        assert download_hub in [
            DownloadSource.HUGGINGFACE,
            DownloadSource.AISTUDIO,
            DownloadSource.MODELSCOPE,
        ], f"download_hub must be one of {DownloadSource.HUGGINGFACE}, {DownloadSource.AISTUDIO}, {DownloadSource.MODELSCOPE}"
        if model_name_or_path not in HF_MODEL_MAPPINGS.keys():
            # repo id set by user
            return model_name_or_path
        model_name = HF_MODEL_MAPPINGS[model_name_or_path]
        if download_hub not in MODEL_MAPPINGS[model_name]:
            return model_name_or_path
        return MODEL_MAPPINGS[model_name][download_hub]
    return model_name_or_path


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


def resolve_file_path(
    repo_id: str = None,
    filenames: Union[str, list] = None,
    subfolder: Optional[str] = None,
    repo_type: Optional[str] = None,
    revision: Optional[str] = None,
    library_version: Optional[str] = __version__,
    cache_dir: Union[str, Path, None] = None,
    local_dir: Union[str, Path, None] = None,
    local_dir_use_symlinks: Union[bool, Literal["auto"]] = "auto",
    user_agent: Union[Dict, str, None] = None,
    force_download: bool = False,
    proxies: Optional[Dict] = None,
    etag_timeout: float = 10,
    resume_download: bool = False,
    token: Union[bool, str, None] = None,
    local_files_only: bool = False,
    endpoint: Optional[str] = None,
    download_hub: Optional[DownloadSource] = None,
    force_return: Optional[bool] = False,
) -> str:
    """
    This is a general download function, mainly called by the from_pretrained function.

    It supports downloading files from four different download sources, including BOS, AiStudio,
    HuggingFace Hub and ModelScope.

    Args:
        repo_id('str'): A path to a folder containing the file, a path of the file, a url or repo name.
        filenames('str' or list): Name of the file to be downloaded. If it is a str, the file will be downloaded directly,
            if it is a list, it will try to download the file in turn, and when one exists, it will be returned directly.
        subfolder('str'): Some repos will exist subfolder.
        repo_type('str'): The default is model.
        cache_dir('str' or Path): Where to save or load the file after downloading.
        download_hub (DownloadSource): The source for model downloading, options include `huggingface`, `aistudio`, `modelscope`, default `aistudio`.
        force_return (Optional[bool]): If True, the function will return None instead of raising an error
            when none of the specified `filenames` can be found or downloaded. Defaults to False (raises error).


    Returns:
        cached_file('str'): The path of file or None.
    """
    assert repo_id is not None, "repo_id cannot be None"
    assert filenames is not None, "filenames cannot be None"

    if isinstance(filenames, str):
        filenames = [filenames]

    # check repo id
    if download_hub is None:
        download_hub = os.environ.get("DOWNLOAD_SOURCE", "huggingface")
        logger.info(f"Using download source: {download_hub}")
    checked_repo_id = check_repo(repo_id, download_hub)
    if repo_id != checked_repo_id:
        repo_id = checked_repo_id
        logger.warning(f"The repo id check failed, changed to {repo_id}")

    download_kwargs = dict(
        repo_id=repo_id,
        filename=filenames[0],
        subfolder=subfolder if subfolder is not None else "",
        repo_type=repo_type,
        revision=revision,
        library_version=library_version,
        cache_dir=cache_dir,
        local_dir=local_dir,
        local_dir_use_symlinks=local_dir_use_symlinks,
        user_agent=user_agent,
        force_download=force_download,
        proxies=proxies,
        etag_timeout=etag_timeout,
        resume_download=resume_download,
        token=token,
        local_files_only=local_files_only,
        endpoint=endpoint,
    )
    cached_file = None
    log_endpoint = "N/A"
    # log_filename = os.path.join(download_kwargs["subfolder"], filename)

    # return file path from local file, eg: /cache/path/model_config.json
    if os.path.isfile(repo_id):
        return repo_id
    # return the file path from local dir with filename, eg: /local/path
    elif os.path.isdir(repo_id):
        for index, filename in enumerate(filenames):
            if os.path.exists(os.path.join(repo_id, download_kwargs["subfolder"], filename)):
                if not os.path.isfile(os.path.join(repo_id, download_kwargs["subfolder"], filename)):
                    raise EnvironmentError(f"{repo_id} does not appear to have file named {filename}.")
                return os.path.join(repo_id, download_kwargs["subfolder"], filename)
            elif index < len(filenames) - 1:
                continue
            else:
                if force_return:
                    return None
                else:
                    raise FileNotFoundError(f"please make sure one of the {filenames} under the dir {repo_id}")

    # check cache
    for filename in filenames:
        cache_file_name = hf_try_to_load_from_cache(repo_id, filename, cache_dir, subfolder, revision, repo_type)
        if download_hub == DownloadSource.HUGGINGFACE and cache_file_name is _CACHED_NO_EXIST:
            cache_file_name = None
        if cache_file_name is not None and os.path.exists(str(cache_file_name)):
            return cache_file_name

    # download file from different origins
    try:
        if download_hub == DownloadSource.MODELSCOPE:
            for index, filename in enumerate(filenames):
                try:
                    from modelscope.hub.file_download import (
                        model_file_download as modelscope_download,
                    )

                    return modelscope_download(repo_id, filename, revision, cache_dir, user_agent, local_files_only)
                except Exception:
                    if index < len(filenames) - 1:
                        continue
                    else:
                        if force_return:
                            return None
                        else:
                            raise EntryNotFoundError(
                                f"please make sure one of the {filenames} under the repo {repo_id}"
                            )

        elif download_hub == DownloadSource.AISTUDIO:
            for index, filename in enumerate(filenames):
                try:
                    from aistudio_sdk.file_download import (
                        model_file_download as aistudio_download,
                    )

                    aistudio_cache_dir = os.path.join(cache_dir, repo_id) if cache_dir is not None else None
                    return aistudio_download(repo_id, filename, revision, local_files_only, aistudio_cache_dir)
                except Exception:
                    if index < len(filenames) - 1:
                        continue
                        if force_return:
                            return None
                        else:
                            raise EntryNotFoundError(
                                f"please make sure one of the {filenames} under the repo {repo_id}"
                            )

        elif download_hub == DownloadSource.HUGGINGFACE:
            log_endpoint = "Huggingface Hub"
            for filename in filenames:
                download_kwargs["filename"] = filename
                is_available = hf_file_exist(
                    repo_id,
                    filename,
                    subfolder=subfolder,
                    repo_type=repo_type,
                    revision=revision,
                    token=token,
                    endpoint=endpoint,
                )
                if is_available:
                    cached_file = hf_hub_download(
                        **download_kwargs,
                    )
                    if cached_file is not None:
                        return cached_file
    except LocalEntryNotFoundError:
        raise EnvironmentError(
            "Cannot find the requested files in the cached path and"
            " outgoing traffic has been disabled. To enable model look-ups"
            " and downloads online, set 'local_files_only' to False."
        )
    except RepositoryNotFoundError:
        raise EnvironmentError(
            f"{repo_id} is not a local folder and is not a valid model identifier "
            f"listed on '{log_endpoint}'\nIf this is a private repository, make sure to pass a "
            "token having permission to this repo."
        )
    except RevisionNotFoundError:
        raise EnvironmentError(
            f"{revision} is not a valid git identifier (branch name, tag name or commit id) that exists for "
            "this model name. Check the model page at "
            f"'{log_endpoint}' for available revisions."
        )
    except EntryNotFoundError:
        raise EnvironmentError(f"Does not appear one of the {filenames} in {repo_id}.")
    except HTTPError as err:
        raise EnvironmentError(f"There was a specific connection error when trying to load {repo_id}:\n{err}")
    except ValueError:
        raise EnvironmentError(
            f"We couldn't connect to '{log_endpoint}' to load this model, couldn't find it"
            f" in the cached files and it looks like {repo_id} is not the path to a"
            f" directory containing one of the {filenames} or"
            " \nCheckout your internet connection or see how to run the library in offline mode."
        )
    except EnvironmentError:
        raise EnvironmentError(
            f"Can't load the model for '{repo_id}'. If you were trying to load it from "
            f"'{log_endpoint}', make sure you don't have a local directory with the same name. "
            f"Otherwise, make sure '{repo_id}' is the correct path to a directory "
            f"containing one of the {filenames}"
        )


def hf_file_exist(
    repo_id: str,
    filename: str,
    *,
    subfolder: Optional[str] = None,
    repo_type: Optional[str] = None,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    endpoint: Optional[str] = None,
):
    assert repo_id is not None, "repo_id cannot be None"
    assert filename is not None, "filename cannot be None"

    if subfolder is None:
        subfolder = ""
    filename = os.path.join(subfolder, filename)
    out = hf_hub_file_exists(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        revision=revision,
        token=token,
    )
    return out


def hf_try_to_load_from_cache(
    repo_id: str,
    filename: str,
    cache_dir: Union[str, Path, None] = None,
    subfolder: str = None,
    revision: Optional[str] = None,
    repo_type: Optional[str] = None,
):
    if subfolder is None:
        subfolder = ""
    load_kwargs = dict(
        repo_id=repo_id,
        filename=os.path.join(subfolder, filename),
        cache_dir=cache_dir,
        revision=revision,
        repo_type=repo_type,
    )
    return hf_hub_try_to_load_from_cache(**load_kwargs)
