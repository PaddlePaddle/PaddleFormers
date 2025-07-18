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

from typing import Optional

from modelscope.hub.file_download import model_file_download as download
from requests import HTTPError


class UnauthorizedError(Exception):
    pass


class EntryNotFoundError(Exception):
    pass


def _add_subfolder(weights_name: str, subfolder: Optional[str] = None) -> str:
    if subfolder is not None and subfolder != "":
        weights_name = "/".join([subfolder, weights_name])
    return weights_name


def modelscope_download(
    repo_id: str,
    filename: str = None,
    cache_dir: Optional[str] = None,
    subfolder: Optional[str] = "",
    revision: Optional[str] = None,
    **kwargs,
):
    if revision is None:
        revision = "master"
    filename = _add_subfolder(filename, subfolder)
    download_kwargs = {}
    if revision is not None:
        download_kwargs["revision"] = revision
    if cache_dir is not None:
        download_kwargs["local_dir"] = cache_dir

    try:
        return download(
            repo_id=repo_id,
            file_path=filename,
            **download_kwargs,
        )
    except ValueError:
        raise EnvironmentError(
            f"Cannot find {filename} in the cached files and it looks like {repo_id} is not the path to a directory containing the {filename} or"
            " \nCheckout your internet connection or see how to run the library in offline mode."
        )
    except EntryNotFoundError:
        raise EnvironmentError(
            f"Cannot find the requested file {filename} in {repo_id}, please make sure the {filename} under the repo {repo_id}"
        )
    except HTTPError as err:
        raise EnvironmentError(f"There was a specific connection error when trying to load {repo_id}:\n{err}")
    except Exception:
        raise EnvironmentError(f"Please make sure the {filename} under the repo {repo_id}")
