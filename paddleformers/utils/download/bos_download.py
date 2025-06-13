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

import logging
import os
import re
import tempfile
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Dict, Literal, Optional, Union

from filelock import FileLock
from huggingface_hub.utils import (
    EntryNotFoundError,
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

logger = logging.getLogger(__name__)

from ..env import MODEL_HOME
from .common import (
    DEFAULT_ETAG_TIMEOUT,
    DEFAULT_REQUEST_TIMEOUT,
    AistudioBosFileMetadata,
    _as_int,
    _chmod_and_replace,
    _normalize_etag,
    _request_wrapper,
    http_get,
    raise_for_status,
)

ENDPOINT = os.getenv("PPNLP_ENDPOINT", "https://bj.bcebos.com/paddlenlp")
ENDPOINT_v2 = "https://paddlenlp.bj.bcebos.com"

BOS_URL_TEMPLATE = ENDPOINT + "/{repo_type}/community/{repo_id}/{revision}/{filename}"
BOS_URL_TEMPLATE_WITHOUT_REVISION = ENDPOINT + "/{repo_type}/community/{repo_id}/{filename}"


REGEX_COMMIT_HASH = re.compile(r"^[0-9a-f]{40}$")
REPO_TYPE = "models"


def get_bos_file_metadata(
    url: str,
    token: Union[bool, str, None] = None,
    proxies: Optional[Dict] = None,
    timeout: Optional[float] = DEFAULT_REQUEST_TIMEOUT,
    library_name: Optional[str] = None,
    library_version: Optional[str] = None,
    user_agent: Union[Dict, str, None] = None,
):
    """Fetch metadata of a file versioned on the Hub for a given url.

    Args:
        url (`str`):
            File url, for example returned by [`bos_url`].
        token (`str` or `bool`, *optional*):
            A token to be used for the download.
                - If `True`, the token is read from the BOS config
                  folder.
                - If `False` or `None`, no token is provided.
                - If a string, it's used as the authentication token.
        proxies (`dict`, *optional*):
            Dictionary mapping protocol to the URL of the proxy passed to
            `requests.request`.
        timeout (`float`, *optional*, defaults to 10):
            How many seconds to wait for the server to send metadata before giving up.
        library_name (`str`, *optional*):
            The name of the library to which the object corresponds.
        library_version (`str`, *optional*):
            The version of the library.
        user_agent (`dict`, `str`, *optional*):
            The user-agent info in the form of a dictionary or a string.

    Returns:
        A [`AistudioBosFileMetadata`] object containing metadata such as location, etag, size and
        commit_hash.
    """
    headers = {}
    headers["Accept-Encoding"] = "identity"  # prevent any compression => we want to know the real size of the file

    # Retrieve metadata
    r = _request_wrapper(
        method="HEAD",
        url=url,
        headers=headers,
        allow_redirects=False,
        follow_relative_redirects=True,
        proxies=proxies,
        timeout=timeout,
    )
    raise_for_status(r)

    # Return
    return AistudioBosFileMetadata(
        commit_hash=None,
        etag=_normalize_etag(r.headers.get("ETag")),
        location=url,
        size=_as_int(r.headers.get("Content-Length")),
    )


def bos_url(
    repo_id: str,
    filename: str,
    *,
    subfolder: Optional[str] = None,
    repo_type: Optional[str] = None,
    revision: Optional[str] = None,
    endpoint: Optional[str] = None,
) -> str:
    if subfolder == "":
        subfolder = None
    if subfolder is not None:
        filename = f"{subfolder}/{filename}"

    url = BOS_URL_TEMPLATE_WITHOUT_REVISION.format(
        repo_type=REPO_TYPE,
        repo_id=repo_id,
        filename=filename,
    )

    # Update endpoint if provided
    if endpoint is not None and url.startswith(ENDPOINT):
        url = endpoint + url[len(ENDPOINT) :]
    return url


def bos_download(
    repo_id: str = None,
    filename: str = None,
    subfolder: Optional[str] = None,
    repo_type: Optional[str] = None,
    revision: Optional[str] = None,
    library_name: Optional[str] = None,
    library_version: Optional[str] = None,
    cache_dir: Union[str, Path, None] = None,
    local_dir: Union[str, Path, None] = None,
    local_dir_use_symlinks: Union[bool, Literal["auto"]] = "auto",
    user_agent: Union[Dict, str, None] = None,
    force_download: bool = False,
    proxies: Optional[Dict] = None,
    etag_timeout: float = DEFAULT_ETAG_TIMEOUT,
    resume_download: bool = False,
    token: Optional[str] = None,
    local_files_only: bool = False,
    endpoint: Optional[str] = None,
    url: Optional[str] = None,
    **kwargs,
):
    if url is not None:
        if repo_id is None:
            if url.startswith(ENDPOINT):
                repo_id = "/".join(url[len(ENDPOINT) + 1 :].split("/")[:-1])
            else:
                repo_id = "/".join(url[len(ENDPOINT_v2) + 1 :].split("/")[:-1])
        if filename is None:
            filename = url.split("/")[-1]
        subfolder = None

    if cache_dir is None:
        cache_dir = MODEL_HOME
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    if subfolder == "":
        subfolder = None
    if subfolder is not None:
        # This is used to create a URL, and not a local path, hence the forward slash.
        filename = f"{subfolder}/{filename}"

    locks_dir = os.path.join(cache_dir, ".locks")

    storage_folder = os.path.join(cache_dir, repo_id)
    os.makedirs(storage_folder, exist_ok=True)
    if subfolder is not None:
        storage_sub_folder = os.path.join(storage_folder, subfolder)
        os.makedirs(storage_sub_folder, exist_ok=True)

    if url is None:
        url = bos_url(repo_id, filename, repo_type=REPO_TYPE, endpoint=endpoint)
    headers = None
    url_to_download = url
    lock_path = os.path.join(locks_dir, repo_id, f"{filename}.lock")
    file_path = os.path.join(cache_dir, repo_id, filename)

    if os.name == "nt" and len(os.path.abspath(lock_path)) > 255:
        lock_path = "\\\\?\\" + os.path.abspath(lock_path)

    if os.name == "nt" and len(os.path.abspath(file_path)) > 255:
        file_path = "\\\\?\\" + os.path.abspath(file_path)

    Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
    with FileLock(lock_path):
        # If the download just completed while the lock was activated.
        if os.path.exists(file_path) and not force_download:
            # Even if returning early like here, the lock will be released.
            return file_path

        if resume_download:
            incomplete_path = file_path + ".incomplete"

            @contextmanager
            def _resumable_file_manager():
                with open(incomplete_path, "ab") as f:
                    yield f

            temp_file_manager = _resumable_file_manager
            if os.path.exists(incomplete_path):
                resume_size = os.stat(incomplete_path).st_size
            else:
                resume_size = 0
        else:
            temp_file_manager = partial(  # type: ignore
                tempfile.NamedTemporaryFile, mode="wb", dir=cache_dir, delete=False
            )
            resume_size = 0

        # Download to temporary file, then copy to cache dir once finished.
        # Otherwise you get corrupt cache entries if the download gets interrupted.
        with temp_file_manager() as temp_file:
            logger.info("downloading %s to %s", url_to_download, temp_file.name)

            http_get(
                url_to_download,
                temp_file,
                proxies=proxies,
                resume_size=resume_size,
                headers=headers,
            )

        logger.info("storing %s in cache at %s", url_to_download, file_path)
        _chmod_and_replace(temp_file.name, file_path)
    try:
        os.remove(lock_path)
    except OSError:
        pass
    return file_path


def bos_file_exists(
    repo_id: str,
    filename: str,
    *,
    repo_type: Optional[str] = None,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    endpoint: Optional[str] = None,
) -> bool:
    url = bos_url(repo_id=repo_id, repo_type=REPO_TYPE, filename=filename, endpoint=endpoint)
    try:
        get_bos_file_metadata(url, token=token)
        return True
    except GatedRepoError:  # raise specifically on gated repo
        raise
    except (RepositoryNotFoundError, EntryNotFoundError, RevisionNotFoundError, HfHubHTTPError):
        return False


def bos_try_to_load_from_cache(
    repo_id: str,
    filename: str,
    cache_dir: Union[str, Path, None] = None,
    revision: Optional[str] = None,
    repo_type: Optional[str] = None,
):
    if cache_dir is None:
        cache_dir = MODEL_HOME

    cached_file = os.path.join(cache_dir, repo_id, filename)
    return cached_file if os.path.isfile(cached_file) else None
