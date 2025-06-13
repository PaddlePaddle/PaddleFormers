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

import io
import logging
import os
import re
import shutil
import tempfile
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Dict, Generator, Literal, Optional, Union
from urllib.parse import quote

import requests
from filelock import FileLock
from huggingface_hub.utils import (
    EntryNotFoundError,
    FileMetadataError,
    GatedRepoError,
    HfHubHTTPError,
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

logger = logging.getLogger(__name__)

from .common import (
    DEFALUT_LOCAL_DIR_AUTO_SYMLINK_THRESHOLD,
    DEFAULT_ETAG_TIMEOUT,
    DEFAULT_REQUEST_TIMEOUT,
    AistudioBosFileMetadata,
    OfflineModeIsEnabled,
    _cache_commit_hash_for_specific_revision,
    _check_disk_space,
    _chmod_and_replace,
    _create_symlink,
    _get_pointer_path,
    _is_true,
    _normalize_etag,
    _request_wrapper,
    _to_local_dir,
    http_get,
    raise_for_status,
    repo_folder_name,
)

VERSION = "0.1.5"
ENDPOINT = os.getenv("AISTUDIO_ENDPOINT", "http://git.aistudio.baidu.com")

AISTUDIO_URL_TEMPLATE = ENDPOINT + "/api/v1/repos/{user_name}/{repo_name}/contents/{filename}"


default_home = os.path.join(os.path.expanduser("~"), ".cache")
AISTUDIO_HOME = os.path.expanduser(
    os.getenv(
        "AISTUDIO_HOME",
        os.path.join(os.getenv("XDG_CACHE_HOME", default_home), "paddle"),
    )
)
default_cache_path = os.path.join(AISTUDIO_HOME, "aistudio")
AISTUDIO_HUB_CACHE = os.getenv("AISTUDIO_HUB_CACHE", default_cache_path)


DEFAULT_REVISION = "master"
REPO_TYPE_MODEL = "model"
REPO_TYPES = [None, REPO_TYPE_MODEL]


REGEX_COMMIT_HASH = re.compile(r"^[0-9a-f]{40}$")


# TOKEN
AISTUDIO_TOKEN_PATH = os.path.join(AISTUDIO_HOME, "token")
AISTUDIO_HUB_DISABLE_IMPLICIT_TOKEN: bool = _is_true(os.environ.get("AISTUDIO_HUB_DISABLE_IMPLICIT_TOKEN"))


class LocalTokenNotFoundError(EnvironmentError):
    """Raised if local token is required but not found."""


def _clean_token(token: Optional[str]) -> Optional[str]:
    """Clean token by removing trailing and leading spaces and newlines.

    If token is an empty string, return None.
    """
    if token is None:
        return None
    return token.replace("\r", "").replace("\n", "").strip() or None


def _get_token_from_environment() -> Optional[str]:
    return _clean_token(os.environ.get("AISTUDIO_ACCESS_TOKEN") or os.environ.get("AISTUDIO_TOKEN"))


def _get_token_from_file() -> Optional[str]:
    try:
        return _clean_token(Path(AISTUDIO_TOKEN_PATH).read_text())
    except FileNotFoundError:
        return None


def get_token() -> Optional[str]:
    """
    Get token if user is logged in.

    Note: in most cases, you should use [`build_aistudio_headers`] instead. This method is only useful
          if you want to retrieve the token for other purposes than sending an HTTP request.

    Token is retrieved in priority from the `AISTUDIO_ACCESS_TOKEN` environment variable. Otherwise, we read the token file located
    in the Aistudio home folder. Returns None if user is not logged in.

    Returns:
        `str` or `None`: The token, `None` if it doesn't exist.
    """
    return _get_token_from_environment() or _get_token_from_file()


def get_token_to_send(token: Optional[Union[bool, str]]) -> Optional[str]:
    """Select the token to send from either `token` or the cache."""
    # Case token is explicitly provided
    if isinstance(token, str):
        return token

    # Case token is explicitly forbidden
    if token is False:
        return None

    # Token is not provided: we get it from local cache
    cached_token = get_token()

    # Case token is explicitly required
    if token is True:
        if cached_token is None:
            raise LocalTokenNotFoundError(
                "Token is required (`token=True`), but no token found. You"
                " to provide a token or be logged in to Aistudio Hub . See"
                "https://ai.baidu.com/ai-doc/AISTUDIO/slmkadt9z#2-%E5%A6%82%E4%BD%95%E4%BD%BF%E7%94%A8%E8%AE%BF%E9%97%AE%E4%BB%A4%E7%89%8C."
            )
        return cached_token

    # Case implicit use of the token is forbidden by env variable
    if AISTUDIO_HUB_DISABLE_IMPLICIT_TOKEN:
        return None

    # Otherwise: we use the cached token as the user has not explicitly forbidden it
    return cached_token


def _validate_token_to_send(token: Optional[str], is_write_action: bool) -> None:
    if is_write_action:
        if token is None:
            raise ValueError(
                "Token is required (write-access action) but no token found. You need"
                " to provide a token or be logged in to Aistudio Hub . See"
                "https://ai.baidu.com/ai-doc/AISTUDIO/slmkadt9z#2-%E5%A6%82%E4%BD%95%E4%BD%BF%E7%94%A8%E8%AE%BF%E9%97%AE%E4%BB%A4%E7%89%8C."
            )


def build_aistudio_headers(
    *,
    token: Optional[Union[bool, str]] = None,
    is_write_action: bool = False,
    library_name: Optional[str] = None,
    library_version: Optional[str] = None,
    user_agent: Union[Dict, str, None] = None,
) -> Dict[str, str]:
    # Get auth token to send
    token_to_send = get_token_to_send(token)
    _validate_token_to_send(token_to_send, is_write_action=is_write_action)

    # Combine headers
    headers = {"Content-Type": "application/json", "SDK-Version": str(VERSION)}
    if token_to_send is not None:
        headers["Authorization"] = f"token {token_to_send}"
    return headers


def get_aistudio_file_metadata(
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
            File url, for example returned by [`aistudio_hub_url`].
        token (`str` or `bool`, *optional*):
            A token to be used for the download.
                - If `True`, the token is read from the Aistudio config
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
    headers = build_aistudio_headers(
        token=token, library_name=library_name, library_version=library_version, user_agent=user_agent
    )
    headers["Accept-Encoding"] = "identity"  # prevent any compression => we want to know the real size of the file

    # Retrieve metadata
    r = _request_wrapper(
        method="GET",
        url=url,
        headers=headers,
        allow_redirects=False,
        follow_relative_redirects=True,
        proxies=proxies,
        timeout=timeout,
    )
    raise_for_status(r)
    res = r.json()

    # Return
    return AistudioBosFileMetadata(
        commit_hash=res["last_commit_sha"],
        etag=_normalize_etag(res["sha"]),
        location=res["git_url"],
        size=res["size"],
    )


def aistudio_hub_url(
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

    if repo_type is None:
        repo_type = REPO_TYPES[-1]
    if repo_type not in REPO_TYPES:
        raise ValueError("Invalid repo type")
    if revision is None:
        revision = DEFAULT_REVISION

    # NEW ADD
    if "/" not in repo_id:
        raise ValueError("repo_id must be in the format of 'namespace/name'")
    user_name, repo_name = repo_id.split("/")
    user_name = user_name.strip()
    repo_name = repo_name.strip()

    url = AISTUDIO_URL_TEMPLATE.format(
        user_name=quote(user_name, safe=""), repo_name=quote(repo_name, safe=""), filename=quote(filename)
    )
    # Update endpoint if provided
    if endpoint is not None and url.startswith(ENDPOINT):
        url = endpoint + url[len(ENDPOINT) :]

    if revision != "master":
        url += f"?ref={quote(revision, safe='')}"
    return url


def aistudio_hub_download(
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
    # TODO
    user_agent: Union[Dict, str, None] = None,
    force_download: bool = False,
    proxies: Optional[Dict] = None,
    etag_timeout: float = DEFAULT_ETAG_TIMEOUT,
    resume_download: bool = False,
    token: Optional[str] = None,
    local_files_only: bool = False,
    endpoint: Optional[str] = None,
    **kwargs,
):

    if cache_dir is None:
        cache_dir = AISTUDIO_HUB_CACHE
    if revision is None:
        revision = DEFAULT_REVISION
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)
    if isinstance(local_dir, Path):
        local_dir = str(local_dir)
    locks_dir = os.path.join(cache_dir, ".locks")

    if subfolder == "":
        subfolder = None
    if subfolder is not None:
        # This is used to create a URL, and not a local path, hence the forward slash.
        filename = f"{subfolder}/{filename}"

    if repo_type is None:
        repo_type = REPO_TYPES[-1]
    if repo_type not in REPO_TYPES:
        raise ValueError(f"Invalid repo type: {repo_type}. Accepted repo types are: {str(REPO_TYPES)}")

    storage_folder = os.path.join(cache_dir, repo_folder_name(repo_id=repo_id, repo_type=repo_type))
    os.makedirs(storage_folder, exist_ok=True)

    # cross platform transcription of filename, to be used as a local file path.
    relative_filename = os.path.join(*filename.split("/"))
    if os.name == "nt":
        if relative_filename.startswith("..\\") or "\\..\\" in relative_filename:
            raise ValueError(
                f"Invalid filename: cannot handle filename '{relative_filename}' on Windows. Please ask the repository"
                " owner to rename this file."
            )

    # if user provides a commit_hash and they already have the file on disk,
    # shortcut everything.
    # TODO, 当前不支持commit id下载，因此这个肯定跑的。
    if not force_download:  # REGEX_COMMIT_HASH.match(revision)
        pointer_path = _get_pointer_path(storage_folder, revision, relative_filename)
        if os.path.exists(pointer_path):
            if local_dir is not None:
                return _to_local_dir(pointer_path, local_dir, relative_filename, use_symlinks=local_dir_use_symlinks)
            return pointer_path

    url = aistudio_hub_url(repo_id, filename, repo_type=repo_type, revision=revision, endpoint=endpoint)

    headers = build_aistudio_headers(
        token=token,
        library_name=library_name,
        library_version=library_version,
        user_agent=user_agent,
    )
    url_to_download = url.replace("/contents/", "/media/")

    etag = None
    commit_hash = None
    expected_size = None
    head_call_error: Optional[Exception] = None
    if not local_files_only:
        try:
            try:
                metadata = get_aistudio_file_metadata(
                    url=url,
                    token=token,
                    proxies=proxies,
                    timeout=etag_timeout,
                    library_name=library_name,
                    library_version=library_version,
                    user_agent=user_agent,
                )
            except EntryNotFoundError as http_error:  # noqa: F841
                raise
            # Commit hash must exist
            # TODO，这里修改了commit hash，强迫为revision了。
            commit_hash = revision  # metadata.commit_hash
            if commit_hash is None:
                raise FileMetadataError(
                    "Distant resource does not seem to be on aistudio hub. It is possible that a configuration issue"
                    " prevents you from downloading resources from aistudio hub. Please check your firewall"
                    " and proxy settings and make sure your SSL certificates are updated."
                )

            # Etag must exist
            etag = metadata.etag
            # We favor a custom header indicating the etag of the linked resource, and
            # we fallback to the regular etag header.
            # If we don't have any of those, raise an error.
            if etag is None:
                raise FileMetadataError(
                    "Distant resource does not have an ETag, we won't be able to reliably ensure reproducibility."
                )

            # Expected (uncompressed) size
            expected_size = metadata.size

        except (requests.exceptions.SSLError, requests.exceptions.ProxyError):
            # Actually raise for those subclasses of ConnectionError
            raise
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            OfflineModeIsEnabled,
        ) as error:
            # Otherwise, our Internet connection is down.
            # etag is None
            head_call_error = error
            pass
        except (RevisionNotFoundError, EntryNotFoundError):
            # The repo was found but the revision or entry doesn't exist on the Hub (never existed or got deleted)
            raise
        except requests.HTTPError as error:
            # Multiple reasons for an http error:
            # - Repository is private and invalid/missing token sent
            # - Repository is gated and invalid/missing token sent
            # - Hub is down (error 500 or 504)
            # => let's switch to 'local_files_only=True' to check if the files are already cached.
            #    (if it's not the case, the error will be re-raised)
            head_call_error = error
            pass
        except FileMetadataError as error:
            # Multiple reasons for a FileMetadataError:
            # - Wrong network configuration (proxy, firewall, SSL certificates)
            # - Inconsistency on the Hub
            # => let's switch to 'local_files_only=True' to check if the files are already cached.
            #    (if it's not the case, the error will be re-raised)
            head_call_error = error
            pass

    # etag can be None for several reasons:
    # 1. we passed local_files_only.
    # 2. we don't have a connection
    # 3. Hub is down (HTTP 500 or 504)
    # 4. repo is not found -for example private or gated- and invalid/missing token sent
    # 5. Hub is blocked by a firewall or proxy is not set correctly.
    # => Try to get the last downloaded one from the specified revision.
    #
    # If the specified revision is a commit hash, look inside "snapshots".
    # If the specified revision is a branch or tag, look inside "refs".
    if etag is None:
        # In those cases, we cannot force download.
        if force_download:
            raise ValueError(
                "We have no connection or you passed local_files_only, so force_download is not an accepted option."
            )

        # Try to get "commit_hash" from "revision"
        commit_hash = None
        if REGEX_COMMIT_HASH.match(revision):
            commit_hash = revision
        else:
            ref_path = os.path.join(storage_folder, "refs", revision)
            if os.path.isfile(ref_path):
                with open(ref_path) as f:
                    commit_hash = f.read()

        # Return pointer file if exists
        if commit_hash is not None:
            pointer_path = _get_pointer_path(storage_folder, commit_hash, relative_filename)
            if os.path.exists(pointer_path):
                if local_dir is not None:
                    return _to_local_dir(
                        pointer_path, local_dir, relative_filename, use_symlinks=local_dir_use_symlinks
                    )
                return pointer_path

        # If we couldn't find an appropriate file on disk, raise an error.
        # If files cannot be found and local_files_only=True,
        # the models might've been found if local_files_only=False
        # Notify the user about that
        if local_files_only:
            raise LocalEntryNotFoundError(
                "Cannot find the requested files in the disk cache and outgoing traffic has been disabled. To enable"
                " aistudio hub look-ups and downloads online, set 'local_files_only' to False."
            )
        elif isinstance(head_call_error, RepositoryNotFoundError) or isinstance(head_call_error, GatedRepoError):
            # Repo not found => let's raise the actual error
            raise head_call_error
        else:
            # Otherwise: most likely a connection issue or Hub downtime => let's warn the user
            raise LocalEntryNotFoundError(
                "An error happened while trying to locate the file on the Hub and we cannot find the requested files"
                " in the local cache. Please check your connection and try again or make sure your Internet connection"
                " is on."
            ) from head_call_error

    # From now on, etag and commit_hash are not None.
    assert etag is not None, "etag must have been retrieved from server"
    assert commit_hash is not None, "commit_hash must have been retrieved from server"
    blob_path = os.path.join(storage_folder, "blobs", etag)
    pointer_path = _get_pointer_path(storage_folder, commit_hash, relative_filename)

    os.makedirs(os.path.dirname(blob_path), exist_ok=True)
    os.makedirs(os.path.dirname(pointer_path), exist_ok=True)
    # if passed revision is not identical to commit_hash
    # then revision has to be a branch name or tag name.
    # In that case store a ref.
    _cache_commit_hash_for_specific_revision(storage_folder, revision, commit_hash)

    if os.path.exists(pointer_path) and not force_download:
        if local_dir is not None:
            return _to_local_dir(pointer_path, local_dir, relative_filename, use_symlinks=local_dir_use_symlinks)
        return pointer_path

    if os.path.exists(blob_path) and not force_download:
        # we have the blob already, but not the pointer
        if local_dir is not None:  # to local dir
            return _to_local_dir(blob_path, local_dir, relative_filename, use_symlinks=local_dir_use_symlinks)
        else:  # or in snapshot cache
            _create_symlink(blob_path, pointer_path, new_blob=False)
            return pointer_path

    # Prevent parallel downloads of the same file with a lock.
    # etag could be duplicated across repos,
    lock_path = os.path.join(locks_dir, repo_folder_name(repo_id=repo_id, repo_type=repo_type), f"{etag}.lock")

    # Some Windows versions do not allow for paths longer than 255 characters.
    # In this case, we must specify it is an extended path by using the "\\?\" prefix.
    if os.name == "nt" and len(os.path.abspath(lock_path)) > 255:
        lock_path = "\\\\?\\" + os.path.abspath(lock_path)

    if os.name == "nt" and len(os.path.abspath(blob_path)) > 255:
        blob_path = "\\\\?\\" + os.path.abspath(blob_path)

    Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
    with FileLock(lock_path):
        # If the download just completed while the lock was activated.
        if os.path.exists(pointer_path) and not force_download:
            # Even if returning early like here, the lock will be released.
            if local_dir is not None:
                return _to_local_dir(pointer_path, local_dir, relative_filename, use_symlinks=local_dir_use_symlinks)
            return pointer_path

        if resume_download:
            incomplete_path = blob_path + ".incomplete"

            @contextmanager
            def _resumable_file_manager() -> Generator[io.BufferedWriter, None, None]:
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
            logger.info("downloading %s to %s", url, temp_file.name)

            if expected_size is not None:  # might be None if HTTP header not set correctly
                # Check tmp path
                _check_disk_space(expected_size, os.path.dirname(temp_file.name))

                # Check destination
                _check_disk_space(expected_size, os.path.dirname(blob_path))
                if local_dir is not None:
                    _check_disk_space(expected_size, local_dir)

            http_get(
                url_to_download,
                temp_file,
                proxies=proxies,
                resume_size=resume_size,
                headers=headers,
                expected_size=expected_size,
            )
        if local_dir is None:
            logger.debug(f"Storing {url} in cache at {blob_path}")
            _chmod_and_replace(temp_file.name, blob_path)
            _create_symlink(blob_path, pointer_path, new_blob=True)
        else:
            local_dir_filepath = os.path.join(local_dir, relative_filename)
            os.makedirs(os.path.dirname(local_dir_filepath), exist_ok=True)

            # If "auto" (default) copy-paste small files to ease manual editing but symlink big files to save disk
            # In both cases, blob file is cached.
            is_big_file = os.stat(temp_file.name).st_size > DEFALUT_LOCAL_DIR_AUTO_SYMLINK_THRESHOLD
            if local_dir_use_symlinks is True or (local_dir_use_symlinks == "auto" and is_big_file):
                logger.debug(f"Storing {url} in cache at {blob_path}")
                _chmod_and_replace(temp_file.name, blob_path)
                logger.debug("Create symlink to local dir")
                _create_symlink(blob_path, local_dir_filepath, new_blob=False)
            elif local_dir_use_symlinks == "auto" and not is_big_file:
                logger.debug(f"Storing {url} in cache at {blob_path}")
                _chmod_and_replace(temp_file.name, blob_path)
                logger.debug("Duplicate in local dir (small file and use_symlink set to 'auto')")
                shutil.copyfile(blob_path, local_dir_filepath)
            else:
                logger.debug(f"Storing {url} in local_dir at {local_dir_filepath} (not cached).")
                _chmod_and_replace(temp_file.name, local_dir_filepath)
            pointer_path = local_dir_filepath  # for return value

    return pointer_path


def aistudio_hub_file_exists(
    repo_id: str,
    filename: str,
    *,
    repo_type: Optional[str] = None,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    endpoint: Optional[str] = None,
) -> bool:
    """
    Checks if a file exists in a repository on the Aistudio Hub.

    Args:
        repo_id (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
        filename (`str`):
            The name of the file to check, for example:
            `"config.json"`
        repo_type (`str`, *optional*):
            Set to `"dataset"` or `"space"` if getting repository info from a dataset or a space,
            `None` or `"model"` if getting repository info from a model. Default is `None`.
        revision (`str`, *optional*):
            The revision of the repository from which to get the information. Defaults to `"main"` branch.
        token (`bool` or `str`, *optional*):
            A valid authentication token (see https://huggingface.co/settings/token).
            If `None` or `True` and machine is logged in (through `huggingface-cli login`
            or [`~login`]), token will be retrieved from the cache.
            If `False`, token is not sent in the request header.

    Returns:
        True if the file exists, False otherwise.

    <Tip>

    Examples:
        ```py
        >>> from huggingface_hub import file_exists
        >>> file_exists("bigcode/starcoder", "config.json")
        True
        >>> file_exists("bigcode/starcoder", "not-a-file")
        False
        >>> file_exists("bigcode/not-a-repo", "config.json")
        False
        ```

    </Tip>
    """
    url = aistudio_hub_url(
        repo_id=repo_id, repo_type=repo_type, revision=revision, filename=filename, endpoint=endpoint
    )
    try:
        if token is None:
            token = get_token()
        get_aistudio_file_metadata(url, token=token)
        return True
    except GatedRepoError:  # raise specifically on gated repo
        raise
    except (RepositoryNotFoundError, EntryNotFoundError, RevisionNotFoundError, HfHubHTTPError):
        return False


def aistudio_hub_try_to_load_from_cache(
    repo_id: str,
    filename: str,
    cache_dir: Union[str, Path, None] = None,
    revision: Optional[str] = None,
    repo_type: Optional[str] = None,
):
    if revision is None:
        revision = DEFAULT_REVISION
    if repo_type is None:
        repo_type = REPO_TYPES[-1]
    if repo_type not in REPO_TYPES:
        raise ValueError(f"Invalid repo type: {repo_type}. Accepted repo types are: {str(REPO_TYPES)}")
    if cache_dir is None:
        cache_dir = AISTUDIO_HUB_CACHE

    object_id = repo_id.replace("/", "--")
    repo_cache = os.path.join(cache_dir, f"{repo_type}s--{object_id}")
    if not os.path.isdir(repo_cache):
        # No cache for this model
        return None

    refs_dir = os.path.join(repo_cache, "refs")
    snapshots_dir = os.path.join(repo_cache, "snapshots")
    no_exist_dir = os.path.join(repo_cache, ".no_exist")

    # Resolve refs (for instance to convert main to the associated commit sha)
    if os.path.isdir(refs_dir):
        revision_file = os.path.join(refs_dir, revision)
        if os.path.isfile(revision_file):
            with open(revision_file) as f:
                revision = f.read()

    # Check if file is cached as "no_exist"
    if os.path.isfile(os.path.join(no_exist_dir, revision, filename)):
        return None

    # Check if revision folder exists
    if not os.path.exists(snapshots_dir):
        return None
    cached_shas = os.listdir(snapshots_dir)
    if revision not in cached_shas:
        # No cache for this revision and we won't try to return a random revision
        return None

    # Check if file exists in cache
    cached_file = os.path.join(snapshots_dir, revision, filename)
    return cached_file if os.path.isfile(cached_file) else None
