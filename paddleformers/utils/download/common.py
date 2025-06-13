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

import copy
import logging
import os
import re
import shutil
import stat
import tempfile
import threading
import time
import uuid
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import BinaryIO, Callable, Dict, Generator, Literal, Optional, Union
from urllib.parse import urlparse

import requests
from huggingface_hub.utils import (
    BadRequestError,
    EntryNotFoundError,
    HfHubHTTPError,
    tqdm,
)
from requests import HTTPError, Response
from requests.adapters import HTTPAdapter
from requests.models import PreparedRequest

logger = logging.getLogger(__name__)

ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}


def _is_true(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.upper() in ENV_VARS_TRUE_VALUES


def _as_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    return int(value)


DISABLE_SYMLINKS_WARNING = False
# Regex to get filename from a "Content-Disposition" header for CDN-served files
HEADER_FILENAME_PATTERN = re.compile(r'filename="(?P<filename>.*?)"')
DOWNLOAD_CHUNK_SIZE = 10 * 1024 * 1024
REPO_ID_SEPARATOR = "--"

DEFAULT_DOWNLOAD_TIMEOUT = 10
DEFAULT_REQUEST_TIMEOUT = 10
DEFAULT_ETAG_TIMEOUT = 10
DEFALUT_LOCAL_DIR_AUTO_SYMLINK_THRESHOLD: int = 5 * 1024 * 1024

OFFLINE = _is_true(os.environ.get("AISTUDIO_BOS_OFFLINE"))
_CACHED_NO_EXIST = object()


def _cache_commit_hash_for_specific_revision(storage_folder: str, revision: str, commit_hash: str) -> None:
    """Cache reference between a revision (tag, branch or truncated commit hash) and the corresponding commit hash.

    Does nothing if `revision` is already a proper `commit_hash` or reference is already cached.
    """
    # if revision != commit_hash:
    ref_path = Path(storage_folder) / "refs" / revision
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    if not ref_path.exists() or commit_hash != ref_path.read_text():
        # Update ref only if has been updated. Could cause useless error in case
        # repo is already cached and user doesn't have write access to cache folder.
        # See https://github.com/huggingface/huggingface_hub/issues/1216.
        ref_path.write_text(commit_hash)


def _check_disk_space(expected_size: int, target_dir: Union[str, Path]) -> None:
    """Check disk usage and log a warning if there is not enough disk space to download the file.

    Args:
        expected_size (`int`):
            The expected size of the file in bytes.
        target_dir (`str`):
            The directory where the file will be stored after downloading.
    """

    target_dir = Path(target_dir)  # format as `Path`
    for path in [target_dir] + list(target_dir.parents):  # first check target_dir, then each parents one by one
        try:
            target_dir_free = shutil.disk_usage(path).free
            if target_dir_free < expected_size:
                warnings.warn(
                    "Not enough free disk space to download the file. "
                    f"The expected file size is: {expected_size / 1e6:.2f} MB. "
                    f"The target location {target_dir} only has {target_dir_free / 1e6:.2f} MB free disk space."
                )
            return
        except OSError:  # raise on anything: file does not exist or space disk cannot be checked
            pass


def http_get(
    url: str,
    temp_file: BinaryIO,
    *,
    proxies=None,
    resume_size: float = 0,
    headers: Optional[Dict[str, str]] = None,
    expected_size: Optional[int] = None,
    _nb_retries: int = 5,
):
    """
    Download a remote file. Do not gobble up errors, and will return errors tailored to the Hugging Face Hub.

    If ConnectionError (SSLError) or ReadTimeout happen while streaming data from the server, it is most likely a
    transient error (network outage?). We log a warning message and try to resume the download a few times before
    giving up. The method gives up after 5 attempts if no new data has being received from the server.
    """
    initial_headers = headers
    headers = copy.deepcopy(headers) or {}
    if resume_size > 0:
        headers["Range"] = "bytes=%d-" % (resume_size,)

    r = _request_wrapper(
        method="GET", url=url, stream=True, proxies=proxies, headers=headers, timeout=DEFAULT_DOWNLOAD_TIMEOUT
    )
    raise_for_status(r)
    content_length = r.headers.get("Content-Length")

    # NOTE: 'total' is the total number of bytes to download, not the number of bytes in the file.
    #       If the file is compressed, the number of bytes in the saved file will be higher than 'total'.
    total = resume_size + int(content_length) if content_length is not None else None

    displayed_name = url
    content_disposition = r.headers.get("Content-Disposition")
    if content_disposition is not None:
        match = HEADER_FILENAME_PATTERN.search(content_disposition)
        if match is not None:
            # Means file is on CDN
            displayed_name = match.groupdict()["filename"]

    # Truncate filename if too long to display
    if len(displayed_name) > 40:
        displayed_name = f"(…){displayed_name[-40:]}"

    consistency_error_message = (
        f"Consistency check failed: file should be of size {expected_size} but has size"
        f" {{actual_size}} ({displayed_name}).\nWe are sorry for the inconvenience. Please retry download and"
        " pass `force_download=True, resume_download=False` as argument.\nIf the issue persists, please let us"
        " know by opening an issue on https://github.com/huggingface/huggingface_hub."
    )

    # Stream file to buffer
    with tqdm(
        unit="B",
        unit_scale=True,
        total=total,
        initial=resume_size,
        desc=displayed_name,
        disable=bool(logger.getEffectiveLevel() == logging.NOTSET),
    ) as progress:
        new_resume_size = resume_size
        try:
            for chunk in r.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    progress.update(len(chunk))
                    temp_file.write(chunk)
                    new_resume_size += len(chunk)
                    # Some data has been downloaded from the server so we reset the number of retries.
                    _nb_retries = 5
        except (requests.ConnectionError, requests.ReadTimeout) as e:
            # If ConnectionError (SSLError) or ReadTimeout happen while streaming data from the server, it is most likely
            # a transient error (network outage?). We log a warning message and try to resume the download a few times
            # before giving up. The retry mechanism is basic but should be enough in most cases.
            if _nb_retries <= 0:
                logger.warning("Error while downloading from %s: %s\nMax retries exceeded.", url, str(e))
                raise
            logger.warning("Error while downloading from %s: %s\nTrying to resume download...", url, str(e))
            time.sleep(1)
            reset_sessions()  # In case of SSLError it's best to reset the shared requests.Session objects
            return http_get(
                url=url,
                temp_file=temp_file,
                proxies=proxies,
                resume_size=new_resume_size,
                headers=initial_headers,
                expected_size=expected_size,
                _nb_retries=_nb_retries - 1,
            )

        if expected_size is not None and expected_size != temp_file.tell():
            raise EnvironmentError(
                consistency_error_message.format(
                    actual_size=temp_file.tell(),
                )
            )


def _chmod_and_replace(src: str, dst: str) -> None:
    """Set correct permission before moving a blob from tmp directory to cache dir.

    Do not take into account the `umask` from the process as there is no convenient way
    to get it that is thread-safe.

    See:
    - About umask: https://docs.python.org/3/library/os.html#os.umask
    - Thread-safety: https://stackoverflow.com/a/70343066
    - About solution: https://github.com/huggingface/huggingface_hub/pull/1220#issuecomment-1326211591
    - Fix issue: https://github.com/huggingface/huggingface_hub/issues/1141
    - Fix issue: https://github.com/huggingface/huggingface_hub/issues/1215
    """
    # Get umask by creating a temporary file in the cached repo folder.
    tmp_file = Path(dst).parent.parent / f"tmp_{uuid.uuid4()}"
    try:
        tmp_file.touch()
        cache_dir_mode = Path(tmp_file).stat().st_mode
        os.chmod(src, stat.S_IMODE(cache_dir_mode))
    finally:
        tmp_file.unlink()

    shutil.move(src, dst)


def repo_folder_name(*, repo_id: str, repo_type: str) -> str:
    """Return a serialized version of a aistudio repo name and type, safe for disk storage
    as a single non-nested folder.

    Example: models--julien-c--EsperBERTo-small
    """
    # remove all `/` occurrences to correctly convert repo to directory name
    parts = [f"{repo_type}s", *repo_id.split("/")]
    return REPO_ID_SEPARATOR.join(parts)


class OfflineModeIsEnabled(ConnectionError):
    """Raised when a request is made but `AISTUDIO_HUB_OFFLINE=1` is set as environment variable."""


class OfflineAdapter(HTTPAdapter):
    def send(self, request: PreparedRequest, *args, **kwargs) -> Response:
        raise OfflineModeIsEnabled(
            f"Cannot reach {request.url}: offline mode is enabled. To disable it, please unset the `AISTUDIO_HUB_OFFLINE` environment variable."
        )


BACKEND_FACTORY_T = Callable[[], requests.Session]


def _default_backend_factory() -> requests.Session:
    session = requests.Session()
    if OFFLINE:
        session.mount("http://", OfflineAdapter())
        session.mount("https://", OfflineAdapter())

    return session


_GLOBAL_BACKEND_FACTORY: BACKEND_FACTORY_T = _default_backend_factory
HTTP_METHOD_T = Literal["GET", "OPTIONS", "HEAD", "POST", "PUT", "PATCH", "DELETE"]


@lru_cache
def _get_session_from_cache(process_id: int, thread_id: int) -> requests.Session:
    """
    Create a new session per thread using global factory. Using LRU cache (maxsize 128) to avoid memory leaks when
    using thousands of threads. Cache is cleared when `configure_http_backend` is called.
    """
    return _GLOBAL_BACKEND_FACTORY()


def reset_sessions() -> None:
    """Reset the cache of sessions.

    Mostly used internally when sessions are reconfigured or an SSLError is raised.
    See [`configure_http_backend`] for more details.
    """
    _get_session_from_cache.cache_clear()


def get_session() -> requests.Session:
    """
    Get a `requests.Session` object, using the session factory from the user.

    Use [`get_session`] to get a configured Session. Since `requests.Session` is not guaranteed to be thread-safe,
    `huggingface_hub` creates 1 Session instance per thread. They are all instantiated using the same `backend_factory`
    set in [`configure_http_backend`]. A LRU cache is used to cache the created sessions (and connections) between
    calls. Max size is 128 to avoid memory leaks if thousands of threads are spawned.

    See [this issue](https://github.com/psf/requests/issues/2766) to know more about thread-safety in `requests`.

    Example:
    ```py
    import requests
    from huggingface_hub import configure_http_backend, get_session

    # Create a factory function that returns a Session with configured proxies
    def backend_factory() -> requests.Session:
        session = requests.Session()
        session.proxies = {"http": "http://10.10.1.10:3128", "https": "https://10.10.1.11:1080"}
        return session

    # Set it as the default session factory
    configure_http_backend(backend_factory=backend_factory)

    # In practice, this is mostly done internally in `huggingface_hub`
    session = get_session()
    ```
    """
    return _get_session_from_cache(process_id=os.getpid(), thread_id=threading.get_ident())


def _request_wrapper(
    method: HTTP_METHOD_T, url: str, *, follow_relative_redirects: bool = False, **params
) -> requests.Response:
    """Wrapper around requests methods to follow relative redirects if `follow_relative_redirects=True` even when
    `allow_redirection=False`.

    Args:
        method (`str`):
            HTTP method, such as 'GET' or 'HEAD'.
        url (`str`):
            The URL of the resource to fetch.
        follow_relative_redirects (`bool`, *optional*, defaults to `False`)
            If True, relative redirection (redirection to the same site) will be resolved even when `allow_redirection`
            kwarg is set to False. Useful when we want to follow a redirection to a renamed repository without
            following redirection to a CDN.
        **params (`dict`, *optional*):
            Params to pass to `requests.request`.
    """
    # Recursively follow relative redirects
    if follow_relative_redirects:
        response = _request_wrapper(
            method=method,
            url=url,
            follow_relative_redirects=False,
            **params,
        )

        # If redirection, we redirect only relative paths.
        # This is useful in case of a renamed repository.
        if 300 <= response.status_code <= 399:
            parsed_target = urlparse(response.headers["Location"])
            if parsed_target.netloc == "":
                # This means it is a relative 'location' headers, as allowed by RFC 7231.
                # (e.g. '/path/to/resource' instead of 'http://domain.tld/path/to/resource')
                # We want to follow this relative redirect !
                #
                # Highly inspired by `resolve_redirects` from requests library.
                # See https://github.com/psf/requests/blob/main/requests/sessions.py#L159
                next_url = urlparse(url)._replace(path=parsed_target.path).geturl()
                return _request_wrapper(method=method, url=next_url, follow_relative_redirects=True, **params)
        return response
    # Perform request and return if status_code is not in the retry list.
    response = get_session().request(method=method, url=url, **params)
    raise_for_status(response)
    return response


def _get_pointer_path(storage_folder: str, revision: str, relative_filename: str) -> str:
    # Using `os.path.abspath` instead of `Path.resolve()` to avoid resolving symlinks
    snapshot_path = os.path.join(storage_folder, "snapshots")
    pointer_path = os.path.join(snapshot_path, revision, relative_filename)
    if Path(os.path.abspath(snapshot_path)) not in Path(os.path.abspath(pointer_path)).parents:
        raise ValueError(
            "Invalid pointer path: cannot create pointer path in snapshot folder if"
            f" `storage_folder='{storage_folder}'`, `revision='{revision}'` and"
            f" `relative_filename='{relative_filename}'`."
        )
    return pointer_path


def _create_symlink(src: str, dst: str, new_blob: bool = False) -> None:
    """Create a symbolic link named dst pointing to src.

    By default, it will try to create a symlink using a relative path. Relative paths have 2 advantages:
    - If the cache_folder is moved (example: back-up on a shared drive), relative paths within the cache folder will
      not brake.
    - Relative paths seems to be better handled on Windows. Issue was reported 3 times in less than a week when
      changing from relative to absolute paths. See https://github.com/huggingface/huggingface_hub/issues/1398,
      https://github.com/huggingface/diffusers/issues/2729 and https://github.com/huggingface/transformers/pull/22228.
      NOTE: The issue with absolute paths doesn't happen on admin mode.
    When creating a symlink from the cache to a local folder, it is possible that a relative path cannot be created.
    This happens when paths are not on the same volume. In that case, we use absolute paths.


    The result layout looks something like
        └── [ 128]  snapshots
            ├── [ 128]  2439f60ef33a0d46d85da5001d52aeda5b00ce9f
            │   ├── [  52]  README.md -> ../../../blobs/d7edf6bd2a681fb0175f7735299831ee1b22b812
            │   └── [  76]  pytorch_model.bin -> ../../../blobs/403450e234d65943a7dcf7e05a771ce3c92faa84dd07db4ac20f592037a1e4bd

    If symlinks cannot be created on this platform (most likely to be Windows), the workaround is to avoid symlinks by
    having the actual file in `dst`. If it is a new file (`new_blob=True`), we move it to `dst`. If it is not a new file
    (`new_blob=False`), we don't know if the blob file is already referenced elsewhere. To avoid breaking existing
    cache, the file is duplicated on the disk.

    In case symlinks are not supported, a warning message is displayed to the user once when loading `huggingface_hub`.
    The warning message can be disable with the `DISABLE_SYMLINKS_WARNING` environment variable.
    """
    try:
        os.remove(dst)
    except OSError:
        pass

    abs_src = os.path.abspath(os.path.expanduser(src))
    abs_dst = os.path.abspath(os.path.expanduser(dst))
    abs_dst_folder = os.path.dirname(abs_dst)

    # Use relative_dst in priority
    try:
        relative_src = os.path.relpath(abs_src, abs_dst_folder)
    except ValueError:
        # Raised on Windows if src and dst are not on the same volume. This is the case when creating a symlink to a
        # local_dir instead of within the cache directory.
        # See https://docs.python.org/3/library/os.path.html#os.path.relpath
        relative_src = None

    try:
        commonpath = os.path.commonpath([abs_src, abs_dst])
        _support_symlinks = are_symlinks_supported(commonpath)
    except ValueError:
        # Raised if src and dst are not on the same volume. Symlinks will still work on Linux/Macos.
        # See https://docs.python.org/3/library/os.path.html#os.path.commonpath
        _support_symlinks = os.name != "nt"
    except PermissionError:
        # Permission error means src and dst are not in the same volume (e.g. destination path has been provided
        # by the user via `local_dir`. Let's test symlink support there)
        _support_symlinks = are_symlinks_supported(abs_dst_folder)

    # Symlinks are supported => let's create a symlink.
    if _support_symlinks:
        src_rel_or_abs = relative_src or abs_src
        logger.debug(f"Creating pointer from {src_rel_or_abs} to {abs_dst}")
        try:
            os.symlink(src_rel_or_abs, abs_dst)
            return
        except FileExistsError:
            if os.path.islink(abs_dst) and os.path.realpath(abs_dst) == os.path.realpath(abs_src):
                # `abs_dst` already exists and is a symlink to the `abs_src` blob. It is most likely that the file has
                # been cached twice concurrently (exactly between `os.remove` and `os.symlink`). Do nothing.
                return
            else:
                # Very unlikely to happen. Means a file `dst` has been created exactly between `os.remove` and
                # `os.symlink` and is not a symlink to the `abs_src` blob file. Raise exception.
                raise
        except PermissionError:
            # Permission error means src and dst are not in the same volume (e.g. download to local dir) and symlink
            # is supported on both volumes but not between them. Let's just make a hard copy in that case.
            pass

    # Symlinks are not supported => let's move or copy the file.
    if new_blob:
        logger.info(f"Symlink not supported. Moving file from {abs_src} to {abs_dst}")
        shutil.move(abs_src, abs_dst)
    else:
        logger.info(f"Symlink not supported. Copying file from {abs_src} to {abs_dst}")
        shutil.copyfile(abs_src, abs_dst)


_are_symlinks_supported_in_dir: Dict[str, bool] = {}


def _set_write_permission_and_retry(func, path, excinfo):
    os.chmod(path, stat.S_IWRITE)
    func(path)


@contextmanager
def SoftTemporaryDirectory(
    suffix: Optional[str] = None,
    prefix: Optional[str] = None,
    dir: Optional[Union[Path, str]] = None,
    **kwargs,
) -> Generator[str, None, None]:
    """
    Context manager to create a temporary directory and safely delete it.

    If tmp directory cannot be deleted normally, we set the WRITE permission and retry.
    If cleanup still fails, we give up but don't raise an exception. This is equivalent
    to  `tempfile.TemporaryDirectory(..., ignore_cleanup_errors=True)` introduced in
    Python 3.10.

    See https://www.scivision.dev/python-tempfile-permission-error-windows/.
    """
    tmpdir = tempfile.TemporaryDirectory(prefix=prefix, suffix=suffix, dir=dir, **kwargs)
    yield tmpdir.name

    try:
        # First once with normal cleanup
        shutil.rmtree(tmpdir.name)
    except Exception:
        # If failed, try to set write permission and retry
        try:
            shutil.rmtree(tmpdir.name, onerror=_set_write_permission_and_retry)
        except Exception:
            pass

    # And finally, cleanup the tmpdir.
    # If it fails again, give up but do not throw error
    try:
        tmpdir.cleanup()
    except Exception:
        pass


def _to_local_dir(
    path: str, local_dir: str, relative_filename: str, use_symlinks: Union[bool, Literal["auto"]]
) -> str:
    """Place a file in a local dir (different than cache_dir).

    Either symlink to blob file in cache or duplicate file depending on `use_symlinks` and file size.
    """
    # Using `os.path.abspath` instead of `Path.resolve()` to avoid resolving symlinks
    local_dir_filepath = os.path.join(local_dir, relative_filename)
    if Path(os.path.abspath(local_dir)) not in Path(os.path.abspath(local_dir_filepath)).parents:
        raise ValueError(
            f"Cannot copy file '{relative_filename}' to local dir '{local_dir}': file would not be in the local"
            " directory."
        )

    os.makedirs(os.path.dirname(local_dir_filepath), exist_ok=True)
    real_blob_path = os.path.realpath(path)

    # If "auto" (default) copy-paste small files to ease manual editing but symlink big files to save disk
    if use_symlinks == "auto":
        use_symlinks = os.stat(real_blob_path).st_size > DEFALUT_LOCAL_DIR_AUTO_SYMLINK_THRESHOLD

    if use_symlinks:
        _create_symlink(real_blob_path, local_dir_filepath, new_blob=False)
    else:
        shutil.copyfile(real_blob_path, local_dir_filepath)
    return local_dir_filepath


def _normalize_etag(etag: Optional[str]) -> Optional[str]:
    """Normalize ETag HTTP header, so it can be used to create nice filepaths.

    The HTTP spec allows two forms of ETag:
      ETag: W/"<etag_value>"
      ETag: "<etag_value>"

    For now, we only expect the second form from the server, but we want to be future-proof so we support both. For
    more context, see `TestNormalizeEtag` tests and https://github.com/huggingface/huggingface_hub/pull/1428.

    Args:
        etag (`str`, *optional*): HTTP header

    Returns:
        `str` or `None`: string that can be used as a nice directory name.
        Returns `None` if input is None.
    """
    if etag is None:
        return None
    return etag.lstrip("W/").strip('"')


@dataclass(frozen=True)
class AistudioBosFileMetadata:
    """Data structure containing information about a file versioned on the Aistudio Hub.

    Returned by [`get_aistudio_file_metadata`] based on a URL.

    Args:
        commit_hash (`str`, *optional*):
            The commit_hash related to the file.
        etag (`str`, *optional*):
            Etag of the file on the server.
        location (`str`):
            Location where to download the file. Can be a Hub url or not (CDN).
        size (`size`):
            Size of the file. In case of an LFS file, contains the size of the actual
            LFS file, not the pointer.
    """

    commit_hash: Optional[str]
    etag: Optional[str]
    location: str
    size: Optional[int]


def raise_for_status(response: Response, endpoint_name: Optional[str] = None) -> None:
    try:
        response.raise_for_status()
    except HTTPError as e:
        if response.status_code == 404:
            message = f"{response.status_code} Client Error." + "\n\n" + f"Entry Not Found for url: {response.url}."
            raise EntryNotFoundError(message, None) from e
        elif response.status_code == 400:
            message = (
                f"\n\nBad request for {endpoint_name} endpoint:" if endpoint_name is not None else "\n\nBad request:"
            )
            raise BadRequestError(message, response=None) from e
        raise HfHubHTTPError(str(e), response=None) from e


def are_symlinks_supported(cache_dir: Union[str, Path, None] = None) -> bool:
    """Return whether the symlinks are supported on the machine.

    Since symlinks support can change depending on the mounted disk, we need to check
    on the precise cache folder.

    Args:
        cache_dir (`str`, `Path`, *optional*):
            Path to the folder where cached files are stored.

    Returns: [bool] Whether symlinks are supported in the directory.
    """
    assert cache_dir is not None
    cache_dir = str(Path(cache_dir).expanduser().resolve())  # make it unique

    # Check symlink compatibility only once (per cache directory) at first time use
    if cache_dir not in _are_symlinks_supported_in_dir:
        _are_symlinks_supported_in_dir[cache_dir] = True

        os.makedirs(cache_dir, exist_ok=True)
        with SoftTemporaryDirectory(dir=cache_dir) as tmpdir:
            src_path = Path(tmpdir) / "dummy_file_src"
            src_path.touch()
            dst_path = Path(tmpdir) / "dummy_file_dst"

            # Relative source path as in `_create_symlink``
            relative_src = os.path.relpath(src_path, start=os.path.dirname(dst_path))
            try:
                os.symlink(relative_src, dst_path)
            except OSError:
                # Likely running on Windows
                _are_symlinks_supported_in_dir[cache_dir] = False

                if not DISABLE_SYMLINKS_WARNING:
                    message = (
                        "cache-system uses symlinks by default to"
                        " efficiently store duplicated files but your machine does not"
                        f" support them in {cache_dir}. Caching files will still work"
                        " but in a degraded version that might require more space on"
                        " your disk. This warning can be disabled by setting the"
                        " `DISABLE_SYMLINKS_WARNING` environment variable."
                    )
                    if os.name == "nt":
                        message += (
                            "\nTo support symlinks on Windows, you either need to"
                            " activate Developer Mode or to run Python as an"
                            " administrator. In order to see activate developer mode,"
                            " see this article:"
                            " https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development"
                        )
                    warnings.warn(message)

    return _are_symlinks_supported_in_dir[cache_dir]
