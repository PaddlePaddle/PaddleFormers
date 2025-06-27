# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import errno
import io
import os
import re
import subprocess
from datetime import datetime

import setuptools

PADDLEFORMERS_STABLE_VERSION = "PADDLEFORMERS_STABLE_VERSION"


def read_requirements_file(filepath):
    with open(filepath) as fin:
        requirements = fin.read()
    return requirements


def is_git_repo(dir: str) -> bool:
    """Is the given directory version-controlled with git?"""
    return os.path.exists(os.path.join(dir, ".git"))


def have_git() -> bool:
    """Can we run the git executable?"""
    try:
        subprocess.check_output(["git", "--help"])
        return True
    except subprocess.CalledProcessError:
        return False
    except OSError:
        return False


def git_revision(dir: str) -> bytes:
    """Get the SHA-1 of the HEAD of a git repository."""
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=dir).strip()


def git_checkout(dir: str, filename: str) -> bytes:
    """Get the SHA-1 of the HEAD of a git repository."""
    return subprocess.check_output(["git", "checkout", filename], cwd=dir).strip()


def is_dirty(dir: str) -> bool:
    """Check whether a git repository has uncommitted changes."""
    output = subprocess.check_output(["git", "status", "-uno", "--porcelain"], cwd=dir)
    return output.strip() != b""


commit = "unknown"
paddleformers_dir = os.path.abspath(os.path.dirname(__file__))
if commit.endswith("unknown") and is_git_repo(paddleformers_dir) and have_git():
    commit = git_revision(paddleformers_dir).decode("utf-8")
    if is_dirty(paddleformers_dir):
        commit += ".dirty"


def write_version_py(filename="paddleformers/version/__init__.py"):
    cnt = '''# THIS FILE IS GENERATED FROM PADDLEFORMERS SETUP.PY
commit           = '%(commit)s'

__all__ = ['show']

def show():
    """Get the corresponding commit id of paddleformers.

    Returns:
        The commit-id of paddleformers will be output.

        full_version: version of paddleformers


    Examples:
        .. code-block:: python

            import paddleformers

            paddleformers.version.show()
            # commit: 1ef5b94a18773bb0b1bba1651526e5f5fc5b16fa

    """
    print("commit:", commit)

'''
    commit_id = commit
    content = cnt % {"commit": commit_id}

    dirname = os.path.dirname(filename)

    try:
        os.makedirs(dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    with open(filename, "w") as f:
        f.write(content)


# only use this file to contral the version
__version__ = "0.0.0.post"
if os.getenv(PADDLEFORMERS_STABLE_VERSION):
    __version__ = __version__.replace(".post", "")
else:
    formatted_date = datetime.now().date().strftime("%Y%m%d")
    __version__ = __version__.replace(".post", ".post{}".format(formatted_date))


# write the version information for the develop version
def append_version_py(filename="paddleformers/__init__.py"):
    assert os.path.exists(filename), f"{filename} does not exist!"

    with open(filename, "r") as file:
        file_content = file.read()

    pattern = r"^# \[VERSION_INFO\].*$"
    modified_content = re.sub(pattern, f'\n__version__ = "{__version__}"\n\n', file_content, flags=re.MULTILINE)
    with open(filename, "w") as file:
        file.write(modified_content)


append_version_py(filename="paddleformers/__init__.py")

extras = {}
REQUIRED_PACKAGES = read_requirements_file("requirements.txt")


def read(*names, **kwargs):
    with io.open(os.path.join(os.path.dirname(__file__), *names), encoding=kwargs.get("encoding", "utf8")) as fp:
        return fp.read()


def get_package_data_files(package, data, package_dir=None):
    """
    Helps to list all specified files in package including files in directories
    since `package_data` ignores directories.
    """
    if package_dir is None:
        package_dir = os.path.join(*package.split("."))
    all_files = []
    for f in data:
        path = os.path.join(package_dir, f)
        if os.path.isfile(path):
            all_files.append(f)
            continue
        for root, _dirs, files in os.walk(path, followlinks=True):
            root = os.path.relpath(root, package_dir)
            for file in files:
                file = os.path.join(root, file)
                if file not in all_files:
                    all_files.append(file)
    return all_files


if commit != "unknown":
    write_version_py(filename="paddleformers/version/__init__.py")

try:
    setuptools.setup(
        name="paddleformers",
        version=__version__,
        author="PaddleFormers Team",
        author_email="paddleformers@baidu.com",
        description="PaddleFormers is an easy-to-use library of pre-trained large language model zoo based on PaddlePaddle.",
        long_description=read("README_en.md"),
        long_description_content_type="text/markdown",
        url="https://github.com/PaddlePaddle/PaddleFormers",
        license_files=("LICENSE",),
        packages=setuptools.find_packages(
            where=".",
            exclude=("tests*"),
        ),
        package_data={},
        setup_requires=["cython", "numpy"],
        install_requires=REQUIRED_PACKAGES,
        entry_points={"console_scripts": ["paddleformers = paddleformers.cli:main"]},
        extras_require=extras,
        python_requires=">=3.8",
        classifiers=[
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
        ],
        license="Apache 2.0",
    )
except Exception as e:
    git_checkout(paddleformers_dir, "paddleformers/version/__init__.py") if commit != "unknown" else None
    git_checkout(paddleformers_dir, "paddleformers/__init__.py") if commit != "unknown" else None
    raise e

git_checkout(paddleformers_dir, "paddleformers/version/__init__.py") if commit != "unknown" else None
git_checkout(paddleformers_dir, "paddleformers/__init__.py") if commit != "unknown" else None
