# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from huggingface_hub import snapshot_download

from paddleformers.utils.log import logger


def HuggingFaceDownload(repo_id, download_path, resume_download=True, max_workers=16):
    hf_download_proxy = os.getenv("https_proxy")
    if hf_download_proxy is None:
        hf_download_proxy = os.getenv("HTTPS_PROXY")
    logger.info(f"HuggingFace dataset downloading..., the proxy is {hf_download_proxy}")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        proxies={"http": hf_download_proxy},
        resume_download=resume_download,
        max_workers=max_workers,
        local_dir=download_path,
    )
