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

import json
import os
import queue
import shutil
import subprocess
import threading
import time
from distutils.dir_util import copy_tree
from enum import Enum
from typing import List

from .log import logger

PDC_AGENT_BIN = "/root/paddlejob/tools/agent"
HASH_SUM_BIN = "/root/paddlejob/afs_tool/bin/b3sum"
TRAIN_CONFIG = "/root/paddlejob/workspace/env_run/longjob/train.conf"
TAR_BIN = "tar"

FLASH_DEVICE = os.getenv("PDC_FLASH_DEVICE", "/shared/dev/shm/flash")


def pdc_flash_device_available():
    # TODO(@gexiao): need better check
    return os.path.exists(FLASH_DEVICE)


class PDCErrorCode(Enum):
    """Error Code For PDCTools usage"""

    # success
    Success = 0

    RemotePathNotExist = 1404
    LocalPathExist = 1405
    DownloadFail = 1406
    AgentConfigInvalid = 1407
    AFSToolsNotExist = 1408
    TrainConfigNotExist = 1409
    LocalPathNotExist = 1410

    CommandFail = 1501
    CalculateHashFail = 1502
    InvalidArgument = 1503
    CommandTimeout = 1504
    CheckSumCommandFail = 1505
    CopyTreeFailed = 1506

    UnknownError = 1999


class PDCTools:
    """PDCTools"""

    def __init__(self):
        """ """
        self._pdc_agent_bin = PDC_AGENT_BIN
        self._hash_sum_bin = HASH_SUM_BIN
        self._train_config = TRAIN_CONFIG
        self._tar_bin = TAR_BIN

    def pdc_upload(self, remote_path: str, local_path: str) -> PDCErrorCode:
        """upload data to afs/bos
        1. tar local path
        2. calculate the hash of tar file
        3. upload tar to remote path

        Args:
        remote_path str: the remote file path, afs/bos, such as afs://user/a/b/xx.tar
        local_path str: local file path

        Return:
        """
        pre_check_status = self._pre_check()
        if pre_check_status != PDCErrorCode.Success:
            return pre_check_status
        # check local path
        if not os.path.exists(local_path):
            logger.error(f"{local_path} not exist")
            return PDCErrorCode.LocalPathNotExist
        if not remote_path.endswith(".tar"):
            logger.warning(f"remote path {remote_path} should end with .tar")
            return PDCErrorCode.InvalidArgument

        try:
            # get tar name
            remote_dir = os.path.dirname(remote_path)
            tar_file_name = os.path.basename(remote_path)
            # tar local path
            status = self._tar_file(local_path, tar_file_name)
            if status != PDCErrorCode.Success:
                logger.error(f"tar local path {local_path} failed")
                return status
            # calc hash
            b3sum_hash, status = self._calculate_hash(tar_file_name)
            if status != PDCErrorCode.Success:
                logger.error(f"calculate hash for {tar_file_name} failed")
                return status
            logger.info(f"local tar: {tar_file_name}, b3sum hash: {b3sum_hash}")

            # upload local tar to remote path
            status = self._upload_file(tar_file_name, remote_path)
            if status != PDCErrorCode.Success:
                logger.error(f"upload file {tar_file_name} failed")
                return status

            # upload b3sum hash to remote path
            local_b3sum_hash_file = f".{time.time()}_b3sum.hash"
            with open(local_b3sum_hash_file, "w") as f:
                f.write(b3sum_hash)
            remote_b3sum_path = os.path.join(remote_dir, self._get_file_hash_name(tar_file_name))
            status = self._upload_file(local_b3sum_hash_file, remote_b3sum_path)
            if status != PDCErrorCode.Success:
                logger.error(f"upload hash file {local_b3sum_hash_file} failed")
                return status

            # clean tmp files
            self._clean_tmp_files([local_b3sum_hash_file, tar_file_name])

            logger.info(f"successfully uploaded ${local_path} to remote path ${remote_path}")
            return PDCErrorCode.Success
        except Exception as e:
            logger.error(f"pdc upload failed: {e}")
            raise e

    def pdc_download(self, remote_path: str, local_path: str, timeout: int) -> PDCErrorCode:
        """
        download data from afs/bos

        Args:
        remote_path str: the remote file path, afs/bos, such as afs://user/a/b/xx.tar
        local_path str: local file directory
        timeout int: max wait time in seconds

        Return:
        PDCErrorCode: indicate the status of pdc download
        """

        def _download_worker(remote_path, local_path, queue):
            try:
                result = self._pdc_download_impl(remote_path, local_path)
                queue.put(result)
            except Exception as e:
                queue.put(str(e))
            return

        result_queue = queue.Queue()
        thread = threading.Thread(
            target=_download_worker,
            args=(
                remote_path,
                local_path,
                result_queue,
            ),
        )
        thread.start()
        logger.info(f"Begin downloading object of {remote_path} to {local_path} from PDC...")
        start_time = time.time()
        end_time = time.time()
        last_log_time = start_time
        while (end_time - start_time) < timeout:
            if not thread.is_alive():
                break
            if end_time - last_log_time > 30:
                # log every 30 seconds to avoid false detection by hangWatcher
                logger.info(f"Still waiting for download, already passed {end_time - start_time} seconds...")
                last_log_time = end_time
            time.sleep(1)
            end_time = time.time()
        if thread.is_alive():
            return PDCErrorCode.CommandTimeout
        result = result_queue.get()
        if isinstance(result, str):
            logger.error(f"Unknown exception occurred during download process, details: {result}")
            return PDCErrorCode.UnknownError
        return result

    def pdc_download_checkpoint(self, resume_step: int, timeout: int) -> PDCErrorCode:
        """
        download checkpoints from afs/bos

        Args:
        resume_step int: the resume step number
        timeout int: max wait time in seconds

        Return:
        PDCErrorCode: indicate the status of pdc download
        """

        def _download_worker(step, queue):
            try:
                result = self._pdc_download_checkpoint_impl(step)
                queue.put(result)
            except Exception as e:
                queue.put(str(e))
            return

        result_queue = queue.Queue()
        thread = threading.Thread(
            target=_download_worker,
            args=(
                resume_step,
                result_queue,
            ),
        )
        thread.start()
        logger.info(f"Begin downloading recovery checkpoint of step {resume_step} from PDC...")
        start_time = time.time()
        end_time = time.time()
        last_log_time = start_time
        while (end_time - start_time) < timeout:
            if not thread.is_alive():
                break
            if end_time - last_log_time > 30:
                # log every 30 seconds to avoid false detection by hangWatcher
                logger.info(f"Still waiting for download, already passed {end_time - start_time} seconds...")
                last_log_time = end_time
            time.sleep(1)
            end_time = time.time()
        if thread.is_alive():
            return PDCErrorCode.CommandTimeout
        result = result_queue.get()
        if isinstance(result, str):
            logger.error(f"Unknown exception occurred during download process, details: {result}")
            return PDCErrorCode.UnknownError
        return result

    def _pdc_download_impl(self, remote_path: str, local_path: str) -> PDCErrorCode:
        """download data from afs/bos

        Args:
        remote_path str: the remote file path, afs/bos, such as afs://user/a/b/xx.tar
        local_path str: local file directory

        Return:
        """
        pre_check_status = self._pre_check()
        if pre_check_status != PDCErrorCode.Success:
            return pre_check_status
        # check local path
        if os.path.exists(local_path):
            logger.info(f"local path {local_path} already exists")
            return PDCErrorCode.LocalPathExist
        if not remote_path.endswith(".tar"):
            logger.warning(f"remote path {remote_path} should end with .tar")
            return PDCErrorCode.InvalidArgument

        try:
            remote_dir = os.path.dirname(remote_path)
            file_name = os.path.basename(remote_path)
            # download remote file to local tmp path
            local_tmp_file_path = f".tmp_{time.time()}_{file_name}"
            status = self._download_file(remote_path, local_tmp_file_path)
            if status != PDCErrorCode.Success:
                logger.error(f"download remote file {file_name} failed")
                return status

            # download hash file to local path
            hash_file_name = self._get_file_hash_name(file_name)
            hash_file_path = os.path.join(remote_dir, hash_file_name)
            status = self._download_file(hash_file_path, hash_file_name)
            if status != PDCErrorCode.Success:
                logger.error(f"download remote hash file {hash_file_path} failed")
                return status
            remote_hash = ""
            with open(hash_file_name, "r") as f:
                remote_hash = f.read().strip()

            # calc hash
            local_hash, status = self._calculate_hash(local_tmp_file_path)
            if status != PDCErrorCode.Success:
                logger.error(f"calculate hash for {local_tmp_file_path} failed")
                return status
            logger.info(f"remote hash: {remote_hash}, local hash: {local_hash}")
            # check hash
            if local_hash != remote_hash:
                logger.error(f"local b3sum hash: {local_hash}, remote b3sum hash: {remote_hash}")
                return PDCErrorCode.CalculateHashFail

            # untar file to local_path
            status = self._untar_file(local_tmp_file_path, local_path)
            if status != PDCErrorCode.Success:
                logger.error(f"untar file {local_tmp_file_path} failed")
                return status
            # clean tmp files
            self._clean_tmp_files([local_tmp_file_path])
            return PDCErrorCode.Success
        except Exception as e:
            logger.error(f"pdc upload failed: {e}")
            raise e

    def _pdc_download_checkpoint_impl(self, step: int) -> PDCErrorCode:
        """ "download checkpoint from afs/bos

        Args:
        step int: the step of checkpoint

        """
        pre_check_status = self._pre_check()
        if pre_check_status != PDCErrorCode.Success:
            return pre_check_status

        conf = json.dumps(
            {
                "download_step": step,
            }
        )
        # download file from remote path
        download_cmd_args = [
            self._pdc_agent_bin,
            "-mode",
            "command",
            "-type",
            "download_checkpoint",
            "-config",
            f"{conf}",
        ]
        try:
            self._pre_check()
            logger.info(f"begin to download checkpoint from step {step}, config: {conf}")
            res, error_code = self._exec_cmd(download_cmd_args)
            if error_code == PDCErrorCode.Success:
                logger.info(f"download checkpoint from step {step} successfully")
            return error_code
        except Exception as e:
            logger.error(f"exec cmd {download_cmd_args} with error: {e}")
            raise Exception(f"exec cmd {download_cmd_args} with error: {e}")

    def _pre_check(self) -> PDCErrorCode:
        """check whether the environment is ready"""
        if not os.path.exists(self._pdc_agent_bin):
            logger.error(f"pdc tool {self._pdc_agent_bin} not found")
            return PDCErrorCode.AFSToolsNotExist
        if not os.path.exists(self._hash_sum_bin):
            logger.error(f"hash tool {self._hash_sum_bin} not found")
            return PDCErrorCode.AFSToolsNotExist
        if not os.path.exists(self._train_config):
            logger.error(f"train config {self._train_config} not found")
            return PDCErrorCode.TrainConfigNotExist
        # TODO(@zezhao): add more check
        return PDCErrorCode.Success

    def _exec_cmd(self, cmd_args: List[str]) -> (str, PDCErrorCode):
        """exec user command

        Args:
        cmd List[str]: command
        """
        error_code = PDCErrorCode.Success
        try:
            result = subprocess.run(cmd_args, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"exec cmd {cmd_args} successfully, result: {result.stdout}; {result.stderr}")
            else:
                logger.error(f"exec cmd {cmd_args} failed, exit code: {result.returncode}, err: {result.stderr}")
                # TODO(@zezhao): add more error code
                error_code = PDCErrorCode.CommandFail
            return result.stdout, error_code
        except Exception as e:
            logger.error(f"exec cmd {cmd_args} with error: {e}")
            raise Exception(f"exec cmd {cmd_args} with error: {e}")

    def _get_file_hash_name(self, file_name: str) -> str:
        """get the hash name of file

        Args:
        file_name str: file name

        Return:
        """
        if file_name.endswith(".tar"):
            file_name = file_name[:-4]
        return f"{file_name}.b3sumhash"

    def _calculate_hash(self, file_path: str) -> (str, PDCErrorCode):
        """calc the hash of file using b3sum

        Args:
        file_path str: file path

        Return:
        """
        cmd_args = [self._hash_sum_bin, "--num-threads", "16", file_path]
        try:
            result, error_code = self._exec_cmd(cmd_args)
            if error_code == PDCErrorCode.Success and len(result) > 0:
                return result.split(" ")[0].strip(), error_code
        except Exception as e:
            logger.error(f"exec cmd {cmd_args} with error: {e}")
            raise Exception(f"exec cmd {cmd_args} with error: {e}")
        return "", PDCErrorCode.CalculateHashFail

    def _tar_file(self, source_path: str, target_path: str) -> PDCErrorCode:
        """tar file with command
           tar -cf target_path -C source_path .

        Args:
        source_path str: source file path for tar
        target_path str: target file path
        """
        if not os.path.exists(source_path):
            logger.error(f"file {source_path} not exist")
            return PDCErrorCode.LocalPathNotExist
        if os.path.exists(target_path):
            os.rename(target_path, f"{target_path}.old")
            logger.warning(f"{target_path} already exists, backup it")

        error_code = PDCErrorCode.Success
        # tar file
        tar_cmd_args = [self._tar_bin, "-cf", target_path, "-C", source_path, "."]
        try:
            res, error_code = self._exec_cmd(tar_cmd_args)
            if error_code == PDCErrorCode.Success:
                logger.info(f"tar {source_path} successfully")
        except Exception as e:
            logger.error(f"exec cmd {tar_cmd_args} failed, error: {e}")
            raise Exception(f"exec cmd {tar_cmd_args} failed, error: {e}")
        return error_code

    def _untar_file(self, source_path: str, target_path: str) -> PDCErrorCode:
        """untar file
        Args:
        source_path str: source file path for untar
        target_path str: target file path
        """
        if not os.path.exists(source_path):
            logger.error(f"{source_path} not exist")
            return PDCErrorCode.LocalPathNotExist
        if not os.path.exists(target_path):
            # create target path if not exists
            os.makedirs(target_path)

        # untar file
        error_code = PDCErrorCode.Success
        untar_cmd_args = [self._tar_bin, "-xf", source_path, "-C", target_path]
        try:
            res, error_code = self._exec_cmd(untar_cmd_args)
            if error_code == PDCErrorCode.Success:
                logger.info(f"untar {source_path} successfully")
        except Exception as e:
            logger.error(f"exec cmd {untar_cmd_args} with error: {e}")
            raise Exception(f"exec cmd {untar_cmd_args} with error: {e}")
        return error_code

    def _upload_file(self, local_file_path: str, remote_path: str) -> PDCErrorCode:
        """upload file
        Args:
        local_file_path str: local file path
        remote_path str: remote file path
        """
        if not os.path.exists(local_file_path):
            logger.error(f"{local_file_path} not exist")
            return PDCErrorCode.LocalPathNotExist

        conf = json.dumps({"remote_path": remote_path, "local_path": local_file_path})
        # upload file to remote path
        upload_cmd_args = [self._pdc_agent_bin, "-mode", "command", "-type", "upload", "-config", f"{conf}"]
        error_code = PDCErrorCode.Success
        try:
            res, error_code = self._exec_cmd(upload_cmd_args)
            if error_code == PDCErrorCode.Success:
                logger.info(f"upload {local_file_path} successfully")
        except Exception as e:
            logger.error(f"exec cmd {upload_cmd_args} with error: {e}")
            raise Exception(f"exec cmd {upload_cmd_args} with error: {e}")
        return error_code

    def _download_file(self, remote_path: str, local_path: str) -> PDCErrorCode:
        """download file

        Args:
        remote_path str: remote file path
        local_path str: local file path
        """
        if os.path.exists(local_path):
            os.rename(local_path, f"{local_path}.old")
            logger.warning(f"{local_path} already exists, backup it to {local_path}.old")

        conf = json.dumps({"remote_path": remote_path, "local_path": local_path})
        # download file from remote path
        download_cmd_args = [self._pdc_agent_bin, "-mode", "command", "-type", "download", "-config", f"{conf}"]
        error_code = PDCErrorCode.Success
        try:
            logger.info(f"begin to download {remote_path}, config: {conf}")
            res, error_code = self._exec_cmd(download_cmd_args)
            if error_code == PDCErrorCode.Success:
                logger.info(f"download {remote_path} successfully")
        except Exception as e:
            logger.error(f"exec cmd {download_cmd_args} with error: {e}")
            raise Exception(f"exec cmd {download_cmd_args} with error: {e}")
        return error_code

    def _pdc_backup_failed_directory(self, path):
        base_dir, target_path = os.path.split(os.path.normpath(path))
        failed_path = os.path.join(base_dir, f"{target_path}_failed")
        if os.path.exists(path):
            if os.path.exists(failed_path):
                shutil.rmtree(failed_path)
            # Backup failed files for debug
            os.rename(path, failed_path)

    def pdc_backup_to_flash_device(self, persistent_path: str, flash_device_path: str) -> PDCErrorCode:
        """backup data to flash device

        Args:
        persistent_path str: persistent path
        flash_device_path str: flash device path
        """
        if not os.path.exists(persistent_path):
            logger.error(f"{persistent_path} not exist")
            return PDCErrorCode.LocalPathNotExist

        logger.info("starting backup to flash device...")

        # step 1: generate checksum for recovery
        result = self.pdc_generate_dir_checksum(persistent_path)
        if result != PDCErrorCode.Success:
            logger.error(f"[Error] [pdc_sdk] generating checksum for {persistent_path} failed")
            return result

        # step 2: copy persistent data to flash device
        try:
            copy_tree(persistent_path, flash_device_path)
            logger.info(f"backup {persistent_path} to {flash_device_path} successed.")
        except Exception as e:
            logger.error(f"[Error] [pdc_sdk] copy tree {persistent_path} to {flash_device_path} failed, error: {e}")
            self._pdc_backup_failed_directory(flash_device_path)
            return PDCErrorCode.CopyTreeFailed

        # step 3: do checksum for storage on flash device
        result = self.pdc_flash_do_check(flash_device_path)
        if result == PDCErrorCode.Success:
            return result

        logger.error(f"[Error] [pdc_sdk] checksum failed on {flash_device_path} after copy, backup for debug")
        self._pdc_backup_failed_directory(flash_device_path)
        return result

    def pdc_generate_dir_checksum(self, path: str) -> PDCErrorCode:
        """
        Args
        :param localPath:
        :return:
        """
        if not os.path.exists(path):
            logger.error(f"pdc_generate_dir_checksum gi{path} not exist")
            return PDCErrorCode.CommandFail
        generate_checksum_args = [self._pdc_agent_bin, "-mode", "command", "-type", "generateSum", "-path", f"{path}"]
        error_code = PDCErrorCode.Success
        try:
            logger.info(f"begin to generate_sum path: {path}")
            res, error_code = self._exec_cmd(generate_checksum_args)
            if error_code == PDCErrorCode.Success:
                logger.info(f"generate_sum {path} successfully")
        except Exception as e:
            logger.error(f"exec cmd {generate_checksum_args} with error: {e}")
            return PDCErrorCode.CheckSumCommandFail
        return error_code

    def pdc_flash_do_check(self, path: str) -> PDCErrorCode:
        """
        Args
        :param localPath:
        :return:
        """
        if not os.path.exists(path):
            logger.error(f"pdc_flash_do_check {path} not exist")
            return PDCErrorCode.CommandFail
        generate_checksum_args = [self._pdc_agent_bin, "-mode", "command", "-type", "checkSum", "-path", f"{path}"]
        error_code = PDCErrorCode.Success
        try:
            logger.info(f"begin to check_sum path: {path}")
            res, error_code = self._exec_cmd(generate_checksum_args)
            if error_code == PDCErrorCode.Success:
                logger.info(f"check_sum {path} successfully")
            else:
                logger.error(f"[Error] [pdc_sdk] check_sum {path} failed, error code: {error_code}")
                self._pdc_backup_failed_directory(path)
        except Exception as e:
            logger.error(f"[Error] [pdc_sdk] exec cmd {generate_checksum_args} with error: {e}")
            self._pdc_backup_failed_directory(path)
            return PDCErrorCode.CheckSumCommandFail
        return error_code

    def _clean_tmp_files(self, tmp_files: List[str]):
        """clean tmp files

        Args:
        tmp_files List[str]: list of tmp file paths
        """
        if len(tmp_files) == 0:
            return
        # clean files
        for file_path in tmp_files:
            if os.path.exists(file_path):
                logger.info(f"clean tmp file: {file_path}")
                os.remove(file_path)


pdc_tool = PDCTools()
PDCErrorMessageMap = {
    PDCErrorCode.Success: "success",
    PDCErrorCode.RemotePathNotExist: "remote path not exist",
    PDCErrorCode.LocalPathExist: "local path exist",
    PDCErrorCode.DownloadFail: "download fail",
    PDCErrorCode.AgentConfigInvalid: "agent config invalid",
    PDCErrorCode.AFSToolsNotExist: "afs tools not exist",
    PDCErrorCode.TrainConfigNotExist: "train config not exist",
    PDCErrorCode.LocalPathNotExist: "local path not exist",
    PDCErrorCode.CommandFail: "pdc agent command fail",
    PDCErrorCode.CalculateHashFail: "calculate hash fail",
    PDCErrorCode.InvalidArgument: "invalid argument",
    PDCErrorCode.CommandTimeout: "pdc agent command timeout",
    PDCErrorCode.CheckSumCommandFail: "checksum command fail",
    PDCErrorCode.CopyTreeFailed: "copy directory failed",
}
