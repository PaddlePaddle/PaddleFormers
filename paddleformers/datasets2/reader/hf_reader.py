import os
import json

from .base_reader import BaseReader

DATA_INFO_FILE = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data_info.json")

with open(DATA_INFO_FILE) as fp:
    hf_repo_config_map = json.load(fp)

def is_hf_dataset(file_path):
    hf_dataset_config = hf_repo_config_map.get(file_path, None)
    if hf_dataset_config is None:
        return False
    else:
        return True


class HuggingFaceReader(BaseReader):
    def __init__(self, file_path, file_type, shuffle_file=True):
        # download
        HuggingFaceDownload(file_path)

    def read(self):
        results = []

        self.download_file_path = self.hf_dataset_config["file_name"]
        self.download_file_type = self.hf_dataset_config["formatting"]

        # hugging face 下载的数据可能是单个文件也可能是多个文件
        if os.path.isdir(self.download_file_path):
            reader = FileListReader(self.download_file_path, self.download_file_type, self._shuffle_file)
        else:
            reader = FileReader(self.download_file_path, self.download_file_type, self._shuffle_file)

        return reader.read()

