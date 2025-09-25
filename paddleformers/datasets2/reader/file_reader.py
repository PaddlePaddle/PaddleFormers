from .base_reader import BaseReader

class FileReader(BaseReader):
    def __init__(self, file_path, file_type, shuffle_file=True):
        super().__init__(file_path=file_path, file_type=file_type, shuffle_file=shuffle_file)

    def read(self):
        ext = self._get_extension()

        if ext not in self.loader_map:
            raise ValueError(f"Unsupported file extension: {ext}")
        res = self.loader_map[ext](self._file_path)

        if self._file_type not in self.convertor_map:
            raise ValueError(f"Unsupported file type: {self._file_type}")
        res = self.convertor_map[self._file_type](res)

        return res


class FileListReader(BaseReader):
    def __init__(self, file_path, file_type, shuffle_file=True):
        if not os.path.isdir(file_path):
            raise ValueError(f"Directory not found: {file_path}")
        super().__init__(file_path=file_path, file_type=file_type, shuffle_file=shuffle_file)

    def read(self):
        results = []
        for file_path in self._get_files():
            reader = FileReader(file_path, self._file_type, self._shuffle_file)
            results.append(reader.read())
        return results

    def _get_files(self):
        files = []
        for filename in os.listdir(self.file_path):
            file_path = os.path.join(self.path, filename)
            if os.path.isfile(file_path):
                files.append(file_path)
        return files