import os

def pytest_configure(config):
    os.environ.setdefault("DOWNLOAD_SOURCE", "aistudio")
