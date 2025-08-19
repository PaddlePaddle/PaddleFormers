import os

def pytest_configure(config):
    os.environ["DOWNLOAD_SOURCE"] = os.getenv("DOWNLOAD_SOURCE", "aistudio")
