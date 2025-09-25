from huggingface_hub import snapshot_download

def HuggingFaceDownload(
    repo_id,
    download_path,
    resume_download,
    max_workers):
    hf_download_proxy = os.getenv("https_proxy")
    if hf_download_proxy is None:
        hf_download_proxy = os.getenv("HTTPS_PROXY")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        proxies={"http": hf_download_proxy},
        resume_download=resume_download,
        max_workers=max_workers,
        local_dir=download_path,
    )