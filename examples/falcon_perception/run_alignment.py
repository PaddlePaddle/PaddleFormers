# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""Run isolated Torch and Paddle GPU processes, then enforce FP32 alignment thresholds."""
import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch-python", required=True)
    parser.add_argument("--paddle-python", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--query", default="cat")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--torch-gpu", default="7")
    parser.add_argument("--paddle-gpu", default="6")
    parser.add_argument("--max-dimension", type=int, default=1024)
    parser.add_argument("--min-dimension", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    if args.torch_gpu == args.paddle_gpu:
        parser.error("Use separate GPUs for the concurrent reference and Paddle processes.")
    scripts = Path(__file__).resolve().parent
    repo = scripts.parents[1]
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    common = [
        "--model",
        str(Path(args.model).resolve()),
        "--image",
        str(Path(args.image).resolve()),
        "--query",
        args.query,
        "--max-dimension",
        str(args.max_dimension),
        "--min-dimension",
        str(args.min_dimension),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--dtype",
        "float32",
        "--trace",
    ]

    def run(framework, python, script, gpu):
        destination = args.output / framework
        destination.mkdir(exist_ok=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
        command = [python, str(scripts / script), *common, "--output", str(destination)]
        log = destination / "run.log"
        print(f"{framework}: GPU {gpu}, log={log}", flush=True)
        with log.open("w") as stream:
            result = subprocess.run(command, env=env, cwd=repo, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"{framework} failed ({result.returncode}); see {log}")

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(run, "torch", args.torch_python, "export_torch_reference.py", args.torch_gpu),
            pool.submit(run, "paddle", args.paddle_python, "infer.py", args.paddle_gpu),
        ]
        errors = []
        for future in futures:
            try:
                future.result()
            except Exception as error:
                errors.append(str(error))
        if errors:
            raise RuntimeError("\n".join(errors))
    subprocess.run(
        [
            args.paddle_python,
            str(scripts / "compare_reference.py"),
            "--reference",
            str(args.output / "torch"),
            "--actual",
            str(args.output / "paddle"),
            "--report",
            str(args.output / "report.json"),
        ],
        check=True,
        cwd=repo,
    )


if __name__ == "__main__":
    main()
