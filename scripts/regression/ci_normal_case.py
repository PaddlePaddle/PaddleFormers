# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
"""
RUN PaddleNLP CI Case
"""
import os
import re
import subprocess
import sys


def get_mode_info(case_path):
    """
    Return: model_info{path,exec_file_list}
    Examples:
        pegasus = {
            "path": "applications/text_summarization/pegasus/"
            "deploy_path": "deploy/paddle_inference/"
            "prepare": "run_prepare.py"
            "train_exec_file": "train.py"
            "eval_exec_file": None
            "predict_exec_file": predict.py
            “export_exec_file”: export_model.py
            "infer_exec_file": inference_pegasus.py
            }
    """
    model_info = {
        "path": case_path,
        "deploy_path": None,
        "prepare_exec_file": None,
        "train_exec_file": [],
        "eval_exec_file": None,
        "predict_exec_file": None,
        "export_exec_file": None,
        "infer_exec_file": None,
    }
    for root, dirs, files in os.walk(case_path):
        infer_deploy_path = case_path + "/deploy/paddle_inference"
        python_deploy_path = case_path + "/deploy/python"

        if files and root == case_path:
            for file in files:
                # TODO .sh file incompatible windows
                if file.split(".")[-1] == "py":
                    if re.compile("prepare.py").findall(file):
                        model_info["prepare_exec_file"] = file

                    elif re.compile("train.py").findall(file):
                        model_info["train_exec_file"].append(file)

                    elif re.compile("finetune").findall(file):
                        model_info["train_exec_file"].append(file)

                    elif re.compile("eval.py").findall(file):
                        model_info["eval_exec_file"] = file

                    elif re.compile("predict.py").findall(file):
                        model_info["predict_exec_file"] = file

                    elif re.compile("export_model.py").findall(file):
                        model_info["export_exec_file"] = file

                    elif re.compile("run_").findall(file):
                        model_info["train_exec_file"].append(file)
                    else:
                        continue
        elif files and root == infer_deploy_path:
            for file in files:
                if file.split(".")[-1] == "py":
                    model_info["deploy_path"] = "deploy/paddle_inference"
                    model_info["infer_exec_file"] = file
        elif files and root == python_deploy_path:
            for file in files:
                if file.split(".")[-1] == "py":
                    model_info["deploy_path"] = "deploy/python"
                    model_info["infer_exec_file"] = file

    print("model_info", model_info)
    return model_info


def save_log(exit_code, output, case_name, file_name):
    """
    save model log
    """
    root_path = "/workspace/PaddleNLP"
    # root_path = '/ssd1/paddlenlp/zhangjunjun/PaddleNLP'
    if exit_code == 0:
        log_file = root_path + "/model_logs/" + os.path.join(case_name + "_" + file_name + "_SUCCESS.log")
        print("{} SUCCESS".format(file_name))
        with open(log_file, "a") as flog:
            flog.write("%s" % (output))
    else:
        log_file = root_path + "/model_logs/" + os.path.join(case_name + "_" + file_name + "_FAIL.log")
        print("{} FAIL".format(file_name))
        with open(log_file, "a") as flog:
            flog.write("%s" % (output))


def run_normal_case(case_path):
    """
    Run new normal case
    params:
    case_path: model path based PaddleNLP from git diff
    """
    case_name = case_path.split("/")[-1]
    model_info = get_mode_info(case_path)
    deploy_path = model_info["deploy_path"]
    prepare_exec_file = model_info["prepare_exec_file"]
    eval_exec_file = model_info["eval_exec_file"]
    predict_exec_file = model_info["predict_exec_file"]
    export_exec_file = model_info["export_exec_file"]
    infer_exec_file = model_info["infer_exec_file"]

    os.chdir(case_path)

    if prepare_exec_file:
        prepare_output = subprocess.getstatusoutput("python %s " % (prepare_exec_file))
        save_log(prepare_output[0], prepare_output[1], case_name, prepare_exec_file.split(".")[0])

    if model_info["train_exec_file"]:
        for train_file in model_info["train_exec_file"]:
            train_output = subprocess.getstatusoutput(
                "python -m paddle.distributed.launch %s --device gpu --max_steps 2 \
                --save_steps 2 --output_dir ./output/"
                % (train_file)
            )
            save_log(train_output[0], train_output[1], case_name, train_file.split(".")[0])
    else:
        print("Train Skipped")

    if eval_exec_file:
        eval_output = subprocess.getstatusoutput("python %s --init_checkpoint_dir ./output/" % (eval_exec_file))
        save_log(eval_output[0], eval_output[1], case_name, eval_exec_file.split(".")[0])
    else:
        print("Evaluation Skipped")
    if predict_exec_file:
        predict_output = subprocess.getstatusoutput("python %s --init_checkpoint_dir ./output/" % (predict_exec_file))
        save_log(predict_output[0], predict_output[1], case_name, predict_exec_file.split(".")[0])
    else:
        print("Predict Skipped")
    if export_exec_file:
        export_output = subprocess.getstatusoutput(
            "python %s --export_output_dir ./inference_model/" % (export_exec_file)
        )
        save_log(export_output[0], export_output[1], case_name, export_exec_file.split(".")[0])
    else:
        print("Export model Skipped")
    if infer_exec_file:
        infer_output = subprocess.getstatusoutput(
            "cd %s && python %s --inference_model_dir ../../inference_model/" % (deploy_path, infer_exec_file)
        )
        save_log(infer_output[0], infer_output[1], case_name, infer_exec_file.split(".")[0])
    else:
        print("python inference Skipped")


if __name__ == "__main__":
    # path ="applications/text_summarization/pegasus"
    path = sys.argv[1]
    if os.path.isdir(path):
        run_normal_case(path)
    else:
        print("not model file path, skepped ")
