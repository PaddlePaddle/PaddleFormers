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
"""Get PaddleNLP develop model list && before merge pr """
import io
import os


def get_model_list():
    """
    get model list from
    <https://github.com/PaddlePaddle/PaddleNLP/model_zoo/>
    <https://github.com/PaddlePaddle/PaddleNLP/slm/examples/>
    """

    CI_MODEL_LIST = [
        "DuEE",
        "DuReader-robust",
        "DuReader-yesno",
        "SQuAD",
        "albert",
        "bert",
        "bigbird",
        "clue",
        "couplet",
        "doc",
        "elmo",
        "ernie",
        "ernie-1.0",
        "ernie-csc",
        "ernie_matching",
        "few_shot",
        "glue",
        "gpt",
        "gpt-3",
        "lexical_analysis",
        "minilmv2",
        "mpnet",
        "msra_ner",
        "msra_ner",
        "ofa",
        "pointer_summarizer",
        "pp-minilm",
        "pretrained_models",
        "question_matching",
        "rnn",
        "semantic_indexing",
        "sentiment_analysis",
        "simbert",
        "simbert",
        "simcse",
        "skep",
        "squad",
        "stacl",
        "stacl",
        "tcn",
        "tinybert",
        "transformer",
        "unimo-text",
        "vae-seq2seq",
        "word_embedding",
    ]
    examples_second_list = ["model_interpretation", "semantic_indexing", "lexical_analysis", "word_embedding"]

    model_list = os.listdir("slm/model_zoo")
    model_list = os.listdir("model_zoo")
    examples_list = os.listdir("slm/examples/")
    app_list = os.listdir("applications/")

    # remove model_list README
    model_list.remove("README.md")
    examples_list.remove("README.md")
    model_list.extend(app_list)
    model_list.extend(examples_second_list)
    for examples_model_list in examples_list:
        if examples_model_list not in examples_second_list:
            examples_model = os.listdir("examples/" + examples_model_list)
            if "README.md" in examples_model:
                examples_model.remove("README.md")
            model_list.extend(examples_model)

    all_examples_dict = set(sorted(model_list))
    no_test_models = []

    # get model list not in CI/CE
    for full_model in all_examples_dict:
        if full_model not in CI_MODEL_LIST:
            no_test_models.append(full_model)

    # save model list for CI run_ci.sh
    with io.open("./scripts/regression/model_list.txt", "w", encoding="utf-8") as list:
        for all_model in all_examples_dict:
            list.write("{}\n".format(all_model))
        list.close()
    return all_examples_dict


if __name__ == "__main__":
    get_model_list()
