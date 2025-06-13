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
"""Test taskflow."""
import os
import unittest

from paddlenlp import Taskflow


def test_knowledge_mining():
    """
    test_knowledge_mining
    """
    wordtag = Taskflow("knowledge_mining", model="wordtag", batch_size=2, max_seq_len=128, linking=True)
    wordtag("《孤女》是2010年九州出版社出版的小说，作者是余兼羽。")

    nptag = Taskflow("knowledge_mining", model="nptag", batch_size=2, max_seq_len=128, linking=True)
    nptag(["糖醋排骨", "红曲霉菌"])


def test_name_entity_recognition():
    """
    test_name_entity_recognition
    """
    ner = Taskflow("ner", batch_size=2)
    ner("《长津湖》收尾，北美是最大海外票仓")
    ner_fast = Taskflow("ner", mode="fast")
    ner_fast("《长津湖》收尾，北美是最大海外票仓")
    ner_entity = Taskflow("ner", mode="accurate", entity_only=True)
    ner_entity("《孤女》是2010年九州出版社出版的小说，作者是余兼羽")


def test_word_segmetation():
    """
    test_word_segmetation
    """
    seg = Taskflow("word_segmentation", batch_size=2)
    seg(["第十四届全运会在西安举办", "三亚是一个美丽的城市"])
    seg_fast = Taskflow("word_segmentation", mode="fast")
    seg_fast(["第十四届全运会在西安举办", "三亚是一个美丽的城市"])
    seg_acc = Taskflow("word_segmentation", mode="accurate")
    seg_acc("李伟拿出具有科学性、可操作性的《陕西省高校管理体制改革实施方案》")


def test_pos_tagging():
    """
    test_pos_tagging
    """
    tag = Taskflow("pos_tagging", batch_size=2)
    tag("第十四届全运会在西安举办")


def test_corrector():
    """
    test_corrector
    """
    corrector = Taskflow("text_correction", batch_size=2)
    corrector("遇到逆竟时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。")


@unittest.skip("dependency_parsing is not support for Paddle >= 2.6.1")
def test_dependency_parsing():
    """
    test_dependency_parsing
    """
    ddp = Taskflow("dependency_parsing", model="ddparser", batch_size=2, prob=True, use_pos=True)
    print(ddp("9月9日上午纳达尔在亚瑟·阿什球场击败俄罗斯球员梅德韦杰夫"))
    print(ddp.from_segments([["9月9日", "上午", "纳达尔", "在", "亚瑟·阿什球场", "击败", "俄罗斯", "球员", "梅德韦杰夫"]]))

    ddp_ernie = Taskflow("dependency_parsing", model="ddparser-ernie-1.0", batch_size=2, prob=True, use_pos=True)
    print(ddp_ernie("9月9日上午纳达尔在亚瑟·阿什球场击败俄罗斯球员梅德韦杰夫"))

    ddp_ernie_gram = Taskflow(
        "dependency_parsing", model="ddparser-ernie-gram-zh", batch_size=2, prob=True, use_pos=True
    )
    print(ddp_ernie_gram("9月9日上午纳达尔在亚瑟·阿什球场击败俄罗斯球员梅德韦杰夫"))


def test_sentiment_analysis():
    """
    test_sentiment_analysis
    """
    skep = Taskflow("sentiment_analysis", batch_size=2)
    skep("这个产品用起来真的很流畅，我非常喜欢")

    skep_ernie = Taskflow("sentiment_analysis", model="skep_ernie_1.0_large_ch", batch_size=2)
    skep_ernie("作为老的四星酒店，房间依然很整洁，相当不错。机场接机服务很好，可以在车上办理入住手续，节省时间。")


def test_text_similarity():
    """
    test_text_similarity
    """
    similarity = Taskflow("text_similarity", batch_size=2)
    similarity([["世界上什么东西最小", "世界上什么东西最小？"]])


def test_question_answering():
    """
    test_question_answering
    """
    qa = Taskflow("question_answering", batch_size=2)
    qa("中国的国土面积有多大？")


def test_poetry():
    """
    test_poetry
    """
    poetry = Taskflow("poetry_generation", batch_size=2)
    poetry("林密不见人")


def test_dialogue():
    """
    test_dialogue
    """
    dialogue = Taskflow("dialogue", batch_size=2, max_seq_len=512)
    dialogue(["吃饭了吗"])


def test_uie():
    """
    test_uie
    """
    schema_ner = ["时间", "选手", "赛事名称"]  # Define the schema for entity extraction
    ie = Taskflow("information_extraction", schema=schema_ner, model="uie-base", batch_size=2, prob=True, use_pos=True)
    ie("2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！")

    ie = Taskflow("information_extraction", schema=schema_ner, model="uie-tiny", batch_size=2, prob=True, use_pos=True)
    schema_re = {"歌曲名称": ["歌手", "所属专辑"]}  # Define the schema for relation extraction
    ie.set_schema(schema_re)  # Reset schema
    ie("《告别了》是孙耀威在专辑爱的故事里面的歌曲")

    ie = Taskflow("information_extraction", schema=schema_ner, prob=True, use_pos=True)
    schema_ee = {"歌曲名称": ["歌手", "所属专辑"]}  # Define the schema for relation extraction
    ie.set_schema(schema_ee)  # Reset schema
    ie("《告别了》是孙耀威在专辑爱的故事里面的歌曲")

    schema_opinion = {"评价维度": "观点词"}  # Define the schema for opinion extraction
    ie.set_schema(schema_opinion)  # Reset schema
    ie("个人觉得管理太混乱了，票价太高了")

    schema_sa = "情感倾向[正向，负向]"  # Define the schema for sentence-level sentiment classification
    ie.set_schema(schema_sa)  # Reset schema
    ie("这个产品用起来真的很流畅，我非常喜欢")

    schema_bre = ["寺庙", {"丈夫": "妻子"}]
    ie.set_schema(schema_bre)
    ie("李治即位后，让身在感业寺的武则天续起头发，重新纳入后宫。")

    schema = {"竞赛名称": ["主办方", "承办方", "已举办次数"]}
    ie.set_schema(schema)
    ie("2022语言与智能技术竞赛由中国中文信息学会和中国计算机学会联合主办，百度公司、中国中文信息学会评测工作委员会和中国计算机学会自然语言处理专委会承办，已连续举办4届，成为全球最热门的中文NLP赛事之一。")

    schema = ["Person", "Organization"]
    ie_en = Taskflow("information_extraction", schema=schema, model="uie-base-en")
    ie_en("In 1997, Steve was excited to become the CEO of Apple.")

    schema = [{"Person": ["Company", "Position"]}]
    ie_en.set_schema(schema)
    ie_en("In 1997, Steve was excited to become the CEO of Apple.")

    schema = [{"Aspect": ["Opinion", "Sentiment classification [negative, positive]"]}]
    ie_en.set_schema(schema)
    ie_en("The teacher is very nice.")

    schema = "Sentiment classification [negative, positive]"
    ie_en.set_schema(schema)
    ie_en("I am sorry but this is the worst film I have ever seen in my life.")


def test_summarizer():
    """
    test_summarizer
    """
    summarizer = Taskflow("text_summarization")
    summarizer("2022年，中国房地产进入转型阵痛期，传统“高杠杆、快周转”的模式难以为继，万科甚至直接喊话，中国房地产进入“黑铁时代”")
    summarizer(
        [
            "据悉，2022年教育部将围绕“巩固提高、深化落实、创新突破”三个关键词展开工作。要进一步强化学校教育主阵地作用，继续把落实“双减”作为学校工作的重中之重，\
            重点从提高作业设计水平、提高课后服务水平、提高课堂教学水平、提高均衡发展水平四个方面持续巩固提高学校“双减”工作水平。",
            "党参有降血脂，降血压的作用，可以彻底消除血液中的垃圾，从而对冠心病以及心血管疾病的患者都有一定的稳定预防工作作用，因此平时口服党参能远离三高的危害。\
            另外党参除了益气养血，降低中枢神经作用，调整消化系统功能，健脾补肺的功能。",
        ]
    )


def test_uiex():
    """UIE-X"""
    path = "./cases/"
    if not os.path.exists(path):
        os.mkdir(path)
    os.system(
        "cd %s && wget %s"
        % (
            path,
            "https://user-images.githubusercontent.com/40840292/203457596-8dbc9241-833d-4b0e-9291-f134a790d0e1.jpeg",
        )
    )
    os.system(
        "cd %s && wget %s"
        % (
            path,
            "https://user-images.githubusercontent.com/40840292/203457719-84a70241-607e-4bb1-ab4c-3d9beee9e254.jpeg",
        )
    )
    os.system(
        "cd %s && wget %s"
        % (
            path,
            "https://user-images.githubusercontent.com/40840292/203457817-76fe638a-3277-4619-9066-d1dffd52c5d4.jpg ",
        )
    )
    ie = Taskflow(
        "information_extraction",
        schema="",
        schema_lang="ch",
        ocr_lang="ch",
        batch_size=16,
        model="uie-x-base",
        layout_analysis=False,
        position_prob=0.5,
        precision="fp32",
        use_fast=True,
    )
    schema = ["姓名", "性别", "学校"]
    ie({"doc": "./cases/203457596-8dbc9241-833d-4b0e-9291-f134a790d0e1.jpeg"})

    schema = ["收发货人", "进口口岸", "进口日期", "申报日期", "提运单号"]
    ie.set_schema(schema)
    print(ie({"doc": "./cases/203457719-84a70241-607e-4bb1-ab4c-3d9beee9e254.jpeg"}))

    schema = {"项目名": "单价"}
    ie.set_schema(schema)
    print(ie({"doc": "./cases/203457817-76fe638a-3277-4619-9066-d1dffd52c5d4.jpg"}))


def test_codegen():
    """ """
    prompt = "def lengthOfLongestSubstring(self, s: str) -> int:"
    codegen = Taskflow(
        "code_generation",
        model="Salesforce/codegen-350M-mono",
        decode_strategy="greedy_search",
        repetition_penalty=1.0,
    )
    print(codegen(prompt))
