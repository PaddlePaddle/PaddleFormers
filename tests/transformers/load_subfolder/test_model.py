# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import os
import tempfile
import unittest

import pytest

from paddleformers.transformers import AutoModel, BertModel
from paddleformers.utils.log import logger
from tests.testing_utils import slow


@unittest.skip("skipping due to connection error!")
class ModelLoadTester(unittest.TestCase):
    @pytest.mark.skip
    def test_config_diff(self, config_1, config_2):
        config_1 = config_1.to_dict()
        config_2 = config_2.to_dict()
        config_1.pop("architectures", None)
        config_2.pop("architectures", None)
        assert config_1 == config_2, "config not equal"

    @pytest.mark.skip
    def test_cache_dir(
        self,
        model_cls,
        repo_id="",
        subfolder=None,
        use_safetensors=False,
        from_aistudio=False,
        from_hf_hub=False,
        from_modelscope=False,
    ):
        with tempfile.TemporaryDirectory() as cache_dir:
            model_cls.from_pretrained(
                repo_id,
                subfolder=subfolder,
                cache_dir=cache_dir,
                use_safetensors=use_safetensors,
                from_aistudio=from_aistudio,
                from_modelscope=from_modelscope,
                from_hf_hub=from_hf_hub,
            )
            file_list = []
            for root, dirs, files in os.walk(cache_dir):
                for file in files:
                    file_list.append(file)
            assert len(file_list) > 0, "cache_dir is empty"
            assert "config.json" in file_list, "config.json not in cache_dir"
            if use_safetensors:
                assert any(".safetensors" in f for f in file_list), "*.safetensors not in cache_dir"
            else:
                if from_hf_hub:
                    assert any(".bin" in f for f in file_list), "*.bin not in cache_dir"
                else:
                    assert any(".pdparams" in f for f in file_list), "*.pdparams not in cache_dir"

    @slow
    def test_bert_load(self):
        # BOS
        logger.info("Download model from PaddleFormers BOS")
        bert_model_bos = BertModel.from_pretrained("baicai/tiny-bert-2", use_safetensors=False, from_hf_hub=False)
        bert_model_bos_auto = AutoModel.from_pretrained("baicai/tiny-bert-2", use_safetensors=False, from_hf_hub=False)
        self.test_config_diff(bert_model_bos.config, bert_model_bos_auto.config)

        logger.info("Download model from PaddleFormers BOS with subfolder")
        bert_model_bos_sub = BertModel.from_pretrained(
            "baicai/paddleformers-test-model", subfolder="tiny-bert", use_safetensors=False, from_hf_hub=False
        )
        self.test_config_diff(bert_model_bos.config, bert_model_bos_sub.config)

        bert_model_bos_sub_auto = AutoModel.from_pretrained(
            "baicai/paddleformers-test-model", subfolder="tiny-bert", use_safetensors=False, from_hf_hub=False
        )
        self.test_config_diff(bert_model_bos_sub.config, bert_model_bos_sub_auto.config)

        # aistudio
        logger.info("Download model from aistudio")
        bert_model_aistudio = BertModel.from_pretrained(
            "aistudio/tiny-bert", use_safetensors=False, from_aistudio=True
        )
        self.test_config_diff(bert_model_bos.config, bert_model_aistudio.config)
        bert_model_aistudio_auto = AutoModel.from_pretrained(
            "aistudio/tiny-bert", use_safetensors=False, from_aistudio=True
        )
        self.test_config_diff(bert_model_aistudio.config, bert_model_aistudio_auto.config)

        # hf
        logger.info("Download model from hf")
        bert_model_hf = BertModel.from_pretrained("Baicai003/tiny-bert", from_hf_hub=True, use_safetensors=False)
        bert_model_hf_auto = AutoModel.from_pretrained("Baicai003/tiny-bert", from_hf_hub=True, use_safetensors=False)
        self.test_config_diff(bert_model_hf.config, bert_model_hf_auto.config)
        logger.info("Download model from hf with subfolder")
        bert_model_hf_sub = BertModel.from_pretrained(
            "Baicai003/paddleformers-test-model", subfolder="tiny-bert", from_hf_hub=True, use_safetensors=False
        )
        self.test_config_diff(bert_model_hf.config, bert_model_hf_sub.config)
        bert_model_hf_sub_auto = AutoModel.from_pretrained(
            "Baicai003/paddleformers-test-model", subfolder="tiny-bert", from_hf_hub=True, use_safetensors=False
        )
        self.test_config_diff(bert_model_hf_sub.config, bert_model_hf_sub_auto.config)
        bert_model_hf = BertModel.from_pretrained("Baicai003/tiny-bert-one", from_hf_hub=True, use_safetensors=False)
        self.test_config_diff(bert_model_hf.config, bert_model_hf.config)
        bert_model_hf_auto = AutoModel.from_pretrained(
            "Baicai003/tiny-bert-one", from_hf_hub=True, use_safetensors=False
        )
        self.test_config_diff(bert_model_hf.config, bert_model_hf_auto.config)
        logger.info("Download model from hf with subfolder")
        bert_model_hf_sub = BertModel.from_pretrained(
            "Baicai003/paddleformers-test-model", subfolder="tiny-bert-one", from_hf_hub=True, use_safetensors=False
        )
        self.test_config_diff(bert_model_hf.config, bert_model_hf_sub.config)
        bert_model_hf_sub_auto = AutoModel.from_pretrained(
            "Baicai003/paddleformers-test-model", subfolder="tiny-bert-one", from_hf_hub=True, use_safetensors=False
        )
        self.test_config_diff(bert_model_hf_sub.config, bert_model_hf_sub_auto.config)

        logger.info("Download model from aistudio with subfolder")
        bert_model_aistudio_sub = BertModel.from_pretrained(
            "aistudio/paddleformers-test-model", subfolder="tiny-bert", use_safetensors=False, from_aistudio=True
        )
        self.test_config_diff(bert_model_aistudio.config, bert_model_aistudio_sub.config)
        bert_model_aistudio_sub_auto = AutoModel.from_pretrained(
            "aistudio/paddleformers-test-model", subfolder="tiny-bert", use_safetensors=False, from_aistudio=True
        )
        self.test_config_diff(bert_model_aistudio_sub.config, bert_model_aistudio_sub_auto.config)

        # local
        logger.info("Download model from local")
        bert_model_bos.save_pretrained("./paddleformers-test-model/tiny-bert", safe_serialization=False)
        bert_model_local = BertModel.from_pretrained(
            "./paddleformers-test-model/", subfolder="tiny-bert", use_safetensors=False
        )
        self.test_config_diff(bert_model_bos.config, bert_model_local.config)
        bert_model_local_auto = AutoModel.from_pretrained(
            "./paddleformers-test-model/", subfolder="tiny-bert", use_safetensors=False
        )
        self.test_config_diff(bert_model_local.config, bert_model_local_auto.config)

        logger.info("Test cache_dir")
        # BOS
        self.test_cache_dir(BertModel, "baicai/tiny-bert-2", use_safetensors=False, from_hf_hub=False)
        self.test_cache_dir(AutoModel, "baicai/tiny-bert-2", use_safetensors=False, from_hf_hub=False)
        self.test_cache_dir(
            BertModel,
            "baicai/paddleformers-test-model",
            subfolder="tiny-bert",
            use_safetensors=False,
            from_hf_hub=False,
        )
        self.test_cache_dir(
            AutoModel,
            "baicai/paddleformers-test-model",
            subfolder="tiny-bert",
            use_safetensors=False,
            from_hf_hub=False,
        )

        # aistudio
        self.test_cache_dir(BertModel, "aistudio/tiny-bert", use_safetensors=False, from_aistudio=True)
        self.test_cache_dir(AutoModel, "aistudio/tiny-bert", use_safetensors=False, from_aistudio=True)
        self.test_cache_dir(
            BertModel,
            "aistudio/paddleformers-test-model",
            subfolder="tiny-bert",
            use_safetensors=False,
            from_aistudio=True,
        )
        self.test_cache_dir(
            AutoModel,
            "aistudio/paddleformers-test-model",
            subfolder="tiny-bert",
            use_safetensors=False,
            from_aistudio=True,
        )

        # hf
        self.test_cache_dir(BertModel, "Baicai003/tiny-bert", from_hf_hub=True, use_safetensors=False)
        self.test_cache_dir(AutoModel, "Baicai003/tiny-bert", from_hf_hub=True, use_safetensors=False)
        self.test_cache_dir(
            BertModel,
            "Baicai003/paddleformers-test-model",
            subfolder="tiny-bert",
            from_hf_hub=True,
            use_safetensors=False,
        )
        self.test_cache_dir(
            AutoModel,
            "Baicai003/paddleformers-test-model",
            subfolder="tiny-bert",
            from_hf_hub=True,
            use_safetensors=False,
        )
        self.test_cache_dir(BertModel, "Baicai003/tiny-bert-one", from_hf_hub=True, use_safetensors=False)
        self.test_cache_dir(AutoModel, "Baicai003/tiny-bert-one", from_hf_hub=True, use_safetensors=False)
        self.test_cache_dir(
            BertModel,
            "Baicai003/paddleformers-test-model",
            subfolder="tiny-bert-one",
            from_hf_hub=True,
            use_safetensors=False,
        )
        self.test_cache_dir(
            AutoModel,
            "Baicai003/paddleformers-test-model",
            subfolder="tiny-bert-one",
            from_hf_hub=True,
            use_safetensors=False,
        )

    @slow
    def test_bert_load_safe(self):
        # BOS
        logger.info("Download model from PaddleFormers BOS")
        bert_model_bos = BertModel.from_pretrained("baicai/tiny-bert-2", use_safetensors=True, from_hf_hub=False)
        bert_model_bos_auto = AutoModel.from_pretrained("baicai/tiny-bert-2", use_safetensors=True, from_hf_hub=False)
        self.test_config_diff(bert_model_bos.config, bert_model_bos_auto.config)

        logger.info("Download model from PaddleFormers BOS with subfolder")
        bert_model_bos_sub = BertModel.from_pretrained(
            "baicai/paddleformers-test-model", subfolder="tiny-bert", use_safetensors=True, from_hf_hub=False
        )
        self.test_config_diff(bert_model_bos.config, bert_model_bos_sub.config)

        bert_model_bos_sub_auto = AutoModel.from_pretrained(
            "baicai/paddleformers-test-model", subfolder="tiny-bert", use_safetensors=True, from_hf_hub=False
        )
        self.test_config_diff(bert_model_bos_sub.config, bert_model_bos_sub_auto.config)

        # aistudio
        logger.info("Download model from aistudio")
        bert_model_aistudio = BertModel.from_pretrained("aistudio/tiny-bert", use_safetensors=True, from_aistudio=True)
        self.test_config_diff(bert_model_bos.config, bert_model_aistudio.config)
        bert_model_aistudio_auto = AutoModel.from_pretrained(
            "aistudio/tiny-bert", use_safetensors=True, from_aistudio=True
        )
        self.test_config_diff(bert_model_aistudio.config, bert_model_aistudio_auto.config)

        # hf
        logger.info("Download model from hf")
        bert_model_hf = BertModel.from_pretrained("Baicai003/tiny-bert", from_hf_hub=True, use_safetensors=True)
        bert_model_hf_auto = AutoModel.from_pretrained("Baicai003/tiny-bert", from_hf_hub=True, use_safetensors=True)
        self.test_config_diff(bert_model_hf.config, bert_model_hf_auto.config)
        logger.info("Download model from hf with subfolder")
        bert_model_hf_sub = BertModel.from_pretrained(
            "Baicai003/paddleformers-test-model", subfolder="tiny-bert", from_hf_hub=True, use_safetensors=True
        )
        self.test_config_diff(bert_model_hf.config, bert_model_hf_sub.config)
        bert_model_hf_sub_auto = AutoModel.from_pretrained(
            "Baicai003/paddleformers-test-model", subfolder="tiny-bert", from_hf_hub=True, use_safetensors=True
        )
        self.test_config_diff(bert_model_hf_sub.config, bert_model_hf_sub_auto.config)
        bert_model_hf = BertModel.from_pretrained("Baicai003/tiny-bert-one", from_hf_hub=True, use_safetensors=True)
        self.test_config_diff(bert_model_hf.config, bert_model_hf.config)
        bert_model_hf_auto = AutoModel.from_pretrained(
            "Baicai003/tiny-bert-one", from_hf_hub=True, use_safetensors=True
        )
        self.test_config_diff(bert_model_hf.config, bert_model_hf_auto.config)
        logger.info("Download model from hf with subfolder")
        bert_model_hf_sub = BertModel.from_pretrained(
            "Baicai003/paddleformers-test-model", subfolder="tiny-bert-one", from_hf_hub=True, use_safetensors=True
        )
        self.test_config_diff(bert_model_hf.config, bert_model_hf_sub.config)
        bert_model_hf_sub_auto = AutoModel.from_pretrained(
            "Baicai003/paddleformers-test-model", subfolder="tiny-bert-one", from_hf_hub=True, use_safetensors=True
        )
        self.test_config_diff(bert_model_hf_sub.config, bert_model_hf_sub_auto.config)

        logger.info("Download model from aistudio with subfolder")
        bert_model_aistudio_sub = BertModel.from_pretrained(
            "aistudio/paddleformers-test-model", subfolder="tiny-bert", use_safetensors=True, from_aistudio=True
        )
        self.test_config_diff(bert_model_aistudio.config, bert_model_aistudio_sub.config)
        bert_model_aistudio_sub_auto = AutoModel.from_pretrained(
            "aistudio/paddleformers-test-model", subfolder="tiny-bert", use_safetensors=True, from_aistudio=True
        )
        self.test_config_diff(bert_model_aistudio_sub.config, bert_model_aistudio_sub_auto.config)

        # local
        logger.info("Download model from local")
        bert_model_bos.save_pretrained("./paddleformers-test-model/tiny-bert", safe_serialization=True)
        bert_model_local = BertModel.from_pretrained(
            "./paddleformers-test-model/", subfolder="tiny-bert", use_safetensors=True
        )
        self.test_config_diff(bert_model_bos.config, bert_model_local.config)
        bert_model_local_auto = AutoModel.from_pretrained(
            "./paddleformers-test-model/", subfolder="tiny-bert", use_safetensors=True
        )
        self.test_config_diff(bert_model_local.config, bert_model_local_auto.config)

        logger.info("Test cache_dir")
        # BOS
        self.test_cache_dir(BertModel, "baicai/tiny-bert-2", use_safetensors=True, from_hf_hub=False)
        self.test_cache_dir(AutoModel, "baicai/tiny-bert-2", use_safetensors=True, from_hf_hub=False)
        self.test_cache_dir(
            BertModel,
            "baicai/paddleformers-test-model",
            subfolder="tiny-bert",
            use_safetensors=True,
            from_hf_hub=False,
        )
        self.test_cache_dir(
            AutoModel,
            "baicai/paddleformers-test-model",
            subfolder="tiny-bert",
            use_safetensors=True,
            from_hf_hub=False,
        )

        # aistudio
        self.test_cache_dir(BertModel, "aistudio/tiny-bert", use_safetensors=True, from_aistudio=True)
        self.test_cache_dir(AutoModel, "aistudio/tiny-bert", use_safetensors=True, from_aistudio=True)
        self.test_cache_dir(
            BertModel,
            "aistudio/paddleformers-test-model",
            subfolder="tiny-bert",
            use_safetensors=True,
            from_aistudio=True,
        )
        self.test_cache_dir(
            AutoModel,
            "aistudio/paddleformers-test-model",
            subfolder="tiny-bert",
            use_safetensors=True,
            from_aistudio=True,
        )

        # hf
        self.test_cache_dir(BertModel, "Baicai003/tiny-bert", from_hf_hub=True, use_safetensors=True)
        self.test_cache_dir(AutoModel, "Baicai003/tiny-bert", from_hf_hub=True, use_safetensors=True)
        self.test_cache_dir(
            BertModel,
            "Baicai003/paddleformers-test-model",
            subfolder="tiny-bert",
            from_hf_hub=True,
            use_safetensors=True,
        )
        self.test_cache_dir(
            AutoModel,
            "Baicai003/paddleformers-test-model",
            subfolder="tiny-bert",
            from_hf_hub=True,
            use_safetensors=True,
        )
        self.test_cache_dir(BertModel, "Baicai003/tiny-bert-one", from_hf_hub=True, use_safetensors=True)
        self.test_cache_dir(AutoModel, "Baicai003/tiny-bert-one", from_hf_hub=True, use_safetensors=True)
        self.test_cache_dir(
            BertModel,
            "Baicai003/paddleformers-test-model",
            subfolder="tiny-bert-one",
            from_hf_hub=True,
            use_safetensors=True,
        )
        self.test_cache_dir(
            AutoModel,
            "Baicai003/paddleformers-test-model",
            subfolder="tiny-bert-one",
            from_hf_hub=True,
            use_safetensors=True,
        )

    # @slow
    # def test_clip_load(self):
    #     # BOS
    #     logger.info("Download model from PaddleFormers BOS")
    #     clip_model_bos = CLIPTextModel.from_pretrained("baicai/tiny-clip", use_safetensors=False, from_hf_hub=False)
    #     clip_model_bos_auto = AutoModel.from_pretrained("baicai/tiny-clip", use_safetensors=False, from_hf_hub=False)
    #     self.test_config_diff(clip_model_bos.config, clip_model_bos_auto.config)

    #     logger.info("Download model from PaddleFormers BOS with subfolder")
    #     clip_model_bos_sub = CLIPTextModel.from_pretrained(
    #         "baicai/paddleformers-test-model", subfolder="tiny-clip", use_safetensors=False, from_hf_hub=False
    #     )
    #     self.test_config_diff(clip_model_bos.config, clip_model_bos_sub.config)

    #     clip_model_bos_sub_auto = AutoModel.from_pretrained(
    #         "baicai/paddleformers-test-model", subfolder="tiny-clip", use_safetensors=False, from_hf_hub=False
    #     )
    #     self.test_config_diff(clip_model_bos_sub.config, clip_model_bos_sub_auto.config)

    #     # aistudio
    #     logger.info("Download model from aistudio")
    #     clip_model_aistudio = CLIPTextModel.from_pretrained(
    #         "aistudio/tiny-clip", use_safetensors=False, from_aistudio=True
    #     )
    #     self.test_config_diff(clip_model_bos.config, clip_model_aistudio.config)
    #     clip_model_aistudio_auto = AutoModel.from_pretrained(
    #         "aistudio/tiny-clip", use_safetensors=False, from_aistudio=True
    #     )
    #     self.test_config_diff(clip_model_aistudio.config, clip_model_aistudio_auto.config)

    #     logger.info("Download model from aistudio with subfolder")
    #     clip_model_aistudio_sub = CLIPTextModel.from_pretrained(
    #         "aistudio/paddleformers-test-model", subfolder="tiny-clip", use_safetensors=False, from_aistudio=True
    #     )
    #     self.test_config_diff(clip_model_aistudio.config, clip_model_aistudio_sub.config)
    #     clip_model_aistudio_sub_auto = AutoModel.from_pretrained(
    #         "aistudio/paddleformers-test-model", subfolder="tiny-clip", use_safetensors=False, from_aistudio=True
    #     )
    #     self.test_config_diff(clip_model_aistudio_sub.config, clip_model_aistudio_sub_auto.config)

    #     # hf
    #     logger.info("Download model from hf")
    #     clip_model_hf = CLIPTextModel.from_pretrained("Baicai003/tiny-clip", from_hf_hub=True, use_safetensors=False)
    #     clip_model_hf_auto = AutoModel.from_pretrained("Baicai003/tiny-clip", from_hf_hub=True, use_safetensors=False)
    #     self.test_config_diff(clip_model_hf.config, clip_model_hf_auto.config)
    #     logger.info("Download model from hf with subfolder")
    #     clip_model_hf_sub = CLIPTextModel.from_pretrained(
    #         "Baicai003/paddleformers-test-model", subfolder="tiny-clip", from_hf_hub=True, use_safetensors=False
    #     )
    #     self.test_config_diff(clip_model_hf.config, clip_model_hf_sub.config)
    #     clip_model_hf_sub_auto = AutoModel.from_pretrained(
    #         "Baicai003/paddleformers-test-model", subfolder="tiny-clip", from_hf_hub=True, use_safetensors=False
    #     )
    #     self.test_config_diff(clip_model_hf_sub.config, clip_model_hf_sub_auto.config)
    #     clip_model_hf = CLIPTextModel.from_pretrained(
    #         "Baicai003/tiny-clip-one", from_hf_hub=True, use_safetensors=False
    #     )
    #     self.test_config_diff(clip_model_hf.config, clip_model_hf.config)
    #     clip_model_hf_auto = AutoModel.from_pretrained(
    #         "Baicai003/tiny-clip-one", from_hf_hub=True, use_safetensors=False
    #     )
    #     self.test_config_diff(clip_model_hf.config, clip_model_hf_auto.config)
    #     logger.info("Download model from hf with subfolder")
    #     clip_model_hf_sub = CLIPTextModel.from_pretrained(
    #         "Baicai003/paddleformers-test-model", subfolder="tiny-clip-one", from_hf_hub=True, use_safetensors=False
    #     )
    #     self.test_config_diff(clip_model_hf.config, clip_model_hf_sub.config)
    #     clip_model_hf_sub_auto = AutoModel.from_pretrained(
    #         "Baicai003/paddleformers-test-model", subfolder="tiny-clip-one", from_hf_hub=True, use_safetensors=False
    #     )
    #     self.test_config_diff(clip_model_hf_sub.config, clip_model_hf_sub_auto.config)

    #     # local
    #     logger.info("Download model from local")
    #     clip_model_bos.save_pretrained("./paddleformers-test-model/tiny-clip", safe_serialization=False)
    #     clip_model_local = CLIPTextModel.from_pretrained(
    #         "./paddleformers-test-model/", subfolder="tiny-clip", use_safetensors=False
    #     )
    #     self.test_config_diff(clip_model_bos.config, clip_model_local.config)
    #     clip_model_local_auto = AutoModel.from_pretrained(
    #         "./paddleformers-test-model/", subfolder="tiny-clip", use_safetensors=False
    #     )
    #     self.test_config_diff(clip_model_local.config, clip_model_local_auto.config)

    #     logger.info("Test cache_dir")
    #     # BOS
    #     self.test_cache_dir(CLIPTextModel, "baicai/tiny-clip", use_safetensors=False, from_hf_hub=False)
    #     self.test_cache_dir(AutoModel, "baicai/tiny-clip", use_safetensors=False, from_hf_hub=False)
    #     self.test_cache_dir(
    #         CLIPTextModel,
    #         "baicai/paddleformers-test-model",
    #         subfolder="tiny-clip",
    #         use_safetensors=False,
    #         from_hf_hub=False,
    #     )
    #     self.test_cache_dir(
    #         AutoModel, "baicai/paddleformers-test-model", subfolder="tiny-clip", use_safetensors=False, from_hf_hub=False
    #     )

    #     # aistudio
    #     self.test_cache_dir(CLIPTextModel, "aistudio/tiny-clip", use_safetensors=False, from_aistudio=True)
    #     self.test_cache_dir(AutoModel, "aistudio/tiny-clip", use_safetensors=False, from_aistudio=True)
    #     self.test_cache_dir(
    #         CLIPTextModel,
    #         "aistudio/paddleformers-test-model",
    #         subfolder="tiny-clip",
    #         use_safetensors=False,
    #         from_aistudio=True,
    #     )
    #     self.test_cache_dir(
    #         AutoModel,
    #         "aistudio/paddleformers-test-model",
    #         subfolder="tiny-clip",
    #         use_safetensors=False,
    #         from_aistudio=True,
    #     )

    #     # hf
    #     self.test_cache_dir(CLIPTextModel, "Baicai003/tiny-clip", from_hf_hub=True, use_safetensors=False)
    #     self.test_cache_dir(AutoModel, "Baicai003/tiny-clip", from_hf_hub=True, use_safetensors=False)
    #     self.test_cache_dir(
    #         CLIPTextModel,
    #         "Baicai003/paddleformers-test-model",
    #         subfolder="tiny-clip",
    #         from_hf_hub=True,
    #         use_safetensors=False,
    #     )
    #     self.test_cache_dir(
    #         AutoModel, "Baicai003/paddleformers-test-model", subfolder="tiny-clip", from_hf_hub=True, use_safetensors=False
    #     )
    #     self.test_cache_dir(CLIPTextModel, "Baicai003/tiny-clip-one", from_hf_hub=True, use_safetensors=False)
    #     self.test_cache_dir(AutoModel, "Baicai003/tiny-clip-one", from_hf_hub=True, use_safetensors=False)
    #     self.test_cache_dir(
    #         CLIPTextModel,
    #         "Baicai003/paddleformers-test-model",
    #         subfolder="tiny-clip-one",
    #         from_hf_hub=True,
    #         use_safetensors=False,
    #     )
    #     self.test_cache_dir(
    #         AutoModel,
    #         "Baicai003/paddleformers-test-model",
    #         subfolder="tiny-clip-one",
    #         from_hf_hub=True,
    #         use_safetensors=False,
    #     )

    # @slow
    # def test_clip_load_safe(self):
    #     # BOS
    #     logger.info("Download model from PaddleFormers BOS")
    #     clip_model_bos = CLIPTextModel.from_pretrained("baicai/tiny-clip", use_safetensors=True, from_hf_hub=False)
    #     clip_model_bos_auto = AutoModel.from_pretrained("baicai/tiny-clip", use_safetensors=True, from_hf_hub=False)
    #     self.test_config_diff(clip_model_bos.config, clip_model_bos_auto.config)

    #     logger.info("Download model from PaddleFormers BOS with subfolder")
    #     clip_model_bos_sub = CLIPTextModel.from_pretrained(
    #         "baicai/paddleformers-test-model", subfolder="tiny-clip", use_safetensors=True, from_hf_hub=False
    #     )
    #     self.test_config_diff(clip_model_bos.config, clip_model_bos_sub.config)

    #     clip_model_bos_sub_auto = AutoModel.from_pretrained(
    #         "baicai/paddleformers-test-model", subfolder="tiny-clip", use_safetensors=True, from_hf_hub=False
    #     )
    #     self.test_config_diff(clip_model_bos_sub.config, clip_model_bos_sub_auto.config)

    #     # aistudio
    #     logger.info("Download model from aistudio")
    #     clip_model_aistudio = CLIPTextModel.from_pretrained(
    #         "aistudio/tiny-clip", use_safetensors=True, from_aistudio=True
    #     )
    #     self.test_config_diff(clip_model_bos.config, clip_model_aistudio.config)
    #     clip_model_aistudio_auto = AutoModel.from_pretrained(
    #         "aistudio/tiny-clip", use_safetensors=True, from_aistudio=True
    #     )
    #     self.test_config_diff(clip_model_aistudio.config, clip_model_aistudio_auto.config)

    #     logger.info("Download model from aistudio with subfolder")
    #     clip_model_aistudio_sub = CLIPTextModel.from_pretrained(
    #         "aistudio/paddleformers-test-model", subfolder="tiny-clip", use_safetensors=True, from_aistudio=True
    #     )
    #     self.test_config_diff(clip_model_aistudio.config, clip_model_aistudio_sub.config)
    #     clip_model_aistudio_sub_auto = AutoModel.from_pretrained(
    #         "aistudio/paddleformers-test-model", subfolder="tiny-clip", use_safetensors=True, from_aistudio=True
    #     )
    #     self.test_config_diff(clip_model_aistudio_sub.config, clip_model_aistudio_sub_auto.config)

    #     # hf
    #     logger.info("Download model from hf")
    #     clip_model_hf = CLIPTextModel.from_pretrained("Baicai003/tiny-clip", from_hf_hub=True, use_safetensors=True)
    #     clip_model_hf_auto = AutoModel.from_pretrained("Baicai003/tiny-clip", from_hf_hub=True, use_safetensors=True)
    #     self.test_config_diff(clip_model_hf.config, clip_model_hf_auto.config)
    #     logger.info("Download model from hf with subfolder")
    #     clip_model_hf_sub = CLIPTextModel.from_pretrained(
    #         "Baicai003/paddleformers-test-model", subfolder="tiny-clip", from_hf_hub=True, use_safetensors=True
    #     )
    #     self.test_config_diff(clip_model_hf.config, clip_model_hf_sub.config)
    #     clip_model_hf_sub_auto = AutoModel.from_pretrained(
    #         "Baicai003/paddleformers-test-model", subfolder="tiny-clip", from_hf_hub=True, use_safetensors=True
    #     )
    #     self.test_config_diff(clip_model_hf_sub.config, clip_model_hf_sub_auto.config)
    #     clip_model_hf = CLIPTextModel.from_pretrained(
    #         "Baicai003/tiny-clip-one", from_hf_hub=True, use_safetensors=True
    #     )
    #     self.test_config_diff(clip_model_hf.config, clip_model_hf.config)
    #     clip_model_hf_auto = AutoModel.from_pretrained(
    #         "Baicai003/tiny-clip-one", from_hf_hub=True, use_safetensors=True
    #     )
    #     self.test_config_diff(clip_model_hf.config, clip_model_hf_auto.config)
    #     logger.info("Download model from hf with subfolder")
    #     clip_model_hf_sub = CLIPTextModel.from_pretrained(
    #         "Baicai003/paddleformers-test-model", subfolder="tiny-clip-one", from_hf_hub=True, use_safetensors=True
    #     )
    #     self.test_config_diff(clip_model_hf.config, clip_model_hf_sub.config)
    #     clip_model_hf_sub_auto = AutoModel.from_pretrained(
    #         "Baicai003/paddleformers-test-model", subfolder="tiny-clip-one", from_hf_hub=True, use_safetensors=True
    #     )
    #     self.test_config_diff(clip_model_hf_sub.config, clip_model_hf_sub_auto.config)

    #     # local
    #     logger.info("Download model from local")
    #     clip_model_bos.save_pretrained("./paddleformers-test-model/tiny-clip", safe_serialization=True)
    #     clip_model_local = CLIPTextModel.from_pretrained(
    #         "./paddleformers-test-model/", subfolder="tiny-clip", use_safetensors=True
    #     )
    #     self.test_config_diff(clip_model_bos.config, clip_model_local.config)
    #     clip_model_local_auto = AutoModel.from_pretrained(
    #         "./paddleformers-test-model/", subfolder="tiny-clip", use_safetensors=True
    #     )
    #     self.test_config_diff(clip_model_local.config, clip_model_local_auto.config)

    #     logger.info("Test cache_dir")
    #     # BOS
    #     self.test_cache_dir(CLIPTextModel, "baicai/tiny-clip", use_safetensors=True, from_hf_hub=False)
    #     self.test_cache_dir(AutoModel, "baicai/tiny-clip", use_safetensors=True, from_hf_hub=False)
    #     self.test_cache_dir(
    #         CLIPTextModel,
    #         "baicai/paddleformers-test-model",
    #         subfolder="tiny-clip",
    #         use_safetensors=True,
    #         from_hf_hub=False,
    #     )
    #     self.test_cache_dir(
    #         AutoModel, "baicai/paddleformers-test-model", subfolder="tiny-clip", use_safetensors=True, from_hf_hub=False
    #     )

    #     # aistudio
    #     self.test_cache_dir(CLIPTextModel, "aistudio/tiny-clip", use_safetensors=True, from_aistudio=True)
    #     self.test_cache_dir(AutoModel, "aistudio/tiny-clip", use_safetensors=True, from_aistudio=True)
    #     self.test_cache_dir(
    #         CLIPTextModel,
    #         "aistudio/paddleformers-test-model",
    #         subfolder="tiny-clip",
    #         use_safetensors=True,
    #         from_aistudio=True,
    #     )
    #     self.test_cache_dir(
    #         AutoModel, "aistudio/paddleformers-test-model", subfolder="tiny-clip", use_safetensors=True, from_aistudio=True
    #     )

    #     # hf
    #     self.test_cache_dir(CLIPTextModel, "Baicai003/tiny-clip", from_hf_hub=True, use_safetensors=True)
    #     self.test_cache_dir(AutoModel, "Baicai003/tiny-clip", from_hf_hub=True, use_safetensors=True)
    #     self.test_cache_dir(
    #         CLIPTextModel,
    #         "Baicai003/paddleformers-test-model",
    #         subfolder="tiny-clip",
    #         from_hf_hub=True,
    #         use_safetensors=True,
    #     )
    #     self.test_cache_dir(
    #         AutoModel, "Baicai003/paddleformers-test-model", subfolder="tiny-clip", from_hf_hub=True, use_safetensors=True
    #     )
    #     self.test_cache_dir(CLIPTextModel, "Baicai003/tiny-clip-one", from_hf_hub=True, use_safetensors=True)
    #     self.test_cache_dir(AutoModel, "Baicai003/tiny-clip-one", from_hf_hub=True, use_safetensors=True)
    #     self.test_cache_dir(
    #         CLIPTextModel,
    #         "Baicai003/paddleformers-test-model",
    #         subfolder="tiny-clip-one",
    #         from_hf_hub=True,
    #         use_safetensors=True,
    #     )
    #     self.test_cache_dir(
    #         AutoModel,
    #         "Baicai003/paddleformers-test-model",
    #         subfolder="tiny-clip-one",
    #         from_hf_hub=True,
    #         use_safetensors=True,
    #     )
