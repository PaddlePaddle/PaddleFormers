# coding=utf-8
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

import base64
import mimetypes
import os
import random
import re
from io import BytesIO

import numpy as np
import requests
from packaging.version import Version
from PIL import Image, ImageDraw, ImageOps

from .image_utils import np2base64
from .log import logger


class DocParser(object):
    """DocParser"""

    def __init__(self, ocr_lang="ch", layout_analysis=False, pdf_parser_config=None, use_gpu=None, device_id=None):
        self.ocr_lang = ocr_lang
        self.use_angle_cls = False
        self.layout_analysis = layout_analysis
        self.pdf_parser_config = pdf_parser_config
        self.ocr_infer_model = None
        self.use_gpu = use_gpu
        self.device_id = device_id

    def parse(self, doc, expand_to_a4_size=False, do_ocr=True):
        """
        parse
        """
        doc_type = mimetypes.guess_type(doc["doc"])[0]

        if not doc_type or doc_type.startswith("image"):
            image = self.read_image(doc["doc"])
        elif doc_type == "application/pdf":
            image = self.read_pdf(doc["doc"])
        offset_x, offset_y = 0, 0
        if expand_to_a4_size:
            image, offset_x, offset_y = self.expand_image_to_a4_size(image, center=True)
        img_h, img_w = image.shape[:2]
        doc["image"] = np2base64(image)
        doc["offset_x"] = offset_x
        doc["offset_y"] = offset_y
        doc["img_w"] = img_w
        doc["img_h"] = img_h
        if do_ocr:
            ocr_result = self.ocr(image)
            if expand_to_a4_size:
                layout = []
                for segment in ocr_result:
                    box = segment[0]
                    org_box = [
                        max(box[0] - offset_x, 0),
                        max(box[1] - offset_y, 0),
                        max(box[2] - offset_x, 0),
                        max(box[3] - offset_y, 0),
                    ]
                    if len(segment) == 2:
                        layout.append((org_box, segment[1]))
                    elif len(segment) == 3:
                        layout.append((org_box, segment[1], segment[2]))
                doc["layout"] = layout
            else:
                doc["layout"] = ocr_result
        return doc

    def __call__(self, *args, **kwargs):
        """
        Call parse
        """
        return self.parse(*args, **kwargs)

    def ocr(self, image, det=True, rec=True, cls=None):
        """
        Call ocr for an image
        """

        def _get_box(box):
            box = [
                min(box[0][0], box[3][0]),  # x1
                min(box[0][1], box[1][1]),  # y1
                max(box[1][0], box[2][0]),  # x2
                max(box[2][1], box[3][1]),  # y2
            ]
            return box

        def _normal_box(box):
            # Ensure the height and width of bbox are greater than zero
            if box[3] - box[1] < 0 or box[2] - box[0] < 0:
                return False
            return True

        def _is_ch(s):
            for ch in s:
                if "\u4e00" <= ch <= "\u9fff":
                    return True
            return False

        if self.ocr_infer_model is None:
            self.init_ocr_inference()
        if cls is None:
            cls = self.use_angle_cls
        remove = False if self.ppocr_version <= Version("2.6.0.1") else True

        layout = []
        if not self.layout_analysis:
            ocr_result = self.ocr_infer_model.ocr(image, det, rec, cls)
            ocr_result = ocr_result[0] if remove else ocr_result
            for segment in ocr_result:
                box = segment[0]
                box = _get_box(box)
                if not _normal_box(box):
                    continue
                text = segment[1][0]
                layout.append((box, text))
        else:
            layout_result = self.layout_analysis_engine(image)
            for region in layout_result:
                if region["type"] != "table":
                    ocr_result = region["res"]
                    for segment in ocr_result:
                        box = segment["text_region"]
                        box = _get_box(box)
                        if not _normal_box(box):
                            continue
                        text = segment["text"]
                        layout.append((box, text, region["type"]))
                else:
                    bbox = region["bbox"]
                    table_result = region["res"]
                    html = table_result["html"]
                    cell_bbox = table_result["cell_bbox"]
                    table_list = []
                    lines = re.findall("<tr>(.*?)</tr>", html)
                    for line in lines:
                        table_list.extend(re.findall("<td.*?>(.*?)</td>", line))
                    for cell_box, text in zip(cell_bbox, table_list):
                        if self.ocr_lang == "ch":
                            box = [
                                bbox[0] + cell_box[0],
                                bbox[1] + cell_box[1],
                                bbox[0] + cell_box[4],
                                bbox[1] + cell_box[5],
                            ]
                        else:
                            box = [
                                bbox[0] + cell_box[0],
                                bbox[1] + cell_box[1],
                                bbox[0] + cell_box[2],
                                bbox[1] + cell_box[3],
                            ]
                        if not _normal_box(box):
                            continue
                        if _is_ch(text):
                            text = text.replace(" ", "")
                        layout.append((box, text, region["type"]))
        return layout

    @classmethod
    def _get_buffer(self, data, file_like=False):
        buff = None
        if len(data) < 1024:
            if os.path.exists(data):
                buff = open(data, "rb").read()
            elif data.startswith("http://") or data.startswith("https://"):
                resp = requests.get(data, stream=True)
                if not resp.ok:
                    raise RuntimeError("Failed to download the file from {}".format(data))
                buff = resp.raw.read()
            else:
                raise FileNotFoundError("Image file {} not found!".format(data))
        if buff is None:
            buff = base64.b64decode(data)
        if buff and file_like:
            return BytesIO(buff)
        return buff

    @classmethod
    def read_image(self, image):
        """
        read image to np.ndarray
        """
        image_buff = self._get_buffer(image)

        # Use exif_transpose to correct orientation
        _image = np.array(ImageOps.exif_transpose(Image.open(BytesIO(image_buff)).convert("RGB")))
        return _image

    @classmethod
    def read_pdf(self, pdf, password=None):
        """
        read pdf
        """
        try:
            import fitz
        except ImportError:
            raise RuntimeError(
                "Need PyMuPDF to process pdf input. " "Please install module by: python3 -m pip install pymupdf"
            )
        if isinstance(pdf, fitz.Document):
            return pdf
        pdf_buff = self._get_buffer(pdf)
        if not pdf_buff:
            logger.warning("Failed to read pdf: %s...", pdf[:32])
            return None
        pdf_doc = fitz.Document(stream=pdf_buff, filetype="pdf")
        if pdf_doc.needs_pass:
            if pdf_doc.authenticate(password) == 0:
                raise ValueError("The password of pdf is incorrect.")

        if pdf_doc.page_count > 1:
            logger.warning("Currently only parse the first page for PDF input with more than one page.")

        page = pdf_doc.load_page(0)
        # The original image is shrunk when convertd from PDF by fitz, so we scale the image size by 10 times
        matrix = fitz.Matrix(10, 10)
        image = np.array(self.get_page_image(page, matrix).convert("RGB"))
        return image

    @classmethod
    def get_page_image(self, page, matrix):
        """
        get page image
        """
        pix = page.get_pixmap(matrix=matrix)
        image_buff = pix.pil_tobytes("jpeg")
        return Image.open(BytesIO(image_buff))

    def init_ocr_inference(self):
        """
        init ocr inference
        """
        if self.ocr_infer_model is not None:
            logger.warning("ocr model has already been initialized")
            return

        try:
            import paddleocr
        except ImportError:
            raise RuntimeError(
                "Need paddleocr to process image input. Please install module by: python3 -m pip install paddleocr"
            )
        self.ppocr_version = Version(paddleocr.__version__)

        if not self.layout_analysis:
            from paddleocr import PaddleOCR

            self.ocr_infer_model = PaddleOCR(show_log=False, lang=self.ocr_lang)
        else:
            from paddleocr import PPStructure

            self.layout_analysis_engine = PPStructure(table=True, ocr=True, show_log=False, lang=self.ocr_lang)

    @classmethod
    def _normalize_box(self, box, old_size, new_size, offset_x=0, offset_y=0):
        """normalize box"""
        return [
            int((box[0] + offset_x) * new_size[0] / old_size[0]),
            int((box[1] + offset_y) * new_size[1] / old_size[1]),
            int((box[2] + offset_x) * new_size[0] / old_size[0]),
            int((box[3] + offset_y) * new_size[1] / old_size[1]),
        ]

    @classmethod
    def expand_image_to_a4_size(self, image, center=False):
        """expand image to a4 size"""
        h, w = image.shape[:2]
        offset_x, offset_y = 0, 0
        if h * 1.0 / w >= 1.42:
            exp_w = int(h / 1.414 - w)
            if center:
                offset_x = int(exp_w / 2)
                exp_img = np.zeros((h, offset_x, 3), dtype="uint8")
                exp_img.fill(255)
                image = np.hstack([exp_img, image, exp_img])
            else:
                exp_img = np.zeros((h, exp_w, 3), dtype="uint8")
                exp_img.fill(255)
                image = np.hstack([image, exp_img])
        elif h * 1.0 / w <= 1.40:
            exp_h = int(w * 1.414 - h)
            if center:
                offset_y = int(exp_h / 2)
                exp_img = np.zeros((offset_y, w, 3), dtype="uint8")
                exp_img.fill(255)
                image = np.vstack([exp_img, image, exp_img])
            else:
                exp_img = np.zeros((exp_h, w, 3), dtype="uint8")
                exp_img.fill(255)
                image = np.vstack([image, exp_img])
        return image, offset_x, offset_y

    @classmethod
    def write_image_with_results(
        self, image, layout=None, result=None, save_path=None, return_image=False, format=None, max_size=None
    ):
        """
        write image with boxes and results
        """

        def _flatten_results(results):
            """flatten results"""
            is_single = False
            if not isinstance(results, list):
                results = [results]
                is_single = True
            flat_results = []

            def _flatten(result):
                flat_result = []
                for key, vals in result.items():
                    for val in vals:
                        new_val = val.copy()
                        if val.get("relations"):
                            new_val["relations"] = _flatten(val["relations"])
                        new_val["label"] = key
                        flat_result.append(new_val)
                return flat_result

            for result in results:
                flat_results.append(_flatten(result))
            if is_single:
                return flat_results[0]
            return flat_results

        def _write_results(results, color=None, root=True, parent_centers=None):
            for segment in results:
                if "bbox" not in segment.keys():
                    continue
                boxes = segment["bbox"]
                if not isinstance(boxes[0], list):
                    boxes = [boxes]
                centers = []
                plot_boxes = []
                for box in boxes:
                    x1, y1, x2, y2 = box
                    plot_box = [
                        (x1, y1),
                        (x2, y1),
                        (x2, y2),
                        (x1, y2),
                    ]
                    plot_boxes.append(plot_box)
                    centers.append(((x2 - x1) / 2 + x1, (y2 - y1) / 2 + y1))
                if root:
                    while True:
                        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                        if sum(color) < 480:
                            break
                for box in plot_boxes:
                    draw_render.polygon(box, fill=color)
                if parent_centers:
                    for p_c in parent_centers:
                        for c in centers:
                            draw_render.line((p_c[0], p_c[1], c[0], c[1]), fill=125, width=3)
                if isinstance(segment, dict) and segment.get("relations"):
                    _write_results(segment["relations"], color, root=False, parent_centers=centers)

        random.seed(0)
        _image = self.read_image(image)
        _image = Image.fromarray(np.uint8(_image))
        h, w = _image.height, _image.width
        img_render = _image.copy()
        draw_render = ImageDraw.Draw(img_render)

        if layout:
            for segment in layout:
                if isinstance(segment, dict):
                    box = segment["bbox"]
                else:
                    box = segment[0]
                box = [
                    (box[0], box[1]),
                    (box[2], box[1]),
                    (box[2], box[3]),
                    (box[0], box[3]),
                ]
                while True:
                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    if sum(color) < 480:
                        break
                draw_render.polygon(box, fill=color)

        elif result:
            flatten_results = _flatten_results(result)
            _write_results(flatten_results, color=None, root=True)

        img_render = Image.blend(_image, img_render, 0.3)
        img_show = Image.new("RGB", (w, h), (255, 255, 255))
        img_show.paste(img_render, (0, 0, w, h))
        w, h = img_show.width, img_show.height
        if max_size and max(w, h) > max_size:
            if max(w, h) == h:
                new_size = (int(w * max_size / h), max_size)
            else:
                new_size = (max_size, int(h * max_size / w))
            img_show = img_show.resize(new_size)

        if save_path:
            dir_path = os.path.dirname(save_path)
            if dir_path and not os.path.isdir(dir_path):
                os.makedirs(dir_path)
            img_show.save(save_path)
            if return_image:
                return np.array(img_show)
        elif return_image:
            return np.array(img_show)
        else:
            buff = BytesIO()
            if format is None:
                format = "jpeg"
            if format.lower() == "jpg":
                format = "jpeg"
            img_show.save(buff, format=format, quality=90)
            return buff
