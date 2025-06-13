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
import random
import re
import uuid
from collections.abc import Sequence
from functools import cmp_to_key
from io import BytesIO

import numpy as np
from PIL import Image


class BaseOperator(object):
    def __init__(self, name=None):
        if name is None:
            name = self.__class__.__name__
        self._id = name + "_" + str(uuid.uuid4())[-6:]

    def __call__(self, sample, context=None):
        """Process a sample.
        Args:
            sample (dict): a dict of sample, eg: {'image':xx, 'label': xxx}
            context (dict): info about this sample processing
        Returns:
            result (dict): a processed sample
        """
        return sample

    def __str__(self):
        return str(self._id)


class DecodeImage(BaseOperator):
    def __init__(self):
        """Transform the image data to numpy format."""
        super(DecodeImage, self).__init__()

    def __call__(self, sample, context=None):
        """load image if 'im_file' field is not empty but 'image' is"""
        if "image" not in sample:
            sample["image"] = base64.b64decode(sample["im_base64"].encode("utf-8"))

        im = sample["image"]
        data = np.frombuffer(bytearray(im), dtype="uint8")
        im = np.array(Image.open(BytesIO(data)).convert("RGB"))  # RGB format
        sample["image"] = im

        if "h" not in sample:
            sample["h"] = im.shape[0]
        elif sample["h"] != im.shape[0]:
            sample["h"] = im.shape[0]
        if "w" not in sample:
            sample["w"] = im.shape[1]
        elif sample["w"] != im.shape[1]:
            sample["w"] = im.shape[1]

        # make default im_info with [h, w, 1]
        sample["im_info"] = np.array([im.shape[0], im.shape[1], 1.0], dtype=np.float32)
        return sample


class ResizeImage(BaseOperator):
    def __init__(self, target_size=0, interp=1):
        """
        Rescale image to the specified target size, and capped at max_size
        if max_size != 0.
        If target_size is list, selected a scale randomly as the specified
        target size.
        Args:
            target_size (int|list): the target size of image's short side,
                multi-scale training is adopted when type is list.
            interp (int): the interpolation method
        """
        super(ResizeImage, self).__init__()
        self.interp = int(interp)
        if not (isinstance(target_size, int) or isinstance(target_size, list)):
            raise TypeError(
                "Type of target_size is invalid. Must be Integer or List, now is {}".format(type(target_size))
            )
        self.target_size = target_size

    def __call__(self, sample, context=None, save_real_img=False):
        """Resize the image numpy."""
        im = sample["image"]
        if not isinstance(im, np.ndarray):
            raise TypeError("{}: image type is not numpy.".format(self))
        im_shape = im.shape
        im_size_min = np.min(im_shape[0:2])
        if isinstance(self.target_size, list):
            # Case for multi-scale training
            selected_size = random.choice(self.target_size)
        else:
            selected_size = self.target_size
        if float(im_size_min) == 0:
            raise ZeroDivisionError("{}: min size of image is 0".format(self))

        resize_w = selected_size
        resize_h = selected_size

        im = Image.fromarray(im.astype("uint8"))
        im = im.resize((int(resize_w), int(resize_h)), self.interp)
        sample["image"] = np.array(im)
        return sample


class Permute(BaseOperator):
    def __init__(self, to_bgr=True):
        """
        Change the channel.
        Args:
            to_bgr (bool): confirm whether to convert RGB to BGR
        """
        super(Permute, self).__init__()
        self.to_bgr = to_bgr

    def __call__(self, sample, context=None):
        samples = sample
        batch_input = True
        if not isinstance(samples, Sequence):
            batch_input = False
            samples = [samples]
        for sample in samples:
            assert "image" in sample, "image data not found"
            for k in sample.keys():
                # hard code
                if k.startswith("image"):
                    im = sample[k]
                    im = np.swapaxes(im, 1, 2)
                    im = np.swapaxes(im, 1, 0)
                    if self.to_bgr:
                        im = im[[2, 1, 0], :, :]
                    sample[k] = im
        if not batch_input:
            samples = samples[0]
        return samples


class NormalizeImage(BaseOperator):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[1, 1, 1], is_channel_first=True, is_scale=False):
        """
        Args:
            mean (list): the pixel mean
            std (list): the pixel variance
            channel_first (bool): confirm whether to change channel
        """
        super(NormalizeImage, self).__init__()
        self.mean = mean
        self.std = std
        self.is_channel_first = is_channel_first
        self.is_scale = is_scale
        from functools import reduce

        if reduce(lambda x, y: x * y, self.std) == 0:
            raise ValueError("{}: std is invalid!".format(self))

    def __call__(self, sample, context=None):
        """Normalize the image.
        Operators:
            1.(optional) Scale the image to [0,1]
            2. Each pixel minus mean and is divided by std
        """
        samples = sample
        batch_input = True
        if not isinstance(samples, Sequence):
            batch_input = False
            samples = [samples]
        for sample in samples:
            for k in sample.keys():
                if k.startswith("image"):
                    im = sample[k]
                    im = im.astype(np.float32, copy=False)
                    if self.is_channel_first:
                        mean = np.array(self.mean)[:, np.newaxis, np.newaxis]
                        std = np.array(self.std)[:, np.newaxis, np.newaxis]
                    else:
                        mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
                        std = np.array(self.std)[np.newaxis, np.newaxis, :]
                    if self.is_scale:
                        im = im / 255.0
                    im -= mean
                    im /= std
                    sample[k] = im
        if not batch_input:
            samples = samples[0]
        return samples


class PadBatch(BaseOperator):
    """
    Pad a batch of samples so they can be divisible by a stride.
    The layout of each image should be 'CHW'.
    Args:
        pad_to_stride (int): If `pad_to_stride > 0`, pad zeros to ensure
            height and width is divisible by `pad_to_stride`.
    """

    def __init__(self, pad_to_stride=0, use_padded_im_info=True):
        super(PadBatch, self).__init__()
        self.pad_to_stride = pad_to_stride
        self.use_padded_im_info = use_padded_im_info

    def __call__(self, samples, context=None):
        """
        Args:
            samples (list): a batch of sample, each is dict.
        """
        coarsest_stride = self.pad_to_stride
        if coarsest_stride == 0:
            return samples
        max_shape = np.array([data["image"].shape for data in samples]).max(axis=0)

        if coarsest_stride > 0:
            max_shape[1] = int(np.ceil(max_shape[1] / coarsest_stride) * coarsest_stride)
            max_shape[2] = int(np.ceil(max_shape[2] / coarsest_stride) * coarsest_stride)

        for data in samples:
            im = data["image"]
            im_c, im_h, im_w = im.shape[:]
            padding_im = np.zeros((im_c, max_shape[1], max_shape[2]), dtype=np.float32)
            padding_im[:, :im_h, :im_w] = im
            data["image"] = padding_im
            if self.use_padded_im_info:
                data["im_info"][:2] = max_shape[1:3]
        return samples


def check(s):
    """Check whether is English"""
    my_re = re.compile(r"[A-Za-z0-9]", re.S)
    res = re.findall(my_re, s)
    if len(res):
        return True
    return False


def img2base64(img_path):
    """get base64"""
    with open(img_path, "rb") as f:
        base64_str = base64.b64encode(f.read()).decode("utf-8")
    return base64_str


def np2base64(image_np):
    img = Image.fromarray(image_np)
    base64_str = pil2base64(img)
    return base64_str


def pil2base64(image, image_type=None, size=False):
    if not image_type:
        image_type = "JPEG"
    img_buffer = BytesIO()
    image.save(img_buffer, format=image_type)

    byte_data = img_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)

    base64_string = base64_str.decode("utf-8")

    if size:
        return base64_string, image.size
    else:
        return base64_string


class Bbox(object):
    """
    The inner store format of `Bbox` is (left, top, width, height).

    The user may instance plenty of `Bbox`, that's why we insist the `Bbox` only contains four variables.
    """

    __slots__ = ["_c_left", "_c_top", "_c_width", "_c_height"]

    def __init__(self, left=0, top=0, width=0, height=0):
        """
        Constructor of `Bbox`.

        >> left: The most left position of bounding box.
        >> right: The most right position of bounding box.
        >> width: The width of bounding box.
        >> height: The height of bounding box.

        ^^ AssertionError: width and height must larger than 0.
        """
        assert width >= 0, "width {} must no less than 0".format(width)
        assert height >= 0, "height {} must no less than 0".format(height)
        self._c_left, self._c_top, self._c_width, self._c_height = left, top, width, height

    def __str__(self):
        """
        Reload the `str` operator.
        """
        return repr(self)

    def __repr__(self):
        """
        Reload the `repr` operator.
        """
        return "(x={}, y={}, w={}, h={})".format(self.left, self.top, self.width, self.height)

    def __eq__(self, other):
        """
        if `self` is equal with given `other` box.

        >> other: The comparing box instance.

        << True if two box is equal else False.
        """
        return (
            self.left == other.left
            and self.top == other.top
            and self.width == other.width
            and self.height == other.height
        )

    def tuple(self, precision=3):
        """
        Return the tuple format box.
        """
        return tuple(round(one, precision) for one in (self.left, self.top, self.width, self.height))

    def list_int(self):
        """
        Return the list(int) format box.
        """
        return list(int(one) for one in (self.left, self.top, self.width, self.height))

    def points_tuple(self, precision=3):
        """
        Return the coordinate of box
        """
        return tuple(round(one, precision) for one in (self.left, self.top, self.right, self.bottom))

    @property
    def left(self):
        """
        Visit the most left position of bounding box.
        """
        return self._c_left

    @left.setter
    def left(self, left):
        """
        Set the most left position of bounding box.
        """
        self._c_left = left

    @property
    def right(self):
        """
        Visit the most right position of bounding box.
        """
        return self._c_left + self._c_width

    @right.setter
    def right(self, right):
        """
        Set the most right position of bounding box.

        ^^ AssertionError: when right is less than left.
        """
        assert right >= self._c_left, "right {} < left {} is forbidden.".format(right, self._c_left)
        self._c_width = right - self._c_left

    @property
    def top(self):
        """
        Visit the most top position of bounding box.
        """
        return self._c_top

    @top.setter
    def top(self, top):
        """
        Set the most top position of bounding box.
        """
        self._c_top = top

    @property
    def bottom(self):
        """
        Visit the most bottom position of bounding box.
        """
        return self._c_top + self._c_height

    @bottom.setter
    def bottom(self, bottom):
        """
        Set the most bottom position of bounding box.

        ^^ AssertionError: when bottom is less than top.
        """
        assert bottom >= self._c_top, "top {} > bottom {} is forbidden.".format(self._c_top, bottom)
        self._c_height = bottom - self._c_top

    @property
    def width(self):
        """
        Visit the width of bounding box.
        """
        return self._c_width

    @width.setter
    def width(self, width):
        """
        Set the width of bounding box.

        ^^ AssertionError: when width is less than 0.
        """
        assert width >= 0, "width {} < 0 is forbidden.".format(width)
        self._c_width = width

    @property
    def height(self):
        """
        Visit the height of bounding box.
        """
        return self._c_height

    @height.setter
    def height(self, height):
        """
        Set the height of bounding box.

        ^^ AssertionError: when height is less than 0.
        """
        assert height >= 0, "height {} < 0 is forbidden.".format(height)
        self._c_height = height

    def is_cross_boundary(self, width, height, top=0, left=0):
        """
        If this box is cross boundary of given boundary. The boundary is start at (0, 0) by default.

        >> width: The width of boundary.
        >> height: The height of boundary.
        >> top: The top-left point location. Default at (0, 0)
        >> left: The top-left point location. Default at (0, 0)
        """
        boundary = Bbox(top, left, width, height)
        return boundary.contain(self)

    def is_vertical(self):
        """
        If this box is vertical.
        """
        return self.width < self.height

    def is_horizontal(self):
        """
        If this box is horizontal.
        """
        return self.width > self.height

    def is_square(self):
        """
        If this box is square.
        """
        return self.width == self.height

    def center(self):
        """
        Return the center point of this box.
        """
        return (self.left + self.width / 2.0, self.top + self.height / 2.0)

    def points(self):
        """
        Convert bounding box to main corner points (left, top) + (right, bottom).

        << Two tuple of points, left-top and right-bottom respectively.
        """
        return (self.left, self.top), (self.right, self.bottom)

    def contain(self, box):
        """
        If given `box` is contained by `self`.

        >> box: The box supposed to be contained.

        << True if `self` contains `box` else False
        """
        return self.left <= box.left and self.top <= box.top and self.right >= box.right and self.bottom >= box.bottom

    def overlap_vertically(self, box):
        """
        If given `box` is overlap with `self` vertically.

        >> box: The comparing box.

        << True if overlap with each others vertically else False.
        """
        return not (self.top >= box.bottom or self.bottom <= box.top)

    def overlap_horizontally(self, box):
        """
        If given `box` is overlap with `self` horizontally.

        >> box: The comparing box.

        << True if overlap with each others horizontally else False.
        """
        return not (self.left >= box.right or self.right <= box.left)

    def overlap(self, box):
        """
        If given `box` is overlap with `self`.

        >> box: The comparing box.

        << True if overlap with each others else False.
        """
        return self.overlap_horizontally(box) and self.overlap_vertically(box)

    def hoverlap(self, box):
        """
        The value of overlapped horizontally.

        >> box: The calculating box.
        """
        if not self.overlap_horizontally(box):
            return 0

        return min(self.right, box.right) - max(self.left, box.left)

    def voverlap(self, box):
        """
        The value of overlap vertically.

        >> box: The calculating box.
        """
        if not self.overlap_vertically(box):
            return 0

        return min(self.bottom, box.bottom) - max(self.top, box.top)

    def hdistance(self, box):
        """
        The distance of two boxes horizontally.

        >> box: The calculating box.
        """
        if self.overlap_horizontally(box):
            return 0

        return max(self.left, box.left) - min(self.right, box.right)

    def vdistance(self, box):
        """
        The distance of two boxes vertically.

        >> box: The calculating box.
        """
        if self.overlap_vertically(box):
            return 0

        return max(self.top, box.top) - min(self.bottom, box.bottom)

    def area(self):
        """
        Calculate the area within the bounding box.
        """
        return self.width * self.height

    def translate(self, vector):
        """
        Translate box in the direction of vector
        """
        return Bbox(self.left + vector[0], self.top + vector[1], self.width, self.height)

    @staticmethod
    def union(*boxes):
        """
        Calculate the union bounding box of given `boxes`.

        >> boxes: The boxes to calculate with.

        << The union `Bbox` of `boxes`.
        """
        left, top = min([box.left for box in boxes]), min([box.top for box in boxes])
        right, bottom = max([box.right for box in boxes]), max([box.bottom for box in boxes])

        return Bbox.from_points((left, top), (right, bottom))

    @staticmethod
    def adjacency(boxa, boxb):
        """
        Calculate the adjacent bounding box of given boxes.

        >> boxa: The box to calculate with.
        >> boxb: The box to calculate with.

        << The adjacent `Bbox` of boxes.
        """
        horizon = [min(boxa.right, boxb.right), max(boxa.left, boxb.left)]
        vertical = [min(boxa.bottom, boxb.bottom), max(boxa.top, boxb.top)]

        left, right = min(horizon), max(horizon)
        top, bottom = min(vertical), max(vertical)

        return Bbox.from_points((left, top), (right, bottom))

    @staticmethod
    def intersection(*boxes):
        """
        Calculate the intersection bounding box of given `boxes`.

        >> boxes: The boxes to calculate with.

        << The intersection `Bbox` of `boxes`.
        """
        left, top = max(box.left for box in boxes), max(box.top for box in boxes)
        right, bottom = min(box.right for box in boxes), min(box.bottom for box in boxes)

        if left > right or top > bottom:
            return Bbox()

        return Bbox.from_points((left, top), (right, bottom))

    @staticmethod
    def iou(boxa, boxb):
        """
        Calculate the union area divided by intersection area.

        >> boxa: The box to calculate with.
        >> boxb: The box to calculate with.
        """
        return Bbox.intersection(boxa, boxb).area() / Bbox.union(boxa, boxb).area()

    @staticmethod
    def from_points(p0, p1):
        """
        Convert main corner points to bounding box.

        >> p0: The left-top points in (x, y).
        >> p1: The right-bottom points in (x, y).

        << The instance of `Bbox`.

        ^^ AssertionError: if width or height is less than 0.
        """
        assert p1[0] >= p0[0], "width {} must larger than 0.".format(p1[0] - p0[0])
        assert p1[1] >= p0[1], "height {} must larger than 0.".format(p1[1] - p0[1])

        return Bbox(p0[0], p0[1], p1[0] - p0[0], p1[1] - p0[1])


def two_dimension_sort_box(box1: Bbox, box2: Bbox, vratio=0.5):
    """bbox sort 2D

    Args:
        box1 (Bbox): [bbox1]
        box2 (Bbox): [bbox2]
        vratio (float, optional): [description]. Defaults to 0.5.

    Returns:
        [type]: [description]
    """
    kernel = [box1.left - box2.left, box1.top - box2.top]
    if box1.voverlap(box2) < vratio * min(box1.height, box2.height):
        kernel = [box1.top - box2.top, box1.left - box2.left]
    return kernel[0] if kernel[0] != 0 else kernel[1]


def two_dimension_sort_layout(layout1, layout2, vratio=0.54):
    """Layout sort"""
    return two_dimension_sort_box(layout1["bbox"], layout2["bbox"])


def ppocr2example(ocr_res, img_path):
    """Transfer paddleocr result to example"""
    segments = []
    for rst in ocr_res:
        left = min(rst[0][0][0], rst[0][3][0])
        top = min(rst[0][0][-1], rst[0][1][-1])
        width = max(rst[0][1][0], rst[0][2][0]) - min(rst[0][0][0], rst[0][3][0])
        height = max(rst[0][2][-1], rst[0][3][-1]) - min(rst[0][0][-1], rst[0][1][-1])
        segments.append({"bbox": Bbox(*[left, top, width, height]), "text": rst[-1][0]})
    segments.sort(key=cmp_to_key(two_dimension_sort_layout))
    img_base64 = img2base64(img_path)
    doc_tokens = []
    doc_boxes = []

    im_w_box = max([seg["bbox"].left + seg["bbox"].width for seg in segments]) + 20 if segments else 0
    im_h_box = max([seg["bbox"].top + seg["bbox"].height for seg in segments]) + 20 if segments else 0
    img = Image.open(img_path)
    im_w, im_h = img.size
    im_w, im_h = max(im_w, im_w_box), max(im_h, im_h_box)

    for segment in segments:
        bbox = segment["bbox"]
        x1, y1, w, h = bbox.left, bbox.top, bbox.width, bbox.height
        bbox = Bbox(*[x1, y1, w, h])
        text = segment["text"]
        char_num = 0
        eng_word = ""
        for char in text:
            if not check(char) and not eng_word:
                doc_tokens.append(char)
                char_num += 1
            elif not check(char) and eng_word:
                doc_tokens.append(eng_word)
                eng_word = ""
                doc_tokens.append(char)
                char_num += 2
            else:
                eng_word += char
        if eng_word:
            doc_tokens.append(eng_word)
            char_num += 1
        char_width = int(w / char_num)
        for char_idx in range(char_num):
            doc_boxes.append([Bbox(*[bbox.left + (char_width * char_idx), bbox.top, char_width, bbox.height])])
    new_doc_boxes = []
    for doc_box in doc_boxes:
        bbox = doc_box[0]
        new_doc_boxes.append([bbox.left, bbox.top, bbox.right, bbox.bottom])
    doc_boxes = new_doc_boxes
    example = {"text": doc_tokens, "bbox": doc_boxes, "width": im_w, "height": im_h, "image": img_base64}
    return example
