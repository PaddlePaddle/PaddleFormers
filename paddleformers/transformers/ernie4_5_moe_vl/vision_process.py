# coding=utf-8
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
"""Processing functions for image and video inputs for ERNIE4.5-MOE-VL."""

import base64
import datetime
import hashlib
import io
import os
import random
import threading
import uuid
from pathlib import Path

import numpy as np
import requests
from decord import VideoReader, cpu
from PIL import Image, ImageDraw, ImageFont
from PIL.ExifTags import TAGS

from ...utils.log import logger

RAW_VIDEO_DIR = "./download_tmp/raw_video/"
RAW_IMAGE_DIR = "./download_tmp/raw_images/"
EXTRACTED_FRAME_DIR = "./download_tmp/extracted_frames/"
TMP_DIR = "./download_tmp/upload_tmp/"

FONT_PATH = os.path.join(Path(__file__).parent.absolute(), "Roboto-Regular.ttf")


def get_filename(url=None):
    """
    Get Filename
    """
    if url is None:
        return str(uuid.uuid4()).replace("-", "")
    t = datetime.datetime.now()
    if not isinstance(url, bytes):
        url = url.encode("utf-8")

    md5_hash = hashlib.md5(url).hexdigest()
    pid = os.getpid()
    tid = threading.get_ident()

    # Remove the suffix to prevent save-jpg from reporting errors
    image_filname = f"{t.year}-{t.month:02d}-{t.day:02d}-{pid}-{tid}-{md5_hash}"
    return image_filname


def file_download(url, download_dir, save_to_disk=False, retry=0, retry_interval=3):
    """
    Description: Download url, if url is PIL, return directly
    Args:
        url(str, PIL): http/local path/io.Bytes, note that io.Bytes is the image byte stream
        download_path: when save_to_disk=True, return the saved address
        save_to_disk: whether to save in the local path
    """

    if isinstance(url, Image.Image):
        return url
    elif isinstance(url, VideoReader):
        return url
    elif url.startswith("http"):
        response = requests.get(url)
        bytes_data = response.content
    elif os.path.isfile(url):
        if save_to_disk:
            return url
        bytes_data = open(url, "rb").read()
    else:
        bytes_data = base64.b64decode(url)
    if not save_to_disk:
        return bytes_data

    download_path = os.path.join(download_dir, get_filename(url))
    Path(download_path).parent.mkdir(parents=True, exist_ok=True)
    with open(download_path, "wb") as f:
        f.write(bytes_data)
    return download_path


def get_downloadable(url, download_dir=RAW_VIDEO_DIR, save_to_disk=False, retry=0, retry_interval=3):
    """download video and store it in the disk

    return downloaded **path** if save_to_disk is set to true
    return downloaded **bytes** if save_to_disk is set to false
    """

    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    downloaded_path = file_download(
        url,
        download_dir,
        save_to_disk=save_to_disk,
        retry=retry,
        retry_interval=retry_interval,
    )
    return downloaded_path


def get_downloadable_image(download_path, need_exif_info, retry_max_time=0, retry_interval=3):
    """
    Get downloadable with exif info and image processing
    """

    def get_image_exif(image):
        exif_data = image._getexif()
        exif_info = {}
        if exif_data is not None:
            for tag, value in exif_data.items():
                tag_name = TAGS.get(tag, tag)
                exif_info[tag_name] = value.strip()
        return exif_info

    def has_transparent_background(img):
        """has_transparent_background"""
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            # Check for any pixel with alpha channel less than 255 (fully opaque)
            alpha = img.convert("RGBA").split()[-1]
            if alpha.getextrema()[0] < 255:
                return True
        return False

    def add_white_background(img):
        """
        Add a white background to a transparent background image
        """
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        # Create an image with a white background and the same size as the original image
        img_white_background = Image.new("RGBA", img.size, (255, 255, 255))

        # Paste the original image onto a white background
        img_white_background.paste(img, (0, 0), img)

        return img_white_background

    def change_I16_to_L(img):
        """
        Convert image from I;16 mode to L mode
        """
        # Since the point function in I mode only supports addition, subtraction, and multiplication,
        # the following * (1 / 256) cannot be changed to division.
        return img.point(lambda i: i * (1 / 256)).convert("L")

    image = get_downloadable(
        download_path,
        save_to_disk=False,
        retry=retry_max_time,
        retry_interval=retry_interval,
    )
    if isinstance(image, Image.Image):
        pil_image = image
    else:
        pil_image = Image.open(io.BytesIO(image))
    if need_exif_info:
        try:
            exif_info = get_image_exif(pil_image)
        except Exception:
            exif_info = {}
    else:
        exif_info = {}

    try:
        if pil_image.mode == "I;16":
            pil_image = change_I16_to_L(pil_image)
        if has_transparent_background(pil_image):
            pil_image = add_white_background(pil_image)
    except Exception:
        pass

    return pil_image.convert("RGB"), exif_info


def read_video_decord(video_path, save_to_disk):
    """get reader and meta by decord"""
    video_path = get_downloadable(video_path, save_to_disk=save_to_disk)
    if isinstance(video_path, VideoReader):
        video_reader = video_path
    else:
        if isinstance(video_path, bytes):
            video_path = io.BytesIO(video_path)
        video_reader = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    vlen = len(video_reader)
    fps = video_reader.get_avg_fps()
    duration = vlen / float(fps)

    video_meta = {"fps": fps, "duration": duration, "num_of_frame": vlen}

    return video_reader, video_meta, video_path


def get_frame_indices(
    vlen,
    target_frames=-1,
    target_fps=-1,
    frames_sample="middle",
    fix_start=None,
    input_fps=-1,
):
    """get_frame_indices"""
    assert frames_sample in ["rand", "middle", "leading"]
    if target_frames > 0:
        assert target_fps <= 0, "target_fps must be negative if target_frames is given."
        if target_frames > vlen:
            acc_samples = vlen
            logger.info(
                f"target_frames={target_frames} is larger than video length {vlen}, "
                f"will sample {acc_samples} frames."
            )
        else:
            acc_samples = target_frames
            logger.debug(f"sampling at target_frames={target_frames}, frames_sample={frames_sample}")

        # split the video into `acc_samples` intervals, and sample from each interval.
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if frames_sample == "rand":
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except Exception:
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                frame_indices.sort()
                frame_indices = list(frame_indices)
        elif fix_start is not None:
            frame_indices = [x[0] + fix_start for x in ranges]
        elif frames_sample == "leading":
            frame_indices = [x[0] for x in ranges]
        elif frames_sample == "middle":
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

    elif target_fps > 0:
        assert target_frames <= 0, "target_frames must be negative if target_fps is given."
        assert input_fps > 0, "input_fps must be provided if target_fps is given."
        logger.info(f"sampling at fps={target_fps}, frames_sample={frames_sample}")
        duration = float(vlen) / input_fps
        delta = 1 / target_fps  # gap between frames, this is also the clip length each frame represents
        if frames_sample == "middle":
            frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        elif frames_sample == "leading":
            frame_seconds = np.arange(0, duration, delta)
        if frames_sample == "rand":
            frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
            rand_offset = np.random.rand(*(frame_seconds.shape)) - 0.5
            frame_seconds += rand_offset * delta
        frame_indices = np.around(frame_seconds * input_fps).astype(int)
        frame_indices = [e for e in frame_indices if e < vlen]

    else:
        raise ValueError("Must provide either positive target_fps or positive target_frames.")

    return frame_indices


def read_frames_decord(
    video_path,
    video_reader,
    video_meta,
    target_frames=-1,
    target_fps=-1,
    frames_sample="middle",
    fix_start=None,
    frame_indices=None,
    tol=10,
):
    """get frames by decord"""

    if frame_indices is None:
        frame_indices = get_frame_indices(
            video_meta["num_of_frame"],
            target_frames=target_frames,
            target_fps=target_fps,
            frames_sample=frames_sample,
            fix_start=fix_start,
            input_fps=video_meta["fps"],
        )

    frames = []
    try:
        frames = video_reader.get_batch(frame_indices).asnumpy()
        video_reader.seek(0)
    except Exception:
        logger.info(f"get {frame_indices} frames error")

    assert len(frames) == len(frame_indices), f"len(frames): {len(frames)} != len(frame_indices): {len(frame_indices)}"

    ret = []
    for idx, frame in enumerate(frames):
        tmp = Image.fromarray(frame, "RGB")
        ret.append(tmp)

    time_stamps = [frame_idx * video_meta["duration"] / video_meta["num_of_frame"] for frame_idx in frame_indices]

    del frame_indices
    assert len(time_stamps) == len(ret)
    return ret, time_stamps


def render_single_image_with_timestamp(image: Image, number: str, rate: float, font_path: str = FONT_PATH):
    """
    Function: Renders a timestamp to the image of pil.image
    The timestamp size is the rate of min(width, height)
    The font color is black, the outline is white, and the outline size is 10% of the font
    Returns an Image object
    """
    draw = ImageDraw.Draw(image)
    width, height = image.size
    font_size = int(min(width, height) * rate)
    outline_size = int(font_size * 0.1)
    font = ImageFont.truetype(font_path, font_size)
    x = 0
    y = 0

    # Draw a black timestamp with a white border
    draw.text(
        (x, y),
        number,
        font=font,
        fill=(0, 0, 0),
        stroke_width=outline_size,
        stroke_fill=(255, 255, 255),
    )

    return image


def timestamp_converting(time_stamp_in_seconds):
    """
    convert timestamp format from seconds to hr:min:sec
    """
    # get hours
    hours = 0
    while time_stamp_in_seconds >= 3600:
        hours += 1
        time_stamp_in_seconds -= 3600
    # get minutes
    mins = 0
    while time_stamp_in_seconds >= 60:
        mins += 1
        time_stamp_in_seconds -= 60
    time_hours = f"{int(hours):02d}"
    time_mins = f"{int(mins):02d}"
    time_secs = f"{time_stamp_in_seconds:05.02f}"
    fi_time_stamp = time_hours + ":" + time_mins + ":" + time_secs

    return fi_time_stamp


def render_frame_timestamp(frame, timestamp, font_rate=0.1):
    """
    Function, given a frame, render the index in order
    Logic: render the index to the upper left corner of the image
    frame: frame, PIL.Image object
    timestamp: timestamp, in seconds
    font_rate: the ratio of font size to min(wi, hei)
    """
    time_stamp = "time: " + timestamp_converting(timestamp)
    new_frame = render_single_image_with_timestamp(frame, time_stamp, font_rate)

    return new_frame
