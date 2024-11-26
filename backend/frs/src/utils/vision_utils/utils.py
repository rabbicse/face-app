import time
from datetime import datetime
from math import floor

import cv2
import imutils
import numpy as np

epoch = datetime.utcfromtimestamp(0)


def unix_time_millis(dt):
    """
    @param dt:
    @return:
    """
    return (dt - epoch).total_seconds() * 1000.0


def hex_to_rgb(hex_color: str) -> tuple:
    """
    @param hex_color:
    @return:
    """
    hex_color = hex_color.lstrip('#')
    # hex_color_len = len(hex_color)
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    # return tuple(int(hex[i:i + hex_color_len // 3], 16) for i in range(0, hex_color_len, hex_color_len // 3))


def hex_to_bgr(hex_color: str) -> tuple:
    """
    @param hex_color:
    @return:
    """
    return hex_to_rgb(hex_color)[::-1]


def rgb_to_hex(rgb: tuple):
    """
    @param rgb:
    @return:
    """
    return "#{:02x}{:02x}{:02x}".format(*rgb).upper()


def resize_image(frame, max_width, max_height):
    """
    @param frame:
    @param max_width:
    @param max_height:
    @return:
    """
    height, width, channels = frame.shape
    ratio = width / height
    w, h = width, height
    if ratio > 1 and w > max_width:  # if width > height then resize based on width
        w = max_width
        h = w / ratio
    elif ratio < 1 and h > max_height:
        h = max_height
        w = h * ratio

    if w > max_width:
        w = max_width
        h = w / ratio

    if h > max_height:
        h = max_height
        w = h * ratio

    return imutils.resize(frame, int(floor(w)), int(floor(h)))


def resize_stretch_image(frame, max_width, max_height):
    """
    @param frame:
    @param max_width:
    @param max_height:
    @return:
    """
    height, width, channels = frame.shape
    ratio = width / height
    w, h = width, height
    if ratio > 1:  # if width > height then resize based on width
        w = max_width
        h = w / ratio
    elif ratio < 1:
        h = max_height
        w = h * ratio

    if w > max_width:
        w = max_width
        h = w / ratio

    if h > max_height:
        h = max_height
        w = h * ratio

    return imutils.resize(frame, int(floor(w)), int(floor(h)))


def current_time_ms():
    """
    @param x:
    @return:
    """
    return int(round(time.time() * 1000))


def parse_json_default(json_data, root, key, default):
    """
    @param json_data:
    @param root:
    @param key:
    @param default:
    @return:
    """
    return json_data[root][key] if key in json_data[root] else default


def parse_dict_default(dictionary, key, default):
    """
    @param dictionary:
    @param key:
    @param default:
    @return:
    """
    return dictionary[key] if key in dictionary else default

def convert_photo_to_bgr(photo):
    frame = np.frombuffer(photo, dtype=np.uint8)
    frame = cv2.imdecode(frame, cv2.IMREAD_UNCHANGED)

    if frame.shape[-1] == 4:  # RGBA
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    if len(frame.shape) == 2:  # Grayscale image
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    return frame