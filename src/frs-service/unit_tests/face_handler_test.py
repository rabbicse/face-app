import base64
import logging
import pickle

import cv2.cv2 as cv2
import numpy as np

from dnn_utils import dnn_converter
from face_handler import FaceHandler
from vision_utils.decorators import timeit, TimeitDecorator
from vision_utils.redis_handler import RedisHandler

logger = logging.getLogger(__name__)

face_handler = FaceHandler()


@TimeitDecorator()
def extract_emb():
    # frame = cv2.imread("test_data/480.jpg", cv2.IMREAD_UNCHANGED)
    frame = cv2.imread("test_data/480.jpg", cv2.IMREAD_UNCHANGED)
    # frame = cv2.imread("test_data/screen.jpg", cv2.IMREAD_UNCHANGED)
    emb = face_handler.extract_embedding(frame)
    print(emb)
    # b = emb.tobytes()
    hex = dnn_converter.encode_np_hex(emb)
    print(hex)
    print(len(hex))


for i in range(10):
    extract_emb()

# en = dnn_converter.encode_np(emb)
# print(len(en))
# print(en)
#
# de = dnn_converter.decode_np(en)
# print(de.shape)

# redis_handler = RedisHandler()
# redis_handler.insert_data('name', emb)
#
# dec_emb = redis_handler.search_data('name')
# if dec_emb is not None:
#     print(emb.shape)
#     print(dec_emb)
#     score = face_handler.match(emb, dec_emb)
#     print(score)
