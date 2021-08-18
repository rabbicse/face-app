import base64
import logging
import pickle

import cv2.cv2 as cv2
import numpy as np
import zfpy

from dnn_utils import dnn_converter
from face_handler import FaceHandler
from vision_utils.redis_handler import RedisHandler

logger = logging.getLogger(__name__)


face_handler = FaceHandler()
# frame = cv2.imread("test_data/480.jpg", cv2.IMREAD_UNCHANGED)
frame = cv2.imread("test_data/480.jpg", cv2.IMREAD_UNCHANGED)
# frame = cv2.imread("test_data/screen.jpg", cv2.IMREAD_UNCHANGED)
emb = face_handler.extract_embedding(frame)
# # print(emb)
# b = emb.tobytes()
# hex = b.hex()
# print(b.hex())
# print(len(hex))
# b2 = bytes.fromhex(hex)
# print(b2)
# d = np.frombuffer(b2)
# print(d.shape)



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