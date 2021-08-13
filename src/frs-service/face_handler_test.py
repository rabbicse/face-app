import logging

import cv2.cv2 as cv2

from face_handler import FaceHandler
from utils.redis_handler import RedisHandler

logger = logging.getLogger(__name__)


face_handler = FaceHandler()
# frame = cv2.imread("test_data/480.jpg", cv2.IMREAD_UNCHANGED)
frame = cv2.imread("test_data/test_rabbi01.jpg", cv2.IMREAD_UNCHANGED)
# frame = cv2.imread("test_data/screen.jpg", cv2.IMREAD_UNCHANGED)
emb = face_handler.extract_embedding(frame)
# print(emb)

# redis_handler = RedisHandler()
# redis_handler.insert_data('name', emb)
#
# dec_emb = redis_handler.search_data('name')
# if dec_emb is not None:
#     print(emb.shape)
#     print(dec_emb)
#     score = face_handler.match(emb, dec_emb)
#     print(score)