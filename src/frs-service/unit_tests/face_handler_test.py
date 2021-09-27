import logging
import os

import cv2.cv2 as cv2

from dnn_utils import dnn_converter
from face_handler import FaceHandler
from vision_utils.decorators import TimeitDecorator

logger = logging.getLogger(__name__)

DETECTOR_MODEL_PATH = os.path.abspath('../models/mobilenet0.25_Final.pth')
DETECTOR_MODEL_TAR_PATH = os.path.abspath('../models/mobilenetV1X0.25_pretrain.tar')
RECOGNIZER_MODEL_PATH = os.path.abspath('../models/backbone-r100m.pth')

dnn_config = {
    'detector_model_path': DETECTOR_MODEL_PATH,
    'detector_model_tar_path': DETECTOR_MODEL_TAR_PATH,
    'recognizer_model_path': RECOGNIZER_MODEL_PATH,
    'recognizer_model_architecture': 'r100'}
face_handler = FaceHandler(detector_network='mobile0.25',
                           dnn_config=dnn_config,
                           debug=True)


@TimeitDecorator()
def extract_emb():
    # frame = cv2.imread("test_data/480.jpg", cv2.IMREAD_UNCHANGED)
    frame = cv2.imread("../test_data/messi.jpg", cv2.IMREAD_UNCHANGED)
    # frame = cv2.imread("test_data/screen.jpg", cv2.IMREAD_UNCHANGED)

    emb = face_handler.extract_embedding(frame)
    print(emb)
    # # b = emb.tobytes()
    # hex = dnn_converter.encode_np_hex(emb)
    # print(hex)
    # print(len(hex))


for i in range(1):
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
