import logging
import os.path

import cv2

from recognition.arc_face import arc_face
from utils.vision_utils.decorators import timeit

logger = logging.getLogger(__name__)

rec = arc_face.ArcFace(os.path.abspath('../models/backbone-r18m.pth'), model_architecture="r18")


@timeit
def test_recognize():
    img = cv2.imread('../samples/480.jpg', cv2.IMREAD_UNCHANGED)
    frm = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    frm = cv2.resize(frm, (112, 112))
    emb = rec.get_embedding(frm)
    print(emb)
    print(emb.shape)

    res = rec.compute_match(emb, emb)
    print(res)


for i in range(5):
    test_recognize()
