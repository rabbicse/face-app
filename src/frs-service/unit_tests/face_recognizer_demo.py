import logging

from cv2 import cv2

# from ..arcface import recognition
import recognition

logger = logging.getLogger(__name__)

rec = recognition.Embedding('../../models/backbone-r100.pth')
img = cv2.imread('../../test_data/screen.jpg', cv2.IMREAD_UNCHANGED)
frm = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
frm = cv2.resize(frm, (112, 112))
emb = rec.get_embedding(frm)
print(emb)

res = rec.compute_match(emb, emb)
print(res)