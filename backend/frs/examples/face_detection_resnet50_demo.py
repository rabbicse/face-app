import logging
import os

import cv2

from detection.retina_face.retina_face_detector import RetinaFaceDetector
from utils.vision_utils.drawing_utils import draw_img, resize_image_to_monitor

logger = logging.getLogger(__name__)

image_path = '../samples/large-selfie-02.jpg'
img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
face_detect = RetinaFaceDetector(model_path=os.path.abspath('../models/Resnet50_Final.pth'),
                                 network='resnet50')

faces = face_detect.detect_faces(img_raw=img_raw)
print(f'Total faces: {len(faces)}')
for face in faces:
    print(face)

# draw over image
canvas = resize_image_to_monitor(img_raw)
draw_img(img_raw, faces, is_fullscreen=True)
cv2.waitKey()
cv2.destroyAllWindows()
