import logging
import os

import cv2

from detection.retina_face.retina_face_detector import RetinaFaceDetector
from utils.vision_utils.drawing_utils import resize_image_to_monitor, draw_img

logger = logging.getLogger(__name__)

image_path = '../samples/large-selfie-02.jpg'
img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
# detect.detect_face('test_data/family.jpg', model_path='models/Resnet50_Final.pth')

# detect.detect_face(image_path, model_path='models/mobilenet0.25_Final.pth', network='mobile0.25')
model_pth_path = os.path.abspath('../models/mobilenet0.25_Self.pth')
face_detect = RetinaFaceDetector(model_path=model_pth_path,
                                 model_tar=os.path.abspath("../models/mobilenetV1X0.25_pretrain.tar"),
                                 network='mobile0.25')

faces = face_detect.detect_faces(img_raw=img_raw)
print(f'Total faces: {len(faces)}')
for face in faces:
    print(face)

# draw over image
canvas = resize_image_to_monitor(img_raw)
draw_img(img_raw, faces, is_fullscreen=True)
cv2.waitKey()
cv2.destroyAllWindows()