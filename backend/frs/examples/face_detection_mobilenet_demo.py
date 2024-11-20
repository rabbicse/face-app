import logging
import os

import cv2

from detection.retina_face.retina_face_detector import RetinaFaceDetector

logger = logging.getLogger(__name__)

image_path = '../samples/large-selfie-02.jpg'
img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
# detect.detect_face('test_data/family.jpg', model_path='models/Resnet50_Final.pth')

# detect.detect_face(image_path, model_path='models/mobilenet0.25_Final.pth', network='mobile0.25')
model_pth_path = os.path.abspath('../models/mobilenet0.25_Final.pth')
face_detect = RetinaFaceDetector(model_path=os.path.abspath('../models/mobilenet0.25_Final.pth'),
                                 model_tar=os.path.abspath("../models/mobilenetV1X0.25_pretrain.tar"),
                                 network='mobile0.25')
for i in range(1):
    faces = face_detect.detect_faces(img_raw=img_raw)
    print(f'Total faces: {len(faces[0])}')
    for face in faces:
        print(face)
