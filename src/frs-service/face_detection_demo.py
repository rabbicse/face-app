import logging

from cv2 import cv2

from retina_face import detect
from retina_face.detect import FaceDetect

logger = logging.getLogger(__name__)

image_path = 'test_data/family.jpg'
img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
# detect.detect_face('test_data/family.jpg', model_path='models/Resnet50_Final.pth')

# detect.detect_face(image_path, model_path='models/mobilenet0.25_Final.pth', network='mobile0.25')

face_detect = FaceDetect(model_path='models/mobilenet0.25_Final.pth', network='mobile0.25')
for i in range(5):
    face_detect.detect_faces(img_raw=img_raw)