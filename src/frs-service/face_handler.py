import os

import numpy as np
import cv2.cv2 as cv2

from arc_face import preprocessor
from arc_face.recognition import Embedding
from retina_face.detect import FaceDetect


class FaceHandler:
    def __init__(self):
        """
        """
        self.thresh = 0.2
        self.scales = [240, 720]
        self.retina_face_model = os.path.abspath('models/mobilenet0.25_Final.pth')
        self.face_detector = FaceDetect(model_path=self.retina_face_model,
                                        network='mobile0.25')
        self.arc_face_model = os.path.abspath('models/backbone-r100.pth')
        self.recognition = Embedding('models/backbone-r100.pth')

    def detect_faces(self, img):
        try:
            # get image shape
            im_shape = img.shape

            target_size, max_size = self.scales
            im_size_min = np.min(im_shape[0:2])
            im_size_max = np.max(im_shape[0:2])

            # if im_size_min>target_size or im_size_max>max_size:
            im_scale = float(target_size) / float(im_size_min)
            # prevent bigger axis from being more than max_size:
            if np.round(im_scale * im_size_max) > max_size:
                im_scale = float(max_size) / float(im_size_max)

            # print('im_scale', im_scale)
            faces, landmarks = self.face_detector.detect_faces(img)
            face_list = []
            for i in range(len(faces)):
                face = {
                    'bbox': faces[i],
                    'landmark': landmarks[i]
                }
                face_list.append(face)

            return face_list
        except Exception as x:
            print(x)

    def pre_process_img(self, img):
        try:
            # get image shape
            im_shape = img.shape

            target_size, max_size = self.scales
            im_size_min = np.min(im_shape[0:2])
            im_size_max = np.max(im_shape[0:2])

            # if im_size_min>target_size or im_size_max>max_size:
            im_scale = float(target_size) / float(im_size_min)
            # prevent bigger axis from being more than max_size:
            if np.round(im_scale * im_size_max) > max_size:
                im_scale = float(max_size) / float(im_size_max)

            print('im_scale', im_scale)
            img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)

            # cv2.imshow('i', img)
            # cv2.waitKey(0)

            faces, landmarks = self.face_detector.detect_faces(img)
            # faces, landmarks = self.face_detector.detect_(img, self.thresh, scales=[im_scale], do_flip=False)

            return img, faces, landmarks
        except Exception as x:
            print(x)

    def pre_process_face(self, img, box, landmark5):
        try:
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

            ch = y2 - y1
            cw = x2 - x1
            margin = 0
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(x1 - margin / 2, 0)
            bb[1] = np.maximum(y1 - margin / 2, 0)
            bb[2] = np.minimum(x2 + margin / 2, img.shape[1])
            bb[3] = np.minimum(y2 + margin / 2, img.shape[0])
            # ret = img[bb[1]:bb[3], bb[0]:bb[2], :]
            ret = img[bb[1]:bb[3], bb[0]:bb[2], :]

            lan = [[l[0] - bb[0], l[1] - bb[1]] for l in landmark5]
            ln = np.array(lan, dtype=np.float32)

            crop_img = preprocessor.preprocess(ret, image_size=[112, 112], landmark=ln)
            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
            # crop_img = cv2.resize(crop_img, (112, 112))

            # cv2.imshow('Images', crop_img)
            # # Hit 'q' on the keyboard to quit!
            # cv2.waitKey(0)

            return crop_img
        except Exception as x:
            print(x)

    def extract_embedding(self, img):
        # detect faces
        img, faces, landmarks = self.pre_process_img(img)

        print(len(faces))

        # if not faces.any():
        #     return

        # extract rect and landmark=5
        box = faces[0]#.astype(np.int)
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        landmark5 = landmarks[0]#.astype(np.float32)

        # print(box)
        # print(landmark5)

        # preprocess face before get embeddings
        crop_img = self.pre_process_face(img, box, landmark5)

        # print(crop_img.shape)

        # extract embedding
        emb = self.recognition.get_embedding(crop_img)

        # print(emb)

        return emb

    def match(self, embedding1, embedding2):
        return self.recognition.compute_match(embedding1, embedding2)