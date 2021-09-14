import os

import numpy as np
import cv2.cv2 as cv2

from arc_face import preprocessor
from arc_face.recognition import Embedding
from dnn_utils.pose_estimation import PoseEstimation
from retina_face.retina_face_detector import RetinaFaceDetector
from vision_utils import log_utils
from vision_utils.decorators import timeit, TimeitDecorator

DETECTOR_MODEL_PATH = 'models/mobilenet0.25_Final.pth'
DETECTOR_MODEL_TAR_PATH = 'models/mobilenetV1X0.25_pretrain.tar'
RECOGNIZER_MODEL_PATH = 'models/backbone-r100m.pth'


class FaceHandler:
    def __init__(self, dnn_config, detector_network='mobile0.25', debug=False):
        self.logger = log_utils.LogUtils().get_logger(self.__class__.__name__)
        self.__debug = debug
        self.thresh = 0.2
        self.scales = [240, 720]

        # initialize detector
        retina_face_model_path = dnn_config['detector_model_path'] \
            if 'detector_model_path' in dnn_config else os.path.abspath(DETECTOR_MODEL_PATH)
        retina_face_model_tar_path = dnn_config['detector_model_tar_path'] \
            if 'detector_model_tar_path' in dnn_config else os.path.abspath(DETECTOR_MODEL_TAR_PATH)
        self.face_detector = RetinaFaceDetector(model_path=retina_face_model_path,
                                                model_tar=retina_face_model_tar_path,
                                                network=detector_network)

        # initialize recognizer
        arc_face_model_path = dnn_config['recognizer_model_path'] \
            if 'recognizer_model_path' in dnn_config else os.path.abspath(DETECTOR_MODEL_PATH)

        self.face_recognizer = Embedding(arc_face_model_path, model_architecture='r100')
        self.pose_estimator = PoseEstimation()

    @TimeitDecorator()
    def detect_faces(self, img):
        """
        :param img:
        :return:
        """
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
            self.logger.error(f'Error when detect faces. Details: {x}')

    @TimeitDecorator()
    def pre_process_img(self, img):
        """
        :param img:
        :return:
        """
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

            self.logger.info(f'Original image scale: {im_scale}')
            img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)

            # cv2.imshow('i', img)
            # cv2.waitKey(0)

            faces, landmarks = self.face_detector.detect_faces(img)
            return img, faces, landmarks
        except Exception as x:
            self.logger.error(f'Error when pre-process0image. Details: {x}')

    @TimeitDecorator()
    def pre_process_face(self, img, box, landmark5):
        """
        :param img:
        :param box:
        :param landmark5:
        :return:
        """
        try:
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

            margin = 0
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(x1 - margin / 2, 0)
            bb[1] = np.maximum(y1 - margin / 2, 0)
            bb[2] = np.minimum(x2 + margin / 2, img.shape[1])
            bb[3] = np.minimum(y2 + margin / 2, img.shape[0])
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
            self.logger.error(f'Error when pre-process-face. Details: {x}')

    @TimeitDecorator()
    def extract_embedding(self, img):
        """
        :param img:
        :return:
        """
        # detect faces
        img, faces, landmarks = self.pre_process_img(img)

        self.logger.info(f'Total faces detected: {len(faces)}')
        if len(faces) <= 0:
            return -1

        # extract rect and landmark=5
        box = faces[0]
        landmark5 = landmarks[0]

        # check face pose
        if not self.pose_estimator.estimate(img, box, landmark5):
            return -2

        # preprocess face before get embeddings
        crop_img = self.pre_process_face(img, box, landmark5)

        # extract embedding
        emb = self.face_recognizer.get_embedding(crop_img)

        return emb

    @TimeitDecorator()
    def match(self, embedding1, embedding2):
        """
        :param embedding1:
        :param embedding2:
        :return: score
        """
        return self.recognition.compute_match(embedding1, embedding2)
