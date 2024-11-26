import os
from typing import Dict

import numpy as np
import cv2

from utils.vision_utils import log_utils
from utils.vision_utils.decorators import timeit
from utils.dnn_utils import preprocessor
from utils.dnn_utils.pose_estimation import PoseEstimation

from detection.retina_face.retina_face_detector import RetinaFaceDetector
from recognition.arc_face.arc_face import ArcFace
from utils.vision_utils.singleton_decorator import SingletonDecorator


@SingletonDecorator
class FaceService:
    def __init__(self,
                 det_model_path: str,
                 det_model_tar: str,
                 rec_model_path: str,
                 det_network: str = 'mobile0.25',
                 rec_network: str = 'r100',
                 device: str = 'cpu'):
        """
        """
        self.logger = log_utils.LogUtils().get_logger(self.__class__.__name__)
        self.thresh = 0.2
        self.scales = [240, 720]
        # self.retina_face_model = det_model_path  # os.path.abspath('models/mobilenet0.25_Final.pth')
        self.face_detector = RetinaFaceDetector(model_path=det_model_path,
                                                model_tar=det_model_tar,
                                                network=det_network,
                                                device=device)
        # self.arc_face_model = os.path.abspath('models/backbone-r100.pth')
        self.recognition = ArcFace(model_path=rec_model_path,
                                   model_architecture=rec_network,
                                   device=device)
        self.pose_estimator = PoseEstimation()

    @timeit
    def detect_faces(self, img: np.ndarray):
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
            faces = self.face_detector.detect_faces(img)
            self.logger.info(f'Total faces found: {len(faces)}')
            # face_list = []
            # for i in range(len(faces)):
            #     face = {
            #         'bbox': faces[i],
            #         'landmark': landmarks[i]
            #     }
            #     face_list.append(face)
            return faces
        except Exception as x:
            self.logger.error(f'Error when detect faces. Details: {x}')

    @timeit
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

            faces = self.face_detector.detect_faces(img)
            return img, faces
        except Exception as x:
            self.logger.error(f'Error when pre-process0image. Details: {x}')

    @timeit
    def pre_process_face(self,
                         img: np.ndarray,
                         face: Dict):
        """
        :param img:
        :param box:
        :param landmark5:
        :return:
        """
        try:
            height, width, _ = img.shape
            box = face['bbox']
            landmarks = face['landmarks']
            x1, y1, x2, y2 = box['x_min'] * width, box['y_min'] * height, box['x_max'] * width, box['y_max'] * height

            margin = 0
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(x1 - margin / 2, 0)
            bb[1] = np.maximum(y1 - margin / 2, 0)
            bb[2] = np.minimum(x2 + margin / 2, width)
            bb[3] = np.minimum(y2 + margin / 2, height)
            ret = img[bb[1]:bb[3], bb[0]:bb[2], :]

            lan = [[l['x'] * width - bb[0], l['y'] * height - bb[1]] for l in landmarks.values()]
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

    @timeit
    def extract_embedding(self, img: np.ndarray):
        """
        :param img:
        :return:
        """
        # detect faces
        img, faces = self.pre_process_img(img)

        self.logger.info(f'Total faces detected: {len(faces)}')
        if len(faces) <= 0:
            return -1

        # extract rect and landmark=5
        box = faces[0]
        face_bbox = box['bbox']
        landmarks = box['landmarks']

        # check face pose
        if not self.pose_estimator.estimate(img, box, landmarks):
            return -2

        # preprocess face before get embeddings
        # todo:
        crop_img = self.pre_process_face(img, box)

        # extract embedding
        emb = self.recognition.get_embedding(crop_img)

        return emb

    def match(self, embedding1, embedding2):
        """
        :param embedding1:
        :param embedding2:
        :return: score
        """
        return self.recognition.compute_match(embedding1, embedding2)
