# -*- coding: utf-8 -*-
import logging
from typing import Tuple

import cv2
import numpy as np
import sklearn
from sklearn import preprocessing
import torch
from numpy.linalg import norm
from skimage import transform as trans
from recognition.arc_face.backbones import get_model
from utils.vision_utils.decorators import TimeitDecorator

logger = logging.getLogger(__name__)


class ArcFace(object):
    def __init__(self,
                 model_path: str,
                 data_shape: Tuple[3] = (3, 112, 112),
                 batch_size: int = 1,
                 model_architecture: str = 'r100',
                 device: str = 'cpu'):
        """
        @param model_path:
        @param data_shape:
        @param batch_size:
        @param model_architecture:
        """
        self.device: torch.device = torch.device(device)
        self.image_size = (112, 112)
        # weight = torch.load(model_path, map_location=self.device, weights_only=True)
        # resnet = get_model(model_architecture, dropout=0, fp16=False).cpu()
        # resnet.load_state_dict(weight)
        # self.model = torch.nn.DataParallel(resnet)
        # self.model.eval()
        self.batch_size = batch_size
        self.data_shape = data_shape
        self.model = self.__load_model(model_path, model_architecture)

    def __load_model(self, model_path: str, model_architecture: str):
        weight = torch.load(model_path, map_location=self.device, weights_only=True)
        resnet = get_model(model_architecture, dropout=0, fp16=False).to(self.device)
        resnet.load_state_dict(weight)
        model = torch.nn.DataParallel(resnet)
        model.eval()
        return model

    @TimeitDecorator()
    @torch.no_grad()
    def forward_db(self, batch_data):
        """
        @param batch_data:
        @return:
        """
        imgs = torch.Tensor(batch_data).cpu()
        imgs.div_(255).sub_(0.5).div_(0.5)
        feat = self.model(imgs)
        return feat.cpu().numpy()

    def get_embedding(self,
                      img: np.ndarray,
                      rgb_convert: bool = False):
        """
        @param img:
        @param rgb_convert:
        @return:
        """
        if img.shape[2] != 3:
            logger.warning('Wrong image!')
            return

        if img.shape[0:2] != self.image_size:
            logger.warning('Wrong image size!')
            return

        if rgb_convert:
            data = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            data = img

        data = np.transpose(data, (2, 0, 1))  # 3*112*112, RGB

        img_orig = torch.from_numpy(data).unsqueeze(0).float()

        emb = self.forward_db(img_orig)
        return sklearn.preprocessing.normalize(emb).flatten()

    def compute_sim(self, img1, img2):
        """
        @param img1:
        @param img2:
        @return:
        """
        emb1 = self.get_embedding(img1).flatten()
        emb2 = self.get_embedding(img2).flatten()
        return np.dot(emb1, emb2) / (norm(emb1) * norm(emb2))

    def compute_match(self, emb1, emb2):
        """
        @param emb1:
        @param emb2:
        @return:
        """
        return np.dot(emb1, emb2) / (norm(emb1) * norm(emb2))
