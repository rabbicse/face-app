from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from utils.dnn_utils.box_utils import decode, decode_landm
from utils.dnn_utils.py_cpu_nms import py_cpu_nms
from detection.retina_face.data import cfg_mnet, cfg_re50
from detection.retina_face.prior_box import PriorBox
from detection.retina_face.retina_face import RetinaFace

CONFIDENCE_THRESHOLD = 0.02
TOP_K = 5000
NMS_THRESHOLD = 0.4
KEEP_TOP_K = 750
VIS_THRESHOLD = 0.5


class RetinaFaceDetector:
    def __init__(self,
                 model_path: str,
                 model_tar: str = None,
                 network: str = "resnet50",
                 device: str = "cpu"):
        """
        @param model_path:
        @param network:
        @param cpu:
        """
        torch.set_grad_enabled(False)
        self.cfg = None
        self.device: torch.device = torch.device(device)

        self.net = self.__create_network(model_path=model_path,
                                         model_tar=model_tar,
                                         network=network)

    def __create_network(self,
                         model_path: str,
                         model_tar: str = None,
                         network: str = "resnet50"):
        """
        @param model_path:
        @param network:
        @param cpu:
        @return:
        """
        if network == "mobile0.25":
            self.cfg = cfg_mnet
        elif network == "resnet50":
            self.cfg = cfg_re50

        # net and model
        net = RetinaFace(cfg=self.cfg, phase=None, model_tar=model_tar)
        net = self.__load_model(net, model_path, load_to_cpu=True)
        return net.to(self.device)

    def __load_model(self,
                     model: nn.Module,
                     pretrained_path: str,
                     load_to_cpu: bool = True):
        """
        @param model:
        @param pretrained_path:
        @param load_to_cpu:
        @return:
        """
        if load_to_cpu:
            pretrained_dict = torch.load(pretrained_path,
                                         map_location=lambda storage, loc: storage,
                                         weights_only=True)
        else:
            device = torch.cuda.current_device()
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = self.remove_prefix(pretrained_dict['state_dict'], 'module.')
        else:
            pretrained_dict = self.remove_prefix(pretrained_dict, 'module.')
        if self.check_keys(model, pretrained_dict):
            model.load_state_dict(pretrained_dict, strict=False)
            return model

    def check_keys(self,
                   model: nn.Module,
                   pretrained_state_dict: Dict):
        """
        @param model:
        @param pretrained_state_dict:
        @return:
        """
        ckpt_keys = set(pretrained_state_dict.keys())
        model_keys = set(model.state_dict().keys())
        used_pretrained_keys = model_keys & ckpt_keys

        return len(used_pretrained_keys) > 0

    def remove_prefix(self, state_dict, prefix):
        """
        Old style model is stored with all names of parameters sharing common prefix 'module.
        @param state_dict:
        @param prefix:
        @return:
        """
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        return {f(key): value for key, value in state_dict.items()}

    def detect_faces(self, img_raw):
        """
        @param img_raw:
        @return:
        """
        resize = 1
        img = np.float32(img_raw)
        # print(img.shape)

        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)

        loc, conf, landms = self.net(img)  # forward pass

        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(self.device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > CONFIDENCE_THRESHOLD)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:TOP_K]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, NMS_THRESHOLD)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:KEEP_TOP_K, :]
        landms = landms[:KEEP_TOP_K, :]

        dets = np.concatenate((dets, landms), axis=1)

        detections = []
        landmarks = []

        # show image
        for b in dets:
            if b[4] < VIS_THRESHOLD:
                continue

            # detections.append((b[0], b[1], b[2], b[3]))
            bbox = {
                'x_min': float(b[0] / im_width),
                'y_min': float(b[1] / im_height),
                'x_max': float(b[2] / im_width),
                'y_max': float(b[3] / im_height),
                'score': float(b[4])
            }
            # landmarks.append([(b[5], b[6]),
            #                   (b[7], b[8]),
            #                   (b[9], b[10]),
            #                   (b[11], b[12]),
            #                   (b[13], b[14])])

            landmarks = {
                'left_eye': {
                    'x': float(b[5] / im_width),
                    'y': float(b[6] / im_height)
                },
                'right_eye': {
                    'x': float(b[7] / im_width),
                    'y': float(b[8] / im_height)
                },
                'nose': {
                    'x': float(b[9] / im_width),
                    'y': float(b[10] / im_height)
                },
                'left_lip': {
                    'x': float(b[11] / im_width),
                    'y': float(b[12] / im_height)
                },
                'right_lip': {
                    'x': float(b[13] / im_width),
                    'y': float(b[14] / im_height)
                }
            }

            detections.append({
                'bbox': bbox,
                'landmarks': landmarks
            })

        # return detections, landmarks

        return detections
