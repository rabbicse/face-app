import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models._utils as _utils

from retina_face.net import FPN as FPN
from retina_face.net import MobileNetV1 as MobileNetV1
from retina_face.net import SSH as SSH


class ClassHead(nn.Module):
    def __init__(self, in_channels=512, num_anchors=3):
        """
        @param in_channels:
        @param num_anchors:
        """
        super(ClassHead, self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(in_channels, self.num_anchors * 2, kernel_size=(1, 1), stride=(1,), padding=0)

    def forward(self, x):
        """
        @param x:
        @return:
        """
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 2)


class BboxHead(nn.Module):
    def __init__(self, in_channels=512, num_anchors=3):
        """
        @param in_channels: 
        @param num_anchors: 
        """
        super(BboxHead, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=(1, 1), stride=(1,), padding=0)

    def forward(self, x):
        """
        @param x:
        @return:
        """
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 4)


class LandmarkHead(nn.Module):
    def __init__(self, in_channels=512, num_anchors=3):
        """
        @param in_channels:
        @param num_anchors:
        """
        super(LandmarkHead, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, num_anchors * 10, kernel_size=(1, 1), stride=(1,), padding=0)

    def forward(self, x):
        """
        @param x:
        @return:
        """
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 10)


class RetinaFace(nn.Module):
    def __init__(self, cfg=None, mobilenet_model_tar="./models/mobilenetV1X0.25_pretrain.tar", phase='train'):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(RetinaFace, self).__init__()
        self.phase = phase
        self.cfg = cfg
        self.mobilenet_model_tar = mobilenet_model_tar

        self.body = None
        self.fpn = None
        self.ssh1 = None
        self.ssh2 = None
        self.ssh3 = None
        self.class_head = None
        self.bbox_head = None
        self.landmark_head = None

        self.initialize_network()

    def initialize_network(self):
        backbone = None
        if self.cfg['name'] == 'mobilenet0.25':
            backbone = MobileNetV1()
            checkpoint = torch.load(self.mobilenet_model_tar, map_location=torch.device('cpu'))
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = k[7:]  # remove module.
                new_state_dict[name] = v
            # load params
            backbone.load_state_dict(new_state_dict)
        elif self.cfg['name'] == 'Resnet50':
            import torchvision.models as models
            backbone = models.resnet50(pretrained=self.cfg['pretrain'])

        self.body = _utils.IntermediateLayerGetter(backbone, self.cfg['return_layers'])
        in_channels_stage2 = self.cfg['in_channel']
        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        out_channels = self.cfg['out_channel']
        self.fpn = FPN(in_channels_list, out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.class_head = self._make_class_head(fpn_num=3, in_channels=self.cfg['out_channel'])
        self.bbox_head = self._make_bbox_head(fpn_num=3, in_channels=self.cfg['out_channel'])
        self.landmark_head = self._make_landmark_head(fpn_num=3, in_channels=self.cfg['out_channel'])

    def _make_class_head(self, fpn_num=3, in_channels=64, anchor_num=2):
        """
        @param fpn_num:
        @param in_channels:
        @param anchor_num:
        @return:
        """
        class_head = nn.ModuleList()
        for i in range(fpn_num):
            class_head.append(ClassHead(in_channels, anchor_num))
        return class_head

    def _make_bbox_head(self, fpn_num=3, in_channels=64, anchor_num=2):
        """
        @param fpn_num:
        @param in_channels:
        @param anchor_num:
        @return:
        """
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(in_channels, anchor_num))
        return bboxhead

    def _make_landmark_head(self, fpn_num=3, in_channels=64, anchor_num=2):
        """
        @param fpn_num:
        @param in_channels:
        @param anchor_num:
        @return:
        """
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(in_channels, anchor_num))
        return landmarkhead

    def forward(self, inputs):
        """
        @param inputs:
        @return:
        """
        out = self.body(inputs)

        # FPN
        fpn = self.fpn(out)

        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        bbox_regressions = torch.cat([self.bbox_head[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.class_head[i](feature) for i, feature in enumerate(features)], dim=1)
        ldm_regressions = torch.cat([self.landmark_head[i](feature) for i, feature in enumerate(features)], dim=1)

        if self.phase == 'train':
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
        return output
