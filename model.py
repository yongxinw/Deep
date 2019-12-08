# Created by yongxinwang at 2019-12-08 17:22
import torch
import torch.nn as nn
import torchvision.models as models


class DeepVP(nn.Module):
    def __init__(self, config):
        super(DeepVP, self).__init__()

        self.feature_extractor = models.resnet50()

    def forward(self, x):

        pass