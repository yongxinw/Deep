# Created by yongxinwang at 2019-12-08 17:22
import torch.nn as nn
import numpy as np
import resnet


class DeepVP(nn.Module):
    def __init__(self, resnet_pretrained, resnet_usebn, grid_resolution, resnet_depth):
        super(DeepVP, self).__init__()

        if resnet_depth == 18:
            resnet_func = resnet.resnet18
            fc_in = 512
        elif resnet_depth == 34:
            resnet_func = resnet.resnet34
            fc_in = 512
        elif resnet_depth == 50:
            resnet_func = resnet.resnet50
            fc_in = 2048
        elif resnet_depth == 101:
            resnet_func = resnet.resnet101
            fc_in = 2048
        else:
            raise NotImplementedError
        fc_out = int(np.prod(grid_resolution))

        self.conv1 = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1)

        self.feature_extractor = resnet_func(pretrained=resnet_pretrained,
                                             use_bn=resnet_usebn)
        self.fc = nn.Sequential(
            nn.Linear(fc_in, fc_out),
            nn.ReLU(inplace=True),
            nn.Linear(fc_out, fc_out)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.feature_extractor(x)
        x = self.fc(x)
        return x