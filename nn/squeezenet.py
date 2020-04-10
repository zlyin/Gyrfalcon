## import packages
import os
import sys
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

"""
- define a SqueezeNet with a classification head
- args:
    - classes = # of prediction classes
    - in_dim = in_dim of feature map
    - squeeze_dim = out_dim of squeeze layer, S1
    - expand_dim = out_dim of 2 branches of expand layer, e1 = e2 = 4*S1
"""
class Fire(nn.Module):
    # initi
    def __init__(self, in_dim, squeeze_dim, expand_dim):
        super(Fire, self).__init__()

        # squeeze layer, conv1x1
        self.conv1 = nn.Conv2d(in_dim, squeeze_dim, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(squeeze_dim)
        self.relu1 = nn.ReLU(inplace=True)

        # expand layer, 2 branches;
        # e1 = conv1x1, e2 = conv3x3 
        self.conv2 = nn.Conv2d(squeeze_dim, expand_dim, kernel_size=1,
                stride=1)
        self.bn2 = nn.BatchNorm2d(expand_dim)

        self.conv3 = nn.Conv2d(squeeze_dim, expand_dim, kernel_size=3,
                stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(expand_dim)
        self.relu2 = nn.ReLU(inplace=True)
        pass

    def forward(self, x):
        # feed in squeeze layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # feed into expand layer
        out1 = self.conv2(x)
        out1 = self.bn2(out1)

        out2 = self.conv3(x)
        out2 = self.bn3(out2)

        # concate in channels
        output = torch.cat([out1, out2], 1)
        output = self.relu2(output)
        return output


"""
- define a SqueezeNet with a classification head
- args:
    - num_classes = # of prediction classes
    - version = 1.0 or 1.1, default=1.0
    - in_dim = input dim, default=3 
"""
class SqueezeNet(nn.Module):
    # define init method
    def __init__(self, version=1.0, num_classes=1000, in_dim=3):
        super(SqueezeNet, self).__init__()
        # params
        if version not in [1.0, 1.1]:
            raise ValueError("Unsupported SqueezeNet version! Either 1.0 or 1.1 is expected!")
        self.num_classes = num_classes
        self.version = version
        self.in_dim = in_dim

        if self.version == 1.0:
            self.features= nn.Sequential(
                    nn.Conv2d(self.in_dim, 96, kernel_size=7, stride=2),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                    Fire(in_dim=96, squeeze_dim=16, expand_dim=64),     # Fire2
                    Fire(in_dim=128, squeeze_dim=16, expand_dim=64),    # Fire3 
                    Fire(in_dim=128, squeeze_dim=32, expand_dim=128),   # Fire4
                    nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                    Fire(in_dim=256, squeeze_dim=32, expand_dim=128),     # Fire5
                    Fire(in_dim=256, squeeze_dim=48, expand_dim=192),     # Fire6
                    Fire(in_dim=384, squeeze_dim=48, expand_dim=192),     # Fire7
                    Fire(in_dim=384, squeeze_dim=64, expand_dim=256),     # Fire8
                    nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                    Fire(in_dim=512, squeeze_dim=64, expand_dim=256),     # Fire9
                    )
        else:
            # change first conv7x7 => conv3x3
            # advance MaxPool2d to reduce size of feature map
            self.features= nn.Sequential(
                    nn.Conv2d(self.in_dim, 64, kernel_size=3, stride=2),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                    Fire(in_dim=64, squeeze_dim=16, expand_dim=64),     # Fire2
                    Fire(in_dim=128, squeeze_dim=16, expand_dim=64),    # Fire3 
                    nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                    Fire(in_dim=128, squeeze_dim=32, expand_dim=128),   # Fire4
                    Fire(in_dim=256, squeeze_dim=32, expand_dim=128),     # Fire5
                    nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                    Fire(in_dim=256, squeeze_dim=48, expand_dim=192),     # Fire6
                    Fire(in_dim=384, squeeze_dim=48, expand_dim=192),     # Fire7
                    Fire(in_dim=384, squeeze_dim=64, expand_dim=256),     # Fire8
                    Fire(in_dim=512, squeeze_dim=64, expand_dim=256),     # Fire9
                    )

        # final conv & classifier
        self.conv10 = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                self.conv10,
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                )
        pass
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        # flatten scores
        out = x.reshape(x.shape[0], self.num_classes)
        return out




