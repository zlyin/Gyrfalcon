## import packages
import os
import sys
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

"""
- define a MobileNetV1 class for classification task
- args:
    - classes = # of prediction classes
    - in_dim = in_dim of feature map
"""
class MobileNetV1(nn.Module):

    # __init__()
    def __init__(self, in_dim=3, num_classes=1000):
        super(MobileNetV1, self).__init__()
        
        # define params
        self.in_dim = in_dim
        self.num_classes = num_classes

        # define model structures
        self.features = nn.Sequential(
                # 1st block
                self.conv_bn(in_dim, 32, 2),
                self.conv_dws(32, 64, 1),
                self.conv_dws(64, 128, 2),
                # 2nd block
                self.conv_dws(128, 128, 1),
                self.conv_dws(128, 256, 2),
                # 3rd block
                self.conv_dws(256, 256, 1),
                self.conv_dws(256, 512, 2),
                # 4th block - 5 repeats
                self.conv_dws(512, 512, 1),
                self.conv_dws(512, 512, 1),
                self.conv_dws(512, 512, 1),
                self.conv_dws(512, 512, 1),
                self.conv_dws(512, 512, 1),
                # 4th block - rest
                self.conv_dws(512, 1024, 2),
                self.conv_dws(1024, 1024, 1),
                )

        # define classifer layers
        self.avgPool = nn.AvgPool2d(kernel_size=7)
        self.fc = nn.Linear(in_features=1024, out_features=self.num_classes)
        pass


    def forward(self, x):
        x = self.features(x)
        x = self.avgPool(x)
        # flatten
        x = x.reshape(-1, 1024)
        scores = self.fc(x)
        return scores


    # define routine conv block
    def conv_bn(self, in_dim, out_dim, stride):
        layers = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=stride,
                    padding=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True),
                )
        return layers
    
    # define Depthwise Separable Conv block
    def conv_dws(self, in_dim, out_dim, stride):
        layers = nn.Sequential(
                # depthwise conv
                nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=stride, 
                    padding=1, groups=in_dim, bias=False),
                nn.BatchNorm2d(in_dim),
                nn.ReLU(inplace=True),
                # elementwise conv
                nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, 
                    padding=0, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True),
                )
        return layers

