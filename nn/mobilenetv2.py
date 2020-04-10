## import packages
import os
import sys
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


"""
- define a InvertedResidual block for MobileNetV2 that consists of Conv1x1 =>
  DwS Conv => Conv1x1
- args:
    - in_dim = in_dim of feature map
    - out_dim = out_dim of feature map
    - stride = stride for DwS Conv, either 1 or 2
    - expand_ratio = out_dim of 1st Conv1x1 = in_dim * expand_ratio
"""
class InvertedResidual(nn.Module):
    
    def __init__(self, in_dim, out_dim, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        
        # define params
        self.stride = stride
        assert stride in [1, 2]
        hid_dim = int(expand_ratio * in_dim)

        # Conv1x1 => DwS Conv3x3 => Conv1x1
        self.conv1 = nn.Conv2d(in_dim, hid_dim, kernel_size=1, stride=1,
                padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(hid_dim)
        self.relu1 = nn.ReLU6(inplace=True)

        self.conv2 = nn.Conv2d(hid_dim, hid_dim, kernel_size=3,
                stride=self.stride, padding=1, groups=hid_dim, bias=False)
        self.bn2 = nn.BatchNorm2d(hid_dim)
        self.relu2 = nn.ReLU6(inplace=True)

        self.conv3 = nn.Conv2d(hid_dim, out_dim, kernel_size=1, stride=1, 
                padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_dim)

        # shortcut
        if (self.stride == 1) and (in_dim != out_dim):
            self.shortcut = nn.Sequential(
                    nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1,
                        padding=0, bias=False),
                    nn.BatchNorm2d(out_dim),
                    )
        else:
            self.shortcut = nn.Sequential()
        pass


    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        # add shortcut
        if self.stride == 1:
            output = out + self.shortcut(x) 
        else:
            output = out
        return output


"""
- define a MobileNetV2 class for classification task
- args:
    - classes = # of prediction classes
    - in_dim = in_dim of feature map
"""
class MobileNetV2(nn.Module):

    # __init__()
    def __init__(self, in_dim=3, num_classes=1000, expand_ratio=1.0):
        super(MobileNetV2, self).__init__()
        
        # define params
        self.num_classes = num_classes
        # [exp_ratio, out_dim, n_times, stride] for each residual block
        self.res_block_cfg = [
                (1,  16, 1, 1),
                (6,  24, 2, 2),     # NOTE: change stride 2 -> 1 for CIFAR10 
                (6,  32, 3, 2),
                (6,  64, 4, 2),
                (6,  96, 3, 1),
                (6, 160, 3, 2),
                (6, 320, 1, 1),
               ]    

        # conv3x3 
        self.conv1 = nn.Conv2d(in_dim, 32, kernel_size=3, stride=1, padding=1,
                bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        # 7 residual blocks 
        self.residual_blocks = self.make_layers(32)        

        # conv1x1 
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0,
                bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        # avgPool7x7
        self.avgPool = nn.AdaptiveAvgPool2d(output_size=(1,1))  # may change

        # flatten & fc layer
        self.fc = nn.Linear(in_features=1280, out_features=self.num_classes)
        pass


    def make_layers(self, in_dim):
        layers = []

        # loop over data from self.res_block_cfg
        for exp_ratio, out_dim, n_times, stride in self.res_block_cfg:

            # create strides for n_times
            strides = [stride] + [1] * (n_times - 1)
            for stride in strides:
                layers.append(InvertedResidual(in_dim, out_dim, stride, exp_ratio))
                # upadte in_dim
                in_dim = out_dim
            pass
        return nn.Sequential(*layers)


    def forward(self, x):
        # conv1
        out = self.conv1(x)
        out = self.bn1(out)

        # residual blocks
        out = self.residual_blocks(out)

        # conv2
        out = self.conv2(out)
        out = self.bn2(out)

        # avgPool
        out = self.avgPool(out)
        # flatten
        out = out.reshape(out.shape[0], -1)
        scores = self.fc(out)
        return scores


