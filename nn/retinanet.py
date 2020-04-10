## import packages
import os
import sys
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from fpn import FPN


"""
- define a RetinaNet class for Object Detection
- args:
    - classes = # of prediction classes
    - in_dim = in_dim of feature map
"""
class RetinaNet(nn.Module):

    # __init__()
    def __init__(self, in_dim=3, num_classes=20):
        super(RetinaNet, self).__init__()
        
        # define params
        self.fpn = FPN(version="resnet50")    # use ResNet50 FPN extract feature maps
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.num_anchors = 9
        # cls & regression branches
        self.loc_head = self.make_head(self.num_anchors * 4)
        self.cls_head = self.make_head(self.num_anchors * self.num_classes)

    def forward(self, x):
        # create feature maps
        fms = self.fpn(x)
        # placeholders for cls & regression predictions
        loc_preds = []
        cls_preds = []

        for fm in fms:
            # do cls & regression calcualation on each fm
            loc_pred = self.loc_head(fm)
            cls_pred = self.cls_head(fm)
            # permute channels & reshape from [batch, 36, H, W] => [batch, H*W*9, 4]
            loc_pred = loc_pred.permute(0, 2, 3, 1).reshape(x.shape[0], -1, 4)
            # permute channels & reshape from [batch, 180, H, W] => [batch, H*W*9, 20]
            cls_pred = cls_pred.permute(0, 2, 3, 1).reshape(x.shape[0], -1, 
                    self.num_classes)
            # record
            loc_preds.append(loc_pred)
            cls_preds.append(cls_pred)
            pass

        # concat across fms
        return torch.cat(loc_preds, 1), torch.cat(cls_preds, 1)
    
    # build loc_head & cls_head network
    def make_head(self, out_dim):
        layers = []
        for i in range(4):
            layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1,
                padding=1))
            layers.append(nn.ReLU(inplace=True))
            pass
        layers.append(nn.Conv2d(256, out_dim, kernel_size=3, stride=1,
                padding=1))
        return nn.Sequential(*layers)

    # batch_size is too small, freeze BN params
    def freeze_bn(self):
        for layer in self.module():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
        pass

