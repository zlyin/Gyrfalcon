## import packages

## import packages
import os
import sys
import torch
from torch import nn
import torch.nn.functional as F


"""
- define DarkNet class
"""
class DarkNet19(nn.Module):
    def __init__(self, classes=20):
        super(DarkNet19, self).__init__()
        
        # predefine network structures
        net_cfgs = [
                # conv1s
                [(32, 3)], 
                ["MaxPool", (64, 3)],
                ["MaxPool", (128, 3), (64, 1), (128, 3)],
                ["MaxPool", (256, 3), (128, 1), (256, 3)],
                ["MaxPool", (512, 3), (256, 1), (512, 3), (256, 1), (512, 3)],
                # conv2
                ["MaxPool", (1024, 3), (512, 1), (1024, 3), (512, 1), (1024, 3)],
                # conv3
                [(1024, 3), (1024, 3)],
                # pass through input here
                # conv4
                [(1024, 3)],
                ]
        
        # define conv modules
        self.conv1s, c1 = self.make_layers(3, net_cfgs[0:5])
        self.conv2, c2 = self.make_layers(c1, net_cfgs[5])
        self.conv3, c3 = self.make_layers(c2, net_cfgs[6])
        # implement passthrough
        stride = 2
        
        # concate, in_dim = conv1s x 4 + c3
        self.conv4, c4 = self.make_layer((c1 * (stride * stride) + c3), net_cfgs[7])
        # final output dim = 5 x 25
        out_dim = 5 * (classes + 5)
        self.conv5 = nn.Sequential([
            nn.Conv2d(c4, out_dim, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_dim, momentum=0.01),
            ])
        # globalAvgPool
        self.avgPool = nn.AvgPool2d((1, 1))
        pass


    # define a make_layer method
    def make_layer(self, in_dim, net_cfgs):
        layers = []
        
        # if input is a list, recursively call make_layers, otherwise, build
        # each block
        if len(net_cfgs) > 0 and isinstance(net_cfgs[0], list):
            for cfg in net_cfgs:
                layer, in_dim = self.make_layers(in_dim, cfg)
                layers.append(layer)
            pass
        else:
            for item in net_cfgs:
                # MaxPooling2d
                if item == "MaxPool":
                    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                # Conv
                else:
                    out_dim, k_size = item
                    padding = int(k_size / 2)
                    layers.extend([
                        nn.Conv2d(in_dim, out_dim, k_size, stride=1, padding=padding),
                        nn.BatchNorm2d(out_dim, momentum=0.01),
                        nn.LeakyReLU(0.1, inplace=True),
                        ]
                    # update in_dim
                    in_dim = out_dim
                pass
            pass
        
        # return
        return nn.Sequential(*layers), in_dim

    



