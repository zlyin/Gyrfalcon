## import packages
import torch
from torch import nn
import numpy as np
from resnet_bottleneck import *


"""
- Define ResNet general components and construct ResNet18, ResNet50, ResNet152
- BasicConv = Conv7x7 => maxpool 3x3
- 4 Conv layers  = different number of Bottleneckv2 block(for 18) or 
Bottleneckv2 blocks (for 50 & 152)
- Head layer = AvgPool => FC => Softmax
"""

class ResNet(nn.Module):
    """
    - inputs:
    - block_sizes = [4 numbers] of bottleneck blocks 
    - classes = num of classes
    """
    def __init__(self, blocks_sizes, classes=1000):
        # inherit from parant class
        super(ResNet, self).__init__()

        # Basic Conv layer = Conv7x7 => MaxPool3x3
        self.basicconv = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                ) 

        # 4 layers consist of multiple bottleneck blocks
        in_dim = 64
        out_dim = 64 
        self.layer1 = self.build_layer(in_dim, out_dim, blocks_sizes[0], stride=1)

        in_dim = out_dim 
        out_dim *= 2
        self.layer2 = self.build_layer(in_dim, out_dim, blocks_sizes[1], stride=2)

        in_dim = out_dim 
        out_dim *= 2
        self.layer3 = self.build_layer(in_dim, out_dim, blocks_sizes[2], stride=2)

        in_dim = out_dim 
        out_dim *= 2
        self.layer4 = self.build_layer(in_dim, out_dim, blocks_sizes[3], stride=2)

        # Head layer = AvgPool => FC 
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(out_dim, classes)
        pass


    """
    - inputs:
    - block_size = num of bottleneck blocks in this layer
    - in_dim, out_dim = input dim & out dim of THIS layer! not for the each
      block!
    - stride = stride for conv3x3/downsample in the FIRST bottleneck block of
      each layer!
    - eg. 2nd layer has 2 blocks (layer_in_dim=64, layer_out_dim=128):
        - 1st block has 2 layers (block_in_dim=64, block_out_dim=128):
            - Conv3x3 => in_dim = 64, out_dim=128
            - Conv3x3 => in_dim = *128*, out_dim=128
        - 2nd block has 2 layers (block_in_dim=*128*, block_out_dim=128):
            - Conv3x3 => in_dim = 128, out_dim=128
            - Conv3x3 => in_dim = 128, out_dim=128
    """
    def build_layer(self, in_dim, out_dim, block_size, stride):
        layers = []
        block_in = in_dim
        block_strides = [stride] + [1] * (block_size - 1)

        # add multiple bottleneck blocks to form a layer
        for bs in block_strides: 
            layers.append(BottleneckV1(block_in, out_dim, bs))
            # after 1st bottleneck block, in_dim == out_dim!
            block_in = out_dim 
        return nn.Sequential(*layers)


    def forward(self, x):
        # basic conv layer
        x = self.basicconv(x) 

        # 4 layers of bottleneck blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # head
        x = self.avgpool(x) # x.shape = [batch, 512, 1, 1]
        x = x.reshape(x.shape[0], -1)    
        x = self.fc(x)
        return x


"""
- define different version of ResNet
"""
def ResNet18(classes=1000):
    return ResNet([2, 2, 2, 2], classes)

def ResNet34(classes=1000):
    return ResNet([3, 4, 6, 3], classes)




