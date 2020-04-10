## import packages
import torch
from torch import nn
import numpy as np
from detnet_bottleneck import DilatedBottleneck
from resnet_bottleneck import BottleneckV2


"""
- Define DetNet general components and construct DetNet59
- use the first 4 stages of ResNet50
- stage 5 & 6 = Version B DilatedBottleneck => Version A DilatedBottleneck x2
- Head layer = AvgPool => FC => Softmax
"""

class DetNet(nn.Module):
    """
    - inputs:
    - block_sizes = [4 numbers] of bottleneck blocks 
    - classes = num of classes
    """
    def __init__(self, blocks_sizes=[3, 4, 6], classes=1000):
        # inherit from parant class
        super(DetNet, self).__init__()

        # Basic Conv layer = Conv7x7 => MaxPool3x3
        self.stage1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                ) 

        # 3 layers consist of multiple residual bottlenecks
        # note that layer_out_dim = 2 * layer_in_dim
        self.stage2 = self.build_layer(64, 256, blocks_sizes[0], stride=1)
        self.stage3 = self.build_layer(256, 512, blocks_sizes[1], stride=2)
        self.stage4 = self.build_layer(512, 1024, blocks_sizes[2], stride=2)
        
        # stage 5
        self.stage5 = nn.Sequential(
                DilatedBottleneck(1024, out_dim=256, stride=1, extra=True),
                DilatedBottleneck(256, out_dim=256, stride=1, extra=False),
                DilatedBottleneck(256, out_dim=256, stride=1, extra=False),
                )

        # stage 6
        self.stage6 = nn.Sequential(
                DilatedBottleneck(256, out_dim=256, stride=1, extra=True),
                DilatedBottleneck(256, out_dim=256, stride=1, extra=False),
                DilatedBottleneck(256, out_dim=256, stride=1, extra=False),
                )
                
        # Head layer = AdaptiveAvgPool => FC 
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(256, classes)
        pass


    """
    - inputs:
    - block_size = num of bottleneck blocks in this layer
    - in_dim, out_dim = input dim & out dim of THIS layer! not for the each
      block!
    - stride = stride for conv3x3/downsample in the FIRST bottleneck block of
      each layer!
    - eg. 1st layer has 3 blocks (layer_in_dim=64, layer_out_dim=256):
        - 1st block has 3 layers (block_in_dim=64, block_out_dim=256):
            - Conv1x1 => in_dim = 64, out_dim=64
            - Conv3x3 => in_dim = 64, out_dim=64
            - Conv1x1 => in_dim = 64, out_dim=256
        - 2nd block has 3 layers (block_in_dim=*256*, block_out_dim=256):
            - Conv1x1 => in_dim = *256*, out_dim=64
            - Conv3x3 => in_dim = 64, out_dim=64
            - Conv1x1 => in_dim = 64, out_dim=256
        - 3rd block has 3 layers (block_in_dim=*256*, block_out_dim=256):
            - Conv1x1 => in_dim = *256*, out_dim=64
            - Conv3x3 => in_dim = 64, out_dim=64
            - Conv1x1 => in_dim = 64, out_dim=256    
    """
    def build_layer(self, in_dim, out_dim, block_size, stride):
        layers = []
        block_ins = [in_dim] + [out_dim] * (block_size - 1)
        block_strides = [stride] + [1] * (block_size - 1)

        # add multiple bottleneck blocks to form a layer
        for i in range(block_size):
            bs = block_strides[i]
            b_in_dim = block_ins[i]
            layers.append(BottleneckV2(b_in_dim, out_dim, bs))
        return nn.Sequential(*layers)


    def forward(self, x):
        # stage1
        x = self.stage1(x)

        # 3 stages of residual bottlenecks 
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
    
        # 2 stages of dilated bottlenecks
        x = self.stage5(x)
        x = self.stage6(x)

        # head
        x = self.avgpool(x) # x.shape = [batch, 512, 1, 1]
        x = x.reshape(x.shape[0], -1)    
        x = self.fc(x)
        return x



