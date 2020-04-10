# import packages
import torch
from torch import nn
import torch.nn.functional as F


"""
- define a basic inception class that includes a Conv2d => BN => ReLU
""" 
class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        # inherit from parent class
        super(BasicConv2d, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        pass

    # define forward method
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


"""
- define inceptionv3 class
- params = in_dim and channels_7x7
"""
class InceptionV3(nn.Module):

    def __init__(self, in_dim, channels_7x7):

        # inherit from parent class
        super(InceptionV3, self).__init__()
        
        # define 4 submodules
        c7 = channels_7x7

        # branch 1 = Conv(1x1)
        self.branch1 = BasicConv2d(in_dim, 192, 1)

        # brach 2 = Conv(1x1) => Conv(1x7) => Conv(7x1)
        self.branch2 = nn.Sequential(
                BasicConv2d(in_dim, c7, 1),
                BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3)),
                BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0)),
                )
        
        # branch 3 = Conv(1x1) => Conv(3x3) => Conv(3x3)
        self.branch3 = nn.Sequential(
                BasicConv2d(in_dim, c7, 1),
                BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0)),
                BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3)),
                BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0)),
                BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3)),
                )

        # branch 4 = AvgPool(3x3) = Conv(1x1)
        self.branch4 = nn.Sequential(
                nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
                BasicConv2d(in_dim, 192, 1),
                )
        pass


    # define forward method
    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        
        # concatenate outputs from 4 modules in channel dimension
        output = torch.cat([x1, x2, x3, x4], dim=1)
        return output

    pass



