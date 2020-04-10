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
- define inceptionv1 class
- params = channels for 4 sub modules
"""
class InceptionV2(nn.Module):

    def __init__(self, in_dim=192, out_1_1=96, hid_2_1=48, out_2_3=64,
            hid_3_1=64, hid_3_2=96, out_3_3=96, out_4_1=64):

        # inherit from parent class
        super(InceptionV2, self).__init__()
        
        # define 4 submodules
        # branch 1 = Conv(1x1)
        self.branch1 = BasicConv2d(in_dim, out_1_1, 1)

        # brach 2 = Conv(1x1) => Conv(3x3)
        self.branch2 = nn.Sequential(
                BasicConv2d(in_dim, hid_2_1, 1),
                BasicConv2d(hid_2_1, out_2_3, 3, padding=1),
                )
        
        # branch 3 = Conv(1x1) => Conv(3x3) => Conv(3x3)
        self.branch3 = nn.Sequential(
                BasicConv2d(in_dim, hid_3_1, 1),
                BasicConv2d(hid_3_1, hid_3_2, 3, padding=1),
                BasicConv2d(hid_3_2, out_3_3, 3, padding=1),
                )
        # branch 4 = AvgPool(3x3) = Conv(1x1)
        self.branch4 = nn.Sequential(
                nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
                BasicConv2d(in_dim, out_4_1, 1),
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



