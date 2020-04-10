# import packages
import torch
from torch import nn
import torch.nn.functional as F

"""
- define a basic inception class that includes a Conv2d => ReLU 
""" 
class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        # inherit from parent class
        super(BasicConv2d, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                padding=padding)
        pass

    # define forward method
    def forward(self, x):
        x = self.conv(x)
        x = nn.ReLU(inplace=True)(x)
        return x


"""
- define inceptionv1 class
- params = channels for 4 sub modules
"""
class InceptionV1(nn.Module):

    def __init__(self, in_dim, hid_1_1, hid_2_1, hid_2_3, hid_3_1, out_3_5,
            out_4_1):
        # inherit from parent class
        super(InceptionV1, self).__init__()
        
        # define 4 submodules
        self.branch1x1 = BasicConv2d(in_dim, hid_1_1, 1)

        self.branch3x3 = nn.Sequential(
                BasicConv2d(in_dim, hid_2_1, 1),
                BasicConv2d(hid_2_1, hid_2_3, 3, padding=1),
                )

        self.branch5x5 = nn.Sequential(
                BasicConv2d(in_dim, hid_3_1, 1),
                BasicConv2d(hid_3_1, out_3_5, 5, padding=2),
                )

        self.branchPool = nn.Sequential(
                nn.MaxPool2d(3, stride=1, padding=1),
                BasicConv2d(in_dim, out_4_1, 1),
                )
        pass


    # define forward method
    def forward(self, x):
        x1 = self.branch1x1(x)
        x2 = self.branch3x3(x)
        x3 = self.branch5x5(x)
        x4 = self.branchPool(x)
        
        # concatenate outputs from 4 modules in channel dimension
        output = torch.cat([x1, x2, x3, x4], dim=1)
        return output

    pass



