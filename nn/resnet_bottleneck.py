## import packages
import torch
from  torch import nn
import torch.nn.functional as F


"""
- define bottleneck/residual block for ResNet18/34 backbone
- each bottleneck block consists of Conv3x3 => Conv1x1 
- for each shortcut, input needs to be downsampled into the same number of
  out_dim so that ADDITION in channel dim can be performed
"""
class BottleneckV1(nn.Module):
    def __init__(self, in_dim, out_dim, stride=1):
        
        # inherit from parent class
        super(BottleneckV1, self).__init__()

        # create conv blocks, output has out_dim
        self.bottleneck = nn.Sequential(
                # Conv3x3, note stride may change as input
                nn.Conv2d(in_dim, out_dim, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True),
                # Conv3x3, note stride must be 1!
                nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_dim),
                )

        # downsample input to out_dim
        self.downsample = nn.Sequential()
        if stride != 1 or in_dim != out_dim:
            self.downsample = nn.Sequential(
                    nn.Conv2d(in_dim, out_dim, 1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_dim),
                    )

        # ReLU after concate
        self.relu = nn.ReLU(inplace=True)
        pass

    # define a forward method
    def forward(self, x):
        # record input & downsample
        identity = x
        identity = self.downsample(identity)

        # pass through
        output = self.bottleneck(x)

        # ADD elementwisely & activate
        output += identity
        return self.relu(output)


"""
- define bottleneck/residual block for ResNet50/101/152 backbone
- each bottleneck block consists of Conv1x1 => Conv3x3 => Conv1x1 
- for each shortcut, input needs to be downsampled into the same number of
  out_dim so that ADDITION in channel dim can be performed
"""
class BottleneckV2(nn.Module):
    def __init__(self, in_dim, out_dim, stride=1):
        
        # inherit from parent class
        super(BottleneckV2, self).__init__()

        block_out_dim = int(out_dim / 4)
        # create conv blocks, output has out_dim
        self.bottleneck = nn.Sequential(
                # Conv1x1
                nn.Conv2d(in_dim, block_out_dim, 1, bias=False),
                nn.BatchNorm2d(block_out_dim),
                nn.ReLU(inplace=True),
                # Conv3x3
                nn.Conv2d(block_out_dim, block_out_dim, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(block_out_dim),
                nn.ReLU(inplace=True),
                # Conv1x1
                nn.Conv2d(block_out_dim, out_dim, 1, bias=False),
                nn.BatchNorm2d(out_dim),
                )

        # downsample input to out_dim
        self.downsample = nn.Sequential()
        if stride != 1 or in_dim != out_dim:
            self.downsample = nn.Sequential(
                    nn.Conv2d(in_dim, out_dim, 1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_dim),
                    )       
        # ReLU after concate
        self.relu = nn.ReLU(inplace=True)
        pass

    # define a forward method
    def forward(self, x):
        # record input & downsample
        identity = x
        identity = self.downsample(identity)

        # pass through
        output = self.bottleneck(x)

        # ADD elementwisely & activate
        output += identity
        return self.relu(output)

