## import packages
import torch
from  torch import nn
import torch.nn.functional as F


"""
- define bottleneck/residual block for DenseNet
- each bottleneck block consists of BN => ReLU => Conv1x1 => BN => ReLU => Conv3x3 
    - BN is in front of Conv!
- for each shortcut, input needs to be downsampled into the same number of
  out_dim so that CONCATENATE in channel dim can be performed
- inputs:
    - in_dim = in_dim of block
    - growthRate = out_dim of each block, default=32
"""

class DenseBottleneck(nn.Module):
    def __init__(self, in_dim, growthRate=32):
        
        # inherit from parent class
        super(DenseBottleneck, self).__init__()

        # interChannels = 4 *  growthRate
        interChannel = 4 * growthRate

        # create conv blocks, output has out_dim 
        # BN => ReLU => Conv1x1 => BN => ReLU => Conv3x3 
        self.bottleneck = nn.Sequential(
                nn.BatchNorm2d(in_dim),
                nn.ReLU(inplace=True),
                # Conv1x1, note stride may change as input
                nn.Conv2d(in_dim, interChannel, 1, bias=False),
                
                nn.BatchNorm2d(interChannel),
                nn.ReLU(inplace=True),
                # Conv3x3 
                nn.Conv2d(interChannel, growthRate, 3, padding=1, bias=False),
                )
        pass

    # define a forward method
    def forward(self, x):
        # record input 
        identity = x

        # pass through
        output = self.bottleneck(x)

        # Concatenate in dim of channel
        out = torch.cat([identity, output], 1)
        return out



