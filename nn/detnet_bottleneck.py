## import packages
import torch
from  torch import nn
import torch.nn.functional as F


"""
- define bottleneck/residual block for DetNet
- 2 versions of such DilatedBottleneck
    - both has structure of Conv1x1 => DilatedConv3x3 => Conv1x1
    - Version B has an extra Conv1x1 in the shortcut
- inputs:
    - in_dim = in_dim of this bottleneck block
    - out_dim =  out_dim of this bottleneck block, default=256
    - stride = stride of dilated kernel, default = 1
    - extra = add an extra Conv1x1 on the shortcut; False for version A, True
      for Version B
"""
class DilatedBottleneck(nn.Module):

    def __init__(self, in_dim, out_dim=256, stride=1, extra=False):
        
        # inherit from parent class
        super(DilatedBottleneck, self).__init__()

        # create conv blocks, output has out_dim
        self.bottleneck = nn.Sequential(
                # Conv1x1
                nn.Conv2d(in_dim, out_dim, 1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True),
                # dialated Conv3x3 to replace the old Conv3x3 with stride=2!
                # this kernel equivalences as kernel_size=5, padding=2!
                nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=2, dilation=2, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True),
                # Conv1x1
                nn.Conv2d(out_dim, out_dim, 1, bias=False),
                nn.BatchNorm2d(out_dim),
                )

        self.extra = extra
        # extra Conv1x1 on the shortcut
        if self.extra:
            self.extra_conv = nn.Sequential(
                    nn.Conv2d(in_dim, out_dim, 1, bias=False),
                    nn.BatchNorm2d(out_dim),
                    )

        # ReLU after concate
        self.relu = nn.ReLU(inplace=True)
        pass

    # define a forward method
    def forward(self, x):
        # record input 
        identity = x
        
        # if Version B, pass x through extra Conv1x1
        if self.extra:
            identity = self.extra_conv(identity)

        # pass through
        output = self.bottleneck(x)

        # ADD elementwisely & activate
        output += identity
        return self.relu(output)



