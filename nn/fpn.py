## import packages
import torch
from torch import nn
import torch.nn.functional as F
from resnet_bottleneck import BottleneckV1, BottleneckV2


"""
- Define a Feature Pyramid Network class that has 4 componets:
    - Bottom-up network = use ResNet50
    - Up-down network = Upsampling
    - Lateral network = Conv1x1 & elelentwise addition
    - Feature merge = Conv3x3
- inputs:
    - layer_sizes = a list of # of blocks in 4 layers of resnet
    - version = resnet18/34/50/101/152
"""
class FPN(nn.Module):
    
    def __init__(self, layer_sizes=None, version="resnet50"):
        # inherit from paraent class
        super(FPN, self).__init__()

        # determin which bottleneck block to use
        self.version_to_sizes = {
                "resnet18" : ([2, 2, 2, 2], "v1"),
                "resnet34" : ([3, 4, 6, 3], "v1"),
                "resnet50" : ([3, 4, 6, 3], "v2"),
                "resnet101" : ([3, 4, 23, 3], "v2"),
                "resnet152" : ([3, 8, 36, 3], "v2"),
                }

        if layer_sizes is None:
            block_sizes, bn_version = self.version_to_sizes[version]
        else:
            block_sizes = layer_sizes
            bn_version = "v2"
       
        # determine in_dims & out_dims of each layer
        layer_in_dims = [64, 64, 128, 256] if bn_version == "v1" else [64, 256, 512, 1024]
        layer_out_dims = [64, 128, 256, 512] if bn_version == "v1" else [256, 512, 1024, 2048]

        # build C1 layer = Conv7x7 => MaxPool3x3
        self.C1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64), 
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                )

        # Bottom-up network, C2 => C3 => C4 => C5
        self.C2 = self.build_layer(layer_in_dims[0], layer_out_dims[0],
                block_sizes[0], stride=1, version=bn_version)

        self.C3 = self.build_layer(layer_in_dims[1], layer_out_dims[1],
                block_sizes[1], stride=2, version=bn_version)

        self.C4 = self.build_layer(layer_in_dims[2], layer_out_dims[2],
                block_sizes[2], stride=2, version=bn_version)

        self.C5 = self.build_layer(layer_in_dims[3], layer_out_dims[3],
                block_sizes[3], stride=2, version=bn_version)

        # laterals layers for C5/4/3/2 in the Up-down network, 
        # C5 --Conv1x1 to 256 channels & elementwise addition--> P5, 
        self.L5 = nn.Conv2d(layer_out_dims[-1], 256, 1, stride=1)
        self.L4 = nn.Conv2d(layer_out_dims[-2], 256, 1, stride=1)
        self.L3 = nn.Conv2d(layer_out_dims[-3], 256, 1, stride=1)
        self.L2 = nn.Conv2d(layer_out_dims[-4], 256, 1, stride=1)

        # Feature Merge of C4-L4, C3-L3, C2-L2 with Conv3x3
        self.merge4 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.merge3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.merge2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        pass
    

    """
    - forward method
    """
    def forward(self, x):
        # Bottom-up network
        c1 = self.C1(x)
        c2 = self.C2(c1)
        c3 = self.C3(c2)
        c4 = self.C4(c3)
        c5 = self.C5(c4)

        # Top-down network
        p5 = self.L5(c5)        # no need to upsample
        p4 = self.upsample(p5, self.L4(c4))
        p3 = self.upsample(p4, self.L3(c3))
        p2 = self.upsample(p3, self.L2(c2))

        # merge features
        p5 = p5
        p4 = self.merge4(p4)
        p3 = self.merge3(p3)
        p2 = self.merge2(p2)

        # return merged features in Bottom-up/(H, W) decreaseing order
        return p2, p3, p4, p5


    """
    - build each layer of bottleneck blocks
    - inputs:
        - in_dim, out_dim = layer_input, layer_output dimensions
        - block_size = # of blocks in the layer
        - stride = stride to use in the first block
        - version = use BottleneckV1 or BottleneckV2 to build layer
    """
    def build_layer(self, in_dim, out_dim, block_size, stride, version):
        layers = []
        block_ins = [in_dim] + [out_dim] * (block_size - 1)
        block_strides = [stride] + [1] * (block_size - 1)

        # add multiple bottleneck blocks to form a layer
        for i in range(block_size):
            bs = block_strides[i]
            b_in_dim = block_ins[i]
            if version == "v2":
                layers.append(BottleneckV2(b_in_dim, out_dim, bs))
            else:
                layers.append(BottleneckV1(b_in_dim, out_dim, bs))
            pass
        return nn.Sequential(*layers)


    """
    - upsample feature map in the Up-down process; 
    - elementwisely add upsmapled features to target features
    - inputs:
        - feature = upsampled features
        - target = features with target H & W
    """
    def upsample(self, feature, target):
        _, _, H, W = target.shape
        return F.interpolate(feature, size=(H, W), mode="nearest") + target














