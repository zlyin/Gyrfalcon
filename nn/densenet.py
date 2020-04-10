## import packages
import torch
from torch import nn
import torch.nn.functional as F
from densenet_bottleneck import DenseBottleneck


"""
- define DenseBlock that includes 4 DenseBottleneck blocks
- for each DenseBottleneck:
    - shortcut = input 
    - after each DenseBottleneck, concat growthRate channels to input
- inputs:
    - in_dim = in_dim of this LAYER
    - growthRate = # of channels to concate after each DenseBottleneck
"""
class DenseBlock(nn.Module):
    
    def __init__(self, in_dim, growthRate, nBlocks):
        # inherit from parent class
        super(DenseBlock, self).__init__()

        # each layer contains 4 DenseBottleneck 
        layers = []
        interChannel = in_dim
        for i in range(int(nBlocks)):
            layers.append(DenseBottleneck(interChannel, growthRate=growthRate))
            # update in_dim for next DenseBottleneck
            interChannel += growthRate
            pass
        # store in a Sequential
        self.denseblock = nn.Sequential(*layers)
        pass

    def forward(self, x):
        return self.denseblock(x)


"""
- define a transition layer inlcuding BN => ReLU => Conv1x1 => AvgPool2x2
- 
"""
class Transition(nn.Module):
    
    def __init__(self, in_dim, out_dim):
        # inherit from parent class
        super(Transition, self).__init__()

        # add following layers to Sequential, BN => ReLU => Conv1x1 => AvgPool2x2
        self.transition = nn.Sequential(
                nn.BatchNorm2d(in_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_dim, out_dim, 1, bias=False),
                nn.AvgPool2d(kernel_size=2, stride=2),
                )
        pass

    def forward(self, x):
        return self.transition(x)


"""
- define DenseNet general class

"""
class DenseNet(nn.Module):
    
    def __init__(self, block_sizes=[6, 12, 24, 32], classes=1000, 
            num_init_channel=64, growthRate=32,):
        # inherit from parent class
        super(DenseNet, self).__init__()

        # BasicConv => Conv7x7 => MaxPool3x3 
        self.basicconv = nn.Sequential(
                nn.Conv2d(3, num_init_channel, 7, stride=2, padding=3,
                    bias=False),
                nn.BatchNorm2d(num_init_channel),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                )

        # 1st DenseBlock
        block_in_dim = num_init_channel
        self.block1 = DenseBlock(block_in_dim, growthRate, block_sizes[0])

        # 1st Transition
        block_in_dim += growthRate * block_sizes[0]
        self.trainsition1  = Transition(block_in_dim, int(block_in_dim / 2))  

        # 2nd DenseBlock
        block_in_dim = int(block_in_dim / 2)
        self.block2 = DenseBlock(block_in_dim, growthRate, block_sizes[1])

        # 2nd Transition
        block_in_dim += growthRate * block_sizes[1]
        self.trainsition2  = Transition(block_in_dim, int(block_in_dim / 2))  

        # 3rd DenseBlock
        block_in_dim = int(block_in_dim / 2)
        self.block3 = DenseBlock(block_in_dim, growthRate, block_sizes[2])

        # 3rd Transition
        block_in_dim += growthRate * block_sizes[2]
        self.trainsition3  = Transition(block_in_dim, int(block_in_dim / 2))  

        # 4th DenseBlock
        block_in_dim = int(block_in_dim / 2)
        self.block4 = DenseBlock(block_in_dim, growthRate, block_sizes[3])

        # head 
        head_in_dim = block_in_dim + growthRate * block_sizes[3]
        self.final_BN = nn.BatchNorm2d(head_in_dim) 
        self.final_ReLU = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Linear(head_in_dim, classes)
        pass

    
    def forward(self, x):
        # pass basic conv
        x = self.basicconv(x)

        # pass 4 DenseBlocks and 3 Transitions
        x = self.block1(x)
        x = self.trainsition1(x)
        x = self.block2(x)
        x = self.trainsition2(x)
        x = self.block3(x)
        x = self.trainsition3(x)
        x = self.block4(x)
        
        # head
        # need to add final BN => ReLU before FC
        x = self.final_BN(x)
        x = self.final_ReLU(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x
        


"""
- define different version of DenseNet
"""
def DenseNet121(classes=1000):
    return DenseNet(block_sizes=[6, 12, 24, 16], classes=classes, 
        num_init_channel=64, growthRate=32)

def DenseNet169(classes=1000):
    return DenseNet(block_sizes=[6, 12, 32, 32], classes=classes, 
        num_init_channel=64, growthRate=32)

def DenseNet201(classes=1000):
    return DenseNet(block_sizes=[6, 12, 48, 32], classes=classes, 
        num_init_channel=64, growthRate=32)

def DenseNet264(classes=1000):
    return DenseNet(block_sizes=[6, 12, 64, 48], classes=classes, 
        num_init_channel=64, growthRate=32)
