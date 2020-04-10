## import packages
import os
import sys
import torch
from torch import nn
import torch.nn.functional as F


## define VGG class
class VGG16(nn.Module):
    # define __init__()
    def __init__(self, num_classes=1000):
        # inherit from parent class in MRO
        super(VGG16, self).__init__()
        
        # note: parent class may not have the following properties
        layers = []
        in_dim = 3
        out_dim = 64
    
        # build network here, 13 conv layers in total
        for i in range(13):
            layers += [
                    nn.Conv2d(in_dim, out_dim, 3, 1, 1),
                    nn.ReLU(inplace=True),
                    ]
            in_dim = out_dim
            
            # add MaxPool2d after 2, 4, 7, 10, 13 layer
            # before 10th conv, output dim x2
            if i in [1, 3, 6, 9, 12]:
                layers.append(nn.MaxPool2d(2, 2))
                
                if i != 9:
                    out_dim *= 2
                pass
        # backbone 
        self.features = nn.Sequential(*layers)     # pass multiple params => *()

        # create head that has 3 FC layers
        self.classifier = nn.Sequential(
                # 1st
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                # 2nd
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                # 3rd
                nn.Linear(4096, num_classes), # softmax requires an input!
                )
        pass

    # define forward method
    def forward(self, x):
        x = self.features(x)
        # flatten!
        x = x.reshape(x.size(0), -1)
        #x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


