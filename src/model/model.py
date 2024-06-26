import torch
import torchvision
import os
import glob
import numpy as np
import math
import random
import torch.nn as nn

### U-net from source torch api. D B S 2024.

def UNet_Block(in_channel, out_channel, pool_k=2, pool_stride=2, pool_pad=0,ceiling=False):
    block = nn.Sequential(nn.conv2D(in_channel, out_channel, 3, 1, padding=1),
                          nn.BatchNorm2d(out_channel),
                          nn.ReLU(),
                          nn.conv2D(out_channel, out_channel, 3, 1, padding=1),
                          nn.BatchNorm2d(out_channel),
                          nn.MaxPool2d(pool_k, pool_stride, pool_pad, ceil_mode=ceiling),
                          nn.ReLU(),
                          )
    
    return block

def Dilated_Block(in_channel, out_channel, kernel_size=3, dilation=1, stride=1, padding=0):
    block = nn.Sequential(nn.conv2D(in_channel, out_channel, kernel_size=kernel_size, 
                                    stride=stride, padding=padding, dilation=dilation),
                          nn.BatchNorm2d(out_channel),
                          nn.ReLU(),
                         )
                         

class myUnet(nn.module): # assume input size 300x300
    def __init__(self):
        super(myUnet, self).__init__()
        self.block1 = UNet_Block(3, 64) # output size 150x150
        self.block2 = UNet_Block(64, 128) # output size 75x75
        self.block3 = UNet_Block(128, 256, ceiling=True) # output size is 38x38
        self.block4 = UNet_Block(256, 512) # output size is 19x19
        self.block5 = UNet_Block(512, 512, pool_k=3, pool_stride=1, pool_pad=1) # output size is 19x19
        self.block6 = Dilated_Block(512, 1024, dilation=6, padding=6)
        self.block7 = nn.Sequential(nn.conv2D(1024, 1024, 1, 1),
                                    nn.BatchNorm2d(1024),
                                    nn.ReLU(),
                                    )
        
    def forward(self, x):
        y = self.block1(x)
        y = self.block2(y)
        y = self.block3(y)
        y = self.block4(y)
        y = self.block5(y)
        y = self.block6(y)
        y = self.block7(y)

        return y

        


