import torch
import torchvision
import os
import glob
import numpy as np
import math
import random
import torch.nn as nn

### U-net from source torch api. D B S 2024.

class myUnet(nn.module):
    def __init__(self):
        self.conv1 = nn.conv2D(3, 128, 3, 1)
        self.conv2 = nn.conv2D(128, 256)
        self.conv3 = nn.conv2D(256, 512)

        super(myUnet)


