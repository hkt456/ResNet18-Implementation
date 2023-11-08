import torch
import torch.nn as nn

from residual_block import Residual_Block

class ResNet18(nn.Module):
    def __init__(self,
                 block, 
                 layers,
                 num_classes: int = 10000):
        super().__init__()

        self.inplanes = 64

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


