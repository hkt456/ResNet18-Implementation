import torch
import torch.nn as nn

from residual_block import Residual_Block
from layer_maker import _make_layer

class ResNet(nn.Module):
    def __init__(self,
                 block, 
                 layers,
                 num_classes: int = 10000):
        super().__init__()
        
        self.make_layer = _make_layer
        self.inplanes = 64

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(block, self.inplanes, 64, layers[0])
        self.layer2 = self.make_layer(block, self.inplanes, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, self.inplanes, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(block, self.inplanes, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)           # 224x224
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)         # 112x112

        x = self.layer1(x)          # 56x56
        x = self.layer2(x)          # 28x28
        x = self.layer3(x)          # 14x14
        x = self.layer4(x)          # 7x7

        x = self.avgpool(x)         # 1x1
        x = torch.flatten(x, 1)     # convert 1 X 1 to vector
        x = self.fc(x)

        return x

    def ResNet18():
        layer = [2,2,2,2]
        model = ResNet18(Residual_Block, layer)
        return model

    def ResNet34():
        layer = [3,4,6,3]
        model = ResNet18(Residual_Block, layer)
        return model

    
    



