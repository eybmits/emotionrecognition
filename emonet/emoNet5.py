import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import transforms
from skimage.feature import local_binary_pattern
import numpy as np
import cv2
import os




class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ELU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

import torch.nn as nn
import torch.nn.functional as F


class emoNet(nn.Module):
    def __init__(self, feature_vector_size, num_of_classes):
        super(emoNet, self).__init__()
        self.in_channels = 32

        # New 1x1 Convolution layer to adapt the 1D feature vector for 2D convolutions
        self.adapt_conv = nn.Conv2d(1, self.in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ELU(inplace=True)

        # Existing layers
        self.layer1 = self._make_layer(BasicBlock, 32, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 64, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 128, num_blocks=2, stride=2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier remains unchanged
        self.classifier = nn.Sequential(
            nn.Linear(128 * BasicBlock.expansion, 64),
            nn.ELU(True),
            nn.Dropout(0.5),
            nn.Linear(64, num_of_classes)
        )

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # Reshape and adapt the 1D feature vector for 2D convolutions
        x = x.view(x.size(0), 1, -1, 1)  # Reshape [batch_size, 1, feature_vector_size, 1]
        x = self.relu(self.bn1(self.adapt_conv(x)))

        # Forward pass through the rest of the layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Example of using the model
# model = emoNet(num_of_channels=1, num_of_classes=6)


"""
  def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.global_avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(x)
        return out"""

# -> Data 
#-> "preprocessing" (qualität, face-alignment, cropping, augmentation)
# fer2013 train test(80%!) Uni -> vald (40!!%) 
# -> 18.01 Latex
 



#Preliminary Report: Write about your approach, potential architectures and your
#initial findings; 2 pages, double column; we will provide you with a latex template.
#Deadline: 18.01.



#11.01 -> Final Report: Write about your approach, potential architectures and your

"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ELU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class emoNet(nn.Module):
    def __init__(self, num_of_channels, num_of_classes):
        super(emoNet, self).__init__()
        self.in_channels = 32
        self.conv1 = nn.Conv2d(num_of_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ELU(inplace=True)
        self.layer1 = self._make_layer(BasicBlock, 32, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 64, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 128, num_blocks=2, stride=2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(128 * BasicBlock.expansion, 64),
            nn.ELU(True),
            nn.Dropout(0.5),
            nn.Linear(64, num_of_classes)
        )

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.global_avg_pool(out)
        out = out.view(out.size(0), -1)  # Flatten the output
        out = self.classifier(out)       # Use the flattened output here
        return out"""
