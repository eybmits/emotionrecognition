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
        self.in_channels = 64  # Increased initial channels
        self.conv1 = nn.Conv2d(num_of_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)  # Changed to ReLU
        self.layer1 = self._make_layer(BasicBlock, 64, num_blocks=3, stride=1)  # Increased depth
        self.layer2 = self._make_layer(BasicBlock, 128, num_blocks=4, stride=2) # Increased depth
        self.layer3 = self._make_layer(BasicBlock, 256, num_blocks=6, stride=2) # Increased depth
        self.layer4 = self._make_layer(BasicBlock, 512, num_blocks=3, stride=2) # New layer added
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512 * BasicBlock.expansion, 256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(128, num_of_classes)
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
        out = self.layer4(out)  # New layer
        out = self.global_avg_pool(out)
        out = out.view(out.size(0), -1)  # Flatten the output
        out = self.classifier(out)
        return out
