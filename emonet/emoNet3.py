import torch.nn as nn
import torch.nn.functional as F


class emoNet(nn.Module):
    network_config = [32, 32, 'M', 64, 64, 'M', 128, 128, 'M']

    def __init__(self, num_of_channels, num_of_classes):
        super(emoNet, self).__init__()
        self.features = self._make_layers(num_of_channels, self.network_config)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.Mish(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.Mish(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(64, num_of_classes)
        )

    def forward(self, x):
        out = self.features(x)
        out = self.classifier(out)
        return out

    def _make_layers(self, in_channels, cfg):
        layers = []
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.Mish(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)