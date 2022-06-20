import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torchvision import models
import math

# comma_large_dropout
class stage1(nn.Module):
    def __init__(self):
        super(stage1, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=24, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(in_features=25 * 25 * 64, out_features=500), nn.ReLU(inplace=True), nn.Dropout(0.5)
        )
        self.layer4 = nn.Sequential(
            nn.Linear(in_features=500, out_features=100), nn.ReLU(inplace=True), nn.Dropout(0.25)
        )
        self.layer5 = nn.Sequential(nn.Linear(in_features=100, out_features=20), nn.ReLU(inplace=True))
        self.layer6 = nn.Linear(in_features=20, out_features=2)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        return out


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2.0 / n))
