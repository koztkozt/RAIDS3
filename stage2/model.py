import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torchvision import models
import math

#  In the prototype of RAIDS, we build a classifier that is mainly composed of two linear layers.
class stage2(nn.Module):
    def __init__(self):
        super(stage2, self).__init__()
        self.fc1 = nn.Linear(in_features=2, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.batchnorm2 = nn.BatchNorm1d(64)

    def forward(self, x):
        out = self.fc1(x)  # Concatenates a sequence of tensors along a new dimension. torch.stack(inp)
        out = self.relu(out)
        out = self.batchnorm1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.batchnorm2(out)
        out = self.dropout(out)
        out = self.fc3(out)
        # out = F.log_softmax(out, dim=-1)
        return out


def weight_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
