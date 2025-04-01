import torch
import torch.nn as nn
import torch.nn.functional as F
from baselines.util.util import sample_and_group 


class Pct_feature(nn.Module):
    def __init__(self, output_channels=40):
        super(Pct_feature, self).__init__()
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, feature):
        x = self.linear3(feature)

        return x
