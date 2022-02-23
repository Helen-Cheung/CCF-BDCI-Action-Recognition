import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .utils import activation_factory 
from ..registry import BACKBONES


class MLP(nn.Layer):
    def __init__(self, in_channels, out_channels, activation='relu', dropout=0):
        super().__init__()
        channels = [in_channels] + out_channels
        self.layers = nn.LayerList()
        for i in range(1, len(channels)):
            if dropout > 0.001:
                self.layers.append(nn.Dropout(p=dropout))
            self.layers.append(nn.Conv2D(channels[i-1], channels[i], kernel_size=1))
            self.layers.append(nn.BatchNorm2D(channels[i]))
            self.layers.append(activation_factory(activation))

    def forward(self, x):
        # Input shape: (N,C,T,V)
        for layer in self.layers:
            x = layer(x)
        return x