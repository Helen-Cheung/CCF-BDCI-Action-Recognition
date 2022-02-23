import paddle
import paddle.nn as nn

from .base import BaseHead
from ..registry import HEADS
from ..weight_init import weight_init_



@HEADS.register()
class MSG3DHead(BaseHead):
    """
    Head for MSG3D model.
    Args:
        in_channels: int, input feature channels. Default: 256.
        num_classes: int, number classes. Default: 10.
    """
    def __init__(self, in_channels=384, num_classes=10, **kwargs):
        super().__init__(num_classes, in_channels, **kwargs)
        self.fc = nn.Linear(in_channels, num_classes)

    def init_weights(self):
        """Initiate the parameters.
        """
        for layer in self.sublayers():
            if isinstance(layer, nn.Linear):
                weight_init_(layer, 'Normal', std=0.02)

    def forward(self, x):
        """Define how the head is going to run.
        """
        x = self.fc(x)
        x = paddle.reshape_(x, (x.shape[0], -1))  # N,C,1,1 --> N,C

        return x