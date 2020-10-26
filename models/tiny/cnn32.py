import torch.nn as nn
from collections import OrderedDict

__all__ = ['cnn32', 'cnn42']


class CNN32(nn.Module):
    """A typical cnn model originated from ActiveThief.

    """

    def __init__(self, num_classes=43, channel=3, **kwargs):
        super(CNN32, self).__init__()
        self.cb1 = make_conv_block(channel, 32)
        self.cb2 = make_conv_block(32, 64)
        self.cb3 = make_conv_block(64, 128)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.cb1(x)
        x = self.cb2(x)
        x = self.cb3(x)
        x = x.view(-1, 2048)
        x = self.fc(x)

        return x


def make_conv_block(inplane: int, num_filter) -> nn.Module:
    block = nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(inplane, num_filter, 3, 1, 1)),
        ('relu1', nn.ReLU()),
        ('bn1', nn.BatchNorm2d(num_filter, eps=1e-3, momentum=0.99)),
        ('conv2', nn.Conv2d(num_filter, num_filter, 3, 1, 1)),
        ('relu2', nn.ReLU()),
        ('bn2', nn.BatchNorm2d(num_filter, eps=1e-3, momentum=0.99)),
        ('mp1', nn.MaxPool2d(2, 2)),
        ('do1', nn.Dropout2d(p=0.5)),
    ])
    )
    return block


def cnn32(num_classes, **kwargs):
    return CNN32(num_classes, **kwargs)

class CNN42(nn.Module):
    def __init__(self, num_classes=43, channel=3, **kwargs):
        super(CNN42, self).__init__()
        self.cb1 = make_conv_block(channel, 32)
        self.cb2 = make_conv_block(32, 64)
        self.cb3 = make_conv_block(64, 128)
        self.cb4 = make_conv_block(128, 256)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.cb1(x)
        x = self.cb2(x)
        x = self.cb3(x)
        x = self.cb4(x)
        x = x.view(-1, 1024)
        x = self.fc(x)

        return x

def cnn42(num_classes, **kwargs):
    return CNN42(num_classes, **kwargs)