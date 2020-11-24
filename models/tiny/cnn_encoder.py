import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Classifier"]


class Classifier(nn.Module):
    def __init__(
        self, channel: int = 1, num_classes: int = 10, complexity: int = 64, **kwargs
    ):
        super(Classifier, self).__init__()

        self.channel = channel
        self.ndf = complexity
        self.num_classes = num_classes

        self.encoder = nn.Sequential(
            # input is (channel) x 32 x 32
            nn.Conv2d(channel, self.ndf, 3, 1, 1),
            nn.BatchNorm2d(self.ndf),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(True),
            # state size. (ndf) x 16 x 16
            nn.Conv2d(self.ndf, self.ndf * 2, 3, 1, 1),
            nn.BatchNorm2d(self.ndf * 2),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(True),
            # state size. (ndf * 2) x 8 x 8
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 3, 1, 1),
            nn.BatchNorm2d(self.ndf * 4),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(True),
            # encoder output size. (ndf * 4) x 4 x 4
        )

        self.fc = nn.Sequential(
            nn.Linear(self.ndf * 4 * 4 * 4, self.num_classes * 5),
            nn.Dropout(0.5),
            nn.Linear(self.num_classes * 5, self.num_classes),
        )

    def forward(self, x, release=False):

        x = x.view(-1, self.channel, 32, 32)
        x = self.encoder(x)
        x = x.view(-1, self.ndf * 4 * 4 * 4)
        x = self.fc(x)

        if release:
            return F.softmax(x, dim=1)
        else:
            return F.log_softmax(x, dim=1)
