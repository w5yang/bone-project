import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['Inversion']


class Inversion(nn.Module):
    def __init__(
            self,
            channel: int = 1,
            num_classes: int = 10,
            truncation: int = 10,
            c: float = 50.,
            complexity: int = 64,
            **params
    ):
        super(Inversion, self).__init__()

        self.channel = channel
        self.ngf = complexity
        self.num_classes = num_classes
        self.truncation = truncation
        self.c = c

        self.decoder = nn.Sequential(
            # input is Z: num_classes x 1 x 1
            nn.ConvTranspose2d(self.num_classes, self.ngf * 4, 4, 1, 0),
            nn.BatchNorm2d(self.ngf * 4),
            nn.Tanh(),
            # state size. (ngf*4) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(self.ngf * 2),
            nn.Tanh(),
            # state size. (ngf*2) x 8 x 8
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1),
            nn.BatchNorm2d(self.ngf),
            nn.Tanh(),
            # state size. (ngf) x 16 x 16
            nn.ConvTranspose2d(self.ngf, self.channel, 4, 2, 1),
            nn.Sigmoid()
            # state size. channel x 32 x 32
        )

    def forward(self, x):
        topk, indices = torch.topk(x, self.truncation)
        topk = torch.clamp(torch.log(topk), min=-1000) + self.c
        topk_min = topk.min(1, keepdim=True)[0]
        topk = topk + F.relu(-topk_min)
        x = torch.zeros_like(x).scatter_(1, indices, topk)

        x = x.view(-1, self.num_classes, 1, 1)
        x = self.decoder(x)
        x = x.view(-1, self.channel, 32, 32)
        return x
