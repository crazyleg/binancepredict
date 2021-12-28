import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from .resnet1d import ResNet1D


class ResNetBSM4(nn.Module):
    def __init__(
        self,
        n_features,
        n_outputs,
        filters=64,
        blocks=6,
    ):
        super(ResNetBSM4, self).__init__()

        self.net1 = ResNet1D(n_features, filters, 3, 1, 1, blocks, n_outputs)
        # self.net2 = ResNet1D(98, 64, 3, 1 , 1, 6, 14)
        self.flat = nn.Flatten()
        self.fcn = nn.Linear(n_outputs, n_outputs)

    def forward(self, x):

        # x_fft = torch.fft.fft(x).real
        x = torch.cat([x, x], dim=2)
        x_res1 = self.net1(x)
        # x_res2 = self.net2(x_fft)
        # resnet1d seems to be overfitting (at least default version), but I still have hopes for FFT
        # fust for fft it seems like i would need to capture ... maybe same?

        result = self.flat(x_res1)

        return self.fcn(result)
