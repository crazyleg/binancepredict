import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from .resnet1d import ResNet1D


class BSM(nn.Module):
    def __init__(self, n_coins=100, n_timesteps=129):
        ## TODO: Add dropout
        super(BSM, self).__init__()

        # self.net1 = ResNet1D(98, 64, 3, 1 , 1, 6, 14)
        # self.net2 = ResNet1D(98, 64, 3, 1 , 1, 6, 14)

        self.conv1 = nn.Conv1d(98, 512, 1)
        self.conv2 = nn.Conv1d(512, 512, 1)
        self.conv3 = nn.Conv1d(512, 14, 1)

        self.relu = nn.LeakyReLU()

        self.conv1t = nn.Conv1d(98, 512, 3)
        self.conv2t = nn.Conv1d(512, 512, 3)
        self.conv3t = nn.Conv1d(512, 512, 3)
        self.conv4t = nn.Conv1d(512, 14, 4)

        self.fcn = nn.Linear(28, 14)

        self.flat = nn.Flatten()

    def forward(self, x):

        # x_fft = torch.fft.fft(x).real
        # x_res1 = self.net1(x)
        # x_res2 = self.net2(x_fft)
        # resnet1d seems to be overfitting (at least default version), but I still have hopes for FFT
        # fust for fft it seems like i would need to capture ... maybe same?

        x_last = x[:, :, -1].unsqueeze(2)
        x_last = self.relu(self.conv1(x_last))
        x_last = self.relu(self.conv2(x_last))
        x_last = self.conv3(x_last)

        # x_temporal_collection_fft = x_fft[:,:,[-1,-2,-4,-8,-16,-32,-64,-128,-256,-512]]
        # x_temporal_collection_fft = self.relu(self.conv1t(x_temporal_collection_fft))
        # x_temporal_collection_fft = self.relu(self.conv2t(x_temporal_collection_fft))
        # x_temporal_collection_fft = self.relu(self.conv3t(x_temporal_collection_fft))
        # x_temporal_collection_fft = self.conv4t(x_temporal_collection_fft)

        x_temporal_collection = x[
            :, :, [-1, -2, -4, -8, -16, -32, -64, -128, -256, -512]
        ]
        x_temporal_collection = self.relu(self.conv1t(x_temporal_collection))
        x_temporal_collection = self.relu(self.conv2t(x_temporal_collection))
        x_temporal_collection = self.relu(self.conv3t(x_temporal_collection))
        x_temporal_collection = self.conv4t(x_temporal_collection)

        result = self.relu(torch.cat([x_last, x_temporal_collection], dim=2))
        result = self.flat(result)

        return self.fcn(result)


class BSM2(nn.Module):
    def __init__(self, n_features, n_outputs):
        ## TODO: Add dropout
        super(BSM2, self).__init__()

        # self.net1 = ResNet1D(98, 64, 3, 1 , 1, 6, 14)
        # self.net2 = ResNet1D(98, 64, 3, 1 , 1, 6, 14)

        self.conv1 = nn.Conv1d(n_features, 512, 1)
        self.conv2 = nn.Conv1d(512, 512, 1)
        self.conv3 = nn.Conv1d(512, n_outputs, 1)

        self.relu = nn.LeakyReLU()

        self.conv1t = nn.Conv1d(n_features, 512, 3)
        self.conv2t = nn.Conv1d(512, 512, 3)
        self.conv3t = nn.Conv1d(512, 512, 3)
        self.conv4t = nn.Conv1d(512, n_outputs, 4)

        self.conv1tt = nn.Conv1d(n_features, 512, 3)
        self.conv2tt = nn.Conv1d(512, 512, 3)
        self.conv3tt = nn.Conv1d(512, 512, 3)
        self.conv4tt = nn.Conv1d(512, n_outputs, 4)

        self.conv1ttt = nn.Conv1d(n_features, 512, 3)
        self.conv2ttt = nn.Conv1d(512, 512, 3)
        self.conv3ttt = nn.Conv1d(512, 512, 3)
        self.conv4ttt = nn.Conv1d(512, n_outputs, 4)

        self.fcn = nn.Linear(n_outputs * 4, n_outputs)
        self.drp = nn.Dropout(0.5)

        self.flat = nn.Flatten()

    def forward(self, x):

        # x_fft = torch.fft.fft(x).real
        # x_res1 = self.net1(x)
        # x_res2 = self.net2(x_fft)
        # resnet1d seems to be overfitting (at least default version), but I still have hopes for FFT
        # fust for fft it seems like i would need to capture ... maybe same?

        x_last = x[:, :, -1].unsqueeze(2)
        x_last = self.relu(self.conv1(x_last))
        x_last = self.relu(self.conv2(x_last))
        x_last = self.conv3(x_last)

        x_temporal_collection = x[
            :, :, [-1, -2, -4, -8, -16, -32, -64, -128, -256, -512]
        ]
        x_temporal_collection = self.relu(self.conv1t(x_temporal_collection))
        x_temporal_collection = self.relu(self.conv2t(x_temporal_collection))
        x_temporal_collection = self.relu(self.conv3t(x_temporal_collection))
        x_temporal_collection = self.conv4t(x_temporal_collection)

        x_temporal_collection2 = x[:, :, [-1, -2, -3, -4, -6, -8, -10, -12, -24, -32]]
        x_temporal_collection2 = self.relu(self.conv1tt(x_temporal_collection2))
        x_temporal_collection2 = self.relu(self.conv2tt(x_temporal_collection2))
        x_temporal_collection2 = self.relu(self.conv3tt(x_temporal_collection2))
        x_temporal_collection2 = self.conv4tt(x_temporal_collection2)

        x_temporal_collection3 = x[
            :, :, [-8, -16, -24, -32, -40, -48, -56, -64, -72, 94]
        ]
        x_temporal_collection3 = self.relu(self.conv1ttt(x_temporal_collection3))
        x_temporal_collection3 = self.relu(self.conv2ttt(x_temporal_collection3))
        x_temporal_collection3 = self.relu(self.conv3ttt(x_temporal_collection3))
        x_temporal_collection3 = self.conv4ttt(x_temporal_collection3)

        result = self.relu(
            torch.cat(
                [
                    x_last,
                    x_temporal_collection,
                    x_temporal_collection2,
                    x_temporal_collection3,
                ],
                dim=2,
            )
        )
        result = self.flat(result)

        return self.fcn(result)


class BSM3(nn.Module):
    def __init__(self, n_features, n_outputs):
        ## TODO: Add dropout
        super(BSM3, self).__init__()

        self.net1 = ResNet1D(n_features, 32, 3, 1, 1, 4, n_outputs)
        # self.net2 = ResNet1D(98, 64, 3, 1 , 1, 6, 14)
        self.flat = nn.Flatten()
        self.fcn = nn.Linear(n_outputs, n_outputs)

    def forward(self, x):

        # x_fft = torch.fft.fft(x).real
        x_res1 = self.net1(x)
        # x_res2 = self.net2(x_fft)
        # resnet1d seems to be overfitting (at least default version), but I still have hopes for FFT
        # fust for fft it seems like i would need to capture ... maybe same?

        result = self.flat(x_res1)

        return self.fcn(result)


class BSM4(nn.Module):
    def __init__(self, n_features, n_outputs):
        ## TODO: Add dropout
        super(BSM4, self).__init__()

        self.net1 = ResNet1D(n_features, 32, 3, 1, 1, 4, n_outputs)
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
