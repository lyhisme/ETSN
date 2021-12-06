import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class NormalizedReLU(nn.Module):
    """
    Normalized ReLU Activation prposed in the original TCN paper.
    the values are divided by the max computed per frame
    """

    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        x = F.relu(x)
        x /= x.max(dim=1, keepdim=True)[0] + self.eps

        return x


class EDTCN(nn.Module):
    """
    Encoder Decoder Temporal Convolutional Network
    """

    def __init__(self, in_channels, n_classes, kernel_size=25, mid_channels=[128, 160]):
        """
            Args:
                in_channels: int. the number of the channels of input feature
                n_classes: int. output classes
                kernel_size: int. 25 is proposed in the original paper
                mid_channels: list. the list of the number of the channels of the middle layer.
                            [96 + 32*1, 96 + 32*2] is proposed in the original paper
            Note that this implementation only supports n_layer=2
        """
        super().__init__()

        # encoder
        self.enc1 = nn.Conv1d(
            in_channels, mid_channels[0], kernel_size,
            stride=1, padding=(kernel_size - 1) // 2
        )
        self.dropout1 = nn.Dropout(0.3)
        self.relu1 = NormalizedReLU()

        self.enc2 = nn.Conv1d(
            mid_channels[0], mid_channels[1], kernel_size,
            stride=1, padding=(kernel_size - 1) // 2
        )
        self.dropout2 = nn.Dropout(0.3)
        self.relu2 = NormalizedReLU()

        # decoder
        self.dec1 = nn.Conv1d(
            mid_channels[1], mid_channels[1], kernel_size,
            stride=1, padding=(kernel_size - 1) // 2
        )
        self.dropout3 = nn.Dropout(0.3)
        self.relu3 = NormalizedReLU()

        self.dec2 = nn.Conv1d(
            mid_channels[1], mid_channels[0], kernel_size,
            stride=1, padding=(kernel_size - 1) // 2
        )
        self.dropout4 = nn.Dropout(0.3)
        self.relu4 = NormalizedReLU()

        self.conv_out = nn.Conv1d(mid_channels[0], n_classes, 1, bias=True)

        self.init_weight()

    def forward(self, x):
        # encoder 1
        x1 = self.relu1(self.dropout1(self.enc1(x)))
        t1 = x1.shape[2]
        x1 = F.max_pool1d(x1, 2)

        # encoder 2
        x2 = self.relu2(self.dropout2(self.enc2(x1)))
        t2 = x2.shape[2]
        x2 = F.max_pool1d(x2, 2)

        # decoder 1
        x3 = F.interpolate(x2, size=(t2, ), mode='nearest')
        x3 = self.relu3(self.dropout3(self.dec1(x3)))

        # decoder 2
        x4 = F.interpolate(x3, size=(t1, ), mode='nearest')
        x4 = self.relu4(self.dropout4(self.dec2(x4)))

        out = self.conv_out(x4)

        return out

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

