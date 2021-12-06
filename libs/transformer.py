import random
import torch


class TempDownSamp(object):
    def __init__(self, downsamp_rate=1):
        super().__init__()
        self.downsamp_rate = downsamp_rate

    def __call__(self, input):
        feature, label = input[0], input[1]

        feature = feature[:, ::self.downsamp_rate]
        label = label[::self.downsamp_rate]

        return [feature, label]


class ToTensor(object):
    def __call__(self, input):
        feature, label = input[0], input[1]

        # from numpy to tensor
        feature = torch.from_numpy(feature).float()
        label = torch.from_numpy(label).long()

        # arrange feature and label in the temporal duration
        if feature.shape[1] > label.shape[0]:
            feature = feature[:, :label.shape[0]]
        else:
            label = label[:feature.shape[1]]

        return [feature, label]
