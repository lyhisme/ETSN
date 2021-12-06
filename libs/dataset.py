import numpy as np
import sys
import os
import pandas as pd
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset


class ActionSegmentationDataset(Dataset):
    """ Action Segmentation Dataset (50salads, gtea, breakfast) """

    def __init__(
            self, dataset, transform=None, mode='training',
            split=1, dataset_dir='./dataset', csv_dir='./csv'
    ):
        super().__init__()
        """
            Args:
                dataset: the name of dataset (50salads, gtea, breakfast)
                transform: torchvision.transforms.Compose([...])
                mode: training, validation, test
                split: which split of train, val and test do you want to use in csv files.(default:1)
                csv_dir: the path to the directory where the csv files are saved
        """

        if (dataset != '50salads') and (dataset != 'gtea') and (dataset != 'breakfast'):
            print("You have to choose 50saladas, gtea, breakfast as dataset.")
            sys.exit(1)

        if mode == 'training':
            self.df = pd.read_csv(
                os.path.join(csv_dir, dataset, 'train{}.csv'.format(split))
            )
        elif mode == 'validation':
            self.df = pd.read_csv(
                os.path.join(csv_dir, dataset, 'val{}.csv'.format(split))
            )
        elif mode == 'trainval':
            df1 = pd.read_csv(
                os.path.join(csv_dir, dataset, 'train{}.csv'.format(split))
            )
            df2 = pd.read_csv(
                os.path.join(csv_dir, dataset, 'val{}.csv'.format(split))
            )
            self.df = pd.concat([df1, df2])
        elif mode == 'test':
            self.df = pd.read_csv(
                os.path.join(csv_dir, dataset, 'test{}.csv'.format(split))
            )
        else:
            print('You have to choose training or validation as the dataset mode')
            sys.exit(1)

        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        feature_path = self.df.iloc[idx]['feature']

        label_path = self.df.iloc[idx]['label']

        feature = np.load(feature_path).astype(np.float32)
        label = np.load(label_path).astype(np.int64)

        if self.transform is not None:
            feature, label = self.transform([feature, label])

        sample = {
            'feature': feature,
            'label': label,
            'feature_path': feature_path
        }

        return sample


def collate_fn(sample):
    max_length = max([s['feature'].shape[1] for s in sample])

    feat_list = []
    label_list = []
    path_list = []

    for s in sample:
        feature = s['feature']
        label = s['label']
        feature_path = s['feature_path']

        _, t = feature.shape
        pad_t = max_length - t

        if pad_t > 0:
            feature = F.pad(
                feature, (0, pad_t), mode='constant', value=0.)
            label = F.pad(label, (0, pad_t), mode='constant', value=255)

        feat_list.append(feature)
        label_list.append(label)
        path_list.append(feature_path)

    # merge features from tuple of 2D tensor to 3D tensor
    features = torch.stack(feat_list, dim=0)
    # merge labels from tuple of 1D tensor to 2D tensor
    labels = torch.stack(label_list, dim=0)

    return {'feature': features, 'label': labels, 'feature_path': path_list}
