import numpy as np
import os
import pandas as pd
import sys
import torch


def get_class_nums(dataset, split=1, csv_dir='./csv'):

    df = pd.read_csv(
        os.path.join(csv_dir, dataset, "train{}.csv").format(split))

    if dataset == '50salads':
        n_classes = 19
    elif dataset == 'gtea':
        n_classes = 11
    elif dataset == 'breakfast':
        n_classes = 48
    else:
        print("You have to select 50salads, gtea or breakfast as dataset.")
        sys.exit(1)

    nums = [0 for i in range(n_classes)]
    for i in range(len(df)):
        label_path = df.iloc[i]['label']
        label = np.load(label_path).astype(np.int64)
        num, cnt = np.unique(label, return_counts=True)
        for n, c in zip(num, cnt):
            nums[n] += c

    return nums


def get_class_weight(dataset, split=1, csv_dir='./csv'):
    """
    Class weight for CrossEntropy in the provided dataset by Softbank
    Class weight is calculated in the way described in:
        D. Eigen and R. Fergus, “Predicting depth, surface normals and semantic labels with a common multi-scale convolutional architecture,” in ICCV,
        openaccess: https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Eigen_Predicting_Depth_Surface_ICCV_2015_paper.pdf
    """

    nums = get_class_nums(dataset, split, csv_dir)

    class_num = torch.tensor(nums)
    total = class_num.sum().item()
    frequency = class_num.float() / total
    median = torch.median(frequency)
    class_weight = median / frequency

    return class_weight
