import sys
import torch.nn as nn

from .tmse import TMSE


class ActionSegmentationLoss(nn.Module):
    """
        Loss Function for Action Segmentation
        You can choose the below loss functions and combine them.
            - Cross Entropy Loss (CE)
            - Temporal MSE (TMSE)
    """

    def __init__(
        self, ce=True, tmse=False, weight=None, threshold=4, ignore_index=255,
        ce_weight=1.0, tmse_weight=0.15,
    ):
        super().__init__()
        self.criterions = []
        self.weights = []

        if ce:
            self.criterions.append(
                nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index))
            self.weights.append(ce_weight)

        if tmse:
            self.criterions.append(
                TMSE(threshold=threshold, ignore_index=ignore_index))
            self.weights.append(tmse_weight)

        if len(self.criterions) == 0:
            print("You have to choose at least one loss function.")
            sys.exit(1)

    def forward(self, preds, gts, feats):
        loss = 0.
        for criterion, weight in zip(self.criterions, self.weights):
            loss += weight * criterion(preds, gts)

        return loss
