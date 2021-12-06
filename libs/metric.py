# Originally written by cthincs1
# https://github.com/cthincsl/TemporalConvthutionalNetworks/blob/master/code/metrics.py

import numpy as np


def get_segments(frame_wise_label, id2class_map, bg_class="background"):
    """
        Args:
            frame-wise label: frame-wise prediction or ground truth. 1D numpy array
        Return:
            segment-label array: list (excluding background class)
            start index list
            end index list
    """

    labels = []
    starts = []
    ends = []

    frame_wise_label = [
        id2class_map[frame_wise_label[i]] for i in range(len(frame_wise_label))]

    # get class, start index and end index of segments
    # background class is excluded
    last_label = frame_wise_label[0]
    if frame_wise_label[0] != bg_class:
        labels.append(frame_wise_label[0])
        starts.append(0)

    for i in range(len(frame_wise_label)):
        # if action labels change
        if frame_wise_label[i] != last_label:
            # if label change from one class to another class
            # it's an action starting point
            if frame_wise_label[i] != bg_class:
                labels.append(frame_wise_label[i])
                starts.append(i)

            # if label change from background to a class
            # it's not an action end point.
            if last_label != bg_class:
                ends.append(i)

            # update last label
            last_label = frame_wise_label[i]

    if last_label != bg_class:
        ends.append(i)

    return labels, starts, ends


def levenshtein(pred, gt, norm=True):
    """
        Levenshtein distance(Edit Distance)
        Args:
            pred: segments list
            gt: segments list
        Return:
            if norm == True:
                (1 - average_edit_distance) * 100
            else:
                edit distance
    """

    n, m = len(pred), len(gt)

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if pred[i - 1] == gt[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,         # insertion
                dp[i][j - 1] + 1,         # deletion
                dp[i - 1][j - 1] + cost)  # replacement

    if norm:
        score = (1 - dp[n][m] / max(n, m)) * 100
    else:
        score = dp[n][m]

    return score


def get_f1_score(p_label, p_start, p_end, g_label, g_start, g_end, threshold, bg_class=["background"]):
    """
        Args:
            p_label, p_start, p_end: return values of get_segments(pred)
            g_label, g_start, g_end: return values of get_segments(gt)
            threshold: threshold (0.1, 0.25, 0.5)
            bg_class: background class
        Return:
            tp: true positive
            fp: false positve
            fn: false negative
    """

    tp = 0
    fp = 0
    hits = np.zeros(len(g_label))

    for j in range(len(p_label)):
        intersection = np.minimum(
            p_end[j], g_end) - np.maximum(p_start[j], g_start)
        union = np.maximum(p_end[j], g_end) - np.minimum(p_start[j], g_start)
        IoU = (1.0 * intersection / union) * \
            ([p_label[j] == g_label[x] for x in range(len(g_label))])
        # Get the best scoring segment
        idx = np.array(IoU).argmax()

        if IoU[idx] >= threshold and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1

    fn = len(g_label) - sum(hits)

    return float(tp), float(fp), float(fn)


class ScoreMeter(object):
    def __init__(self, id2class_map, thresholds=[0.1, 0.25, 0.5], ignore_index=255):
        self.thresholds = thresholds    # threshold for f score
        self.ignore_index = ignore_index
        self.id2class_map = id2class_map
        self.edit_score = 0
        self.tp = [0 for _ in range(len(thresholds))]    # true positive
        self.fp = [0 for _ in range(len(thresholds))]    # false positive
        self.fn = [0 for _ in range(len(thresholds))]    # false negative
        self.n_correct = 0
        self.n_frames = 0
        self.n_videos = 0
        self.n_classes = len(self.id2class_map)
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

    def _fast_hist(self, pred, gt):
        mask = (gt >= 0) & (gt < self.n_classes)
        hist = np.bincount(
            self.n_classes * gt[mask].astype(int) + pred[mask], minlength=self.n_classes ** 2
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def update(self, pred, gt):
        """
            Args:
                pred, gt: shape => 1D array
            only evaluation of single video is supported.
        """
        pred = pred[gt != self.ignore_index]
        gt = gt[gt != self.ignore_index]

        for lt, lp in zip(pred, gt):
            self.confusion_matrix += \
                self._fast_hist(lt.flatten(), lp.flatten())

        self.n_videos += 1
        # count the correct frame
        self.n_frames += len(pred)
        for i in range(len(pred)):
            if pred[i] == gt[i]:
                self.n_correct += 1

        # calculate the edit distance
        p_label, p_start, p_end = get_segments(pred, self.id2class_map)
        g_label, g_start, g_end = get_segments(gt, self.id2class_map)

        self.edit_score += levenshtein(p_label, g_label, norm=True)

        for i, th in enumerate(self.thresholds):
            tp, fp, fn = get_f1_score(
                p_label, p_start, p_end, g_label, g_start, g_end, th)
            self.tp[i] += tp
            self.fp[i] += fp
            self.fn[i] += fn

    def get_scores(self):
        """
            Return:
                Accuracy
                Normlized Edit Distance
                F1 Score of Each Threshold
        """

        # accuracy
        acc = 100 * float(self.n_correct) / self.n_frames

        # edit distance
        edit_score = float(self.edit_score) / self.n_videos

        # F1 Score
        f1s = []
        for i in range(len(self.thresholds)):
            if float(self.tp[i] + self.fp[i]) == 0:
                precision = 0
            else:    
                precision = self.tp[i] / float(self.tp[i] + self.fp[i])
            recall = self.tp[i] / float(self.tp[i] + self.fn[i])

            f1 = 2.0 * (precision * recall) / (precision + recall + 1e-7)
            f1 = np.nan_to_num(f1) * 100

            f1s.append(f1)

        # Accuracy, Edit Distance, F1 Score
        return acc, edit_score, f1s

    def return_confusion_matrix(self):
        return self.confusion_matrix

    def reset(self):
        self.edit_score = 0
        self.tp = [0 for _ in range(len(self.thresholds))]    # true positive
        self.fp = [0 for _ in range(len(self.thresholds))]    # false positive
        self.fn = [0 for _ in range(len(self.thresholds))]    # false negative
        self.n_correct = 0
        self.n_frames = 0
        self.n_videos = 0
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
