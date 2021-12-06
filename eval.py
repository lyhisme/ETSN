import argparse
import csv
import os
import pandas as pd
import time
import torch
import yaml

from addict import Dict
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from libs import models
from libs.dataset import ActionSegmentationDataset
from libs.metric import ScoreMeter
from libs.transformer import TempDownSamp, ToTensor
from libs.lbs import LBS
from utils.class_id_map import get_id2class_map, get_n_classes


def get_arguments():
    '''
    parse all the arguments from command line inteface
    return a list of parsed arguments
    '''

    parser = argparse.ArgumentParser(
        description='train a network for action recognition')
    parser.add_argument('config', type=str, help='path to a config file')
    parser.add_argument('--mode', type=str, default='test', help='validation or test')
    parser.add_argument(
        '--model', type=str, default=None,
        help='path to the trained model. If you do not specify, the trained model, \'best_acc1_model.prm\' in result directory will be used.')
    parser.add_argument(
        '--cpu', action='store_true', help='Add --cpu option if you use cpu.')

    return parser.parse_args()


def test(loader, model, config, device):
    scores = ScoreMeter(
        id2class_map=get_id2class_map(
            config.dataset, dataset_dir=config.dataset_dir),
        thresholds=config.thresholds
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for sample in loader:
            x = sample['feature']
            t = sample['label']
            x = x.to(device)
            t = t.to(device)

            # compute output
            output = model(x)

            # measure pixel accuracy, mean accuracy, Frequency Weighted IoU, mean IoU, class IoU
            pred = output.data.max(1)[1].squeeze(0).cpu().numpy()

            if config.lbs:
                pred = LBS(pred,output,config)

            gt = t.data.cpu().squeeze(0).numpy()
            scores.update(pred, gt)

    acc, edit_score, f1s = scores.get_scores()
    c_matrix = scores.return_confusion_matrix()

    return acc, edit_score, f1s, c_matrix


def main():
    # measure elapsed time
    start = time.time()

    args = get_arguments()

    # configuration
    CONFIG = Dict(yaml.safe_load(open(args.config)))
    CONFIG.result_path = CONFIG.result_path + '/split' + str(CONFIG.split)

    # cpu or gpu
    if args.cpu:
        device = 'cpu'
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            torch.backends.cudnn.benchmark = True

    # Dataloader
    downsamp_rate = 2 if CONFIG.dataset == '50salads' else 1

    data = ActionSegmentationDataset(
        CONFIG.dataset,
        transform=Compose([
            ToTensor(),
            TempDownSamp(downsamp_rate)
        ]),
        mode=args.mode,
        split=CONFIG.split,
        dataset_dir=CONFIG.dataset_dir,
        csv_dir=CONFIG.csv_dir
    )

    loader = DataLoader(
        data,
        batch_size=1,
        shuffle=False,
        num_workers=CONFIG.num_workers
    )

    # load model
    print('\n------------------------Loading Model------------------------\n')

    n_classes = get_n_classes(CONFIG.dataset, dataset_dir=CONFIG.dataset_dir)

    print('Multi Stage TCN will be used as a model.')
    print('stages: {}\tn_features: {}\tn_layers of dilated TCN: {}\tkernel_size of ED-TCN: {}'
          .format(CONFIG.stages, CONFIG.n_features, CONFIG.dilated_n_layers, CONFIG.kernel_size))
    model = models.MultiStageTCN(
        in_channel=CONFIG.in_channel,
        n_classes=n_classes,
        stages=CONFIG.stages,
        n_features=CONFIG.n_features,
        dilated_n_layers=CONFIG.dilated_n_layers,
        kernel_size=CONFIG.kernel_size
    )

    # send the model to cuda/cpu
    model.to(device)

    # load the state dict of the model
    if args.model is not None:
        state_dict = torch.load(args.model)
    else:
        state_dict = torch.load(
            os.path.join(CONFIG.result_path, 'epoch_best_model.prm'), map_location='cuda:0')
    model.load_state_dict(state_dict ,strict=False)

    # train and validate model
    print('\n------------------------Start testing------------------------\n')

    # validation
    acc, edit_score, f1s, c_matrix = test(loader, model, CONFIG, device)

    # save log
    columns = ['acc', 'edit']
    columns += ["f1s@{}".format(CONFIG.thresholds[i])
                for i in range(len(CONFIG.thresholds))]
    log = pd.DataFrame(columns=columns)

    tmp = [acc, edit_score]
    tmp += [f1s[i] for i in range(len(CONFIG.thresholds))]
    tmp_df = pd.Series(tmp, index=log.columns)

    log = log.append(tmp_df, ignore_index=True)
    log.to_csv(
        os.path.join(CONFIG.result_path, '{}_log.csv').format(args.mode), index=False)

    with open(os.path.join(CONFIG.result_path, '{}_c_matrix.csv').format(args.mode), 'w') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerows(c_matrix)

    elapsed_time = (time.time() - start) / 60
    print(
        'elapsed_time: {:.1f}min\tacc: {:.5f}\tedit: {:.5f}\tf1s@{}: {}'.format(
            elapsed_time, acc, edit_score, CONFIG.thresholds, f1s
        )
    )


if __name__ == '__main__':
    main()
