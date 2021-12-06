import sys
import os


def get_class2id_map(dataset, dataset_dir='./dataset'):
    """
    Args:
        dataset: 50salads, gtea, breakfast
        dataset_dir: the path to the datset directory
    """

    if (dataset == '50salads') or (dataset == 'gtea') or (dataset == 'breakfast')or (dataset == 'Skating_segmentation')or (dataset == 'Skating22'):
        with open(os.path.join(dataset_dir, "{}/mapping.txt".format(dataset)), 'r') as f:
            actions = f.read().split('\n')[:-1]

        class2id_map = dict()
        for a in actions:
            class2id_map[a.split()[1]] = int(a.split()[0])
    else:
        print('You have to choose 50salads, gtea or breakfast as dataset.')
        sys.exit(1)

    return class2id_map


def get_id2class_map(dataset, dataset_dir='./dataset'):
    class2id_map = get_class2id_map(dataset, dataset_dir)

    return {val: key for key, val in class2id_map.items()}


def get_n_classes(dataset, dataset_dir='./dataset'):
    return len(get_class2id_map(dataset, dataset_dir))
