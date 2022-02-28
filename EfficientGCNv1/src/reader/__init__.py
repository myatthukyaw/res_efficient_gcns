import logging

from .ntu_reader import NTU_Reader
from .my_dataset_reader import MyDataset_Reader

__generator = {
    'ntu': NTU_Reader,
    'my_dataset' : MyDataset_Reader
}

def create(args):
    if args.dataset.split('_')[0] == 'ntu':
        dataset = args.dataset.split('-')[0]
    else:
        dataset = args.dataset
    dataset_args = args.dataset_args[dataset]
    if dataset not in __generator.keys():
        logging.info('')
        logging.error('Error: Do NOT exist this dataset: {}!'.format(dataset))
        raise ValueError()
    return __generator[dataset](args, **dataset_args)
