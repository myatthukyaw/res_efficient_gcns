import logging

from .ntu_generator import NTU_Generator
from .kinetics_generator import Kinetics_Generator
from .cmu_generator import CMU_Generator
from .my_dataset_generator import MyDataset_Generator

__generator = {
    'ntu': NTU_Generator,
    'kinetics': Kinetics_Generator,
    'cmu': CMU_Generator,
    'zensho' : MyDataset_Generator
}

def create(args):
    dataset = args.dataset
    #dataset = args.dataset.split('-')[0]
    dataset_args = args.dataset_args[dataset]
    print(dataset_args)
    if dataset not in __generator.keys():
        logging.info('')
        logging.error('Error: Do NOT exist this dataset: {}!'.format(args.dataset))
        raise ValueError()
    print('here again')
    return __generator[dataset](args, dataset_args)
