import argparse
import time
import os
import time
import logging, numpy as np

from src import utils as U
from utils import  Model,  update_parameters

logging.getLogger().setLevel(logging.INFO)


def init_parameters():
    parser = argparse.ArgumentParser(description='Skeleton-based Action Recognition')

    # Setting
    parser.add_argument('--config', '-c', type=str, default='', help='ID of the using config', required=True)
    parser.add_argument('--gpus', '-g', type=int, nargs='+', default=[], help='Using GPUs')
    parser.add_argument('--seed', '-s', type=int, default=1, help='Random seed')
    parser.add_argument('--debug', '-db', default=False, action='store_true', help='Debug')
    parser.add_argument('--pretrained_path', '-pp', type=str, default='', help='Path to pretrained models')
    parser.add_argument('--work_dir', '-w', type=str, default='', help='Work dir')
    parser.add_argument('--no_progress_bar', '-np', default=False, action='store_true', help='Do not show progress bar')
    parser.add_argument('--path', '-p', type=str, default='', help='Path to save preprocessed skeleton files')

    # # Processing
    # parser.add_argument('--resume', '-r', default=False, action='store_true', help='Resume from checkpoint')
    parser.add_argument('--evaluate', '-e', default=False, action='store_true', help='Evaluate')
    parser.add_argument('--extract', '-ex', default=False, action='store_true', help='Extract')
    parser.add_argument('--visualization', '-v', default=False, action='store_true', help='Visualization')
    parser.add_argument('--generate_data', '-gd', default=False, action='store_true', help='Generate skeleton data')
    parser.add_argument('--generate_label', '-gl', default=False, action='store_true', help='Only generate label')

    # Visualization
    parser.add_argument('--visualization_class', '-vc', type=int, default=0, help='Class: 1 ~ 60, 0 means true class')
    parser.add_argument('--visualization_sample', '-vs', type=int, default=0, help='Sample: 0 ~ batch_size-1')
    parser.add_argument('--visualization_frames', '-vf', type=int, nargs='+', default=[], help='Frame: 0 ~ max_frame-1')

    # Dataloader
    parser.add_argument('--dataset', '-d', type=str, default='', help='Select dataset')
    parser.add_argument('--dataset_args', default=dict(), help='Args for creating dataset')

    # # Model
    parser.add_argument('--model_type', '-mt', type=str, default='', help='Model type')
    parser.add_argument('--model_args', default=dict(), help='Args for creating model')

    # Optimizer
    parser.add_argument('--optimizer', '-o', type=str, default='', help='Initial optimizer')
    parser.add_argument('--optimizer_args', default=dict(), help='Args for optimizer')

    # LR_Scheduler
    parser.add_argument('--lr_scheduler', '-ls', type=str, default='', help='Initial learning rate scheduler')
    parser.add_argument('--scheduler_args', default=dict(), help='Args for scheduler')

    return parser

if __name__ == '__main__':

    os.chdir(os.getcwd())

    # Loading Parameters
    parser = init_parameters()
    args, _ = parser.parse_known_args()
    # Updating Parameters (cmd > yaml > default)
    args = update_parameters(parser, args)

    # load action recognition model
    model = Model(args, './workdir')
    action_model = './workdir/2022-01-31 03-39-53/1001_pa-resgcn-b29_my_dataset.pth.tar'
    model.load(action_model)
    total_predtime = 0

    logging.info('Making predictions on random generated data')
    for i in range(50):
        start = time.time()
        rand_kps = np.random.randn(2,600,18,1)
        model.preprocess(rand_kps)
        actions = model.predict()
        print(actions)
        pred_time = time.time() - start
        total_predtime += pred_time
    
    logging.info('Average prediction time : {}'.format(total_predtime/50))
