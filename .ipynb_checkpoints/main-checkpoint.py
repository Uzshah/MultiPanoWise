from __future__ import absolute_import, division, print_function
import os
import argparse

from trainer_s2d3d import Trainer
from trainer_structured3d import Trainer_s3d
import warnings
warnings.filterwarnings("ignore")


def get_args_parser():
    parser = argparse.ArgumentParser(
        'MultiPanoWise training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--num_epochs', default=50, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='START of CHECKPOINT',
                        help='Checkpoint and resume ')

    # Model parameters
    parser.add_argument('--model_name', default='MPW', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=[512, 1024],
                        type=int, help='images input size')
    parser.add_argument('--data_path', default='Data/full', type=str, metavar='DATASET PATH',
                        help='Path to dataset')


    # Optimizer parameters
    parser.add_argument('--opt', default='Adam', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--weighted_loss', default=False, type=bool, metavar='WEIGHTED LOSS',
                        help='')
    
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='CosineAnnealingWarmRestarts', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--learning_rate', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--alpha', type=float, default=0, metavar='Weighting',
                        help='weighting scheme')
    parser.add_argument('--split', default=['Data/full'], type = str, 
                        nargs="*",help='Image Net dataset path')
    parser.add_argument("--load_weights_dir", default=None, type=str, help="folder of model to load")
    
    parser.add_argument('--dataset', default='s3d', choices=['s3d', 'ade20k', 's2d3d'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--target', default='all', choices=['all', 'depth', 'shading', 'albedo', 'normal', 'semantic'],
                        type=str, help='task selection')
    
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=2023, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    parser.add_argument("--log_dir", type=str, default=os.path.join(os.path.dirname(__file__), "tmp"), help="log directory")
    parser.add_argument("--log_frequency", type=int, default=100, help="number of batches between each tensorboard log")
    parser.add_argument("--save_frequency", type=int, default=1, help="number of epochs between each save")
    
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    
    return parser
    
def main(args):
    if args.dataset =="s3d":
         trainer = Trainer_s3d(args)
    else:
        trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'MultiPanoWise training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
