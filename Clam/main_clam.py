from __future__ import print_function

import argparse
import pdb
import os
import math

# internal imports
from utils.file_utils import save_pkl, load_pkl
from utils.utils import *
from utils.core_utils_clam import train
from dataset_modules.dataset_generic_clam import Generic_MIL_Dataset

# pytorch imports
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np


def main(train_dataset, test_dataset, args):
    # create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    all_test_f1 = []
    all_val_f1 = []
    all_test_f1w = []
    all_val_f1w = []
    folds = np.arange(start, end)
    for i in folds:
        seed_torch(args.seed)
        train_dataset = train_dataset.return_splits(from_id=False,csv_path='/mnt/c8f0e3a8-8cda-42ea-b586-9c2ac93e02a4/Clam/features/train_split.csv')
        test_dataset = test_dataset.return_splits(from_id=False, csv_path='/mnt/c8f0e3a8-8cda-42ea-b586-9c2ac93e02a4/Clam/features/test_split.csv',sp='test')

        datasets = (train_dataset, test_dataset, test_dataset)
        results, test_auc, val_auc, test_acc, val_acc, test_f1, test_f1w, val_f1, val_f1w = train(datasets, i, args, embedding_dim)
        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        all_test_f1.append(test_f1)
        all_test_f1w.append(test_f1w)
        all_val_f1.append(val_f1)
        all_val_f1w.append(val_f1w)
        # write results to pkl
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, results)

    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_test_auc, 'val_auc': all_val_auc, 'val_f1': all_val_f1, 'val_f1w': all_val_f1w,
                             'test_acc': all_test_acc, 'val_acc': all_val_acc, 'test_f1': all_test_f1, 'test_f1w': all_test_f1w})

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))


# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_root_dir', type=str, default=None,
                    help='data directory')
parser.add_argument('--max_epochs', type=int, default=200,
                    help='maximum number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--label_frac', type=float, default=1.0,
                    help='fraction of training labels (default: 1.0)')
parser.add_argument('--reg', type=float, default=1e-5,
                    help='weight decay (default: 1e-5)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=1, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--results_dir', default='/home/huron/Documents/Clam/CLAM/results/pkgh-410', help='results directory (default: ./results)')
parser.add_argument('--split_dir', type=str, default=None,
                    help='manually specify the set of splits to use, '
                         + 'instead of infering from the task and label_frac argument (default: None)')
parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
parser.add_argument('--opt', type=str, choices=['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out', action='store_true', default=False, help='enable dropout (p=0.25)')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce'], default='ce',
                    help='slide-level classification loss function (default: ce)')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil'], default='clam_sb',
                    help='type of model (default: clam_sb, clam w/ single attention branch)')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small',
                    help='size of model, does not affect mil')
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal', 'task_2_tumor_subtyping'])
### CLAM specific options
parser.add_argument('--no_inst_cluster', action='store_true', default=False,
                    help='disable instance-level clustering')
parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', None], default=None,
                    help='instance-level clustering loss function (default: None)')
parser.add_argument('--subtyping', action='store_true', default=False,
                    help='subtyping problem')
parser.add_argument('--bag_weight', type=float, default=0.7,
                    help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--B', type=int, default=8, help='numbr of positive/negative patches to sample for clam')

#parser.add_argument('--arch', type=str, help='Encoder Architecture to use. ')
parser.add_argument('--image_size', default=224, type=int, help='Image Size of global views.')
parser.add_argument("--source_level", type=str)
parser.add_argument("--target_level", type=str)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_torch(args.seed)

settings = {'num_splits': args.k,
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs,
            'results_dir': args.results_dir,
            'lr': args.lr,
            'experiment': args.exp_code,
            'reg': args.reg,
            'label_frac': args.label_frac,
            'bag_loss': args.bag_loss,
            'seed': args.seed,
            'model_type': args.model_type,
            'model_size': args.model_size,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'opt': args.opt}

if args.model_type in ['clam_sb', 'clam_mb']:
    settings.update({'bag_weight': args.bag_weight,
                     'inst_loss': args.inst_loss,
                     'B': args.B})

args.n_classes = 5
#embedding_dim = 100352
#embedding_dim = 768
embedding_dim = 2048 #100352 #2048 #768 #2048 
# embedding_map = {'vit-t': 768,
#                  'vim-t': 192,
#                  'vit-s': 1536,
#                  'vim-s': 384,
#                  'vim-s2': 384
#                  }
# embedding_dim = embedding_map[args.arch]

train_dataset = Generic_MIL_Dataset(csv_path='/mnt/c8f0e3a8-8cda-42ea-b586-9c2ac93e02a4/Clam/features/pkgh-410/features-sup/KGH_slides.csv',
                              data_dir='/mnt/c8f0e3a8-8cda-42ea-b586-9c2ac93e02a4/Clam/features/pkgh-410/features-sup',
                              shuffle=True,
                              seed=args.seed,
                              print_info=True,
                              label_dict = {"N" : 0, "TA": 1, "TVA": 2, "HP": 3, "SSL": 4},
                              patient_strat=False,
                              ignore=[])

test_dataset = Generic_MIL_Dataset(csv_path='/mnt/c8f0e3a8-8cda-42ea-b586-9c2ac93e02a4/Clam/features/pkgh-410/features-test-sup/KGH_slides.csv',
                              data_dir='/mnt/c8f0e3a8-8cda-42ea-b586-9c2ac93e02a4/Clam/features/pkgh-410/features-test-sup',
                              shuffle=True,
                              seed=args.seed,
                              print_info=True,
                              label_dict = {"N" : 0, "TA": 1, "TVA": 2, "HP": 3, "SSL": 4},
                              patient_strat=False,
                              ignore=[])

if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)


with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))

if __name__ == "__main__":
    results = main(train_dataset, test_dataset, args)
    print("finished!")
    print("end script")