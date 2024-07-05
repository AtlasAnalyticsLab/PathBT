# Original code from: https://github.com/facebookresearch/barlowtwins?tab=MIT-1-ov-file
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license

from pathlib import Path
import argparse
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from tqdm import tqdm
import wandb
import torch.distributed as dist
import torch
from torch import nn, optim
import numpy as np

#add
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.transforms import v2

from factory import Transform,TransformPath,BarlowTwins, adjust_learning_rate,LARS,set_mean_std, choice_dataset


####################
###### PARSER ######
####################
parser = argparse.ArgumentParser(description='Barlow Twins Training')

# dataset and model
parser.add_argument('--dataset',choices = ['pcam','tinykgh','tinykgh_10000','pkgh','pkgh-800','pkgh-600','pkgh-410'],default = 'tinykgh',type = str)
parser.add_argument("--backbone",default = 'resnet50',choices = ['resnet50','resnet18','resnet101','vit_b16','vit_b32','vit_s','swin_s','swin_t','swin_b','swin2_s','swin2_t','swin2_b'],type = str)
parser.add_argument("--ROI",  type=bool, default=False)
parser.add_argument('--projector', default='8192-8192-8192', type=str,
                    metavar='MLP', help='projector MLP')

# training parameter
parser.add_argument('--epochs', default=10, type=int, metavar='N', 
                    help='number of total epochs to run') #1000
parser.add_argument('--batch-size', default=256, type=int, metavar='N',
                    help='mini-batch size') #2048

parser.add_argument('--learning-rate-weights', default=0.2, type=float, metavar='LR',
                    help='base learning rate for weights')
parser.add_argument('--learning-rate-biases', default=0.0048, type=float, metavar='LR',
                    help='base learning rate for biases and batch norm parameters')
parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',
                    help='weight decay')

parser.add_argument('--lambd', default=0.0051, type=float, metavar='L',
                    help='weight on off-diagonal terms')
parser.add_argument("--optimizer",default = 'LARS',choices = ['LARS','AdamW','Adam'],type = str)

parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loader workers')

# pretraining 
parser.add_argument("--pretrained", default=None, choices = [None,"imagenet","pcam"], type=str)

# transformations to be applied
parser.add_argument('--transform', choices = ['vanilla','patho'],default = 'vanilla',type = str)

# parameters to save the models
parser.add_argument('--print-freq', default=100, type=int, metavar='N',
                    help='print frequency')
parser.add_argument('--checkpoint-dir', default='./checkpoint/', type=Path,
                    metavar='DIR', help='path to checkpoint directory')

parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
         distributed training; see https://pytorch.org/docs/stable/distributed.html""")

# Weights&Biases
parser.add_argument('--wandb_login', default="your wandb key", type=str)
parser.add_argument("--wandb", default=True, type=bool,help="set to False if you do not want WandB support")


#torchrun
local_rank = int(os.environ["LOCAL_RANK"])


# function which iterativelt gives an ID to each experiments (provided by Joe Yang from Atlas Analytics Lab)
def get_next_id(filename="id_store.txt"):
    try:
        with open(filename, "r") as f:
            last_id = int(f.read().strip())
    except (FileNotFoundError, ValueError):
        last_id = 0

    next_id = last_id + 1

    with open(filename, "w") as f:
        f.write(str(next_id))

    return next_id


def main():
    print("Entering main")
    args = parser.parse_args()
    

    #args.wandb = False
    if args.wandb:
        wandb.login(key = args.wandb_login)
    args.ngpus_per_node = torch.cuda.device_count()
    print("args.ngpu_per_node = ",args.ngpus_per_node)

    exp_id = get_next_id()
    args.exp_id = exp_id
    print("----- runnning experiment ",args.exp_id," -----")

    #wandb
    if 'SLURM_JOB_ID' in os.environ:
 
        cmd = 'scontrol show hostnames ' + os.getenv('SLURM_JOB_NODELIST')
        stdout = subprocess.check_output(cmd.split())
        host_name = stdout.decode().splitlines()[0]
        args.rank = int(os.getenv('SLURM_NODEID')) * args.ngpus_per_node
        args.world_size = int(os.getenv('SLURM_NNODES')) * args.ngpus_per_node
    else:
        # single-node distributed training
        args.rank = 0
        args.dist_url = 'tcp://localhost:58472'

    args.dist_url = f'tcp://localhost:{random.randrange(49152, 65535)}'
    
    # launch training
    print("----- launch training -----")
    torch.multiprocessing.spawn(main_worker, (args,), nprocs = args.ngpus_per_node)  
    if args.wandb:
        wandb.finish()
    


def main_worker(gpu, args):
    print("In main worker")
    args.rank += gpu
    # config file for WandB
    config={
        "experiment id": args.exp_id,
        "architecture": args.backbone,
        "dataset": args.dataset,
        "fov": args.fovs,
        "epochs": args.epochs,
        "projector": args.projector,
        "lambda":args.lambd,
        "batch size":args.batch_size,
        "nb of GPUs":args.ngpus_per_node,
        "optimizer":args.optimizer,
        "transformations":args.transform,
        "learning_rate_weights": args.learning_rate_weights,
        "learning_rate_biases": args.learning_rate_biases,
        }
    if args.wandb:
        wandb.init(project="KGH project", name = "BT-ENCS-"+args.backbone+"-"+str(args.exp_id),config=config)
    
    print("args.rank = ",args.rank)
    print("--initializing multiprocessing")

    torch.distributed.init_process_group(backend="nccl",init_method = args.dist_url, rank=args.rank, world_size=args.ngpus_per_node)
    print("--- entering main worker")


    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True
    print("---- initializing the model ----")

    ############## SETTING UP BARLOW TWINS MODEL WITH SPECIFIC ENCODER ##############
    model = BarlowTwins(args).cuda(gpu)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # parameters
    param_weights = []
    param_biases = []
    for param in model.parameters():
        if param.ndim == 1:
            param_biases.append(param)
        else:
            param_weights.append(param)
    parameters = [{'params': param_weights}, {'params': param_biases}]
    #sending model on device
    print("--- sending on device")
    if torch.cuda.device_count() > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    print("--- model on GPU")

    ############## OPTIMIZER ##############
    learning_rate_w = args.learning_rate_weights * args.batch_size / 256 #benchmark value
    learning_rate_b = args.learning_rate_biases * args.batch_size / 256
    params_to_optimize = [
    {"params": [param for name, param in model.named_parameters() if 'bn' not in name and 'bias' not in name],
    "lr":learning_rate_w,
    "weight_decay": 1.5e-6},
    {"params": [param for name, param in model.named_parameters() if 'bn' in name or 'bias' in name], 
    "lr":learning_rate_b,
    "weight_decay": 0.0}]
    

    learning_rate = args.learning_rate_weights * args.batch_size / 256 #benchmark value

    if args.optimizer == "LARS":
        optimizer = LARS(parameters, lr=0, weight_decay=args.weight_decay,
                        weight_decay_filter=True,lars_adaptation_filter=True)
        print("--- setting up LARS optimizer")
    if args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(params_to_optimize,lr=learning_rate,weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
        print("--- setting up AdamW optimizer and excluding biases and bn from optimization and weight decay")
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(params_to_optimize,lr=learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
        print("--- setting up Adam optimizer and excluding biases and bn from optimization and weight decay")


    ############## DATASET ##############
    mean, std = set_mean_std(args)
    #size of the input
    size = 224
    if args.backbone == 'resnet18' or args.backbone == 'resnet50' or args.backbone == 'resnet101' or args.dataset == 'tinykgh':
        size = 224
    print("feeding the model with images of size ",size)

    #transformations
    transform = Transform(size,mean,std)

    if args.transform == 'patho':
        print("--- Applying transformation for pathological images")
        transform = TransformPath(size,mean,std)
      
    train_dataset, val_dataset = choice_dataset(args,transform)
    
    if torch.cuda.device_count() > 1:
        train_sampler = torch.utils.data.DistributedSampler(train_dataset, shuffle=True)
        val_sampler = torch.utils.data.DistributedSampler(val_dataset, shuffle=True)
 
        print("sampler multi applied")
    else:
        train_sampler, val_sampler = None, None
        print("sampler unique applied")

    ############## SETTING BATCH SIZE PER DEVICE AND DATA LOADERS ##############
    print("--- batch size = ",args.batch_size)
    print("--- world size = ",args.world_size)
    per_device_batch_size = args.batch_size // args.world_size
    per_device_batch_size_validation = per_device_batch_size//2
    print("batch size per device = ", per_device_batch_size)
    print("For validation, batch size per device = ",per_device_batch_size_validation)
    if args.backbone == 'vit_s':
        train_sampler = torch.utils.data.DistributedSampler(train_dataset, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=per_device_batch_size, num_workers=args.workers,
        pin_memory=True, sampler=train_sampler,drop_last = True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=per_device_batch_size_validation, shuffle = False, num_workers=args.workers,
        pin_memory=True, sampler=val_sampler, drop_last = True)
    print("--- data loaded ---")

    start_time = time.time()
    scaler = torch.cuda.amp.GradScaler()
    start_epoch=0


    
    ############## TRAINING ##############
    for epoch in range(start_epoch, args.epochs):
        
        if torch.cuda.device_count() > 1:
            train_sampler.set_epoch(epoch)
        losses = []
        model.train()
        
        ############## TRAINING ##############
        for step, ((y1,y2), t)  in enumerate(tqdm(train_loader), start=epoch * len(train_loader)):            
            
            y1 = y1.cuda(gpu, non_blocking=True)
            y2 = y2.cuda(gpu, non_blocking=True)
            if args.optimizer == 'LARS':
                adjust_learning_rate(args, optimizer, train_loader, step)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                r1,r2,loss = model.forward(y1, y2)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.detach().cpu())


        ############## VALIDATION ##############
        print("---- VALIDATION ")
        val_losses = []
        model.eval()
        for step, ((y1, y2), t)  in enumerate(tqdm(val_loader), start=epoch * len(val_loader)):
            with torch.no_grad():
                y1 = y1.cuda(gpu, non_blocking=True)
                y2 = y2.cuda(gpu, non_blocking=True)
                t = t
                with torch.cuda.amp.autocast():
                    r1,r2,loss = model.forward(y1, y2)
                val_losses.append(loss.detach().cpu())

        ############## SAVE MODEL EVERY 20 EPOCHS ##############
        if args.rank == 0 and epoch%10==0:
            if args.ngpus_per_node==1:

                state = dict(epoch=epoch + 1, model=model.state_dict(),
                         optimizer=optimizer.state_dict())
            else:
                print("Using module")
                state = dict(epoch=epoch + 1, model=model.state_dict(),
                         optimizer=optimizer.state_dict())
            name_file = 'checkpoint-{}-{}-{}-{}-{}.pth'.format(args.backbone,args.dataset, args.exp_id, args.batch_size, epoch)
            file_path = os.path.join(args.checkpoint_dir,name_file)
            torch.save(state, file_path)

        #saving training curves to WandB
        if args.wandb:
            wandb.log({"training loss": np.mean(losses),"validation loss": np.mean(val_losses)})

        ############## SCHEDULER STEP ##############
        if args.optimizer == 'Adam' or args.optimizer == 'AdamW':
            scheduler.step()

    ############## SAVE FINAL ENCODER BACKBONE ##############
    if args.rank == 0:
        state = dict(epoch=epoch + 1, model=model.state_dict(),
                         optimizer=optimizer.state_dict())
        name_file = 'checkpoint-{}-{}-{}-{}-{}.pth'.format(args.backbone,args.dataset, args.exp_id, args.batch_size, epoch)
        torch.save(state, os.path.join(args.checkpoint_dir,name_file))
        file_path = os.path.join(args.checkpoint_dir,name_file)

        #just backbone
        if args.ngpus_per_node > 1:
            ckpt = model.state_dict()
        else:
            ckpt = model.state_dict()
        name_file = '{}-{}-{}-{}-gpus{}-{}.pth'.format(args.backbone, args.dataset,args.exp_id,args.batch_size,args.ngpus_per_node, args.epochs)
        file_location = os.path.join(args.checkpoint_dir,name_file)
        torch.save(ckpt,
                  file_location)
        if args.wandb:
            wandb.save(file_location)




if __name__ == '__main__':
    main()