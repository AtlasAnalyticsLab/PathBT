# Original code from: https://github.com/facebookresearch/barlowtwins?tab=MIT-1-ov-file
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license
# added AUC computation

from pathlib import Path
import argparse
import json
import os
import random
import signal
import sys
import time
import urllib
from tqdm import tqdm
from torch import nn, optim
from torchvision import models, datasets, transforms
import torch
import torchvision
import wandb
import numpy as np
import pandas as pd
from factory import BarlowTwins, backbone_eval, modify_last_layer, set_mean_std, choice_dataset_eval
from RandStainNA.randstainna import RandStainNA
from sklearn.metrics import roc_auc_score


####################
###### PARSER ######
####################
parser = argparse.ArgumentParser(description='Evaluate resnet50 features on ImageNet')

# dataset and model
parser.add_argument('--dataset',choices = ['pcam','tinykgh','tinykgh_10000','imagenet','pkgh','pkgh_tuning','pkgh-800','pkgh-600','pkgh-410'],default = 'tinykgh',type = str)
parser.add_argument('--pretrained',  type=Path, metavar='FILE',
                    help='path to pretrained model')
parser.add_argument("--backbone_ckpt",default = False, type = bool)
parser.add_argument("--backbone",default = 'resnet50',choices = ['resnet50','resnet18','resnet101','vit_b16','vit_b32','swin_s','swin_t','swin_b','swin2_s','swin2_t','swin2_b'],type = str)
parser.add_argument('--projector', default='4096-4096-4096', type=str, metavar='MLP', help='projector MLP')
parser.add_argument('--data', default= "/home/ubuntu/Documents/Datasets/tinyKGH_5000", type=Path, metavar='DIR',
                    help='path to dataset')

# training parameter
parser.add_argument('--weights', default='freeze', type=str,
                    choices=('finetune', 'freeze'),
                    help='finetune or freeze resnet weights')
parser.add_argument("--optimizer",default = False, type = bool)
parser.add_argument('--train-percent', default=100, type=int,
                    choices=(100, 10, 1),
                    help='size of traing set in percent')
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=64, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--lr-backbone', default=0.0, type=float, metavar='LR',
                    help='backbone base learning rate')
parser.add_argument('--lr-classifier', default=0.3, type=float, metavar='LR',
                    help='classifier base learning rate')
parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',
                    help='weight decay')

# parameters to save the models
parser.add_argument('--print-freq', default=100, type=int, metavar='N',
                    help='print frequency')
parser.add_argument('--checkpoint-dir', default='./checkpoint/lincls/', type=Path,
                    metavar='DIR', help='path to checkpoint directory')
parser.add_argument('--folder_eval', default = "/home/ubuntu/Documents/KGH_RepLearn/Experiments/BarlowTwins/models_evaluation/resnet50/tinyKGH_5000/tuning/projector", type = Path)

# Weights&Biases
parser.add_argument('--wandb_login', default="your wandb key", type=str)
parser.add_argument("--wandb", default=True, type=bool,help="set to False if you do not want WandB support")

# if randstain should be used for evaluation
parser.add_argument("--randstain",default = False, type = bool)


def get_next_id(filename="id_eval_store.txt"):
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
    args = parser.parse_args()
    print("evaluating model ",args.pretrained)
    print(" on dataset ",args.dataset)
    if args.wandb:
        wandb.login(key = args.wandb_login)
    if args.train_percent in {1, 10}:
        args.train_files = urllib.request.urlopen(f'https://raw.githubusercontent.com/google-research/simclr/master/imagenet_subsets/{args.train_percent}percent.txt').readlines()
    args.ngpus_per_node = torch.cuda.device_count()
    exp_id = get_next_id()
    args.exp_id = exp_id
    if 'SLURM_JOB_ID' in os.environ:
        signal.signal(signal.SIGUSR1, handle_sigusr1)
        signal.signal(signal.SIGTERM, handle_sigterm)
    # single-node distributed training
    args.rank = 0
    args.dist_url = f'tcp://localhost:{random.randrange(49152, 65535)}'
    args.world_size = args.ngpus_per_node
    torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)


def main_worker(gpu, args):
    print("--args.backbone = ",args.backbone_ckpt)
    config={
    "experiment id": args.exp_id,
    "name model/checkpoint": os.path.basename(args.pretrained),
    "architecture": "Barlow Twins with ResNet50",
    "dataset": args.dataset,
    "epochs": args.epochs,
    "batch size":args.batch_size,
    "optimizer":"SGD",
    "transformations":"default"
    }
    if args.wandb:
        wandb.init(project="KGH project", name = "BT-eval-"+args.backbone+"-desk_"+str(args.exp_id),config=config)
    args.rank += gpu
    torch.distributed.init_process_group(
        backend='nccl', init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)

    if args.rank == 0:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.checkpoint_dir / 'stats.txt', 'a', buffering=1)
        print(' '.join(sys.argv))
        print(' '.join(sys.argv), file=stats_file)

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True
    
    
    ############## LOAD WEIGHTS INTO RIGHT MODEL ##############
    state_dict = torch.load(args.pretrained, map_location='cpu')
    #print(state_dict)
    if args.backbone_ckpt:
        state_dict = state_dict['model']
    # if saved final model (and only the model)
    if "backbone.conv1.weight" in state_dict or 'projector.1.weight' in state_dict:
        model = BarlowTwins(args).cuda(gpu) 
        print("BACKBONE = ",args.backbone)   
        #model.load_state_dict(state_dict['model'], strict = False)
        #from March 22nd
        model.load_state_dict(state_dict, strict = True)
        print("--- extracting model from the backbone ---")
        state_dict = model.backbone.state_dict()
    
    # if saved model, optimizer, scheduler, epoch   
    if "module.backbone.conv1.weight" in state_dict or 'module.projector.1.weight' in state_dict:
        model = BarlowTwins(args).cuda(gpu) 
        print("BACKBONE = ",args.backbone)   
        #model.load_state_dict(state_dict['model'], strict = False)
        #from March 22nd
        new_state_dict = {k[7:]:v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        print("--- extracting model from the backbone ---")
        state_dict = model.backbone.state_dict()
    

    # if saved backbone
    model = backbone_eval(args).cuda(gpu)
    model.load_state_dict(state_dict, strict=False)

    args.num_classes = 5
    if args.dataset == 'pcam':
        args.num_classes = 2
    model = modify_last_layer(model,args)
    

    ############## PARAMETERS TO UPDATE ##############
    classifier_parameters, model_parameters = [], []
    for name, param in model.named_parameters():
        #print(name)
        if name in {'fc.weight', 'fc.bias','head.weight','head.bias','heads.head.weight','heads.head.bias'}:
            #print("Classifier parameters = ",classifier_parameters)
            classifier_parameters.append(param)
        else:
            model_parameters.append(param)

    model = torch.nn.parallel.DistributedDataParallel(model.cuda(gpu), device_ids=[gpu])


    ############## LOSS ##############
    criterion = nn.CrossEntropyLoss().cuda(gpu)

    ############## OPTIMIZER & SCHEDULER ##############
    param_groups = [dict(params=classifier_parameters, lr=args.lr_classifier)]
    if args.weights == 'finetune':
        param_groups.append(dict(params=model_parameters, lr=args.lr_backbone))
    optimizer = optim.SGD(param_groups, 0, momentum=0.9, weight_decay=args.weight_decay)
    #optimizer = optim.Adam(param_groups,lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    start_epoch = 0
    best_acc = argparse.Namespace(top1=0, top2=0, top3=0)

    # Data loading code
    traindir = args.data / 'train'
    valdir = args.data / 'val'
    
    ############## DATASET ##############
    mean, std = set_mean_std(args)
    
    normalize = transforms.Normalize(mean=mean,
                                     std=std)
    # transform = transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])
    transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.1, contrast=0.1,
                                        saturation=0.1, hue=0.1)],
                p=0.8
            ),
            transforms.ToTensor(),
            normalize,
        ])
    if args.randstain :
        print("Setting up transformation with RandStainNA")
        rd_na = RandStainNA(
            yaml_file="/home/ubuntu/Documents/KGH_RepLearn/Experiments/BarlowTwins/RandStainNA/my_output/KGH.yaml",
            std_hyper=-0.3,
            probability=1.0,
            distribution="normal",
            is_train=True,)
        
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.1, contrast=0.1,
                                        saturation=0.1, hue=0.1)],
                p=0.8
            ),
            rd_na,
            transforms.ToTensor(),
        ])
        
    train_dataset, val_dataset = choice_dataset_eval(args,transform)
    print(f"--train set = {len(train_dataset)} images and test set = {val_dataset} images")
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    kwargs = dict(batch_size=args.batch_size // args.world_size, num_workers=args.workers, pin_memory=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, **kwargs)


    start_time = time.time()

    if args.weights == 'freeze':
        # reference_state_dict = torch.load(args.pretrained, map_location='cpu')
        reference_state_dict = state_dict
        model_state_dict = model.module.state_dict()
        for k in reference_state_dict:
            assert torch.equal(model_state_dict[k].cpu(), reference_state_dict[k].cpu()), k
        print("--- sanity check done")
    best_auc = 0
    ############## TRAINING ##############
    for epoch in range(start_epoch, args.epochs):
        # train
        if args.weights == 'finetune':
            model.train()
        elif args.weights == 'freeze':
            model.eval()
        else:
            assert False
        train_sampler.set_epoch(epoch)
        training_losses = []
        losses = []

        for step, (images, target) in enumerate(train_loader, start=epoch * len(train_loader)):
            output = model(images.cuda(gpu, non_blocking=True))
            target = target.squeeze()
            loss = criterion(output, target.cuda(gpu, non_blocking=True))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % args.print_freq == 0:
                torch.distributed.reduce(loss.div_(args.world_size), 0)
                if args.rank == 0:
                    pg = optimizer.param_groups
                    lr_classifier = pg[0]['lr']
                    lr_backbone = pg[1]['lr'] if len(pg) == 2 else 0
                    stats = dict(epoch=epoch, step=step, lr_backbone=lr_backbone,
                                 lr_classifier=lr_classifier, loss=loss.item(),
                                 time=int(time.time() - start_time))
                    print(json.dumps(stats))
                    print(json.dumps(stats), file=stats_file)
            losses.append(loss.detach().cpu())

              
        ############## EVALUATION ##############
        print("--evaluation--")
        val_losses = []
        outputs, true_labels = [], []
        sf = nn.Softmax(dim=1)
        if args.rank == 0:
            top1 = AverageMeter('Acc@1')
            top2 = AverageMeter('Acc@2')
            top3 = AverageMeter('Acc@3')
            with torch.no_grad():
                for images, target in tqdm(val_loader):
                    output = model(images.cuda(gpu, non_blocking=True))
                    t = target.squeeze()
                    loss = criterion(output, t.cuda(gpu, non_blocking=True))
                    outputs.extend(sf(output).cpu().numpy())
                    _,predicted = torch.max(output.data, 1)
                    true_labels.extend(t.cpu().numpy())

                    acc1, acc2 = accuracy(output, target.cuda(gpu, non_blocking=True), topk=(1, 2))
                    _, acc3 = accuracy(output, target.cuda(gpu, non_blocking=True), topk=(1, 3))
                    top1.update(acc1[0].item(), images.size(0))
                    top2.update(acc2[0].item(), images.size(0))
                    top3.update(acc3[0].item(), images.size(0))
                    val_losses.append(loss.detach().cpu())

            # AUC computation + save model if best AUC    
            AUC_ovo = roc_auc_score(true_labels, outputs, multi_class='ovo')
            if AUC_ovo > best_auc:
                best_auc = AUC_ovo
                print(f"Saving new best model with AUC of {best_auc}")
                state = model.state_dict()
                name_file = 'BEST-AUC-exp{}-{}-{}-{}.pth'.format(args.exp_id, args.backbone,args.dataset, args.batch_size)
                torch.save(state, args.folder_eval / name_file)

            # save model if best ACC
            if top1.avg > best_acc.top1:
                #save best test model
                print(f"Saving new best model with top-1 accuracy of {top1.avg}")
                state = model.state_dict()
                name_file = 'BEST-exp{}-{}-{}-{}.pth'.format(args.exp_id, args.backbone,args.dataset, args.batch_size)
                torch.save(state, args.folder_eval / name_file)
                

            best_acc.top1 = max(best_acc.top1, top1.avg)
            best_acc.top2 = max(best_acc.top2, top2.avg)
            best_acc.top3 = max(best_acc.top3, top3.avg)
            stats = dict(epoch=epoch, acc1=top1.avg, acc2=top2.avg, acc3=top3.avg, best_acc1=best_acc.top1, best_acc2=best_acc.top2, best_acc3=best_acc.top3)
            
            print(json.dumps(stats))
            print(json.dumps(stats), file=stats_file)
        if args.wandb:
            wandb.log({"training loss": np.mean(losses),"validation loss": np.mean(val_losses), "acc1": top1.avg, "AUC":AUC_ovo, "acc2": top2.avg, "acc3": top3.avg})
        # sanity check
        if args.weights == 'freeze':
            # reference_state_dict = torch.load(args.pretrained, map_location='cpu')
            reference_state_dict = state_dict
            model_state_dict = model.module.state_dict()
            for k in reference_state_dict:
                assert torch.equal(model_state_dict[k].cpu(), reference_state_dict[k].cpu()), k

        scheduler.step()

        ############## SAVE MODEL EVERY 25 EPOCHS ##############
        if args.rank == 0 and epoch%25==0:
            state = dict(
                epoch=epoch + 1, best_acc=best_acc, model=model.state_dict(),
                optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict())
            name_file = 'ckpt-eval-{}-{}-{}-{}-{}.pth'.format(args.backbone,args.dataset, args.exp_id, args.batch_size, epoch)
            torch.save(state, args.folder_eval / name_file)

    accs = np.array([best_acc.top1,best_acc.top3])
    df = pd.DataFrame(accs)
    folder_eval = args.folder_eval 
    state = model=model.state_dict()
    name_file = 'final-exp{}-{}-{}-{}.pth'.format(args.exp_id, args.backbone,args.dataset, args.batch_size)
    torch.save(state, args.folder_eval / name_file)
    accs_file_path = os.path.join(folder_eval, os.path.basename(args.pretrained) + '.csv')
    df.to_csv(accs_file_path, index=False)
    print(f"Best AUC = {best_auc}")
    if args.wandb:
        wandb.save(accs_file_path)





############## USEFUL FUNCTIONS ##############
def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()


def handle_sigterm(signum, frame):
    pass


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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()