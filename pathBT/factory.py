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
from pkgh_1fov import pkgh_1fov
from pkgh import pkgh
import wandb
import torch.distributed as dist
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from torchvision.transforms import v2

#add
from torch.nn.parallel import DistributedDataParallel as DDP


from PIL import Image, ImageOps, ImageFilter
from torch import nn, optim
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import h5py
import random
import os
from functools import partial

def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    optimizer.param_groups[0]['lr'] = lr * args.learning_rate_weights
    optimizer.param_groups[1]['lr'] = lr * args.learning_rate_biases


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()


def handle_sigterm(signum, frame):
    pass


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def save_checkpoint(args,epoch,model,optimizer):
    state = dict(epoch=epoch + 1, model=model.state_dict(),
                         optimizer=optimizer.state_dict())
    name_file = 'checkpoint-{}-{}-{}-{}-{}.pth'.format(args.exp_id,args.dataset,args.batch_size,epoch)
    torch.save(state, args.checkpoint_dir / name_file)
    file_path = os.path.join(args.checkpoint_dir,name_file)
    wandb.save(file_path)

class BarlowTwins(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        weights = None
        if args.pretrained == 'imagenet':
            weights = 'IMAGENET1K_V2'
            print("--- weights for imagenet")
        if args.pretrained == 'pcam':
            weights = torch.load("./PCam_weights_ResNet50.pth")
        ## RESNET ##
        #print("Weights = ",weights)
        if args.backbone == "resnet50":
            self.backbone = torchvision.models.resnet50()
            if args.pretrained == 'imagenet':
                self.backbone = torchvision.models.resnet50(weights = weights)
            if args.pretrained == 'pcam':
                self.backbone = torchvision.models.resnet50(num_classes = 2)
                self.backbone.load_state_dict(weights["model"])

                print("---loaded PCam weights")
            self.backbone.fc = nn.Identity()
            # projector
            sizes = [2048] + list(map(int, args.projector.split('-')))
        if args.backbone == "resnet18":
            print("Loading RN18 with weights = ",weights)
            self.backbone = torchvision.models.resnet18(weights = torchvision.models.ResNet18_Weights.DEFAULT)
            self.backbone.fc = nn.Identity()
            # projector
            sizes = [512] + list(map(int, args.projector.split('-')))
        if args.backbone == "resnet101":
            self.backbone = torchvision.models.resnet101(zero_init_residual=True, weights = None)
            self.backbone.fc = nn.Identity()
            # projector
            sizes = [2048] + list(map(int, args.projector.split('-')))
    
        ## ViT ##            
        if args.backbone == "vit_b16":
            self.backbone = torchvision.models.vit_b_16()
            self.backbone.heads[-1] = nn.Identity()
            # projector
            sizes = [768] + list(map(int, args.projector.split('-')))           
        if args.backbone == "vit_b32":
            self.backbone = torchvision.models.vit_b_32()
            self.backbone.heads[-1] = nn.Identity()
            # projector
            sizes = [768] + list(map(int, args.projector.split('-'))) 
        
        ## Swin ##
        if args.backbone == "swin_s":
            self.backbone = torchvision.models.swin_s(weights='IMAGENET1K_V1')
            self.backbone.head = nn.Identity()
            # projector
            sizes = [768] + list(map(int, args.projector.split('-')))  
        if args.backbone == "swin_t":
            self.backbone = torchvision.models.swin_t(weights='IMAGENET1K_V1')
            self.backbone.head = nn.Identity()
            # projector
            sizes = [768] + list(map(int, args.projector.split('-')))
        if args.backbone == "swin_b":
            self.backbone = torchvision.models.swin_b(weights = weights)
            self.backbone.head = nn.Identity()
            # projector
            sizes = [1024] + list(map(int, args.projector.split('-')))
        if args.backbone == "swin2_s":
            self.backbone = torchvision.models.swin_v2_s(weights = weights)
            self.backbone.head = nn.Identity()
            # projector
            sizes = [768] + list(map(int, args.projector.split('-')))
        if args.backbone == "swin2_t":
            self.backbone = torchvision.models.swin_v2_t(weights = weights)
            self.backbone.head = nn.Identity()
            # projector
            sizes = [768] + list(map(int, args.projector.split('-')))
        if args.backbone == "swin2_b":
            self.backbone = torchvision.models.swin_v2_b(weights = weights)
            self.backbone.head = nn.Identity()
            # projector
            sizes = [1024] + list(map(int, args.projector.split('-')))
        
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, y1, y2):
        r1 = self.backbone(y1)
        r2 = self.backbone(y2)
        z1 = self.projector(r1)
        z2 = self.projector(r2)

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size)
        if torch.cuda.device_count() > 1:
          torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.args.lambd * off_diag
        return r1,r2,loss

class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=False, lars_adaptation_filter=False):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)


    def exclude_bias_and_norm(self, p):
        return p.ndim == 1

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if not g['weight_decay_filter'] or not self.exclude_bias_and_norm(p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if not g['lars_adaptation_filter'] or not self.exclude_bias_and_norm(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])



class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class Transform:
    def __init__(self,size,mean,std):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.4, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0),
            Solarization(p=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,
                                 std=std)
        ])
        self.transform_prime = transforms.Compose([
            transforms.RandomResizedCrop(size, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.4, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,
                                 std=std)
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2


class TransformPath:
    def __init__(self,size,mean,std):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply([transforms.v2.RandomAffine(degrees = (0,45), translate = (0.5,0.5))],p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.1, contrast=0.1,
                                        saturation=0.1, hue=0.1)],
                p=0.2
            ),
            transforms.v2.RandomSolarize(threshold = 250,p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,
                                 std=std)
        ])
        self.transform_prime = transforms.Compose([
            transforms.RandomResizedCrop(size, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply([transforms.v2.RandomAffine(degrees = (0,45), translate = (0.5,0.5))],p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.1, contrast=0.1,
                                        saturation=0.1, hue=0.1)],
                p=0.2
            ),
            transforms.v2.RandomPosterize(bits = 7, p = 0.5),
            transforms.v2.RandomSolarize(threshold = 250,p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,
                                 std=std)            
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2


def get_mean_std(data):
    mean, std = torch.zeros(3), torch.zeros(3)
    for img,_ in data:
        #img = el[0]
        #print(img)
        mean += torch.mean(img, dim=(1, 2))
        std += torch.std(img, dim=(1, 2))
        print("mean, std = ",mean, std)
    mean /= len(data)
    std /= len(data)
    print("--- FINAL mean, std = ",mean, std)
    return mean,std


def choice_dataset(args,transform):

    if args.dataset=="pcam":
        args.datafolder = "CHANGE PATH"
        train_dataset = datasets.PCAM(root=args.datafolder, split= 'train', transform=transform,download = True)
        val_dataset = datasets.PCAM(root=args.datafolder, split= 'val', transform=transform,download = True)
        print("PCAM loaded")
    
    
    if args.dataset == "tinykgh":
        root = "CHANGE PATH"
        
        dataset = pkgh_1fov(root_dir = root, transform=transform)

        #splitting dataset
        tr,va,te = 0.80, 0.10 , 0.10
        print("Applying a split of ",tr,va,te)
        train_size = int(tr * len(dataset))
        val_size = int(va * len(dataset))
        test_size = len(dataset) - train_size - val_size
        #fixing the seed
        generator1 = torch.Generator().manual_seed(42)
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size],generator=generator1)
        print("Length of train dataset = ",len(train_dataset))
    
    if args.dataset == "pkgh":
        root = "CHANGE PATH" #atlas
        
        dataset = pkgh(root_dir = root, split = 'train', ROI = args.ROI, transform=transform)
        #splitting dataset
        tr,va= 0.90, 0.10
        print("Applying a split of ",tr,va)
        train_size = int(tr * len(dataset))
        val_size = len(dataset) - train_size
        #fixing the seed
        generator1 = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size,],generator=generator1)
        print("Length of train dataset = ",len(train_dataset))
    
    if args.dataset == "pkgh-800":
        root = "CHANGE PATH"
        
        dataset = pkgh(root_dir = root, split = 'train', ROI = args.ROI, transform=transform)
        #splitting dataset
        tr,va= 0.90, 0.10
        print("Applying a split of ",tr,va)
        train_size = int(tr * len(dataset))
        val_size = len(dataset) - train_size
        #fixing the seed
        generator1 = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size,],generator=generator1)
        print("Length of train dataset = ",len(train_dataset))
    
    if args.dataset == "pkgh-600":
        root = "CHANGE PATH"
        
        dataset = pkgh(root_dir = root, split = 'train', ROI = args.ROI, transform=transform)
        #splitting dataset
        tr,va= 0.90, 0.10
        print("Applying a split of ",tr,va)
        train_size = int(tr * len(dataset))
        val_size = len(dataset) - train_size
        #fixing the seed
        generator1 = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size,],generator=generator1)
        print("Length of train dataset = ",len(train_dataset))
    
    if args.dataset == "pkgh-410":
        root = "CHANGE PATH"
        
        dataset = pkgh(root_dir = root, split = 'train', ROI = args.ROI, transform=transform)
        #splitting dataset
        tr,va= 0.90, 0.10
        print("Applying a split of ",tr,va)
        train_size = int(tr * len(dataset))
        val_size = len(dataset) - train_size
        #fixing the seed
        generator1 = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size,],generator=generator1)
        print("Length of train dataset = ",len(train_dataset))
    
    return train_dataset,val_dataset

def set_mean_std(args):
    # imagenet values
    mean=[0.8549, 0.6397, 0.7591]
    std=[0.1263, 0.1987, 0.1488]
    if args.dataset == 'tinykgh':
        mean = [0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
    elif args.dataset == 'pcam':
        mean=[0.676,0.566,0.664]
        std=[0.227,0.253,0.217]
    elif args.dataset == 'pkgh':
        mean = [0.8977, 0.7313, 0.8159]
        std = [0.0855, 0.1797, 0.1237]
    elif args.dataset == 'pkgh-800':
        mean = [0.8816, 0.7241, 0.8087]
        std = [0.1132, 0.2006, 0.1436]
    elif args.dataset == 'pkgh-600' or args.dataset == 'pkgh-410':
        mean = [0.8733, 0.7039, 0.7952]
        std = [0.1078, 0.1912, 0.1369]
    return mean,std


def choice_dataset_eval(args,transform):

    if args.dataset == "tinykgh":
        # to change
        root = "CHANGE PATH"
        args.batch_size = 32 #64
        dataset = pkgh_1fov(root_dir = root, transform=transform)
        tr,va,te = 0.8, 0.1 , 0.1
        print("Applying a split of ",tr,va,te)
        train_size = int(tr * len(dataset ))
        val_size = int(va * len(dataset))
        test_size = len(dataset) - train_size - val_size
        generator1 = torch.Generator().manual_seed(42)
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset , [train_size, val_size, test_size],generator=generator1)

    elif args.dataset == "pkgh":
        print("--loading from pkgh")
        root = "CHANGE PATH"
        train_dataset = pkgh(root_dir = root, split = 'train', ROI = True, balance = True, transform = transform )
        test_dataset = pkgh(root_dir = root, split = 'test', ROI = True, balance = False, transform = transform )
    
    elif args.dataset == "pkgh-800":
        print("--loading from pkgh")
        root = "CHANGE PATH"
        train_dataset = pkgh(root_dir = root, split = 'train', ROI = True, balance = True, transform = transform )
        test_dataset = pkgh(root_dir = root, split = 'test', ROI = True, balance = False, transform = transform )
    elif args.dataset == "pkgh-600":
        print("--loading from pkgh")
        root = "CHANGE PATH"
        train_dataset = pkgh(root_dir = root, split = 'train', ROI = True, balance = True, transform = transform )
        test_dataset = pkgh(root_dir = root, split = 'test', ROI = True, balance = False, transform = transform )
    
    elif args.dataset == "pkgh-410":
        print("--loading from pkgh")
        root = "CHANGE PATH"
        train_dataset = pkgh(root_dir = root, split = 'train', ROI = True, balance = True, transform = transform )
        test_dataset = pkgh(root_dir = root, split = 'test', ROI = True, balance = False, transform = transform )
    

    return train_dataset, test_dataset

