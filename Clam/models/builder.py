import os
from functools import partial
import timm
from .timm_wrapper import TimmCNNEncoder
import torch
from utils.constants import MODEL2CONSTANTS
from utils.transform_utils import get_eval_transforms
import torchvision
import torch.nn as nn
from torchvision.models.resnet import Bottleneck, ResNet
from torchvision import models



def has_CONCH():
    HAS_CONCH = False
    CONCH_CKPT_PATH = ''
    # check if CONCH_CKPT_PATH is set and conch is installed, catch exception if not
    try:
        from conch.open_clip_custom import create_model_from_pretrained
        # check if CONCH_CKPT_PATH is set
        if 'CONCH_CKPT_PATH' not in os.environ:
            raise ValueError('CONCH_CKPT_PATH not set')
        HAS_CONCH = True
        CONCH_CKPT_PATH = os.environ['CONCH_CKPT_PATH']
    except Exception as e:
        print(e)
        print('CONCH not installed or CONCH_CKPT_PATH not set')
    return HAS_CONCH, CONCH_CKPT_PATH

def has_UNI():
    HAS_UNI = False
    UNI_CKPT_PATH = ''
    # check if UNI_CKPT_PATH is set, catch exception if not
    try:
        # check if UNI_CKPT_PATH is set
        if 'UNI_CKPT_PATH' not in os.environ:
            raise ValueError('UNI_CKPT_PATH not set')
        HAS_UNI = True
        UNI_CKPT_PATH = os.environ['UNI_CKPT_PATH']
    except Exception as e:
        print(e)
    return HAS_UNI, UNI_CKPT_PATH
def backbone_eval(encoder):
    if encoder == "resnet50":
        backbone = torchvision.models.resnet50()
        backbone.fc = nn.Identity()

    if encoder == "resnet18":
        backbone = torchvision.models.resnet18()

    if encoder == "resnet101":
        backbone = torchvision.models.resnet101()

    ## Swin ##
    if encoder == "swin_t":
        backbone = torchvision.models.swin_t()
        backbone.head = nn.Identity()

    return backbone

class BarlowTwins(nn.Module):
    def __init__(self, encoder, proj):
        super().__init__()
        projector = f"{proj}-{proj}-{proj}"
        ## RESNET ##
        if encoder == "resnet50":
            self.backbone = torchvision.models.resnet50(zero_init_residual=True)
            self.backbone.fc = nn.Identity()
            # projector
            sizes = [2048] + list(map(int, projector.split('-')))
        if encoder == "resnet18":
            self.backbone = torchvision.models.resnet18(zero_init_residual=True, weights = None)
            self.backbone.fc = nn.Identity()
            # projector
            sizes = [512] + list(map(int, projector.split('-')))
        if encoder == "resnet101":
            self.backbone = torchvision.models.resnet101(zero_init_residual=True, weights = None)
            self.backbone.fc = nn.Identity()
            # projector
            sizes = [2048] + list(map(int, projector.split('-')))
    
        # swin
        if encoder == "swin_t":
            self.backbone = torchvision.models.swin_t()
            self.backbone.head = nn.Identity()
            # projector
            sizes = [768] + list(map(int, projector.split('-')))
        
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
def make_encoder_from_barlow(encoder, proj, weights):
    model = BarlowTwins(encoder,proj)
    state_dict = torch.load(weights)
    state_dict = {k[7:]:v for k, v in state_dict.items()}
    print(f"State dict = {[k for k,v in state_dict.items()]}")
    print(f"--MODEL = {model}")
    model.load_state_dict(state_dict, strict = True)
    state_dict = model.backbone.state_dict()
    print(f"new state dict = {[k for k,v in state_dict.items()]}")
    model = backbone_eval(encoder)
    model.load_state_dict(state_dict, strict = True)

    return model
def make_encoder_from_barlow_checkpoints(encoder, proj, weights):
    model = BarlowTwins(encoder,proj)
    state_dict = torch.load(weights)
    state_dict = state_dict['model']
    state_dict = {k[7:]:v for k, v in state_dict.items()}
    print(f"State dict = {[k for k,v in state_dict.items()]}")
    print(f"--MODEL = {model}")
    model.load_state_dict(state_dict, strict = True)
    state_dict = model.backbone.state_dict()
    print(f"new state dict = {[k for k,v in state_dict.items()]}")
    model = backbone_eval(encoder)
    model.load_state_dict(state_dict, strict = True)

    return model
class ResNetTrunk(ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.fc  # remove FC layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

def resnet50(pretrained, **kwargs):
    model = ResNetTrunk(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        weights = "/home/huron/Documents/Clam/CLAM/models/BT-rn50/bt_rn50_ep200.torch"
        verbose = model.load_state_dict(torch.load(weights))
        print(verbose)
    return model

class resnet_benchmark(nn.Module):
    def __init__(self,model,num_classes = 5):
        super(resnet_benchmark, self).__init__()
        self.backbone = model 
        self.fc = nn.Linear(in_features=100352, out_features=num_classes, bias=True)

    def forward(self, x):
        #print(f"input of shape {x.shape}")
        x = self.backbone(x)
        #print(f"before flatten {x.shape}")
        x = torch.flatten(x, 1)
        #print(f"after flatten {x.shape}")
        x = self.fc(x)

        return x
        
def get_encoder(model_name, target_img_size=224):
    print('loading model checkpoint')
    if model_name == 'resnet50_trunc':
        model = TimmCNNEncoder()
    elif model_name == 'resnet50_BT_800':
        weights = "/home/huron/Documents/Clam/CLAM/models/final-exp61-resnet50-pkgh-64.pth"
        model = torchvision.models.resnet50()
        num_ftrs = model.fc.in_features
        print(f"Number of features before fc = {num_ftrs}")
        model.fc = nn.Linear(num_ftrs, 5)

        state_dict = torch.load(weights)
        new_state_dict = {k[7:]:v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=True)
        model.fc = nn.Identity()
        print("Loaded ResNet50 model trained on pkgh")
    elif model_name == 'resnet50-pathBT':
        weights = "/home/huron/Documents/Clam/CLAM/models/BT-rn50/resnet50-pkgh-209-100-pathBT-2.pth"
        model = make_encoder_from_barlow('resnet50', 2048, weights)

    elif model_name == 'resnet50-basicBT':
        weights = "/home/huron/Documents/Clam/CLAM/models/BT-rn50/resnet50-pkgh-213-100-basic.pth"
        model = make_encoder_from_barlow('resnet50', 8192, weights)

    elif model_name == 'resnet50-imBT':
        weights = "/home/huron/Documents/Clam/CLAM/models/BT-rn50/resnet50-pkgh-212-100-imagenet.pth"
        model = make_encoder_from_barlow('resnet50', 8192, weights)

    elif model_name == 'swin-BT':
        #weights = "/home/huron/Documents/Clam/CLAM/models/BT-rn50/swin_t-pkgh-207-512-gpus2-100.pth"
        weights = "/home/huron/Documents/Clam/CLAM/models/BT-rn50/pkgh-1-25/swin_t-pkgh-7-100-swinPath.pth"
        model = make_encoder_from_barlow('swin_t', 2048, weights)
    
    elif model_name == 'resnet50_bench':
        
        model = resnet50(pretrained=True)
        model = resnet_benchmark(model)
        model.fc = nn.Identity()
    
    # pkgh-800
    elif model_name == 'resnet50-pathBT-800':
        weights = "/home/huron/Documents/Clam/CLAM/models/BT-rn50/pkgh-800/resnet50-pkgh-800-206-pathBT.pth" #not latest one
        model = make_encoder_from_barlow('resnet50', 2048, weights)

    elif model_name == 'resnet50-basicBT-800':
        weights = "/home/huron/Documents/Clam/CLAM/models/BT-rn50/pkgh-800/resnet50-pkgh-800-210-basicBT.pth"
        model = make_encoder_from_barlow('resnet50', 8192, weights)

    elif model_name == 'resnet50-imBT-800':
        weights = "/home/huron/Documents/Clam/CLAM/models/BT-rn50/pkgh-800/resnet50-pkgh-800-211-imBT.pth"
        model = make_encoder_from_barlow('resnet50', 8192, weights)

    elif model_name == 'swin-BT-800':
        weights = "/home/huron/Documents/Clam/CLAM/models/BT-rn50/pkgh-800/swin_t-pkgh-800-208-512-gpus2-100.pth" #to change
        model = make_encoder_from_barlow('swin_t', 2048, weights)
    
    #### pkgh-600
    elif model_name == 'resnet50-pathBT-600':
        #weights = "/home/huron/Documents/Clam/CLAM/models/BT-rn50/pkgh-600/resnet50-pkgh-600-1-100-pathBT.pth"
        weights = "/home/huron/Documents/Clam/CLAM/models/BT-rn50/pkgh-600/checkpoint-resnet50-pkgh-600-1-512-20.pth"
        #model = make_encoder_from_barlow('resnet50', 8192, weights)
        model = make_encoder_from_barlow_checkpoints('resnet50', 8192, weights)

    elif model_name == 'resnet50-basicBT-600':
        weights = "/home/huron/Documents/Clam/CLAM/models/BT-rn50/pkgh-600/resnet50-pkgh-600-3-100-basic.pth"
        model = make_encoder_from_barlow('resnet50', 8192, weights)

    elif model_name == 'resnet50-imBT-600':
        weights = "/home/huron/Documents/Clam/CLAM/models/BT-rn50/pkgh-600/resnet50-pkgh-600-4-100-imBT.pth" # changed
        model = make_encoder_from_barlow('resnet50', 8192, weights)

    elif model_name == 'swin-BT-600':
        weights = "/home/huron/Documents/Clam/CLAM/models/BT-rn50/pkgh-600/swin_t-pkgh-600-5-100-swinPath.pth" # changed
        model = make_encoder_from_barlow('swin_t', 8192, weights)
    

    #### pkgh-410
    elif model_name == 'resnet50-pathBT-410':
        weights = "/home/huron/Documents/Clam/CLAM/models/BT-rn50/pkgh-410/resnet50-pkgh-410-10-pathBT.pth" #changed
        model = make_encoder_from_barlow('resnet50', 8192, weights)
        
    elif model_name == 'resnet50-imBT-410':
        weights = "/home/huron/Documents/Clam/CLAM/models/BT-rn50/pkgh-410/resnet50-pkgh-410-11-imBT.pth" # changed
        model = make_encoder_from_barlow('resnet50', 8192, weights)
    
    if model_name == 'resnet50-basicBT-410':
        weights = "/home/huron/Documents/Clam/CLAM/models/BT-rn50/pkgh-410/resnet50-pkgh-410-16-basicBT.pth" # to change
        model = make_encoder_from_barlow('resnet50', 8192, weights)

    if model_name == 'resnet50-swin-410':
        weights = "/home/huron/Documents/Clam/CLAM/models/BT-rn50/pkgh-410/swin_t-pkgh-410-15-swinBT.pth" # to change
        model = make_encoder_from_barlow('swin_t', 8192, weights)

    if model_name == 'resnet_sup-410':
        weights = "/home/huron/Documents/Supervised/results/101/Trained_Model20.pt"
        weights = torch.load(weights)
        model  = models.__dict__['resnet50'](weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs,5)
        model.load_state_dict(weights, strict = True)
        model.fc = nn.Identity()

    elif model_name == 'uni_v1':
        HAS_UNI, UNI_CKPT_PATH = has_UNI()
        assert HAS_UNI, 'UNI is not available'
        model = timm.create_model("vit_large_patch16_224",
                            init_values=1e-5, 
                            num_classes=0, 
                            dynamic_img_size=True)
        model.load_state_dict(torch.load(UNI_CKPT_PATH, map_location="cpu"), strict=True)
    elif model_name == 'conch_v1':
        HAS_CONCH, CONCH_CKPT_PATH = has_CONCH()
        assert HAS_CONCH, 'CONCH is not available'
        from conch.open_clip_custom import create_model_from_pretrained
        model, _ = create_model_from_pretrained("conch_ViT-B-16", CONCH_CKPT_PATH)
        model.forward = partial(model.encode_image, proj_contrast=False, normalize=False)
    else:
        raise NotImplementedError('model {} not implemented'.format(model_name))
    
    print(model)
    # constants = MODEL2CONSTANTS[model_name]
    # img_transforms = get_eval_transforms(mean=constants['mean'],
    #                                      std=constants['std'],
    #                                      target_img_size = target_img_size)
    mean = [0.8816, 0.7241, 0.8087]
    std = [0.1132, 0.2006, 0.1436]
    img_transforms = get_eval_transforms(mean=mean,
                                            std=std,
                                            target_img_size = target_img_size)

    return model, img_transforms