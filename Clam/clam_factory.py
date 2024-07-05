# internal imports
from wsi_core.WholeSlideImage import WholeSlideImage
from wsi_core.wsi_utils import StitchCoords
from wsi_core.batch_process_utils import initialize_df
# other imports
import os
import numpy as np
import time
import argparse
import pdb
import pandas as pd
from tqdm import tqdm
from functools import partial

import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
from PIL import Image
import h5py
import openslide
import torchvision

from utils.file_utils import save_hdf5
from dataset_modules.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP, Whole_Slide_Bag_KGH
from models import get_encoder
from models.timm_wrapper import TimmCNNEncoder
from utils.constants import MODEL2CONSTANTS
from utils.transform_utils import get_eval_transforms
from torchvision import models
from torchvision.models.resnet import Bottleneck, ResNet



########## PATCH CREATION ##########

def stitching(file_path, wsi_object, downscale = 64):
	start = time.time()
	heatmap = StitchCoords(file_path, wsi_object, downscale=downscale, bg_color=(0,0,0), alpha=-1, draw_grid=False)
	total_time = time.time() - start
	
	return heatmap, total_time

def segment(WSI_object, seg_params = None, filter_params = None, mask_file = None):
	### Start Seg Timer
	start_time = time.time()
	# Use segmentation file
	if mask_file is not None:
		WSI_object.initSegmentation(mask_file)
	# Segment	
	else:
		WSI_object.segmentTissue(**seg_params, filter_params=filter_params)

	### Stop Seg Timers
	seg_time_elapsed = time.time() - start_time   
	return WSI_object, seg_time_elapsed

def patching(WSI_object, **kwargs):
	### Start Patch Timer
	start_time = time.time()

	# Patch
	file_path = WSI_object.process_contours(**kwargs)


	### Stop Patch Timer
	patch_time_elapsed = time.time() - start_time
	return file_path, patch_time_elapsed


def seg_and_patch(source, save_dir, patch_save_dir, mask_save_dir, stitch_save_dir, 
				  patch_size = 256, step_size = 256, 
				  seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
				  'keep_ids': 'none', 'exclude_ids': 'none'},
				  filter_params = {'a_t':100, 'a_h': 16, 'max_n_holes':8}, 
				  vis_params = {'vis_level': -1, 'line_thickness': 500},
				  patch_params = {'use_padding': True, 'contour_fn': 'four_pt'},
				  patch_level = 0,
				  use_default_params = False, 
				  seg = False, save_mask = True, 
				  stitch= False, 
				  patch = False, auto_skip=True, process_list = None):
	



    slides = sorted(os.listdir(source))
    slides = [slide for slide in slides if os.path.isfile(os.path.join(source, slide))]
    
    if process_list is None:
        df = initialize_df(slides, seg_params, filter_params, vis_params, patch_params)
    
    else:
        df = pd.read_csv(process_list)
        df = initialize_df(df, seg_params, filter_params, vis_params, patch_params)
    
    mask = df['process'] == 1
    process_stack = df[mask]

    total = len(process_stack)

    legacy_support = 'a' in df.keys()
    if legacy_support:
        print('detected legacy segmentation csv file, legacy support enabled')
        df = df.assign(**{'a_t': np.full((len(df)), int(filter_params['a_t']), dtype=np.uint32),
        'a_h': np.full((len(df)), int(filter_params['a_h']), dtype=np.uint32),
        'max_n_holes': np.full((len(df)), int(filter_params['max_n_holes']), dtype=np.uint32),
        'line_thickness': np.full((len(df)), int(vis_params['line_thickness']), dtype=np.uint32),
        'contour_fn': np.full((len(df)), patch_params['contour_fn'])})

    seg_times = 0.
    patch_times = 0.
    stitch_times = 0.

    for i in tqdm(range(total)):
        df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
        idx = process_stack.index[i]
        slide = process_stack.loc[idx, 'slide_id']
        print("\n\nprogress: {:.2f}, {}/{}".format(i/total, i, total))
        print('processing {}'.format(slide))
        
        df.loc[idx, 'process'] = 0
        slide_id, _ = os.path.splitext(slide)

        if auto_skip and os.path.isfile(os.path.join(patch_save_dir, slide_id + '.h5')):
            print('{} already exist in destination location, skipped'.format(slide_id))
            df.loc[idx, 'status'] = 'already_exist'
            continue

        # Inialize WSI
        full_path = os.path.join(source, slide)
        WSI_object = WholeSlideImage(full_path)

        if use_default_params:
            current_vis_params = vis_params.copy()
            current_filter_params = filter_params.copy()
            current_seg_params = seg_params.copy()
            current_patch_params = patch_params.copy()
            
        else:
            current_vis_params = {}
            current_filter_params = {}
            current_seg_params = {}
            current_patch_params = {}


            for key in vis_params.keys():
                if legacy_support and key == 'vis_level':
                    df.loc[idx, key] = -1
                current_vis_params.update({key: df.loc[idx, key]})

            for key in filter_params.keys():
                if legacy_support and key == 'a_t':
                    old_area = df.loc[idx, 'a']
                    seg_level = df.loc[idx, 'seg_level']
                    scale = WSI_object.level_downsamples[seg_level]
                    adjusted_area = int(old_area * (scale[0] * scale[1]) / (512 * 512))
                    current_filter_params.update({key: adjusted_area})
                    df.loc[idx, key] = adjusted_area
                current_filter_params.update({key: df.loc[idx, key]})

            for key in seg_params.keys():
                if legacy_support and key == 'seg_level':
                    df.loc[idx, key] = -1
                current_seg_params.update({key: df.loc[idx, key]})

            for key in patch_params.keys():
                current_patch_params.update({key: df.loc[idx, key]})

        if current_vis_params['vis_level'] < 0:
            if len(WSI_object.level_dim) == 1:
                current_vis_params['vis_level'] = 0
            
            else:	
                wsi = WSI_object.getOpenSlide()
                best_level = wsi.get_best_level_for_downsample(64)
                current_vis_params['vis_level'] = best_level

        if current_seg_params['seg_level'] < 0:
            if len(WSI_object.level_dim) == 1:
                current_seg_params['seg_level'] = 0
            
            else:
                wsi = WSI_object.getOpenSlide()
                best_level = wsi.get_best_level_for_downsample(64)
                current_seg_params['seg_level'] = best_level

        keep_ids = str(current_seg_params['keep_ids'])
        if keep_ids != 'none' and len(keep_ids) > 0:
            str_ids = current_seg_params['keep_ids']
            current_seg_params['keep_ids'] = np.array(str_ids.split(',')).astype(int)
        else:
            current_seg_params['keep_ids'] = []

        exclude_ids = str(current_seg_params['exclude_ids'])
        if exclude_ids != 'none' and len(exclude_ids) > 0:
            str_ids = current_seg_params['exclude_ids']
            current_seg_params['exclude_ids'] = np.array(str_ids.split(',')).astype(int)
        else:
            current_seg_params['exclude_ids'] = []

        w, h = WSI_object.level_dim[current_seg_params['seg_level']] 
        if w * h > 1e8:
            print('level_dim {} x {} is likely too large for successful segmentation, aborting'.format(w, h))
            df.loc[idx, 'status'] = 'failed_seg'
            continue

        df.loc[idx, 'vis_level'] = current_vis_params['vis_level']
        df.loc[idx, 'seg_level'] = current_seg_params['seg_level']


        seg_time_elapsed = -1
        if seg:
            WSI_object, seg_time_elapsed = segment(WSI_object, current_seg_params, current_filter_params) 

        if save_mask:
            mask = WSI_object.visWSI(**current_vis_params)
            mask_path = os.path.join(mask_save_dir, slide_id+'.jpg')
            mask.save(mask_path)

        patch_time_elapsed = -1 # Default time
        if patch:
            current_patch_params.update({'patch_level': patch_level, 'patch_size': patch_size, 'step_size': step_size, 
                                            'save_path': patch_save_dir})
            file_path, patch_time_elapsed = patching(WSI_object = WSI_object,  **current_patch_params,)
        
        stitch_time_elapsed = -1
        if stitch:
            file_path = os.path.join(patch_save_dir, slide_id+'.h5')
            if os.path.isfile(file_path):
                heatmap, stitch_time_elapsed = stitching(file_path, WSI_object, downscale=64)
                stitch_path = os.path.join(stitch_save_dir, slide_id+'.jpg')
                heatmap.save(stitch_path)

        print("segmentation took {} seconds".format(seg_time_elapsed))
        print("patching took {} seconds".format(patch_time_elapsed))
        print("stitching took {} seconds".format(stitch_time_elapsed))
        df.loc[idx, 'status'] = 'processed'

        seg_times += seg_time_elapsed
        patch_times += patch_time_elapsed
        stitch_times += stitch_time_elapsed

    seg_times /= total
    patch_times /= total
    stitch_times /= total

    df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
    print("average segmentation time in s per slide: {}".format(seg_times))
    print("average patching time in s per slide: {}".format(patch_times))
    print("average stiching time in s per slide: {}".format(stitch_times))
        
    return seg_times, patch_times


def create_patches(source, save_dir, step_size=256, patch_size=256, patch=False, seg=False, stitch=False, no_auto_skip=True, preset=None, patch_level=0, process_list =None):

	patch_save_dir = os.path.join( save_dir, 'patches')
	mask_save_dir = os.path.join( save_dir, 'masks')
	stitch_save_dir = os.path.join( save_dir, 'stitches')

	if  process_list:
		process_list = os.path.join( save_dir,  process_list)

	else:
		process_list = None

	print('source: ',  source)
	print('patch_save_dir: ', patch_save_dir)
	print('mask_save_dir: ', mask_save_dir)
	print('stitch_save_dir: ', stitch_save_dir)
	
	directories = {'source':  source, 
				   'save_dir':  save_dir,
				   'patch_save_dir': patch_save_dir, 
				   'mask_save_dir' : mask_save_dir, 
				   'stitch_save_dir': stitch_save_dir} 

	for key, val in directories.items():
		print("{} : {}".format(key, val))
		if key not in ['source']:
			os.makedirs(val, exist_ok=True)

	seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
				  'keep_ids': 'none', 'exclude_ids': 'none'}
	filter_params = {'a_t':100, 'a_h': 16, 'max_n_holes':8}
	vis_params = {'vis_level': -1, 'line_thickness': 250}
	patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}

	if  preset:
		preset_df = pd.read_csv(os.path.join('presets',  preset))
		for key in seg_params.keys():
			seg_params[key] = preset_df.loc[0, key]

		for key in filter_params.keys():
			filter_params[key] = preset_df.loc[0, key]

		for key in vis_params.keys():
			vis_params[key] = preset_df.loc[0, key]

		for key in patch_params.keys():
			patch_params[key] = preset_df.loc[0, key]
	
	parameters = {'seg_params': seg_params,
				  'filter_params': filter_params,
	 			  'patch_params': patch_params,
				  'vis_params': vis_params}

	print(parameters)

	seg_times, patch_times = seg_and_patch(**directories, **parameters,
											patch_size =  patch_size, step_size= step_size, 
											seg =  seg,  use_default_params=False, save_mask = True, 
											stitch=  stitch,
											patch_level= patch_level, patch =  patch,
											process_list = process_list, auto_skip= no_auto_skip)

########## FEATURE EXTRACTION ##########

# configuration of the encoder
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


def compute_w_loader(output_path, device, loader, model, verbose = 0):
    """
    args:
        output_path: directory to save computed features (.h5 file)
        model: pytorch model
        verbose: level of feedback
    """
    if verbose > 0:
        print(f'processing a total of {len(loader)} batches'.format(len(loader)))

    mode = 'w'
    for count, data in enumerate(tqdm(loader)):
        with torch.inference_mode():	
            batch = data['img']
            coords = data['coord'][0]
            coords = coords.numpy().astype(np.int32)
            batch = batch.to(device, non_blocking=True)
            
            ### might need to change this line with another encoder
            features = model(batch)
            print(f"Features of dimension {features.shape}")
            features = features.cpu().numpy().astype(np.float32)

            asset_dict = {'features': features, 'coords': coords}
            save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
            print(f"Saved to {output_path}")
            mode = 'a'

    return output_path

# define backbone model based on Barlow Twins model 
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

# define backbone model based on Barlow Twins model with optimizer, epochs...
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

# useful functions to use the benchmark weights
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
        weights = "path/to/bt_rn50_ep200.torch"
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

# get encoder based on the one we want
def get_encoder(model_name, target_img_size=224):
    print('loading model checkpoint')
    if model_name == 'resnet50_trunc':
        model = TimmCNNEncoder() 

    #### pkgh 1.25
    if model_name == 'resnet50-pathBT-1-25':
        #weights = "/home/huron/Documents/Clam/CLAM/models/BT-rn50/pkgh-1-25/resnet50-pkgh-209-100-pathBT-2.pth"
        weights = "/home/huron/Documents/Clam/CLAM/models/BT-rn50/pkgh-1-25/checkpoint-resnet50-pkgh-2-512-50.pth"
        #model = make_encoder_from_barlow('resnet50', 2048, weights)
        model = make_encoder_from_barlow_checkpoints('resnet50', 8192, weights)

    if model_name == 'resnet50-basicBT-1-25':
        weights = "/home/huron/Documents/Clam/CLAM/models/BT-rn50/pkgh-1-25/resnet50-pkgh-213-100-basic.pth"
        model = make_encoder_from_barlow('resnet50', 8192, weights)

    if model_name == 'resnet50-imBT-1-25':
        weights = "/home/huron/Documents/Clam/CLAM/models/BT-rn50/pkgh-1-25/resnet50-pkgh-212-100-imagenet.pth"
        model = make_encoder_from_barlow('resnet50', 8192, weights)

    if model_name == 'swin-BT-1-25':
        weights = "/home/huron/Documents/Clam/CLAM/models/BT-rn50/pkgh-1-25/swin_t-pkgh-7-100-swinPath.pth" #changed
        model = make_encoder_from_barlow('swin_t', 8192, weights)

    if model_name == 'resnet_sup-1.25':
        weights = "/home/huron/Documents/Supervised/results/41/Trained_Model10.pt"
        weights = torch.load(weights)
        model  = models.__dict__['resnet50'](weights=None)
        num_ftrs = model.fc.in_features
        model.load_state_dict(weights, strict = True)
        model.fc = nn.Identity()

    #### pkgh-800
    if model_name == 'resnet50-pathBT-800':
        weights = "/home/huron/Documents/Clam/CLAM/models/BT-rn50/pkgh-800/resnet50-pkgh-800-206-pathBT.pth" #not latest one
        model = make_encoder_from_barlow('resnet50', 2048, weights)

    if model_name == 'resnet50-basicBT-800':
        weights = "/home/huron/Documents/Clam/CLAM/models/BT-rn50/pkgh-800/resnet50-pkgh-800-210-basicBT.pth"
        model = make_encoder_from_barlow('resnet50', 8192, weights)

    if model_name == 'resnet50-imBT-800':
        weights = "/home/huron/Documents/Clam/CLAM/models/BT-rn50/pkgh-800/resnet50-pkgh-800-211-imBT.pth"
        model = make_encoder_from_barlow('resnet50', 8192, weights)

    if model_name == 'swin-BT-800':
        weights = "/home/huron/Documents/Clam/CLAM/models/BT-rn50/pkgh-800/swin_t-pkgh-800-8-512-gpus4-100.pth" #changed
        model = make_encoder_from_barlow('swin_t', 8192, weights)
    
    if model_name == 'resnet_sup-800':
        weights = "/home/huron/Documents/Supervised/results/42/Trained_Model10.pt"
        weights = torch.load(weights)
        model  = models.__dict__['resnet50'](weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs,5)
        model.load_state_dict(weights, strict = True)
        model.fc = nn.Identity()

    #### pkgh-600
    if model_name == 'resnet50-pathBT-600':
        #weights = "/home/huron/Documents/Clam/CLAM/models/BT-rn50/pkgh-600/resnet50-pkgh-600-1-100-pathBT.pth"
        weights = "/home/huron/Documents/Clam/CLAM/models/BT-rn50/pkgh-600/checkpoint-resnet50-pkgh-600-1-512-20.pth"
        #model = make_encoder_from_barlow('resnet50', 8192, weights)
        model = make_encoder_from_barlow_checkpoints('resnet50', 8192, weights)

    if model_name == 'resnet50-basicBT-600':
        weights = "/home/huron/Documents/Clam/CLAM/models/BT-rn50/pkgh-600/resnet50-pkgh-600-3-100-basic.pth"
        model = make_encoder_from_barlow('resnet50', 8192, weights)

    if model_name == 'resnet50-imBT-600':
        weights = "/home/huron/Documents/Clam/CLAM/models/BT-rn50/pkgh-600/resnet50-pkgh-600-4-100-imBT.pth" # changed
        model = make_encoder_from_barlow('resnet50', 8192, weights)

    if model_name == 'swin-BT-600':
        weights = "/home/huron/Documents/Clam/CLAM/models/BT-rn50/pkgh-600/swin_t-pkgh-600-5-100-swinPath.pth" # changed
        model = make_encoder_from_barlow('swin_t', 8192, weights)
    
    if model_name == 'resnet_sup-600':
        weights = "/home/huron/Documents/Supervised/results/42/Trained_Model10.pt"
        weights = torch.load(weights)
        model  = models.__dict__['resnet50'](weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs,5)
        model.load_state_dict(weights, strict = True)
        model.fc = nn.Identity()

     #### pkgh-410
    if model_name == 'resnet50-pathBT-410':
        weights = "/home/huron/Documents/Clam/CLAM/models/BT-rn50/pkgh-410/resnet50-pkgh-410-10-pathBT.pth" #changed
        model = make_encoder_from_barlow('resnet50', 8192, weights)
    if model_name == 'resnet50-imBT-410':
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

    #### benchmark weights
    if model_name == 'resnet50_bench':
        model = resnet50(pretrained=True)
        model = resnet_benchmark(model)
        model.fc = nn.Identity()


    if model_name == 'resnet50_BT_800':
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

    # if model_name == 'uni_v1':
    #     HAS_UNI, UNI_CKPT_PATH = has_UNI()
    #     assert HAS_UNI, 'UNI is not available'
    #     model = timm.create_model("vit_large_patch16_224",
    #                         init_values=1e-5, 
    #                         num_classes=0, 
    #                         dynamic_img_size=True)
    #     model.load_state_dict(torch.load(UNI_CKPT_PATH, map_location="cpu"), strict=True)

	



    # if model_name == 'conch_v1':
    #     HAS_CONCH, CONCH_CKPT_PATH = has_CONCH()
    #     assert HAS_CONCH, 'CONCH is not available'
    #     from conch.open_clip_custom import create_model_from_pretrained
    #     model, _ = create_model_from_pretrained("conch_ViT-B-16", CONCH_CKPT_PATH)
    #     model.forward = partial(model.encode_image, proj_contrast=False, normalize=False)
	
	
	

	
    # else:
    #     raise NotImplementedError('model {} not implemented'.format(model_name))
    
    #print(model)
	#constants = MODEL2CONSTANTS[model_name]
    mean = [0.8816, 0.7241, 0.8087]
    std = [0.1132, 0.2006, 0.1436]
    img_transforms = get_eval_transforms(mean=mean,
                                            std=std,
                                            target_img_size = target_img_size)

    return model, img_transforms



def feature_extraction(device, data_h5_dir = None, data_slide_dir = None, slide_ext = '.tif', csv_path = None, feat_dir = None, model_name = 'resnet50_trunc', batch_size = 256, no_auto_skip = False, target_patch_size = 224):
    print('initializing dataset')
    csv_path =   csv_path
    if csv_path is None:
        raise NotImplementedError

    bags_dataset = Dataset_All_Bags(csv_path)
    print(f"bags_dataset = {bags_dataset}")

    os.makedirs(  feat_dir, exist_ok=True)
    os.makedirs(os.path.join(  feat_dir, 'pt_files'), exist_ok=True)
    os.makedirs(os.path.join(  feat_dir, 'h5_files'), exist_ok=True)
    dest_files = os.listdir(os.path.join(  feat_dir, 'pt_files'))

    model, img_transforms = get_encoder(  model_name, target_img_size=  target_patch_size)
            
    _ = model.eval()
    model = model.to(device)
    total = len(bags_dataset)

    loader_kwargs = {'num_workers': 8, 'pin_memory': True} if device.type == "cuda" else {}
    csv_path = '/mnt/c8f0e3a8-8cda-42ea-b586-9c2ac93e02a4/Clam/patches-410/KGH_slides.csv'
    slide_data = pd.read_csv(csv_path)
    wsis = list(slide_data['slide_id'])
    print(f"All wsis = {wsis}")
    #adapted to KGH_WSIs folder
    for bag_candidate_idx in tqdm(wsis):

        slide_id = bag_candidate_idx
        bag_name = slide_id+'.h5'
        h5_file_path = os.path.join(  data_h5_dir, 'patches', bag_name)
        slide_file_path = os.path.join(  data_slide_dir, slide_id+'.tif')
        print(f"----slide_file_path = {slide_file_path}")
        print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
        print(slide_id)

        if not   no_auto_skip and slide_id+'.pt' in dest_files:
            print('skipped {}'.format(slide_id))
            continue 

        output_path = os.path.join(feat_dir, 'h5_files', bag_name)
        time_start = time.time()
        wsi = openslide.open_slide(slide_file_path)
        dataset = Whole_Slide_Bag_FP(file_path=h5_file_path, 
                                        wsi=wsi, 
                                        img_transforms=img_transforms)
        print(f'--Name of slide {os.path.basename(slide_file_path)[:-4]}')
        print(f'--dataset of length {len(dataset)}')
        loader = DataLoader(dataset=dataset, batch_size=8, **loader_kwargs)
        output_file_path = compute_w_loader(output_path, device, loader = loader, model = model, verbose = 1)

        time_elapsed = time.time() - time_start
        print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))

        with h5py.File(output_file_path, "r") as file:
            features = file['features'][:]
            print('features size: ', features.shape)
            print('coordinates size: ', file['coords'].shape)
            features = torch.from_numpy(features)
            torch.save(features, os.path.join(  feat_dir, 'pt_files', slide_id +'.pt'))
	
# create CSV with cases id, slides id and labels
def make_csv(feat_dir = None):
    slide_ids = os.listdir(os.path.join(feat_dir,'pt_files'))
    slide_ids = [slide[:-3] for slide in slide_ids]
    labels = []
    for slide in slide_ids:
        if slide[0] == 'N':
            labels.append('N')
        elif slide[:2] == 'HP':
            labels.append('HP')
        elif slide[:2] == 'TA':
            labels.append('TA')
        elif slide[:3] == 'TVA':
            labels.append('TVA')
        elif slide[:3] == 'SSL':
            labels.append('SSL')

    df = pd.DataFrame([slide_ids,slide_ids,labels]).T
    df.columns = ['case_id','slide_id','label']
    df.to_csv(os.path.join(feat_dir,'KGH_slides.csv'))
    print(df)
    print(f"Saved cases, ids and labels for {feat_dir} to {os.path.join(feat_dir,'KGH_slides.csv')}")