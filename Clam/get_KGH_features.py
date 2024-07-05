from clam_factory import feature_extraction

import torch

# basicBT

DEVICE = device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
DATA_H5 = "/mnt/c8f0e3a8-8cda-42ea-b586-9c2ac93e02a4/Clam/patches-410"
DATA_SLIDE = "/home/huron/Documents/Datasets/KGH_WSIs/train/"
CSV = "/mnt/c8f0e3a8-8cda-42ea-b586-9c2ac93e02a4/Clam/patches-410/process_list_autogen.csv"
TARGET = "/mnt/c8f0e3a8-8cda-42ea-b586-9c2ac93e02a4/Clam/features/pkgh-410/features-sup"
BATCH_SIZE = 64
SLIDE_EXT = '.tif'
MODEL = 'resnet_sup-410'
NO_AUTO_SKIP = False
TARGET_PATCH_SIZE = 224

feature_extraction(device = DEVICE, data_h5_dir = DATA_H5, data_slide_dir = DATA_SLIDE, slide_ext = SLIDE_EXT, csv_path = CSV, feat_dir = TARGET, model_name = MODEL, batch_size = BATCH_SIZE, no_auto_skip = NO_AUTO_SKIP, target_patch_size = TARGET_PATCH_SIZE)