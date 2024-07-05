from clam_factory import create_patches

SOURCE = '/home/huron/Documents/Datasets/KGH_WSIs/test/'
TARGET = '/mnt/c8f0e3a8-8cda-42ea-b586-9c2ac93e02a4/Clam/patches-test-600'
STEP_SIZE = 1025
PATCH_SIZE = 1025 #fov 410
PATCH = True
SEG = True
STITCH = True
NO_AUTO_SKIP = True
PRESET = None
PATCH_LEVEL = 0
PROCESS_LIST = None


create_patches(source = SOURCE, save_dir = TARGET, step_size = STEP_SIZE, patch_size = PATCH_SIZE, patch = PATCH, seg = SEG, stitch = STITCH, no_auto_skip = NO_AUTO_SKIP, preset = PRESET, patch_level = PATCH_LEVEL, process_list = PROCESS_LIST)
