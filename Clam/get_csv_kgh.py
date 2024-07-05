from clam_factory import make_csv

DATA_H5 = "/mnt/c8f0e3a8-8cda-42ea-b586-9c2ac93e02a4/Clam/patches-test-410"
DATA_SLIDE = "/home/huron/Documents/Datasets/KGH_WSIs/train"
CSV = "/mnt/c8f0e3a8-8cda-42ea-b586-9c2ac93e02a4/Clam/patches-410/process_list_autogen.csv"
TARGET = "/mnt/c8f0e3a8-8cda-42ea-b586-9c2ac93e02a4/Clam/features/pkgh-410/features-sup"
make_csv(TARGET)