import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image
import pandas as pd
import os
import pickle

class pkgh(Dataset):
    def __init__(self, root_dir, size = 256, split = 'train', ROI = False, transform=None):
        print("initialization of PKGH")
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.Resize((size,size)),
            transforms.ToTensor(),
            ])

        if transform:
            self.transform = transform

        self.data = []

        pathologies = ['CP_TA','CP_TVA','CP_HP','CP_SSL','Normal']
    
        if split == 'train':
            dataset = os.path.join(self.root_dir,'train')
            for patho in pathologies:
                
                if patho != 'Normal':
                    # extracting all patches ROI + nonROI                              
                    all_images = os.listdir(os.path.join(self.root_dir,dataset,patho,'ROI'))
                    images = [os.path.join("ROI",img) for img in all_images if img[-3:]=='png']    
                    #print(f"{len(images)} patches from ROI for train for {patho}") 

                    # adding images not from ROI as well
                    if not ROI:
                        all_images = os.listdir(os.path.join(self.root_dir,dataset,patho,'nonROI'))
                        images= images + [os.path.join("nonROI",img) for img in all_images if img[-3:]=='png']
                        #print(f"{len(images)} patches from ROI and non ROI for train for {patho}") 
                else:
                    all_images = os.listdir(os.path.join(self.root_dir,dataset,patho))
                    images = [img for img in all_images if img[-3:]=='png']    
                    #print(f"{len(images)} patches from ROI and non ROI for train for {patho}") 


                # assigning labels
                for image in images:
                    path_to_image = os.path.join(self.root_dir,dataset,patho,image)
                    class_name = 'N'
                    # assign label
                    if image[:2] == 'TA':
                        class_name = 'TA'
                    elif image[:2] == 'HP':
                        class_name = 'HP'
                    elif image[:3] == 'TVA':
                        class_name = 'TVA'
                    elif image[:3] == 'SSL':
                        class_name = 'SSL'

                    self.data.append([path_to_image, class_name])

        if split == 'test':
            dataset = os.path.join(self.root_dir,'test')

            for patho in pathologies:

                if patho != 'Normal':
                    # extracting all patches ROI + nonROI                              
                    all_images = os.listdir(os.path.join(self.root_dir,dataset,patho,'ROI'))
                    images = [os.path.join("ROI",img) for img in all_images if img[-3:]=='png']    
                    #print(f"{len(images)} patches from ROI for train for {patho}") 

                    # adding images not from ROI as well
                    if not ROI:
                        all_images = os.listdir(os.path.join(self.root_dir,dataset,patho,'nonROI'))
                        images= images + [os.path.join("nonROI",img) for img in all_images if img[-3:]=='png']
                        #print(f"{len(images)} patches from ROI and non ROI for test for {patho}") 
                else:
                    all_images = os.listdir(os.path.join(self.root_dir,dataset,patho))
                    images = [img for img in all_images if img[-3:]=='png']    
                    #print(f"{len(images)} patches from ROI and non ROI for test for {patho}") 

                # assigning labels
                for image in images:
                    path_to_image = os.path.join(self.root_dir,dataset,patho,image)

                    class_name = 'N'
                    # assign label
                    if image[:2] == 'TA':
                        class_name = 'TA'
                    elif image[:2] == 'HP':
                        class_name = 'HP'
                    elif image[:3] == 'TVA':
                        class_name = 'TVA'
                    elif image[:3] == 'SSL':
                        class_name = 'SSL'
                    self.data.append([path_to_image, class_name])
                
                       

        print("We have", len(self.data)," in this dataset")
        self.class_map = {"N" : 0, "TA": 1, "TVA": 2, "HP": 3, "SSL": 4}
        self.img_dim = (size, size)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = Image.open(img_path)
        img = self.transform(img)

        class_id = torch.tensor([self.class_map[class_name]])        
        return img, class_id

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

# dataset = pkgh(root_dir = '/home/atlas-gp/tinyKGH/kgh-800', split = 'train', ROI = False )
# mean, std = get_mean_std(dataset)

# with open('/home/c_notton/KGH_project/experiments/BarlowTwins/out/pkgh_mean.pkl', 'wb') as f:
#     pickle.dump(mean, f)

# with open('/home/c_notton/KGH_project/experiments/BarlowTwins/out/pkgh_std.pkl', 'wb') as f:
#     pickle.dump(std, f)