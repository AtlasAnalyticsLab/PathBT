import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image
import pandas as pd
import os

#example of root /home/ubuntu/Documents/KGH_dataset/PKGH/patch_tinyKGH_320 then TVA,TA,SSL,HP,Normal then all slides in it
class pkgh_1fov(Dataset):
    def __init__(self, root_dir, transform=None):
        print("initialization of PKGH")
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            ])

        if transform:
            self.transform = transform

        self.data = []

        pathologies = os.listdir(self.root_dir) 
        for patho in pathologies:
            if patho !="Normal":
                wsis = os.listdir(os.path.join(self.root_dir,patho))
                
                for wsi in wsis:
                    if wsi[-3:]!='csv': 
                        all_files = os.listdir(os.path.join(self.root_dir,patho,wsi))  
                        rois = [folders for folders in all_files if len(folders)<=2]
                        #print(rois)
                        for roi in rois:
                            all_images = os.listdir(os.path.join(self.root_dir,patho,wsi,roi))  
                            images = [img for img in all_images if img[-3:]=='png']
                            for image in images:
                                path_to_image = os.path.join(self.root_dir,patho,wsi,roi,image)
                                #pil_image = Image.open(path_to_image)
                    
                                # assign label
                                if image[:2] == 'TA':
                                    class_name = 'TA'
                                elif image[:2] == 'HP':
                                    class_name = 'HP'
                                elif image[:3] == 'TVA':
                                    class_name = 'TVA'
                                elif image[:3] == 'SSL':
                                    class_name = 'SSL'

                                #print("For image ",image, " label is ",class_name)
                                self.data.append([path_to_image, class_name])

            elif patho == "Normal":
                #print("dealing with normal")
                wsis = os.listdir(os.path.join(self.root_dir,patho))
                #print("WSIs in ",patho," folder are ",wsis)
                for wsi in wsis:
                    if wsi[-3:]!='csv': 
                        patch_folder = wsi+"_tissue"
                        
                        all_images = os.listdir(os.path.join(self.root_dir,patho,wsi,patch_folder))  
                        images = [img for img in all_images if img[-3:]=='png']
                        for image in images:
                            path_to_image = os.path.join(self.root_dir,patho,wsi,patch_folder, image)
                        
                
                            # assign label
                            class_name = "N"

                            #print("For image ",image, " label is ",class_name)          
                            self.data.append([path_to_image, class_name])

        print("We have", len(self.data)," in this dataset")
        self.class_map = {"N" : 0, "TA": 1, "TVA": 2, "HP": 3, "SSL": 4}
        self.img_dim = (224, 224)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = Image.open(img_path)
        img = self.transform(img)

        class_id = torch.tensor([self.class_map[class_name]])        
        return img, class_id

                


