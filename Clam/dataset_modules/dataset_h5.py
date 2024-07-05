import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image
import h5py
import os 
from torch.utils.data import DataLoader
import re
from tqdm import tqdm

class Whole_Slide_Bag(Dataset):
	def __init__(self,
		file_path,
		img_transforms=None):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			roi_transforms (callable, optional): Optional transform to be applied on a sample
		"""
		self.roi_transforms = img_transforms
		self.file_path = file_path

		with h5py.File(self.file_path, "r") as f:
			dset = f['imgs']
			self.length = len(dset)

		self.summary()
			
	def __len__(self):
		return self.length

	def summary(self):
		with h5py.File(self.file_path, "r") as hdf5_file:
			dset = hdf5_file['imgs']
			for name, value in dset.attrs.items():
				print(name, value)

		print('transformations:', self.roi_transforms)

	def __getitem__(self, idx):
		with h5py.File(self.file_path,'r') as hdf5_file:
			img = hdf5_file['imgs'][idx]
			coord = hdf5_file['coords'][idx]
		
		img = Image.fromarray(img)
		img = self.roi_transforms(img)
		return {'img': img, 'coord': coord}

class Whole_Slide_Bag_FP(Dataset):
	def __init__(self,
		file_path,
		wsi,
		img_transforms=None):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			img_transforms (callable, optional): Optional transform to be applied on a sample
		"""
		self.wsi = wsi
		self.roi_transforms = img_transforms

		self.file_path = file_path

		with h5py.File(self.file_path, "r") as f:
			dset = f['coords']
			self.patch_level = f['coords'].attrs['patch_level']
			self.patch_size = f['coords'].attrs['patch_size']
			self.length = len(dset)
			
		self.summary()
			
	def __len__(self):
		return self.length

	def summary(self):
		hdf5_file = h5py.File(self.file_path, "r")
		dset = hdf5_file['coords']
		for name, value in dset.attrs.items():
			print(name, value)

		print('\nfeature extraction settings')
		print('transformations: ', self.roi_transforms)

	def __getitem__(self, idx):
		with h5py.File(self.file_path,'r') as hdf5_file:
			coord = hdf5_file['coords'][idx]
		#print(f"--coords = {coord}")
		img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')

		img = self.roi_transforms(img)
		return {'img': img, 'coord': coord}

class Dataset_All_Bags(Dataset):

	def __init__(self, csv_path):
		self.df = pd.read_csv(csv_path)
	
	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		return self.df['slide_id'][idx]




class Whole_Slide_Bag_KGH(Dataset):
	def __init__(self,
		data_path,
		wsi_name,
		img_transforms=None):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			img_transforms (callable, optional): Optional transform to be applied on a sample
		"""
		self.wsi_name = wsi_name
		self.roi_transforms = img_transforms

		self.data_path = data_path
		self.patho = 'Normal'

		if self.wsi_name[:2] == 'TA':
			self.patho = 'CP_TA'
		elif self.wsi_name[:2] == 'HP':
			self.patho = 'CP_HP'
		elif self.wsi_name[:3] == 'TVA':
			self.patho = 'CP_TVA'
		elif self.wsi_name[:3] == 'SSL':
			self.patho = 'CP_SSL'

		if self.patho != 'Normal':
			patches_patho_ROI = os.listdir(os.path.join(data_path,self.patho,'ROI'))
			for k in range(len(patches_patho_ROI)):
				patches_patho_ROI[k]=os.path.join('ROI',patches_patho_ROI[k])
			
			patches_patho_nonROI = os.listdir(os.path.join(data_path,self.patho,'nonROI'))
			for k in range(len(patches_patho_nonROI)):
				patches_patho_nonROI[k]=os.path.join('nonROI',patches_patho_nonROI[k])
			patches_patho = patches_patho_ROI + patches_patho_nonROI

		elif self.patho == 'Normal':
			patches_patho = os.listdir(os.path.join(data_path,self.patho))
		
		for k in range(len(patches_patho)):
				patches_patho[k]=os.path.join(self.patho,patches_patho[k])

		self.patches = [patches for patches in patches_patho if os.path.basename(patches).startswith(self.wsi_name)]

		# RegEx to find coordinates from the name of the patches
		self.pattern = r"-(\d+)-(\d+)-(\d+)-(\d+)\.png"
		if self.wsi_name[:2] == 'HP':
			print("Dealing with HP patches")
			self.pattern = r"-(\d+)-(\d+)-(\d+)-(\d+)-rot(\d+)\.png"
		
		self.coords = []
		for patch in self.patches:
			
			match = re.search(self.pattern, os.path.basename(patch) )
			if match:
				X1 = int(match.group(1))
				Y1 = int(match.group(2))
				X2 = int(match.group(3))
				Y2 = int(match.group(4))
				self.coords.append(np.array([X1,Y1]))
				
			else:
				print(f"----- PATCH {os.path.basename(patch)}")
				print("Pattern not found in filename.")

			
		#print(f"--- PATH TO IMAGE = {self.patches}")

		self.length = len(self.coords)

			
	def __len__(self):
		return self.length


	def __getitem__(self, idx):
		coord = self.coords[idx]
		img_path = os.path.join(self.data_path,self.patches[idx])
		
		img = Image.open(img_path)
		img = self.roi_transforms(img)

		return {'img': img, 'coord': [coord]}


# data_path='/home/ubuntu/Documents/Datasets/kgh_complete/1-25X/train'
# wsi_name='N152'
# dataset = Whole_Slide_Bag_KGH(data_path=data_path,wsi_name=wsi_name)
# print(f"Dataset from {wsi_name} of length {len(dataset)}")
# loader = DataLoader(dataset=dataset, batch_size=  32)
# print("--- loader done")
# for count, data in enumerate(tqdm(loader)):
# 	print(data[1])

# patches = ['Normal/N146-92--402-1608-626-1832.png', 'Normal/N146-29--603-603-827-827.png', 'Normal/N146-71--402-1206-626-1430.png', 'Normal/N146-118--1206-2010-1430-2234.png', 'Normal/N146-128--1206-2211-1430-2435.png', 'Normal/N146-129--1407-2211-1631-2435.png', 'Normal/N146-60--1005-1005-1229-1229.png', 'Normal/N146-108--1608-1809-1832-2033.png', 'Normal/N146-12--2412-201-2636-425.png', 'Normal/N146-84--603-1407-827-1631.png', 'Normal/N146-19--1407-402-1631-626.png', 'Normal/N146-147--1407-2613-1631-2837.png', 'Normal/N146-36--2010-603-2234-827.png', 'Normal/N146-107--1407-1809-1631-2033.png', 'Normal/N146-4--2613-0-2837-224.png', 'Normal/N146-136--804-2412-1028-2636.png', 'Normal/N146-26--2814-402-3038-626.png', 'Normal/N146-78--2010-1206-2234-1430.png', 'Normal/N146-61--1206-1005-1430-1229.png', 'Normal/N146-64--1809-1005-2033-1229.png', 'Normal/N146-122--2010-2010-2234-2234.png', 'Normal/N146-111--2211-1809-2435-2033.png', 'Normal/N146-127--1005-2211-1229-2435.png', 'Normal/N146-105--1005-1809-1229-2033.png', 'Normal/N146-11--2211-201-2435-425.png', 'Normal/N146-101--201-1809-425-2033.png', 'Normal/N146-48--1407-804-1631-1028.png', 'Normal/N146-50--1809-804-2033-1028.png', 'Normal/N146-117--1005-2010-1229-2234.png', 'Normal/N146-81--0-1407-224-1631.png', 'Normal/N146-123--2211-2010-2435-2234.png', 'Normal/N146-119--1407-2010-1631-2234.png', 'Normal/N146-97--2211-1608-2435-1832.png', 'Normal/N146-145--1005-2613-1229-2837.png', 'Normal/N146-21--1809-402-2033-626.png', 'Normal/N146-65--2010-1005-2234-1229.png', 'Normal/N146-58--603-1005-827-1229.png', 'Normal/N146-40--2814-603-3038-827.png', 'Normal/N146-45--804-804-1028-1028.png', 'Normal/N146-141--1809-2412-2033-2636.png', 'Normal/N146-77--1809-1206-2033-1430.png', 'Normal/N146-94--804-1608-1028-1832.png', 'Normal/N146-121--1809-2010-2033-2234.png', 'Normal/N146-110--2010-1809-2234-2033.png', 'Normal/N146-133--2211-2211-2435-2435.png', 'Normal/N146-30--804-603-1028-827.png', 'Normal/N146-70--201-1206-425-1430.png', 'Normal/N146-22--2010-402-2234-626.png', 'Normal/N146-33--1407-603-1631-827.png', 'Normal/N146-103--603-1809-827-2033.png', 'Normal/N146-126--804-2211-1028-2435.png', 'Normal/N146-82--201-1407-425-1631.png', 'Normal/N146-47--1206-804-1430-1028.png', 'Normal/N146-130--1608-2211-1832-2435.png', 'Normal/N146-17--1005-402-1229-626.png', 'Normal/N146-13--2613-201-2837-425.png', 'Normal/N146-38--2412-603-2636-827.png', 'Normal/N146-93--603-1608-827-1832.png', 'Normal/N146-51--2010-804-2234-1028.png', 'Normal/N146-53--2412-804-2636-1028.png', 'Normal/N146-142--2010-2412-2234-2636.png', 'Normal/N146-139--1407-2412-1631-2636.png', 'Normal/N146-137--1005-2412-1229-2636.png', 'Normal/N146-132--2010-2211-2234-2435.png', 'Normal/N146-146--1206-2613-1430-2837.png', 'Normal/N146-42--201-804-425-1028.png', 'Normal/N146-112--2412-1809-2636-2033.png', 'Normal/N146-28--402-603-626-827.png', 'Normal/N146-73--804-1206-1028-1430.png', 'Normal/N146-83--402-1407-626-1631.png', 'Normal/N146-90--0-1608-224-1832.png', 'Normal/N146-18--1206-402-1430-626.png', 'Normal/N146-39--2613-603-2837-827.png', 'Normal/N146-106--1206-1809-1430-2033.png', 'Normal/N146-44--603-804-827-1028.png', 'Normal/N146-102--402-1809-626-2033.png', 'Normal/N146-124--2412-2010-2636-2234.png', 'Normal/N146-120--1608-2010-1832-2234.png', 'Normal/N146-34--1608-603-1832-827.png', 'Normal/N146-20--1608-402-1832-626.png', 'Normal/N146-35--1809-603-2033-827.png', 'Normal/N146-62--1407-1005-1631-1229.png', 'Normal/N146-43--402-804-626-1028.png', 'Normal/N146-49--1608-804-1832-1028.png', 'Normal/N146-52--2211-804-2435-1028.png', 'Normal/N146-74--1005-1206-1229-1430.png', 'Normal/N146-3--2412-0-2636-224.png', 'Normal/N146-85--804-1407-1028-1631.png', 'Normal/N146-56--201-1005-425-1229.png', 'Normal/N146-23--2211-402-2435-626.png', 'Normal/N146-24--2412-402-2636-626.png', 'Normal/N146-72--603-1206-827-1430.png', 'Normal/N146-37--2211-603-2435-827.png', 'Normal/N146-57--402-1005-626-1229.png', 'Normal/N146-59--804-1005-1028-1229.png', 'Normal/N146-14--2814-201-3038-425.png', 'Normal/N146-25--2613-402-2837-626.png', 'Normal/N146-31--1005-603-1229-827.png', 'Normal/N146-66--2211-1005-2435-1229.png', 'Normal/N146-138--1206-2412-1430-2636.png', 'Normal/N146-79--2211-1206-2435-1430.png', 'Normal/N146-109--1809-1809-2033-2033.png', 'Normal/N146-131--1809-2211-2033-2435.png', 'Normal/N146-32--1206-603-1430-827.png', 'Normal/N146-140--1608-2412-1832-2636.png', 'Normal/N146-91--201-1608-425-1832.png', 'Normal/N146-63--1608-1005-1832-1229.png', 'Normal/N146-46--1005-804-1229-1028.png']
# for patch in patches:
# 	img_path = os.path.join(data_path,patch)
# 	print(f"--- {img_path}")	
# 	img = Image.open(img_path)
# 	print(f"- {type(img)}")