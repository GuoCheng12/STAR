import os
import pdb
import numpy as np
import torch
import warnings
from torch.utils.data import Dataset

warnings.filterwarnings('ignore', category=UserWarning)

class SR_dataset(Dataset):
    def __init__(self, split, root_dir, **kargs):
        self.root_dir = root_dir
        self.split = split
        self.mode = kargs.get('mode', 'default') 

        if split == 'train':
            with open(kargs['filenames_file_train'], 'r') as f:
                self.filenames = [line.strip() for line in f.readlines()]# if line.startswith('train')]
        elif split == 'eval':
            with open(kargs['filenames_file_eval'], 'r') as f:
                self.filenames = [line.strip() for line in f.readlines()]# if line.startswith('eval')]
        else:
            raise ValueError("split must be 'train' or 'eval'")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        line = self.filenames[index]
        hr_file_path, lr_file_path ,_,_= line.split(',')
        try:
            hr_data = np.load(hr_file_path, allow_pickle=True).item()
        except:
            return self[(index + 1) % len(self)]  #
        hr_image = hr_data['image'] 
        mask = hr_data['mask']
        attn_map = hr_data['attn_map']

        try:
            lr_data = np.load(lr_file_path, allow_pickle=True).item()
        except:
            return self[(index + 1) % len(self)] 
        lr_image = lr_data['image'] 
        lr_mask = lr_data['mask']

        try:
            flux_lr_map = lr_data['attn_map']
        except:
            flux_lr_map = lr_mask####
        hr_image = self.normalize(hr_image, mask)
        lr_image = self.normalize(lr_image,lr_mask)
        hr_image = np.expand_dims(hr_image, axis=0)
        lr_image = np.expand_dims(lr_image, axis=0)
        mask = torch.from_numpy(mask).float()
        mask = np.expand_dims(mask, axis=0)
        return {'input': torch.from_numpy(lr_image).float(), 
                'hr': torch.from_numpy(hr_image).float(), 
                'mask': mask,
                'attn_map': torch.from_numpy(attn_map).float(),
                'flux_map':torch.from_numpy(flux_lr_map).float(),
                'filename': hr_file_path.split('/')[-1],
                'item': index}

    def normalize(self, image, mask=None):
        if mask is not None:
            valid_pixels = image[mask]
            if len(valid_pixels) > 0:
                min_val = np.min(valid_pixels)
                max_val = np.max(valid_pixels)
            else:
                min_val = 0
                max_val = 1
            image_normalized = (image - min_val) / (max_val - min_val + 1e-8)
            image_normalized[~mask] = 0  
        else:
            min_val = np.min(image)
            max_val = np.max(image)
            image_normalized = (image - min_val) / (max_val - min_val + 1e-8)
        
        return image_normalized
