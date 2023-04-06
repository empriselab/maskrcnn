import torch 
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np
import os

class MaskRCNNDataset(Dataset):
    def __init__(self, directories):
        self.file_list = []
        
        # Collect list of npz files
        for directory in directories:
            for fi in os.listdir(directory):
                if '.npz' in fi:
                    full_path = os.path.join(directory, fi)
                    self.file_list.append(full_path)
            
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        # Load data from npz file
        npz_data = np.load(self.file_list[idx])
        image = torch.tensor(npz_data['image']) / 255.
        masks = torch.tensor(npz_data['masks'], dtype=torch.int64)
        labels = torch.tensor(npz_data['labels'], dtype=torch.int64)
        boxes = torch.tensor(npz_data['boxes'])

        return {'image':image, 'masks':masks, 'boxes':boxes,'labels':labels}