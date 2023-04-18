import torch 
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np
import os

class RetinaNetDataset(Dataset):
    def __init__(self, directories, transform=None):
        self.transform = transform
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
        try:
            # Load data from npz file
            npz_data = np.load(self.file_list[idx])
            image = torch.tensor(npz_data['image']) / 255.
            labels = torch.tensor(npz_data['labels'], dtype=torch.int64)
            boxes = torch.tensor(npz_data['boxes'])

            return {'image':image,'boxes':boxes,'labels':labels}
        except:
            print('Pickle file!!')
            print(self.file_list[idx])
            print('!!!')