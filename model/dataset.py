import torch 
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np
import os

class MaskRCNNDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.transforms = transforms.Compose([
            transforms.ToTensor()
        ])
        self.file_list = []
        
        # Collect list of npz files
        for subdir in os.listdir(root_dir):
            subdir_path = os.path.join(root_dir, subdir)
            
            for fi in os.listdir(subdir_path):
                if '.npz' in fi and 'bag_1' not in subdir:
                    full_path = os.path.join(root_dir, subdir, fi)
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