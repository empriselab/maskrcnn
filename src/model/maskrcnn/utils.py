import os
import torch
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn
import torch.utils.data as data
from torch import utils 
from torch.utils.data import DataLoader, SubsetRandomSampler


def collate_fn(batch):
    """
    Our batch is a list of dictionaries with and `image`, `boxes`, `masks`, and `labels` 
    as fields. Ultimately our call to `model()` will take `inputs` and targets` so the two
    objectives are to stack the images in our batch and then stack the other three
    fields that comprise the "target"
    """
    stacked_inputs = []
    stacked_targets = []
    for sample in batch:

        # process image
        image = sample['image']
        chw_image = image.permute(2, 0, 1)    # MaskRCNN expects channels first
        stacked_inputs.append(chw_image)

        # target is everything BUT image
        target = {k:v for k,v in sample.items() if k != 'image'}
        stacked_targets.append(target)
    
    inputs = torch.stack(stacked_inputs, dim=0)
    return inputs, stacked_targets

def get_data_loaders(dataset, batch_size, num_workers, test_split=0.2, device="cpu"):
    # Get the total number of samples in the dataset
    num_samples = len(dataset)
    
    # Split the dataset into training and validation sets
    indices = list(range(num_samples))
    split = int(np.floor(test_split * num_samples))
    np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]
    
    # Create the sampler objects for the training and validation sets
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    
    # Create the DataLoader objects for the training and validation sets
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, num_workers=num_workers, collate_fn=collate_fn)

    return train_loader, test_loader