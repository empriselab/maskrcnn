import os
from PIL import Image
import torch.utils.data as data

class SegmentationDataset(data.Dataset):
    def __init__(self, train_dir, transform=None):
        self.train_dir = train_dir 
        self.transform = transform
        
        # Get a list of all the image and mask file names
        self.image_dir = os.path.join(self.train_dir, 'images')
        self.mask_dir = os.path.join(self.train_dir, 'masks')
        self.image_filenames = [fi for fi in os.listdir(self.image_dir)] 
        
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, index):
       # Load the image from the given index
        image_filename = self.image_filenames[index]
        image_path = os.path.join(self.image_dir, image_filename)
        image = Image.open(image_path).convert('RGB')

        # Find the corresponding mask filename
        # TODO Possibly optimize, this method could be faster  
        path_list = image_filename.split('_')
        path_list[-1] = 'mask.png'
        mask_filename = '_'.join(path_list)
        mask_path = os.path.join(self.mask_dir, mask_filename)

        # Load the mask if it exists, or return None otherwise
        if os.path.exists(mask_path):
            mask = Image.open(mask_path)
        else:
            print(f"Warning: mask not found for image {image_filename}")
            mask = None

        # Apply the transform to the image and mask (if it exists)
        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)

        
        return image, mask
