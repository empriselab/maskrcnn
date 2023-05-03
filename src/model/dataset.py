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

    def box_xywh_to_xyxy(self, boxes):
        """
        Convert (N x 4) tensor from (x,y,w,h) to (x1,y1,x2,y2)
        """
        boxes_xyxy = torch.zeros_like(boxes)
        boxes_xyxy[:,0] = boxes[:,0]
        boxes_xyxy[:,1] = boxes[:,1]
        boxes_xyxy[:,2] = boxes[:,0] + boxes[:,2]
        boxes_xyxy[:,3] = boxes[:,1] + boxes[:,3]
        return boxes_xyxy
    
    def __getitem__(self, idx):
        try:
            # Load data from npz file
            npz_data = np.load(self.file_list[idx])
            image = torch.tensor(npz_data['image']) / 255.
            masks = torch.tensor(npz_data['masks'], dtype=torch.int64)
            labels = torch.tensor(npz_data['labels'], dtype=torch.int64)
            boxes = torch.tensor(npz_data['boxes'])

            if labels.shape[0] == 0:    # if the data point has no bounding boxes/labels
                return {'image':-1, 'masks':-1, 'boxes':-1, 'labels':-1}

            boxes = self.box_xywh_to_xyxy(boxes)
            return {'image':image, 'masks':masks, 'boxes':boxes,'labels':labels}
            
        except Exception as e:
            print(e)
            print('Pickle file!!')
            print(self.file_list[idx])
            print('!!!')