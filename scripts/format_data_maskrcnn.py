import os
import time
from pathlib import Path
import gc
import multiprocessing

from PIL import Image 
import numpy as np
import cv2
import torch


TRAINING_SET_VERSION = '1'
BASE_DIR = Path(__file__).absolute().parents[1]
print("BASE DIRECTORY :: {}".format(BASE_DIR))
TRAINING_DATA_DIR = str(BASE_DIR / 'data' / 'training' / 'v{}'.format(TRAINING_SET_VERSION))
MASKRCNN_DIR = str(BASE_DIR / 'data' / 'training' / 'maskrcnn')
IMAGES_DIR = os.path.join(TRAINING_DATA_DIR, 'images')
MASKS_DIR = os.path.join(TRAINING_DATA_DIR, 'masks')

def create_maskrcnn_data(image_filename):
    """
    Uses an image filename to load an image and coresponding segmentation
    mask. Then uses this to create bounding boxes, binary masks, and labels
    all of which are required by MaskRCNN. These are saved as a compressed
    .npz format.
    """
    # load image and corresponding mask
    split_filename = image_filename.split('_')[:-1]
    example_name = '_'.join(split_filename)
    split_filename.append('mask.png')
    mask_filename = '_'.join(split_filename)
    example_path = os.path.join(MASKRCNN_DIR, example_name) 
    if os.path.exists(example_path):
        return 

    image_path = os.path.join(IMAGES_DIR, image_filename)
    mask_path = os.path.join(MASKS_DIR, mask_filename)
    image = np.array(Image.open(image_path).convert('RGB'))
    mask = np.array(Image.open(mask_path).convert('L'))

    # extract bbox, labels 
    boxes, labels, binary_masks = [], [], []
    class_labels = torch.unique(torch.tensor(mask)).tolist()[1:]    # exclude zero
    if len(class_labels) == 0:     # if there are no objects annotated in the scene
       return 

    for label in class_labels:
        class_mask = (mask == label).astype('uint8') 
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        class_bounding_boxes = []
        class_object_masks = []
        labels_for_this_class = [] 
        for contour in contours:
            contour_area = cv2.contourArea(contour)
            if contour_area > 0.0:    # sometimes finds 0 dim contours?
                x, y, w, h = cv2.boundingRect(contour)
                class_bounding_boxes.append((x, y, x+w, y+h))
                object_mask = np.zeros_like(mask)
                cv2.drawContours(object_mask, [contour], -1, 1, -1)
                class_object_masks.append(object_mask)
                labels_for_this_class.append(label)

        boxes.extend(class_bounding_boxes)
        labels.extend(labels_for_this_class)
        binary_masks.extend(class_object_masks)

    boxes = np.array(boxes).astype(np.int8)
    labels = np.array(labels).astype(np.int8)
    masks = np.array(binary_masks).astype(np.int8)
    os.makedirs(example_path, exist_ok=True)

    # compressed here saves a TON of data
    np.savez_compressed(
        os.path.join(example_path, 'data'),
        image=image,
        masks=masks,
        labels=labels,
        boxes=boxes
    )
    gc.collect()

if __name__ == '__main__':
    examples = os.listdir(IMAGES_DIR)
    pool = multiprocessing.Pool()
    pool.map(create_maskrcnn_data, examples)
    pool.close()
    pool.join()