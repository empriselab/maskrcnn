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

MAX_OBJECTS = 20     # adjustable parameter, should be the max segmentations you expect in a scene

def create_maskrcnn_data(image_filename):
    """
    Uses an image filename to load an image and coresponding segmentation
    mask. Then uses this to create bounding boxes, binary masks, and labels
    all of which are required by MaskRCNN. These are saved as a compressed
    .npz format.
    """
    # load image and corresponding mask
    split_filename = image_filename.split('_')[:-1]
    bag_nbr = '_'.join(split_filename[:2])
    callback_nbr = '_'.join(split_filename[-2:])
    example_name = '_'.join(split_filename)
    split_filename.append('mask.png')
    mask_filename = '_'.join(split_filename)
    bag_path = os.path.join(MASKRCNN_DIR, bag_nbr)
    # example_path = os.path.join(MASKRCNN_DIR, example_name) 
    
    image_path = os.path.join(IMAGES_DIR, image_filename)
    mask_path = os.path.join(MASKS_DIR, mask_filename)
    image = np.array(Image.open(image_path).convert('RGB'))
    mask = np.array(Image.open(mask_path).convert('L'))

    # extract bbox, labels 
    boxes, labels, binary_masks = [], [], []
    class_labels = torch.unique(torch.tensor(mask)).tolist()[1:]    # exclude zero

    for label in class_labels:
        class_mask = (mask == label).astype('uint8') 
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        class_bounding_boxes = []
        class_object_masks = []
        labels_for_this_class = [] 
        for contour in contours:
            contour_area = cv2.contourArea(contour)
            if contour_area > 400.0:    # avoid 0 dimensional / speck contours
                x, y, w, h = cv2.boundingRect(contour)
                class_bounding_boxes.append((x, y, x+w, y+h))
                object_mask = np.zeros_like(mask)
                cv2.drawContours(object_mask, [contour], -1, 1, -1)
                class_object_masks.append(object_mask)
                labels_for_this_class.append(label)

        boxes.extend(class_bounding_boxes)
        labels.extend(labels_for_this_class)
        binary_masks.extend(class_object_masks)

    boxes = np.array(boxes).astype(np.int16)    # need 16bit precision for 720x1280 images
    labels = np.array(labels).astype(np.int8)
    masks = np.array(binary_masks).astype(np.int8)
    os.makedirs(bag_path, exist_ok=True)

    # verify N's are equal and we actually found objects in the scene
    assert (boxes.shape[0] == labels.shape[0]) and (labels.shape[0] == masks.shape[0])
    if boxes.shape[0] > 0:
        # compressed here saves a TON of data
        np.savez_compressed(
            os.path.join(bag_path, callback_nbr),    # also, saving in subdirs speeds up filesave time
            image=image,
            masks=masks,
            labels=labels,
            boxes=boxes
        )
    gc.collect()

if __name__ == '__main__':
    n_processes = 16    # adjust as needed, 16 on the g2 cluster is fast
    examples = os.listdir(IMAGES_DIR)
    pool = multiprocessing.Pool(n_processes)
    pool.map(create_maskrcnn_data, examples)
    pool.close()
    pool.join()