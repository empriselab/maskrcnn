import os
import time

from PIL import Image 
import numpy as np
import cv2

import torch 
import torchvision.transforms as T

TRAINING_SET_VERSION = '1'
TRAINING_DATA_DIR = '../data/training/v{}'.format(TRAINING_SET_VERSION)
MASKRCNN_DIR = '../data/training/maskrcnn'
IMAGES_DIR = os.path.join(TRAINING_DATA_DIR, 'images')
MASKS_DIR = os.path.join(TRAINING_DATA_DIR, 'masks')

t0 = time.time()
examples = os.listdir(IMAGES_DIR)
N = len(examples)
for i in range(N):
    if i % 100 == 0:
        print(float(i/15000.))

    # load image and corresponding mask
    image_filename = examples[i]
    split_filename = image_filename.split('_')[:-1]
    example_name = '_'.join(split_filename)
    split_filename.append('mask.png')
    mask_filename = '_'.join(split_filename)

    image_path = os.path.join(IMAGES_DIR, image_filename)
    mask_path = os.path.join(MASKS_DIR, mask_filename)
    image = np.array(Image.open(image_path).convert('RGB'))
    mask = np.array(Image.open(mask_path).convert('L'))

    t1 = time.time()
    print(t1-t0)

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
            x, y, w, h = cv2.boundingRect(contour)
            class_bounding_boxes.append((x, y, x+w, y+h))
            object_mask = np.zeros_like(mask)
            cv2.drawContours(object_mask, [contour], -1, 1, -1)
            class_object_masks.append(object_mask)
            labels_for_this_class.append(label)

        boxes.extend(class_bounding_boxes)
        labels.extend(labels_for_this_class)
        binary_masks.extend(class_object_masks)

    t2 = time.time()
    print(t2-t1)

    boxes = np.array(boxes)
    labels = np.array(labels)
    masks = np.array(binary_masks)
    example_path = os.path.join(MASKRCNN_DIR, example_name) 
    os.makedirs(example_path, exist_ok=True)

    np.save(os.path.join(example_path, 'image.npy'), image)
    np.save(os.path.join(example_path, 'boxes.npy'), boxes)
    np.save(os.path.join(example_path, 'masks.npy'), masks)
    np.save(os.path.join(example_path, 'labels.npy'), labels)
    print(time.time() - t2)

    exit()