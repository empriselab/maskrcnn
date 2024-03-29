# correct_masks.py
# Author: Thomas Patton (tjp93)
#
# Purpose: Use the "Segment Anything Model" from Meta AI research to correct our potentially incorrect
# masks. This program uses the masks generated by SAM and then finds the closest corresponding mask in the 
# potentially incorrect mask to give the mask a label. 

import os
import time
import glob
import argparse
import json
from typing import List, Tuple
from pathlib import Path
import gc

import cv2
import numpy as np
from tqdm import tqdm

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

BASE_DIR = Path(__file__).absolute().parents[1]
TRAINING_DATA_DIR = str(BASE_DIR / 'data' / 'training' / 'v1')
OUTPUT_DIR = str(BASE_DIR / 'data' / 'training' / 'sam_plus_maskrcnn')
VERBOSE = 0

def verbose_log(msg):
    """
    Log outputs when var `VERBOSE` is true
    """
    if VERBOSE: print(msg)

def load_model():
    """
    Loads in the SAM model from local
    """
    device = 'cuda'
    model_type = 'vit_b'
    sam_checkpoint = 'sam_vit_b_01ec64.pth'

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    verbose_log('Model loaded...')
    return mask_generator

def batch_load_examples(image_paths:List) -> Tuple[np.ndarray]:
    """
    Loads a batch of images and masks into memory for prediction
    """
    images = [cv2.imread(path) for path in image_paths]
    mask_paths = [path.replace('img.png', 'mask.png') for path in image_paths]
    masks = [cv2.imread(path, 0) for path in mask_paths]
    verbose_log('{} examples loaded into memory'.format(len(image_paths)))
    return images, masks

def batch_predict_examples(mask_generator:SamAutomaticMaskGenerator, images:List[np.ndarray]) -> List:
    """
    Use our loaded SAM model to generate predictions for a batch of images
    """
    return [mask_generator.generate(i) for i in images]

def create_fork_mask():
    """
    Creates a mask image represeting the fork in the frame

    (stolen from annotate.py)
    """
    white = np.ones((720,1280))
    fork_outline = np.array([
        [714, 717], [707, 640], [733, 636], [724, 534],
        [778, 534], [785, 567], [815, 577], [865, 637],
        [987, 635], [988, 657], [1059, 718]
    ])
    cv2.fillPoly(white, [fork_outline], 0)
    return white

def create_projection_image(color_img: np.array, projected_mask: np.array, id_to_color:dict) -> np.array:
    """
    Create a new image with our projection overlayed on top 

    (stolen from annotate.py)
    """
    b, g, r = [{k:v[i] for k,v in id_to_color.items()} for i in range(3)]
    f_b, f_g, f_r = [np.vectorize(x.get) for x in [b,g,r]]
    b_layer, g_layer, r_layer = [f(projected_mask).astype('uint8') for f in [f_b, f_g, f_r]]
    color_mapped_mask = np.dstack((b_layer, g_layer, r_layer))
    empty = np.where(color_mapped_mask == [0,0,0])
    color_mapped_mask[empty] = color_img[empty]    # yes
    return cv2.addWeighted(color_mapped_mask, 0.75, color_img, 0.25, 0.0)

def mask_is_shadow(img, predicted_mask, shadow_thresh=0.25) -> bool:
    """
    Filter out masks that SAM things are objects that are actually shadows.
    Return a boolean if mask is predicted to be a shadow
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lower = np.array([0, 0, 0], dtype=np.uint8)
    upper = np.array([135, 135, 135], dtype=np.uint8)
    shadow_mask = cv2.inRange(lab, lower, upper) / 255

    predicted_mask_nz = np.nonzero(predicted_mask)
    num_nz = len(predicted_mask_nz[0])
    corresponding_shadow_mask = shadow_mask[predicted_mask_nz]
    shadow_score = corresponding_shadow_mask.sum() / num_nz   # lol shadow score??
    return shadow_score > shadow_thresh

def get_label_for_mask(predicted_mask:np.ndarray, true_mask:np.ndarray, distance_thresh:float) -> int:
    """
    Given our predicted mask of an object, find the closest nonzero
    value in our "true" mask. Returns the predicted mask "filled"
    with the closest value

    predicted_mask: mask predicted by SAM, boolean array (hxw)
    true_mask: mask produced by our autolabeling algorithm, int array (hxw)
    """
    predicted_mask_nz_idx = np.nonzero(predicted_mask)
    true_mask_nz_idx = np.nonzero(true_mask)

    if len(true_mask_nz_idx[0]) < 100:    # if we have less than 100 mask pixels in the scene, skip it
        return 0

    predicted_mask_mean_y = np.mean(predicted_mask_nz_idx[0])
    predicted_mask_mean_x = np.mean(predicted_mask_nz_idx[1])
        
    distances = np.sqrt((true_mask_nz_idx[1] - predicted_mask_mean_x) ** 2 + (true_mask_nz_idx[0] - predicted_mask_mean_y) ** 2)
    if min(distances) < distance_thresh:
        closest_index = np.argmin(distances)
        closest_value = true_mask[
            true_mask_nz_idx[0][closest_index], 
            true_mask_nz_idx[1][closest_index]
        ]
        # return predicted_mask_nz_idx, closest_value
        return closest_value
    else:
        return 0

def get_bbox_from_mask(mask:np.ndarray) -> np.ndarray:
    """
    Converts the mask of a single object into a bounding box for that mask
    """
    processed_mask = mask.astype('uint8')
    nz = processed_mask.nonzero()
    min_y, min_x = min(nz[0]), min(nz[1])
    max_y, max_x = max(nz[0]), max(nz[1])
    bbox = np.array([min_x, min_y, (max_x-min_x), (max_y-min_y)])
    return bbox


def label_sam_predictions(img, sam_output, true_mask, area_lower_thresh=500, area_upper_thresh=15000, distance_thresh=40):
    """
    Expects `sam_output` to be a list of dictionaries as produced 
    by the SAM model. `true_mask` is the mask produced by our 
    autolabeling algorithm. `area_thresh` is the max area acceptable
    for a predicted mask.
    """
    labels, boxes, masks = [], [] ,[]
    fork_mask = create_fork_mask()
    for i, output in enumerate(sam_output):
        predicted_mask = output['segmentation']
        area = output['area']

        is_shadow = mask_is_shadow(img, predicted_mask, area)
        if (not is_shadow) and area > area_lower_thresh and area < area_upper_thresh:
            closest_value = get_label_for_mask(predicted_mask, true_mask, distance_thresh=distance_thresh)
            if closest_value != 0:
                bbox = get_bbox_from_mask(predicted_mask)
                valid_mask = np.logical_and(predicted_mask, fork_mask)
                
                masks.append(valid_mask)
                boxes.append(bbox)
                labels.append(closest_value)

    masks = np.array(masks).astype(np.int8)
    boxes = np.array(boxes).astype(np.int16)
    labels = np.array(labels).astype(np.int8)
    
    return masks, boxes, labels

def save_output(filenames:List[str], images:List[np.ndarray], outputs:List[Tuple]) -> None:
    """
    Saves the resulting output data. `images` will be the original list of images,
    `outputs` will be a same length list with each index being a tuple of masks,
    boxes, and labels for the example
    """
    assert len(images) == len(outputs) and len(outputs) == len(filenames)
    N = len(images)
    for i in range(N):
        filename = filenames[i]
        image = images[i]
        masks, boxes, labels = outputs[i]

        if len(masks) > 0:   # did we actually find any objects in the scene
            # config filenames and save
            split_filename = filename.split('_')
            bag_number = split_filename[-4]
            callback_number = split_filename[-2]
            os.makedirs(os.path.join(OUTPUT_DIR, 'bag_{}'.format(bag_number)), exist_ok=True)

            np.savez_compressed(
                os.path.join(OUTPUT_DIR, 'bag_{}/callback_{}'.format(bag_number, callback_number)),  
                image=image,
                masks=masks,
                labels=labels,
                boxes=boxes
            )

def run_batch(model:SamAutomaticMaskGenerator, files:List[str], batch_indices:np.ndarray):
    """
    Load in a batch of images/masks, predict using SAM, then save
    the results accordingly
    """
    # load examples
    batch_filenames = [files[i] for i in batch_indices]
    images, masks = batch_load_examples(batch_filenames)

    # batch predict examples with SAM
    start_time = time.time()
    sam_output = batch_predict_examples(model, images)
    end_time = time.time()
    verbose_log('{} examples predicted in {}s ({}s/example)'.format(len(images), (end_time-start_time), (end_time-start_time)/len(images)))

    # convert SAM predictions to tuples of (masks, boxes, labels)
    labeled_outputs = [label_sam_predictions(images[i], sam_output[i], masks[i]) for i in range(len(images))]

    # save as a compressed .npz
    save_output(batch_filenames, images, labeled_outputs)

def get_batch_indices(total_size:int, batch_size:int):
    """
    Helper function to split some size N into batch indicies with size batch_size
    """
    index = np.arange(total_size)
    n_subgroups = (total_size // batch_size) + 1
    batch_index = np.array_split(index, n_subgroups)
    return batch_index

def get_files_in_bag(bag_path:str, sampling_rate:int=12):
    """
    Use glob to get files in a bag. Downsample modulo `sampling_rate`.
    """
    files = glob.glob('{}/*_img.png'.format(bag_path))
    valid_files = [f for f in files if (int(f.split('_')[-2]) % sampling_rate) == 0]
    return valid_files

def run(batch_size:int=4):
    """
    Run the program and use TQDM to track progress
    """
    start_time = time.time()
    sam = load_model()
    bags = glob.glob('{}/*'.format(TRAINING_DATA_DIR))

    for bag in tqdm(bags, desc='bag'):
        files = get_files_in_bag(bag)
        batched_indices = get_batch_indices(len(files), batch_size)
        for batch_indices in tqdm(batched_indices, desc='batch'):
            run_batch(sam, files, batch_indices)

        gc.collect()
        
    end_time = time.time()
    print('Total Time :: {}'.format(end_time-start_time))


if __name__ == '__main__':
    run()
