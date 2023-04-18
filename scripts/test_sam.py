import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import numpy as np
import cv2
import time
import torch

# sam_checkpoint = "sam_vit_h_4b8939.pth"
# model_type = "vit_h"

model_type = 'vit_b'
sam_checkpoint = 'sam_vit_b_01ec64.pth'

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(
    sam
)

n_images = 20
start = 50
print('Starting...')
start_time = time.time()
outputs = []
for i in range(start, start+n_images+1):
    img = cv2.imread('../data/training/v1/bag_1/bag_1_callback_{}_img.png'.format(i))
    output = mask_generator.generate(img)
    # img_tensor = torch.as_tensor(img, device=device).permute(2,0,1).contiguous()
    # input = [{'image':img_tensor, 'original_size':img_tensor.shape[:2]}]
    # batched_output = sam(input, multimask_output=False)
    outputs.append(output)
end_time = time.time()

print('Total {} time :: {}'.format(device, end_time-start_time))
print('Average {} time :: {}'.format(device, (end_time-start_time)/n_images))




# img = cv2.imread('../data/training/v1/bag_1/bag_1_callback_50_img.png')
# mask = cv2.imread('../data/training/v1/bag_1/bag_1_callback_50_mask.png')

# print('Extracting individual masks')
# extract bbox, labels 
# boxes, labels, binary_masks = [], [], []
# class_labels = np.unique(mask).tolist()[1:]    # exclude zero

# for label in class_labels:
#     class_mask = (mask == label).astype('uint8') 
#     processed_mask = (cv2.blur(class_mask, (2,2)) > 0.5).astype('uint8')
#     contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     class_bounding_boxes = []
#     class_object_masks = []
#     labels_for_this_class = [] 
#     for contour in contours:
#         contour_area = cv2.contourArea(contour)
#         if contour_area > 50.0:    # avoid 0 dimensional / speck contours
#             x, y, w, h = cv2.boundingRect(contour)
#             class_bounding_boxes.append((x, y, x+w, y+h))
#             object_mask = np.zeros_like(mask)
#             cv2.drawContours(object_mask, [contour], -1, 1, -1)
#             class_object_masks.append(object_mask)
#             labels_for_this_class.append(label)

#     boxes.extend(class_bounding_boxes)
#     labels.extend(labels_for_this_class)
#     binary_masks.extend(class_object_masks)

# print('Generating Masks...')
# start_time = time.time()
# masks = mask_generator.generate(img)
# end_time = time.time()
# print('Total {} time :: {}'.format(device, end_time-start_time))
