import os

image_dir = '../data/training/v1/images'
mask_dir  = '../data/training/v1/masks'
fnames = os.listdir(image_dir)

for f in fnames:
    split_path = f.split('_')
    split_path[-1] = 'mask.png'
    mask_name = '_'.join(split_path)
    mask_path = os.path.join(mask_dir, mask_name)

    print(mask_path)
    if not os.path.exists(mask_path):
        print(mask_path)