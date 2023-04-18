import torch

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