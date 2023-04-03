import time
import torch
import numpy as np
# import torchvision.utils as utils
from torch.utils.data import DataLoader, SubsetRandomSampler

def loader_to_device(loader, device):
    for imgs, targets in loader:
        imgs = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    return None


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
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, num_workers=num_workers)

    # loader_to_device(train_loader, device)
    # loader_to_device(test_loader, device)
    
    return train_loader, test_loader


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = _get_metric_logger()
    metric_logger.add_meter('lr', _get_lr(optimizer))
    header = 'Epoch: [{}]'.format(epoch)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        assert torch.isfinite(losses).all(), "Loss is not finite. Training stopped."

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        metric_logger.update(loss=losses, **loss_dict)
        metric_logger.update(lr=_get_lr(optimizer))

    return metric_logger


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    metric_logger = _get_metric_logger()

    header = 'Test:'
    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        output = model(images)

        for i, target in enumerate(targets):
            target = {k: v for k, v in target.items() if k != 'image_id'}
            model.boxes = output[i]['boxes']
            model.labels = output[i]['labels']
            model.scores = output[i]['scores']
            model.masks = output[i]['masks']
            res = {target['image_id'].item(): output[i]}
            evaluator_time = time.time()
            metric_logger.update(evaluator_time=evaluator_time)

    return metric_logger


def _get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def _get_metric_logger():
    from collections import defaultdict
    metric_logger = defaultdict(float)
    return metric_logger
