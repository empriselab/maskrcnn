import sys
import os
import gc
from pathlib import Path
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn
import torch.utils.data as data
from torch import utils 
import numpy as np

from dataset import MaskRCNNDataset
import utils

BASE_DIR = Path(__file__).absolute().parents[1]
TRAINING_SET_VERSION = '1'
DATA_PATH = str(BASE_DIR / 'data' / 'training' / 'maskrcnn')
BATCH_SIZE = 8
NUM_CLASSES = 34        # 33 food items + a zero class


bag_numbers = os.listdir(DATA_PATH)
def load_data():
    random_bag = np.random.choice(bag_numbers)
    bag_path = os.path.join(DATA_PATH, random_bag)
    callbacks = os.listdir(bag_path)
    n_callbacks = len(callbacks)
    random_callback = callbacks[np.random.randint(0, n_callbacks)]
    full_npz_path = os.path.join(bag_path, random_callback)
    print(full_npz_path)
    npz_data = np.load(full_npz_path)

    image = (torch.tensor(npz_data['image']) / 255.).permute(2, 0, 1)[None, :]
    masks = torch.tensor(npz_data['masks'], dtype=torch.int64)
    labels = torch.tensor(npz_data['labels'], dtype=torch.int64)
    boxes = torch.tensor(npz_data['boxes'])

    return image, [{'masks':masks, 'labels':labels, 'boxes':boxes}]

# Define the device to train the model on
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device('cpu')
print("DEVICE :: {}".format(device))

# load data and create data loaders
dataset = MaskRCNNDataset(DATA_PATH)
train_loader, test_loader = utils.get_data_loaders(dataset, batch_size=BATCH_SIZE, num_workers=0)

# Load the pre-trained Mask RCNN model
# Replace the final layer with a new fully connected layer with 34 output channels
model = maskrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, NUM_CLASSES)
model.to(device)

# Freeze all the layers except the final layer
for param in model.parameters():
    param.requires_grad = False
for param in model.roi_heads.box_predictor.parameters():
    param.requires_grad = True

# Define the optimizer and the learning rate scheduler
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Set up loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.roi_heads.box_predictor.parameters(), lr=0.001, momentum=0.9)

# Train only the last layer for a few epochs
num_epochs = 5
for epoch in range(num_epochs):

    # train step
    model.train()
    for i, batch in enumerate(train_loader):

        images, targets = batch 
        images = images.to(device)
        targets = [{k: v.to(device) for k,v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        # total_loss = sum(loss for loss in loss_dict.values())
        total_loss = 0
        for loss_key in loss_dict:
            if 'classifier' in loss_key:
                total_loss += 2*loss_dict[loss_key]
            else:
                total_loss += loss_dict[loss_key]

        # backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # release computational graph
        total_loss.detach_()
        for loss in loss_dict.values():
            loss.detach_()

        print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {total_loss.detach().item():.4f}")

    lr_scheduler.step()

best_model = model.state_dict()
torch.save(best_model, 'best_weighted_model.pth')

# lr_scheduler.step()

    # evaluation step
    # model.eval()
    # with torch.no_grad():
    #     for j, batch in enumerate(test_loader):

    #         images, targets = batch
    #         loss_dict = model(images, targets)
    #         sum_loss = sum(loss for loss in loss_dict.values())

    #         print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{j+1}/{len(test_loader)}], Loss: {sum_loss.item():.4f}")
