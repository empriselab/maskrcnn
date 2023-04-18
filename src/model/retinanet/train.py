import sys
import os
import gc
from pathlib import Path
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.models.detection import RetinaNet_ResNet50_FPN_Weights
import torch.utils.data as data
from torch import utils 
from torch.utils.data import DataLoader
import numpy as np

from dataset import RetinaNetDataset 
import utils

BASE_DIR = Path(__file__).absolute().parents[3]
TRAINING_SET_VERSION = '1'
DATA_PATH = str(BASE_DIR / 'data' / 'training' / 'maskrcnn')    # can use MaskRCNN data in this case
BATCH_SIZE = 8
NUM_CLASSES = 34        # 33 food items + a zero class
TEST_BAG_INDEXES = [1, 15]

# Define the device to train the model on
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device('cpu')
print("DEVICE :: {}".format(device))

# load data and create data loaders
all_bags = os.listdir(DATA_PATH)
train_bags = [os.path.join(DATA_PATH, x) for x in all_bags if int(x.split('_')[-1]) not in TEST_BAG_INDEXES]
test_bags = [os.path.join(DATA_PATH, x)for x in all_bags if int(x.split('_')[-1]) in TEST_BAG_INDEXES]

target_transform = transforms.Compose([
    transforms.Resize((800, 1333)),
    transforms.ToTensor(),
])

train_dataset = RetinaNetDataset(train_bags, transform=target_transform)
test_dataset = RetinaNetDataset(test_bags, transform=target_transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=0, collate_fn=utils.collate_fn)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=0, collate_fn=utils.collate_fn)

# Load the pre-trained RetinaNet
# Replace the final layer with a new fully connected layer with 34 output channels
model = torchvision.models.detection.retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.DEFAULT)
in_channels = model.head.classification_head.cls_logits.in_channels
num_anchors = model.head.classification_head.num_anchors
classification_layer = torch.nn.Conv2d(in_channels, NUM_CLASSES * num_anchors, kernel_size=3, stride=1, padding=1)
model.head.classification_head.cls_logits = classification_layer

# Replace the anchor generator with a new one for the new dataset
anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
anchor_generator = torchvision.models.detection.anchor_utils.AnchorGenerator(
    anchor_sizes, aspect_ratios
)
model.anchor_generator = anchor_generator

# Replace the box predictor with a new one for the new dataset
box_predictor = torchvision.models.detection.retinanet.RetinaNetHead(
    in_channels=in_channels,
    num_classes=NUM_CLASSES,
    num_anchors=num_anchors
)
model.head.box_predictor = box_predictor
model.to(device)

# Freeze all the layers except the final layer
# for param in model.parameters():
#     param.requires_grad = False
# for param in model.roi_heads.box_predictor.parameters():
#     param.requires_grad = True

# Define the optimizer and the learning rate scheduler
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=0.0005)

# Set up loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Train only the last layer for a few epochs
num_epochs = 1
best_val_loss = float('inf')
for epoch in range(num_epochs):

    # train step

    for i, batch in enumerate(train_loader):
        model.train()
        optimizer.zero_grad()

        images, targets = batch 
        images = images.to(device)
        targets = [{k: v.to(device) for k,v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        classifier_loss = loss_dict['loss_classifier']
        # total_loss = sum(loss for loss in loss_dict.values())

        # backward pass
        # total_loss.backward()
        classifier_loss.backward()
        optimizer.step()

        # release computational graph
        # total_loss.detach_()
        # for loss in loss_dict.values():
        #     loss.detach_()

        # print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {total_loss.detach().item():.4f}")
        print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {classifier_loss.detach().item():.4f}")

        if i % 50 == 0:
            print('Evaluating...')
            with torch.no_grad():
                total_val_loss = 0
                for j, batch in enumerate(test_loader):        
                    if j >= 5:
                        break
                    images, targets = batch 
                    images = images.to(device)
                    targets = [{k: v.to(device) for k,v in t.items()} for t in targets]

                    loss_dict = model(images, targets)
                    val_loss = sum(loss for loss in loss_dict.values()).detach().item()
                    total_val_loss += val_loss

                    print(f"Batch [{j+1}/{5}], Loss: {val_loss:.4f}")

                print('Average Val Loss :: {}'.format(total_val_loss/5))
                if total_val_loss < best_val_loss:
                    print('New best validation loss! Saving model...')
                    best_model = model.state_dict()
                    torch.save(best_model, 'best_model.pth') 
                    best_val_loss = total_val_loss


# best_model = model.state_dict()
# torch.save(best_model, 'best_model.pth')

# lr_scheduler.step()

    # evaluation step
    # model.eval()
    # with torch.no_grad():
    #     for j, batch in enumerate(test_loader):

    #         images, targets = batch
    #         loss_dict = model(images, targets)
    #         sum_loss = sum(loss for loss in loss_dict.values())

    #         print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{j+1}/{len(test_loader)}], Loss: {sum_loss.item():.4f}")
