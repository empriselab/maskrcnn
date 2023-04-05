from pathlib import Path
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn
import torch.utils.data as data
from torch import utils 

from dataset import MaskRCNNDataset
import utils

BASE_DIR = Path(__file__).absolute().parents[1]
TRAINING_SET_VERSION = '1'
DATA_PATH = str(BASE_DIR / 'data' / 'training' / 'maskrcnn')
BATCH_SIZE = 4
NUM_CLASSES = 34        # 33 food items + a zero class

# Define the device to train the model on
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("DEVICE :: {}".format(device))

# load data and create data loaders
dataset = MaskRCNNDataset(DATA_PATH)
train_loader, test_loader = utils.get_data_loaders(dataset, batch_size=BATCH_SIZE, num_workers=1)

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
num_epochs = 10
for epoch in range(num_epochs):

    # train step
    model.train()
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()

        images, targets = batch
        try:
            loss_dict = model(images, targets)
            sum_loss = sum(loss for loss in loss_dict.values())
        
            sum_loss.backward()
            optimizer.step()
        except:
            for target in targets:
                print(target['boxes'].shape) 
                print(target['boxes'])
            exit()

        print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {sum_loss.item():.4f}")

    lr_scheduler.step()

    # evaluation step
    model.eval()
    with torch.no_grad():
        for j, batch in enumerate(test_loader):

            images, targets = batch
            loss_dict = model(images, targets)
            sum_loss = sum(loss for loss in loss_dict.values())

            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{j+1}/{len(test_loader)}], Loss: {sum_loss.item():.4f}")
