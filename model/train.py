import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn
import torch.utils.data as data
from torch import utils 

from dataset import SegmentationDataset
import utils

TRAINING_SET_VERSION = '1'
DATA_PATH = '../data/training/v{}'.format(TRAINING_SET_VERSION)
IMAGE_PATH = DATA_PATH + '/images'
MASK_PATH = DATA_PATH + '/masks'
BATCH_SIZE = 64
NUM_CLASSES = 34        # 33 food items + a zero class

# Define the device to train the model on
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')
print("DEVICE :: {}".format(device))

# load data and create data loaders
transform = transforms.Compose([transforms.ToTensor()])
dataset = SegmentationDataset(DATA_PATH, transform=transform)
train_loader, test_loader = utils.get_data_loaders(dataset, batch_size=BATCH_SIZE, num_workers=1)

# Load the pre-trained Mask RCNN model
model = maskrcnn_resnet50_fpn(pretrained=True)

# Replace the final layer with a new fully connected layer with 34 output channels
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, NUM_CLASSES)

# Freeze all the layers except the final layer
for param in model.parameters():
    param.requires_grad = False
for param in model.roi_heads.box_predictor.parameters():
    param.requires_grad = True


# Define the optimizer and the learning rate scheduler
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Move the model and the data to the device
model.to(device)
for imgs, targets in train_loader:
    imgs = [img.to(device) for img in imgs]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    # Train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
    # Update the learning rate
    lr_scheduler.step()
    # Evaluate on the test dataset
    evaluate(model, test_loader, device=device)

# Save the trained model
torch.save(model.state_dict(), 'maskrcnn_model.pth')
