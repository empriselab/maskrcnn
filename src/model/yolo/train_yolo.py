# import sys
# import os
# import gc
# from pathlib import Path
# import torch
# from torch import nn
# import torchvision
# import torchvision.transforms as transforms
# from torchvision.models.detection import maskrcnn_resnet50_fpn
# from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
# import torch.utils.data as data
# from torch import utils 
# from torch.utils.data import DataLoader
# import numpy as np

# from dataset import MaskRCNNDataset
# import utils

# BASE_DIR = Path(__file__).absolute().parents[1]
# TRAINING_SET_VERSION = '1'
# DATA_PATH = str(BASE_DIR / 'data' / 'training' / 'maskrcnn')
# BATCH_SIZE = 8
# NUM_CLASSES = 34        # 33 food items + a zero class
# TEST_BAG_INDEXES = [1]


# # Define the device to train the model on
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# # device = torch.device('cpu')
# print("DEVICE :: {}".format(device))

# # load data and create data loaders
# all_bags = os.listdir(DATA_PATH)
# train_bags = [os.path.join(DATA_PATH, x) for x in all_bags if int(x.split('_')[-1]) not in TEST_BAG_INDEXES]
# test_bags = [os.path.join(DATA_PATH, x)for x in all_bags if int(x.split('_')[-1]) in TEST_BAG_INDEXES]

# train_dataset = MaskRCNNDataset(train_bags)
# test_dataset = MaskRCNNDataset(test_bags)
# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=0, collate_fn=utils.collate_fn)
# test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=0, collate_fn=utils.collate_fn)

# # Load the pre-trained Mask RCNN model
# # Replace the final layer with a new fully connected layer with 34 output channels
# model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
# in_features = model.roi_heads.box_predictor.cls_score.in_features
# model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, NUM_CLASSES)
# model.to(device)

# # Freeze all the layers except the final layer
# for param in model.parameters():
#     param.requires_grad = False
# for param in model.roi_heads.box_predictor.parameters():
#     param.requires_grad = True

# # Define the optimizer and the learning rate scheduler
# params = [p for p in model.parameters() if p.requires_grad]
# optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=0.0005)
# # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# # Set up loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.AdamW(model.roi_heads.box_predictor.parameters(), lr=1e-5)

# # Train only the last layer for a few epochs
# num_epochs = 1
# best_val_loss = float('inf')
# for epoch in range(num_epochs):

#     # train step

#     for i, batch in enumerate(train_loader):
#         model.train()
#         optimizer.zero_grad()

#         images, targets = batch 
#         images = images.to(device)
#         targets = [{k: v.to(device) for k,v in t.items()} for t in targets]

#         loss_dict = model(images, targets)
#         classifier_loss = loss_dict['loss_classifier']
#         # total_loss = sum(loss for loss in loss_dict.values())

#         # backward pass
#         # total_loss.backward()
#         classifier_loss.backward()
#         optimizer.step()

#         # release computational graph
#         # total_loss.detach_()
#         # for loss in loss_dict.values():
#         #     loss.detach_()

#         # print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {total_loss.detach().item():.4f}")
#         print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {classifier_loss.detach().item():.4f}")

#         if i % 50 == 0:
#             print('Evaluating...')
#             with torch.no_grad():
#                 total_val_loss = 0
#                 for j, batch in enumerate(test_loader):        
#                     if j >= 5:
#                         break
#                     images, targets = batch 
#                     images = images.to(device)
#                     targets = [{k: v.to(device) for k,v in t.items()} for t in targets]

#                     loss_dict = model(images, targets)
#                     val_loss = sum(loss for loss in loss_dict.values()).detach().item()
#                     total_val_loss += val_loss

#                     print(f"Batch [{j+1}/{5}], Loss: {val_loss:.4f}")

#                 print('Average Val Loss :: {}'.format(total_val_loss/5))
#                 if total_val_loss < best_val_loss:
#                     print('New best validation loss! Saving model...')
#                     best_model = model.state_dict()
#                     torch.save(best_model, 'best_model.pth') 
#                     best_val_loss = total_val_loss


# # best_model = model.state_dict()
# # torch.save(best_model, 'best_model.pth')

# # lr_scheduler.step()

#     # evaluation step
#     # model.eval()
#     # with torch.no_grad():
#     #     for j, batch in enumerate(test_loader):

#     #         images, targets = batch
#     #         loss_dict = model(images, targets)
#     #         sum_loss = sum(loss for loss in loss_dict.values())

#     #         print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{j+1}/{len(test_loader)}], Loss: {sum_loss.item():.4f}")
