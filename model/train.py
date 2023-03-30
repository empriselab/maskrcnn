import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from model import SegmentationDataset, SegmentationNet

# Define hyperparameters
batch_size = 8
learning_rate = 0.001
num_epochs = 10
num_classes = 2  # Include background and foreground classes

# Define data transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize input images to 256x256
    transforms.ToTensor(),  # Convert PIL images to tensors
])

# Create dataset and dataloader
dataset = SegmentationDataset('path/to/data', transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Create model, loss function, and optimizer
model = SegmentationNet(num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training loop
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Print statistics
        running_loss += loss.item()
        if i % 10 == 9:  # Print every 10 mini-batches
            print('[Epoch %d, Batch %d] Loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0
