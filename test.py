# %%
import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader,random_split
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms


# %%
# parameters
batch_size = 16
num_epoch = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
# Train transforms
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=224),  # Randomly crop the image to size 224x224
    transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
    transforms.RandomRotation(degrees=15),  # Randomly rotate the image by 15 degrees
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])

# Test transforms
test_transform = transforms.Compose([
    transforms.Resize(size=256),  # Resize the image to size 256x256
    transforms.CenterCrop(size=224),  # Crop the center of the image to size 224x224
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])

# %%
# data dir
train_dir = "C:/Users/gunja/Downloads/archive/seg_train/seg_train/"
test_dir = "C:/Users/gunja/Downloads/archive/seg_test/seg_test/"

# %%
train_dataset = ImageFolder(train_dir,transform=train_transform)
test_dataset = ImageFolder(test_dir,transform=test_transform)
test_dataset, val_dataset = random_split(test_dataset, [2000, 1000])


# %%


# %%
class_to_idx = train_dataset.class_to_idx
idx_to_class = {val:key for key, val in class_to_idx.items()}

# %%
# dataloader 
train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_dataloader = DataLoader(test_dataset,batch_size=batch_size,shuffle=True)
validation_dataloader = DataLoader(val_dataset,batch_size=batch_size,shuffle=True)

# %%
# model
from ComputerVision.ImageClassification.VGG import VGG16
model = VGG16(n_channel=3,n_classes=len(class_to_idx)).to(device=device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# %%
# training loop

for epoch_no in tqdm(range(num_epoch),desc='Training Loop',leave=True):
    model.train()
    train_loss = 0.0
    for i,(images, labels) in tqdm(enumerate(train_dataloader),desc=f'Loop: {epoch_no+1}',total=len(train_dataloader),leave=False):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_dataloader)

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(validation_dataloader,total=len(validation_dataloader),desc='Validation'):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(validation_dataloader)
    accuracy = correct / total * 100
    print(f"Epoch [{epoch_no+1}/{num_epoch}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%")

# %%



