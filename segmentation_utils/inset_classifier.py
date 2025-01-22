import os
from pathlib import Path
import torchvision
from torchvision.io import read_image
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim


class CustomImageDataset(Dataset):
    def __init__(self, insets_dir, non_insets_dir, transform=None, target_transform=None):
        self.image_labels = []
        self.image_paths = []
        for inset in os.listdir(insets_dir):
            self.image_labels.append(0)
            self.image_paths.append(insets_dir / inset)
        for non_inset in os.listdir(non_insets_dir):
            self.image_labels.append(1)
            self.image_paths.append(non_insets_dir / non_inset)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        print(f"About to load {self.image_paths[idx]}")
        image = read_image(str(self.image_paths[idx]))
        label = self.image_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

train_insets_dir = Path("./inset_detection/train/insets_dir").expanduser()
train_non_insets_dir = Path("./inset_detection/train/non_insets_dir").expanduser()
train_inset_datasets = CustomImageDataset(train_insets_dir, train_non_insets_dir)
train_dataloader = DataLoader(train_inset_datasets, batch_size=2, shuffle=True)
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
classes = ('inset', 'non_inset')

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# for epoch in range(2):  # loop over the dataset multiple times

#     running_loss = 0.0
#     for i, data in enumerate(train_dataloader, 0):
#         # get the inputs; data is a list of [inputs, labels]
#         inputs, labels = data

#         # zero the parameter gradients
#         optimizer.zero_grad()

#         # forward + backward + optimize
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         # print statistics
#         running_loss += loss.item()
#         if i % 2000 == 1999:    # print every 2000 mini-batches
#             print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
#             running_loss = 0.0

# print('Finished Training')

# PATH = './cifar_net.pth'
# torch.save(net.state_dict(), PATH)

# dataiter = iter(test_dataloader)
# images, labels = dataiter.next()

# # print images
# imshow(torchvision.utils.make_grid(images))
# print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))


# # get some random training images
# dataiter = iter(train_dataloader)
# images, labels = dataiter.next()
# classes = ('inset', 'non_inset')
# batch_size = 2
# # show images
# imshow(torchvision.utils.make_grid(images))
# # print labels
# print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))