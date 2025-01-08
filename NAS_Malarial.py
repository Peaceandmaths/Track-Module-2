
# Imports 
#from __future__ import print_function, division
import sys
import torch
import torch.nn as nn
import torch.optim as optim
# Learning rate scheduler 
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
# for loading data 
import torchvision
from torchvision import datasets, models, transforms 
import matplotlib.pyplot as plt
import time
import os
import copy
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # progressbar
import torchmetrics
import pickle as pkl
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # prevents "OSError: broken data stream when reading image file" in some cases
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import nni.retiarii.nn.pytorch as nn
from nni.retiarii import model_wrapper
from nni.retiarii.nn.pytorch import LayerChoice, InputChoice, ValueChoice
from nni.common.serializer import dump

#nni.common.serializer.dump(obj, fp=None, *, use_trace=True, pickle_size_limit=3000, allow_nan=True, **json_tricks_kwargs)
# Loading and augmenting data 

def get_data():
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    img_path = '/home/golubeka/Trackmodule2/Malarial_cells/'
    # create dataset of original images, one with augmentation operations, one without

    image_dataset_augmented = datasets.ImageFolder(os.path.join(img_path, 'train'), data_transforms['train'])
    image_dataset = datasets.ImageFolder(os.path.join(img_path, 'train'), data_transforms['val'])


    dataset_size = len(image_dataset_augmented)
    class_names = image_dataset_augmented.classes
    num_classes = len(class_names)

    # Hyperparameters 
    num_epochs = 15
    batch_size = 32
    learning_rate = 0.001
    train_verbose = True  # show epoch progressbar
    device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')

    # split data with stratified kfold
    dataset_indices = list(range(dataset_size))

    # split test data from train data
    train_idx, test_idx = train_test_split(dataset_indices, test_size=0.2, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=42)

    # np.unique(train_image_dataset.targets, return_counts=True)[1]
    y_test = [image_dataset.targets[x] for x in test_idx]
    y_train = [image_dataset.targets[x] for x in train_idx]
    y_val = [image_dataset.targets[x] for x in val_idx]

    len(y_train)
    len(y_test)
    len(y_val)


    train_dataset = torch.utils.data.Subset(image_dataset_augmented, train_idx)
    val_dataset = torch.utils.data.Subset(image_dataset, val_idx)
    test_dataset = torch.utils.data.Subset(image_dataset, test_idx)

    # define data loaders
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True)

    # DataLoader accepts pin_memory argument, which defaults to False. When using a GPU it’s better to set pin_memory=True, 
    # this instructs DataLoader to use pinned memory and enables faster and asynchronous memory copy from the host to the GPU.

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True)

    dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset), 'test': len(test_dataset)}

    print(type(train_dataset))
    # load my data, load everything together, then train in batches 
    train_loader = dataloaders['train']
    print(type(train_loader))
    test_loader = dataloaders['test']

    return train_loader, test_loader

# Define Base Model 

SIZE = 64 # size of an image 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.norm1 = nn.BatchNorm2d(32)
        self.drop1 = nn.Dropout2d(p=0.2)
        
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.norm2 = nn.BatchNorm2d(32)
        self.drop2 = nn.Dropout2d(p=0.2)
        
        self.fc1 = nn.Linear(32 * SIZE * SIZE // 16, 512)
        self.norm3 = nn.BatchNorm1d(512)
        self.drop3 = nn.Dropout(p=0.2)
        
        self.fc2 = nn.Linear(512, 256)
        self.norm4 = nn.BatchNorm1d(256)
        self.drop4 = nn.Dropout(p=0.2)
        
        self.fc3 = nn.Linear(256, 2)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.norm1(x)
        x = self.drop1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.norm2(x)
        x = self.drop2(x)
        
        x = x.view(x.size(0), -1)  # Flatten the tensor
        
        x = F.relu(self.fc1(x))
        x = self.norm3(x)
        x = self.drop3(x)
        
        x = F.relu(self.fc2(x))
        x = self.norm4(x)
        x = self.drop4(x)
        
        x = F.sigmoid(self.fc3(x))
        
        return x

model = Net()
criterion = nn.BCELoss()  # Binary cross-entropy loss
optimizer = torch.optim.Adam(model.parameters())

# The rest of your code for training and compilation goes here...


# Visualize the experiment 

import os
from pathlib import Path

""" 
def evaluate_model_with_visualization(model_cls):
    model = model_cls()
    # dump the model into an onnx
    if 'NNI_OUTPUT_DIR' in os.environ:
        dummy_input = torch.zeros(1, 3, 32, 32)
        torch.onnx.export(model, (dummy_input, ),
                          Path(os.environ['NNI_OUTPUT_DIR']) / 'model.onnx')
    evaluate_model(model_cls)


###### Application

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
num_epochs = 10

model = Net()  # Instantiate your model
model.to(device)


for epoch in range(num_epochs):
    train_epoch(model, device, train_loader, optimizer, epoch)

accuracy = test_epoch(model, device, test_loader)
print(f"Test accuracy: {accuracy}%")

evaluate_model(model)
 """


