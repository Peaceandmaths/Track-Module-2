""" Experiment with Malarial cells dataset. Running the pretrained model
 with the following parameters :
 model = Resnet18
 batch size = 32 
 learning rate = 0.001
 number of epochs = 20 
Using the CNN as fixed feature extractor 
(freeze the weights, change the last layer)
 """

""" Version 2 : I slpit train and test differently. I don't use the 20 images validation set and just split the training data I have  
 """
# Imports 
from __future__ import print_function, division
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


cudnn.benchmark = True
plt.ion() 


# Set up Tensorboard 

from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/Experiment_malarial_Resnet')

# Data augmentation and normalization for training
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}



data_dir = '/home/golubeka/Trackmodule2/Malarial_cells'

# Load the entire dataset
full_dataset = datasets.ImageFolder(data_dir, transform=data_transforms['train'])

# Split the full dataset into a new training set and a test set
train_dataset, test_dataset = train_test_split(full_dataset, test_size=0.2, random_state=42)
train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.2, random_state=42)

# Create dataloaders for the new training set and the test set
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=4)

# Get the dataset sizes and class names for the new training set and the test set
train_size = len(train_dataset)
test_size = len(test_dataset)
class_names = full_dataset.classes


device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

# Writing images in Tensorboard 

# get some random training images
dataiter = iter(train_loader)
images, labels = next(dataiter)
images, labels = images.to(device), labels.to(device)

# create grid of images
img_grid = torchvision.utils.make_grid(images)


# write to tensorboard
writer.add_image('Malarial_cells_Resnet', img_grid)



# Visualization 

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(train_loader))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

#imshow(out, title=[class_names[x] for x in classes])


# Visulaizing the model prediction 
# Generic function to display predictions for a few images


def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

def train_model(model, criterion, optimizer, scheduler, num_epochs=15):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
                
                running_loss = 0.0
                running_corrects = 0

                for i, (inputs, labels) in enumerate(train_loader):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                # scheduler.step()

                epoch_loss = running_loss / train_size
                epoch_acc = running_corrects.double() / train_size

                scheduler.step(epoch_loss)

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                writer.add_scalar('training loss', epoch_loss, epoch)
                writer.add_scalar('train Accuracy', epoch_acc, epoch)

            else: # evaluation loop
                model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for i, (inputs, labels) in enumerate(test_loader):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    # optimizer.zero_grad()

                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
               
                epoch_loss = running_loss / test_size
                epoch_acc = running_corrects.double() / test_size

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                writer.add_scalar('validation loss', epoch_loss, epoch)
                writer.add_scalar('val. Accuracy', epoch_acc, epoch)
            #writer.add_figure('predictions vs. actuals',visualize_model(model,num_images=6),global_step=epoch * len(dataloaders['train']) + i)

                # deep copy the model
                if  epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())


    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


""" ConvNet as fixed feature extractor
Here, we need to freeze all the network except the final layer. 
We need to set requires_grad = False to freeze the parameters so that the gradients 
are not computed in backward().

 """

model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
# scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_conv, factor = 0.1, patience = 5, verbose = True)

""" 
model_conv = model_conv.to(device='cpu')
images = images.to(device='cpu')
writer.add_graph(model_conv, images)
writer.close() """



""" Train and evaluate
On CPU this will take about half the time compared to previous scenario. 
This is expected as gradients don’t need to be computed for most of the network. 
However, forward does need to be computed. """

model_conv = train_model(model_conv, criterion, optimizer_conv, scheduler, num_epochs=15)

#visualize_model(model_conv)

plt.ioff()
#plt.show() 

# Tensorboard 
""" writer.add_graph(model_conv,inputs)
writer.close()
 """
