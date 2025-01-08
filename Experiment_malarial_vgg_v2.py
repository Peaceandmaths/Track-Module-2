""" Experiment with Malarial cells dataset. Running the pretrained model
 with the following parameters :
 model = VGG
 batch size = 32 
 learning rate = 0.001
 number of epochs = 20 
Using the CNN as fixed feature extractor 
(freeze the weights, change the last layer)
 """

""" Version 4 : I merge Stefan's code on regnet and my v2 
 """
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

cudnn.benchmark = True
plt.ion() 


# default `log_dir` is "runs" - we'll be more specific here
# writer = SummaryWriter('runs/Experiment_malarial_Resnet')

# Data augmentation and normalization for training
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
project_path = '/home/golubeka/Trackmodule2/'
result_path = project_path + 'runs/Experiments/'


# create dataset of original images, one with augmentation operations, one without
image_dataset_augmented = datasets.ImageFolder(os.path.join(img_path, 'train'), data_transforms['train'])
image_dataset = datasets.ImageFolder(os.path.join(img_path, 'train'), data_transforms['val'])

dataset_size = len(image_dataset_augmented)
class_names = image_dataset_augmented.classes
num_classes = len(class_names)

num_epochs = 15
batch_size = 32
learning_rate = 0.001
train_verbose = True  # show epoch progressbar
device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')


# split data with stratified kfold
dataset_indices = list(range(dataset_size))

experiment_name = 'vgg' + \
                  '_epochs' + str(num_epochs) + \
                  '_lr' + str(learning_rate) + \
                  '_batchsize' + str(batch_size)

print('Starting experiment:', experiment_name)

# create path to store results
act_result_path = result_path + experiment_name + '/'
try:
    os.mkdir(act_result_path)
except OSError as error:
    print(error)
try:
    os.mkdir(act_result_path + 'plots/')
except OSError as error:
    print(error)



# Writer will output to ./runs/ directory by default
writer = SummaryWriter(act_result_path)

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

model_conv = torchvision.models.vgg16(pretrained=False)
for param in model_conv.parameters():
    param.requires_grad = False

num_ftrs = model_conv.classifier[6].in_features
model_conv.classifier[6] = nn.Linear(num_ftrs, 2)
model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_conv.parameters(), lr=0.001, momentum=0.9)
# Decay LR by a factor of 0.1 every 7 epochs
# scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience = 5, verbose = True)


def train_model(model, criterion, optimizer, scheduler, num_epochs=15):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
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

                epoch_loss = running_loss / dataset_sizes['train']
                epoch_acc = running_corrects.double() / dataset_sizes['train']

                scheduler.step(epoch_loss)

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                writer.add_scalar('training loss', epoch_loss, epoch)
                writer.add_scalar('train Accuracy', epoch_acc, epoch)

            else: # evaluation loop
                model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for i, (inputs, labels) in enumerate(val_loader):
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
                
                epoch_loss = running_loss / dataset_sizes['val']
                epoch_acc = running_corrects.double() / dataset_sizes['val']

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
    return model, epoch_loss, epoch_acc, best_acc


def eval_model(model, num_classes, data_loader):
    # init tensor to model outputs and targets
    eval_targets = torch.empty(0, device=device)
    eval_predictions = torch.empty(0, device=device)

    # initialize metric
    eval_metric_acc = torchmetrics.Accuracy().to(device) # accuracy

    # 'macro': Calculate the metric for each class separately, and average the metrics across classes (with equal weights for each class).
    eval_metric_weighted_acc = torchmetrics.Accuracy(num_classes=num_classes, average='macro').to(device) # weigthed accuracy

    # evaluate the model
    model.eval()  # Set model to evaluate mode

    # iterate over data of the epoch (evaluation)
    for batch_id, (inputs, labels) in enumerate(data_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)


        # forward
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

        # store batch model outputs and targets
        eval_predictions = torch.cat((eval_predictions, preds.data))
        eval_targets = torch.cat((eval_targets, labels.data))

        # compute batch evaluation metric
        eval_metric_acc.update(preds, labels.data)
        eval_metric_weighted_acc.update(preds, labels.data)

    # compute metrics for complete data
    eval_acc = eval_metric_acc.compute().item()
    eval_weighted_acc = eval_metric_weighted_acc.compute().item()

    return eval_acc, eval_weighted_acc, eval_predictions, eval_targets


# train model
model, epoch_loss, epoch_acc, best_acc = train_model(model=model_conv,
                                                    criterion=criterion,
                                                    optimizer=optimizer,
                                                    scheduler=plateau_scheduler,
                                                    num_epochs=num_epochs)
""" # show/store learning curves
plt.plot(epoch_loss)
plt.legend(['train'])
plt.title('Loss')
plt.savefig(act_result_path + 'plots/loss' + '.png')
plt.close()
# plt.show()

plt.plot(epoch_acc)
plt.legend(['train'])
plt.title('Acc')
plt.savefig(act_result_path + 'plots/acc' + '.png')
plt.close()
# plt.show()
plt.close()
# plt.show()

# store model
torch.save(model, act_result_path + '_model.pth')

# eval model on test data
eval_acc, eval_weighted_acc, eval_predictions, eval_targets = eval_model(model=model, num_classes=num_classes, data_loader=dataloaders['test'])

eval_targets = eval_targets.cpu()
eval_predictions = eval_predictions.cpu()
target_classes = np.unique(eval_targets)
pred_classes = np.unique(eval_predictions)
eval_classes = np.union1d(target_classes, pred_classes)
eval_class_names = [class_names[int(x)] for x in eval_classes]

print('Got ' + str(len(target_classes)) + ' target classes')
print('Got ' + str(len(pred_classes)) + ' prediction classes')
print('Resulting in ' + str(len(eval_classes)) + ' total classes')
print(eval_class_names)

save_dict = {'train_acc': epoch_acc,
            'train_loss': epoch_loss,
            'test_acc': eval_acc,
            'test_weighted_acc': eval_weighted_acc,
            'test_predictions': eval_predictions,
            'test_targets': eval_targets,
            'class_names': class_names}
save_filename = 'results' + '.pkl'

outfile = open(act_result_path + save_filename, 'wb')
pkl.dump(save_dict, outfile)
outfile.close()

writer.flush()
writer.close()
 """