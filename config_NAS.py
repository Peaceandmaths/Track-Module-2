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
from NAS_Malarial import get_data
import pickle 

train_loader, test_loader = get_data()

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


@model_wrapper
class ModelSpace(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(32, 32, 3, 1) # change based on my input 125x125x24 ? 
        # LayerChoice is used to select a layer between Conv2d and DwConv.
        self.conv2 = nn.LayerChoice([
            nn.Conv2d(32, 64, 3, 1),
            DepthwiseSeparableConv(32, 64)
        ])
        # ValueChoice is used to select a dropout rate.
        # ValueChoice can be used as parameter of modules wrapped in `nni.retiarii.nn.pytorch`
        # or customized modules wrapped with `@basic_unit`.
        self.dropout1 = nn.Dropout(nn.ValueChoice([0.25, 0.5, 0.75]))  # choose dropout rate from 0.25, 0.5 and 0.75
        self.dropout2 = nn.Dropout(0.5)
        feature = nn.ValueChoice([64, 128, 256])
        self.fc1 = nn.Linear(9216, feature)
        self.fc2 = nn.Linear(feature, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(self.conv2(x), 2)
        x = torch.flatten(self.dropout1(x), 1)
        x = self.fc2(self.dropout2(F.relu(self.fc1(x))))
        output = F.log_softmax(x, dim=1) # good for small images 
        return output


model_space = ModelSpace()
model_space

# Exploration startegy

import nni.retiarii.strategy as strategy
search_strategy = strategy.Random(dedup=True)  
# dedup=False if deduplication is not wanted

# Evaluator 

# Model evaluator 

import nni

from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader


def train_epoch(model, device, train_loader, optimizer, epoch):
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        print(data[0:10])
        optimizer.zero_grad()
        print(model(data))
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test_epoch(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
          correct, len(test_loader.dataset), accuracy))

    return accuracy

def evaluate_model(model_cls):
    # "model_cls" is a class, need to instantiate
    model = model_cls()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    transf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    for epoch in range(3):
        # train the model for one epoch
        train_epoch(model, device, train_loader, optimizer, epoch)
        # test the model for one epoch
        accuracy = test_epoch(model, device, test_loader)
        # call report intermediate result. Result can be float or dict
        nni.report_intermediate_result(accuracy)

    # report final test result
    nni.report_final_result(accuracy)


from nni.retiarii.evaluator import FunctionalEvaluator
evaluator = FunctionalEvaluator(evaluate_model) 


# Launch an experiment 

from nni.retiarii.experiment.pytorch import RetiariiExperiment
#from retiarii_class import RetiariiExeConfig
from nni.retiarii.experiment.pytorch import RetiariiExeConfig

exp = RetiariiExperiment(model_space, evaluator, [], search_strategy)
exp_config = RetiariiExeConfig('local')
exp_config.experiment_name = 'malarial_search'
exp_config.max_trial_number = 4
exp_config.trial_concurrency = 2
exp_config.trial_gpu_number = 1
exp_config.training_service.use_active_gpu = True
exp.run(exp_config, 6030)



#evaluate_model_with_visualization(model)


# Export top models 

for model_dict in exp.export_top_models(formatter='dict'):
    print(model_dict)

exp_config.execution_engine = 'base'
export_formatter = 'code'
""" 
for model_dict in exp.export_top_models(formatter='dict'):
    # Process each exported model dictionary
    print(model_dict) """



