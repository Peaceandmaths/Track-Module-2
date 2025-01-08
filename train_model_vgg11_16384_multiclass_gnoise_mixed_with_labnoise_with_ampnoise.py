import numpy as np
import pandas as pd
import os
import time
import copy
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import lib.model_VGG
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from tqdm import tqdm  # progressbar
import torchmetrics
import pickle as pkl


class DroneSignalsDataset(Dataset):
    """
    Class for custom dataset of drone data comprised of
    x (torch.tensor.float): signals (n_samples x 2 x input_vec_length)
    y (torch.tensor.long): targets (n_samples)
    snrs (torch.tensor.int): SNRs per sample (n_samples) 
    Args:
        Dataset (torch tensor): 
    """
    def __init__(self, x_tensor, y_tensor, snr_tensor):
        self.x = x_tensor
        self.y = y_tensor
        self.snr = snr_tensor

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.snr[idx]  

    def targets(self):
        return self.y 

    def snrs(self):
        return self.snr
    


def eval_model(model, num_classes, data_loader):
    # init tensor to model outputs and targets
    eval_targets = torch.empty(0, device=device)
    eval_predictions = torch.empty(0, device=device)
    eval_snrs = torch.empty(0, device=device)

    # initialize metric
    eval_metric_acc = torchmetrics.Accuracy().to(device) # accuracy

    # 'macro': Calculate the metric for each class separately, and average the metrics across classes (with equal weights for each class).
    eval_metric_weighted_acc = torchmetrics.Accuracy(num_classes=num_classes, average='macro').to(device) # weigthed accuracy

    # evaluate the model
    model.eval()  # Set model to evaluate mode

    # iterate over data of the epoch (evaluation)
    for batch_id, (inputs, labels, snrs) in enumerate(data_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        snrs = snrs.to(device)

        # forward
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

        # store batch model outputs and targets
        eval_predictions = torch.cat((eval_predictions, preds.data))
        eval_targets = torch.cat((eval_targets, labels.data))
        eval_snrs = torch.cat((eval_snrs, snrs.data))

        # compute batch evaluation metric
        eval_metric_acc.update(preds, labels.data)
        eval_metric_weighted_acc.update(preds, labels.data)

    # compute metrics for complete data
    eval_acc = eval_metric_acc.compute().item()
    eval_weighted_acc = eval_metric_weighted_acc.compute().item()

    return eval_acc, eval_weighted_acc, eval_predictions, eval_targets, eval_snrs


def get_model(model_name, num_classes):
    if(model_name == 'vgg11'):
        return lib.model_VGG.vgg11(num_classes=num_classes)
    elif(model_name == 'vgg11_bn'):
        return lib.model_VGG.vgg11_bn(num_classes=num_classes)
    elif(model_name == 'vgg13'):
        return lib.model_VGG.vgg13(num_classes=num_classes)
    elif(model_name == 'vgg13_bn'):
        return lib.model_VGG.vgg13_bn(num_classes=num_classes)
    elif(model_name == 'vgg16'):
        return lib.model_VGG.vgg16(num_classes=num_classes)
    elif(model_name == 'vgg16_bn'):
        return lib.model_VGG.vgg16_bn(num_classes=num_classes)
    elif(model_name == 'vgg19'):
        return lib.model_VGG.vgg13(num_classes=num_classes)
    elif(model_name == 'vgg19_bn'):
        return lib.model_VGG.vgg13_bn(num_classes=num_classes)
    else:
        print('Error: no valid model name:', model_name)
        exit()


def train_model_observe_snr_performance(model, criterion, optimizer, scheduler, num_classes, num_epochs, snr_list_for_observation):
    
    since = time.time()
    train_loss = []
    train_acc = []
    train_weighted_acc = []
    lr = []

    val_loss = []
    val_acc = []
    val_weighted_acc = []

    # create variables to store acc for different SNR samples
    num_snrs_to_observe = len(snr_list_for_observation)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0

    print('start training')
    print('-' * 10)

    for epoch in range(num_epochs):
        # initialize metric
        # accuracy
        train_metric_acc = torchmetrics.Accuracy().to(device)
        val_metric_acc = torchmetrics.Accuracy().to(device)

        # weigthed accuracy
        # 'macro': Calculate the metric for each class separately, and average the metrics across classes (with equal weights for each class).
        train_metric_weighted_acc = torchmetrics.Accuracy(num_classes=num_classes, average='macro').to(device)
        val_metric_weighted_acc = torchmetrics.Accuracy(num_classes=num_classes, average='macro').to(device)

        # snr dependent accuracies metrics
        snr_val_metric_acc_list =[torchmetrics.Accuracy().to(device) for i in range(num_snrs_to_observe)]
        snr_val_metric_weighted_acc_list =[torchmetrics.Accuracy(num_classes=num_classes, average='macro').to(device) for i in range(num_snrs_to_observe)]
        
        # snr dependent accuracies storage for epoch
        snr_epoch_acc = torch.zeros([num_snrs_to_observe], dtype=torch.float)
        snr_epoch_weighted_acc = torch.zeros([num_snrs_to_observe], dtype=torch.float)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            # phase = 'train'
            if phase == 'train':
                model.train()  # Set model to training mode
                running_loss = 0.0
                epoch_train_loop = tqdm(dataloaders[phase])  # setup progress bar

                # iterate over data of the epoch (training)
                # inputs, labels, snrs = next(iter(epoch_train_loop))
                for batch_id, (inputs, labels, snrs) in enumerate(epoch_train_loop):                                     
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    # add model graph to tensorboard
                    if (batch_id==0) & (epoch==0):
                        writer.add_graph(model, inputs)

                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(True):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        loss.backward()
                        optimizer.step()

                    # compute scores for the epoch
                    running_loss += loss.item() * inputs.size(0)

                    # compute scores for batch
                    train_metric_acc.update(preds, labels.data)
                    train_metric_weighted_acc.update(preds, labels.data)

                    # print(f"Accuracy on batch: {batch_train_acc}")
                    if train_verbose:
                        # show progress bar for the epoch
                        epoch_train_loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
                        # epoch_train_loop.set_postfix(loss=loss.item(), acc=torch.sum(preds == labels.data).item()/batch_size)
                        epoch_train_loop.set_postfix()

                # apply learning rate scheduler after training epoch (for exp_lr_scheduler
                # scheduler.step()

                # compute and show metrics for the epoch
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = train_metric_acc.compute().item()
                epoch_weighted_acc = train_metric_weighted_acc.compute().item()

                print('{} Loss: {:.4f} Acc: {:.4f}  Balanced Acc: {:.4f} |'.format(phase, epoch_loss, epoch_acc, epoch_weighted_acc), end=' ')

                # store metric for epoch
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
                train_weighted_acc.append(epoch_weighted_acc)
                lr.append(optimizer.param_groups[0]['lr'])

                # add to tensor board
                writer.add_scalar('Loss/train', epoch_loss, epoch)
                writer.add_scalar('Accuracy/train', epoch_acc, epoch)
                writer.add_scalar('BalancedAccuracy/train', epoch_weighted_acc, epoch)
                writer.add_scalar('Learnigrate', optimizer.param_groups[0]['lr'], epoch)
            else:
                # phase = 'val'
                model.eval()   # Set model to evaluate mode
                running_loss = 0.0

                # iterate over data of the epoch (evaluation)
                inputs, labels, snrs =  next(iter(dataloaders[phase]))
                for batch_id, (inputs, labels, snrs) in enumerate(dataloaders[phase]):
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    snrs = snrs.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    with torch.set_grad_enabled(False):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                    # statistics
                    running_loss += loss.item() * inputs.size(0)

                    # compute scores for batch
                    val_metric_acc.update(preds, labels.data)
                    val_metric_weighted_acc.update(preds, labels.data)

                    # compute accuracies for diffrent SNRs
                    for i, snr in enumerate(snr_list_for_observation):
                        act_snr_sample_indices = torch.where(snrs == snr)[0]
                        if act_snr_sample_indices.size(0) > 0: # if there are some samples with current SNR
                            snr_val_metric_acc_list[i].update(preds[act_snr_sample_indices], labels.data[act_snr_sample_indices])
                            snr_val_metric_weighted_acc_list[i].update(preds[act_snr_sample_indices], labels.data[act_snr_sample_indices])
                            
                # compute and show metrics for the epoch
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = val_metric_acc.compute().item()
                epoch_weighted_acc = val_metric_weighted_acc.compute().item()

                for i in range(num_snrs_to_observe):
                    snr_epoch_acc[i] = snr_val_metric_acc_list[i].compute().item()
                    snr_epoch_weighted_acc[i] = snr_val_metric_weighted_acc_list[i].compute().item()

                # apply LR scheduler ... looking for plateau in val loss
                scheduler.step(epoch_loss)

                print('{} Loss: {:.4f} Acc: {:.4f}  Balanced Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc, epoch_weighted_acc))
                # store validation loss
                val_loss.append(epoch_loss)
                val_acc.append(epoch_acc)
                val_weighted_acc.append(epoch_weighted_acc)

                # add to tensor board
                writer.add_scalar('Loss/val', epoch_loss, epoch)
                writer.add_scalar('Accuracy/val', epoch_acc, epoch)
                writer.add_scalar('BalancedAccuracy/val', epoch_weighted_acc, epoch)

                # SNR measures to tensorboard
                for i, snr in enumerate(snr_list_for_observation):
                    writer.add_scalar('SNR/val Accuracy SNR' + str(snr), snr_epoch_acc[i], epoch)
                    writer.add_scalar('SNR/val BalancedAccuracy SNR' + str(snr), snr_epoch_weighted_acc[i], epoch)

                #     best_epoch = epoch
                if epoch_weighted_acc > best_acc:
                    best_acc = epoch_weighted_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_epoch = epoch

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('Best epoch: {}'.format(best_epoch))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, train_loss, train_acc, val_loss, val_acc, train_weighted_acc, val_weighted_acc, best_epoch, lr


project_path = '/home/glue/work/drones/'
result_path = project_path + 'results/experiments/'
store_data_path = '/data/glue/drones/preprocessed/'

# params for data preprocessing
signal_freq = 2440  # frequenzy of the measurments to use in MHz (5.8GHz 2.44GHz od 869MHz )
sampling_rate = 14 # downsample frequency in MHz
input_vec_length = 16384 # lenght of the input vector

# create string to be used for filnames for this dataset
file_name_extension = 'sigfreq_' + str(signal_freq) + \
    '_samplefreq_' + str(sampling_rate) + \
    '_inputlength_' + str(input_vec_length) + \
    '_SNR_all_gnoise_mixed_with_labnoise_with_ampnoise'

# global params
num_workers = 12  # for the data loader
num_folds = 5
num_epochs = 25
batch_size = 64
learning_rate = 0.005
train_verbose = True  # show epoch progressbar
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_name = 'vgg11'

# on G001 devices:
# cuda:0-2 = A100
# cuda: = P100
# </editor-fold">

experiment_name = model_name + \
                '_CV' + str(num_folds) +\
                '_epochs' + str(num_epochs) + \
                '_lr' + str(learning_rate) + \
                '_batchsize' + str(batch_size) + \
                '_freq' + str(signal_freq) + \
                '_input' + str(input_vec_length) + \
                '_observeSNR_multiclass_gnoise_mixed_with_labnoise_with_ampnoise'


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


# read statistics/class count of the dataset
dataset_stats = pd.read_csv(store_data_path + 'class_stats_' + file_name_extension + '.csv', index_col=0)
class_names = dataset_stats['class'].values

# read SNR count of the dataset
snr_stats = pd.read_csv(store_data_path + 'SNR_stats_' + file_name_extension + '.csv', index_col=0)
snr_list = snr_stats['SNR'].values

# load data
dataset_dict = torch.load(store_data_path + 'dataset_' + file_name_extension + '.pt')
dataset_dict.keys()

x = dataset_dict['x']
y = dataset_dict['y']
snrs = dataset_dict['snr']

# x.type()
# y.type()
# snrs.type()

# create pytorch dataset form tensors
dataset = DroneSignalsDataset(x,y,snrs)
del(x,y,snrs,dataset_dict)


# i,t = next(iter(dataset))
# split data with stratified kfold
dataset_indices = list(range(len(dataset)))

# split test data from train data in stratified k-fold manner
num_folds = 5

fold = 0
# Tensorboard writer will output to ./runs/ directory by default
writer = SummaryWriter(act_result_path + 'runs/fold' + str(fold))

train_idx, test_idx = train_test_split(dataset_indices, test_size=1/num_folds, stratify=dataset.targets())
# 20% as test set, create a list of data points => create dataloaders from this list 
# np.unique(train_image_dataset.targets, return_counts=True)[1]
y_test = [dataset.targets()[x] for x in test_idx]
y_train = [dataset.targets()[x] for x in train_idx]

# split val data from train data in stratified k-fold manner
train_idx, val_idx = train_test_split(train_idx, test_size=1/num_folds, stratify=y_train)
y_val = [dataset.targets()[x] for x in val_idx]
y_train = [dataset.targets()[x] for x in train_idx]

# print(np.unique(y_test, return_counts=True)[1])
# print(np.unique(y_val, return_counts=True)[1])
# print(np.unique(y_train, return_counts=True)[1])

# get train samples weight by class weight for each train target
class_weights = 1. / dataset_stats['count']

train_samples_weight = np.array([class_weights[int(i)] for i in y_train])
train_samples_weight = torch.from_numpy(train_samples_weight)

train_dataset = torch.utils.data.Subset(dataset, train_idx)
val_dataset = torch.utils.data.Subset(dataset, val_idx)
test_dataset = torch.utils.data.Subset(dataset, test_idx)

# define weighted random sampler with the weighted train samples
train_sampler = torch.utils.data.WeightedRandomSampler(train_samples_weight.type('torch.DoubleTensor'), len(train_samples_weight))

# define data loaders
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    sampler=train_sampler,
    # shuffle=True,
    num_workers=12,
    pin_memory=True)

# num_workers (int, optional) – how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: 0)
# pin_memory: If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
# Setting num_workers > 0 enables asynchronous data loading and overlap between the training and data loading. num_workers should be tuned depending on the workload, CPU, GPU, and location of training data.
# DataLoader accepts pin_memory argument, which defaults to False. When using a GPU it’s better to set pin_memory=True, this instructs DataLoader to use pinned memory and enables faster and asynchronous memory copy from the host to the GPU.

val_loader = torch.utils.data.DataLoader(
  dataset=val_dataset,
  batch_size=batch_size,
  shuffle=True,
  num_workers=12,
  pin_memory=True)

test_loader = torch.utils.data.DataLoader(
  dataset=test_dataset,
  batch_size=batch_size,
  shuffle=False,
  num_workers=12,
  pin_memory=True)

dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset), 'test': len(test_dataset)}

num_classes = len(np.unique(y_val))

# model = lib.model_VGG.vgg11_bn(num_classes=num_classes)
model = get_model(model_name, num_classes)

model = model.to(device)
# criterion = nn.CrossEntropyLoss(weight=torch.Tensor(class_weights).to(device))
criterion = nn.CrossEntropyLoss()  # don't use class weights in the loss

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
plateau_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.1, patience=3, threshold=0.0001,
                                                    threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True)

# # train model
# model, train_loss, train_acc, val_loss, val_acc, train_weighted_acc, val_weighted_acc, best_epoch, lr = train_model(model=model,
#                                                                                                                        criterion=criterion,
#                                                                                                                        optimizer=optimizer_ft,
#                                                                                                                        scheduler=plateau_scheduler,
#                                                                                                                        num_classes=num_classes,
#                                                                                                                        num_epochs=num_epochs)

# train model
model, train_loss, train_acc, val_loss, val_acc, train_weighted_acc, val_weighted_acc, best_epoch, lr = train_model_observe_snr_performance(model=model,
                                                                                                                       criterion=criterion,
                                                                                                                       optimizer=optimizer_ft,
                                                                                                                       scheduler=plateau_scheduler,
                                                                                                                       num_classes=num_classes,
                                                                                                                       num_epochs=num_epochs,
                                                                                                                       snr_list_for_observation=[0, -10, -20])


# show/store learning curves
plt.plot(train_loss)
plt.plot(val_loss)
plt.legend(['train', 'val'])
plt.title('Loss')
plt.savefig(act_result_path + 'plots/loss_fold' + str(fold) + '.png')
plt.close()
# plt.show()

plt.plot(train_acc)
plt.plot(val_acc)
plt.legend(['train', 'val'])
plt.title('Acc')
plt.savefig(act_result_path + 'plots/acc_fold' + str(fold) + '.png')
plt.close()
# plt.show()

plt.plot(train_weighted_acc)
plt.plot(val_weighted_acc)
plt.legend(['train', 'val'])
plt.title('Weighted Acc')
plt.savefig(act_result_path + 'plots/weigthed_acc_fold' + str(fold) + '.png')
plt.close()
# plt.show()

# store model
torch.save(model, act_result_path + 'model.pth')

# eval model on test data
eval_acc, eval_weighted_acc, eval_predictions, eval_targets, eval_snrs = eval_model(model=model, num_classes=num_classes, data_loader=dataloaders['test'])

eval_targets = eval_targets.cpu()
eval_predictions = eval_predictions.cpu()
eval_snrs = eval_snrs.cpu()
target_classes = np.unique(eval_targets)
pred_classes = np.unique(eval_predictions)
eval_classes = np.union1d(target_classes, pred_classes)
eval_class_names = [class_names[int(x)] for x in eval_classes]

print('Got ' + str(len(target_classes)) + ' target classes')
print('Got ' + str(len(pred_classes)) + ' prediction classes')
print('Resulting in ' + str(len(eval_classes)) + ' total classes')
print(eval_class_names)

save_dict = {'train_weighted_acc': train_weighted_acc,
                'train_acc': train_acc,
                'train_loss': train_loss,
                'val_weighted_acc': val_weighted_acc,
                'val_acc': val_acc,
                'val_loss': val_loss,
                'test_acc': eval_acc,
                'test_weighted_acc': eval_weighted_acc,
                'test_predictions': eval_predictions,
                'test_targets': eval_targets,
                'test_snrs': eval_snrs,
                'class_names': class_names
                }
save_filename = 'results_fold' + str(fold) + '.pkl'

outfile = open(act_result_path + save_filename, 'wb')
pkl.dump(save_dict, outfile)
outfile.close()