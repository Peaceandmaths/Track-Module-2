import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from tqdm import tqdm  # progressbar
import torchmetrics
import pickle as pkl
from torchvision import models
from sklearn.model_selection import train_test_split
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # prevents "OSError: broken data stream when reading image file" in some cases
from torch.utils.tensorboard import SummaryWriter

# <editor-fold desc="Helper functions">
# </editor-fold">


# <editor-fold desc="model training and evaluation functions">

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

# model = model
# optimizer = optimizer_ft
# scheduler = exp_lr_scheduler
# num_epochs = 5
# epoch=0
# phase = 'train'


def train_model(model, criterion, optimizer, scheduler, num_classes, num_epochs):
    since = time.time()
    train_loss = []
    train_acc = []
    train_weighted_acc = []
    lr = []

    val_loss = []
    val_acc = []
    val_weighted_acc = []

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

        train_metric_precision = torchmetrics.Precision(num_classes=num_classes, average='macro').to(device)
        val_metric_precision = torchmetrics.Precision(num_classes=num_classes, average='macro').to(device)

        train_metric_recall = torchmetrics.Recall(num_classes=num_classes, average='macro').to(device)
        val_metric_recall = torchmetrics.Recall(num_classes=num_classes, average='macro').to(device)

        # weigthed accuracy
        # 'macro': Calculate the metric for each class separately, and average the metrics across classes (with equal weights for each class).
        train_metric_weighted_acc = torchmetrics.Accuracy(num_classes=num_classes, average='macro').to(device)
        val_metric_weighted_acc = torchmetrics.Accuracy(num_classes=num_classes, average='macro').to(device)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:

            if phase == 'train':
                model.train()  # Set model to training mode

                running_loss = 0.0

                epoch_train_loop = tqdm(dataloaders[phase])  # setup progress bar

                # iterate over data of the epoch (training)
                # (inputs, labels) = next(iter(epoch_train_loop))
                for batch_id, (inputs, labels) in enumerate(epoch_train_loop):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(True):
                        outputs = model(inputs)
                        # outputs.shape
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        loss.backward()
                        optimizer.step()

                    # compute scores for the epoch
                    running_loss += loss.item() * inputs.size(0)

                    # compute scores for batch
                    train_metric_acc.update(preds, labels.data)
                    train_metric_weighted_acc.update(preds, labels.data)
                    train_metric_precision.update(preds, labels.data)
                    train_metric_recall.update(preds, labels.data)

                    # print(f"Accuracy on batch: {batch_train_acc}")
                    if train_verbose:
                        # show progress bar for the epoch
                        epoch_train_loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
                        # epoch_train_loop.set_postfix(loss=loss.item(), acc=torch.sum(preds == labels.data).item()/batch_size)
                        epoch_train_loop.set_postfix()

                # apply learning rate scheduler after training epoch (for exp_lr_scheduler)
                # scheduler.step()

                # compute and show metrics for the epoch
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = train_metric_acc.compute().item()
                epoch_weighted_acc = train_metric_weighted_acc.compute().item()
                epoch_precision = train_metric_precision.compute().item()
                epoch_recall = train_metric_recall.compute().item()

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
                writer.add_scalar('Precision/train', epoch_precision, epoch)
                writer.add_scalar('Recall/train', epoch_recall, epoch)
                writer.add_scalar('Learnigrate', optimizer.param_groups[0]['lr'], epoch)

            else:
                model.eval()   # Set model to evaluate mode
                running_loss = 0.0

                # iterate over data of the epoch (evaluation)
                for batch_id, (inputs, labels) in enumerate(dataloaders[phase]):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

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
                    val_metric_precision.update(preds, labels.data)
                    val_metric_recall.update(preds, labels.data)


                # compute and show metrics for the epoch
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = val_metric_acc.compute().item()
                epoch_weighted_acc = val_metric_weighted_acc.compute().item()
                epoch_precision = val_metric_precision.compute().item()
                epoch_recall = val_metric_recall.compute().item()

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
                writer.add_scalar('Precision/val', epoch_precision, epoch)
                writer.add_scalar('Recall/val', epoch_recall, epoch)

                # save best validation model
                # deep copy the model
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
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
# </editor-fold">

cudnn.benchmark = True
# plt.ion()   # interactive mode

# dataset color channel means and std for the BM_cytomorphology_data (computed in comp_channel_means_std.py)
color_means = torch.tensor([0.5630, 0.4959, 0.7353])
color_std = torch.tensor([0.2421, 0.2835, 0.1767])

# MacOS
# project_path = '/Users/glue/work/usz_heme/heme_prototype/heme_01/'
# img_path = project_path + 'data/raw/'

# g001
project_path = '/home/glue/work/heme_01/'
img_path = '/data/glue/heme/data/raw/'

# # gcloud
# project_path = '/home/glue/work/heme/'
# img_path = '/home/glue/work/heme/data/raw/'

result_path = project_path + 'results/experiments/'

# load dataset stats
dataset_stats_path = img_path + 'dataset_class_stats.pkl'
infile = open(dataset_stats_path, 'rb')
dataset_stats_dict = pkl.load(infile)
infile.close()
class_weights_balanced = dataset_stats_dict['class_weights']  # class weights precomuted with sklearns compute_class_weight() function
class_counts = dataset_stats_dict['class_counts']  # class counts precomuted

# <editor-fold desc="create datasets with/ without augmentations">
input_size = 224
train_transform = data_transform = transforms.Compose([
    transforms.RandomResizedCrop(input_size),
    transforms.RandomRotation(degrees=(0, 180)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(color_means, color_std)
])

val_transform = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor(),
    transforms.Normalize(color_means, color_std)
])

# create dataset of original images, one with augmentation operations, one without
image_dataset_augmented = datasets.ImageFolder(os.path.join(img_path, 'BM_cytomorphology_data'), train_transform)
image_dataset = datasets.ImageFolder(os.path.join(img_path, 'BM_cytomorphology_data'), val_transform)

# get number of classes
dataset_size = len(image_dataset_augmented)
class_names = image_dataset_augmented.classes
num_classes = len(class_names)

# </editor-fold">

# <editor-fold desc="define training parameter">
num_workers = 12  # for the data loader
split_num_folds = 5  # folds of stratifies train/test split
num_folds = 5  # experiment runs ... should match split_num_folds in the final experiment
num_epochs = 70
batch_size = 32
learning_rate = 0.004
train_verbose = True  # show epoch progressbar
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# on G001 devices:
# cuda:0-1 = A100
# cuda: = P100
# </editor-fold">


# split data with stratified kfold
dataset_indices = list(range(dataset_size))

experiment_name = 'regnet_fromScratch' + \
                  '_CV' + str(num_folds) +\
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

for fold in range(num_folds):  # just a single run
    print('CV:', fold)

    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter(act_result_path + 'runs/fold' + str(fold))

    # split test data from train data in stratified k-fold manner
    train_idx, test_idx = train_test_split(dataset_indices, test_size=1/split_num_folds, stratify=image_dataset.targets)
    # np.unique(train_image_dataset.targets, return_counts=True)[1]
    y_test = [image_dataset.targets[x] for x in test_idx]
    y_train = [image_dataset.targets[x] for x in train_idx]

    # print(np.unique(y_test, return_counts=True)[1])
    # if fold == 0:
    #     y_test_class_counts = np.unique(y_test, return_counts=True)[1]
    # else:
    #     y_test_class_counts += np.unique(y_test, return_counts=True)[1]

    # split val data from train data in stratified k-fold manner
    train_idx, val_idx = train_test_split(train_idx, test_size=1/split_num_folds, stratify=y_train)
    y_val = [image_dataset.targets[x] for x in val_idx]
    y_train = [image_dataset.targets[x] for x in train_idx]

    len(y_train)
    len(y_test)
    len(y_val)

    # get train samples weight by class weight for each train target
    class_weights = 1. / class_counts
    train_samples_weight = np.array([class_weights[i] for i in y_train])
    train_samples_weight = torch.from_numpy(train_samples_weight)

    train_dataset = torch.utils.data.Subset(image_dataset_augmented, train_idx)
    val_dataset = torch.utils.data.Subset(image_dataset, val_idx)
    test_dataset = torch.utils.data.Subset(image_dataset, test_idx)


    # define weighted random sampler with the weighted train samples
    train_sampler = torch.utils.data.WeightedRandomSampler(train_samples_weight.type('torch.DoubleTensor'), len(train_samples_weight))


    # define data loaders
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        # shuffle=True,
        num_workers=num_workers,
        pin_memory=True)

    # num_workers (int, optional) – how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: 0)
    # pin_memory: If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
    # Setting num_workers > 0 enables asynchronous data loading and overlap between the training and data loading. num_workers should be tuned depending on the workload, CPU, GPU, and location of training data.
    # DataLoader accepts pin_memory argument, which defaults to False. When using a GPU it’s better to set pin_memory=True, this instructs DataLoader to use pinned memory and enables faster and asynchronous memory copy from the host to the GPU.

    val_loader = torch.utils.data.DataLoader(
      dataset=val_dataset,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
      dataset=test_dataset,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=True)

    dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset), 'test': len(test_dataset)}

    # <editor-fold desc="train model">
    model = models.regnet_y_32gf(pretrained=False)

    num_ftrs = model.fc.in_features
    # Here the size of each output sample the number of classes
    model.fc = nn.Linear(num_ftrs, len(class_names))

    model = model.to(device)

    # criterion = nn.CrossEntropyLoss(weight=torch.Tensor(class_weights).to(device))
    criterion = nn.CrossEntropyLoss()  # don't use class weights in the loss

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    plateau_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.1, patience=3, threshold=0.0001,
                                               threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True)
    # train model
    model, train_loss, train_acc, val_loss, val_acc, train_weighted_acc, val_weighted_acc, best_epoch, lr = train_model(model=model,
                                                                                                                       criterion=criterion,
                                                                                                                       optimizer=optimizer_ft,
                                                                                                                       scheduler=plateau_scheduler,
                                                                                                                       num_classes=num_classes,
                                                                                                                       num_epochs=num_epochs)

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
    torch.save(model, act_result_path + 'fold' + str(fold) + '_model.pth')

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

    save_dict = {'train_weighted_acc': train_weighted_acc,
                 'train_acc': train_acc,
                 'train_loss': train_loss,
                 'lr': lr,
                 'val_weighted_acc': val_weighted_acc,
                 'val_acc': val_acc,
                 'val_loss': val_loss,
                 'test_acc': eval_acc,
                 'test_weighted_acc': eval_weighted_acc,
                 'test_predictions': eval_predictions,
                 'test_targets': eval_targets,
                 'class_names': class_names
                 }
    save_filename = 'results_fold' + str(fold) + '.pkl'

    outfile = open(act_result_path + save_filename, 'wb')
    pkl.dump(save_dict, outfile)
    outfile.close()

    writer.flush()
    writer.close()