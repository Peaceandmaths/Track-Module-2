import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets 
import torchvision.transforms as transforms 
import torchvision


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_classes = 10
lr = 0.1 
batch_size = 1024
num_epochs = 100 

model = torchvision.models.googlenet(pretrained = True)

for param in model.parameters():
    param.requires_grad= False 

model.fc = nn.Linear(1024, num_classes)
model.to(device)


train_dataset = datasets.CIFAR10(root ='dataset/', train = True,
                                 transform = transforms.ToTensor(), download = True)

train_loader = DataLoader(dataste = train_dataset, batch_size = batch_size, shuffle = True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience = 5, verbose = True)
# if the loss doesn't decrease after 5 epochs then we decrease the lr by the factor * original lr 

for epoch in range(1, num_epochs):
    losses = []
    for batch_idx, (data,targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        scores = model(data)
        loss = criterion(scores,targets) 

        losses.append(loss.item())

        loss.backward()

        optimizer.step()
        optimizer.zero_grad() 
    
    mean_loss = sum(losses)/len(losses)
    scheduler.step(mean_loss)
    print(f'Cost at epoch {epoch} is {mean_loss}')



