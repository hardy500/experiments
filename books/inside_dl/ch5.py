#%%
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms

from torch.utils.data import Dataset, DataLoader

from tqdm.autonotebook import tqdm

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

import pandas as pd

from sklearn.metrics import accuracy_score

import time

from idlmam import train_simple_network, Flatten, weight_reset, set_seed, run_epoch, train_network
#%%


device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 50
B = 256
train_data = torchvision.datasets.FashionMNIST("./data", train=True, transform=transforms.ToTensor(), download=True)
test_data = torchvision.datasets.FashionMNIST("./data", train=False, transform=transforms.ToTensor(), download=True)

train_loader = DataLoader(train_data, batch_size=B, shuffle=True)
test_loader = DataLoader(test_data, batch_size=B)

D = 28*28
n = 128
C = 1
classes = 10

fc_model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(D, n),
    nn.Tanh(),
    nn.Linear(n, n),
    nn.Tanh(),
    nn.Linear(n, n),
    nn.Tanh(),
    nn.Linear(n, classes),
)

fc_model.apply(weight_reset)
lr = 0.1
lr_min = 0.0001 # disired final learning rate
#gamma_expo = (lr_min/lr)**(1/epochs) # compute gamme that result in lr_min
train_sub_set, val_sub_set = torch.utils.data.random_split(train_data, [int(len(train_data)*0.8), int(len(train_data)*0.2)])
train_sub_loader = DataLoader(train_sub_set, batch_size=B, shuffle=True)
val_sub_loader = DataLoader(val_sub_set, batch_size=B)

loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(fc_model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10)

#%%

fc_result = train_network(
    fc_model,
    loss_func,
    train_loader=train_loader,
    val_loader=val_sub_loader,
    test_loader=test_loader,
    epochs=epochs,
    optimizer=optimizer,
    lr_schedule=scheduler,
    score_funcs={'Accuracy': accuracy_score},
    device=device
)
# %%
sns.lineplot(x='epoch', y='val Accuracy', data=fc_result, label='FC');

# %%
