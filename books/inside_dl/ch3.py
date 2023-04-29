#%%
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms

from idlmam import train_simple_network, set_seed
import time

from sklearn.metrics import accuracy_score

mnist_data_train = torchvision.datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
mnist_data_test = torchvision.datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())

#%%

device = 'cuda' if torch.cuda.is_available() else 'cpu'
B = 32

train_loader = DataLoader(mnist_data_train, batch_size=B, shuffle=True)
test_loader = DataLoader(mnist_data_test, batch_size=B)
#%%

D = 28*28
C = 1
classes = 10
filters = 16
K = 3

model_linear = nn.Sequential(
    nn.Flatten(), # (B, C, W, H) -> (B, C*W*H) = (B, D)
    nn.Linear(D, 256),
    nn.Tanh(),
    nn.Linear(256, classes)
)

model_cnn = nn.Sequential(
    nn.Conv2d(in_channels=C, out_channels=filters, kernel_size=K, padding=K//2),
    nn.Tanh(),
    nn.Flatten(),  # (B, C, W, H) -> (B, D)
    nn.Linear(filters*D, classes)
)

model_cnn_pool = nn.Sequential(
    nn.Conv2d(C, filters, 3, padding=3//2),
    nn.Tanh(),
    nn.Conv2d(filters, filters, 3, padding=3//2),
    nn.Tanh(),
    nn.Conv2d(filters, filters, 3, padding=3//2),
    nn.Tanh(),
    nn.MaxPool2d(2),
    nn.Conv2d(filters, 2*filters, 3, padding=3//2),
    nn.Tanh(),
    nn.Conv2d(2*filters, 2*filters, 3, padding=3//2),
    nn.Tanh(),
    nn.Conv2d(2*filters, 2*filters, 3, padding=3//2),
    nn.Tanh(),
    nn.MaxPool2d(2),

    nn.Flatten(),
    #Why did we reduce the number of units into the Linear layer by a factor of $4^2$? Because pooling a 2x2 grid down to one value means we go from 4 values, down to 1, and we did this two times.
    nn.Linear(2*filters*D//(4**2), classes),
)


loss_func = nn.CrossEntropyLoss()
# %%
cnn_results = train_simple_network(
    model_cnn,
    loss_func,
    train_loader,
    test_loader,
    score_funcs={'Accuracy': accuracy_score},
    device=device,
    epochs=20
)

#%%
cnn_pool_results = train_simple_network(
    model_cnn_pool,
    loss_func,
    train_loader,
    test_loader,
    score_funcs={'Accuracy': accuracy_score},
    device=device,
    epochs=20
)


#%%

fc_results = train_simple_network(
    model_linear,
    loss_func,
    train_loader,
    test_loader,
    score_funcs={'Accuracy': accuracy_score},
    device=device,
    epochs=20
)

# %%
import seaborn as sns

sns.lineplot(x='epoch', y='test Accuracy', data=cnn_results, label='CNN')
sns.lineplot(x='epoch', y='test Accuracy', data=fc_results, label='Linear');
sns.lineplot(x='epoch', y='test Accuracy', data=cnn_pool_results, label='CNN_POOL');

# %%
