import os
import random

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import torch
from torch.utils.data import TensorDataset

class CatDogDataset(Dataset):
  def __init__(self, img_paths, transform):
    super().__init__()
    self.paths = img_paths
    self.transform = transform

  def __len__(self):
    return len(self.paths)

  def __getitem__(self, idx):
    path = self.paths[idx]
    img = Image.open(path).convert('RGB')
    img = self.transform(img)
    label = 0 if 'cat' in path.split('/')[-1] else 1
    return (img, label)

def get_data():
  #SPLIT = 24000
  SPLIT = 300
  BS = 64
  IMG_SIZE = 256

  img_files = os.listdir('data/dogvcat/train')
  img_files = ["data/dogvcat/train/" + i for i in img_files]
  print(len(img_files))

  # image normalization
  transform = transforms.Compose([
      transforms.Resize((IMG_SIZE, IMG_SIZE)),
      transforms.CenterCrop((IMG_SIZE, IMG_SIZE)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])

  # split data
  random.shuffle(img_files)
  train = img_files[:SPLIT]
  val = img_files[SPLIT:350]

  # create train dataset
  train_data = CatDogDataset(train, transform)
  train_loader = DataLoader(train_data, batch_size=BS)

  # create validation dataset
  val_data = CatDogDataset(val, transform)
  val_loader = DataLoader(val_data, batch_size=BS)

  return (train_loader, val_loader)

def preprocessed_dataset(model, loader, device=None):
  if device is None:
    device = next(model.parameters()).device

  features = None
  labels = None

  for i, (x, y) in enumerate(loader):
    model.eval()
    x = x.to(device)
    output = model(x)
    if i == 0:
      features = output.detach().cpu()
      labels = y.cpu()
    else:
      features = torch.cat([features, output.detach().cpu()])
      labels = torch.cat([labels, y.cpu()])

  dataset = TensorDataset(features, labels)
  return dataset