"""
    Dataset loader, to load the clean and poisoning datasets
"""
import os
import gc
import sys
import pickle
import numpy as np

# torch/torchvision module
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


# ----------------------------------------------------------------
#   Configurations
# ----------------------------------------------------------------
CIFAR10_ROOT      = "../data"
TINYIMAGENET_ROOT = "datasets/originals/tiny_imagenet"

# ------------------------------------------------------------------------------
#   Dataset class (to load the custom data)
# ------------------------------------------------------------------------------
class DenoisedDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.pdata     = data
        self.plabels   = labels
        self.transform = transform
        print ('DenoisedDataset: read the data - {}'.format(self.pdata.shape))


    def __len__(self):
        # return the number of instances in a dataset
        return len(self.pdata)


    def __getitem__(self, idx):
        # return (data, label) where label is index of the label class
        data, label = self.pdata[idx], self.plabels[idx]

        # transform into the Pytorch format
        if self.transform:
            data = self.transform(data)
        return (data, label)

def load_denoised_cifar(augment=True, datpath=None, train_transform=None, valid_transform=None):
    # compose the transformation
    # transform_train = []
    # transform_valid = []

    # # augmentation
    # if augment:
    #     transform_train += [transforms.RandomCrop(32, padding=4),
    #                         transforms.RandomHorizontalFlip()]

    # # normalization
    # if normalize:
    #     transform_train += [transforms.Normalize((0.4914, 0.4822, 0.4465),
    #                                              (0.2023, 0.1994, 0.2010))]
    #     transform_valid += [transforms.Normalize((0.4914, 0.4822, 0.4465),
    #                                              (0.2023, 0.1994, 0.2010))]

    # extract the data
    custom_dataset = torch.load(datpath)

    # compose the dataset
    trainset = DenoisedDataset(custom_dataset['tdata'],
                               custom_dataset['tlabels'],
                               transform=train_transform)
    validset = DenoisedDataset(custom_dataset['vdata'],
                               custom_dataset['vlabels'],
                               transform=valid_transform)
    return trainset, validset

def data_transforms_denoised_cifar(augment, normalize):
  # compose the transformation
  transform_train = []
  transform_valid = []

  # augmentation
  if augment:
      transform_train += [transforms.RandomCrop(32, padding=4),
                          transforms.RandomHorizontalFlip()]

  # normalization
  if normalize:
      transform_train += [transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                (0.2023, 0.1994, 0.2010))]
      transform_valid += [transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                (0.2023, 0.1994, 0.2010))]
  
  return transforms.Compose(transform_train), transforms.Compose(transform_valid)

