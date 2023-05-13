import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


def get_dataset(configs):
    data_config = configs['DATA']
    dataset_name = data_config['name']

    if dataset_name == 'MNIST':
        train_dataset = datasets.MNIST('../data', train=True, download=True,
                                        transform=transforms.ToTensor())
        test_dataset = datasets.MNIST('../data', train=False, download=True,
                                        transform=transforms.ToTensor())
    else:
        raise NotImplementedError('Dataset not implemented.')
    
    return train_dataset, test_dataset


def get_dataloader(configs):
    data_config = configs['DATA']

    batch_size = data_config['batch_size']
    num_workers = data_config['num_workers']
    pin_memory = data_config['pin_memory']

    train_dataset, test_dataset = get_dataset(configs)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=pin_memory)
    
    return train_loader, test_loader