import os

import torch

from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


def get_dataset(configs):
    data_config = configs['DATA']
    dataset_name = data_config['name']
    data_path = data_config['data_path']

    if dataset_name == 'MNIST':
        train_dataset, test_dataset = get_mnist(data_path)
    elif dataset_name == 'SVHN':
        train_dataset, test_dataset = get_svhn(data_path)
    elif dataset_name == 'MNIST-SVHN':
        train_dataset, test_dataset = get_mnist_svhn(configs)
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


def rand_match_on_idx(l1, idx1, l2, idx2, max_d=10000, dm=10):
    """
    l*: sorted labels
    idx*: indices of sorted labels in original list
    """
    _idx1, _idx2 = [], []
    for l in l1.unique():  # assuming both have same idxs
        l_idx1, l_idx2 = idx1[l1 == l], idx2[l2 == l]
        n = min(l_idx1.size(0), l_idx2.size(0), max_d)
        l_idx1, l_idx2 = l_idx1[:n], l_idx2[:n]
        for _ in range(dm):
            _idx1.append(l_idx1[torch.randperm(n)])
            _idx2.append(l_idx2[torch.randperm(n)])
    return torch.cat(_idx1), torch.cat(_idx2)


def mnist_svhn(configs):
    max_d = 10000  # maximum number of datapoints per class
    dm = 30        # data multiplier: random permutations to match
    data_path = configs['DATA']['data_path']

    # get the individual datasets
    tx = transforms.ToTensor()
    train_mnist = datasets.MNIST(data_path, train=True, download=True, transform=tx)
    test_mnist = datasets.MNIST(data_path, train=False, download=True, transform=tx)
    train_svhn = datasets.SVHN(data_path, split='train', download=True, transform=tx)
    test_svhn = datasets.SVHN(data_path, split='test', download=True, transform=tx)

    train_svhn.labels = torch.LongTensor(train_svhn.labels.squeeze()) % 10
    test_svhn.labels = torch.LongTensor(test_svhn.labels.squeeze()) % 10

    # svhn labels need extra work
    train_svhn.labels = torch.LongTensor(train_svhn.labels.squeeze()) % 10
    test_svhn.labels = torch.LongTensor(test_svhn.labels.squeeze()) % 10

    mnist_l, mnist_li = train_mnist.targets.sort()
    svhn_l, svhn_li = train_svhn.labels.sort()
    idx1, idx2 = rand_match_on_idx(mnist_l, mnist_li, svhn_l, svhn_li, max_d=max_d, dm=dm)
    print('len train idx:', len(idx1), len(idx2))
    
    train_mnist_path = os.path.join(data_path, 'train-ms-mnist-idx.pt')
    train_svhn_path = os.path.join(data_path, 'train-ms-svhn-idx.pt')
    torch.save(idx1, train_mnist_path)
    torch.save(idx2, train_svhn_path)

    mnist_l, mnist_li = test_mnist.targets.sort()
    svhn_l, svhn_li = test_svhn.labels.sort()
    idx1, idx2 = rand_match_on_idx(mnist_l, mnist_li, svhn_l, svhn_li, max_d=max_d, dm=dm)
    print('len test idx:', len(idx1), len(idx2))
    
    test_mnist_path = os.path.join(data_path, 'test-ms-mnist-idx.pt')
    test_svhn_path = os.path.join(data_path, 'test-ms-svhn-idx.pt')
    torch.save(idx1, test_mnist_path)
    torch.save(idx2, test_svhn_path)


def get_mnist(data_path):
    train_dataset = datasets.MNIST(data_path, train=True, download=True,
                                        transform=transforms.ToTensor())
    test_dataset = datasets.MNIST(data_path, train=False, download=True,
                                        transform=transforms.ToTensor())
    return train_dataset, test_dataset


def get_svhn(data_path):
    train_dataset = datasets.SVHN(data_path, split='train', download=True,
                                        transform=transforms.ToTensor())
    test_dataset = datasets.SVHN(data_path, split='test', download=True,
                                        transform=transforms.ToTensor())
    return train_dataset, test_dataset

class MNISTSVHN(Dataset):
    def __init__(self, mnist, svhn):
        super().__init__()

        self.mnist = mnist
        self.svhn = svhn

    def __len__(self):
        return len(self.mnist) 
    
    def __getitem__(self, idx):
        return self.mnist[idx], self.svhn[idx]


def get_mnist_svhn(configs):
    data_path = configs['DATA']['data_path']

    train_mnist_path = os.path.join(data_path, 'train-ms-mnist-idx.pt')
    train_svhn_path = os.path.join(data_path, 'train-ms-svhn-idx.pt')
    test_mnist_path = os.path.join(data_path, 'test-ms-mnist-idx.pt')
    test_svhn_path = os.path.join(data_path, 'test-ms-svhn-idx.pt')
    if not (os.path.exists(train_mnist_path) and
            os.path.exists(train_svhn_path) and
            os.path.exists(test_mnist_path) and
            os.path.exists(test_svhn_path)):
        mnist_svhn(configs)
    
    train_mnist = torch.load(train_mnist_path)
    train_svhn = torch.load(train_svhn_path)
    test_mnist = torch.load(test_mnist_path)
    test_svhn = torch.load(test_svhn_path)

    train_mnist_dataset, test_mnist_dataset = get_mnist(data_path)
    train_svhn_dataset, test_svhn_dataset = get_svhn(data_path)

    train_mnist_dataset = Subset(train_mnist_dataset, train_mnist)
    train_svhn_dataset = Subset(train_svhn_dataset, train_svhn)
    test_mnist_dataset = Subset(test_mnist_dataset, test_mnist)
    test_svhn_dataset = Subset(test_svhn_dataset, test_svhn)

    train_mnist_svhn = MNISTSVHN(train_mnist_dataset, train_svhn_dataset)
    test_mnist_svhn = MNISTSVHN(test_mnist_dataset, test_svhn_dataset)

    return train_mnist_svhn, test_mnist_svhn