import tqdm

import torch
import torch.nn as nn

def train_epoch(train_loader, model, optimizer, objective, device='cpu'):
    model.train()
    train_loss = 0
    for (data, label) in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = -objective(model, data)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    train_loss /= len(train_loader.dataset)
    return train_loss


def test_epoch(test_loader, model, objective, device='cpu'):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for (data, label) in test_loader:
            data = data.to(device)
            loss = -objective(model, data)
            test_loss += loss.item()

    test_loss /= len(test_loader.dataset)
    return test_loss