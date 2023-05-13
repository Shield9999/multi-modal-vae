import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd

def load_model(loading_path):
    model = torch.load(loading_path)
    return model


def get_model(config):
    model_configs = config['MODEL']

    if model_configs['pretrained']:
        try:
            model = load_model(model_configs['pretrain_path'])
        except:
            raise ValueError('Pretrained model not found.')
    else:
        dataset_name = config['DATA']['name']
        if dataset_name == 'MNIST':
            from models.vae import MNISTVAE
            model = MNISTVAE(model_configs)
        else:
            raise NotImplementedError('Model not implemented.')

    return model


def get_optimizer(model, config):
    optim_configs = config['OPTIMIZER']
    optim_type = optim_configs['type']
    optim_params = optim_configs['params']

    if optim_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), **optim_params)
    elif optim_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), **optim_params)
    else:
        raise ValueError('Optimizer type not recognized.')

    return optimizer


def tensor_to_df(tensor, ax_names=None):
    assert tensor.ndim == 2, "Can only currently convert 2D tensors to dataframes"
    df = pd.DataFrame(data=tensor, columns=np.arange(tensor.shape[1]))
    return df.melt(value_vars=df.columns,
                   var_name=('variable' if ax_names is None else ax_names[0]),
                   value_name=('value' if ax_names is None else ax_names[1]))


def tensors_to_df(tensors, head=None, keys=None, ax_names=None):
    dfs = [tensor_to_df(tensor, ax_names=ax_names) for tensor in tensors]
    df = pd.concat(dfs, keys=(np.arange(len(tensors)) if keys is None else keys))
    df.reset_index(level=0, inplace=True)
    if head is not None:
        df.rename(columns={'level_0': head}, inplace=True)
    return df

