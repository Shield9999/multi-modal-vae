import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.lines import Line2D

def exp_str(config):
    components = []
    components.append(config['project'])
    components.append(config['DATA']['name'])
    components.append(config['MODEL']['name'])
    components.append(config['postfix'])
    return '_'.join(components)


def custom_cmap(n):
    """Create customised colormap for scattered latent plot of n categories.
    Returns colormap object and colormap array that contains the RGB value of the colors.
    See official matplotlib document for colormap reference:
    https://matplotlib.org/examples/color/colormaps_reference.html
    """
    # first color is grey from Set1, rest other sensible categorical colourmap
    cmap_array = sns.color_palette("Set1", 9)[-1:] + sns.husl_palette(n - 1, h=.6, s=0.7)
    cmap = colors.LinearSegmentedColormap.from_list('mmdgm_cmap', cmap_array)
    return cmap, cmap_array


def plot_embeddings(emb, emb_l, labels, file_path):
    cmap_obj, cmap_arr = custom_cmap(n=len(labels))
    plt.figure()
    plt.scatter(emb[:, 0], emb[:, 1], c=emb_l, cmap=cmap_obj, s=25, alpha=0.2, edgecolors='none')
    l_elems = [Line2D([0], [0], marker='o', color=cm, label=l, alpha=0.5, linestyle='None')
               for (cm, l) in zip(cmap_arr, labels)]
    plt.legend(frameon=False, loc=2, handles=l_elems)
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()


def plot_kls_df(df, file_path):
    _, cmap_arr = custom_cmap(df[df.columns[0]].nunique() + 1)
    with sns.plotting_context("notebook", font_scale=2.0):
        g = sns.FacetGrid(df, height=12, aspect=2)
        g = g.map(sns.boxplot, df.columns[1], df.columns[2], df.columns[0], palette=cmap_arr[1:],
                  order=None, hue_order=None)
        g = g.set(yscale='log').despine(offset=10)
        plt.legend(loc='best', fontsize='22')
        plt.savefig(file_path, bbox_inches='tight')
        plt.close()


def log_recon_analysis(model, data_loader, file_path, epoch, device='cpu'):
    model.eval()
    with torch.no_grad():
        for (data, label) in data_loader:
            data = data.to(device)
            model.reconstruct(data, file_path, epoch)
            model.analyse(data, file_path, epoch)
            break


def get_writer(config):
    logging_config = config['LOGGING']
    log_str = exp_str(config)

    log_path = logging_config['log_path']
    log_dir = os.path.join(log_path, log_str)
    writer = SummaryWriter(log_dir=log_dir)

    return writer


def log_scalars(writer, train_loss, test_loss, epoch):
    writer.add_scalar('train_loss', train_loss, epoch)
    writer.add_scalar('test_loss', test_loss, epoch)


def save_model(model, config):
    log_str = exp_str(config)
    save_path = os.path.join(config['LOGGING']['save_path'], log_str + '.pt')

    torch.save(model.state_dict(), save_path)