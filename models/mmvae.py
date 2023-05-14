import numpy as np
from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

import pandas as pd
from umap import UMAP

import utils.logging as logging
from utils.criteria import kl_divergence
from models.vae import MNISTVAE, SVHNVAE

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

class MMVAE(nn.Module):
    def __init__(self,
                 prior_dist,
                 params,
                 *vaes) -> None:
        
        super(MMVAE, self).__init__()

        self.pz = prior_dist
        self.vaes = nn.ModuleList([vae(params) for vae in vaes])

        self.modulename = None
        self.params = params

        self._pz_params = None

    @property
    def pz_params(self):
        return self._pz_params
    
    def forward(self, x, K=1):
        qz_xs, zss = [], []

        px_zs = [[None for _ in range(len(self.vaes))] for _ in range(len(self.vaes))]

        for m, vae in enumerate(self.vaes):
            qz_x, px_z, zs = vae(x[m], K=K)
            qz_xs.append(qz_x)
            zss.append(zs)
            px_zs[m][m] = px_z

        for e, zs in enumerate(zss):
            for d, vae in enumerate(self.vaes):
                if e != d:
                    px_zs[e][d] = vae.px_z(*vae.decoder(zs))

        return qz_xs, px_zs, zss
    
    def generate(self, N):
        self.eval()
        with torch.no_grad():
            x = []
            pz = self.pz(*self.pz_params)
            latents = pz.rsample(torch.Size([N]))
            for d, vae in enumerate(self.vaes):
                px_z = vae.px_z(*vae.decoder(latents))
                x.append(px_z.mean.view(-1, *px_z.mean.size()[2:]))

        return x
    
    def reconstruct(self, x):
        self.eval()
        with torch.no_grad():
            _, px_zs, _ = self.forward(x)
            recons = [[px_z.mean for px_z in r] for r in px_zs]

        return recons
    
    def analyse(self, x, K):
        self.eval()
        with torch.no_grad():
            qz_xs, _, zss = self.forward(x, K)
            pz = self.pz(*self.pz_params)
            zss = [pz.sample(torch.Size([K, x[0].size(0)])).view(-1, pz.batch_shape[-1]), 
                   *[zs.view(-1, zs.size(-1)) for zs in zss]]
            zsl = [torch.zeros(zs.size(0)).fill_(i) for i, zs in enumerate(zss)]
            kls_df = tensors_to_df(
                [*[kl_divergence(qz_x, pz).cpu().numpy() for qz_x in qz_xs],
                 *[0.5 * (kl_divergence(p, q) + kl_divergence(q, p)).cpu().numpy()
                   for p, q in combinations(qz_xs, 2)]],
                head='KL',
                keys=[*[r'KL$(q(z|x_{})\,||\,p(z))$'.format(i) for i in range(len(qz_xs))],
                      *[r'J$(q(z|x_{})\,||\,q(z|x_{}))$'.format(i, j)
                        for i, j in combinations(range(len(qz_xs)), 2)]],
                ax_names=['Dimensions', r'KL$(q\,||\,p)$']
            )
        
        ret = UMAP(metric='euclidean',
                   n_neighbors=5,
                   transform_seed=torch.initial_seed())
        ret = ret.fit_transform(torch.cat(zss, dim=0).cpu().numpy())

        return ret, torch.cat(zsl, 0).cpu().numpy(), kls_df
            

class MNISTSVHNMMVAE(MMVAE):
    def __init__(self, params):
        super(MNISTSVHNMMVAE, self).__init__(torch.distributions.Laplace, params, MNISTVAE, SVHNVAE)
        
        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params['latent_dim']), requires_grad=False), # mu
            nn.Parameter(torch.ones(1, params['latent_dim']), requires_grad=params['learn_prior']) # logvar
        ])

        self.modelname = 'mnist-svhn'

    @property
    def pz_params(self):
        return self._pz_params[0], F.softmax(self._pz_params[1], dim=-1) * self._pz_params[1].size(-1)
    
    def generate(self, run_path, epoch):
        N = 64
        samples_list = super(MNISTSVHNMMVAE, self).generate(N)
        for i, samples in enumerate(samples_list):
            samples = samples.data.cpu()
            samples = samples.view(N, *samples.size()[1:])
            save_image(samples, run_path + '/generated_{}_{}.png'.format(self.modulename, i), nrow=8)

    def reconstruct(self, x, run_path, epoch):
        recons_mat = super(MNISTSVHNMMVAE, self).reconstruct([d[:8] for d in x])
        for r, recons in enumerate(recons_mat):
            for o, recon in enumerate(recons):
                _data = x[r][:8].cpu()
                recon = recon.squeeze(0).cpu()

                _data = _data if r == 1 else resize_img(_data, self.vaes[1].data_size)
                recon = recon if o == 1 else resize_img(recon, self.vaes[1].data_size)
                comp = torch.cat([_data, recon])
                save_image(comp, '{}/recon_{}x{}_{:03d}.png'.format(run_path, r, o, epoch))

    def analyse(self, x, run_path, epoch):
        z_emb, zsl, kls_df = super(MNISTSVHNMMVAE, self).analyse(x, K=10)
        labels = ['Prior', self.modelname.lower()]
        logging.plot_embeddings(z_emb, zsl, labels, '{}/emb_umap_{:03d}.png'.format(run_path, epoch))
        logging.plot_kls_df(kls_df, '{}/kldistance_{:03d}.png'.format(run_path, epoch))

def resize_img(img, refsize):
    return F.pad(img, (2, 2, 2, 2)).expand(img.size(0), *refsize)