import torch

def kl_divergence(d1, d2, K=100):
    if (type(d1), type(d2)) in torch.distributions.kl._KL_REGISTRY:
        return torch.distributions.kl_divergence(d1, d2)
    else:
        samples = d1.rsample(torch.Size([K]))
        return (d1.log_prob(samples) - d2.log_prob(samples)).mean(0)


def elbo(model, x, K=1):
    qz_x, px_z, _ = model(x, K)
    lpx_z = px_z.log_prob(x).view(*px_z.batch_shape[:2], -1) * model.scaling_factor
    kld = kl_divergence(qz_x, model.pz(*model.pz_params))
    return (lpx_z.sum(-1) - kld.sum(-1)).mean(0).sum()