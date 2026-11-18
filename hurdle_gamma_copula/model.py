import torch
from torch import nn
from torch.nn import functional as F


class HurdleVAE(nn.Module):
    """Point mass at zero + Gamma density for positive log-expression.

    This is a continuous likelihood on the supplied scale, not a count model.
    Genes are conditionally independent given z; no cellular labels are used.
    """

    def __init__(self, n_genes, latent=16, hidden=64):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(n_genes, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, latent)
        self.lv = nn.Linear(hidden, latent)
        self.hidden = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU())
        self.gate = nn.Linear(hidden, n_genes)
        self.positive = nn.Linear(hidden, n_genes)
        self.log_shape = nn.Parameter(torch.zeros(n_genes))

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.lv(h).clamp(-10, 5)

    def parameters_at(self, z):
        h = self.hidden(z)
        logits = self.gate(h)
        mean = F.softplus(self.positive(h)) + 1e-5
        shape = self.log_shape.clamp(-2, 3).exp()
        return logits, mean, shape

    def decoder(self, z):
        logits, mean, _ = self.parameters_at(z)
        return logits.sigmoid() * mean

    def objective(self, x, beta, stochastic=True):
        mu, lv = self.encode(x)
        z = mu + torch.randn_like(mu) * (.5 * lv).exp() if stochastic else mu
        logits, mean, shape = self.parameters_at(z)
        nonzero = (x > 0).to(x.dtype)
        rate = shape / mean
        positive_nll = (torch.lgamma(shape) - shape * rate.log()
                        - (shape - 1) * x.clamp_min(1e-8).log() + rate * x)
        reconstruction = (F.binary_cross_entropy_with_logits(logits, nonzero, reduction='none')
                          + nonzero * positive_nll).mean()
        kl = -.5 * (1 + lv - mu.square() - lv.exp()).sum(1).mean()
        return reconstruction + beta * kl, reconstruction, kl


def normalize_library(x, library_size):
    """Rescale in linear normalized space; preserves zeros, never logs twice."""
    import numpy as np
    y = np.expm1(np.clip(x, 0, 15)).astype('float32')
    sums = y.sum(1, keepdims=True)
    y *= library_size / np.maximum(sums, 1e-8)
    return np.log1p(y)


def decode(model, z, rng, stochastic=True, batch=32):
    import numpy as np
    output = np.empty((len(z), model.gate.out_features), dtype='float32')
    for start in range(0, len(z), batch):
        with torch.no_grad():
            logits, mean, shape = model.parameters_at(z[start:start+batch])
            p = logits.sigmoid().cpu().numpy()
            mean = mean.cpu().numpy()
            shape = shape.cpu().numpy()
        if not stochastic:
            output[start:start+len(p)] = p * mean
        else:
            gate = rng.random(p.shape) < p
            values = rng.gamma(shape, mean / shape)
            # A fixed numerical bound, not a target-derived statistic.
            output[start:start+len(p)] = np.where(gate, np.minimum(values, 15), 0)
    return output
