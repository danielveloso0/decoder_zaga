import numpy as np
import torch
from scipy.special import gammainc, gammaincinv, ndtr, ndtri
from sklearn.utils.extmath import randomized_svd


def parameters(model, z):
    with torch.no_grad():
        logits, mean, shape = model.parameters_at(z)
        return (np.clip(logits.sigmoid().numpy(), 1e-6, 1-1e-6),
                mean.numpy(), shape.numpy())


def randomized_pit(x, probability, mean, shape, rng):
    """Randomized CDF at the atom; ordinary CDF for the continuous positive part."""
    zero_mass = 1-probability
    positive = zero_mass + probability*gammainc(shape, x*shape/mean)
    return np.clip(np.where(x > 0, positive, zero_mass*rng.random(x.shape)), 1e-6, 1-1e-6)


def inverse_hurdle(u, probability, mean, shape):
    q = (u-(1-probability))/probability
    active = q > 0
    result = np.zeros(mean.shape, dtype='float32')
    shape = np.broadcast_to(shape, mean.shape)
    result[active] = np.minimum(15, gammaincinv(shape[active],
        np.clip(q[active], 1e-6, 1-1e-6))*mean[active]/shape[active])
    return result


def fit(model, train, rank=32, seed=0, batch=32):
    if not 1 <= rank < min(train.shape):
        raise ValueError('Factor rank must be below both dimensions')
    residual = np.empty(train.shape, dtype='float32')
    rng = np.random.default_rng(seed+30000)
    model.eval()
    for start in range(0, len(train), batch):
        x = train[start:start+batch]
        with torch.no_grad():
            z = model.encode(torch.as_tensor(x))[0]
        p, mean, shape = parameters(model, z)
        residual[start:start+len(x)] = ndtri(randomized_pit(x, p, mean, shape, rng))
    residual -= residual.mean(0)
    residual /= np.maximum(residual.std(0, ddof=1), .1)
    _, singular, vt = randomized_svd(residual, n_components=rank, n_iter=3, random_state=seed)
    loadings = (vt.T*singular/np.sqrt(len(train)-1)).astype('float32')
    return loadings, dict(rank=rank, train_cells=len(train),
        residual_factor_variance_fraction=float(np.sum(singular**2)/np.sum(residual**2)),
        randomized_zero_pit=True, fitted_on='training_last_stage_only')


def factor_parameters(loadings, rank, strength):
    if not 0 <= rank <= loadings.shape[1] or not 0 <= strength <= 1:
        raise ValueError('Invalid rank or shrinkage strength')
    b = loadings[:, :rank]*np.sqrt(strength)
    variance = (b*b).sum(1)
    scale = np.minimum(1, np.sqrt(.95/np.maximum(variance, 1e-8)))
    b = b*scale[:, None]
    noise = np.sqrt(np.maximum(1-(b*b).sum(1), .05))
    return b.astype('float32'), noise.astype('float32')


def sample(model, z, loadings, rank, strength, seed=0, batch=32):
    b, noise = factor_parameters(loadings, rank, strength)
    result = np.empty((len(z), len(b)), dtype='float32')
    rng = np.random.default_rng(seed+40000)
    for start in range(0, len(z), batch):
        zz = z[start:start+batch]
        # Same innovations across ranks/strengths, including the rank-zero control.
        factors = rng.standard_normal((len(zz), loadings.shape[1])).astype('float32')
        independent = rng.standard_normal((len(zz), len(b))).astype('float32')
        normal = factors[:, :rank]@b.T + independent*noise
        p, mean, shape = parameters(model, zz)
        result[start:start+len(zz)] = inverse_hurdle(ndtr(normal), p, mean, shape)
    return result


def covariance_diagnostics(x, target, panel):
    a, b = np.cov(x[:, panel], rowvar=False), np.cov(target[:, panel], rowvar=False)
    off = np.triu_indices(len(panel), k=1)
    av, bv = a[off], b[off]
    correlation = float(np.corrcoef(av, bv)[0,1]) if min(av.std(), bv.std()) > 1e-12 else float('nan')
    return dict(covariance_relative_error=float(np.linalg.norm(av-bv)/max(np.linalg.norm(bv),1e-10)),
                covariance_correlation=correlation)
