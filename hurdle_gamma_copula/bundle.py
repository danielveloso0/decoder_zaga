"""Load portable weights and reproduce the original CPU sampling protocol."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scipy.special import ndtr
from .model import HurdleVAE, normalize_library
from .copula import factor_parameters, parameters, inverse_hurdle


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def matrix_hash(x):
    x=sparse.csr_matrix(x)
    h=hashlib.sha256()
    for a in (x.data,x.indices,x.indptr):h.update(a.tobytes())
    return h.hexdigest()


class PretrainedGenerator:
    """CPU generator with fixed gene order and log1p(CP10k) convention.

    ``decode`` accepts arbitrary compatible latent vectors. Frozen presets are
    historical E10.5 forecasts, not newly fitted dynamics or a time-conditioned VAE.
    """
    def __init__(self,assets=None):
        self.assets=Path(assets) if assets else Path(__file__).parent/'assets'
        self.config=json.loads((self.assets/'config.json').read_text(encoding='utf-8'))
        for name,digest in self.config['asset_sha256'].items():
            if sha256(self.assets/name)!=digest:raise ValueError(f'Asset checksum mismatch: {name}')
        self.genes=(self.assets/'genes.txt').read_text(encoding='utf-8').splitlines()
        if len(self.genes)!=self.config['n_genes'] or len(set(self.genes))!=len(self.genes):
            raise ValueError('Invalid gene panel')
        self.model=HurdleVAE(len(self.genes),self.config['latent'],self.config['hidden'])
        self.model.encoder=torch.nn.Sequential(torch.nn.Linear(len(self.genes),64),torch.nn.LayerNorm(64),torch.nn.SiLU())
        state=torch.load(self.assets/'weights.pt',map_location='cpu',weights_only=True)
        self.model.load_state_dict(state,strict=True)
        self.model.eval()
        for p in self.model.parameters():p.requires_grad_(False)
        with np.load(self.assets/'decoder_state.npz',allow_pickle=False) as z:
            self.loadings=z['loadings'].copy();self.reference_z=z['latest_z'].copy()
            self.library=float(z['library'])
        self.b,self.noise=factor_parameters(self.loadings,self.config['copula_rank'],self.config['copula_strength'])

    def frozen_latents(self,preset='V0'):
        """Return one of four stored 2000-cell E10.5 reference forecasts."""
        if preset not in self.config['presets']:raise ValueError(f'Unknown preset: {preset}')
        with np.load(self.assets/'forecasts.npz',allow_pickle=False) as saved:return saved[preset].copy()

    def bootstrap_latents(self,n_cells,seed=50000):
        """Bootstrap the historical E9.5 embedding bank; no temporal drift (V0)."""
        if n_cells<1:raise ValueError('n_cells must be positive')
        ids=np.random.default_rng(seed).choice(len(self.reference_z),n_cells,replace=True)
        return self.reference_z[ids].copy()

    def encode(self,x,genes=None,batch=32):
        """Encode preprocessed log1p(CP10k), returning posterior mean/log-variance.

        Supply genes to enforce order. Raw count normalization is not automatic.
        """
        if genes is not None and list(genes)!=self.genes:raise ValueError('Genes must match assets/genes.txt exactly, including order')
        if len(x.shape)!=2 or x.shape[1]!=len(self.genes):raise ValueError('Expected cells x 32285 expression matrix')
        if batch<1:raise ValueError('batch must be positive')
        mus=[];lvs=[]
        for start in range(0,x.shape[0],batch):
            chunk=x[start:start+batch]
            chunk=chunk.toarray() if sparse.issparse(chunk) else np.asarray(chunk)
            if not np.isfinite(chunk).all() or (chunk<0).any():raise ValueError('Expression must be finite and nonnegative')
            with torch.no_grad():mu,lv=self.model.encode(torch.as_tensor(chunk,dtype=torch.float32))
            mus.append(mu.numpy());lvs.append(lv.numpy())
        if not mus:raise ValueError('Empty expression matrix')
        return np.concatenate(mus),np.concatenate(lvs)

    def decode(self,z,copula_seed=40000,normalize=True):
        """Sample with the historical 128-cell RNG blocks and 8-cell inversions.

        Keep block sizes fixed for numerical/RNG reproducibility. Output is CSR.
        """
        z=np.asarray(z,dtype='float32')
        if z.ndim!=2 or z.shape[1]!=self.config['latent'] or not len(z):raise ValueError('Expected nonempty (cells, 32) latent matrix')
        if not np.isfinite(z).all():raise ValueError('Nonfinite latent vector')
        rng=np.random.default_rng(copula_seed);chunks=[]
        for start in range(0,len(z),128):
            zz=z[start:start+128]
            f=rng.standard_normal((len(zz),self.config['copula_rank'])).astype('float32')
            e=rng.standard_normal((len(zz),len(self.genes))).astype('float32')
            x=np.empty_like(e)
            for local in range(0,len(zz),8):
                stop=local+8
                p,mean,shape=parameters(self.model,torch.from_numpy(zz[local:stop]))
                u=ndtr(f[local:stop]@self.b.T+e[local:stop]*self.noise)
                x[local:stop]=inverse_hurdle(u,p,mean,shape)
            if normalize:x=normalize_library(x,self.library)
            chunks.append(sparse.csr_matrix(x))
        return sparse.vstack(chunks,format='csr')

    def write_h5ad(self,z,path,copula_seed=40000,label='custom',submission=True):
        """Write generated expression; refuse to overwrite an existing prediction."""
        import anndata as ad
        path=Path(path)
        if path.exists():raise FileExistsError(path)
        if submission and not 1000<=len(z)<=5118:raise ValueError('Historical Task1 contract: 1000 to 5118 cells')
        x=self.decode(z,copula_seed=copula_seed)
        if not np.isfinite(x.data).all() or (x.data<0).any():raise ValueError('Invalid generated expression')
        a=ad.AnnData(x,obs=pd.DataFrame(index=[f'generated_{i:05d}' for i in range(len(z))]),var=pd.DataFrame(index=self.genes))
        a.uns['generator']=dict(model='HurdleGammaVAE',latent=self.config['latent'],copula_rank=self.config['copula_rank'],model_seed=0,copula_seed=int(copula_seed),library_size=self.library,preset=label)
        path.parent.mkdir(parents=True,exist_ok=True)
        a.write_h5ad(path,compression='gzip')
        ready=dict(shape=list(x.shape),dtype=str(x.dtype),format='CSR',file_sha256=sha256(path),matrix_content_sha256=matrix_hash(x),copula_seed=copula_seed,label=label,zero_fraction=float(1-x.nnz/(x.shape[0]*x.shape[1])))
        path.with_suffix('.json').write_text(json.dumps(ready,indent=2),encoding='utf-8')
        return ready
