"""Generate from pretrained weights without any research-repository imports."""
import argparse
import json
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from hurdle_gamma_copula import PretrainedGenerator


def main():
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--preset',choices=['V0','V3','ot_local','ot_ridge'])
    group.add_argument('--latent-file',help='NPY or NPZ containing compatible latent vectors')
    group.add_argument('--bootstrap',action='store_true',help='V0-style generation at any requested cell count')
    p.add_argument('--latent-key',default='z');p.add_argument('--cells',type=int,default=2000)
    p.add_argument('--latent-seed',type=int,default=50000);p.add_argument('--copula-seed',type=int,default=40000)
    p.add_argument('--threads',type=int,default=2);p.add_argument('--output',required=True)
    args=p.parse_args()
    if args.threads<1:p.error('--threads must be positive')
    torch.set_num_threads(args.threads);threadpool_limits(limits=args.threads)
    model=PretrainedGenerator()
    if args.preset:
        if args.cells!=2000:p.error('Frozen presets have 2000 cells. Use --bootstrap or --latent-file for other counts.')
        z=model.frozen_latents(args.preset);label=args.preset
    elif args.bootstrap:z=model.bootstrap_latents(args.cells,args.latent_seed);label='bootstrap_E95'
    else:
        if args.latent_file.endswith('.npy'):z=np.load(args.latent_file,allow_pickle=False)
        else:
            with np.load(args.latent_file,allow_pickle=False) as saved:z=saved[args.latent_key].copy()
        label='custom_latents'
    print(json.dumps(model.write_h5ad(z,args.output,args.copula_seed,label),indent=2))

if __name__=='__main__':main()
