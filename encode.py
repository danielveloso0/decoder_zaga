"""Encode compatible preprocessed cells; gene order is mandatory."""
import argparse
from pathlib import Path
import anndata as ad
import numpy as np
import torch
from hurdle_gamma_copula import PretrainedGenerator

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();destination=Path(args.output)
    if destination.exists():raise FileExistsError(destination)
    torch.set_num_threads(2);m=PretrainedGenerator();a=ad.read_h5ad(args.input,backed='r')
    try:mu,lv=m.encode(a.X,genes=list(a.var_names))
    finally:a.file.close()
    destination.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(destination,z=mu,log_variance=lv)
    print(f'Saved {mu.shape} posterior means to {destination}')

if __name__=='__main__':main()
