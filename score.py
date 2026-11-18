"""Run the preserved score_local against explicit external target/reference paths."""
import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prediction',required=True);p.add_argument('--e105',required=True);p.add_argument('--e95',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();destination=Path(args.output)
    if destination.exists():raise FileExistsError(destination)
    vendor=Path(__file__).resolve().parent/'evaluation_vendor'
    sys.path.insert(0,str(vendor))
    from score_local import score_local
    from common import core_metrics
    if Path(core_metrics.__file__).resolve()!=vendor/'common/core_metrics.py':raise RuntimeError('Scorer imported an unexpected metric core')
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):result=score_local(args.e105,args.prediction,local_e95=args.e95)
    def digest(path):
        h=hashlib.sha256()
        with Path(path).open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
        return h.hexdigest()
    report=dict(score=result,prediction_sha256=digest(args.prediction),scorer_sha256=digest(vendor/'score_local.py'),core_sha256=digest(core_metrics.__file__),versions={n:importlib.metadata.version(n) for n in ['torch','numpy','scipy','anndata','pandas','h5py','scikit-learn']},training=False,uploaded=False)
    destination.parent.mkdir(parents=True,exist_ok=True);destination.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':main()
