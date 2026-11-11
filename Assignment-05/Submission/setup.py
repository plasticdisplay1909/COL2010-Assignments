"""Generate reproducible synthetic and real-world regression datasets.
Real-world diabetes data are bundled with scikit-learn (no network needed).
Optional California housing is downloaded with sklearn only if --download is supplied.
"""
import argparse
from pathlib import Path
import numpy as np

def main(download=False):
    out=Path(__file__).resolve().parent/'regression_datasets';out.mkdir(exist_ok=True)
    rng=np.random.default_rng(2026)
    x=rng.uniform(-1,1,500); y=1+2*x-3*x*x+0.25*rng.normal(size=len(x))
    np.savez(out/'quadratic.npz',X=x,y=y)
    x=rng.uniform(-1,1,500); y=np.sin(3*np.pi*x)+0.25*rng.normal(size=len(x))
    np.savez(out/'oscillatory.npz',X=x,y=y)
    x=rng.uniform(-1,1,500); y=1+2*x+0.6*rng.normal(size=len(x)); y[:20]+=rng.normal(0,6,20)
    np.savez(out/'outliers.npz',X=x,y=y)
    x=rng.uniform(-1,1,120); y=0.5+1.2*x+0.2*rng.normal(size=len(x))
    np.savez(out/'small_sample.npz',X=x,y=y)
    try:
        from sklearn.datasets import load_diabetes
        data=load_diabetes(); np.savez(out/'diabetes.npz',X=data.data[:,2],y=data.target)
    except ImportError:
        print('Install scikit-learn for bundled diabetes data')
    if download:
        from sklearn.datasets import fetch_california_housing
        data=fetch_california_housing(data_home=str(out/'cache'))
        np.savez(out/'california_housing.npz',X=data.data[:,0],y=data.target)
    print('Datasets ready:',*[q.name for q in sorted(out.glob('*.npz'))],sep='\n  ')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--download',action='store_true',help='Optional internet download of California housing'); main(parser.parse_args().download)
