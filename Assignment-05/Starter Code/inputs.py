"""HW5 inputs: one get_taskN_inputs() helper per assessed function.
All arrays are copied; no student function should modify them.
"""
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent/'regression_datasets'
def available_datasets() -> list[str]:
    return sorted(f.stem for f in ROOT.glob('*.npz'))
def load_dataset(name: str, seed: int = 2026) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    with np.load(ROOT/(name+'.npz')) as d:
        X=np.asarray(d['X'],dtype=float).copy(); y=np.asarray(d['y'],dtype=float).copy()
    ids=np.random.default_rng(seed).permutation(len(y)); a=int(.6*len(y)); b=int(.8*len(y))
    return X[ids[:a]],y[ids[:a]],X[ids[a:b]],y[ids[a:b]],X[ids[b:]],y[ids[b:]]
def _sample():
    return load_dataset('quadratic')
def get_task1_inputs():
    return (np.array([-1.,0.,1.,2.]), 2)
def get_task2_inputs():
    return (np.array([[1.,-1.],[1.,0.],[1.,1.]]), np.array([2.,1.,4.]))
def get_task3_inputs():
    return (np.array([1.,2.,4.]),np.array([1.5,1.5,3.]))
def get_task4_inputs():
    return get_task2_inputs()
def get_task5_inputs():
    return get_task2_inputs()
def get_task6_inputs():
    return (*get_task2_inputs(), 0.5)
def get_task7_inputs():
    return (20,0.6,0.2,42)
def get_task8_inputs():
    return (get_task2_inputs()[0].copy(), 1e-8)
def get_task9_inputs():
    X,y,V,v,_,_=_sample(); return (X[:,0] if X.ndim==2 else X,y,V[:,0] if V.ndim==2 else V,v,(1,2,3,5,9))
def get_task10_inputs():
    X,y,V,v,_,_=_sample(); return (X[:,0] if X.ndim==2 else X,y,V[:,0] if V.ndim==2 else V,v,9,(0.,0.001,0.01,0.1,1.,10.))
def get_task11_inputs():
    X,y,V,v,_,_=_sample(); return (X[:,0] if X.ndim==2 else X,y,V[:,0] if V.ndim==2 else V,v,3,(20,40,80,120),42)
