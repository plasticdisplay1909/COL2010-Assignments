"""Reference smoke tests; replace solutions.py with student file to run locally."""
import numpy as np
from solutions import *
from inputs import load_dataset
x,y,xv,yv,xt,yt=load_dataset('quadratic')
A=design_matrix(x,2)
assert A.shape==(len(x),3)
w=ols(A,y)
assert mse(y,predict(A,w))<.12
yp,e,err=projection_diagnostics(A,y)
assert err<1e-8
assert np.isfinite(ridge(A,y,1)).all()
a,b,c=split_indices(len(x),.6,.2,42)
assert len(set(a)&set(b))==0 and len(a)+len(b)+len(c)==len(x)
S=standardize(A)
assert S.shape==A.shape and np.allclose(S[:,0],A[:,0])
assert np.allclose(np.mean(S[:,1:],axis=0),0,atol=1e-7) and np.allclose(np.std(S[:,1:],axis=0),1,atol=1e-4)
d,ds=select_degree(x,y,xv,yv,[1,2,3,8]); assert d==2
lam,ls=select_ridge(x,y,xv,yv,5,[0,.001,.01,.1,1]);assert len(ls)==5
tr,va=learning_curve(x,y,xv,yv,3,[30,60,120],7);assert len(tr)==3 and np.isfinite(va).all()
print('All smoke tests passed')
