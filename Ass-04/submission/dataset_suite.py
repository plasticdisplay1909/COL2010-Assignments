"""COL2010 classification comparisons. Run after setup.py, or import generate_suite.
No network is required. sklearn is used here only for bundled data/generation/splits.
"""
from pathlib import Path
import json
import numpy as np
from sklearn.datasets import make_moons, make_circles, load_breast_cancer, load_wine, load_digits
from sklearn.model_selection import train_test_split

def generate_suite(output_dir='classification_datasets', seed=42):
    out=Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    rng=np.random.default_rng(seed); cases=[]; n=1000
    def add(name,X,y,purpose,kind='continuous',source='Synthetic; generated locally'):
        cases.append((name,np.asarray(X,dtype=float),np.asarray(y,dtype=int),purpose,kind,source))
    y=np.repeat([0,1],n//2); signal=rng.normal(size=(n,2))+(2*y-1)[:,None]*1.1
    add('gaussian_independent',signal,y,'Gaussian NB assumptions hold; linear boundary and limited-training-size comparison.')
    # Keep the original informative features unchanged, add correlated copies.
    dup=np.c_[signal,np.repeat(signal[:,0,None],8,axis=1)+rng.normal(0,.08,(n,8))]
    add('gaussian_correlated',dup,y,'Paired with gaussian_independent: duplicated evidence violates NB independence; compare F1 and log loss.')
    yy=np.repeat([0,1],n//2); a=rng.normal(0,.5,(n//2,2));b=rng.normal(0,1.8,(n//2,2))
    add('gaussian_unequal_variance',np.vstack([a,b]),yy,'Same class means, different variances: Gaussian NB has a curved boundary; linear log-odds models underfit.')
    X,y2=make_moons(n_samples=n,noise=.22,random_state=seed)
    add('moons',X,y2,'Nonlinear boundary: ordinary K-NN and RBF SVM versus linear logistic/SVM.')
    X,y2=make_circles(n_samples=n,noise=.10,factor=.4,random_state=seed)
    add('circles',X,y2,'Radial geometry: linear boundaries fail; compare RBF gamma and local neighborhoods.')
    add('scale_distortion',signal*np.array([1,1000]),y,'Same rows/labels as gaussian_independent: raw versus training-standardized distances.')
    noise=rng.normal(size=(n,80))
    add('irrelevant_dimensions',np.c_[signal,noise],y,'Same signal and row split as gaussian_independent: effect of 80 irrelevant coordinates on K-NN and regularized models.')
    X=rng.uniform(-2,2,(n,2)); y2=((X[:,0]>.3)&(X[:,1]>-.5)).astype(int)
    flip=rng.random(n)<.05;y2[flip]=1-y2[flip]
    add('axis_aligned_rules',X,y2,'Two threshold rules with 5% label noise: tree interpretability, depth, pruning and forest variance.')
    y2=rng.binomial(1,.05,n);X=rng.normal(size=(n,4))+(2*y2-1)[:,None]*.65
    add('imbalanced',X,y2,'Approximately 5% positives: majority baseline, accuracy versus precision/recall/F1 and validation threshold tuning.')
    y2=rng.integers(0,2,n); p0=np.r_[np.full(8,.10),np.full(8,.45),np.full(8,.25)];p1=np.r_[np.full(8,.55),np.full(8,.15),np.full(8,.25)]
    X=rng.binomial(1,np.where(y2[:,None]==1,p1,p0))
    add('bernoulli_binary',X,y2,'Binary occurrence features: Bernoulli NB including absent-feature evidence and Laplace smoothing.','binary')
    for name,load,purpose in [('breast_cancer',load_breast_cancer,'Real binary measurements; correlated features, scaling and minority errors.'),('wine',load_wine,'Real three-class data; multiclass posterior/voting and softmax evaluation.'),('digits',load_digits,'Real ten-class image vectors; geometry, prediction time and multiclass evaluation.')]:
        d=load(); X=d.data;y2=d.target
        if name=='digits':
            rows,_=train_test_split(np.arange(len(y2)),train_size=1000,stratify=y2,random_state=seed);X=X[rows];y2=y2[rows]
        add(name,X,y2,purpose,source='scikit-learn bundled '+name+' dataset; no download required')
    manifest=[]
    for name,X,y2,purpose,kind,source in cases:
        rows=np.arange(len(y2));train,rest=train_test_split(rows,test_size=.4,stratify=y2,random_state=seed)
        val,test=train_test_split(rest,test_size=.5,stratify=y2[rest],random_state=seed)
        mu=X[train].mean(0);sd=X[train].std(0);sd=np.where(sd==0,1,sd)
        arrays={'classes':np.unique(y2),'mean':mu,'scale':sd}
        for split,ix in [('train',train),('validation',val),('test',test)]:
            arrays['X_'+split]=X[ix];arrays['y_'+split]=y2[ix];arrays['rows_'+split]=ix
            arrays['X_'+split+'_std']=(X[ix]-mu)/sd
        np.savez_compressed(out/(name+'.npz'),**arrays)
        manifest.append(dict(name=name,n=len(y2),features=X.shape[1],classes=np.unique(y2).tolist(),kind=kind,source=source,purpose=purpose,sizes={s:len(arrays['y_'+s]) for s in ['train','validation','test']}))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Generated {len(cases)} datasets in {out}')
    return manifest

def load_dataset(name, output_dir='classification_datasets', standardized=False):
    """Return six arrays: X_train,y_train,X_validation,y_validation,X_test,y_test.
    Bernoulli NB must use raw binary X, even though standardized arrays are stored.
    """
    with np.load(Path(output_dir)/(name+'.npz'),allow_pickle=False) as d:
        return tuple(d['X_'+s+('_std' if standardized else '')] if item=='X' else d['y_'+s]
                     for s in ['train','validation','test'] for item in ['X','y'])
if __name__=='__main__':generate_suite()
