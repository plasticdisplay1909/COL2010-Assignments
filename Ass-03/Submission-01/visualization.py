"""Supplied plotting and visualization helpers for COL2010 Homework 3."""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

def plot_clusters(X_2d, labels, title="Clustering", ax=None, noise_label=-1):
    X_2d=np.asarray(X_2d); labels=np.asarray(labels)
    if X_2d.ndim!=2 or X_2d.shape[1]!=2: raise ValueError("X_2d must have shape (N, 2)")
    if ax is None: _,ax=plt.subplots(figsize=(5,4))
    for lab in np.unique(labels):
        mask=labels==lab; marker="x" if lab==noise_label else "o"; name="noise" if lab==noise_label else f"cluster {lab}"
        ax.scatter(X_2d[mask,0],X_2d[mask,1],s=24,marker=marker,label=name)
    ax.set_title(title); ax.set_xlabel("coordinate 1"); ax.set_ylabel("coordinate 2"); ax.legend(fontsize=8); return ax

def compare_clusterings(X_2d,label_sets,title=None):
    names=list(label_sets.keys()); fig,axes=plt.subplots(1,len(names),figsize=(4.2*len(names),3.8),squeeze=False)
    for ax,name in zip(axes[0],names): plot_clusters(X_2d,label_sets[name],name,ax=ax)
    if title: fig.suptitle(title)
    fig.tight_layout(); return fig,axes

def show_digit_examples(X,true_labels=None,cluster_labels=None,n=12,image_shape=(8,8)):
    X=np.asarray(X); n=min(int(n),len(X)); cols=min(6,n); rows=int(np.ceil(n/cols)); fig,axes=plt.subplots(rows,cols,figsize=(2*cols,2*rows),squeeze=False)
    for i,ax in enumerate(axes.ravel()):
        ax.axis("off")
        if i>=n: continue
        ax.imshow(X[i].reshape(image_shape),cmap="gray_r"); bits=[]
        if true_labels is not None: bits.append(f"true={true_labels[i]}")
        if cluster_labels is not None: bits.append(f"cluster={cluster_labels[i]}")
        ax.set_title(" | ".join(bits),fontsize=8)
    fig.tight_layout(); return fig,axes

def _pairwise_sq_dist(X):
    sq=np.sum(X*X,axis=1); return np.maximum(sq[:,None]+sq[None,:]-2*X@X.T,0.0)

def compute_tsne_projection(X,perplexity=30.0,n_iters=350,seed=42):
    """Exact small-data t-SNE helper for visualization only."""
    X=np.asarray(X,float)
    if X.ndim!=2 or len(X)<3: raise ValueError("X must be a 2D array with at least three rows")
    n=len(X); perplexity=float(min(perplexity,max(2.0,(n-1)/3))); Xc=X-X.mean(axis=0)
    if Xc.shape[1]>30:
        _,_,Vt=np.linalg.svd(Xc,full_matrices=False); Xc=Xc@Vt[:30].T
    D=_pairwise_sq_dist(Xc); target=np.log(perplexity); Pcond=np.zeros((n,n),float)
    for i in range(n):
        beta=1.0; lo,hi=-np.inf,np.inf; mask=np.arange(n)!=i; Di=D[i,mask]; p=None
        for _ in range(45):
            w=np.exp(-Di*beta); p=w/max(w.sum(),np.finfo(float).tiny); H=-np.sum(p*np.log(np.maximum(p,np.finfo(float).tiny)))
            if abs(H-target)<1e-5: break
            if H>target: lo=beta; beta=beta*2 if np.isinf(hi) else (beta+hi)/2
            else: hi=beta; beta=beta/2 if np.isinf(lo) else (beta+lo)/2
        Pcond[i,mask]=p
    P=np.maximum((Pcond+Pcond.T)/(2*n),1e-12); rng=np.random.default_rng(seed); Y=rng.normal(0,1e-4,(n,2)); vel=np.zeros_like(Y)
    for it in range(int(n_iters)):
        num=1/(1+_pairwise_sq_dist(Y)); np.fill_diagonal(num,0); Q=np.maximum(num/max(num.sum(),np.finfo(float).tiny),1e-12); Puse=P*(4 if it<100 else 1); coeff=(Puse-Q)*num
        grad=4*np.sum(coeff[:,:,None]*(Y[:,None,:]-Y[None,:,:]),axis=1); mom=.5 if it<100 else .8; vel=mom*vel-200*grad; Y+=vel; Y-=Y.mean(axis=0)
    return Y
