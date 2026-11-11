from __future__ import annotations
import numpy as np
DATA_FILENAME='module4_data.npz'

def sigmoid(z): return 1/(1+np.exp(-np.clip(z,-40,40)))

def main(seed:int=42):
    rng = np.random.default_rng(seed)
    # Main binary dataset, moderately imbalanced and overlapping
    n0, n1, d = 900,400,5
    A = rng.normal(size=(d,d)); cov0 = A@A.T/d + 0.4*np.eye(d)
    B = rng.normal(size=(d,d)); cov1 = B@B.T/d+0.5*np.eye(d)
    X0 = rng.multivariate_normal(np.array([1.6,1.0,0.0,1.4,-0.8]),cov0,n0)
    X1 = rng.multivariate_normal(np.array([-1.1,-0.7,1.5,-1.0,1.7]),cov1,n1)
    X = np.vstack([X0,X1]); y=np.r_[np.zeros(n0,int),np.ones(n1,int)]
    p = rng.permutation(len(y)); X, y = X[p], y[p]
    split = 950; Xtr, Xte = X[:split], X[split:]; ytr, yte = y[:split], y[split:]
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd==0] = 1
    Xtr_s = (Xtr-mu)/sd; Xte_s = (Xte-mu)/sd
    Xtr_a = np.c_[np.ones(len(Xtr_s)),Xtr_s]; Xte_a = np.c_[np.ones(len(Xte_s)),Xte_s]
    w = np.array([-0.25,-1.0,-0.7,0.9,-0.8,1.1])
    probs = sigmoid(Xtr_a[:240]@w)
    pred = (probs>=0.45).astype(int); truth = ytr[:240]
    # Bernoulli text-like data
    V = 12; m = 300
    yb = rng.integers(0,2,m)
    pb0 = np.linspace(.05,.35,V); pb1=np.linspace(.25,.75,V)
    Xb = np.vstack([rng.binomial(1,pb1 if c else pb0) for c in yb])
    # Multiclass softmax example
    Z = rng.normal(size=(8,4)); Wm=rng.normal(size=(4,3)); tm=rng.integers(0,3,8)
    # SVM example
    ypm = 2*ytr[:220]-1; svmX = Xtr_s[:220]; svmw = np.array([.6,-.4,.8,-.5,.7]); scores=svmX@svmw
    # tree split example
    tree_x = np.array([1.0,1.4,2.2,2.8,3.1,3.6,4.0,4.4,5.0,5.7]); tree_y = np.array([0,0,0,1,0,1,1,1,1,0])
    np.savez(DATA_FILENAME,X_train=Xtr,y_train=ytr,X_test=Xte,y_test=yte,
             X_train_std=Xtr_s,X_test_std=Xte_s,X_train_aug=Xtr_a,X_test_aug=Xte_a,
             metric_true=truth,metric_pred=pred,probabilities=probs,
             knn_query=Xte_s[0],knn_reference=Xtr_s[:80],knn_labels=ytr[:80],
             gaussian_X=Xtr_s[:300],gaussian_y=ytr[:300],
             bernoulli_X=Xb,bernoulli_y=yb,
             softmax_Z=Z,softmax_W=Wm,softmax_t=tm,
             svm_X=svmX,svm_y=ypm,svm_w=svmw,
             tree_feature=tree_x,tree_labels=tree_y,
             thresholds=np.linspace(.1,.9,17),bootstrap_n=np.array(500))
    print(f'[+] Generated {DATA_FILENAME}: train={Xtr.shape}, test={Xte.shape}')
if __name__=='__main__':
    main()
    from dataset_suite import generate_suite
    generate_suite()
