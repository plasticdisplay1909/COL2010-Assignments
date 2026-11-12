"""Provided helpers; plotting, baseline solvers and numerical safeguard are not assessed."""
import numpy as np

def safeguarded_newton_update(X,y,w,direction,l2):
    def loss(v):return np.mean(np.logaddexp(0,X@v)-y*(X@v))+.5*l2*np.sum(v[1:]**2)
    before=loss(w);a=1.
    for _ in range(60):
        candidate=w-a*direction
        if loss(candidate)<=before+1e-14:return candidate
        a*=.5
    return w.copy()

def plot_loss(history,title='Loss'):
    import matplotlib.pyplot as plt
    plt.plot(np.arange(len(history)),history);plt.xlabel('Newton updates');plt.ylabel('Regularized mean loss');plt.title(title);plt.show()

def plot_decision_regions(predict,X,y,title='Decision regions'):
    """predict accepts a (M,2) matrix. Use the same plotting bounds for comparisons."""
    import matplotlib.pyplot as plt
    if X.shape[1]!=2:raise ValueError('Use original 2D data, or its specified standardized coordinates.')
    xs=np.linspace(X[:,0].min()-.4,X[:,0].max()+.4,150);ys=np.linspace(X[:,1].min()-.4,X[:,1].max()+.4,150)
    a,b=np.meshgrid(xs,ys);q=np.c_[a.ravel(),b.ravel()];classes=np.unique(y);pred=predict(q)
    encoded=np.array([np.flatnonzero(classes==c)[0] for c in pred]).reshape(a.shape)
    plt.contourf(a,b,encoded,alpha=.22,cmap='tab10');plt.scatter(X[:,0],X[:,1],c=np.searchsorted(classes,y),cmap='tab10',s=12);plt.title(title);plt.show()

def baseline_predict(name,X_train,y_train,X_query,**parameters):
    if name in ('linear_svm','rbf_svm'):
        from sklearn.svm import SVC
        model=SVC(kernel='linear' if name=='linear_svm' else 'rbf',**parameters)
    elif name=='pruned_tree':
        from sklearn.tree import DecisionTreeClassifier
        model=DecisionTreeClassifier(random_state=42,**parameters)
    elif name=='multiclass_logistic':
        from sklearn.linear_model import LogisticRegression
        model=LogisticRegression(max_iter=2000,**parameters)
    else:raise ValueError('Unknown baseline')
    return model.fit(X_train,y_train).predict(X_query)
