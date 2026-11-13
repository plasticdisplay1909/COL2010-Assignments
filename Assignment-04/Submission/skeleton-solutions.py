from __future__ import annotations
import numpy as np
from inputs import *
# from support import safeguarded_newton_update

## ----------------         Task - 01       ----------------- ##
def confusion_matrix(y_true, y_pred, classes) -> np.ndarray:
    C=len(classes)

    matrix=np.zeros((C,C),dtype=int)

    true_idx=np.searchsorted(classes,y_true)
    pred_idx=np.searchsorted(classes,y_pred)

    np.add.at(matrix,(true_idx,pred_idx),1)
    return matrix

## ----------------         Task - 02       ----------------- ##
def binary_metrics(y_true,y_pred) -> dict[str, float]:
    # Calculating each terms
    TP=np.sum((y_true==1) & (y_pred==1))
    FP=np.sum((y_true==0) & (y_pred==1))

    TN=np.sum((y_true==0) & (y_pred==0))
    FN=np.sum((y_true==1) & (y_pred==0))

    def helper(numerator,denominator):
        "Returns safe divide"
        if denominator==0: return 0.0
        return float(numerator/denominator)

    return {
        'accuracy' : helper(TP+TN,TP+TN+FP+FN),
        'precision' : helper(TP,TP+FP),
        'recall' : helper(TP,TP+FN),
        'f1': helper(2*TP,2*TP+FP+FN)
    }

## ----------------         Task - 03       ----------------- ##
def standardize_train_apply(X_train,X_other) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean=np.mean(X_train,axis=0)
    scale=np.std(X_train,axis=0,ddof=0)

    # Replace zero std to prevent zero divison
    scale=np.where(scale==0,1.0,scale)

    # Standardization
    X_train=(X_train-mean)/scale
    X_other=(X_other-mean)/scale

    return X_train,X_other,mean,scale

## ----------------         Task - 04       ----------------- ##
def knn_posterior(x,X_train,y_train,K,classes) -> np.ndarray:
    # Calculate Euclidian distance
    distance=np.linalg.norm(X_train-x,axis=1)

    # Find k-nearest neighbours
    near_idx=np.argsort(distance,kind='stable')[:K]
    near_label=y_train[near_idx]

    # Caluclating probability
    prob=np.array([np.sum(near_label==c)/K for c in classes],dtype=float)
    return prob

## ----------------         Task - 05       ----------------- ##
def bayes_and_one_nn_error(p) -> tuple[np.ndarray, np.ndarray]:

    bayes_error=np.minimum(p,1-p)
    one_nn_error=2*p*(1-p)

    return bayes_error,one_nn_error

## ----------------         Task - 06       ----------------- ##
def gaussian_log_likelihood(x,mean,var) -> float:
    # Standardization of variance to avoid zero division error
    var=np.maximum(var,1e-9)

    likelihood= np.sum(-(0.5) * np.log(2*np.pi*var) - (x-mean)**2/(2*var))

    return float(likelihood)

## ----------------         Task - 07       ----------------- ##
def gaussian_nb_parameters(X,y) -> dict[str, np.ndarray]:
    pass

## ----------------         Task - 08       ----------------- ##
def bernoulli_nb_parameters(X,y,alpha=1.) -> dict[str, np.ndarray]:
    classes=np.unique(y)
    N,d=X.shape

    C=len(classes)

    # Initialize probability
    prior,prob=np.zeros(C,dtype=float), np.zeros((C,d),dtype=float)

    for idx,c in enumerate(classes):
        # Current class
        curr=X[y==c]
        N_curr=len(curr)            # Number of samples

        prior[idx]=N_curr/N         # Prior Probability

        # Feature probability
        count=np.sum(curr,axis=0)
        prob[idx]=(count+alpha) / (N_curr + 2*alpha)

    # Output
    return {
        'classes' : classes,
        'prior': prior,
        'prob':prob
    }


def stable_sigmoid(z) -> np.ndarray:
    raise NotImplementedError

def binary_cross_entropy(y,p) -> float:
    raise NotImplementedError

def logistic_gradient(X,y,w,l2=0.) -> np.ndarray:
    raise NotImplementedError

def logistic_hessian(X,w,l2=0.) -> np.ndarray:
    raise NotImplementedError

def softmax_loss_probabilities(X,W,t) -> tuple[float, np.ndarray]:
    raise NotImplementedError

def svm_primal_objective(X,y_pm1,w,b,C) -> tuple[float, float]:
    raise NotImplementedError

def rbf_kernel(X,Z,gamma) -> np.ndarray:
    raise NotImplementedError

def node_impurities(class_counts) -> tuple[float, float]:
    raise NotImplementedError

def best_gini_split(X,y) -> tuple[int | None, float | None, float]:
    raise NotImplementedError

def bootstrap_oob(n_samples,seed=42) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError

def majority_vote(predictions,classes) -> np.ndarray:
    raise NotImplementedError

def select_f1_threshold(y_validation,p_validation,thresholds) -> tuple[float, np.ndarray]:
    raise NotImplementedError

def knn_model_selection(X_train,y_train,X_validation,y_validation,k_values) -> tuple[int, np.ndarray]:
    raise NotImplementedError

def naive_bayes_predict(X_train,y_train,X_query,kind='gaussian',alpha=1.) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raise NotImplementedError

def logistic_gradient_descent(X,y,learning_rate=0.1,max_steps=500,tol=1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Task 23 (required): batch gradient descent on mean binary cross-entropy."""
    raise NotImplementedError

# Optional extension: retained unchanged for TA compatibility.
def logistic_newton(X,y,l2=.1,max_steps=50,tol=1e-6) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError

def decision_tree_train_predict(X_train,y_train,X_query,max_depth=4,min_samples_split=2) -> tuple[dict[str, object], np.ndarray]:
    raise NotImplementedError

def forest_oob(X,y,X_query,n_estimators=30,max_depth=None,max_features=None,seed=42) -> dict[str, object]:
    raise NotImplementedError
