from __future__ import annotations
import numpy as np
from inputs import *
# from support import safeguarded_newton_update

## -------------            Task - 01       -----------------  ##
def confusion_matrix(y_true, y_pred, classes) -> np.ndarray:
    n=len(classes)
    classes.sort()
    matrix=[[0]*n for _ in range(n)]

    dict={}
    curr=0
    for i in classes:
        dict[i]=curr
        curr+=1

    for i in range(len(y_true)):
        row=dict[y_true[i]]
        column=dict[y_pred[i]]
        matrix[row][column]+=1

    matrix=np.array(matrix,dtype=int)
    print(matrix)
    return matrix

## ------       Testing Task 1      ------ 
# y1,y2,classes=get_task1_inputs()

## -------------            Task - 02       -----------------  ##
def binary_metrics(y_true,y_pred) -> dict[str, float]:
    tp,fp,tn,fn=0,0,0,0
    n=len(y_true)

    # Calculating each indicator
    for i in range(n):
        a,b=y_true[i],y_pred[i]
        if (a==1 and b==1): tp+=1
        elif (a==1 and b==0): fn+=1
        elif (a==0 and b==1): fp+=1
        else: tn+=1

    # Calculation as per assignment
    acc= (tp+tn) / n if n!=0 else 0.0
    prec= (tp)/(tp+fp) if (tp+fp)!=0 else 0.0
    recall=(tp)/(tp+fn) if (tp+fn)!=0 else 0.0
    f1= (2*tp) / (2*tp+fp+fn) if (2*tp+fp+fn)!=0 else 0.0

    # Output
    return {
        'accuracy':acc,
        'precision':prec,
        'recall':recall,
        'f1':f1
    }

## ------       Testing Task 2      ------ 
# y1,y2,classes=get_task1_inputs()
# print(y1,y2)
# binary_metrics(y1,y2)

## -------------            Task - 03       -----------------  ##
def standardize_train_apply(X_train,X_other) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean=np.mean(X_train,axis=0)
    scale=np.std(X_train,axis=0,ddof=0)

    # Avoid division by zero
    scale=np.where(scale==0,1.0,scale)

    # Standardize training data
    X_train_std= (X_train - mean) / scale
    X_other_std =(X_other - mean) / scale

    return X_train_std, X_other_std,mean,scale
## ------       Testing Task 3      ------ 
# X,Y=get_task3_inputs()
# print(X,Y)

def knn_posterior(x,X_train,y_train,K,classes) -> np.ndarray:
    raise NotImplementedError

x,x_train,y_train,K,classes=get_task4_inputs()
print(y_train.shape)

def bayes_and_one_nn_error(p) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError

def gaussian_log_likelihood(x,mean,var) -> float:
    raise NotImplementedError

def gaussian_nb_parameters(X,y) -> dict[str, np.ndarray]:
    raise NotImplementedError

def bernoulli_nb_parameters(X,y,alpha=1.) -> dict[str, np.ndarray]:
    raise NotImplementedError

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
