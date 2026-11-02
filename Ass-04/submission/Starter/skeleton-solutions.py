from __future__ import annotations
import numpy as np
from inputs import *
from support import safeguarded_newton_update

def confusion_matrix(y_true, y_pred, classes) -> np.ndarray:
    raise NotImplementedError

def binary_metrics(y_true,y_pred) -> dict[str, float]:
    raise NotImplementedError

def standardize_train_apply(X_train,X_other) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raise NotImplementedError

def knn_posterior(x,X_train,y_train,K,classes) -> np.ndarray:
    raise NotImplementedError

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
