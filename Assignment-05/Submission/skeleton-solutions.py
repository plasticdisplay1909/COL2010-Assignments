"""COL2010 Regression HW5: complete the eleven functions. Do not change signatures."""
import numpy as np

def design_matrix(X: np.ndarray, degree: int) -> np.ndarray:
    phi=np.ones((len(X),degree+1))

    for i in range(degree+1):
        phi[:,i]=X**i
    return phi

def predict(Phi: np.ndarray, w: np.ndarray) -> np.ndarray:
    return Phi@w

def mse(y: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y-y_pred)**2))

def ols(Phi: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(Phi,y,rcond=None)[0]

def projection_diagnostics(Phi: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    w=ols(Phi,y)
    y_pred=predict(Phi,w)

    residual=y-y_pred
    violation=np.max(np.abs(Phi.T @ residual))

    return y_pred,residual,float(violation)

def ridge(Phi: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    N,M=Phi.shape

    ## Create identity matrix
    P=np.eye(M)
    P[0,0]=0        ## Remove intercept penalty

    # Solves linear system
    return np.linalg.solve(Phi.T @ Phi + lam*P,Phi.T @ y)


def split_indices(n: int, train_fraction: float, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng=np.random.default_rng(seed)

    idx=rng.permutation(n)

    n_train=int(np.floor(n*train_fraction))
    n_val=int(np.floor(n*val_fraction))

    train_idx=idx[:n_train]
    val_idx=idx[n_train:n_train+n_val]
    test_idx=idx[n_train+n_val:]

    return train_idx,val_idx,test_idx

def standardize(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    raise NotImplementedError("Implement this task")

normalize_features = standardize
feature_normalization = standardize

def select_degree(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, degrees: tuple[int, ...]) -> tuple[int, np.ndarray]:
    raise NotImplementedError("Implement this task")

def select_ridge(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, degree: int, lambdas: tuple[float, ...]) -> tuple[float, np.ndarray]:
    raise NotImplementedError("Implement this task")

def learning_curve(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, degree: int, sizes: tuple[int, ...], seed: int) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError("Implement this task")

