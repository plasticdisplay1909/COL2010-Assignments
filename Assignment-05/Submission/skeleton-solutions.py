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
    # Calculate mean and std except first column
    mean=np.mean(X[:,1:], axis=0)
    std=np.std(X[:,1:],axis=0)

    # Standardization as asked in the question
    X_new=X.copy()      # To not modify the input array
    X_new[:,1:]=(X[:,1:] - mean) / (std+eps)        # Eps used to avoid division by zero

    return X_new

normalize_features = standardize
feature_normalization = standardize

def select_degree(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, degrees: tuple[int, ...]) -> tuple[int, np.ndarray]:
    error=[]

    for d in degrees:
        # Create training,validation matrix
        phi_train=design_matrix(x_train,d)
        phi_val=design_matrix(x_val,d)

        # Find ols
        w=ols(phi_train,y_train)

        # Predict validation values
        y_pred=predict(phi_val,w)

        error.append(mse(y_val,y_pred))

    best_idx=np.argmin(error)
    return int(degrees[best_idx]),np.array(error)

def select_ridge(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, degree: int, lambdas: tuple[float, ...]) -> tuple[float, np.ndarray]:
    phi_train=design_matrix(x_train,degree)
    phi_val=design_matrix(x_val,degree)

    error=[]
    for x in lambdas:
        # Find ridge
        w=ridge(phi_train,y_train,x)

        # Predict validation values
        y_pred=predict(phi_val,w)
        error.append(mse(y_val,y_pred))

    best_idx=np.argmin(error)
    return float(lambdas[best_idx]),np.array(error)

def learning_curve(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, degree: int, sizes: tuple[int, ...], seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng=np.random.default_rng(seed)

    n=len(x_train)
    index=rng.permutation(n)

    error_train,error_val=[],[]

    for x in sizes:
        idx=index[:x]

        x_sub,y_sub=x_train[idx],y_train[idx]

        # Create design matrics
        phi_train,phi_val=design_matrix(x_sub,degree),design_matrix(x_val,degree)

        w=ols(phi_train,y_sub)
        train_pred=predict(phi_train,w)
        val_pred=predict(phi_val,w)

        error_train.append(mse(y_sub,train_pred))
        error_val.append(mse(y_val,val_pred))

    return np.array(error_train),np.array(error_val)

