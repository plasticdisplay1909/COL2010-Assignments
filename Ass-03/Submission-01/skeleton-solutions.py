"""COL2010 Homework 3: Clustering - student skeleton."""
from __future__ import annotations

from collections import deque
import numpy as np

DATA_PATH = "module3_data.npz"


def _load(path: str = DATA_PATH):
    return np.load(path, allow_pickle=True)


# -----------------------------------------------------------------------------
# Input helpers used by the autograder / students for local testing
# -----------------------------------------------------------------------------
def get_task1_inputs(): d = _load(); return d['query'], d['reference']
def get_task2_inputs(): d = _load(); return d['X_small'], d['C_small']
def get_task3_inputs(): d = _load(); return d['X_small'], np.eye(3, dtype=int)[d['labels_small'].astype(int)], 3
def get_task4_inputs(): d = _load(); return d['X_small'], d['C_small'], d['labels_small']
def get_task5_inputs(): d = _load(); return d['X_small'], 3, 7
def get_task6_inputs(): d = _load(); return d['X_blobs'], d['C_small'], d['batch_idx'], np.ones(3, int)
def get_task7_inputs(): d = _load(); return d['X_small'], d['labels_small'], 0, 1
def get_task8_inputs(): d = _load(); return d['X_small'], d['labels_small'], 0, 1
def get_task9_inputs(): d = _load(); return d['X_small'], d['labels_small'], 0, 1
def get_task10_inputs(): return 1.2, 2.5, 3.0, 4, 6, 5, 'average'
def get_task11_inputs(): d = _load(); return d['merge_distances'], 2.0
def get_task12_inputs(): d = _load(); return d['X_small'], d['labels_small']
def get_task13_inputs(): d = _load(); return d['contingency']
def get_task14_inputs(): d = _load(); return d['contingency']
def get_task15_inputs(): d = _load(); return d['labels_small'], np.array([1,1,1,0,0,2,2,2,2,0])
def get_task16_inputs(): d = _load(); return d['X_small'], float(d['eps']), int(d['min_pts'])
def get_task17_inputs(): d = _load(); return d['X_high'], 2
def get_task18_inputs(): d = _load(); return d['X_small'][:6]
def get_task19_inputs(): d = _load(); return d['X_blobs'], 3
def get_task20_inputs(): d = _load(); return d['X_blobs'], 3, 40
def get_task21_inputs(): d = _load(); return d['X_small'], 3, 'average'
def get_task22_inputs(): d = _load(); return d['X_moons'], .20, 5
def get_task23_inputs(): d = _load(); return d['X_blobs'], 3, .80
def get_task24_inputs():
    d = _load()
    return d['synth_blobs_X'], d['synth_blobs_y'], 3, .55, 5, .65


def get_laboratory_dataset_names():
    return ['synth_blobs','synth_aniso','synth_unequal','synth_moons','synth_circles','synth_noise','synth_highdim','iris','wine','penguins','wholesale','digits','scale_500','scale_2000','scale_6000']


def get_laboratory_dataset(name: str):
    d = _load(); key = str(name).strip().lower()
    if key not in get_laboratory_dataset_names(): raise ValueError(f"unknown dataset {name!r}")
    return d[f'{key}_X'], d[f'{key}_y']


def available_datasets(): return get_laboratory_dataset_names()
def load_dataset(name: str): return get_laboratory_dataset(name)


# -----------------------------------------------------------------------------
# Part I: foundational components
# -----------------------------------------------------------------------------

# -------------------------     Q-1          -----------------------------------
def calculate_squared_euclidean_distance(x_q: np.ndarray, x_r: np.ndarray) -> float:
    ans=0
    n=x_q.shape[0]

    for i in range(n):
        ans+=(abs(x_q[i] - x_r[i]))**2

    # print(ans)
    return ans

## ---- Testing Task-01     -----
# x,y=get_task1_inputs()
# print(f"X:{x}\nY:{y}")
# calculate_squared_euclidean_distance(x,y)


# -------------------------     Q-2          -----------------------------------
def build_assignment_indicator_matrix(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    a,b=X.shape     # N x D matrix
    c,d=C.shape     # K x D matrix

    distances=[[0]*c for _ in range(a)]
    # print(distances)

    for i in range(a):
        for j in range(c):
            distances[i][j]=float(calculate_squared_euclidean_distance(X[i],C[j]))
    # print(distances)

    cluster=[[0]*c for _ in range(a)]
    for i in range(a):
        mn=min(distances[i])

        for j in range(c):
            if distances[i][j]==mn: 
                cluster[i][j]=1
                break

    # print(cluster)
    cluster=np.array(cluster,int)
    return cluster

## ---- Testing Task-02     -----
# X,C=get_task2_inputs()
# print(f"X:{X}\nC:{C}")
# res=build_assignment_indicator_matrix(X,C)


# -------------------------     Q-3          -----------------------------------
def update_cluster_centroids(X: np.ndarray, assignments: np.ndarray, K: int) -> np.ndarray:
    N,D=X.shape
    "Case when assignment is already as cluster"
    if assignments.ndim==1:
        tmp=[[0]*K for _ in range(N)]

        for i in range(N):
            tmp[i][assignments[i]]=1

        assignments=np.array(tmp)
    
    "Calculating number of points in each cluster"
    points=[0]*K
    for i in range(N):
        for j in range(K):
            points[j]+=assignments[i][j]
    # print(points)


    "Calculating global mean"
    mean=[0]*D
    for i in range(N):
        for j in range(D):
            mean[j]+=X[i][j]
    # print(mean)
    for i in range(D): mean[i]/=N

    "Calculating centres"
    centres=[[0]*D for _ in range(K)]

    for i in range(N):
        for j in range(K):
            for k in range(D):
                centres[j][k] +=assignments[i][j] * X[i][k]

    "Calculating mean for each centres"
    for i in range(K):
        if points[i]==0:
            "Case of empty cluster"
            centres[i]=mean.copy()
        else:
            for j in range(D):
                centres[i][j]/=points[i]

    centres=np.array(centres,float)
    # print(centres)
    return centres


## ---- Testing Task-03     -----
# X,ass,K=get_task3_inputs()
# print(f"X:{X}\nassignments:{ass}\nK:{K}")
# update_cluster_centroids(X,ass,K)


# -------------------------     Q-4          -----------------------------------
def compute_kmeans_distortion(X: np.ndarray, C: np.ndarray, labels: np.ndarray) -> float:
    N,D=X.shape
    K=C.shape[0]

    J=0
    for i in range(N):
        J+=float(calculate_squared_euclidean_distance(X[i],C[labels[i]]))
        
    # print(J)
    return J

## ---- Testing Task-04     -----
# X,C,labels=get_task4_inputs()
# compute_kmeans_distortion(X,C,labels)

# -------------------------     Q-5          -----------------------------------
def initialize_kmeans_plus_plus(X: np.ndarray, K: int, seed: int = 42) -> np.ndarray:
    raise NotImplementedError

 
## ---- Testing Task-05     -----
# X,K,seed=get_task5_inputs()
# initialize_kmeans_plus_plus(X,K,seed)


# -------------------------     Q-6          -----------------------------------
def mini_batch_centroid_update(X: np.ndarray, C: np.ndarray, batch_indices: np.ndarray, counts: np.ndarray) -> tuple:
    C=C.copy()
    counts=counts.copy()

    N,D=X.shape
    K,D=C.shape
    for idx in batch_indices:
        x=X[idx]

        "Finding cluster to which the centroid belongs"
        min_dist=float('inf')
        cluster=-1

        for i in range(K):
            dist=float(calculate_squared_euclidean_distance(x,C[i]))

            if dist<min_dist:
                min_dist=dist
                cluster=i

        # Update count
        counts[cluster]+=1

        # Update Centroid
        for i in range(D):
            C[cluster][i] += (X[idx][i] - C[cluster][i]) / counts[cluster]

    return (C,counts)

## ---- Testing Task-06     -----
# X,C,idx,count=get_task6_inputs()
# mini_batch_centroid_update(X,C,idx,count)

# print(f"{X.shape,C.shape,idx,count}")

# -------------------------     Q-7          -----------------------------------
def compute_single_linkage_distance(X: np.ndarray, labels: np.ndarray, a: int, b: int) -> float:
    clustera=[]
    clusterb=[]

    "Collecting points from cluster"
    N=X.shape[0]
    for i in range(N):
        if labels[i]==a: clustera.append(X[i].copy())
        elif labels[i]==b: clusterb.append(X[i].copy())

    "Computing minimum distance"
    min_dist=float('inf')
    A,B=len(clustera),len(clusterb)

    for i in range(A):
        for j in range(B):
            dist=float(np.sqrt(calculate_squared_euclidean_distance(clustera[i],clusterb[j])))
            min_dist=min(dist,min_dist)

    return min_dist

## ---- Testing Task-07     -----
# X,labels,a,b=get_task7_inputs()
# print(f"{X.shape,labels,a,b}")
# print(X)
# compute_single_linkage_distance(X,labels,a,b)

# -------------------------     Q-8          -----------------------------------
def compute_complete_linkage_distance(X: np.ndarray, labels: np.ndarray, a: int, b: int) -> float:
    clustera=[]
    clusterb=[]

    "Collecting points from cluster"
    N=X.shape[0]
    for i in range(N):
        if labels[i]==a: clustera.append(X[i].copy())
        elif labels[i]==b: clusterb.append(X[i].copy())

    "Computing maximum distance"
    max_dist=-float('inf')
    A,B=len(clustera),len(clusterb)

    for i in range(A):
        for j in range(B):
            dist=float(np.sqrt(calculate_squared_euclidean_distance(clustera[i],clusterb[j])))
            max_dist=max(dist,max_dist)

    return max_dist


# -------------------------     Q-9          -----------------------------------
def compute_average_linkage_distance(X: np.ndarray, labels: np.ndarray, a: int, b: int) -> float:
    clustera=[]
    clusterb=[]

    "Collecting points from cluster"
    N=X.shape[0]
    for i in range(N):
        if labels[i]==a: clustera.append(X[i].copy())
        elif labels[i]==b: clusterb.append(X[i].copy())


    "Computing total distance"
    total_dist=0
    A,B=len(clustera),len(clusterb)

    for i in range(A):
        for j in range(B):
            dist=float(np.sqrt(calculate_squared_euclidean_distance(clustera[i],clusterb[j])))
            total_dist+=dist

    return float(total_dist / (A*B))


# -------------------------     Q-10          -----------------------------------
def lance_williams_update(d_ac: float, d_bc: float, d_ab: float,
                          n_a: int, n_b: int, n_c: int, linkage: str) -> float:
    """Return the updated dissimilarity as a float without printing.

    Ward inputs/output are squared dissimilarities; see the task coefficients."""
    if linkage=='single': 
        alpha_a,alpha_b,beta,gamma=0.5,0.5,0,-0.5
    elif linkage=='complete':
        alpha_a,alpha_b,beta,gamma=0.5,0.5,0,0.5
    elif linkage=='average':
        beta=gamma=0
        alpha_a = (n_a)/(n_a+n_b)
        alpha_b = (n_b)/(n_a+n_b)
    elif linkage=='ward':
        gamma=0
        T=n_a+n_b+n_c

        alpha_a,alpha_b = (n_a+n_c)/T,(n_b+n_c)/T
        beta=-n_c/T
    else: raise ValueError

    return alpha_a*d_ac + alpha_b*d_bc + beta*d_ab + gamma*abs(d_ac-d_bc)

# -------------------------     Q-11          -----------------------------------
def count_clusters_at_dendrogram_cut(merge_distances: np.ndarray, cut_height: float) -> int:
    """Return the cluster count as an int without printing.

    Use the complete binary merge history; include heights equal to the cut."""
    if len(merge_distances)<1: return 1

    count=np.sum(merge_distances<=cut_height)
    return int(len(merge_distances))+1-count

## ---- Testing Task-11     -----
dist,cut=get_task11_inputs()
# print(f"{dist,cut}")

def compute_mean_silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """Return the mean Euclidean silhouette as a float without printing.

    Singleton/one-cluster scores are zero; treat -1 as an ordinary cluster."""
    # TODO Task 12
    raise NotImplementedError


def compute_cluster_purity(contingency: np.ndarray) -> float:
    """Return cluster purity as a float without printing.

    Rows=predicted clusters, columns=true classes; a zero total gives zero."""
    # TODO Task 13
    raise NotImplementedError


def compute_normalized_mutual_information(contingency: np.ndarray) -> float:
    """Return geometric NMI as a float without printing.

    Empty total: 0; both entropies zero: 1; just one zero: 0."""
    # TODO Task 14
    raise NotImplementedError


def compute_rand_index(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Return the Rand index as a float without printing.

    Count unordered pair agreements; return 1.0 for fewer than two rows."""
    # TODO Task 15
    raise NotImplementedError


def classify_dbscan_points(X: np.ndarray, eps: float, min_pts: int) -> np.ndarray:
    """Return an integer array of shape (N,) without printing.

    Use 2=core, 1=border, 0=noise; include self-neighbours and eps boundaries."""
    # TODO Task 16
    raise NotImplementedError



def compute_pca_projection(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Return PCA coordinates of shape (N, n_components) without printing.

    Require 1<=n_components<=min(N,D); constant data project to zero."""
    # TODO Task 17
    raise NotImplementedError


def compute_birch_cf_statistics(X: np.ndarray) -> tuple:
    """Return (N, LS, SS, centroid, radius) without printing.

    N is an int, LS/centroid are (D,) arrays, SS/radius are floats.
    A centered RMS radius is permitted."""
    # TODO Task 18
    raise NotImplementedError


# -----------------------------------------------------------------------------
# Part II: complete algorithms
# -----------------------------------------------------------------------------
def custom_kmeans_clustering(X: np.ndarray, K: int, max_iters: int = 100,
                             seed: int = 42, init: str = 'kmeans++') -> tuple:
    # TODO Task 19
    raise NotImplementedError


def custom_minibatch_kmeans(X: np.ndarray, K: int, batch_size: int,
                            max_iters: int = 200, seed: int = 42) -> tuple:
    # TODO Task 20
    raise NotImplementedError


def custom_agglomerative_clustering(X: np.ndarray, K: int, linkage: str = 'average') -> tuple:
    # TODO Task 21
    raise NotImplementedError


def custom_dbscan(X: np.ndarray, eps: float, min_pts: int) -> np.ndarray:
    # TODO Task 22
    raise NotImplementedError



def custom_birch_clustering(X: np.ndarray, K: int, threshold: float, seed: int = 42) -> tuple:
    # TODO Task 23
    raise NotImplementedError


def run_clustering_laboratory(X: np.ndarray, labels_true: np.ndarray, K: int,
                              dbscan_eps: float, dbscan_min_pts: int,
                              birch_threshold: float = 0.75, seed: int = 42) -> dict:
    # TODO Task 24
    raise NotImplementedError
