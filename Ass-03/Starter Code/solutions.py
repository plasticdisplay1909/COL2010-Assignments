"""COL2010 Homework 3: Clustering - reference solutions."""
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
def calculate_squared_euclidean_distance(x_q: np.ndarray, x_r: np.ndarray) -> float:
    value = float(np.sum((np.asarray(x_q, float) - np.asarray(x_r, float)) ** 2))
    # print(f"SQUARED_EUCLIDEAN_DISTANCE: {value:.4f}")
    return value


def build_assignment_indicator_matrix(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    D = np.sum((X[:, None, :] - C[None, :, :]) ** 2, axis=2)
    labels = np.argmin(D, axis=1)
    R = np.zeros((len(X), len(C)), int)
    R[np.arange(len(X)), labels] = 1
    # print(f"ASSIGNMENT_INDICATOR_ACTIVE_COUNT: {int(R.sum())}")
    return R


def update_cluster_centroids(X: np.ndarray, assignments: np.ndarray, K: int) -> np.ndarray:
    A = np.asarray(assignments)
    labels = np.argmax(A, axis=1) if A.ndim == 2 else A.astype(int)
    C = np.zeros((K, X.shape[1]))
    gm = X.mean(axis=0)
    for k in range(K):
        C[k] = X[labels == k].mean(axis=0) if np.any(labels == k) else gm
    # print(f"UPDATED_CENTROID_SUM: {C.sum():.4f}")
    return C


def compute_kmeans_distortion(X: np.ndarray, C: np.ndarray, labels: np.ndarray) -> float:
    value = float(np.sum((X - C[np.asarray(labels, int)]) ** 2))
    # print(f"KMEANS_DISTORTION: {value:.4f}")
    return value


def initialize_kmeans_plus_plus(X: np.ndarray, K: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = [int(rng.integers(n))]
    while len(idx) < K:
        D = np.min(np.sum((X[:, None, :] - X[idx][None, :, :]) ** 2, axis=2), axis=1)
        D[idx] = 0
        if D.sum() <= 0:
            remaining = np.setdiff1d(np.arange(n), np.asarray(idx, int))
            idx.append(int(rng.choice(remaining)))
        else:
            idx.append(int(rng.choice(n, p=D / D.sum())))
    C = X[idx].astype(float).copy()
    # print(f"KMEANS_PLUS_PLUS_INDEX_SUM: {sum(idx)}")
    return C


def mini_batch_centroid_update(X: np.ndarray, C: np.ndarray, batch_indices: np.ndarray, counts: np.ndarray) -> tuple:
    C = C.astype(float).copy()
    counts = counts.astype(int).copy()
    B = X[np.asarray(batch_indices, int)]
    for x in B:
        k = int(np.argmin(np.sum((C - x) ** 2, axis=1)))
        counts[k] += 1
        C[k] += (x - C[k]) / counts[k]
    # print(f"MINIBATCH_UPDATED_COUNT_SUM: {int(counts.sum())}")
    return C, counts


def _linkage(X: np.ndarray, labels: np.ndarray, a: int, b: int, method: str) -> float:
    
    A = X[labels == a]
    B = X[labels == b]

    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x.y, for all pairs at once
    d2 = (A * A).sum(axis=1)[:, None] + (B * B).sum(axis=1)[None, :] - 2.0 * (A @ B.T)
    np.maximum(d2, 0.0, out=d2)  # clip tiny negatives from rounding

    if method == 'single':
        return float(np.sqrt(d2.min()))
    if method == 'complete':
        return float(np.sqrt(d2.max()))
    if method == 'average':
        return float(np.sqrt(d2).mean())
    raise ValueError(f"Unknown linkage method: {method!r}")

def compute_single_linkage_distance(X:np.ndarray, labels:np.ndarray, a:int, b:int)->float:
    v = _linkage(X, labels, a, b, 'single')
    #print(f"SINGLE_LINKAGE_DISTANCE: {v:.4f}")
    return v


def compute_complete_linkage_distance(X:np.ndarray, labels:np.ndarray, a:int, b:int)->float:
    v = _linkage(X, labels, a, b, 'complete')
    #print(f"COMPLETE_LINKAGE_DISTANCE: {v:.4f}")
    return v


def compute_average_linkage_distance(X:np.ndarray, labels:np.ndarray, a:int, b:int)->float:
    v = _linkage(X, labels, a, b, 'average')
    #print(f"AVERAGE_LINKAGE_DISTANCE: {v:.4f}")
    return v


def lance_williams_update(d_ac: float, d_bc: float, d_ab: float,
                          n_a: int, n_b: int, n_c: int, linkage: str) -> float:
    """Update d(A union B, C); Ward inputs/output are squared dissimilarities."""
    if linkage == 'single':
        v = min(d_ac, d_bc)
    elif linkage == 'complete':
        v = max(d_ac, d_bc)
    elif linkage == 'average':
        v = (n_a*d_ac + n_b*d_bc) / (n_a + n_b)
    elif linkage == 'ward':
        v = ((n_a+n_c)*d_ac + (n_b+n_c)*d_bc - n_c*d_ab) / (n_a+n_b+n_c)
    else:
        raise ValueError('linkage must be single, complete, average, or ward')
    return float(v)


def count_clusters_at_dendrogram_cut(merge_distances: np.ndarray, cut_height: float) -> int:
    """Count clusters after all merges at height <= cut in a full binary tree."""
    n = len(merge_distances) + 1
    merges = int(np.sum(np.asarray(merge_distances) <= cut_height))
    k = n - merges
    return k


def compute_mean_silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """Mean Euclidean silhouette; singleton and one-cluster scores are zero.

    Every distinct label, including -1, is treated as an ordinary cluster.
    """
    X = np.asarray(X, float)
    labels = np.asarray(labels)
    if X.ndim != 2 or len(X) == 0 or labels.shape != (len(X),):
        raise ValueError("X must be non-empty and 2D, with one label per row")
    labs = np.unique(labels)
    if len(labs) < 2:
        return 0.0
    D = np.sqrt(np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2))
    vals = []
    for i in range(len(X)):
        same = labels == labels[i]
        if same.sum() == 1:
            vals.append(0.0)
            continue
        a = D[i, same].sum() / (same.sum() - 1)
        bs = [D[i, labels == lab].mean() for lab in labs if lab != labels[i]]
        b = min(bs) if bs else 0.0
        den = max(a, b)
        vals.append(0.0 if den == 0 else (b - a) / den)
    v = float(np.mean(vals))
    return v


def compute_cluster_purity(contingency: np.ndarray) -> float:
    """Rows are predicted clusters; columns are true classes. Empty total: 0."""
    c = np.asarray(contingency, float)
    v = float(c.max(axis=1).sum() / c.sum()) if c.sum() else 0.0
    return v


def compute_normalized_mutual_information(contingency: np.ndarray) -> float:
    """Geometric NMI; empty total: 0, two constant partitions: 1, one: 0."""
    c = np.asarray(contingency, float)
    n = c.sum()
    if n == 0:
        v = 0.0
    else:
        p = c / n
        r, q = p.sum(1), p.sum(0)
        if np.sum(r > 0) == 1 and np.sum(q > 0) == 1:
            v = 1.0
        elif np.sum(r > 0) == 1 or np.sum(q > 0) == 1:
            v = 0.0
        else:
            mi = sum(p[i, j] * (np.log(p[i, j]) - np.log(r[i]) - np.log(q[j]))
                     for i, j in np.argwhere(p > 0))
            hr = -sum(x * np.log(x) for x in r if x > 0)
            hc = -sum(x * np.log(x) for x in q if x > 0)
            v = float(min(1.0, max(0.0, mi / np.sqrt(hr * hc))))
    return v


def compute_rand_index(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Return the Rand index between two labelings; return 1 when N < 2."""
    y = np.asarray(labels_true)
    z = np.asarray(labels_pred)
    if y.ndim != 1 or y.shape != z.shape:
        raise ValueError("label arrays must be 1D and have the same shape")
    n = len(y)
    if n < 2:
        score = 1.0
    else:
        same_y = y[:, None] == y[None, :]
        same_z = z[:, None] == z[None, :]
        iu = np.triu_indices(n, k=1)
        score = float(np.mean(same_y[iu] == same_z[iu]))
    return score


def classify_dbscan_points(X: np.ndarray, eps: float, min_pts: int) -> np.ndarray:
    """Inclusive self-neighbourhoods; 2=core, 1=border, 0=noise."""
    X = np.asarray(X, float)
    D = np.sqrt(np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2))
    cnt = (D <= eps).sum(1)
    types = np.zeros(len(X), int)
    types[cnt >= min_pts] = 2
    for i in range(len(X)):
        if types[i] == 0 and np.any((D[i] <= eps) & (cnt >= min_pts)):
            types[i] = 1
    return types



def compute_pca_projection(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Return centered reduced-SVD coordinates; constant data project to zero."""
    X = np.asarray(X, float)
    if X.ndim != 2 or min(X.shape) == 0:
        raise ValueError("X must be a non-empty 2D array")
    if not isinstance(n_components, (int, np.integer)) or not 1 <= n_components <= min(X.shape):
        raise ValueError("n_components must be an integer between 1 and min(N, D)")
    # Translate before centering to reduce cancellation at large offsets.
    shifted = X - X[0]
    Xc = shifted - shifted.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = Xc @ Vt[:n_components].T
    return Z


def compute_birch_cf_statistics(X: np.ndarray) -> tuple:
    """Return additive CF statistics and geometry, using a stable RMS radius."""
    X = np.asarray(X, float)
    if X.ndim != 2 or min(X.shape) == 0:
        raise ValueError("X must be a non-empty 2D array")
    N = int(len(X))
    LS = X.sum(axis=0)
    SS = float(np.sum(X * X))
    shifted = X - X[0]
    mean_shift = shifted.mean(axis=0)
    centroid = X[0] + mean_shift
    radius = float(np.sqrt(np.mean(np.sum((shifted - mean_shift) ** 2, axis=1))))
    return N, LS, SS, centroid, radius


# -----------------------------------------------------------------------------
# Part II: complete algorithms
# -----------------------------------------------------------------------------
def custom_kmeans_clustering(X: np.ndarray, K: int, max_iters: int = 100,
                             seed: int = 42, init: str = 'kmeans++') -> tuple:
    rng = np.random.default_rng(seed)
    if init == 'kmeans++':
        C = initialize_kmeans_plus_plus(X, K, seed)
    elif init == 'random':
        C = X[rng.choice(len(X), K, replace=False)].astype(float).copy()
    else:
        raise ValueError("init must be 'random' or 'kmeans++'")
    history = []
    old_labels = None
    for _ in range(max_iters):
        labels = np.argmin(np.sum((X[:, None, :] - C[None, :, :])**2, axis=2), axis=1)
        C_new = update_cluster_centroids(X, labels, K)
        new_labels = np.argmin(np.sum((X[:, None, :] - C_new[None, :, :])**2, axis=2), axis=1)
        history.append(float(np.sum((X - C_new[new_labels])**2)))
        if old_labels is not None and (np.array_equal(new_labels, old_labels) or np.linalg.norm(C_new-C) < 1e-6):
            C = C_new
            labels = new_labels
            break
        C, old_labels = C_new, new_labels.copy()
    else:
        labels = np.argmin(np.sum((X[:, None, :] - C[None, :, :])**2, axis=2), axis=1)
    if not history:
        history.append(float(np.sum((X - C[labels])**2)))
    return labels, C, np.array(history)


def custom_minibatch_kmeans(X: np.ndarray, K: int, batch_size: int,
                            max_iters: int = 200, seed: int = 42) -> tuple:
    rng = np.random.default_rng(seed)
    C = initialize_kmeans_plus_plus(X, K, seed)
    counts = np.zeros(K, int)
    history = []
    for _ in range(max_iters):
        idx = rng.choice(len(X), min(batch_size, len(X)), replace=False)
        C, counts = mini_batch_centroid_update(X, C, idx, counts)
        labels = np.argmin(np.sum((X[:, None, :] - C[None, :, :])**2, axis=2), axis=1)
        history.append(float(np.sum((X - C[labels])**2)))
    labels = np.argmin(np.sum((X[:, None, :] - C[None, :, :])**2, axis=2), axis=1)
    return labels, C, np.array(history), counts


def custom_agglomerative_clustering(X: np.ndarray, K: int, linkage: str = 'average') -> tuple:
    """Agglomerative clustering using pairwise distances + Lance-Williams-style updates."""
    if linkage not in {'single', 'complete', 'average'}:
        raise ValueError("linkage must be single, complete, or average")
    n = len(X)
    if not (1 <= K <= n):
        raise ValueError("K must be between 1 and N")
    clusters = {i: [i] for i in range(n)}
    sizes = {i: 1 for i in range(n)}
    history = []
    next_id = n
    base = np.sqrt(np.sum((X[:, None, :] - X[None, :, :])**2, axis=2))
    dist = {(i, j): float(base[i, j]) for i in range(n) for j in range(i+1, n)}

    def getd(a, b):
        return dist[(a, b) if a < b else (b, a)]

    while len(clusters) > K:
        active = set(clusters)
        candidates = [(d, a, b) for (a, b), d in dist.items() if a in active and b in active]
        d_ab, a, b = min(candidates, key=lambda t: (t[0], t[1], t[2]))
        members = clusters.pop(a) + clusters.pop(b)
        na, nb = sizes.pop(a), sizes.pop(b)
        new_id = next_id; next_id += 1
        history.append((a, b, d_ab, len(members)))
        clusters[new_id] = members
        sizes[new_id] = na + nb
        for c in list(clusters):
            if c == new_id:
                continue
            da, db = getd(a, c), getd(b, c)
            if linkage == 'single':
                newd = min(da, db)
            elif linkage == 'complete':
                newd = max(da, db)
            else:
                newd = (na*da + nb*db) / (na+nb)
            dist[(c, new_id) if c < new_id else (new_id, c)] = float(newd)
    labels = np.empty(n, int)
    for lab, members in enumerate(clusters.values()):
        labels[members] = lab
    H = np.array(history, float) if history else np.empty((0, 4))
    return labels, H


def custom_dbscan(X: np.ndarray, eps: float, min_pts: int) -> np.ndarray:
    n = len(X)
    D = np.sqrt(np.sum((X[:, None, :] - X[None, :, :])**2, axis=2))
    neigh = [np.where(D[i] <= eps)[0] for i in range(n)]
    labels = np.full(n, -1, int)
    visited = np.zeros(n, bool)
    cluster_id = 0
    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        if len(neigh[i]) < min_pts:
            continue
        labels[i] = cluster_id
        q = deque(neigh[i].tolist())
        queued = set(q)
        while q:
            j = q.popleft()
            if not visited[j]:
                visited[j] = True
                if len(neigh[j]) >= min_pts:
                    for h in neigh[j]:
                        h = int(h)
                        if h not in queued:
                            q.append(h)
                            queued.add(h)
            if labels[j] == -1:
                labels[j] = cluster_id
        cluster_id += 1
    return labels


def _cf_radius(N: int, LS: np.ndarray, SS: float) -> float:
    centroid=LS/N
    return float(np.sqrt(max(SS/N-float(centroid@centroid),0.0)))


def custom_birch_clustering(X: np.ndarray, K: int, threshold: float, seed: int = 42) -> tuple:
    X=np.asarray(X,float)
    if X.ndim!=2 or len(X)==0: raise ValueError("X must be a non-empty 2D array")
    if K<1 or threshold<0: raise ValueError("K must be positive and threshold non-negative")
    entries=[]
    for x in X:
        if not entries:
            entries.append([1,x.copy(),float(x@x)]); continue
        centroids=np.vstack([LS/N for N,LS,SS in entries])
        j=int(np.argmin(np.sum((centroids-x)**2,axis=1)))
        N,LS,SS=entries[j]; newN=N+1; newLS=LS+x; newSS=SS+float(x@x)
        if _cf_radius(newN,newLS,newSS)<=threshold: entries[j]=[newN,newLS,newSS]
        else: entries.append([1,x.copy(),float(x@x)])
    cf_counts=np.array([e[0] for e in entries],int)
    cf_centroids=np.vstack([e[1]/e[0] for e in entries])
    cf_radii=np.array([_cf_radius(e[0],e[1],e[2]) for e in entries])
    leaf_labels,_,_=custom_kmeans_clustering(cf_centroids,min(K,len(entries)),max_iters=100,seed=seed,init='kmeans++')
    nearest=np.argmin(np.sum((X[:,None,:]-cf_centroids[None,:,:])**2,axis=2),axis=1)
    labels=leaf_labels[nearest]
    return labels,cf_counts,cf_centroids,cf_radii


def _contingency(y, z):
    uy = np.unique(y); uz = np.unique(z)
    return np.array([[np.sum((y == a) & (z == b)) for b in uz] for a in uy])




def run_clustering_laboratory(X: np.ndarray, labels_true: np.ndarray, K: int,
                              dbscan_eps: float, dbscan_min_pts: int,
                              birch_threshold: float = 0.75, seed: int = 42) -> dict:
    labels_true=np.asarray(labels_true); results={}
    def metrics(labels):
        return {'silhouette':compute_mean_silhouette_score(X,labels),'nmi':compute_normalized_mutual_information(_contingency(labels_true,labels)),'rand_index':compute_rand_index(labels_true,labels)}
    lk,C,h=custom_kmeans_clustering(X,K,seed=seed); results['kmeans']={'labels':lk,**metrics(lk),'distortion':float(h[-1]),'centroids':C}
    lm,Cm,hm,counts=custom_minibatch_kmeans(X,K,batch_size=min(40,len(X)),seed=seed+1); results['minibatch_kmeans']={'labels':lm,**metrics(lm),'distortion':float(hm[-1]),'centroids':Cm,'counts':counts}
    la,H=custom_agglomerative_clustering(X,K,linkage='average'); results['agglomerative_average']={'labels':la,**metrics(la),'merge_history':H}
    ld=custom_dbscan(X,dbscan_eps,dbscan_min_pts); results['dbscan']={'labels':ld,**metrics(ld),'noise_count':int((ld==-1).sum())}
    lb,cc,ccen,cr=custom_birch_clustering(X,K,birch_threshold,seed=seed+2); results['birch']={'labels':lb,**metrics(lb),'cf_counts':cc,'cf_centroids':ccen,'cf_radii':cr,'leaf_cf_count':int(len(cc))}
    return results
