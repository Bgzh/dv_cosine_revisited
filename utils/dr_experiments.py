import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.datasets import fetch_openml, fetch_olivetti_faces, fetch_20newsgroups_vectorized
from sklearn.utils import check_random_state
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from .dbowe import DBOWEmbedding

def shuffle_and_sample(X, y, n=10000, random_seed=0):
    if X.shape[0] <= n:
        random_state = check_random_state(random_seed)
        permutation = random_state.permutation(X.shape[0])
        X = X[permutation]
        y = y[permutation]
    else:
        X, _, y, _ = train_test_split(X, y, train_size=n, stratify=y, random_state=random_seed)
    return X, y

def get_datasets_dr():
    '''
    return {
        'names': ["mnist", "olivetti_faces", "20newgroups"],
        'datasets': [(X0, y0), ...]
    }
    data are shuffled and sampled by size of min(original_size, 10000)
    '''
    names = ["mnist", "olivetti_faces", "20newgroups"]
    datasets = []
    datasets.append(shuffle_and_sample(*fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)))
    datasets.append(shuffle_and_sample(*fetch_olivetti_faces(return_X_y=True, shuffle=False)))
    datasets.append(shuffle_and_sample(*fetch_20newsgroups_vectorized(return_X_y=True, normalize=True)))
    r = {
        'names': names,
        'datasets': datasets
    }
    return r

def pca_dbowe(X, qt, n_components=50, n_epoch=20):
    '''
    return (X, X_pca, X_dbowe)
    '''
    db = DBOWEmbedding(n=n_components, n_epoch=n_epoch)
    X_db = db.fit_transform((X/qt).astype(np.int64))
    pca = PCA(n_components) if isinstance(X, np.ndarray) else TruncatedSVD(n_components)
    X_pca = pca.fit_transform(X.copy())
    return X_pca, X_db

def get_rank(X_ind):
    '''
    X_ind = np.argmin(X, axis=-1)
    '''
    rank = np.empty_like(X_ind, dtype=np.int64)
    arange = np.arange(rank.shape[0])
    rank[arange[:, None], X_ind] = arange + 1
    return rank

def get_m12_scores(X, X_embedded, ks=[5,], metric="euclidean"):
    '''
    get "trustworthiness" and "continuity", for a list of k values (n_neighbours)
    X_embedded_metric: metric for X_embedded, X always uses euclidean
    ks: list of ints or floats of (0, 0.5)
    return [trustworthiness_k for k in ks], [continuity_k for k in ks]
    '''
    if ks[0] < 1:
        ks = [int(k*X.shape[0]) for k in ks]
    X_dis = pairwise_distances(X, metric=metric)
    np.fill_diagonal(X_dis, np.inf)
    X_emb_dis = pairwise_distances(X_embedded, metric=metric)
    np.fill_diagonal(X_emb_dis, np.inf)
    X_ind = np.argsort(X_dis, axis=-1)
    X_emb_ind = np.argsort(X_emb_dis, axis=-1)
    X_rank = get_rank(X_ind)
    X_emb_rank = get_rank(X_emb_ind)

    n = X_rank.shape[0]
    arange = np.arange(n)
    k_max = max(ks)
    R_m1 = X_rank[arange[:, None], X_emb_ind[:, :k_max]]
    R_m2 = X_emb_rank[arange[:, None], X_ind[:, :k_max]]

    m1_list = []
    m2_list = []
    for k in ks:
        if k:
            factor = 2 / (n * k * (2 * n - 3 * k - 1))
            T_m1 = R_m1[:, :k] - k
            m1 = 1 - factor * np.sum(T_m1[T_m1>0])
            T_m2 = R_m2[:, :k] - k
            m2 = 1 - factor * np.sum(T_m2[T_m2>0])
            m1_list.append(m1)
            m2_list.append(m2)
        else:
            m1_list.append(1.)
            m2_list.append(1.)
    return m1_list, m2_list

def batch_dr(dr_datasets, qt_list, repeat=1):
    reduced = []
    for _ in range(repeat):
        for (X, y), qt in zip(dr_datasets["datasets"], qt_list):
            reduced.append(pca_dbowe(X, qt))
    return reduced

def evaluate_dr_methods(dr_datasets, reduced, ks, metric="euclidean", repeat=1):
    '''
    repeat should equal to repeat in preceding batch_dr call
    return df_error, df_m1, df_m2
    '''
    knn = KNeighborsClassifier(n_neighbors=1, metric=metric)
    method_names = ["None", "PCA", "DBOWE"]
    nn_errors = []
    m1_scores = []
    m2_scores = []
    for i, dataset_name in enumerate(dr_datasets["names"]*repeat):
        X, y = dr_datasets["datasets"][i%len(dr_datasets["datasets"])]
        X_pca, X_cb = reduced[i]
        for mname, X_ in zip(method_names, [X, X_pca, X_cb]):
            nn_error = 1 - cross_val_score(knn, X_, y, cv=5).mean()
            nn_errors.append({
                "method":mname,
                "dataset": dataset_name,
                "error": nn_error
            })
            if mname=="None":
                continue
            m1s, m2s = get_m12_scores(X, X_, ks=ks, metric=metric)
            for k, m1, m2 in zip(ks, m1s, m2s):
                m1_scores.append({
                    "method": mname,
                    "dataset": dataset_name,
                    "trustworthiness": m1,
                    "k": k,
                })
                m2_scores.append({
                    "method": mname,
                    "dataset": dataset_name,
                    "continuity": m2,
                    "k": k,
                })
    df_error = pd.DataFrame.from_records(nn_errors).pivot_table(index="method", columns="dataset", values="error", aggfunc=[np.mean, np.std])
    df_m1 = pd.DataFrame.from_records(m1_scores)
    df_m2 = pd.DataFrame.from_records(m2_scores)
    return df_error, df_m1, df_m2
