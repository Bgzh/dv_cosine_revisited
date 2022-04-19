import json
import subprocess
import os
import numpy as np
from scipy.sparse import spmatrix
from collections import deque

class DBOWEmbedding:
    def __init__(self, n, keep_files=False,
            lr=1e-4, n_epoch=20, sub_samp=False,
            filename="temp_data_path", n_dim=0, min_tf=0, nbA=2, nbB=3,
            vec_path="temp_vectors.jsonl", log_path="temp_log.jsonl",
            test=False, Cs=[], verbose=0):
        '''
        n: number of dimensions of the target vectors
        keep_files:
            True: keep all the temp files
            False: delete temp files on destruction of this object
        lr: learning rate
        nEpoch: number of epochs
        sub_samp: whether to use sub-sampling with naive bayesian weights
        filename: path to temp data file
        n_dim: number of dimensions of the original vectors
        min_tf: min token frequency
        nbA, nbB: params in sub-sampling
        vec_path: path to temp file of vectors
        log_path: path to temp file of log
        test: whether to test during training
        Cs: grid of C in test during training
        verbose: 0, 1 or 2
        '''
        config={
            'filename': filename,
            'nDim': n_dim,
            'nb': sub_samp,
            'n': n,
            'minTf': min_tf,
            'lr': lr,
            'nEpoch': n_epoch,
            'subSamp': sub_samp,
            'nbA': nbA,
            'nbB': nbB,
            'vecPath': vec_path,
            'logPath': log_path,
            'test': test,
            'Cs': Cs,
            'verbose': verbose
        }
        self.config_path = "config.json"
        self.config = config
        self.keep_files = keep_files
        subprocess.run(['javac', '-cp', 'dvscript;build/jars/gson-2.8.9.jar', '-d', 'build/classes', 'dvscript/dv/cosine/java/Run.java'])

    def fit_transform(self, X_train, y_train=None, 
        X_dev=None, y_dev=None, X_test=None, y_test=None,
        **parmas):
        '''
        X: 2d np array or scipy sparse of int
        y: 1d array-like of int, needed for sub-sampling or testing
        '''
        n_dim = X_train.shape[1]
        self.config.update(parmas)
        self.set_params(nDim=n_dim)
        Xs = [X_train, X_dev, X_test]
        ys = [y_train, y_dev, y_test]
        splits = ["train", "dev", "test"]
        arange = np.arange(n_dim)
        data = []
        item_id = 0
        for X, y, split in zip(Xs, ys, splits):
            if isinstance(X, np.ndarray):
                for i in range(len(X)):
                    x = X[i]
                    label = y[i] if y else 0
                    item = {
                        "elementIds": list(map(int, np.repeat(arange, x))),
                        "label": label,
                        "split": split,
                        "itemId": item_id,
                    }
                    data.append(item)
                    item_id += 1
            if isinstance(X, spmatrix):
                for i in range(X.shape[0]):
                    x = X[i]
                    label = y[i] if y else 0
                    elements = []
                    for j, x_j in zip(x.indices, x.data):
                        for _ in range(x_j):
                            elements.append(int(j))
                    item = {
                        "elementIds": elements,
                        "label": label,
                        "split": split,
                        "itemId": item_id,
                    }
                    data.append(item)
                    item_id += 1
        with open(self.config["filename"], 'w') as f:
            for item in data:
                f.write(json.dumps(item))
                f.write('\n')

        subprocess.run(['java', '-cp', 'build/classes;build/jars/gson-2.8.9.jar', 'dv.cosine.java.Run'])

        vectors = deque()
        with open(self.config["vecPath"]) as f:
            for line in f:
                vectors.appendleft(json.loads(line))
        if X_dev or X_test:
            X_splits = {}
            y_splits = {}
            for split_name in splits: 
                X_splits[split_name] = []
                y_splits[split_name] = []
            while vectors:
                item = vectors.pop()
                item_split = item["split"]
                X_splits[item_split].append(item["embs"])
                y_splits[item_split].append(item["label"])
            res = []
            for split in splits:
                res.append(np.array(X_splits[split]))
                res.append(np.array(y_splits[split]))
            return res
        else:
            X_list = []
            y_list = []
            while vectors:
                item = vectors.pop()
                X_list.append(item["embs"])
                y_list.append(item["label"])
            if y_train:
                return np.array(X_list), np.array(y_list)
            else:
                return np.array(X_list)

    def set_params(self, **params):
        self.config.update(params)
        self.save_config()

    def save_config(self):
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f)
    
    def __del__(self):
        if not self.keep_files:
            paths = [self.config["filename"], self.config["vecPath"], self.config["logPath"]]
            for path in paths:
                os.remove(path)
    
