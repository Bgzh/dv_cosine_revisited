import io
import json
import numpy as np
import pandas as pd
from collections import deque


def load_log(log_path):
    '''
    read log and return as pandas DataFrame
    '''
    logs = {}
    with open(log_path) as f:
        for line in f:
            fields = line.split()
            for i in range(0, len(fields), 2):
                logs.setdefault(fields[i], []).append(float(fields[i+1]))
    return pd.DataFrame(logs)


def load_vectors(vector_path, split=True):
    '''
    read vectors and return as np ndarrays
    split is True : X_train, y_train, X_dev, y_dev, X_test, y_test
    split is False: X, y
    '''
    vectors = deque()
    with open(vector_path) as f:
        for line in f:
            vectors.appendleft(json.loads(line))
    if split:
        splits = ["train", "dev", "test"]
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
        return np.array(X_list), np.array(y_list)

