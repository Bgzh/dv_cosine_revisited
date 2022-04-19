import numpy as np
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix, spmatrix
import pandas as pd
import os

import pickle
from test_with_origin import read_file, restore_order


class ImdbSentimentClf:
    def __init__(self, **params):
        self.params = params
        self.lr = linear_model.LogisticRegression(tol=0.001)
        self.dv_scaler = params['dv_scaler']
        self.bon_scaler = params['bon_scaler']
        self.rec = []
        self.best_param = None

    def fit_data(self, dv, bon):
        self.dv_scaler.fit(dv)
        self.bon_scaler.fit(bon)
        return
    
    def transform_data(self, dv, bon):
        dv = self.dv_scaler.transform(dv)
        bon = self.bon_scaler.transform(bon)
        return dv, bon
    
    def fit_gridsearchcv(self, dv, bon, y):
        self.fit_data(dv, bon)
        dv, bon = self.transform_data(dv, bon)
        param_grid = {'C': self.params['C']}
        cv = self.params.get('cv', 5)
        for r in self.params['r']:
            X = self.hstack_([dv * r, bon])
            gscv = GridSearchCV(self.lr, param_grid, cv=cv, 
                scoring='accuracy', n_jobs=4, refit=False, 
                verbose=4)
            gscv.fit(X, y)
            result = gscv.cv_results_
            for param in result['params']:
                param['r'] = r
            self.rec.append(result)
        best_score = 0
        for record in self.rec:
            best_local  = np.max(record['mean_test_score'])
            if best_local > best_score:
                best_score = best_local
                self.best_param = record['params'][np.argmax(record['mean_test_score'])]
        self.lr.set_params(C=self.best_param['C'])
        r = self.best_param['r']
        X = self.hstack_([dv * r, bon])
        self.lr.fit(X, y)
        return best_score, self.best_param

    def predict(self, dv, bon):
        dv, bon = self.transform_data(dv, bon)
        X = self.hstack_([dv * self.best_param['r'], bon])
        return self.lr.predict(X)

    def gridsearch(self, dv_train, bon_train, y_train, dv_val, bon_val, y_val):
        self.fit_data(dv_train, bon_train)
        dv_train, bon_train = self.transform_data(dv_train, bon_train)
        dv_val, bon_val = self.transform_data(dv_val, bon_val)
        #param_grid = {'C': self.params['C']}
        #cv = self.params.get('cv', 5)
        best_score = 0
        best_param = None
        for r in self.params['r']:
            X_train = self.hstack_([dv_train * r, bon_train])
            X_val = self.hstack_([dv_val * r, bon_val])
            for c in self.params["C"]:
                self.lr.set_params(C=c)
                self.lr.fit(X_train, y_train)
                score = self.lr.score(X_val, y_val)
                h = {
                    "C": c,
                    "r": r,
                    "acc": score
                }
                self.rec.append(h)
                if score > best_score:
                    best_score = score
                    best_param = h
                    #print('new best:', best_param)
        return best_param
    
    @staticmethod
    def hstack_(vs):
        if any(map(lambda x: isinstance(x, spmatrix), vs)):
            return hstack(vs)
        else:
            return np.hstack(vs)


class TrivialScaler:
    def __init__(self):
        pass

    def fit(self, X):
        pass

    def transform(self, X):
        return X


class VectorRescaler:
    def __init__(self, feature_scaler, non_zero=False):
        self.feature_scaler = feature_scaler
        self.non_zero = non_zero
    
    def fit(self, X):
        self.feature_scaler.fit(X)
    
    def transform(self, X):
        X = self.feature_scaler.transform(X)
        if isinstance(X, spmatrix):
            if self.non_zero:
                n = np.array((X != 0).sum(axis=1))
            else:
                n = X.shape[1]
            return X.multiply(1/np.sqrt(n))
        else:
            return X / np.sqrt(X.shape[1])

def read_logs(his):
    '''
    read the log file of gridsearchk (no CV)
    '''
    df_list = []
    for k, log in his.items():
        dv_scaling, bon_scaling = k.split("+")
        dv_scaling = dv_scaling[dv_scaling.find('_')+1:]
        bon_scaling = bon_scaling[bon_scaling.find('_')+1:]
        df = pd.DataFrame.from_records(log)
        df["DV scaling"] = dv_scaling
        df["BON scaling"] = bon_scaling
        df_list.append(df)
    return pd.concat(df_list)

def parse_logs(logs):
    '''
    read the log file of gridsearchCV
    '''
    df_list = []
    for k, log in logs.items():
        dv_scaling, bon_scaling = k.split('+')
        results = log["cv_results"]
        for result in results:
            Cs = result["param_C"]
            rs = [d['r'] for d in result["params"]]
            df = pd.DataFrame({"dv_scaling": dv_scaling, "bon_scaling": bon_scaling, "C": Cs, "r": rs, "mean_acc": result["mean_test_score"], "std_acc": result["std_test_score"]})
            df_list.append(df)
    res = pd.concat(df_list, ignore_index=True)
    res["dv_scaling"] = res.dv_scaling.str.extract(r"\w*?_([a-z]*)")
    res["bon_scaling"] = res.bon_scaling.str.extract(r"\w*?_([a-z]*)")
    return res
    
def read_embeddings(filename:str):
    '''
    reads and returns the embeddings from file
    '''
    print(f"reading embeddings from {filename}")
    X=[]
    with open(filename) as file:
        for line in file:
            vector = line.split()[1:]
            X.append(vector)
    return np.array(X, dtype=np.float64)
    
def read_1gram(filename:str):
    '''
    filename: name of the 1gram file
    '''
    train_docs = []
    test_docs = []
    y_train = []
    y_test = []

    with open(filename, encoding='utf-8') as alldata:
        for line_no, line in enumerate(alldata):
            if line_no == 50000:  # only need train/test sets
                break
            tokens = line.split()
            words = tokens[1:]
            split = ['train','test'][line_no//25000]  # 25k train, 25k test, 50k extra (unlabeled)
            sentiment = [1.0, 0.0, 1.0, 0.0][line_no//12500] # [12.5K pos, 12.5K neg]*2 then unknown
            if split == 'train':
                train_docs.append(words)
                y_train.append(sentiment)
            elif split == 'test':
                test_docs.append(words)
                y_test.append(sentiment)

    return train_docs, test_docs, y_train, y_test

def read_embeddings(filename:str):
    '''
    reads and returns the embeddings from file
    '''
    print(f"reading embeddings from {filename}")
    X=[]
    with open(filename) as file:
        for line in file:
            vector = line.split()[1:]
            X.append(vector)
    return np.array(X, dtype=np.float64)

def get_vectors_for_ensemble(path_1gram, path_3gram, path_embedding_train, path_embedding_test, train_val_test=False, shuffle=True):
    '''
    if train_val_test:
        return bon_train, bon_val, bon_test, embedding_train, embedding_val, embedding_test, y_train, y_val, y_test
        always shuffled
    else:
        return bon_train, bon_test, embedding_train, embedding_test, y_train, y_test
        shuffled by default, could be turned off by shuffle=False
    '''
    docs_1gram = read_file(path_1gram, 0, 25000)
    docs_3gram = read_file(path_3gram, 0, 25000)
    train_order = restore_order(docs_1gram, docs_3gram)

    docs_1gram = read_file(path_1gram, 25000, 50000)
    docs_3gram = read_file(path_3gram, 25000, 50000)
    test_order = restore_order(docs_1gram, docs_3gram)

    embedding_train = read_embeddings(path_embedding_train)
    embedding_test = read_embeddings(path_embedding_test)

    embedding_train = embedding_train[train_order]
    embedding_test = embedding_test[test_order]

    train_docs, test_docs, y_train, y_test = read_1gram(path_1gram)

    if train_val_test:
        train_docs, val_docs, embedding_train, embedding_val, y_train, y_val = train_test_split(train_docs, embedding_train, y_train, 
                                                                        test_size=0.2, random_state=2, stratify=y_train, shuffle=True)
        count_vect = CountVectorizer(tokenizer=lambda text: text ,preprocessor=lambda text:text, binary=True,ngram_range=(1,3))
        bon_train = count_vect.fit_transform(train_docs)
        bon_val = count_vect.transform(val_docs)
        bon_test = count_vect.transform(test_docs)

        nb = BernoulliNB()
        nb.fit(bon_train, y_train)
        prob=nb.feature_log_prob_
        r=np.abs(prob[0]-prob[1])
        bon_train = bon_train.multiply(r).tocsr()
        bon_val = bon_val.multiply(r).tocsr()
        bon_test = bon_test.multiply(r).tocsr()
        return bon_train, bon_val, bon_test, embedding_train, embedding_val, embedding_test, y_train, y_val, y_test
    else:
        count_vect = CountVectorizer(tokenizer=lambda text: text ,preprocessor=lambda text:text, binary=True,ngram_range=(1,3))
        bon_train = count_vect.fit_transform(train_docs)
        bon_test = count_vect.transform(test_docs)

        nb = BernoulliNB()
        nb.fit(bon_train, y_train)
        prob=nb.feature_log_prob_
        r=np.abs(prob[0]-prob[1])
        bon_train = bon_train.multiply(r).tocsr()
        bon_test = bon_test.multiply(r).tocsr()
        if shuffle:
            rng = np.random.default_rng(seed=2)
            train_inds = rng.permutation(bon_train.shape[0])
            test_inds = rng.permutation(bon_test.shape[0])
            y_train = np.array(y_train)
            y_test = np.array(y_test)
            return bon_train[train_inds], bon_test[test_inds], embedding_train[train_inds], embedding_test[test_inds], y_train[train_inds], y_test[test_inds]
        return bon_train, bon_test, embedding_train, embedding_test, y_train, y_test


if __name__=="__main__":
    os.chdir("files_root")

    path_1gram = 'alldata-id_p1gram.txt'
    path_3gram = "alldata-id_p3gram.txt"
    path_embedding_train = 'train_vectors.txt'
    path_embedding_test = 'test_vectors.txt'

    bon_train, bon_val, bon_test, embedding_train, embedding_val, embedding_test, y_train, y_val, y_test = get_vectors_for_ensemble(
                            path_1gram, path_3gram, path_embedding_train, path_embedding_test, train_val_test=True)
    

    dv_scalers = {
        'none': TrivialScaler(), 
        #'standard': StandardScaler(with_mean=False), 
        'standard_with_mean': StandardScaler(with_mean=True), 
    }
    bon_scalers = {
        'none': TrivialScaler(), 
        'standard': StandardScaler(with_mean=False), 
        #'minmax': MaxAbsScaler(), 
    }
    if not os.path.isdir("../test_logs"): os.mkdir("../test_logs")

    print("testing")
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.simplefilter("ignore", category=ConvergenceWarning)
    np.random.seed(12)
    his = {}
    rs = np.logspace(-4, 4, 17)
    Cs = np.logspace(-4, 4, 17)
    for dv_scaler_name, dv_scaler in dv_scalers.items():
        for bon_scaler_name, bon_scaler in bon_scalers.items():
            stclf = ImdbSentimentClf(r=rs, C=Cs, dv_scaler=dv_scaler, bon_scaler=bon_scaler)
            best = stclf.gridsearch(embedding_train, bon_train, y_train, embedding_val, bon_val, y_val)
            test_name = f'dv_{dv_scaler_name}+bon_{bon_scaler_name}'
            his[test_name] = stclf.rec
            with open('../test_logs/dv+bon_logs.pkl', 'wb') as f:
                pickle.dump(his, f)
            print(test_name, best)
    print("done")





