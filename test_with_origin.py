import numpy as np
import re

import subprocess
from os.path import join
import os
from shutil import copyfile
from typing import List
from datetime import datetime
from tqdm import tqdm




def read_file(file_name:str, start_at:int, stop_before:int)->List[str]:
    '''
    read a part of the file: lines[start_at: stop_before]
    return the cleaned documents, for the purpose of restoring the order
    '''
    docs = []
    with open(file_name, encoding='utf8') as f:
        for i, line in enumerate(f):
            if i < start_at:
                continue
            if i == stop_before:
                break
            line = line[line.find(' ') + 1:]  # remove the index
            # in the 3gram file, "@$" is used as a word-separator inside tokens, 
            # so here remove every token with this separator (along with very few 
            # 1grams with "@$" to avoid interference)
            line = ''.join([t for t in line.lower().split(' ') if "@$" not in t])
            # due to different preprocessing for the 2 files, here only retain 
            # the alphabetics/'common symbols' in order to ignore noises and 
            # achieve exact matching (they are enough to retrieve the order). 
            line = re.sub(r"[^a-z,\.\?\!]", '', line)
            docs.append(line)     
    return docs

def restore_order(docs_1gram:List[str], docs_3gram:List[str], check=True)->np.ndarray:
    '''
    docs_1gram: docs returned by "read_file", read from the 1gram file
    docs_3gram: docs returned by "read_file", read from the 3gram file
    returns the matched order, with counting sort:
            argsort(docs_3gram, order=docs_1gram)
    '''
    n = len(docs_1gram)
    tri_dict = {}
    for i, doc in enumerate(docs_3gram):
        # repetitions exist in the dataset, so here use lists to deal with them
        tri_dict.setdefault(doc, []).append(i)

    uni2tri = []
    for i, doc in enumerate(docs_1gram):
        uni2tri.append(tri_dict[doc].pop())

    uni2tri = np.array(uni2tri)
    if check:
        assert len(np.unique(uni2tri)) == len(uni2tri)  # no repetitions (valid permutation)
        assert sum(uni2tri[:n//2] >= n//2) == 0  # positive remain positive
        assert sum(uni2tri[n//2: n] < n//2) == 0  # negative remain negative
    return uni2tri

def read_write_embeddings(filename:str, write_filename:str, order:np.ndarray):
    '''
    reads embeddings from file and write (after potential shuffling) in 
        a new file for testing
    '''
    with open(filename) as f:
        embeds = f.readlines()
    with open(write_filename, 'w') as f:
        for i in order:
            f.write(embeds[i])

def get_shuffled_inds(inds, seed, inblock=True):
    '''
    makes permutations of the indices of embeddings
    the embedding has 25000 rows, in which the first 12500 rows are positive, 
        while the second 12500 rows are negative. With inblock=True, the permutation
        will respect the class blocks, otherwise not.
    returns the permutated indices
    '''
    np.random.seed(seed)
    inds = inds.copy()
    if inblock:
        p1 = np.random.permutation(12500)
        p2 = np.random.permutation(12500)
        inds[:12500] = inds[p1]
        inds[12500:] = inds[p2 + 12500]
    else:
        inds = np.random.permutation(inds)
    return inds


if __name__=="__main__":
    repeat_times = 30  # times to repeat shuffling test 
    files_root = 'files_root'
    filename_1gram = "alldata-id_p1gram.txt"
    filename_3gram = "alldata-id_p3gram.txt"
    filename_embedding_train = "train_vectors.txt"
    filename_embedding_test = "test_vectors.txt"

    # create the log dir and prepare the filenames for log and report
    log_root = 'test_logs'
    if not os.path.isdir(log_root):
        os.mkdir(log_root)
    log_file = join(log_root, 'log.txt')
    report_file = join(log_root, 'report.txt')

    results = {}  # results of the tests

    print('retrieving order')
    # train set contains the first 25000 items in both files
    docs_1gram = read_file(join(files_root, filename_1gram), 0, 25000)
    docs_3gram = read_file(join(files_root, filename_3gram), 0, 25000)
    train_order = restore_order(docs_1gram, docs_3gram)

    # test set contains the second 25000 items in both files
    docs_1gram = read_file(join(files_root, filename_1gram), 25000, 50000)
    docs_3gram = read_file(join(files_root, filename_3gram), 25000, 50000)
    test_order = restore_order(docs_1gram, docs_3gram)

    # copy 1gram file from files_root to the root
    copyfile(join(files_root, filename_1gram), filename_1gram)

    print('testing with the original order')
    # read the pretrained embeddings and save them in a particular order in a new file
    read_write_embeddings(join(files_root, filename_embedding_train), 
        filename_embedding_train, np.arange(25000))
    read_write_embeddings(join(files_root, filename_embedding_test), 
        filename_embedding_test, np.arange(25000))
    with open(log_file, 'a') as f:
        f.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
        f.write(" original order\n")
        cp = subprocess.run(['python', 'ensemble.py'], capture_output=True, encoding='ascii')  # run ensemble.py on the prepared files
        res = float(cp.stdout.split()[-1])  # catch the test score from stdout of ensemble.py
        f.write(cp.stdout)  # log the stdout
        f.write("#####################")
    results['original'] = res

    print('testing with the correct order')
    read_write_embeddings(join(files_root, filename_embedding_train), 
        filename_embedding_train, train_order)
    read_write_embeddings(join(files_root, filename_embedding_test), 
        filename_embedding_test, test_order)
    with open(log_file, 'a') as f:
        f.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
        f.write(" correct order\n")
        cp = subprocess.run(['python', 'ensemble.py'], capture_output=True, encoding='ascii')
        res = float(cp.stdout.split()[-1])
        f.write(cp.stdout)
        f.write("#####################")
    results['correct'] = res

    print('testing with shuffled test set (within class)')
    results['shuffle_test_inclass'] = []
    for i in tqdm(range(repeat_times)):
        read_write_embeddings(join(files_root, filename_embedding_train), 
            filename_embedding_train, train_order)
        read_write_embeddings(join(files_root, filename_embedding_test), 
            filename_embedding_test, get_shuffled_inds(test_order, i+11, True))

        with open(log_file, 'a') as f:
            f.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
            f.write(f" shuffle test set within class repeat{i}\n")
            cp = subprocess.run(['python', 'ensemble.py'], capture_output=True, encoding='ascii')
            res = float(cp.stdout.split()[-1])
            f.write(cp.stdout)
            f.write("#####################")
        results['shuffle_test_inclass'].append(res)

    print('testing with shuffled test set (whole)')
    results['shuffle_test_whole'] = []
    for i in tqdm(range(repeat_times)):
        read_write_embeddings(join(files_root, filename_embedding_train), 
            filename_embedding_train, train_order)
        read_write_embeddings(join(files_root, filename_embedding_test), 
            filename_embedding_test, get_shuffled_inds(test_order, i+11, False))

        with open(log_file, 'a') as f:
            f.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
            f.write(f" shuffle whole test set repeat{i}\n")
            cp = subprocess.run(['python', 'ensemble.py'], capture_output=True, encoding='ascii')
            res = float(cp.stdout.split()[-1])
            f.write(cp.stdout)
            f.write("#####################")
        results['shuffle_test_whole'].append(res)

    print('testing with shuffled train and test sets (inclass)')
    results['shuffle_train_test_inclass'] = []
    for i in tqdm(range(repeat_times)):
        read_write_embeddings(join(files_root, filename_embedding_train), 
            filename_embedding_train, get_shuffled_inds(train_order, i+111, True))
        read_write_embeddings(join(files_root, filename_embedding_test), 
            filename_embedding_test, get_shuffled_inds(test_order, i+11, True))

        with open(log_file, 'a') as f:
            f.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
            f.write(f" shuffle train and test sets in class repeat{i}\n")
            cp = subprocess.run(['python', 'ensemble.py'], capture_output=True, encoding='ascii')
            res = float(cp.stdout.split()[-1])
            f.write(cp.stdout)
            f.write("#####################")
        results['shuffle_train_test_inclass'].append(res)

    print('testing with shuffled train and test sets (whole)')
    results['shuffle_train_test_whole'] = []
    for i in tqdm(range(repeat_times)):
        read_write_embeddings(join(files_root, filename_embedding_train), 
            filename_embedding_train, get_shuffled_inds(train_order, i+111, False))
        read_write_embeddings(join(files_root, filename_embedding_test), 
            filename_embedding_test, get_shuffled_inds(test_order, i+11, False))

        with open(log_file, 'a') as f:
            f.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
            f.write(f" shuffle whole train and test sets repeat{i}\n")
            cp = subprocess.run(['python', 'ensemble.py'], capture_output=True, encoding='ascii')
            res = float(cp.stdout.split()[-1])
            f.write(cp.stdout)
            f.write("#####################")
        results['shuffle_train_test_whole'].append(res)

    # save the summary of results as report
    with open(report_file, 'w') as f:
        f.write(f"original score: {results['original']}\n")
        f.write(f"correct score: {results['correct']}\n")

        res = np.array(results['shuffle_test_inclass'])
        f.write(f"shuffle test set in class: mean {res.mean():.2f}, std {res.std():.2f}\n")

        res = np.array(results['shuffle_test_whole'])
        f.write(f"shuffle whole test set: mean {res.mean():.2f}, std {res.std():.2f}\n")

        res = np.array(results['shuffle_train_test_inclass'])
        f.write(f"shuffle train and test sets in class: mean {res.mean():.2f}, std {res.std():.2f}\n")

        res = np.array(results['shuffle_train_test_whole'])
        f.write(f"shuffle whole train and test sets: mean {res.mean():.2f}, std {res.std():.2f}\n")
    print(f'saved report to {report_file}')
    
    os.remove(filename_1gram)
    os.remove(filename_embedding_train)
    os.remove(filename_embedding_test)
    print('finished')

        

    



    


