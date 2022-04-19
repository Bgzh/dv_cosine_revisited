import numpy as np
import re
import glob
import os
from download import download
import tarfile


def normalize_text(text):
    '''
    preprocess a doc from the original imdb dataset
    '''
    text = re.sub(r'([\.",\(\)\!\?:;])', r' \1 ', text.lower())  # find listed punctuation marks and add a space in each side
    text = re.sub('<br />|\x85', ' ', text)  # replace non-informational tag/symbol with space (remove them)
    return text

if __name__ == "__main__":
    # download the original dataset
    if not os.path.isdir('aclImdb'):
        print('downloading the original dataset')
        download('https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz', '', 'aclImdb_v1.tar.gz')
        print('extracting')
        tarfile.open('aclImdb_v1.tar.gz').extractall()
        os.remove('aclImdb_v1.tar.gz')

    # read 1gram file
    print('reading')
    unigram = []  # list of docs from 1gram file
    with open('files_root/alldata-id_p1gram.txt', encoding='utf8') as f:
        for i, line in enumerate(f):
            line = line[line.find(' ') + 1:].rstrip('\n')  # in each line, remove the sequence number (first token) and the trailing '\n'
            unigram.append(line)

    # read original dataset
    original = []  # list of docs from the original dataset
    for fname in sorted(glob.glob('aclImdb/*/*/*.txt')):
        with open(fname, encoding='utf8') as f:
            original.append(f.read())

    # align the order of the original dataset to the 1gram file
    # the natural order of the original dataset:
    #   test -> train
    #   neg -> pos -> unsup
    # the order of the 1gram file:
    #   train (sup) -> test -> train (unsup)
    #   pos -> neg
    inds = np.arange(100000)
    inds[:12500] = np.arange(37500, 50000)
    inds[12500:25000] = np.arange(25000, 37500)
    inds[25000:37500] = np.arange(12500, 25000)
    inds[37500:50000] = np.arange(12500)

    # check if the preprocessing method "normalize_text" was indeed used to make the 1gram file
    print('checking')
    for i in range(100000):
        # assert the preprocessed doc from original dataset is the same as the doc from the 1gram file
        assert normalize_text(original[inds[i]]) == unigram[i], i
    print('matched')
