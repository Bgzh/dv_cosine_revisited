import json
import re
from collections import Counter

import os

os.chdir("IMDB_splits")

def normalize_text(text):
    '''
    preprocess a doc from the original imdb dataset
    '''
    text = re.sub(r'([\.",\(\)\!\?:;])', r' \1 ', text.lower())  # find listed punctuation marks and add a space in each side
    text = re.sub('<br />|\x85', ' ', text)  # replace non-informational tag/symbol with space (remove them)
    return text

def read_data(fp):
    items = []
    with open(fp, encoding='utf8') as f:
        for line in f:
            item = json.loads(line)
            item["text"] = get_3gram_lines(item["text"])
            item.setdefault('label', -1)
            items.append(item)
    return items
        
def get_3gram_lines(doc):
    words = normalize_text(doc).split()
    list_3gram = words.copy()
    list_3gram.extend([words[i] + "@$" + words[i+1] for i in range(len(words)-1)])
    list_3gram.extend([words[i] + "@$" + words[i+1] + "@$" + words[i+2] for i in range(len(words)-2)])
    return list_3gram
    
def main():
    print("reading files")
    train_set = read_data("train.jsonl")
    dev_set = read_data("dev.jsonl")
    test_set = read_data("test.jsonl")
    unsup_set = read_data("unlabelled.jsonl")

    print("processing")
    ngram_2_id = {}
    counter = Counter()
    for dataset in [train_set, dev_set, test_set, unsup_set]:
        for doc in dataset:
            counter.update(doc['text'])
    n_ngram = 0
    for ngram, c in counter.items():
        if c > 3:
            ngram_2_id[ngram] = n_ngram
            n_ngram += 1

    print("saving")
    with open("../files_root/imdb_data.jsonl", "w") as f:
        i = 0
        datasets = [train_set, dev_set, test_set, unsup_set]
        splits = ["train", "dev", "test", "extra"]
        for dataset, split in zip(datasets, splits):
            for doc in dataset:
                item = {}
                item["elementIds"] = [ngram_2_id[g] for g in doc["text"] if g in ngram_2_id]
                item["label"] = doc["label"]
                item["split"] = split
                item["itemId"] = i
                i += 1
                f.write(json.dumps(item))
                f.write('\n')
    

if __name__ == "__main__":
    main()
