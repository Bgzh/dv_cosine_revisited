import json
import glob
from tqdm import tqdm
import glob
import os
import tarfile
import requests

def download(url: str, dest_folder: str, filename:str):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)  # create folder if it does not exist

    file_path = os.path.join(dest_folder, filename)
    r = requests.get(url, stream=True)
    if r.ok:
        print("saving to", os.path.abspath(file_path))
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024 * 8):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
    else:  # HTTP status code 4XX/5XX
        print("Download failed: status code {}\n{}".format(r.status_code, r.text))

print("downloading splits")
download('https://s3-us-west-2.amazonaws.com/allennlp/dont_stop_pretraining/data/imdb/train.jsonl', 'IMDB_splits', 'train.jsonl')
download('https://s3-us-west-2.amazonaws.com/allennlp/dont_stop_pretraining/data/imdb/dev.jsonl', 'IMDB_splits', 'dev.jsonl')
download('https://s3-us-west-2.amazonaws.com/allennlp/dont_stop_pretraining/data/imdb/test.jsonl', 'IMDB_splits', 'test.jsonl')

if not os.path.isdir('aclImdb'):
    print('downloading the original IMDB dataset')
    download('https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz', '', 'aclImdb_v1.tar.gz')
    print('extracting')
    tarfile.open('aclImdb_v1.tar.gz').extractall()
    os.remove('aclImdb_v1.tar.gz')

print("collecting unlabelled data")
jlist = []
for i, fp in tqdm(enumerate(list(sorted(glob.glob("aclImdb/train/unsup/*.txt"))))):
    with open(fp, encoding="utf8") as f:
        text = f.read()
        jlist.append(json.dumps({"id": f"unsup_{i}", "text": text}))

with open("IMDB_splits/unlabelled.jsonl", "w", encoding="utf8") as f:
    for jline in jlist:
        f.write(jline)
        f.write('\n')


