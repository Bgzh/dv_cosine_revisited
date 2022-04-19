import json
import subprocess
import os

def save_config(nDim, nb:bool, lr, randomSeed, earlyStoppingPatience, imdb_data_path, results_dir, suffix='', **kargs):
    nb_string = 'nb' if nb else 'full'
    config = {
        'filename': imdb_data_path,
        'nDim': nDim,
        'nb': nb,
        'n': 500,
        'minTf': 0,
        'lr': lr,
        'nEpoch': 40,
        'subSamp': nb,
        'nbA': 2,
        'nbB': 3,
        'randomSeed': randomSeed,
        'earlyStoppingPatience': earlyStoppingPatience,
        'vecPath': f'{results_dir}/imdb_vectors_{nb_string}_{lr:.0e}{suffix}.jsonl',
        'logPath': f'{results_dir}/imdb_log_{nb_string}_{lr:.0e}{suffix}.txt',
        'test': True,
        'Cs': [0.01, 0.1, 1, 10, 20, 100, 1000],
        'verbose': 0
    }
    config.update(kargs)
    with open('config.json', 'w') as f:
        json.dump(config, f)
    return


if __name__=="__main__":
    imdb_data_path = "files_root/imdb_data.jsonl"
    results_dir = "imdb_trainsize_experiment"
    max_ind = 0
    with open(imdb_data_path) as f:
        for line in f:
            item = json.loads(line)
            max_ind = max(max_ind, max(item["elementIds"]))
    nDim = max_ind + 1
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
    subprocess.run(['javac', '-cp', 'dvscript;build/jars/gson-2.8.9.jar', '-d', 'build/classes', 'dvscript/dv/cosine/java/Run.java'])

    random_seed = 22
    use_nb = False
    nepoch = 120
    save_config(nDim, use_nb, 1e-3, random_seed, 30, imdb_data_path, results_dir, suffix="_120epoch_p30", nEpoch=nepoch)
    subprocess.run(['java', '-cp', 'build/classes;build/jars/gson-2.8.9.jar', 'dv.cosine.java.Run'])

    

    
    