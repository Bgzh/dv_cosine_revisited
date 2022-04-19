import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaModel
from torch.utils.data import Dataset, DataLoader
from torch import optim
import torch.nn.functional as F
import numpy as np

import os
import json
from tqdm import tqdm


def read_jsonl(path):
    ids = []
    texts = []
    labels = []
    with open(path, encoding='utf8') as f:
        for line in f:
            d = json.loads(line)
            ids.append(d['id'])
            texts.append(d['text'])
            labels.append(d['label'])
    return ids, texts, labels

class IMDBDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length):
        '''
        dataset: list ~ [ids, texts, labels]
        ids ~ [train_021, ...]
        texts ~ ['...', ...]
        labels ~ [0, 1, 1, ...]
        '''
        self.tokenizer = tokenizer
        self.ids = dataset[0]
        self.docs = dataset[1]
        self.targets = dataset[2]
        self.max_length = max_length

    def __len__(self):
        return len(self.docs)
    
    def __getitem__(self, i):
        doc = self.docs[i]
        r = self.tokenizer.encode_plus(doc, return_tensors="pt", truncation=False, padding='max_length', max_length=self.max_length)
        tids = r['input_ids'].squeeze()
        mask = r['attention_mask'].squeeze()
        if len(tids) > self.max_length:
            tids = tids[-self.max_length:]
            mask = mask[-self.max_length:]
            tids[0] = 0
            mask[0] = 0
        target = self.targets[i]
        return tids, mask, target


if __name__=="__main__":
    train_size_list = [20, 200, 2000, 20000, 60, 30, 10]
    repeat_times_list = [10, 10, 10, 1, 10, 10, 10]
    nepochs_per_epoch_list = [50, 5, 1, 1, 20, 35, 100] # number of repeats through the training set per "epoch" 
    result_fn = "imdb_trainsize_roberta_results.json" # to save the results of this experiment
    sampled_inds_fn = "imdb_trainsize_sampled_inds.json"

    results_dir = "imdb_trainsize_experiment"
    if os.path.isfile(os.path.join(results_dir, result_fn)):
        with open(os.path.join(results_dir, result_fn)) as f:
                    res = json.load(f)
    else:
        res = {
        "id": [], "train_size": [], "train_acc": [], "dev_acc": [], "test_acc": []
        }

    n_epochs=10
    patience=3
    batch_size=2
    update_freq = 16
    
    #os.chdir("roberta")
    data_dir = "IMDB_splits"
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    trainset = read_jsonl(os.path.join(data_dir, "train.jsonl"))
    devset = read_jsonl(os.path.join(data_dir, "dev.jsonl"))
    testset = read_jsonl(os.path.join(data_dir, "test.jsonl"))

    # load sampled inds
    with open(os.path.join(results_dir, sampled_inds_fn)) as f:
        sampled_inds_dict = json.load(f)

    test_id = 0
    for train_size, repeat_times, nepochs_per_epoch in zip(train_size_list, repeat_times_list, nepochs_per_epoch_list):
        for i in range(repeat_times):

            train_inds = sampled_inds_dict[str(test_id)]
            assert len(train_inds) == train_size
            trainset_sampled = [[ls[si] for si in train_inds] for ls in trainset]

            model = RobertaForSequenceClassification.from_pretrained('roberta-base')

            max_length = 512
            train_set = IMDBDataset(trainset_sampled, tokenizer, max_length)
            dev_set = IMDBDataset(devset, tokenizer, max_length)
            test_set = IMDBDataset(testset, tokenizer, max_length)

            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
            dev_loader = DataLoader(dev_set, batch_size=batch_size*2, shuffle=True)
            test_loader = DataLoader(test_set, batch_size=batch_size*2, shuffle=True)

            model.roberta.requires_grad_(True)
            model.classifier.requires_grad_(True)
            model.to('cuda');
            optimizer = optim.Adam(
            params=model.parameters(),
                lr=1e-5,
                betas=(0.9, 0.98),
                eps=1e-6
                )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 9, 1e-6, verbose=True)
            # finetune
            val_acc_his = []
            train_acc_his = []
            for epoch in range(n_epochs):
                i = 0
                optimizer.zero_grad()
                n_batches = nepochs_per_epoch * len(train_set) // batch_size # the number of batches in this "epoch"
                # here assume the training set size is always divisable by the batch_size
                model.train()
                train_n_correct = 0
                pbar = tqdm(total=n_batches)
                for _ in range(nepochs_per_epoch):
                    for X_train, mask_train, targets_train in train_loader:
                        i += 1
                        
                        X_train = X_train.to('cuda', non_blocking=True)
                        mask_train = mask_train.to('cuda', non_blocking=True)
                        targets_train = targets_train.to('cuda', non_blocking=True)
                        
                        logits = model(X_train, mask_train).logits
                        loss = F.cross_entropy(logits, targets_train)
                        loss.backward()
                        with torch.no_grad():
                            train_n_correct += ((logits[:, 1] > logits[:, 0]).type(torch.int32) == targets_train).sum().item()
                        if i % update_freq == 0 or i == n_batches:
                            optimizer.step()
                            optimizer.zero_grad()
                            # print("ss", train_n_correct / batch_size / i)
                        pbar.update()
                pbar.close()
                #print("train acc:", train_n_correct / nepochs_per_epoch / len(train_set))
                
                n_correct = 0
                model.eval()
                with torch.no_grad():
                    for X_val, mask_val, targets_val in tqdm(dev_loader):
                        X_val = X_val.to('cuda', non_blocking=True)
                        mask_val = mask_val.to('cuda', non_blocking=True)
                        targets_val = targets_val.to('cuda', non_blocking=True)
                        
                        logits_val = model(X_val, mask_val).logits
                        n_correct += ((logits_val[:, 1] > logits_val[:, 0]).type(torch.int32) == targets_val).sum().item()
                        
                train_acc = train_n_correct / nepochs_per_epoch / len(train_set)
                val_acc = n_correct / len(dev_set)
                if not val_acc_his or val_acc > max(val_acc_his):
                    torch.save(model.state_dict(), 'roberta/cp/temp.pt')
                val_acc_his.append(val_acc)
                train_acc_his.append(train_acc)
                
                print('epoch', epoch, ' | ', 'train acc', train_acc, ' | ', 'val acc', val_acc)
                scheduler.step()
                # early stop
                if len(val_acc_his) - max(range(len(val_acc_his)), key=lambda x: val_acc_his[x]) > 3:
                    break
            
            # test
            model.load_state_dict(torch.load('roberta/cp/temp.pt'))
            n_correct = 0
            model.eval()
            with torch.no_grad():
                for X_test, mask_test, targets_test in tqdm(test_loader):
                    X_test = X_test.to('cuda', non_blocking=True)
                    mask_test = mask_test.to('cuda', non_blocking=True)
                    targets_test = targets_test.to('cuda', non_blocking=True)
                    
                    logits_test = model(X_test, mask_test).logits
                    n_correct += ((logits_test[:, 1] > logits_test[:, 0]).type(torch.int32) == targets_test).sum().item()
                    
            test_acc = n_correct / len(test_set)
            best_epoch_ind = max(range(len(val_acc_his)), key=lambda x: val_acc_his[x])
            # save results
            if test_id < len(res['id']):
                res['id'][test_id] = test_id
                res['train_size'][test_id] = len(train_inds)
                res['train_acc'][test_id] = train_acc_his[best_epoch_ind]
                res['dev_acc'][test_id] = max(val_acc_his)
                res['test_acc'][test_id] = test_acc
            else:
                res['id'].append(test_id)
                res['train_size'].append(len(train_inds))
                res['train_acc'].append(train_acc_his[best_epoch_ind])
                res['dev_acc'].append(max(val_acc_his))
                res['test_acc'].append(test_acc)

            with open(os.path.join(results_dir, result_fn), 'w') as f:
                json.dump(res, f)

            print(f"test_id: {test_id}, train_size: {len(train_inds)} finished, train_acc: {train_acc_his[best_epoch_ind]:.4f}, dev_acc: {max(val_acc_his):.4f}, test_acc: {test_acc:.4f}")
            del model
            
            test_id += 1
            

    
    


        



    