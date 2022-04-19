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
    n_epochs=10
    batch_size=2
    update_freq = 16
    
    os.chdir("roberta")
    data_dir = "../IMDB_splits"
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained('roberta-base')

    trainset = read_jsonl(os.path.join(data_dir, "train.jsonl"))
    devset = read_jsonl(os.path.join(data_dir, "dev.jsonl"))
    testset = read_jsonl(os.path.join(data_dir, "test.jsonl"))

    max_length = 512
    train_set = IMDBDataset(trainset, tokenizer, max_length)
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
    #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.774, verbose=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 9, 1e-6, verbose=True)
    # finetune
    train_loss_his = []
    val_acc_his = []
    for epoch in range(n_epochs):
        model.train()
        train_loss_list = []
        i = 0
        optimizer.zero_grad()
        for X_train, mask_train, targets_train in tqdm(train_loader):
            i += 1
            
            X_train = X_train.to('cuda', non_blocking=True)
            mask_train = mask_train.to('cuda', non_blocking=True)
            targets_train = targets_train.to('cuda', non_blocking=True)
            
            logits = model(X_train, mask_train).logits
            loss = F.cross_entropy(logits, targets_train)
            loss.backward()
            train_loss_list.append(loss.item())
            if i % update_freq == 0:
                optimizer.step()
                optimizer.zero_grad()
        
        n_correct = 0
        model.eval()
        with torch.no_grad():
            for X_val, mask_val, targets_val in tqdm(dev_loader):
                X_val = X_val.to('cuda', non_blocking=True)
                mask_val = mask_val.to('cuda', non_blocking=True)
                targets_val = targets_val.to('cuda', non_blocking=True)
                
                logits_val = model(X_val, mask_val).logits
                n_correct += ((logits_val[:, 1] > logits_val[:, 0]).type(torch.int32) == targets_val).sum().item()
                
        tl = sum(train_loss_list) / len(train_loss_list)
        train_loss_his.append(tl)
        val_acc = n_correct / len(dev_set)
        val_acc_his.append(val_acc)
        
        print('epoch', epoch, ' | ', 'train loss', tl, ' | ', 'val acc', val_acc)
        
        scheduler.step()

    torch.save(model.state_dict(), 'cp/ms512.pt')



    # extract vectors
    model.load_state_dict(torch.load('cp/ms512.pt'))
    train_loader = DataLoader(train_set, batch_size=batch_size*2, shuffle=False)
    dev_loader = DataLoader(dev_set, batch_size=batch_size*2, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size*2, shuffle=False)

    emb_list = []
    model.eval()
    for X, mask, _ in tqdm(train_loader):
        X = X.to('cuda', non_blocking=True)
        mask = mask.to('cuda', non_blocking=True)
        
        with torch.no_grad():
            emb = torch.tanh(model.classifier.dense(model.roberta(X, mask)[0][:, 0, :]))
            emb_list.append(emb.cpu().numpy())
    np.save(f'embs/emb_train.npy', np.concatenate(emb_list))

    emb_list = []
    model.eval()
    for X, mask, _ in tqdm(dev_loader):
        X = X.to('cuda', non_blocking=True)
        mask = mask.to('cuda', non_blocking=True)
        
        with torch.no_grad():
            emb = torch.tanh(model.classifier.dense(model.roberta(X, mask)[0][:, 0, :]))
            emb_list.append(emb.cpu().numpy())
    np.save(f'embs/emb_dev.npy', np.concatenate(emb_list))

    emb_list = []
    model.eval()
    for X, mask, _ in tqdm(test_loader):
        X = X.to('cuda', non_blocking=True)
        mask = mask.to('cuda', non_blocking=True)
        
        with torch.no_grad():
            emb = torch.tanh(model.classifier.dense(model.roberta(X, mask)[0][:, 0, :]))
            emb_list.append(emb.cpu().numpy())
    np.save(f'embs/emb_test.npy', np.concatenate(emb_list))
