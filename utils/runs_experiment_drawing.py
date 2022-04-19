import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns


def get_df_from_all_logs(all_logs, columns):
    temp_list = []
    for ss, logs in all_logs.items():
        for lr, df in logs.items():
            temp_df = df[columns].copy()
            temp_df.loc[:, "learning rate"] = lr
            temp_df.loc[:, "type"] = ss
            temp_list.append(temp_df)
    return pd.concat(temp_list, axis=0)

def plot_lr_dev_loss(logs:dict, keys, plot_name, savedir=None):
    '''
    plot over nepoch
    logs ~ dict{name: log as dataframe}
    keys: list of names
    '''
    palette = sns.color_palette()
    nepoch = logs[keys[0]].shape[0]
    f, ax1 = plt.subplots(1, 1, figsize=(12, 8))
    ax2 = ax1.twinx()
    lines = []
    for i, lr in enumerate(keys):
        line1, = ax1.plot(np.arange(1, nepoch+1), logs[lr]["dev_acc"], label=f"lr={lr:.0e} dev_acc", color=palette[i])
        line2, = ax2.plot(np.arange(1, nepoch+1), logs[lr]["mean_loss"], "--", label=f"lr={lr:.0e} mean_loss", color=palette[i])
        lines.extend((line1, line2))
    ax1.set_ylim((80, 94))
    #ax1.set_yticks(range(90, 94))
    ax1.grid()
    ax1.set_xlabel("number of epochs")
    ax1.set_ylabel("accuracy %")
    ax2.set_ylim((1.8, 2.4))
    ax2.set_ylabel("mean loss")
    ax1.set_title(plot_name)
    plt.xlim((1, 40))
    plt.legend(handles=lines, loc=7)
    if savedir:
        f.savefig(savedir)

def plot_vec_grad_norm(logs:dict, keys, plot_name, savedir=None):
    '''
    plot over nepoch
    logs ~ dict{name: log as dataframe}
    keys: list of names
    '''
    palette = sns.color_palette()
    nepoch = logs[keys[0]].shape[0]
    f, ax1 = plt.subplots(1, 1, figsize=(12, 8))
    ax2 = ax1.twinx()
    lines = []
    for i, lr in enumerate(keys):
        line1, = ax1.plot(np.arange(1, nepoch+1), logs[lr]["mean_dv_norm"], label=f"lr={lr:.0e} mean dv norm", color=palette[i])
        line2, = ax2.plot(np.arange(1, nepoch+1), logs[lr]["mean_grad_norm"], "--", label=f"lr={lr:.0e} mean grad norm", color=palette[i])
        lines.extend((line1, line2))
    #ax1.set_ylim((80, 94))
    #ax1.set_yticks(range(90, 94))
    ax1.set_yscale("log")
    ax1.grid()
    ax1.set_xlabel("number of epochs")
    ax1.set_ylabel("mean document vector norm")
    #ax2.set_ylim((1.8, 2.4))
    ax2.set_yscale("log")
    ax2.set_ylabel("mean gradient norm")
    ax1.set_title(plot_name)
    plt.xlim((1, 40))
    plt.legend(handles=lines)
    if savedir:
        f.savefig(savedir)

def plot_compare_dev_loss_niter(log1:pd.DataFrame, log2, name1:str, name2, plot_name, savedir=None):
    '''
    plot over nstep instead of nepoch
    log1, log2: dataframe
    name1, name2: str, names of corresponding logs
    '''
    palette = sns.color_palette()
    nsteps1 = (log1.epoch * log1.mean_ngram_count).values
    nsteps2 = (log2.epoch * log2.mean_ngram_count).values
    ul_nstep = min(nsteps1[-1], nsteps2[-1])
    f, ax1 = plt.subplots(1, 1, figsize=(12, 8))
    ax2 = ax1.twinx()
    lines = []
    for i, (nsteps, log, name) in enumerate([(nsteps1, log1, name1), (nsteps2, log2, name2)]):
        line1, = ax1.plot(nsteps, log["dev_acc"], label=f"lr={name} dev_acc", color=palette[i])
        line2, = ax2.plot(nsteps, log["mean_loss"], "--", label=f"lr={name} mean_loss", color=palette[i])
        lines.extend((line1, line2))
    ax1.set_ylim((80, 94))
    #ax1.set_yticks(range(90, 94))
    ax1.grid()
    ax1.set_xlabel("number of steps")
    ax1.set_ylabel("accuracy %")
    ax2.set_ylim((1.8, 2.4))
    ax2.set_ylabel("mean loss")
    ax1.set_title(plot_name)
    plt.xlim((1, ul_nstep))
    plt.legend(handles=lines, loc=7)
    if savedir:
        f.savefig(savedir)
