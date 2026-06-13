import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


def plot_sample_recons_combined(dgd, train_loader, sample_index, epoch):
    x_mrna_n = [None] * len(sample_index)
    x_mirna_n = [None] * len(sample_index)
    lib_mrna_n = [None] * len(sample_index)
    lib_mirna_n = [None] * len(sample_index)
    res_mrna_n = [None] * len(sample_index)
    res_mirna_n = [None] * len(sample_index)

    # Get the training data and reconstructions for each sample
    for i, j in enumerate(sample_index):
        # Get training data
        x_mrna_n[i], x_mirna_n[i], lib_mrna_n[i], lib_mirna_n[i], _ = train_loader.dataset[j]
        ## mRNA
        x_mrna_n[i] = x_mrna_n[i].cpu().detach().numpy() + 1
        lib_mrna_n[i] = lib_mrna_n[i].cpu().detach().numpy()
        ## miRNA
        x_mirna_n[i] = x_mirna_n[i].cpu().detach().numpy() + 1
        lib_mirna_n[i] = lib_mirna_n[i].cpu().detach().numpy()

        # Get reconstructions via forward method
        res_mrna_n[i], res_mirna_n[i] = dgd.forward(dgd.train_rep.z[j])
        ## mRNA
        res_mrna_n[i] = [tensor.cpu().detach().numpy() for tensor in res_mrna_n[i]]
        res_mrna_n[i] = (res_mrna_n[i] * lib_mrna_n[i]) + 1
        res_mrna_n[i] = np.array(res_mrna_n[i])
        ## miRNA
        res_mirna_n[i] = [tensor.cpu().detach().numpy() for tensor in res_mirna_n[i]]
        res_mirna_n[i] = (res_mirna_n[i] * lib_mirna_n[i]) + 1
        res_mirna_n[i] = np.array(res_mirna_n[i])

    # Plot initialization and cosmetics
    fig, ax = plt.subplots(nrows=2, ncols=len(sample_index), figsize=(20,8))
    fig.subplots_adjust(hspace=0.5)
    sns.set_theme(style="whitegrid")

    # Plot row 1 for mRNA
    for i, j in enumerate(sample_index):
        data = {
            'value': np.concatenate((x_mrna_n[i], res_mrna_n[i])),
            'type': ['Original'] * len(x_mrna_n[i]) + ['Reconstruction'] * len(res_mrna_n[i])
        }
        plotdata = pd.DataFrame(data)
        
        sns.histplot(data=plotdata, x='value', hue='type', ax=ax[0][i], log_scale=True, bins=40)
        ax[0][i].set_title(f'mRNA Sample {j} in epoch {epoch}')
        ax[0][i].set_xlabel('counts+1 (log scale)')
        ax[0][i].set_ylabel(None)

    # Plot row 2 for miRNA
    for i, j in enumerate(sample_index):
        data = {
            'value': np.concatenate((x_mirna_n[i], res_mirna_n[i])),
            'type': ['Original'] * len(x_mirna_n[i]) + ['Reconstruction'] * len(res_mirna_n[i])
        }
        plotdata = pd.DataFrame(data)
        
        sns.histplot(data=plotdata, x='value', hue='type', ax=ax[1][i], log_scale=True, bins=40)
        ax[1][i].set_title(f'miRNA Sample {j} in epoch {epoch}')
        ax[1][i].set_xlabel('counts+1 (log scale)')
        ax[1][i].set_ylabel(None)

    ax[0][0].set_ylabel('Frequency')
    ax[1][0].set_ylabel('Frequency')

    plt.show()


def plot_gene_recons_combined(dgd, train_loader, sample_index, epoch):
    x_mrna_n = [None] * len(sample_index)
    x_mirna_n = [None] * len(sample_index)
    lib_mrna_n = [None] * len(sample_index)
    lib_mirna_n = [None] * len(sample_index)
    res_mrna_n = [None] * len(sample_index)
    res_mirna_n = [None] * len(sample_index)

    # Get mRNA data
    for i, j in enumerate(sample_index):
        # Get training data
        x_mrna_n[i] = train_loader.dataset.mrna_data[:,j] + 1
        x_mirna_n[i] = train_loader.dataset.mirna_data[:,j] + 1
        ## mRNA
        x_mrna_n[i] = x_mrna_n[i].cpu().detach().numpy()
        lib_mrna_n[i] = torch.mean(train_loader.dataset.mrna_data).cpu().detach().numpy()
        ## miRNA
        x_mirna_n[i] = x_mirna_n[i].cpu().detach().numpy()
        lib_mirna_n[i] = torch.mean(train_loader.dataset.mirna_data).cpu().detach().numpy()
        
        # Get gene reconstructions via forward method
        res_mrna_n[i], res_mirna_n[i] = dgd.forward(dgd.train_rep.z)
        ## mRNA
        res_mrna_n[i] = res_mrna_n[i].cpu().detach().numpy()
        res_mrna_n[i] = res_mrna_n[i][:,j] 
        res_mrna_n[i] = (res_mrna_n[i] * lib_mrna_n[i]) + 1
        ## miRNA
        res_mirna_n[i] = res_mirna_n[i].cpu().detach().numpy()
        res_mirna_n[i] = res_mirna_n[i][:,j] 
        res_mirna_n[i] = (res_mirna_n[i] * lib_mirna_n[i]) + 1
        
    # Plot initialization and cosmetics
    fig, ax = plt.subplots(nrows=2, ncols=len(sample_index), figsize=(20,8))
    fig.subplots_adjust(hspace=0.5)
    sns.set_theme(style="whitegrid")

    # Plot row 1 for mRNA
    for i, j in enumerate(sample_index):
        data = {
            'value': np.concatenate((x_mrna_n[i], res_mrna_n[i])),
            'type': ['Original'] * len(x_mrna_n[i]) + ['Reconstruction'] * len(res_mrna_n[i])
        }
        plotdata = pd.DataFrame(data)
        
        sns.histplot(data=plotdata, x='value', hue='type', ax=ax[0][i], log_scale=True, bins=40)
        ax[0][i].set_title(f'Gene {j} in epoch {epoch}')
        ax[0][i].set_xlabel('counts+1 (log scale)')
        ax[0][i].set_ylabel(None)

    # Plot row 2 for miRNA
    for i, j in enumerate(sample_index):
        data = {
            'value': np.concatenate((x_mirna_n[i], res_mirna_n[i])),
            'type': ['Original'] * len(x_mirna_n[i]) + ['Reconstruction'] * len(res_mirna_n[i])
        }
        plotdata = pd.DataFrame(data)
        
        sns.histplot(data=plotdata, x='value', hue='type', ax=ax[1][i], log_scale=True, bins=40)
        ax[1][i].set_title(f'miRNA {j} in epoch {epoch}')
        ax[1][i].set_xlabel('counts+1 (log scale)')
        ax[1][i].set_ylabel(None)

    ax[0][0].set_ylabel('Frequency')
    ax[1][0].set_ylabel('Frequency')
    
    plt.show()


def plot_sample_recons_combined_test(dgd, test_loader, sample_index, epoch):
    x_mrna_n = [None] * len(sample_index)
    x_mirna_n = [None] * len(sample_index)
    lib_mrna_n = [None] * len(sample_index)
    lib_mirna_n = [None] * len(sample_index)
    res_mrna_n = [None] * len(sample_index)
    res_mirna_n = [None] * len(sample_index)

    # Get the training data and reconstructions for each sample
    for i, j in enumerate(sample_index):
        # Get training data
        x_mrna_n[i], x_mirna_n[i], lib_mrna_n[i], lib_mirna_n[i], _ = test_loader.dataset[j]
        ## mRNA
        x_mrna_n[i] = x_mrna_n[i].cpu().detach().numpy() + 1
        lib_mrna_n[i] = lib_mrna_n[i].cpu().detach().numpy()
        ## miRNA
        x_mirna_n[i] = x_mirna_n[i].cpu().detach().numpy() + 1
        lib_mirna_n[i] = lib_mirna_n[i].cpu().detach().numpy()

        # Get reconstructions via forward method
        res_mrna_n[i], res_mirna_n[i] = dgd.forward(dgd.val_rep.z[j])
        ## mRNA
        res_mrna_n[i] = [tensor.cpu().detach().numpy() for tensor in res_mrna_n[i]]
        res_mrna_n[i] = (res_mrna_n[i] * lib_mrna_n[i]) + 1
        res_mrna_n[i] = np.array(res_mrna_n[i])
        ## miRNA
        res_mirna_n[i] = [tensor.cpu().detach().numpy() for tensor in res_mirna_n[i]]
        res_mirna_n[i] = (res_mirna_n[i] * lib_mirna_n[i]) + 1
        res_mirna_n[i] = np.array(res_mirna_n[i])

    # Plot initialization and cosmetics
    fig, ax = plt.subplots(nrows=2, ncols=len(sample_index), figsize=(20,8))
    fig.subplots_adjust(hspace=0.5)
    sns.set_theme(style="whitegrid")

    # Plot row 1 for mRNA
    for i, j in enumerate(sample_index):
        data = {
            'value': np.concatenate((x_mrna_n[i], res_mrna_n[i])),
            'type': ['Original'] * len(x_mrna_n[i]) + ['Reconstruction'] * len(res_mrna_n[i])
        }
        plotdata = pd.DataFrame(data)
        
        sns.histplot(data=plotdata, x='value', hue='type', ax=ax[0][i], log_scale=True, bins=40)
        ax[0][i].set_title(f'mRNA Sample {j} in epoch {epoch}')
        ax[0][i].set_xlabel('counts+1 (log scale)')
        ax[0][i].set_ylabel(None)

    # Plot row 2 for miRNA
    for i, j in enumerate(sample_index):
        data = {
            'value': np.concatenate((x_mirna_n[i], res_mirna_n[i])),
            'type': ['Original'] * len(x_mirna_n[i]) + ['Reconstruction'] * len(res_mirna_n[i])
        }
        plotdata = pd.DataFrame(data)
        
        sns.histplot(data=plotdata, x='value', hue='type', ax=ax[1][i], log_scale=True, bins=40)
        ax[1][i].set_title(f'miRNA Sample {j} in epoch {epoch}')
        ax[1][i].set_xlabel('counts+1 (log scale)')
        ax[1][i].set_ylabel(None)

    ax[0][0].set_ylabel('Frequency')
    ax[1][0].set_ylabel('Frequency')

    plt.show()


def plot_gene_recons_combined_test(dgd, test_loader, sample_index, epoch):
    x_mrna_n = [None] * len(sample_index)
    x_mirna_n = [None] * len(sample_index)
    lib_mrna_n = [None] * len(sample_index)
    lib_mirna_n = [None] * len(sample_index)
    res_mrna_n = [None] * len(sample_index)
    res_mirna_n = [None] * len(sample_index)

    # Get mRNA data
    for i, j in enumerate(sample_index):
        # Get training data
        x_mrna_n[i] = test_loader.dataset.mrna_data[:,j] + 1
        x_mirna_n[i] = test_loader.dataset.mirna_data[:,j] + 1
        ## mRNA
        x_mrna_n[i] = x_mrna_n[i].cpu().detach().numpy()
        lib_mrna_n[i] = torch.mean(test_loader.dataset.mrna_data).cpu().detach().numpy()
        ## miRNA
        x_mirna_n[i] = x_mirna_n[i].cpu().detach().numpy()
        lib_mirna_n[i] = torch.mean(test_loader.dataset.mirna_data).cpu().detach().numpy()
        
        # Get gene reconstructions via forward method
        res_mrna_n[i], res_mirna_n[i] = dgd.forward(dgd.val_rep.z)
        ## mRNA
        res_mrna_n[i] = res_mrna_n[i].cpu().detach().numpy()
        res_mrna_n[i] = res_mrna_n[i][:,j] 
        res_mrna_n[i] = (res_mrna_n[i] * lib_mrna_n[i]) + 1
        ## miRNA
        res_mirna_n[i] = res_mirna_n[i].cpu().detach().numpy()
        res_mirna_n[i] = res_mirna_n[i][:,j] 
        res_mirna_n[i] = (res_mirna_n[i] * lib_mirna_n[i]) + 1
        
    # Plot initialization and cosmetics
    fig, ax = plt.subplots(nrows=2, ncols=len(sample_index), figsize=(20,8))
    fig.subplots_adjust(hspace=0.5)
    sns.set_theme(style="whitegrid")

    # Plot row 1 for mRNA
    for i, j in enumerate(sample_index):
        data = {
            'value': np.concatenate((x_mrna_n[i], res_mrna_n[i])),
            'type': ['Original'] * len(x_mrna_n[i]) + ['Reconstruction'] * len(res_mrna_n[i])
        }
        plotdata = pd.DataFrame(data)
        
        sns.histplot(data=plotdata, x='value', hue='type', ax=ax[0][i], log_scale=True, bins=40)
        ax[0][i].set_title(f'Gene {j} in epoch {epoch}')
        ax[0][i].set_xlabel('counts+1 (log scale)')
        ax[0][i].set_ylabel(None)

    # Plot row 2 for miRNA
    for i, j in enumerate(sample_index):
        data = {
            'value': np.concatenate((x_mirna_n[i], res_mirna_n[i])),
            'type': ['Original'] * len(x_mirna_n[i]) + ['Reconstruction'] * len(res_mirna_n[i])
        }
        plotdata = pd.DataFrame(data)
        
        sns.histplot(data=plotdata, x='value', hue='type', ax=ax[1][i], log_scale=True, bins=40)
        ax[1][i].set_title(f'miRNA {j} in epoch {epoch}')
        ax[1][i].set_xlabel('counts+1 (log scale)')
        ax[1][i].set_ylabel(None)

    ax[0][0].set_ylabel('Frequency')
    ax[1][0].set_ylabel('Frequency')
    
    plt.show()