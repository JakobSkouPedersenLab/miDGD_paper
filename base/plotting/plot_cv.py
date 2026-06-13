import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns


def plot_mirna_recons(dgd, train_loader, sample_index, epoch, type="Train", save_dir='plot/mirna_reconstruction_train.png'):
    x_mirna_n = [None] * len(sample_index)
    lib_mirna_n = [None] * len(sample_index)
    res_mirna_n = [None] * len(sample_index)

    # Get mRNA data
    for i, j in enumerate(sample_index):
        # Get training data
        x_mirna_n[i] = train_loader.dataset.mirna_data[:,j] + 1
        ## miRNA
        x_mirna_n[i] = x_mirna_n[i].cpu().detach().numpy()
        lib_mirna_n[i] = torch.mean(train_loader.dataset.mirna_data).cpu().detach().numpy()
        
        # Get gene reconstructions via forward method
        res_mirna_n[i] = dgd.forward(dgd.train_rep.z)
        ## miRNA
        res_mirna_n[i] = res_mirna_n[i][0].cpu().detach().numpy()
        res_mirna_n[i] = res_mirna_n[i][:,j] 
        res_mirna_n[i] = (res_mirna_n[i] * lib_mirna_n[i]) + 1
        
    # Plot initialization and cosmetics
    fig, ax = plt.subplots(nrows=1, ncols=len(sample_index), figsize=(20,4))
    fig.subplots_adjust(hspace=0.5)
    sns.set_theme(style="whitegrid")

    # Plot row 1 for miRNA
    for i, j in enumerate(sample_index):
        data = {
            'value': np.concatenate((x_mirna_n[i], res_mirna_n[i])),
            'type': ['Original'] * len(x_mirna_n[i]) + ['Reconstruction'] * len(res_mirna_n[i])
        }
        plotdata = pd.DataFrame(data)
        
        sns.histplot(data=plotdata, x='value', hue='type', ax=ax[i], log_scale=True, bins=40)
        ax[i].set_title(f'{type} miRNA {j} in epoch {epoch}')
        ax[i].set_xlabel('counts+1 (log scale)')
        ax[i].set_ylabel(None)

    ax[0].set_ylabel('Frequency')
    
    plt.savefig(save_dir, dpi=300)
    plt.show()


def plot_mirna_recons_test(dgd, validation_loader, sample_index, epoch, type="Test", save_dir='plot/mirna_reconstruction_test.png'):
    x_mirna_n = [None] * len(sample_index)
    lib_mirna_n = [None] * len(sample_index)
    res_mirna_n = [None] * len(sample_index)

    # Get mRNA data
    for i, j in enumerate(sample_index):
        # Get training data
        x_mirna_n[i] = validation_loader.dataset.mirna_data[:,j] + 1
        ## miRNA
        x_mirna_n[i] = x_mirna_n[i].cpu().detach().numpy()
        lib_mirna_n[i] = torch.mean(validation_loader.dataset.mirna_data).cpu().detach().numpy()
        
        # Get gene reconstructions via forward method
        res_mirna_n[i] = dgd.forward(dgd.val_rep.z)
        ## miRNA
        res_mirna_n[i] = res_mirna_n[i][0].cpu().detach().numpy()
        res_mirna_n[i] = res_mirna_n[i][:,j] 
        res_mirna_n[i] = (res_mirna_n[i] * lib_mirna_n[i]) + 1
        
    # Plot initialization and cosmetics
    fig, ax = plt.subplots(nrows=1, ncols=len(sample_index), figsize=(20,4))
    fig.subplots_adjust(hspace=0.5)
    sns.set_theme(style="whitegrid")

    # Plot row 1 for miRNA
    for i, j in enumerate(sample_index):
        data = {
            'value': np.concatenate((x_mirna_n[i], res_mirna_n[i])),
            'type': ['Original'] * len(x_mirna_n[i]) + ['Reconstruction'] * len(res_mirna_n[i])
        }
        plotdata = pd.DataFrame(data)
        
        sns.histplot(data=plotdata, x='value', hue='type', ax=ax[i], log_scale=True, bins=40)
        ax[i].set_title(f'{type} miRNA {j} in epoch {epoch}')
        ax[i].set_xlabel('counts+1 (log scale)')
        ax[i].set_ylabel(None)

    ax[0].set_ylabel('Frequency')
    
    plt.savefig(save_dir, dpi=300)
    plt.show()


def plot_gene_recons(dgd, train_loader, sample_index, epoch, type="Train", save_dir='plot/mrna_reconstruction.png'):
    x_mrna_n = [None] * len(sample_index)
    lib_mrna_n = [None] * len(sample_index)
    res_mrna_n = [None] * len(sample_index)

    # Get mRNA data
    for i, j in enumerate(sample_index):
        # Get training data
        x_mrna_n[i] = train_loader.dataset.mrna_data[:,j] + 1
        ## mRNA
        x_mrna_n[i] = x_mrna_n[i].cpu().detach().numpy()
        lib_mrna_n[i] = torch.mean(train_loader.dataset.mrna_data).cpu().detach().numpy()
        
        # Get gene reconstructions via forward method
        res_mrna_n[i] = dgd.forward(dgd.train_rep.z)
        ## mRNA
        res_mrna_n[i] = res_mrna_n[i][0].cpu().detach().numpy()
        res_mrna_n[i] = res_mrna_n[i][:,j] 
        res_mrna_n[i] = (res_mrna_n[i] * lib_mrna_n[i]) + 1
        
    # Plot initialization and cosmetics
    fig, ax = plt.subplots(nrows=1, ncols=len(sample_index), figsize=(20,4))
    fig.subplots_adjust(hspace=0.5)
    sns.set_theme(style="whitegrid")

    # Plot row 1 for mRNA
    for i, j in enumerate(sample_index):
        data = {
            'value': np.concatenate((x_mrna_n[i], res_mrna_n[i])),
            'type': ['Original'] * len(x_mrna_n[i]) + ['Reconstruction'] * len(res_mrna_n[i])
        }
        plotdata = pd.DataFrame(data)
        
        sns.histplot(data=plotdata, x='value', hue='type', ax=ax[i], log_scale=True, bins=40)
        ax[i].set_title(f'{type} gene {j} in epoch {epoch}')
        ax[i].set_xlabel('counts+1 (log scale)')
        ax[i].set_ylabel(None)

    ax[0].set_ylabel('Frequency')
    
    plt.savefig(save_dir, dpi=300)
    plt.show()


def plot_gene_recons_test(dgd, test_loader, sample_index, epoch, type="Test", save_dir='plot/mrna_reconstruction_test.png'):
    x_mrna_n = [None] * len(sample_index)
    lib_mrna_n = [None] * len(sample_index)
    res_mrna_n = [None] * len(sample_index)

    # Get mRNA data
    for i, j in enumerate(sample_index):
        # Get training data
        x_mrna_n[i] = test_loader.dataset.mrna_data[:,j] + 1
        ## mRNA
        x_mrna_n[i] = x_mrna_n[i].cpu().detach().numpy()
        lib_mrna_n[i] = torch.mean(test_loader.dataset.mrna_data).cpu().detach().numpy()
        
        # Get gene reconstructions via forward method
        res_mrna_n[i] = dgd.forward(dgd.val_rep.z)
        ## mRNA
        res_mrna_n[i] = res_mrna_n[i][0].cpu().detach().numpy()
        res_mrna_n[i] = res_mrna_n[i][:,j] 
        res_mrna_n[i] = (res_mrna_n[i] * lib_mrna_n[i]) + 1
        
    # Plot initialization and cosmetics
    fig, ax = plt.subplots(nrows=1, ncols=len(sample_index), figsize=(20,4))
    fig.subplots_adjust(hspace=0.5)
    sns.set_theme(style="whitegrid")

    # Plot row 1 for mRNA
    for i, j in enumerate(sample_index):
        data = {
            'value': np.concatenate((x_mrna_n[i], res_mrna_n[i])),
            'type': ['Original'] * len(x_mrna_n[i]) + ['Reconstruction'] * len(res_mrna_n[i])
        }
        plotdata = pd.DataFrame(data)
        
        sns.histplot(data=plotdata, x='value', hue='type', ax=ax[i], log_scale=True, bins=40)
        ax[i].set_title(f'{type} gene {j} in epoch {epoch}')
        ax[i].set_xlabel('counts+1 (log scale)')
        ax[i].set_ylabel(None)

    ax[0].set_ylabel('Frequency')
    
    plt.savefig(save_dir, dpi=300)
    plt.show()