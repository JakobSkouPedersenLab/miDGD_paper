import os
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
import torch
import matplotlib.pyplot as plt


def plot_latent_space(rep, means, samples, 
                      tissue_labels,
                      color_mapping, epoch=None, fold=0, dataset="Train", 
                      save = False, outdir='plot', outfile='latent_space.png'):
    # get PCA
    pca = PCA(n_components=2)
    pca.fit(rep)
    rep_pca = pca.transform(rep)    
    means_pca = pca.transform(means)
    samples_pca = pca.transform(samples)
    
    # Create main DataFrame with representations
    df = pd.DataFrame(rep_pca, columns=["PC1", "PC2"])
    df["type"] = "Representation"
    df["tissue_label"] = tissue_labels  # Tissue labels for coloring

    df_temp = pd.DataFrame(samples_pca, columns=["PC1", "PC2"])
    df_temp["type"] = "GMM samples"
    df = pd.concat([df,df_temp])
    df_temp = pd.DataFrame(means_pca, columns=["PC1", "PC2"])
    df_temp["type"] = "GMM means"
    df = pd.concat([df,df_temp])

    # make a figure with 2 subplots
    # set a small text size for figures
    plt.rcParams.update({'font.size': 6})
    sns.set_theme(style="white")
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    # add spacing between subplots
    fig.subplots_adjust(wspace=0.2, top=0.9)

    # first plot: representations, means and samples
    sns.scatterplot(data=df, x="PC1", y="PC2", hue="type", size="type", sizes=[3,3,12], alpha=0.8, ax=ax[0], palette=["steelblue","orange","black"])
    ax[0].set_title("E"+str(epoch)+": "+str(dataset)+" Latent Space (by type)")
    ax[0].legend(loc='upper right', fontsize='small')
    
    # add explained variance to x-label and y-label for first plot
    ax[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")

    # second plot: representations by label
    sns.scatterplot(data=df[df["type"] == "Representation"], x="PC1", y="PC2", hue="tissue_label", s=3, alpha=0.8, ax=ax[1], palette=color_mapping)
    ax[1].set_title("E"+str(epoch)+": "+str(dataset)+" Latent Space (by label)")
    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2, markerscale=3)
    
    # add explained variance to x-label and y-label for second plot
    ax[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")

    # if epoch is None, set title without epoch
    if epoch is None:
        plt.suptitle(f'PCA of {dataset} Latent Space', fontsize=16)
    else:
        # if epoch is given, set title with epoch
        plt.suptitle(f'PCA of {dataset} Latent Space in Epoch {epoch}', fontsize=16)
    plt.tight_layout()
    
    if save:
        plt.savefig(os.path.join(outdir, outfile+"_"+dataset+"_F"+str(fold)+"_E"+str(epoch)+".png"))
    else:
        plt.show()


def plot_latent_space_multi(
    rep, means, samples, 
    tissue_labels, batch_labels,
    color_mapping, epoch=None, fold=0, dataset="Train", 
    save = False, outdir='plot', outfile='latent_space.png'
):
    # PCA
    pca = PCA(n_components=2)
    pca.fit(rep)
    rep_pca = pca.transform(rep)
    means_pca = pca.transform(means)
    samples_pca = pca.transform(samples)

    df = pd.DataFrame(rep_pca, columns=["PC1", "PC2"])
    df["type"] = "Representation"
    df["tissue_label"] = tissue_labels
    df["batch_label"] = batch_labels

    df_temp = pd.DataFrame(samples_pca, columns=["PC1", "PC2"])
    df_temp["type"] = "GMM samples"
    df = pd.concat([df, df_temp])
    df_temp = pd.DataFrame(means_pca, columns=["PC1", "PC2"])
    df_temp["type"] = "GMM means"
    df = pd.concat([df, df_temp])

    plt.rcParams.update({'font.size': 6})
    sns.set_theme(style="white")
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    fig.subplots_adjust(wspace=0.2, top=0.9)

    # First plot
    p1 = sns.scatterplot(
        data=df, x="PC1", y="PC2", hue="type", size="type", sizes=[3,3,12], alpha=0.8, 
        ax=ax[0], palette=["steelblue", "orange", "black"]
    )
    ax[0].set_title("")
    ax[0].set_xlabel("")
    ax[0].set_ylabel("")
    l1 = ax[0].legend(loc='upper left', fontsize='small')
    if l1:
        l1.set_title("")

    # Second plot
    p2 = sns.scatterplot(
        data=df[df["type"] == "Representation"], x="PC1", y="PC2", 
        hue="tissue_label", style="batch_label", 
        markers={"TCGA": "D", "GTEx": "s", "SC": "o", "R2": "X"},
        s=5, alpha=0.8, ax=ax[1], palette=color_mapping
    )
    ax[1].set_title("")
    ax[1].set_xlabel("")
    ax[1].set_ylabel("")
    # Remove the default legend of seaborn from the second plot
    ax[1].legend_.remove()

    # Add combined legend to the figure
    handles, labels = ax[1].get_legend_handles_labels()
    # If markers (batch_labels) are not needed in legend, filter out by label name
    # handles, labels = zip(*[(h, l) for h, l in zip(handles, labels) if l in color_mapping])
    fig.legend(
        handles, labels, title="Cancer Type", loc='upper center', bbox_to_anchor=(0.5, -0.02), 
        ncol=8, markerscale=2, fontsize='small'
    )

    # Set explained variance for both subplots using fig.supxlabel/supylabel
    fig.supxlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=14)
    fig.supylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=14)
    
    # Title handling (optional)
    if epoch is None:
        plt.suptitle(f'PCA of {dataset} Latent Space', fontsize=16)
    else:
        plt.suptitle(f'PCA of {dataset} Latent Space in Epoch {epoch}', fontsize=16)
    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(outdir, outfile + "_" + dataset + "_F" + str(fold) + "_E" + str(epoch) + ".png"), bbox_inches='tight', dpi=300)
    else:
        plt.show()

'''
def plot_latent_space_multi(rep, means, samples, 
                            tissue_labels, batch_labels,
                            color_mapping, epoch=None, fold=0, dataset="Train", 
                            save = False, outdir='plot', outfile='latent_space.png'):
    # get PCA
    pca = PCA(n_components=2)
    pca.fit(rep)
    rep_pca = pca.transform(rep)
    means_pca = pca.transform(means)
    samples_pca = pca.transform(samples)
    # Create main DataFrame with representations
    df = pd.DataFrame(rep_pca, columns=["PC1", "PC2"])
    df["type"] = "Representation"
    df["tissue_label"] = tissue_labels  # Tissue labels for coloring
    df["batch_label"] = batch_labels    # Batch labels for marker shapes
    
    df_temp = pd.DataFrame(samples_pca, columns=["PC1", "PC2"])
    df_temp["type"] = "GMM samples"
    df = pd.concat([df,df_temp])
    df_temp = pd.DataFrame(means_pca, columns=["PC1", "PC2"])
    df_temp["type"] = "GMM means"
    df = pd.concat([df,df_temp])

    # make a figure with 2 subplots
    # set a small text size for figures
    plt.rcParams.update({'font.size': 6})
    sns.set_theme(style="white")
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    # add spacing between subplots
    fig.subplots_adjust(wspace=0.2, top=0.9)

    # first plot: representations, means and samples
    sns.scatterplot(data=df, x="PC1", y="PC2", hue="type", size="type", sizes=[3,3,12], alpha=0.8, ax=ax[0], palette=["steelblue","orange","black"])
    ax[0].set_title("E"+str(epoch)+": "+str(dataset)+" Latent Space (by type)")
    ax[0].legend(loc='upper right', fontsize='small')
    
    # add explained variance to x-label and y-label for first plot
    ax[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")

    # second plot: representations by label
    sns.scatterplot(data=df[df["type"] == "Representation"], x="PC1", y="PC2", 
                    hue="tissue_label", style="batch_label", 
                    markers={"TCGA": "D", "GTEx": "s", "SC": "o", "R2": "X"},
                    s=5, alpha=0.8, ax=ax[1], palette=color_mapping)
    ax[1].set_title("E"+str(epoch)+": "+str(dataset)+" Latent Space (by label)")
    #ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2, markerscale=3)
    ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=8, markerscale=2, fontsize='small')
    
    # add explained variance to x-label and y-label for second plot
    ax[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")

    # if epoch is None, set title without epoch
    if epoch is None:
        plt.suptitle(f'PCA of {dataset} Latent Space', fontsize=16)
    else:
        # if epoch is given, set title with epoch
        plt.suptitle(f'PCA of {dataset} Latent Space in Epoch {epoch}', fontsize=16)
    plt.tight_layout()
    
    if save:
        plt.savefig(os.path.join(outdir, outfile+"_"+dataset+"_F"+str(fold)+"_E"+str(epoch)+".png"))
    else:
        plt.show()
'''


def plot_gene(dgd, loader, sample_index, epoch, device, fold=0, type="Train", 
              save=False, scaling_type='mean'):
    x_mrna_n = [None] * len(sample_index)
    res_mrna_n = [None] * len(sample_index)

    # Get mRNA data
    with torch.no_grad():
        for i, j in enumerate(sample_index):
            # Get training data
            x_mrna_n[i] = loader.dataset.mrna_data[:,j].clone().detach().cpu().numpy() + 1

            # Get scaling
            if scaling_type == 'mean':
                scaling = torch.mean(loader.dataset.mrna_data, axis=1)
            elif scaling_type == 'max':
                scaling = torch.max(loader.dataset.mrna_data, axis=1)
            elif scaling_type == 'sum':
                scaling = torch.sum(loader.dataset.mrna_data, axis=1)

            # Get gene reconstructions via forward method
            if type == "Train":
                _, res_mrna_n[i] = dgd.forward(dgd.train_rep())
            elif type == "Validation":
                _, res_mrna_n[i] = dgd.forward(dgd.val_rep())
            elif type == "Test":
                _, res_mrna_n[i] = dgd.forward(dgd.test_rep())
            ## mRNA
            res_mrna_n[i] = res_mrna_n[i] * scaling.unsqueeze(1).to(device)
            res_mrna_n[i] = res_mrna_n[i][:,j].clone().detach().cpu().numpy() + 1
        
    # Create a DataFrame to store the data
    data = []
    for i, j in enumerate(sample_index):
        data.extend([
            {'value': val, 'type': 'Original', 'sample_index': j} for val in x_mrna_n[i]
        ])
        data.extend([
            {'value': val, 'type': 'Reconstruction', 'sample_index': j} for val in res_mrna_n[i]
        ])
    plotdata = pd.DataFrame(data)
    
    # Create the FacetGrid
    sns.set_theme(style="white")
    
    # Map the histplot to each facet
    g = sns.displot(data=plotdata, x='value', hue='type', 
                    bins=40, log_scale=True, col='sample_index', 
                    col_wrap=4, height=3, aspect=1.6, legend=True)
    
    # Set the titles and labels
    g.set_titles(col_template=f'{type} gene ' + '{col_name}' + f' in epoch {epoch}')
    g.set_xlabels('counts+1 (log scale)')
    g.set_ylabels('Frequency')
    
    # Adjust the spacing between subplots
    g.tight_layout()
    g.fig.suptitle(f'{type} mRNA Reconstruction in Epoch {epoch}', fontsize=16)
    g.fig.subplots_adjust(top=0.8)  # Adjust the top margin
    
    sns.move_legend(
        g, "upper center",
        bbox_to_anchor=(0.5, -0.01), ncol=2, title=None, frameon=False
    )
    
    # Save the plot
    path = 'plot'
    if save:
        g.savefig(os.path.join(path, save+"_"+type+"_F"+str(fold)+"_E"+str(epoch)+".png"))
    else:
        plt.show()


def plot_mirna(dgd, loader, sample_index, epoch, device, fold=0, type="Train", save=False, scaling_type='mean'):
    x_mirna_n = [None] * len(sample_index)
    res_mirna_n = [None] * len(sample_index)

    # Get mRNA data
    with torch.no_grad():
        for i, j in enumerate(sample_index):
            # Get training data
            x_mirna_n[i] = loader.dataset.mirna_data[:,j].detach().cpu().numpy() + 1

            # Get scaling
            if scaling_type == 'mean':
                scaling = torch.mean(loader.dataset.mirna_data, axis=1)
            elif scaling_type == 'max':
                scaling = torch.max(loader.dataset.mirna_data, axis=1)
            elif scaling_type == 'sum':
                scaling = torch.sum(loader.dataset.mirna_data, axis=1)

            # Get mirna reconstructions via forward method
            if type == "Train":
                res_mirna_n[i], _ = dgd.forward(dgd.train_rep())
            elif type == "Validation":
                res_mirna_n[i], _ = dgd.forward(dgd.val_rep())
            elif type == "Test":
                res_mirna_n[i], _ = dgd.forward(dgd.test_rep())
            ## miRNA
            res_mirna_n[i] = res_mirna_n[i] * scaling.unsqueeze(1).to(device)
            res_mirna_n[i] = res_mirna_n[i][:,j].detach().cpu().numpy() + 1
        
    # Create a DataFrame to store the data
    data = []
    for i, j in enumerate(sample_index):
        data.extend([
            {'value': val, 'type': 'Original', 'sample_index': j} for val in x_mirna_n[i]
        ])
        data.extend([
            {'value': val, 'type': 'Reconstruction', 'sample_index': j} for val in res_mirna_n[i]
        ])
    plotdata = pd.DataFrame(data)
    
    # Create the FacetGrid
    sns.set_theme(style="white")
    
    # Map the histplot to each facet
    g = sns.displot(data=plotdata, x='value', hue='type', 
                    bins=40, log_scale=True, col='sample_index', 
                    col_wrap=4, height=3, aspect=1.6, legend=True)
    
    # Set the titles and labels
    g.set_titles(col_template=f'{type} miRNA ' + '{col_name}' + f' in epoch {epoch}')
    g.set_xlabels('counts+1 (log scale)')
    g.set_ylabels('Frequency')
    
    # Adjust the spacing between subplots
    g.tight_layout()
    g.fig.suptitle(f'{type} miRNA Reconstruction in Epoch {epoch}', fontsize=16)
    g.fig.subplots_adjust(top=0.8)  # Adjust the top margin
    sns.move_legend(
        g, "upper center",
        bbox_to_anchor=(0.5, -0.01), ncol=2, title=None, frameon=False
    )
    
    # Save the plot
    path = 'plot'
    if save:
        g.savefig(os.path.join(path, save+"_"+type+"_F"+str(fold)+"_E"+str(epoch)+".png"))
    else:
        plt.show()