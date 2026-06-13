import os
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
import torch
import matplotlib.pyplot as plt

def plot_latent_space(rep, means, samples, labels, color_mapping, epoch=None, fold=0, dataset="Train", 
                      save = False, outdir='plot', outfile='latent_space.png'):
    # get PCA
    pca = PCA(n_components=2)
    pca.fit(rep)
    rep_pca = pca.transform(rep)
    means_pca = pca.transform(means)
    samples_pca = pca.transform(samples)
    df = pd.DataFrame(rep_pca, columns=["PC1", "PC2"])
    df["type"] = "Representation"
    df["label"] = labels
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
    sns.scatterplot(data=df[df["type"] == "Representation"], x="PC1", y="PC2", hue="label", s=3, alpha=0.8, ax=ax[1], palette=color_mapping)
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



def plot_latent_space_covar(rep, means, samples, 
                      tissue_labels, batch_labels, 
                      color_mapping, 
                      epoch=None, fold=0, dataset="Train", 
                      save = False, outdir='plot', outfile='latent_space.png'):
    # Get PCA
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

    # Add GMM samples and means
    df_temp = pd.DataFrame(samples_pca, columns=["PC1", "PC2"])
    df_temp["type"] = "GMM samples"
    df = pd.concat([df, df_temp])
    
    df_temp = pd.DataFrame(means_pca, columns=["PC1", "PC2"])
    df_temp["type"] = "GMM means"
    df = pd.concat([df, df_temp])

    # Make a figure with 2 subplots
    # Set a small text size for figures
    plt.rcParams.update({'font.size': 6})
    sns.set_theme(style="white")
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    # Add spacing between subplots
    fig.subplots_adjust(wspace=0.2, top=0.9)

    # First plot: representations, means and samples
    sns.scatterplot(data=df, 
                    x="PC1", 
                    y="PC2", 
                    hue="type", 
                    size="type", 
                    sizes=[3,3,12], 
                    alpha=0.8, ax=ax[0], 
                    palette=["steelblue","orange","black"])
    ax[0].set_title("E"+str(epoch)+": "+str(dataset)+" Latent Space (by type)")
    ax[0].legend(loc='upper right', fontsize='small')
    
    # add explained variance to x-label and y-label for first plot
    ax[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")

    # second plot: representations by label
    sns.scatterplot(data=df[df["type"] == "Representation"], 
                    x="PC1", y="PC2", 
                    hue="tissue_label", 
                    style="batch_label",
                    s=8, alpha=0.8, ax=ax[1], 
                    palette=color_mapping)
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

def plot_latent_space_covar_multi(
    rep_train, means_train, samples_train, tissue_train, batch_train,
    rep_val, means_val, samples_val, tissue_val, batch_val,
    rep_test, means_test, samples_test, tissue_test, batch_test,
    color_mapping,
    epoch=None, fold=0,
    save=False, outdir='plots', outfile=None
):
    # Fit PCA on train only
    pca = PCA(n_components=2)
    pca.fit(rep_train)
    rep_train_pca = pca.transform(rep_train)
    means_train_pca = pca.transform(means_train)
    samples_train_pca = pca.transform(samples_train)

    rep_val_pca = pca.transform(rep_val)
    means_val_pca = pca.transform(means_val)
    samples_val_pca = pca.transform(samples_val)

    rep_test_pca = pca.transform(rep_test)
    means_test_pca = pca.transform(means_test)
    samples_test_pca = pca.transform(samples_test)

    def build_df(rep_pca, means_pca, samples_pca, tissue, batch, dataset_name):
        df = pd.DataFrame(rep_pca, columns=["PC1", "PC2"])
        df["type"] = "Representation"
        df["dataset"] = dataset_name
        df["tissue_label"] = tissue
        df["batch_label"] = batch

        df_samples = pd.DataFrame(samples_pca, columns=["PC1", "PC2"])
        df_samples["type"] = "GMM samples"
        df_samples["dataset"] = dataset_name
        df_samples["tissue_label"] = None
        df_samples["batch_label"] = None

        df_means = pd.DataFrame(means_pca, columns=["PC1", "PC2"])
        df_means["type"] = "GMM means"
        df_means["dataset"] = dataset_name
        df_means["tissue_label"] = None
        df_means["batch_label"] = None

        return pd.concat([df, df_samples, df_means])

    df_train = build_df(rep_train_pca, means_train_pca, samples_train_pca, tissue_train, batch_train, "Train")
    df_val = build_df(rep_val_pca, means_val_pca, samples_val_pca, tissue_val, batch_val, "Validation")
    df_test = build_df(rep_test_pca, means_test_pca, samples_test_pca, tissue_test, batch_test, "Test")

    # Concatenate everything for axis limit calculation
    df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)
    # Find global min/max for PC1 and PC2 across all relevant points
    x_min = df_all["PC1"].min()-0.05
    x_max = df_all["PC1"].max()+0.05
    y_min = df_all["PC2"].min()-0.05
    y_max = df_all["PC2"].max()+0.05

    explained1 = f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)"
    explained2 = f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)"

    datasets = [("Train", df_train), ("Validation", df_val), ("Test", df_test)]

    plt.rcParams.update({'font.size': 6})
    sns.set_theme(style="white")
    fig, ax = plt.subplots(2, 3, figsize=(24, 12))
    fig.subplots_adjust(wspace=0.25, hspace=0.3, top=0.92)

    for i, (name, df) in enumerate(datasets):
        # Top row: type coloring
        sns.scatterplot(data=df,
                        x="PC1", y="PC2",
                        hue="type", size="type",
                        sizes=[3, 3, 12],
                        alpha=0.8, ax=ax[0, i],
                        palette=["steelblue","orange","black"])
        ax[0, i].set_title(f"E{str(epoch)}: {name} Latent Space (by type)")
        ax[0, i].legend(loc='upper right', fontsize='small')
        ax[0, i].set_xlabel(explained1)
        ax[0, i].set_ylabel(explained2)
        ax[0, i].set_xlim(x_min, x_max)
        ax[0, i].set_ylim(y_min, y_max)

        # Bottom row: tissue label coloring (only Representation type)
        df_repr = df[df['type'] == 'Representation']
        sns.scatterplot(data=df_repr,
                        x="PC1", y="PC2",
                        hue="tissue_label",
                        style="batch_label",
                        s=8, alpha=0.8, ax=ax[1, i],
                        palette=color_mapping,
                        legend=False if i < 2 else True)
        ax[1, i].set_title(f"E{str(epoch)}: {name} Latent Space (by label)")
        ax[1, i].set_xlabel(explained1)
        ax[1, i].set_ylabel(explained2)
        ax[1, i].set_xlim(x_min, x_max)
        ax[1, i].set_ylim(y_min, y_max)

    fig.suptitle(
        f'PCA of Latent Spaces (fixed to Train)\nEpoch {epoch}' if epoch is not None else 'PCA of Latent Spaces (fixed to Train)', fontsize=20
    )

    # Add legend outside last subplot if needed
    handles, labels = ax[1,2].get_legend_handles_labels()
    ax[1,2].legend(handles=handles, labels=labels, loc='center left', bbox_to_anchor=(1, 0.5), ncol=2, markerscale=3, fontsize='small')

    plt.tight_layout(rect=[0, 0, 0.95, 0.97])

    if outfile:
        plt.savefig(os.path.join(outdir, outfile))
    
    plt.show()


def plot_latent_space_covar_two(
    rep_train, means_train, samples_train, tissue_train, batch_train,
    rep_val, means_val, samples_val, tissue_val, batch_val,
    color_mapping,
    epoch=None, fold=0,
    save=False, outdir='plots', outfile=None
):
    # Fit PCA on train only
    pca = PCA(n_components=2)
    pca.fit(rep_train)
    rep_train_pca = pca.transform(rep_train)
    means_train_pca = pca.transform(means_train)
    samples_train_pca = pca.transform(samples_train)

    rep_val_pca = pca.transform(rep_val)
    means_val_pca = pca.transform(means_val)
    samples_val_pca = pca.transform(samples_val)

    def build_df(rep_pca, means_pca, samples_pca, tissue, batch, dataset_name):
        df = pd.DataFrame(rep_pca, columns=["PC1", "PC2"])
        df["type"] = "Representation"
        df["dataset"] = dataset_name
        df["tissue_label"] = tissue
        df["batch_label"] = batch

        df_samples = pd.DataFrame(samples_pca, columns=["PC1", "PC2"])
        df_samples["type"] = "GMM samples"
        df_samples["dataset"] = dataset_name
        df_samples["tissue_label"] = None
        df_samples["batch_label"] = None

        df_means = pd.DataFrame(means_pca, columns=["PC1", "PC2"])
        df_means["type"] = "GMM means"
        df_means["dataset"] = dataset_name
        df_means["tissue_label"] = None
        df_means["batch_label"] = None

        return pd.concat([df, df_samples, df_means])

    df_train = build_df(rep_train_pca, means_train_pca, samples_train_pca, tissue_train, batch_train, "Train")
    df_val = build_df(rep_val_pca, means_val_pca, samples_val_pca, tissue_val, batch_val, "Validation")

    # Concatenate everything for axis limit calculation
    df_all = pd.concat([df_train, df_val], ignore_index=True)
    x_min = df_all["PC1"].min() - 0.05
    x_max = df_all["PC1"].max() + 0.05
    y_min = df_all["PC2"].min() - 0.05
    y_max = df_all["PC2"].max() + 0.05

    explained1 = f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)"
    explained2 = f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)"

    datasets = [("Train", df_train), ("Validation", df_val)]

    plt.rcParams.update({'font.size': 6})
    sns.set_theme(style="white")
    fig, ax = plt.subplots(2, 2, figsize=(16, 12))
    fig.subplots_adjust(wspace=0.25, hspace=0.3, top=0.92)

    for i, (name, df) in enumerate(datasets):
        # Top row: type coloring
        sns.scatterplot(data=df,
                        x="PC1", y="PC2",
                        hue="type", size="type",
                        sizes=[3, 3, 12],
                        alpha=0.8, ax=ax[0, i],
                        palette=["steelblue", "orange", "black"])
        ax[0, i].set_title(f"E{str(epoch)}: {name} Latent Space (by type)")
        ax[0, i].legend(loc='upper right', fontsize='small')
        ax[0, i].set_xlabel(explained1)
        ax[0, i].set_ylabel(explained2)
        ax[0, i].set_xlim(x_min, x_max)
        ax[0, i].set_ylim(y_min, y_max)

        # Bottom row: tissue label coloring (only Representation type)
        df_repr = df[df['type'] == 'Representation']
        sns.scatterplot(data=df_repr,
                        x="PC1", y="PC2",
                        hue="tissue_label",
                        style="batch_label",
                        s=8, alpha=0.8, ax=ax[1, i],
                        palette=color_mapping,
                        legend=False if i < 1 else True)
        ax[1, i].set_title(f"E{str(epoch)}: {name} Latent Space (by label)")
        ax[1, i].set_xlabel(explained1)
        ax[1, i].set_ylabel(explained2)
        ax[1, i].set_xlim(x_min, x_max)
        ax[1, i].set_ylim(y_min, y_max)

    fig.suptitle(
        f'PCA of Latent Spaces (fixed to Train)\nEpoch {epoch}' if epoch is not None else 'PCA of Latent Spaces (fixed to Train)', fontsize=20
    )

    # Add legend outside last subplot if needed
    handles, labels = ax[1, 1].get_legend_handles_labels()
    ax[1, 1].legend(handles=handles, labels=labels, loc='center left', bbox_to_anchor=(1, 0.5), ncol=2, markerscale=3, fontsize='small')

    plt.tight_layout(rect=[0, 0, 0.95, 0.97])
    if outfile:
        plt.savefig(os.path.join(outdir, outfile))
    plt.show()
