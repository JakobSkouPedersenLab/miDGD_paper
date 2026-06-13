import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA


# Barplot

def barplot_dataset(df, groupby='cancer_type', figsize=(9, 5.5), savedir=None, filename=None):
    color_mapping = dict(zip(df[groupby], df['color']))

    primary_site_counts = df[groupby].value_counts().reset_index()
    primary_site_counts.columns = [groupby, 'count']

    sns.set_theme(style="ticks")
    # Create a bar plot
    plt.figure(figsize=figsize)
    barplot = sns.barplot(x=groupby, hue=groupby, y='count', data=primary_site_counts, palette=color_mapping)
    plt.xlabel('', fontsize=14)
    plt.ylabel('Number of Samples', fontsize=14)
    plt.title('')
    plt.xticks(rotation=90, ha='center', fontsize=14)  # Rotate the x labels to show them more clearly
    plt.yticks(fontsize=14, visible=False)  # Rotate the x labels to show them more clearly

    
    # Annotate each bar with the count
    for p in barplot.patches:
        barplot.annotate(format(p.get_height(), '.0f'),  # Format the count as a string with no decimal places
                         (p.get_x() + p.get_width() / 2., p.get_height()),  # Position the text at the center of the bar
                         ha='center', va='center',  # Center the text horizontally and vertically
                         rotation=90, fontsize=12,
                         xytext=(0, 20),  # Offset the text by 10 points vertically
                         textcoords='offset points')  # Use offset points for the text coordinates
    # Change the color of x-axis tick labels to match the bar colors
    for i, tick in enumerate(barplot.get_xticklabels()):
        tick.set_color(color_mapping[primary_site_counts.iloc[i][groupby]])
        
    plt.tight_layout()  # Adjust the layout to fit the x labels
    sns.despine()
    if filename:
        os.makedirs(savedir, exist_ok=True)
        plt.savefig(os.path.join(savedir, filename), bbox_inches='tight')
    plt.show()


# Latent space

def plot_latent_space_multi_grid(
    train_vals, val_vals, test_vals,
    train_labels, val_labels, test_labels,
    train_batch, val_batch, test_batch,
    color_mapping,
    save=False, savedir=None, filename=None
):

    # Unpack
    tr_rep, tr_means, tr_samples = train_vals
    va_rep, va_means, va_samples = val_vals
    te_rep, te_means, te_samples = test_vals

    # Fit PCA on all reps
    combined_rep = np.concatenate([tr_rep, va_rep, te_rep], axis=0)
    pca = PCA(n_components=2)
    pca.fit(combined_rep)

    # Transform
    tr_rep_pca = pca.transform(tr_rep)
    tr_means_pca = pca.transform(tr_means)
    tr_samples_pca = pca.transform(tr_samples)
    va_rep_pca = pca.transform(va_rep)
    va_means_pca = pca.transform(va_means)
    va_samples_pca = pca.transform(va_samples)
    te_rep_pca = pca.transform(te_rep)
    te_means_pca = pca.transform(te_means)
    te_samples_pca = pca.transform(te_samples)

    # Axis limits
    all_pc1 = np.concatenate([tr_rep_pca[:,0], va_rep_pca[:,0], te_rep_pca[:,0]])
    all_pc2 = np.concatenate([tr_rep_pca[:,1], va_rep_pca[:,1], te_rep_pca[:,1]])
    margin = 0.05
    x_min, x_max = all_pc1.min(), all_pc1.max()
    y_min, y_max = all_pc2.min(), all_pc2.max()
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_lim = (x_min - margin*x_range, x_max + margin*x_range)
    y_lim = (y_min - margin*y_range, y_max + margin*y_range)

    # Helper
    def make_df(rep_pca, means_pca, samples_pca, tissue_labels, batch_labels):
        df = pd.DataFrame(rep_pca, columns=["PC1", "PC2"])
        df["type"] = "Representation"
        df["tissue_label"] = tissue_labels
        df["batch_label"] = batch_labels

        df_samples = pd.DataFrame(samples_pca, columns=["PC1", "PC2"])
        df_samples["type"] = "GMM samples"
        df_samples["tissue_label"] = np.nan
        df_samples["batch_label"] = np.nan
        df = pd.concat([df, df_samples])

        df_means = pd.DataFrame(means_pca, columns=["PC1", "PC2"])
        df_means["type"] = "GMM means"
        df_means["tissue_label"] = np.nan
        df_means["batch_label"] = np.nan
        df = pd.concat([df, df_means])
        return df

    dfs = [
        make_df(tr_rep_pca, tr_means_pca, tr_samples_pca, train_labels, train_batch),
        make_df(va_rep_pca, va_means_pca, va_samples_pca, val_labels, val_batch),
        make_df(te_rep_pca, te_means_pca, te_samples_pca, test_labels, test_batch)
    ]
    names = ["Train", "Val", "Test"]

    plt.rcParams.update({'font.size': 9})
    sns.set_theme(style="white")
    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    fig.subplots_adjust(wspace=0.2, hspace=0.25)

    # top row: by type
    for j, (df, name) in enumerate(zip(dfs, names)):
        ax1 = axes[0, j]
        sns.scatterplot(
            data=df, x="PC1", y="PC2", hue="type", size="type", sizes=[3,3,12], alpha=0.8,
            ax=ax1, palette=["steelblue", "orange", "black"]
        )
        ax1.set_title(f'{name} Latent Space (by type)', fontsize=15)
        ax1.set_xlim(*x_lim)
        ax1.set_ylim(*y_lim)
        ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)" if j == 1 else "")
        ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)" if j == 0 else "")
        if j == 0:
            l = ax1.legend(loc='upper left')
            if l: l.set_title("")
        else:
            ax1.get_legend().remove()

    # bottom row: by label
    legend_ax = axes[1, 1]  # Collect legend from middle axis for reliability!
    scatter = None
    for j, (df, name) in enumerate(zip(dfs, names)):
        ax2 = axes[1, j]
        scatter = sns.scatterplot(
            data=df[df["type"] == "Representation"], x="PC1", y="PC2",
            hue="tissue_label", style="batch_label",
            markers={"TCGA": "D", "GTEx": "s", "SC": "o", "R2": "X"},
            s=5, alpha=0.8, ax=ax2, palette=color_mapping, legend=True
        )
        ax2.set_title(f'{name} Latent Space (by label)', fontsize=15)
        ax2.set_xlim(*x_lim)
        ax2.set_ylim(*y_lim)
        ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax2.set_ylabel("")
        ax2.get_legend().remove()  # remove automatic legend

    # Manually place the legend below the figure, shows all tissue labels
    # Use legend_ax as source for handles/labels
    handles, labels = legend_ax.get_legend_handles_labels()
    fig.legend(
        handles, labels, title="Cancer Type", loc='lower center', bbox_to_anchor=(0.5, -0.12),
        ncol=8, markerscale=2, fontsize='medium'
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if filename:
        plt.savefig(os.path.join(savedir, filename), bbox_inches="tight", dpi=300)
    else:
        plt.show()

def plot_single_latent_space(
    rep, tissue_labels, batch_labels, color_mapping,
    title="Latent Space", save=False, savedir=None, filename=None
):
    pca = PCA(n_components=2)
    rep_pca = pca.fit_transform(rep)
    df = pd.DataFrame(rep_pca, columns=["PC1", "PC2"])
    df["tissue_label"] = tissue_labels
    df["batch_label"] = batch_labels

    # Smaller figure
    plt.figure(figsize=(6, 4))
    sns.set_theme(style="white")

    # Larger points, no legend for batch
    scatter = sns.scatterplot(
        data=df, x="PC1", y="PC2",
        hue="tissue_label",           # keep only tissue in legend
        style=None,                   # remove batch_label style
        s=25,                         # larger dots
        alpha=0.8,
        palette=color_mapping,
        legend="full",
        linewidth=0.1
    )

    scatter.set_title("", fontsize=15)
    scatter.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    scatter.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    
    # Remove x and y ticks
    scatter.set_xticks([])
    scatter.set_yticks([])

    # Now legend only reflects tissue_label (Cancer Type)
    handles, labels = scatter.get_legend_handles_labels()
    plt.legend(
        handles=handles, labels=labels, title="Cancer Type",
        loc='center left', bbox_to_anchor=(1.02, 0.5),
        ncol=2, fontsize='x-small', markerscale=1.2, frameon=True
    )
    plt.tight_layout()

    if filename:
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        plt.savefig(os.path.join(savedir, filename), bbox_inches="tight", dpi=300)
    
    plt.show()


def plot_latent_space_ms(rep, means, samples, gmm, labels, color_mapping, data_loader, 
                         title="Train", savedir=None, filename=None, ari=None):
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
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    # add spacing between subplots
    fig.subplots_adjust(wspace=0.2, top=0.9)

    # first plot: representations, means and samples
    sns.scatterplot(data=df, x="PC1", y="PC2", hue="type", size="type", sizes=[8,5,20], 
                    edgecolor = "none", alpha=1, ax=ax[0], palette=["steelblue","orange","black"])
    ax[0].set_title("")
    ax[0].legend(loc='upper left', fontsize='medium', markerscale=3)
    
    # add explained variance to x-label and y-label for first plot
    ax[0].set_xlabel("")
    ax[0].set_ylabel("")

    # second plot: representations by label
    sns.scatterplot(data=df[df["type"] == "Representation"], x="PC1", y="PC2", hue="label", s=10,
                    edgecolor = "none", alpha=1, ax=ax[1], palette=color_mapping)
    ax[1].set_title("")
    ax[1].legend(loc='center right', bbox_to_anchor=(1, 0.5), ncol=2, markerscale=3, fontsize='x-small').remove()
    # Create a combined legend for the second subplot
    
    # add explained variance to x-label and y-label for second plot
    ax[1].set_xlabel("")
    ax[1].set_ylabel("")
    #ax[1].set_xticklabels(fontsize=14)
    ax[1].set_yticklabels("")

    # Add ARI text inside first subplot (axes coordinates)
    if ari is not None:
        ari_text = f"ARI: {ari:.3f}"
        ax[1].text(
            0.98, 0.98, ari_text,           # near top-right inside axes
            transform=ax[1].transAxes,
            va='top', ha='right',           # anchor text to that corner
            fontsize=14
        )

    # plot
    #plt.suptitle(f'{title} PCA', fontsize=16).set_visible(False)
    fig.supxlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=14, y=0.05)
    fig.supylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=14, x=0.04)
    
    handles, labels = ax[1].get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    fig.legend(handles, labels, title="Cancer Type", loc='upper center', bbox_to_anchor=(0.53, 0.05), ncol=8, markerscale=2, fontsize='small')
    #fig.legend(handles, labels, title="Cancer Type", loc='upper left', bbox_to_anchor=(1, 0.9), ncol=2, markerscale=3, fontsize='medium')

    #ax[0].cla()              # clear the second axes
    #ax[0].axis('off')        # hide it completely
    #ax[1].cla()              # clear the second axes
    #ax[1].axis('off')        # hide it completely
    plt.tight_layout()
    
    if filename:
        plt.savefig(os.path.join(savedir, filename), bbox_inches='tight')
    plt.show()

    return pca

def plot_latent_space_dataonly(rep, means, samples, gmm, labels, color_mapping, title="Train", 
                               savedir=None, filename=None, xlim=None, ylim=None, return_limits=False, ari=None):
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

    # Set up the plot
    plt.rcParams.update({'font.size': 8})  
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(6, 6))

    sns.scatterplot(data=df[df["type"] == "Representation"], x="PC1", y="PC2", hue="label", s=10,
                    edgecolor="none", alpha=1, ax=ax, palette=color_mapping)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=14)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=14)
    plt.title(f'{title} PCA', fontsize=16, pad=10).set_visible(False)

    handles, leg_labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax.legend(handles, leg_labels, title="Cancer Types", loc='center left', bbox_to_anchor=(1, 0.5), 
              ncol=2, markerscale=2, fontsize='small', title_fontsize='small').set_visible(False)

    # Add ARI text inside first subplot (axes coordinates)
    if ari is not None:
        ari_text = f"ARI: {ari:.3f}"
        ax.text(
            0.98, 0.98, ari_text,           # near top-right inside axes
            transform=ax.transAxes,
            va='top', ha='right',           # anchor text to that corner
            fontsize=12
        )

    # Use given limits or get from plot
    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        xlim = ax.get_xlim()
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        ylim = ax.get_ylim()

    plt.tight_layout()
    if filename:
        plt.savefig(os.path.join(savedir, filename), bbox_inches='tight', dpi=300)
    plt.show()

    if return_limits:
        return xlim, ylim


# Cancer legends

def plot_cancer_legend(color_mapping, savedir=None, filename=None):
    import matplotlib.lines as mlines

    # make a dummy figure
    fig = plt.figure(figsize=(6, 1.5))
    ax = fig.add_subplot(111)
    ax.axis("off")  # no axes

    # build dummy handles with the right colors
    handles = []
    labels = []
    for label, color in sorted(color_mapping.items(), key=lambda t: t[0]):
        h = mlines.Line2D(
            [], [], color=color, marker='o', linestyle='None',
            markersize=6, label=label
        )
        handles.append(h)
        labels.append(label)

    # figure-level legend only
    fig.legend(
        handles, labels, title="Cancer Type",
        loc="center", bbox_to_anchor=(0.5, 0.5),
        ncol=4, markerscale=2, fontsize='small'
    )

    if filename:
        os.makedirs(savedir, exist_ok=True)
        plt.savefig(os.path.join(savedir, filename), bbox_inches="tight")
    plt.show()

