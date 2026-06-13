import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA


# Barplot

def barplot_dataset(
    df,
    groupby='cancer_type',
    figsize=(9, 5.5),
    savedir=None,
    filename=None,
    xlabel='',
    ylabel='Number of Samples',
    xtick_fontsize=14,
    ytick_fontsize=14,
    annotate_fontsize=12,
    save_format=None,
):
    color_mapping = dict(zip(df[groupby], df['color']))

    primary_site_counts = df[groupby].value_counts().reset_index()
    primary_site_counts.columns = [groupby, 'count']

    sns.set_theme(style="ticks")
    # Create a bar plot
    plt.figure(figsize=figsize)
    barplot = sns.barplot(x=groupby, hue=groupby, y='count', data=primary_site_counts, palette=color_mapping)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title('')
    plt.xticks(rotation=90, ha='center', fontsize=xtick_fontsize)
    plt.yticks(fontsize=ytick_fontsize, visible=False)

    
    # Annotate each bar with the count
    for p in barplot.patches:
        barplot.annotate(format(p.get_height(), '.0f'),  # Format the count as a string with no decimal places
                         (p.get_x() + p.get_width() / 2., p.get_height()),  # Position the text at the center of the bar
                         ha='center', va='center',  # Center the text horizontally and vertically
                         rotation=90, fontsize=annotate_fontsize,
                         xytext=(0, 20),  # Offset the text by 10 points vertically
                         textcoords='offset points')  # Use offset points for the text coordinates
    # Change the color of x-axis tick labels to match the bar colors
    for i, tick in enumerate(barplot.get_xticklabels()):
        tick.set_color(color_mapping[primary_site_counts.iloc[i][groupby]])
        
    plt.tight_layout()  # Adjust the layout to fit the x labels
    sns.despine()
    if filename:
        os.makedirs(savedir, exist_ok=True)
        save_kwargs = {"bbox_inches": "tight"}
        if save_format:
            save_kwargs["format"] = save_format
        plt.savefig(os.path.join(savedir, filename), **save_kwargs)
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


def plot_projection(df, x_proj, y_proj, color_col="cols", title="Projection"):
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=df[x_proj], y=df[y_proj], hue=df[color_col], s=30)
    plt.xlabel(x_proj)
    plt.ylabel(y_proj)
    plt.title(title)
    plt.show()


def plot_average_correlation(corr_data, savedir=None, filename=None):
    sns.set_style("white")
    plt.rcParams.update({"xtick.labelsize": 10, "ytick.labelsize": 10})
    fig = plt.figure(figsize=(8, 3))

    plt.subplot(1, 2, 1)
    sns.histplot(data=corr_data, x="spearman", bins=50, color="#4B7F52", edgecolor="black")
    plt.xlabel("Spearman's Correlation", fontsize=10)
    plt.ylabel("Number of miRNAs", fontsize=10)
    mean_sp = corr_data["spearman"].mean()
    plt.axvline(mean_sp, color="red", linestyle="dashed", linewidth=1)
    plt.text(mean_sp - 0.02, plt.gca().get_ylim()[1] * 0.9, f"Mean: {mean_sp:.3f}",
             color="red", ha="right", fontsize=10)

    plt.subplot(1, 2, 2)
    sns.histplot(data=corr_data, x="pearson", bins=50, color="#2A6F97", edgecolor="black")
    plt.xlabel("Pearson's Correlation", fontsize=10)
    plt.ylabel("")
    mean_pe = corr_data["pearson"].mean()
    plt.axvline(mean_pe, color="red", linestyle="dashed", linewidth=1)
    plt.text(mean_pe - 0.02, plt.gca().get_ylim()[1] * 0.9, f"Mean: {mean_pe:.3f}",
             color="red", ha="right", fontsize=10)

    sns.despine()
    plt.tight_layout()
    if filename:
        savedir = savedir or "."
        os.makedirs(savedir, exist_ok=True)
        plt.savefig(os.path.join(savedir, filename), bbox_inches="tight")
    plt.show()


def plot_poisson_pseudo_r2_hist(pseudo_r2_df, savedir=None, filename=None):
    sns.set_style("white")
    plt.rcParams.update({"xtick.labelsize": 10, "ytick.labelsize": 10})
    fig = plt.figure(figsize=(6, 3))
    sns.histplot(data=pseudo_r2_df, x="pseudo_r2", bins=50, color="#2A6F97", edgecolor="black")
    mean_r2 = pseudo_r2_df["pseudo_r2"].dropna().mean()
    plt.axvline(mean_r2, color="red", linestyle="dashed", linewidth=1)
    plt.text(mean_r2, plt.gca().get_ylim()[1] * 0.9, f"Mean: {mean_r2:.3f}", color="red", fontsize=10)
    plt.xlabel("Poisson Pseudo R²", fontsize=10)
    plt.ylabel("Number of miRNAs", fontsize=10)
    sns.despine()
    plt.tight_layout()
    if filename:
        savedir = savedir or "."
        os.makedirs(savedir, exist_ok=True)
        plt.savefig(os.path.join(savedir, filename), bbox_inches="tight")
    plt.show()


# ── Latent space: batch-split grid ──────────────────────────────────────────

def plot_latent_space_multi_split(
    rep, means, samples,
    tissue_labels, batch_labels,
    color_mapping, savedir=None, filename=None,
):
    from sklearn.decomposition import PCA
    unique_batches = sorted(set(batch_labels))
    n = len(unique_batches)

    pca = PCA(n_components=2)
    pca.fit(rep)
    rep_pca     = pca.transform(rep)
    means_pca   = pca.transform(means)
    samples_pca = pca.transform(samples)

    df = pd.DataFrame(rep_pca, columns=["PC1", "PC2"])
    df["type"]         = "Representation"
    df["tissue_label"] = tissue_labels
    df["batch_label"]  = batch_labels

    df_means   = pd.DataFrame(means_pca,   columns=["PC1", "PC2"])
    df_means["type"] = "GMM means"
    df_samples = pd.DataFrame(samples_pca, columns=["PC1", "PC2"])
    df_samples["type"] = "GMM samples"

    sns.set_theme(style="white")
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 10))
    if n == 1:
        axes = axes.reshape(2, 1)

    x_min, x_max = rep_pca[:, 0].min(), rep_pca[:, 0].max()
    y_min, y_max = rep_pca[:, 1].min(), rep_pca[:, 1].max()

    MARKER_MAP = {"TCGA": "D", "GTEx": "s", "SC": "o", "R2": "X"}

    for i, batch in enumerate(unique_batches):
        ax1 = axes[0, i]
        df_rep = df[df["batch_label"] == batch]
        sns.scatterplot(data=df_rep, x="PC1", y="PC2", hue="type", s=3, alpha=0.8,
                        ax=ax1, palette={"Representation": "steelblue"})
        sns.scatterplot(data=df_samples, x="PC1", y="PC2", ax=ax1,
                        hue="type", palette={"GMM samples": "orange"}, s=3, legend=False, alpha=0.8)
        sns.scatterplot(data=df_means,   x="PC1", y="PC2", ax=ax1,
                        hue="type", palette={"GMM means": "black"},   s=12, legend=False, alpha=0.8)
        ax1.set_title(f"{batch} - Types")
        ax1.set_xlim(x_min, x_max); ax1.set_ylim(y_min, y_max)
        l1 = ax1.legend(loc="upper left", fontsize="small")
        if l1: l1.set_title("")
        ax1.set_xlabel(""); ax1.set_ylabel("")

        ax2 = axes[1, i]
        df_batch = df[df["batch_label"] == batch]
        markers = {b: MARKER_MAP.get(b, "o") for b in df_batch["batch_label"].unique()}
        sns.scatterplot(data=df_batch, x="PC1", y="PC2",
                        hue="tissue_label", style="batch_label", markers=markers,
                        s=5, alpha=0.8, ax=ax2, palette=color_mapping)
        ax2.set_title(f"{batch} - Tissues")
        ax2.set_xlim(x_min, x_max); ax2.set_ylim(y_min, y_max)
        ax2.set_xlabel(""); ax2.set_ylabel("")
        if ax2.get_legend(): ax2.get_legend().remove()

    fig.supxlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=14)
    fig.supylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=14)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    if filename:
        os.makedirs(savedir or ".", exist_ok=True)
        plt.savefig(os.path.join(savedir or ".", filename), bbox_inches="tight", dpi=150)
    plt.show()
    return pca


def plot_latent_space_multi_grid(
    train_vals, val_vals, test_vals,
    train_labels, val_labels, test_labels,
    train_batch, val_batch, test_batch,
    color_mapping, savedir=None, filename=None,
):
    from sklearn.decomposition import PCA
    tr_rep, tr_means, tr_samples = train_vals
    va_rep, va_means, va_samples = val_vals
    te_rep, te_means, te_samples = test_vals

    combined_rep = np.concatenate([tr_rep, va_rep, te_rep], axis=0)
    pca = PCA(n_components=2)
    pca.fit(combined_rep)

    tr_rep_pca = pca.transform(tr_rep); tr_means_pca = pca.transform(tr_means); tr_samples_pca = pca.transform(tr_samples)
    va_rep_pca = pca.transform(va_rep); va_means_pca = pca.transform(va_means); va_samples_pca = pca.transform(va_samples)
    te_rep_pca = pca.transform(te_rep); te_means_pca = pca.transform(te_means); te_samples_pca = pca.transform(te_samples)

    all_pc1 = np.concatenate([tr_rep_pca[:,0], va_rep_pca[:,0], te_rep_pca[:,0]])
    all_pc2 = np.concatenate([tr_rep_pca[:,1], va_rep_pca[:,1], te_rep_pca[:,1]])
    margin = 0.05
    x_range = all_pc1.max() - all_pc1.min(); y_range = all_pc2.max() - all_pc2.min()
    x_lim = (all_pc1.min() - margin*x_range, all_pc1.max() + margin*x_range)
    y_lim = (all_pc2.min() - margin*y_range, all_pc2.max() + margin*y_range)

    def make_df(rep_pca, means_pca, samples_pca, tissue_labels, batch_labels):
        df = pd.DataFrame(rep_pca, columns=["PC1", "PC2"])
        df["type"] = "Representation"; df["tissue_label"] = tissue_labels; df["batch_label"] = batch_labels
        df_s = pd.DataFrame(samples_pca, columns=["PC1", "PC2"])
        df_s["type"] = "GMM samples"; df_s["tissue_label"] = np.nan; df_s["batch_label"] = np.nan
        df_m = pd.DataFrame(means_pca, columns=["PC1", "PC2"])
        df_m["type"] = "GMM means"; df_m["tissue_label"] = np.nan; df_m["batch_label"] = np.nan
        return pd.concat([df, df_s, df_m])

    dfs = [
        make_df(tr_rep_pca, tr_means_pca, tr_samples_pca, train_labels, train_batch),
        make_df(va_rep_pca, va_means_pca, va_samples_pca, val_labels,   val_batch),
        make_df(te_rep_pca, te_means_pca, te_samples_pca, test_labels,  test_batch),
    ]
    names = ["Train", "Val", "Test"]

    MARKER_MAP = {"TCGA": "D", "GTEx": "s", "SC": "o", "R2": "X"}

    plt.rcParams.update({"font.size": 9})
    sns.set_theme(style="white")
    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    fig.subplots_adjust(wspace=0.2, hspace=0.25)

    for j, (df, name) in enumerate(zip(dfs, names)):
        ax1 = axes[0, j]
        sns.scatterplot(data=df, x="PC1", y="PC2", hue="type", size="type",
                        sizes=[3, 3, 12], alpha=0.8, ax=ax1,
                        palette=["steelblue", "orange", "black"])
        ax1.set_title(f"{name} Latent Space (by type)", fontsize=15)
        ax1.set_xlim(*x_lim); ax1.set_ylim(*y_lim)
        ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)" if j == 1 else "")
        ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)" if j == 0 else "")
        if j == 0:
            l = ax1.legend(loc="upper left"); l.set_title("") if l else None
        else:
            ax1.get_legend().remove()

    legend_ax = axes[1, 1]
    for j, (df, name) in enumerate(zip(dfs, names)):
        ax2 = axes[1, j]
        df_rep = df[df["type"] == "Representation"].copy()
        markers = {b: MARKER_MAP.get(b, "o") for b in df_rep["batch_label"].dropna().unique()}
        sns.scatterplot(data=df_rep, x="PC1", y="PC2",
                        hue="tissue_label", style="batch_label", markers=markers,
                        s=5, alpha=0.8, ax=ax2, palette=color_mapping, legend=True)
        ax2.set_title(f"{name} Latent Space (by label)", fontsize=15)
        ax2.set_xlim(*x_lim); ax2.set_ylim(*y_lim)
        ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax2.set_ylabel("")
        ax2.get_legend().remove()

    handles, leg_labels = legend_ax.get_legend_handles_labels()
    fig.legend(handles, leg_labels, title="Tissue", loc="lower center",
               bbox_to_anchor=(0.5, -0.08), ncol=8, markerscale=2, fontsize="medium")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if filename:
        os.makedirs(savedir or ".", exist_ok=True)
        plt.savefig(os.path.join(savedir or ".", filename), bbox_inches="tight", dpi=150)
    plt.show()


# ── Correlation vs mean expression ──────────────────────────────────────────

def plot_corr_vs_mean_expression_pub(
    y_test, corr_data, corr_type="spearman",
    savedir=None, filename=None, log_y=True,
):
    if corr_type not in ["pearson", "spearman"]:
        raise ValueError("corr_type must be 'pearson' or 'spearman'")
    sns.set_style("white")
    plt.rcParams.update({
        "font.family": "sans-serif", "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })
    mean_expr = y_test.iloc[:, :-4].mean(axis=0).reset_index()
    mean_expr.columns = ["mirna", "mean_expression"]
    plot_df = corr_data.merge(mean_expr, on="mirna", how="inner")
    if log_y:
        plot_df = plot_df[plot_df["mean_expression"] > 0]

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    sns.scatterplot(data=plot_df, x=corr_type, y="mean_expression",
                    s=8, alpha=0.5, edgecolor="none", ax=ax)
    if log_y:
        ax.set_yscale("log")
    ax.set_xlim(0, 1)
    ax.set_xlabel(f"{corr_type.capitalize()} correlation")
    ax.set_ylabel("Mean miRNA expression")
    ax.tick_params(direction="out", length=3, width=0.6)
    sns.despine(ax=ax)
    fig.tight_layout()
    if filename:
        os.makedirs(savedir or ".", exist_ok=True)
        fig.savefig(os.path.join(savedir or ".", filename))
    plt.show()
    plt.close(fig)
    return plot_df


# ── Predicted vs observed regression for a single miRNA ─────────────────────

def regression_plot(
    X_test, y_test, mir, cancer_type=None,
    savedir=None, filename=None,
):
    from scipy import stats
    from matplotlib.ticker import ScalarFormatter, MaxNLocator

    X = X_test[mir] if mir in X_test.columns else X_test.iloc[:, X_test.columns.get_loc(mir)]
    y = y_test[mir] if mir in y_test.columns else y_test.iloc[:, y_test.columns.get_loc(mir)]
    data = pd.DataFrame({"X": X, "y": y})

    if cancer_type is not None and "cancer_type" in X_test.columns:
        mask = X_test["cancer_type"] == cancer_type
        data = data[mask.values]

    data = data[(data["X"] > 0) & (data["y"] > 0)]
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    sns.regplot(data=data, x="X", y="y",
                scatter_kws={"s": 15, "alpha": 0.7, "edgecolor": "none", "color": "black"},
                line_kws={"color": "royalblue", "alpha": 0.5}, ax=ax)
    sns.despine(ax=ax)

    sp_r, sp_p = stats.spearmanr(data["X"], data["y"])
    pe_r, pe_p = stats.pearsonr(data["X"], data["y"])
    p_fmt = lambda p: "< 2.2e-8" if p < 2.2e-8 else f"= {p:.2f}"
    ax.text(0.05, 1.00, rf"$r_s$ = {sp_r:.2f}, p {p_fmt(sp_p)}", transform=ax.transAxes, fontsize=11, va="top")
    ax.text(0.05, 0.93, rf"$r$ = {pe_r:.2f}, p {p_fmt(pe_p)}",   transform=ax.transAxes, fontsize=11, va="top")

    ax.set_xlabel(f"{mir} predicted expression", fontsize=12)
    ax.set_ylabel(f"{mir} true expression", fontsize=12)
    for axis, fmt in [(ax.xaxis, ScalarFormatter(useMathText=True)), (ax.yaxis, ScalarFormatter(useMathText=True))]:
        fmt.set_powerlimits((3, 4)); axis.set_major_formatter(fmt)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    min_v = min(data["X"].min(), data["y"].min()); max_v = max(data["X"].max(), data["y"].max())
    ax.plot([min_v, max_v], [min_v, max_v], linestyle="dashed", linewidth=0.5, color="black")
    plt.tight_layout()
    if filename:
        os.makedirs(savedir or ".", exist_ok=True)
        plt.savefig(os.path.join(savedir or ".", filename), bbox_inches="tight")
    plt.show()


# ── Jitter (box+strip) plot per cancer type for a single miRNA ───────────────

def plot_jitter_grouped_tissue(
    X_test, y_test, mir,
    savedir=None, filename=None,
):
    from sklearn.metrics import root_mean_squared_error
    from matplotlib.ticker import ScalarFormatter, MaxNLocator
    from scipy import stats

    x_col = X_test[mir].values if mir in X_test.columns else None
    y_col = y_test[mir].values if mir in y_test.columns else None
    cancer = X_test["cancer_type"].values if "cancer_type" in X_test.columns else y_test["cancer_type"].values

    data = pd.DataFrame({"X": x_col, "y": y_col, "cancer_type": cancer})
    cat_order = sorted(data["cancer_type"].unique())
    data["cancer_type"] = pd.Categorical(data["cancer_type"], categories=cat_order, ordered=True)
    palette = {ct: X_test.loc[X_test["cancer_type"] == ct, "color"].iloc[0]
               if "color" in X_test.columns else "#888888" for ct in cat_order}

    rmse = root_mean_squared_error(data["y"], data["X"])
    sns.set_style("white")
    fig, axes = plt.subplots(2, 1, figsize=(7, 4), sharey=False)
    plt.subplots_adjust(hspace=0.2)

    for ax, col, label in [(axes[0], "y", "True"), (axes[1], "X", "Prediction")]:
        sns.boxplot(data=data, x="cancer_type", y=col, hue="cancer_type",
                    order=cat_order, hue_order=cat_order,
                    boxprops=dict(alpha=0.3), showfliers=False, ax=ax, palette=palette, legend=False)
        sns.stripplot(data=data, x="cancer_type", y=col, hue="cancer_type",
                      order=cat_order, hue_order=cat_order,
                      alpha=0.8, zorder=0, ax=ax, palette=palette, dodge=False, legend=False)
        ax.set_ylabel(label, fontsize=10)
        fmt = ScalarFormatter(useMathText=True); fmt.set_powerlimits((3, 4))
        ax.yaxis.set_major_formatter(fmt); ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.get_offset_text().set_fontsize(10)
        if ax is axes[0]:
            ax.set_xlabel(""); ax.set_xticks([])
            ax.text(1., 1., f"RMSE: {rmse:.2f}", ha="right", va="top",
                    transform=ax.transAxes, fontsize=9, bbox=dict(facecolor="white", alpha=0.5))
        else:
            ax.set_xlabel("Cancer Type", fontsize=12)
            ax.tick_params(axis="x", rotation=90, labelsize=11)
            for lbl, ct in zip(ax.get_xticklabels(), cat_order):
                lbl.set_color(palette.get(ct, "#000000"))

    fig.suptitle(mir, fontsize=12, y=0.99)
    fig.supylabel("Expression count", fontsize=11, x=0.04)
    sns.despine()
    plt.tight_layout()
    if filename:
        os.makedirs(savedir or ".", exist_ok=True)
        plt.savefig(os.path.join(savedir or ".", filename), bbox_inches="tight")
    plt.show()


# ── miRNA (predicted) vs mRNA (predicted) regression ─────────────────────────

def mirna_vs_mrna_regression(
    X_mirna, X_mrna, mir, mrna_gene,
    cancer_type=None, anno=None,
    ylabel=None, savedir=None, filename=None,
):
    from scipy import stats
    from matplotlib.ticker import ScalarFormatter, MaxNLocator

    x = X_mirna[mir].astype(float) if mir in X_mirna.columns else pd.Series(dtype=float)
    y = X_mrna[mrna_gene].astype(float) if mrna_gene in X_mrna.columns else pd.Series(dtype=float)

    if cancer_type is not None and anno is not None:
        idx = anno[anno["cancer_type"] == cancer_type].index
        x = x.loc[x.index.intersection(idx)]
        y = y.loc[y.index.intersection(idx)]

    mask = pd.notnull(x) & pd.notnull(y) & (x > 0) & (y > 0)
    x, y = x[mask], y[mask]

    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    sns.regplot(x=x, y=y,
                scatter_kws={"s": 15, "alpha": 0.7, "edgecolor": "none", "color": "black"},
                line_kws={"color": "royalblue", "alpha": 0.5}, ax=ax)
    sns.despine(ax=ax)

    sp_r, sp_p = stats.spearmanr(x, y)
    pe_r, pe_p = stats.pearsonr(x, y)
    p_fmt = lambda p: "< 2.2e-3" if p < 2.2e-3 else f"= {p:.2f}"
    ax.text(0.05, 1.00, rf"$\rho$ = {sp_r:.2f}, p {p_fmt(sp_p)}", transform=ax.transAxes, fontsize=10, va="top")
    ax.text(0.05, 0.95, rf"$r$ = {pe_r:.2f}, p {p_fmt(pe_p)}",    transform=ax.transAxes, fontsize=10, va="top")

    ax.set_xlabel(f"{mir} predicted expression", fontsize=11)
    ax.set_ylabel(f"{ylabel or mrna_gene} predicted expression", fontsize=11)
    for axis, fmt in [(ax.xaxis, ScalarFormatter(useMathText=True)), (ax.yaxis, ScalarFormatter(useMathText=True))]:
        fmt.set_powerlimits((-5, 6)); axis.set_major_formatter(fmt)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    plt.tight_layout()
    if filename:
        os.makedirs(savedir or ".", exist_ok=True)
        plt.savefig(os.path.join(savedir or ".", filename), bbox_inches="tight")
    plt.show()
