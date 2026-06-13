import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch


def _device_for_model(dgd):
    try:
        return next(dgd.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gmm_clustering(dgd, rep):
    clustering = []
    device = _device_for_model(dgd)
    for i in range(len(rep.z)):
        cluster = dgd.to(device).gmm.clustering(rep(i)).unsqueeze(0).detach().cpu().numpy()
        clustering.extend(cluster)
    return np.array(clustering)


def gmm_clustering_plot_diagonal(
    cluster,
    data_loader,
    tissue_mapping,
    axis="comp",
    is_save=False,
    savedir="plots",
    filename=None,
    label_attr="primary_site",
    figsize=(20, 30),
):
    labels = getattr(data_loader.dataset, label_attr)
    gmm_comp = pd.DataFrame({"component": cluster, label_attr: labels})
    df = pd.crosstab(gmm_comp[label_attr], gmm_comp["component"])
    if axis == "comp":
        df = df.div(df.sum(axis=0))
    elif axis in {"cancer", "tissue"}:
        df = df.div(df.sum(axis=1), axis=0)

    dominant_label_per_component = df.idxmax(axis=0)
    sorting_df = pd.DataFrame(
        {
            "component": df.columns,
            "dominant_label": dominant_label_per_component,
            "max_value": [df[comp].max() for comp in df.columns],
        }
    )

    sorted_groups = []
    for label, group in sorting_df.groupby("dominant_label"):
        sorted_groups.append((label, group, group["max_value"].max()))
    sorted_groups.sort(key=lambda item: -item[2])

    component_order = []
    for _, group, _ in sorted_groups:
        component_order.extend(group.sort_values("max_value", ascending=False)["component"])

    label_component_pairs = []
    for label in df.index:
        max_component = df.loc[label].idxmax()
        max_value = df.loc[label, max_component]
        position = (
            component_order.index(max_component)
            if max_component in component_order
            else float("inf")
        )
        label_component_pairs.append((label, position, max_value))
    label_component_pairs.sort(key=lambda item: (item[1], -item[2]))
    label_order = [item[0] for item in label_component_pairs]

    df_ordered = df.loc[label_order, component_order].T
    label_counts = pd.Series(labels).value_counts()
    component_counts = pd.Series(cluster).value_counts()
    y_labels = [f"{comp} ({component_counts.get(comp, 0)})" for comp in component_order]
    x_labels = [f"{label} ({label_counts[label]})" for label in label_order]

    sns.set_style(style="white")
    _, ax = plt.subplots(figsize=figsize)
    annot = df_ordered.map(lambda value: f"{value:.2f}" if value >= 1e-1 else "")
    cmap = mpl.colormaps.get_cmap("Blues")
    cmap.set_bad("white")
    norm = mpl.colors.Normalize(vmin=0, vmax=1)

    heatmap = sns.heatmap(
        df_ordered,
        annot=annot,
        mask=False,
        cmap=cmap,
        norm=norm,
        fmt="",
        linewidth=0.8,
        annot_kws={"size": 9.5},
        cbar_kws={
            "shrink": 0.8,
            "label": "Sample Proportion per Cancer Type",
            "pad": 0.01,
        },
        ax=ax,
        cbar=False,
    )
    heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0)
    ax.set_xlabel("Cancer Type (#samples)", fontsize=18)
    ax.set_ylabel("GMM Components (#samples)", fontsize=18)
    ax.set_xticklabels(x_labels, fontsize=12, rotation=90)
    ax.set_yticklabels(y_labels, fontsize=10)

    for xtick, label in zip(ax.get_xticklabels(), df_ordered.columns):
        xtick.set_color(tissue_mapping.get(label, "black"))

    plt.tight_layout()
    if filename is not None:
        os.makedirs(savedir, exist_ok=True)
        plt.savefig(os.path.join(savedir, filename), bbox_inches="tight", dpi=300)
    plt.show()


def build_component_to_primary_mapping(dgd, train_loader):
    z_train = dgd.train_rep.z.detach().cpu()
    device = _device_for_model(dgd)
    with torch.no_grad():
        comp_train = (
            dgd.gmm.clustering(z_train.to(device)).detach().cpu().numpy().astype(int)
        )

    primary_site = np.array(train_loader.dataset.primary_site)
    df = pd.DataFrame({"component": comp_train, "primary_site": primary_site})
    comp_primary_counts = (
        df.groupby(["component", "primary_site"], observed=False)
        .size()
        .reset_index(name="count")
    )
    majority_rows = comp_primary_counts.loc[
        comp_primary_counts.groupby("component", observed=False)["count"].idxmax()
    ]
    component_to_primary = dict(
        zip(majority_rows["component"], majority_rows["primary_site"])
    )
    return component_to_primary, comp_primary_counts


def assign_pscsr_to_tissue(dgd, pscsr_loader, component_to_primary, rep_attr="pscsr_rep"):
    z_pscsr = getattr(dgd, rep_attr).z.detach().cpu()
    device = _device_for_model(dgd)
    with torch.no_grad():
        comp_pscsr = (
            dgd.gmm.clustering(z_pscsr.to(device)).detach().cpu().numpy().astype(int)
        )

    cell_lines = np.array(pscsr_loader.dataset.str_labels)
    closest_primary = np.array(
        [component_to_primary.get(component, "Unknown") for component in comp_pscsr]
    )
    return pd.DataFrame(
        {
            "component": comp_pscsr,
            "closest_primary": closest_primary,
            "cell_line": cell_lines,
        }
    )


def gmm_clustering_plot_diagonal_new(
    cluster,
    data_loader,
    tissue_mapping,
    axis="comp",
    savedir="plots",
    filename=None,
    label_attr="primary_site",
    figsize=(30, 20),
):
    labels = getattr(data_loader.dataset, label_attr)
    gmm_comp = pd.DataFrame({"component": cluster, label_attr: labels})
    df = pd.crosstab(gmm_comp[label_attr], gmm_comp["component"])

    if axis == "comp":
        df = df.div(df.sum(axis=0))
    elif axis in {"cancer", "tissue"}:
        df = df.div(df.sum(axis=1), axis=0)

    dominant_comp_per_label = df.idxmax(axis=1)
    sorting_df = pd.DataFrame(
        {
            "label": df.index,
            "dominant_comp": dominant_comp_per_label,
            "max_value": [df.loc[lbl].max() for lbl in df.index],
        }
    )

    sorted_groups = []
    for comp, group in sorting_df.groupby("dominant_comp"):
        sorted_groups.append((comp, group, group["max_value"].max()))
    sorted_groups.sort(key=lambda item: -item[2])

    label_order = []
    for _, group, _ in sorted_groups:
        label_order.extend(group.sort_values("max_value", ascending=False)["label"])

    comp_label_pairs = []
    for comp in df.columns:
        max_label = df[comp].idxmax()
        max_value = df.loc[max_label, comp]
        position = label_order.index(max_label) if max_label in label_order else float("inf")
        comp_label_pairs.append((comp, position, max_value))
    comp_label_pairs.sort(key=lambda item: (item[1], -item[2]))
    component_order = [item[0] for item in comp_label_pairs]

    df_ordered = df.loc[label_order, component_order]

    label_counts   = pd.Series(list(labels)).value_counts()
    component_counts = pd.Series(list(cluster)).value_counts()

    y_labels = [f"{lbl} ({label_counts.get(lbl, 0)})"    for lbl  in label_order]
    x_labels = [f"{comp} ({component_counts.get(comp, 0)})" for comp in component_order]

    sns.set_style(style="white")
    _, ax = plt.subplots(figsize=figsize)
    annot = df_ordered.map(lambda value: f"{value:.2f}" if value >= 1e-1 else "")
    cmap = mpl.colormaps.get_cmap("Blues")
    cmap.set_bad("white")
    norm = mpl.colors.Normalize(vmin=0, vmax=1)

    heatmap = sns.heatmap(
        df_ordered,
        annot=annot,
        mask=False,
        cmap=cmap,
        norm=norm,
        fmt="",
        linewidth=0.8,
        annot_kws={"size": 9.5},
        cbar_kws={"shrink": 0.8, "pad": 0.01},
        ax=ax,
        cbar=False,
    )
    heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0)
    ax.set_xlabel("GMM Components (#samples)", fontsize=18)
    ax.set_ylabel("Cancer / Tissue Type (#samples)", fontsize=18)
    ax.set_xticklabels(x_labels, fontsize=10, rotation=90)
    ax.set_yticklabels(y_labels, fontsize=12)

    for ytick, label in zip(ax.get_yticklabels(), label_order):
        ytick.set_color(tissue_mapping.get(label, "black"))

    plt.tight_layout()
    if filename is not None:
        os.makedirs(savedir, exist_ok=True)
        plt.savefig(os.path.join(savedir, filename), bbox_inches="tight", dpi=300)
    plt.show()
