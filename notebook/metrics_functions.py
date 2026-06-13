from sklearn.metrics.cluster import rand_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from scipy import stats

import numpy as np
import pandas as pd
import statsmodels.api as sm


def calculate_corr(X, y, axis=0):
    X = X.iloc[:, :-4]
    y = y.iloc[:, :-4]
    spearman_corr, pearson_corr, r2 = [], [], []
    spearman_pval, pearson_pval, r2_pval = [], [], []

    if axis == 0:
        iter_range = y.shape[1]
        get = lambda df, i: df.iloc[:, i]
        names = list(X.columns)
    elif axis == 1:
        iter_range = y.shape[0]
        get = lambda df, i: df.iloc[i, :]
        names = list(X.index)
    else:
        raise ValueError("axis must be 0 or 1")

    for i in range(iter_range):
        xi, yi = get(X, i), get(y, i)
        sp, sp_p = stats.spearmanr(xi, yi)
        pe, pe_p = stats.pearsonr(xi, yi)
        _, _, r, r_p, _ = stats.linregress(xi, yi)
        spearman_corr.append(sp)
        pearson_corr.append(pe)
        r2.append(r ** 2)
        spearman_pval.append(sp_p)
        pearson_pval.append(pe_p)
        r2_pval.append(r_p)

    return pd.DataFrame(
        {
            "pearson": pearson_corr,
            "spearman": spearman_corr,
            "r2": r2,
            "pearson_p": pearson_pval,
            "spearman_p": spearman_pval,
            "r2_p": r2_pval,
            "mirna": names,
        }
    ).sort_values(by="pearson", ascending=False, ignore_index=True)

def calculate_ari(
    dgd,
    rep,
    data_loader,
    method='gmm',
    n_clusters=32,
    n_neighbors=15,
    split='train',          # 'train', 'val', or 'test'
):
    """
    Calculate ARI for a given split and clustering method.

    Parameters
    ----------
    dgd : object
        Your DGD model with a .gmm.clustering() method.
    rep : iterable
        Representation(s) to cluster with GMM (e.g. dgd.train_rep.z or dgd.test_rep.z).
    data_loader : tuple
        (train_loader, validation_loader, test_loader).
    method : str
        'gmm', 'kmeans', or 'leiden'.
    n_clusters : int
        Number of clusters for k-means.
    n_neighbors : int
        Number of neighbors for Leiden.
    split : str
        Which split to use: 'train', 'val', or 'test'.
    """

    train_loader, validation_loader, test_loader = data_loader

    if split == 'train':
        loader = train_loader
    elif split in ['val', 'validation']:
        loader = validation_loader
    elif split == 'test':
        loader = test_loader
    else:
        raise ValueError(f"Unknown split: {split}. Use 'train', 'val', or 'test'.")

    labels = loader.dataset.label
    mirna_data = loader.dataset.mrna_data.detach().cpu().numpy()
    df = pd.DataFrame(labels, columns=["label"])
    clusters = []

    if method == 'gmm':
        # rep should correspond to the same split as `loader`
        for i in rep:
            cluster = (
                dgd.gmm.clustering(i)
                .unsqueeze(0)
                .detach()
                .cpu()
                .numpy()
            )
            clusters.extend(cluster)

        df["cluster"] = clusters
        df["cluster"] = df["cluster"].astype('category')

        # majority-vote label per cluster (if you actually want cluster→label mapping)
        label_counts = (
            df.groupby(['cluster', 'label'], observed=False)
              .size()
              .reset_index(name='counts')
        )
        most_frequent_labels = label_counts.loc[
            label_counts.groupby('cluster', observed=False)['counts'].idxmax()
        ]
        cluster_to_label = dict(
            zip(most_frequent_labels['cluster'], most_frequent_labels['label'])
        )
        # If you want **cluster IDs**, use df["cluster"] below.
        # If you want **majority labels**, use df["cluster"].map(cluster_to_label).
        df['cluster_name'] = df['cluster']  # raw cluster IDs

    elif method == 'kmeans':
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(mirna_data)
        clusters = kmeans.labels_
        df['cluster_name'] = clusters

    elif method == 'leiden':
        import scanpy as sc

        adata = sc.AnnData(mirna_data)
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X')
        sc.tl.leiden(
            adata,
            resolution=1,
            key_added='cluster',
            flavor="igraph",
            n_iterations=2,
        )
        clusters = adata.obs['cluster'].values.astype(int)
        df['cluster_name'] = clusters

    else:
        raise ValueError(f"Unknown method: {method}")

    return adjusted_rand_score(df['label'], df['cluster_name'])


def calculate_poisson_regression_stats(observed, predicted):
    poisson_models = []
    for gene_idx in range(observed.shape[1]):
        y = observed.iloc[:, gene_idx].values
        X = sm.add_constant(predicted.iloc[:, gene_idx].values)
        if not (np.isfinite(y).all() and np.isfinite(X).all()):
            print(f"Non-finite values for gene {gene_idx}, skipping.")
            poisson_models.append(None)
            continue
        try:
            poisson_model = sm.GLM(y, X, family=sm.families.Poisson())
            poisson_results = poisson_model.fit()
            poisson_models.append(poisson_results)
        except Exception as exc:
            print(f"Error for gene {gene_idx}: {exc}")
            poisson_models.append(None)
    return poisson_models


def calculate_gaussian_regression_stats(observed, predicted):
    gaussian_models = []
    for gene_idx in range(observed.shape[1]):
        y = observed.iloc[:, gene_idx].values
        X = sm.add_constant(predicted.iloc[:, gene_idx].values)
        if not (np.isfinite(y).all() and np.isfinite(X).all()):
            print(f"Non-finite values for gene {gene_idx}, skipping.")
            gaussian_models.append(None)
            continue
        try:
            gaussian_model = sm.GLM(y, X, family=sm.families.Gaussian())
            gaussian_results = gaussian_model.fit()
            gaussian_models.append(gaussian_results)
        except Exception as exc:
            print(f"Error for gene {gene_idx}: {exc}")
            gaussian_models.append(None)
    return gaussian_models


def poisson_deviance(y, mu):
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    mu = np.clip(mu, 1e-12, None)
    term = np.where(y == 0, 0, y * np.log(np.clip(y, 1e-12, None) / mu))
    return 2 * np.sum(term - (y - mu))


def poisson_pseudo_r2_vector(y_true, y_pred):
    values = []
    for i in range(y_true.shape[1]):
        y = y_true.iloc[:, i].values
        mu = y_pred.iloc[:, i].values
        null_mu = np.repeat(np.mean(y), len(y))
        dev_model = poisson_deviance(y, mu)
        dev_null = poisson_deviance(y, null_mu)
        values.append(1 - dev_model / dev_null if dev_null != 0 else np.nan)
    return pd.DataFrame({"mirna": y_true.columns, "pseudo_r2": values})
