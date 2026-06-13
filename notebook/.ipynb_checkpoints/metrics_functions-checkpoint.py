from sklearn.metrics.cluster import rand_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import scanpy as sc

import pandas as pd

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