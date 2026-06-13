import re

import pandas as pd
import torch


def _device_for_model(dgd):
    try:
        return next(dgd.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _add_columns(X, y, data_loader):
    dataset = data_loader.dataset
    for name, attr in (
        ("tissue", "tissue_type"),
        ("sample", "primary_site"),
        ("color", "color"),
        ("cancer_type", "cancer_type"),
    ):
        if hasattr(dataset, attr):
            values = getattr(dataset, attr)
            X = X.assign(**{name: values})
            y = y.assign(**{name: values})
    return X, y


def _get_data_prediction(dgd, data_loader, mirna_column_name, dataset="train"):
    device = _device_for_model(dgd)
    with torch.inference_mode():
        scaling = torch.sum(data_loader.dataset.mirna_data, axis=1)
        if dataset == "train":
            X, _ = dgd.forward(dgd.train_rep())
        elif dataset == "val":
            X, _ = dgd.forward(dgd.val_rep())
        elif dataset == "test":
            X, _ = dgd.forward(dgd.test_rep())
        else:
            raise ValueError("dataset must be one of 'train', 'val', or 'test'.")

        X = X * scaling.unsqueeze(1).to(device)
        X = pd.DataFrame(X.detach().cpu().numpy(), columns=mirna_column_name)
        y = pd.DataFrame(
            data_loader.dataset.mirna_data.detach().cpu().numpy(),
            columns=mirna_column_name,
        )
        return _add_columns(X, y, data_loader)


def _get_data_pred_from_rep(dgd, test_rep, data_loader, mirna_column_name):
    device = _device_for_model(dgd)
    with torch.inference_mode():
        scaling = torch.nansum(data_loader.dataset.mirna_data, axis=1)
        X, _ = dgd.forward(test_rep.z)
        X = X * scaling.unsqueeze(1).to(device)
        X = pd.DataFrame(X.detach().cpu().numpy(), columns=mirna_column_name)
        y = pd.DataFrame(
            data_loader.dataset.mirna_data.detach().cpu().numpy(),
            columns=mirna_column_name,
        )
        return _add_columns(X, y, data_loader)


def _subset_frame(X, y, data_loader, subset, label_attr="cancer_type", include_batch=None):
    dataset = data_loader.dataset
    values = {
        "X": X[subset],
        "y": y[subset],
        "cancer_type": getattr(dataset, label_attr),
        "tissue": dataset.tissue_type,
        "color": dataset.color,
    }
    if include_batch is None:
        include_batch = hasattr(dataset, "batch")
    if include_batch:
        values["batch"] = dataset.batch
    return pd.DataFrame(data=values)


def generate_analysis_data(
    dgd,
    train_loader,
    validation_loader,
    test_loader,
    mirna_column_name,
    subset=False,
    dataset="test",
    label_attr="cancer_type",
    include_batch=None,
):
    loaders = {
        "train": train_loader,
        "val": validation_loader,
        "validation": validation_loader,
        "test": test_loader,
    }
    if subset:
        if dataset not in loaders:
            raise ValueError("dataset must be one of 'train', 'val', or 'test'.")
        loader = loaders[dataset]
        split_name = "val" if dataset == "validation" else dataset
        X, y = _get_data_prediction(dgd, loader, mirna_column_name, dataset=split_name)
        return _subset_frame(
            X,
            y,
            loader,
            subset,
            label_attr=label_attr,
            include_batch=include_batch,
        )

    X_train, y_train = _get_data_prediction(
        dgd, train_loader, mirna_column_name, dataset="train"
    )
    X_val, y_val = _get_data_prediction(
        dgd, validation_loader, mirna_column_name, dataset="val"
    )
    X_test, y_test = _get_data_prediction(
        dgd, test_loader, mirna_column_name, dataset="test"
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def get_mirna_data(
    dgd,
    data_loader,
    subset,
    mirna_column_name,
    dataset="test",
    label_attr="cancer_type",
    include_batch=None,
):
    return generate_analysis_data(
        dgd,
        *data_loader,
        mirna_column_name,
        subset=subset,
        dataset=dataset,
        label_attr=label_attr,
        include_batch=include_batch,
    )


def _get_data_pred_from_rep_with_mrna(
    dgd, test_rep, data_loader,
    mirna_column_name, mrna_column_name,
):
    device = _device_for_model(dgd)
    with torch.inference_mode():
        scaling_mirna = torch.nansum(data_loader.dataset.mirna_data, axis=1)
        X_mirna, X_mrna = dgd.forward(test_rep.z)
        X_mirna = X_mirna * scaling_mirna.unsqueeze(1).to(device)
        X_mirna = pd.DataFrame(X_mirna.detach().cpu().numpy(), columns=mirna_column_name)
        X_mrna  = pd.DataFrame(X_mrna.detach().cpu().numpy(),  columns=mrna_column_name)
        y_mirna = pd.DataFrame(
            data_loader.dataset.mirna_data.detach().cpu().numpy(),
            columns=mirna_column_name,
        )
        y_mrna = pd.DataFrame(
            data_loader.dataset.mrna_data.detach().cpu().numpy(),
            columns=mrna_column_name,
        )
        X_mirna, y_mirna = _add_columns(X_mirna, y_mirna, data_loader)
        X_mrna,  y_mrna  = _add_columns(X_mrna,  y_mrna,  data_loader)
    return X_mirna, y_mirna, X_mrna, y_mrna


def replace_tissue_anno(df):
    df = df.copy()
    replacements = {
        "Cervix Uteri": "Cervix",
        "Colon": "Colorectal",
        "EBV-transformed lymphocytes": "LCL cell",
    }
    df["primary_site"] = df["primary_site"].replace(replacements)
    df["cancer_type"] = df["cancer_type"].replace(replacements)
    return df


def collapse_mirna_df(df: pd.DataFrame, mirna_anno: pd.DataFrame) -> pd.DataFrame:
    def normalize_primary(name: str) -> str:
        if name.count("-") > 2 and re.search(r"-\d+$", name):
            return re.sub(r"-\d+$", "", name)
        return name

    mature_to_baseprimary = {}
    for _, row in mirna_anno.iterrows():
        base_primary = normalize_primary(row["primary_name"])
        for arm in ("3p", "5p", "NA"):
            mature_col = f"mature_mir_{arm}_name"
            if pd.notna(row.get(mature_col, None)):
                mature_name = re.sub(r"_\d+$", "", str(row[mature_col])).lower()
                mature_to_baseprimary.setdefault(mature_name, set()).add(base_primary)

    baseprimary_to_columns = {}
    for col in df.columns:
        base_col = re.sub(r"_\d+$", "", col).lower()
        if base_col in mature_to_baseprimary:
            for base_primary in mature_to_baseprimary[base_col]:
                baseprimary_to_columns.setdefault(base_primary, []).append(col)

    collapsed = {
        base_primary: df[cols].sum(axis=1)
        for base_primary, cols in baseprimary_to_columns.items()
    }
    return pd.DataFrame(collapsed, index=df.index)


def filter_zeros(df, thres=0.9):
    zero_counts = (df == 0).mean()
    selected_features = zero_counts[zero_counts < thres].index
    return df[selected_features]


def filter_tumor(df, anno, tissue_col="cancer_type", tumor_col="sample_type"):
    tumor_ids = anno.index[anno[tumor_col].eq("Primary Tumor")]
    return df.loc[df.index.intersection(tumor_ids)]


def filter_variance(df, top_n=None):
    variances = df.var(axis=0).sort_values(ascending=False)
    if top_n is None:
        return df.loc[:, variances.index]
    return df.loc[:, variances.head(top_n).index]
