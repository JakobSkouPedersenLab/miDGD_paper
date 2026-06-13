import torch
from sklearn.model_selection import StratifiedKFold
from base.data.combined import GeneExpressionDatasetCombined
from base.dgd.DGD import DGD
from base.engine.train import train_midgd, train_midgd_all

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cross_validate(mrna_data, mirna_data, batch_size, num_workers, 
                   decoder, n_tissues, latent_dim, gmm_spec,
                   learning_rates, weight_decay, betas, reduction_type, scaling_type, nepochs,
                   pr, plot, sample_index, subset, is_plot=False, n_splits=5, 
                   mode="all", wandb_log=True, seed=42, early_stopping=False):

    # Create a KFold object with the desired number of splits
    skf = StratifiedKFold(n_splits=n_splits)

    # Initialize lists to store DGD model and performance metrics for each fold
    dgd_cv = {}
    loss_cv = {}

    # Iterate over the folds
    for fold, (train_idx, val_idx) in enumerate(skf.split(mrna_data, mrna_data['cancer_type'])):
        print(f"Fold: {fold+1}...")

        # Split the data into training and validation sets
        train_mrna = mrna_data.iloc[train_idx]
        train_mirna = mirna_data.iloc[train_idx]
        val_mrna = mrna_data.iloc[val_idx]
        val_mirna = mirna_data.iloc[val_idx]

        # Create the datasets and data loaders
        train_dataset = GeneExpressionDatasetCombined(train_mrna, train_mirna, scaling_type=scaling_type)
        val_dataset = GeneExpressionDatasetCombined(val_mrna, val_mirna, scaling_type=scaling_type)

        train_loader = torch.utils.data.DataLoader(train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True,
                                           num_workers=num_workers)
        validation_loader = torch.utils.data.DataLoader(val_dataset, 
                                                        batch_size=batch_size, 
                                                        shuffle=False,
                                                        num_workers=num_workers)
        # Initialize the model, optimizer, and other components
        dgd = DGD(decoder=decoder, 
                  n_mix=n_tissues, 
                  rep_dim=latent_dim,
                  gmm_spec=gmm_spec
            )

        # Train the model
        if mode == "midgd":
            loss_tab = train_midgd(
                dgd, train_loader, validation_loader, device,
                learning_rates=learning_rates, weight_decay=weight_decay, betas=betas, 
                nepochs=nepochs, fold=fold, pr=pr, plot=plot, reduction_type=reduction_type,
                sample_index=sample_index, subset=subset, wandb_log=wandb_log, early_stopping=early_stopping, is_plot=is_plot)
        elif mode == "all":
            loss_tab = train_midgd_all(
                dgd, train_loader, validation_loader, device,
                learning_rates=learning_rates, weight_decay=weight_decay, betas=betas, 
                nepochs=nepochs, fold=fold, pr=pr, plot=plot, reduction_type=reduction_type,
                sample_index=sample_index, subset=subset, wandb_log=wandb_log, early_stopping=early_stopping, is_plot=is_plot)
        
        # Store the model for this fold
        dgd_cv[fold] = dgd

        # Store the performance metrics for this fold
        loss_cv[fold] = loss_tab

        print(f"Fold {fold+1} done...")

    return dgd_cv, loss_cv