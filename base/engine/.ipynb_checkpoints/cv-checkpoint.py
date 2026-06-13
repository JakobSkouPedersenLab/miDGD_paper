import torch
import numpy as np

from sklearn.model_selection import KFold

from base.data.combined import GeneExpressionDatasetCombined
from base.dgd.latent import RepresentationLayer
from base.dgd.DGD import DGD
# from base.plotting.plot_cv import plot_mirna_recons, plot_mirna_recons_test
from base.plotting.plot_cv2 import plot_latent_space, plot_gene, plot_mirna
from torchmetrics.functional.regression import mean_squared_error, mean_absolute_error, r2_score, explained_variance, pearson_corrcoef, mean_squared_log_error, spearman_corrcoef

from tqdm import tqdm
import wandb


def cross_validate(mrna_data, mirna_data, decoder, n_tissues, latent_dim, gmm_spec,
                   device, num_workers, learning_rates, weight_decay, betas, sample_index, subset,
                   nepochs, batch_size, is_plot, pr, plot, reduction_type, n_splits=5, mode=0, wandb_log=False):
    
    # Create a KFold object with the desired number of splits
    kf = KFold(n_splits=n_splits)

    # Initialize lists to store DGD model and performance metrics for each fold
    dgd_cv = {}
    loss_cv = {}

    # Iterate over the folds
    for fold, (train_idx, val_idx) in enumerate(kf.split(mrna_data)):
        # Split the data into training and validation sets
        print(f"Fold: {fold+1}...")

        train_mrna = mrna_data.iloc[train_idx]
        train_mirna = mirna_data.iloc[train_idx]
        val_mrna = mrna_data.iloc[val_idx]
        val_mirna = mirna_data.iloc[val_idx]

        # Create the datasets and data loaders
        train_dataset = GeneExpressionDatasetCombined(train_mrna, train_mirna)
        val_dataset = GeneExpressionDatasetCombined(val_mrna, val_mirna)

        # Initialize the model, optimizer, and other components
        dgd = DGD(decoder=decoder, 
                  n_mix=n_tissues, 
                  rep_dim=latent_dim,
                  gmm_spec=gmm_spec
            )

        # Train the model
        if mode == 0:
            loss_tab = train_midgd_cv(dgd, train_dataset, val_dataset, device, num_workers, fold,
                                    learning_rates, weight_decay, betas, nepochs, batch_size, is_plot, pr, plot, reduction_type,
                                    sample_index, subset, wandb_log)
        elif mode == 1:
            loss_tab = train_midgd_cv_full(dgd, train_dataset, val_dataset, device, num_workers, fold, 
                                    learning_rates, weight_decay, betas, nepochs, batch_size, is_plot, pr, plot, reduction_type,
                                    sample_index, subset, wandb_log)
        elif mode == 2:
            loss_tab = train_midgd_cv_switch(dgd, train_dataset, val_dataset, device, num_workers, fold, 
                                    learning_rates, weight_decay, betas, nepochs, batch_size, is_plot, pr, plot, reduction_type,
                                    sample_index, subset, wandb_log)
        elif mode == 3:
            loss_tab = train_midgd_cv_full_switch(dgd, train_dataset, val_dataset, device, num_workers, fold, 
                                    learning_rates, weight_decay, betas, nepochs, batch_size, is_plot, pr, plot, reduction_type,
                                    sample_index, subset, wandb_log)

        # Store the model for this fold
        dgd_cv[fold] = dgd

        # Store the performance metrics for this fold
        loss_cv[fold] = loss_tab

        print(f"Fold {fold+1} done...")

    return dgd_cv, loss_cv


def train_midgd_cv(dgd, train_dataset, val_dataset, device, num_workers, fold,
                learning_rates={'dec':0.001,'rep':0.01,'gmm':0.01}, 
                weight_decay=0., betas=(0.5, 0.9), nepochs=100, batch_size=128, is_plot=True, pr=1, plot=10, reduction_type="sum", 
                sample_index=[0,11,22,33], subset=None, 
                wandb_log=False):
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Normalization factor
    if reduction_type == "sum":
        tlen=len(train_loader.dataset)*dgd.decoder.n_out_features
        vlen=len(validation_loader.dataset)*dgd.decoder.n_out_features
        tlen_gmm=len(train_loader.dataset)*dgd.gmm.n_mix_comp*dgd.gmm.dim
        vlen_gmm=len(validation_loader.dataset)*dgd.gmm.n_mix_comp*dgd.gmm.dim
    else:
        tlen=len(train_loader)
        vlen=len(validation_loader)
        tlen_gmm=len(train_loader)
        vlen_gmm=len(validation_loader)
    
    Ntrain=len(train_loader.dataset)
    if dgd.train_rep is None:
        dgd.train_rep = RepresentationLayer(dgd.rep_dim,Ntrain).to(device)

    Nvalidation=len(validation_loader.dataset)
    if dgd.val_rep is None:
        dgd.val_rep = RepresentationLayer(dgd.rep_dim,Nvalidation).to(device)

    dec_optimizer = torch.optim.AdamW(dgd.decoder.parameters(), lr=learning_rates['dec'], weight_decay=weight_decay, betas=betas)
    gmm_optimizer = torch.optim.AdamW(dgd.gmm.parameters(), lr=learning_rates['gmm'], weight_decay=weight_decay, betas=betas)
    train_rep_optimizer = torch.optim.AdamW(dgd.train_rep.parameters(), lr=learning_rates['rep'], weight_decay=weight_decay, betas=betas)
    val_rep_optimizer = torch.optim.AdamW(dgd.val_rep.parameters(), lr=learning_rates['rep'], weight_decay=weight_decay, betas=betas)

    loss_tab = {"epoch":{},
                "train_recon":[],"test_recon":[],
                "train_gmm":[],"test_gmm":[], 
                "train_mse":[], "test_mse":[],
                "train_mae":[], "test_mae":[],
                "train_r2":[], "test_r2":[],
                "train_spearman":[], "test_spearman":[],
                "test_pearson":[], "train_pearson":[],
                "train_expl_var":[], "test_expl_var":[],
                "train_msle":[], "test_msle":[]}
    best_loss=1.e20
    gmm_loss=True

    # For custom color mapping
    color_mapping = dict(zip(train_loader.dataset.label, train_loader.dataset.color))

    # For lib_mirna scaling
    scale_mirna_train = torch.mean(train_dataset.mirna_data, dim=-1)
    scale_mirna_test = torch.mean(val_dataset.mirna_data, dim=-1)

    # Start training
    for epoch in tqdm(range(nepochs)):
        # Train step
        loss_tab["epoch"].append(epoch)
        loss_tab["train_recon"].append(0.)
        loss_tab["train_gmm"].append(0.)
        loss_tab["train_mse"].append(0.)
        loss_tab["train_mae"].append(0.)
        loss_tab["train_r2"].append(0.)
        loss_tab["train_spearman"].append(0.)
        loss_tab["train_pearson"].append(0.)
        loss_tab["train_expl_var"].append(0.)
        loss_tab["train_msle"].append(0.)

        train_rep_optimizer.zero_grad()
        dgd.train() # Training mode
        for (mrna_data, mirna_data, lib_mrna, lib_mirna, index) in train_loader:
            dec_optimizer.zero_grad()
            if gmm_loss: 
                gmm_optimizer.zero_grad()
            recon_loss, gmm_loss = dgd.forward_and_loss(
                z=dgd.train_rep(index),
                target=[mrna_data.to(device), mirna_data.to(device)],  # Pass both mRNA and miRNA data
                scale=[lib_mrna.unsqueeze(1).to(device), lib_mirna.unsqueeze(1).to(device)],  # Pass both scales
                gmm_loss=gmm_loss,
                reduction=reduction_type,
                type="combined"
            )            

            loss_tab["train_recon"][-1] += recon_loss.item()
            loss_tab["train_gmm"][-1] += gmm_loss.item()

            loss = recon_loss + gmm_loss
            loss.backward()
            dec_optimizer.step()
            if gmm_loss: 
                gmm_optimizer.step()
        train_rep_optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            # Get data
            _, mirna_recon = dgd.forward(dgd.train_rep())
            mirna_recon = mirna_recon * scale_mirna_train.unsqueeze(1).to(device)  # Scale reconstructed miRNA
            mirna_data = train_dataset.mirna_data.to(device)
            # Get subset
            mirna_recon = mirna_recon[:,subset]
            mirna_data = mirna_data[:,subset]
            # Normalize mirna_recon and mirna_data using TPM or FPKM
            mirna_recon_tpm = mirna_recon / mirna_recon.sum() * 1e6  # TPM normalization
            mirna_data_tpm = mirna_data.to(device) / mirna_data.to(device).sum() * 1e6  # TPM normalization
            # Calculate metrics
            mirna_mse = mean_squared_error(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["train_mse"][-1] += mirna_mse.item()
            mirna_mae = mean_absolute_error(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["train_mae"][-1] += mirna_mae.item()
            mirna_r2 = r2_score(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["train_r2"][-1] += mirna_r2.item()
            mirna_spearman = spearman_corrcoef(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["train_spearman"][-1] += mirna_spearman.item()
            mirna_pearson = pearson_corrcoef(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["train_pearson"][-1] += mirna_pearson.item()
            mirna_expl_var = explained_variance(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["train_expl_var"][-1] += mirna_expl_var.item()
            mirna_msle = mean_squared_log_error(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["train_msle"][-1] += mirna_msle.item()
                    
        loss_tab["train_recon"][-1] /= tlen
        loss_tab["train_gmm"][-1] /= tlen_gmm

        # Validation step
        loss_tab["test_recon"].append(0.)
        loss_tab["test_gmm"].append(0.)
        loss_tab["test_mse"].append(0.)
        loss_tab["test_mae"].append(0.)
        loss_tab["test_r2"].append(0.)
        loss_tab["test_spearman"].append(0.)
        loss_tab["test_pearson"].append(0.)
        loss_tab["test_expl_var"].append(0.)
        loss_tab["test_msle"].append(0.)

        # Train the validation representation layer only using mRNA data
        val_rep_optimizer.zero_grad()
        dgd.eval() # Validation mode
        for (mrna_data, _, lib_mrna, _, index) in validation_loader:
            recon_loss, gmm_loss = dgd.forward_and_loss(
                z=dgd.val_rep(index),
                target=[mrna_data.to(device)],  # Pass mRNA data
                scale=[lib_mrna.unsqueeze(1).to(device)],  # Pass scales
                gmm_loss=gmm_loss,
                reduction=reduction_type,
                type="miDGD"
            )
            loss_tab["test_recon"][-1] += recon_loss.item()
            loss_tab["test_gmm"][-1] += gmm_loss.item()
            
            loss = recon_loss + gmm_loss
            loss.backward()
        val_rep_optimizer.step()

        # Calculate metrics
        with torch.no_grad():
            # Get data
            _, mirna_recon = dgd.forward(dgd.val_rep())
            mirna_recon = mirna_recon * scale_mirna_test.unsqueeze(1).to(device)
            mirna_data = val_dataset.mirna_data.to(device)
            # Go through mirna_recon
            for subset in range(len(mirna_recon)):
                # Get subset
                mirna_recon = mirna_recon[:,subset]
                mirna_data = mirna_data[:,subset]
                # Normalize mirna_recon and mirna_data using TPM or FPKM
                mirna_recon_tpm = mirna_recon / mirna_recon.sum() * 1e6  # TPM normalization
                mirna_data_tpm = mirna_data.to(device) / mirna_data.to(device).sum() * 1e6  # TPM normalization
                # Calculate metrics
                mirna_mse = mean_squared_error(mirna_recon_tpm, mirna_data_tpm)
                loss_tab["test_mse"][-1] += mirna_mse.item()
                mirna_mae = mean_absolute_error(mirna_recon_tpm, mirna_data_tpm)
                loss_tab["test_mae"][-1] += mirna_mae.item()
                mirna_r2 = r2_score(mirna_recon_tpm, mirna_data_tpm)
                loss_tab["test_r2"][-1] += mirna_r2.item()
                mirna_spearman = spearman_corrcoef(mirna_recon_tpm, mirna_data_tpm)
                loss_tab["test_spearman"][-1] += mirna_spearman.item()
                mirna_pearson = pearson_corrcoef(mirna_recon_tpm, mirna_data_tpm)
                loss_tab["test_pearson"][-1] += mirna_pearson.item()
                mirna_expl_var = explained_variance(mirna_recon_tpm, mirna_data_tpm)
                loss_tab["test_expl_var"][-1] += mirna_expl_var.item()
                mirna_msle = mean_squared_log_error(mirna_recon_tpm, mirna_data_tpm)
                loss_tab["test_msle"][-1] += mirna_msle.item()

        loss_tab["test_recon"][-1] /= vlen
        loss_tab["test_gmm"][-1] /= vlen_gmm

        if pr>=0 and (epoch)%pr==0:
            print(epoch,
                  f"train_recon: {loss_tab['train_recon'][-1]}", 
                  f"train_gmm: {loss_tab['train_gmm'][-1]}",
                  f"train_mse: {loss_tab['train_mse'][-1]}",
                  f"train_mae: {loss_tab['train_mae'][-1]}",
                  f"train_r2: {loss_tab['train_r2'][-1]}",
                  f"train_spearman: {loss_tab['train_spearman'][-1]}",
                  f"train_pearson: {loss_tab['train_pearson'][-1]}",
                  f"train_expl_var: {loss_tab['train_expl_var'][-1]}",
                  f"train_msle: {loss_tab['train_msle'][-1]}")
            print(epoch, 
                  f"test_recon: {loss_tab['test_recon'][-1]}", 
                  f"test_gmm: {loss_tab['test_gmm'][-1]}",
                  f"test_mse: {loss_tab['test_mse'][-1]}",
                  f"test_mae: {loss_tab['test_mae'][-1]}",
                  f"test_r2: {loss_tab['test_r2'][-1]}",
                  f"test_spearman: {loss_tab['test_spearman'][-1]}",
                  f"test_pearson: {loss_tab['test_pearson'][-1]}",
                  f"test_expl_var: {loss_tab['test_expl_var'][-1]}",
                  f"test_msle: {loss_tab['test_msle'][-1]}")
        if is_plot:
            if plot>=0 and (epoch)%plot==0:
                plot_latent_space(*dgd.get_latent_space_values("train",1000), train_loader.dataset.label, color_mapping, epoch, "Train")
                plot_latent_space(*dgd.get_latent_space_values("val",1000), validation_loader.dataset.label, color_mapping, epoch, "Validation")
                plot_gene(dgd, train_loader, sample_index, epoch, fold=fold, type="Train")
                plot_gene(dgd, validation_loader, sample_index, epoch, fold=fold, type="Test")
                plot_mirna(dgd, train_loader, sample_index, epoch, fold=fold, type="Train")
                plot_mirna(dgd, validation_loader, sample_index, epoch, fold=fold, type="Test")
        
        if wandb_log: 
            wandb.log({"train_recon": loss_tab["train_recon"][-1],
                       "train_gmm": loss_tab["train_gmm"][-1],
                       "train_mse": loss_tab["train_mse"][-1],
                       "train_mae": loss_tab["train_mae"][-1],
                       "train_r2": loss_tab["train_r2"][-1],
                       "train_spearman": loss_tab["train_spearman"][-1],
                       "train_pearson": loss_tab["train_pearson"][-1],
                       "train_expl_var": loss_tab["train_expl_var"][-1],
                       "train_msle": loss_tab["train_msle"][-1],
                       "test_recon": loss_tab["test_recon"][-1],
                       "test_gmm": loss_tab["test_gmm"][-1],
                       "test_mse": loss_tab["test_mse"][-1],
                       "test_mae": loss_tab["test_mae"][-1],
                       "test_r2": loss_tab["test_r2"][-1],
                       "test_spearman": loss_tab["test_spearman"][-1],
                       "test_pearson": loss_tab["test_pearson"][-1],
                       "test_expl_var": loss_tab["test_expl_var"][-1],
                       "test_msle": loss_tab["test_msle"][-1]})
    
    return loss_tab


def train_midgd_cv_full(dgd, train_dataset, val_dataset, device, num_workers, fold,
                learning_rates={'dec':0.001,'rep':0.01,'gmm':0.01}, 
                weight_decay=0., betas=(0.5, 0.9), nepochs=100, batch_size=128, is_plot=True, pr=1, plot=10, reduction_type="sum", 
                sample_index=[0,11,22,33], subset=None, 
                wandb_log=False):
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Normalization factor
    if reduction_type == "sum":
        tlen=len(train_loader.dataset)*dgd.decoder.n_out_features
        vlen=len(validation_loader.dataset)*dgd.decoder.n_out_features
        tlen_gmm=len(train_loader.dataset)*dgd.gmm.n_mix_comp*dgd.gmm.dim
        vlen_gmm=len(validation_loader.dataset)*dgd.gmm.n_mix_comp*dgd.gmm.dim
    else:
        tlen=len(train_loader)
        vlen=len(validation_loader)
        tlen_gmm=len(train_loader)
        vlen_gmm=len(validation_loader)
    
    Ntrain=len(train_loader.dataset)
    if dgd.train_rep is None:
        dgd.train_rep = RepresentationLayer(dgd.rep_dim,Ntrain).to(device)

    Nvalidation=len(validation_loader.dataset)
    if dgd.val_rep is None:
        dgd.val_rep = RepresentationLayer(dgd.rep_dim,Nvalidation).to(device)

    dec_optimizer = torch.optim.AdamW(dgd.decoder.parameters(), lr=learning_rates['dec'], weight_decay=weight_decay, betas=betas)
    gmm_optimizer = torch.optim.AdamW(dgd.gmm.parameters(), lr=learning_rates['gmm'], weight_decay=weight_decay, betas=betas)
    train_rep_optimizer = torch.optim.AdamW(dgd.train_rep.parameters(), lr=learning_rates['rep'], weight_decay=weight_decay, betas=betas)
    val_rep_optimizer = torch.optim.AdamW(dgd.val_rep.parameters(), lr=learning_rates['rep'], weight_decay=weight_decay, betas=betas)

    loss_tab = {"epoch":[],
                "train_recon":[],"test_recon":[],
                "train_gmm":[],"test_gmm":[], 
                "train_mse":[], "test_mse":[],
                "train_mae":[], "test_mae":[],
                "train_r2":[], "test_r2":[],
                "train_spearman":[], "test_spearman":[],
                "train_pearson":[], "test_pearson":[],
                "train_expl_var":[], "test_expl_var":[],
                "train_msle":[], "test_msle":[]}
    best_loss=1.e20
    gmm_loss=True

    # For custom color mapping
    color_mapping = dict(zip(train_loader.dataset.label, train_loader.dataset.color))

    # For lib_mirna scaling
    scale_mirna_train = torch.mean(train_dataset.mirna_data, dim=-1)
    scale_mirna_test = torch.mean(val_dataset.mirna_data, dim=-1)

    # Start training
    for epoch in tqdm(range(nepochs)):
        # Train step
        loss_tab["epoch"].append(epoch)
        loss_tab["train_recon"].append(0.)
        loss_tab["train_gmm"].append(0.)
        loss_tab["train_mse"].append(0.)
        loss_tab["train_mae"].append(0.)
        loss_tab["train_r2"].append(0.)
        loss_tab["train_spearman"].append(0.)
        loss_tab["train_pearson"].append(0.)
        loss_tab["train_expl_var"].append(0.)
        loss_tab["train_msle"].append(0.)

        train_rep_optimizer.zero_grad()
        dgd.train() # Training mode
        for (mrna_data, mirna_data, lib_mrna, lib_mirna, index) in train_loader:
            dec_optimizer.zero_grad()
            if gmm_loss: 
                gmm_optimizer.zero_grad()
            recon_loss, gmm_loss = dgd.forward_and_loss(
                z=dgd.train_rep(index),
                target=[mrna_data.to(device), mirna_data.to(device)],  # Pass both mRNA and miRNA data
                scale=[lib_mrna.unsqueeze(1).to(device), lib_mirna.unsqueeze(1).to(device)],  # Pass both scales
                gmm_loss=gmm_loss,
                reduction=reduction_type,
                type="combined"
            )            

            loss_tab["train_recon"][-1] += recon_loss.item()
            loss_tab["train_gmm"][-1] += gmm_loss.item()

            loss = recon_loss + gmm_loss
            loss.backward()
            dec_optimizer.step()
            if gmm_loss: 
                gmm_optimizer.step()
        train_rep_optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            # Get data
            _, mirna_recon = dgd.forward(dgd.train_rep())
            mirna_recon = mirna_recon * scale_mirna_train.unsqueeze(1).to(device)  # Scale reconstructed miRNA
            mirna_data = train_dataset.mirna_data.to(device)
            # Get subset
            mirna_recon = mirna_recon[:,subset]
            mirna_data = mirna_data[:,subset]
            # Normalize mirna_recon and mirna_data using TPM or FPKM
            mirna_recon_tpm = mirna_recon / mirna_recon.sum() * 1e6  # TPM normalization
            mirna_data_tpm = mirna_data.to(device) / mirna_data.to(device).sum() * 1e6  # TPM normalization
            # Calculate metrics
            mirna_mse = mean_squared_error(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["train_mse"][-1] += mirna_mse.item()
            mirna_mae = mean_absolute_error(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["train_mae"][-1] += mirna_mae.item()
            mirna_r2 = r2_score(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["train_r2"][-1] += mirna_r2.item()
            mirna_spearman = spearman_corrcoef(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["train_spearman"][-1] += mirna_spearman.item()
            mirna_pearson = pearson_corrcoef(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["train_pearson"][-1] += mirna_pearson.item()
            mirna_expl_var = explained_variance(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["train_expl_var"][-1] += mirna_expl_var.item()
            mirna_msle = mean_squared_log_error(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["train_msle"][-1] += mirna_msle.item()
                    
        loss_tab["train_recon"][-1] /= tlen
        loss_tab["train_gmm"][-1] /= tlen_gmm

        # Validation step
        loss_tab["test_recon"].append(0.)
        loss_tab["test_gmm"].append(0.)
        loss_tab["test_mse"].append(0.)
        loss_tab["test_mae"].append(0.)
        loss_tab["test_r2"].append(0.)
        loss_tab["test_spearman"].append(0.)
        loss_tab["test_pearson"].append(0.)
        loss_tab["test_expl_var"].append(0.)
        loss_tab["test_msle"].append(0.)

        # Train the validation representation layer only using mRNA data
        val_rep_optimizer.zero_grad()
        dgd.eval() # Validation mode
        for (mrna_data, mirna_data, lib_mrna, lib_mirna, index) in validation_loader:
            recon_loss, gmm_loss = dgd.forward_and_loss(
                z=dgd.val_rep(index),
                target=[mrna_data.to(device), mirna_data.to(device)],  # Pass both mRNA and miRNA data
                scale=[lib_mrna.unsqueeze(1).to(device), lib_mirna.unsqueeze(1).to(device)],  # Pass both scales
                gmm_loss=gmm_loss,
                reduction=reduction_type,
                type="combined"
            )
            loss_tab["test_recon"][-1] += recon_loss.item()
            loss_tab["test_gmm"][-1] += gmm_loss.item()
            
            loss = recon_loss + gmm_loss
            loss.backward()
        val_rep_optimizer.step()

        # Calculate metrics
        with torch.no_grad():
            # Get data
            _, mirna_recon = dgd.forward(dgd.val_rep())
            mirna_recon = mirna_recon * scale_mirna_test.unsqueeze(1).to(device)
            mirna_data = val_dataset.mirna_data.to(device)
            # Get subset
            mirna_recon = mirna_recon[:,subset]
            mirna_data = mirna_data[:,subset]
            # Normalize mirna_recon and mirna_data using TPM or FPKM
            mirna_recon_tpm = mirna_recon / mirna_recon.sum() * 1e6  # TPM normalization
            mirna_data_tpm = mirna_data.to(device) / mirna_data.to(device).sum() * 1e6  # TPM normalization
            # Calculate metrics
            mirna_mse = mean_squared_error(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["test_mse"][-1] += mirna_mse.item()
            mirna_mae = mean_absolute_error(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["test_mae"][-1] += mirna_mae.item()
            mirna_r2 = r2_score(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["test_r2"][-1] += mirna_r2.item()
            mirna_spearman = spearman_corrcoef(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["test_spearman"][-1] += mirna_spearman.item()
            mirna_pearson = pearson_corrcoef(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["test_pearson"][-1] += mirna_pearson.item()
            mirna_expl_var = explained_variance(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["test_expl_var"][-1] += mirna_expl_var.item()
            mirna_msle = mean_squared_log_error(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["test_msle"][-1] += mirna_msle.item()

        loss_tab["test_recon"][-1] /= vlen
        loss_tab["test_gmm"][-1] /= vlen_gmm

        if pr>=0 and (epoch)%pr==0:
            print(epoch,
                  f"train_recon: {loss_tab['train_recon'][-1]}", 
                  f"train_gmm: {loss_tab['train_gmm'][-1]}",
                  f"train_mse: {loss_tab['train_mse'][-1]}",
                  f"train_mae: {loss_tab['train_mae'][-1]}",
                  f"train_r2: {loss_tab['train_r2'][-1]}",
                  f"train_spearman: {loss_tab['train_spearman'][-1]}",
                  f"train_pearson: {loss_tab['train_pearson'][-1]}",
                  f"train_expl_var: {loss_tab['train_expl_var'][-1]}",
                  f"train_msle: {loss_tab['train_msle'][-1]}")
            print(epoch, 
                  f"test_recon: {loss_tab['test_recon'][-1]}", 
                  f"test_gmm: {loss_tab['test_gmm'][-1]}",
                  f"test_mse: {loss_tab['test_mse'][-1]}",
                  f"test_mae: {loss_tab['test_mae'][-1]}",
                  f"test_r2: {loss_tab['test_r2'][-1]}",
                  f"test_spearman: {loss_tab['test_spearman'][-1]}",
                  f"test_pearson: {loss_tab['test_pearson'][-1]}",
                  f"test_expl_var: {loss_tab['test_expl_var'][-1]}",
                  f"test_msle: {loss_tab['test_msle'][-1]}")
        if is_plot:
            if plot>=0 and (epoch)%plot==0:
                plot_latent_space(*dgd.get_latent_space_values("train",1000), train_loader.dataset.label, color_mapping, epoch, fold=fold, type="Train")
                plot_latent_space(*dgd.get_latent_space_values("val",1000), validation_loader.dataset.label, color_mapping, epoch, fold=fold, type="Validation")
                plot_gene(dgd, train_loader, sample_index, epoch, fold=fold, type="Train")
                plot_gene(dgd, validation_loader, sample_index, epoch, fold=fold, type="Test")
                plot_mirna(dgd, train_loader, sample_index, epoch, fold=fold, type="Train")
                plot_mirna(dgd, validation_loader, sample_index, epoch, fold=fold, type="Test")
        
        if wandb_log: 
            wandb.log({"train_recon": loss_tab["train_recon"][-1],
                       "train_gmm": loss_tab["train_gmm"][-1],
                       "train_mse": loss_tab["train_mse"][-1],
                       "train_mae": loss_tab["train_mae"][-1],
                       "train_r2": loss_tab["train_r2"][-1],
                       "train_spearman": loss_tab["train_spearman"][-1],
                       "train_pearson": loss_tab["train_pearson"][-1],
                       "train_expl_var": loss_tab["train_expl_var"][-1],
                       "train_msle": loss_tab["train_msle"][-1],
                       "test_recon": loss_tab["test_recon"][-1],
                       "test_gmm": loss_tab["test_gmm"][-1],
                       "test_mse": loss_tab["test_mse"][-1],
                       "test_mae": loss_tab["test_mae"][-1],
                       "test_r2": loss_tab["test_r2"][-1],
                       "test_spearman": loss_tab["test_spearman"][-1],
                       "test_pearson": loss_tab["test_pearson"][-1],
                       "test_expl_var": loss_tab["test_expl_var"][-1],
                       "test_msle": loss_tab["test_msle"][-1]})
    
    return loss_tab


def train_midgd_cv_switch(dgd, train_dataset, val_dataset, device, num_workers, fold,
                learning_rates={'dec':0.001,'rep':0.01,'gmm':0.01}, 
                weight_decay=0., betas=(0.5, 0.9), nepochs=100, batch_size=128, is_plot=True, pr=1, plot=10, reduction_type="sum", 
                sample_index=[0,11,22,33], subset=None, 
                wandb_log=False):
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Normalization factor
    if reduction_type == "sum":
        tlen=len(train_loader.dataset)*dgd.decoder.n_out_features
        vlen=len(validation_loader.dataset)*dgd.decoder.n_out_features
        tlen_gmm=len(train_loader.dataset)*dgd.gmm.n_mix_comp*dgd.gmm.dim
        vlen_gmm=len(validation_loader.dataset)*dgd.gmm.n_mix_comp*dgd.gmm.dim
    else:
        tlen=len(train_loader)
        vlen=len(validation_loader)
        tlen_gmm=len(train_loader)
        vlen_gmm=len(validation_loader)
    
    Ntrain=len(train_loader.dataset)
    if dgd.train_rep is None:
        dgd.train_rep = RepresentationLayer(dgd.rep_dim,Ntrain).to(device)

    Nvalidation=len(validation_loader.dataset)
    if dgd.val_rep is None:
        dgd.val_rep = RepresentationLayer(dgd.rep_dim,Nvalidation).to(device)

    dec_optimizer = torch.optim.AdamW(dgd.decoder.parameters(), lr=learning_rates['dec'], weight_decay=weight_decay, betas=betas)
    gmm_optimizer = torch.optim.AdamW(dgd.gmm.parameters(), lr=learning_rates['gmm'], weight_decay=weight_decay, betas=betas)
    train_rep_optimizer = torch.optim.AdamW(dgd.train_rep.parameters(), lr=learning_rates['rep'], weight_decay=weight_decay, betas=betas)
    val_rep_optimizer = torch.optim.AdamW(dgd.val_rep.parameters(), lr=learning_rates['rep'], weight_decay=weight_decay, betas=betas)

    loss_tab = {"epoch":[],
                "train_recon":[],"test_recon":[],
                "train_gmm":[],"test_gmm":[], 
                "train_mse":[], "test_mse":[],
                "train_mae":[], "test_mae":[],
                "train_r2":[], "test_r2":[],
                "train_spearman":[], "test_spearman":[],
                "test_pearson":[], "train_pearson":[],
                "train_expl_var":[], "test_expl_var":[],
                "train_msle":[], "test_msle":[]}
    best_loss=1.e20
    gmm_loss=True

    # For custom color mapping
    color_mapping = dict(zip(train_loader.dataset.label, train_loader.dataset.color))

    # For lib_mirna scaling
    scale_mirna_train = torch.mean(train_dataset.mirna_data, dim=-1)
    scale_mirna_test = torch.mean(val_dataset.mirna_data, dim=-1)

    # Start training
    for epoch in tqdm(range(nepochs)):
        # Train step
        loss_tab["epoch"].append(epoch)
        loss_tab["train_recon"].append(0.)
        loss_tab["train_gmm"].append(0.)
        loss_tab["train_mse"].append(0.)
        loss_tab["train_mae"].append(0.)
        loss_tab["train_r2"].append(0.)
        loss_tab["train_spearman"].append(0.)
        loss_tab["train_pearson"].append(0.)
        loss_tab["train_expl_var"].append(0.)
        loss_tab["train_msle"].append(0.)

        train_rep_optimizer.zero_grad()
        dgd.train() # Training mode
        for (mrna_data, mirna_data, lib_mrna, lib_mirna, index) in train_loader:
            dec_optimizer.zero_grad()
            if gmm_loss: 
                gmm_optimizer.zero_grad()
            recon_loss, gmm_loss = dgd.forward_and_loss(
                z=dgd.train_rep(index),
                target=[mirna_data.to(device), mrna_data.to(device)],  # Pass both mRNA and miRNA data
                scale=[lib_mirna.unsqueeze(1).to(device), lib_mrna.unsqueeze(1).to(device)],  # Pass both scales
                gmm_loss=gmm_loss,
                reduction=reduction_type,
                type="combined"
            )            

            loss_tab["train_recon"][-1] += recon_loss.item()
            loss_tab["train_gmm"][-1] += gmm_loss.item()

            loss = recon_loss + gmm_loss
            loss.backward()
            dec_optimizer.step()
            if gmm_loss: 
                gmm_optimizer.step()
        train_rep_optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            # Get data
            mirna_recon, _ = dgd.forward(dgd.train_rep())
            mirna_recon = mirna_recon * scale_mirna_train.unsqueeze(1).to(device)  # Scale reconstructed miRNA
            mirna_data = train_dataset.mirna_data.to(device)
            # Get subset
            mirna_recon = mirna_recon[:,subset]
            mirna_data = mirna_data[:,subset]
            # Normalize mirna_recon and mirna_data using TPM or FPKM
            mirna_recon_tpm = mirna_recon / mirna_recon.sum() * 1e6  # TPM normalization
            mirna_data_tpm = mirna_data.to(device) / mirna_data.to(device).sum() * 1e6  # TPM normalization
            # Calculate metrics
            mirna_mse = mean_squared_error(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["train_mse"][-1] += mirna_mse.item()
            mirna_mae = mean_absolute_error(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["train_mae"][-1] += mirna_mae.item()
            mirna_r2 = r2_score(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["train_r2"][-1] += mirna_r2.item()
            mirna_spearman = spearman_corrcoef(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["train_spearman"][-1] += mirna_spearman.item()
            mirna_pearson = pearson_corrcoef(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["train_pearson"][-1] += mirna_pearson.item()
            mirna_expl_var = explained_variance(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["train_expl_var"][-1] += mirna_expl_var.item()
            mirna_msle = mean_squared_log_error(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["train_msle"][-1] += mirna_msle.item()
                    
        loss_tab["train_recon"][-1] /= tlen
        loss_tab["train_gmm"][-1] /= tlen_gmm

        # Validation step
        loss_tab["test_recon"].append(0.)
        loss_tab["test_gmm"].append(0.)
        loss_tab["test_mse"].append(0.)
        loss_tab["test_mae"].append(0.)
        loss_tab["test_r2"].append(0.)
        loss_tab["test_spearman"].append(0.)
        loss_tab["test_pearson"].append(0.)
        loss_tab["test_expl_var"].append(0.)
        loss_tab["test_msle"].append(0.)

        # Train the validation representation layer only using mRNA data
        val_rep_optimizer.zero_grad()
        dgd.eval() # Validation mode
        for (mrna_data, _, lib_mrna, _, index) in validation_loader:
            recon_loss, gmm_loss = dgd.forward_and_loss(
                z=dgd.val_rep(index),
                target=[mrna_data.to(device)],  # Pass mRNA data
                scale=[lib_mrna.unsqueeze(1).to(device)],  # Pass scales
                gmm_loss=gmm_loss,
                reduction=reduction_type,
                type="switch"
            )
            loss_tab["test_recon"][-1] += recon_loss.item()
            loss_tab["test_gmm"][-1] += gmm_loss.item()
            
            loss = recon_loss + gmm_loss
            loss.backward()
        val_rep_optimizer.step()

        # Calculate metrics
        with torch.no_grad():
            # Get data
            mirna_recon, _ = dgd.forward(dgd.val_rep())
            mirna_recon = mirna_recon * scale_mirna_test.unsqueeze(1).to(device)
            mirna_data = val_dataset.mirna_data.to(device)
            # Get subset
            mirna_recon = mirna_recon[:,subset]
            mirna_data = mirna_data[:,subset]
            # Normalize mirna_recon and mirna_data using TPM or FPKM
            mirna_recon_tpm = mirna_recon / mirna_recon.sum() * 1e6  # TPM normalization
            mirna_data_tpm = mirna_data.to(device) / mirna_data.to(device).sum() * 1e6  # TPM normalization
            # Calculate metrics
            mirna_mse = mean_squared_error(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["test_mse"][-1] += mirna_mse.item()
            mirna_mae = mean_absolute_error(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["test_mae"][-1] += mirna_mae.item()
            mirna_r2 = r2_score(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["test_r2"][-1] += mirna_r2.item()
            mirna_spearman = spearman_corrcoef(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["test_spearman"][-1] += mirna_spearman.item()
            mirna_pearson = pearson_corrcoef(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["test_pearson"][-1] += mirna_pearson.item()
            mirna_expl_var = explained_variance(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["test_expl_var"][-1] += mirna_expl_var.item()
            mirna_msle = mean_squared_log_error(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["test_msle"][-1] += mirna_msle.item()

        loss_tab["test_recon"][-1] /= vlen
        loss_tab["test_gmm"][-1] /= vlen_gmm

        if pr>=0 and (epoch)%pr==0:
            print(epoch,
                  f"train_recon: {loss_tab['train_recon'][-1]}", 
                  f"train_gmm: {loss_tab['train_gmm'][-1]}",
                  f"train_mse: {loss_tab['train_mse'][-1]}",
                  f"train_mae: {loss_tab['train_mae'][-1]}",
                  f"train_r2: {loss_tab['train_r2'][-1]}",
                  f"train_spearman: {loss_tab['train_spearman'][-1]}",
                  f"train_pearson: {loss_tab['train_pearson'][-1]}",
                  f"train_expl_var: {loss_tab['train_expl_var'][-1]}",
                  f"train_msle: {loss_tab['train_msle'][-1]}")
            print(epoch, 
                  f"test_recon: {loss_tab['test_recon'][-1]}", 
                  f"test_gmm: {loss_tab['test_gmm'][-1]}",
                  f"test_mse: {loss_tab['test_mse'][-1]}",
                  f"test_mae: {loss_tab['test_mae'][-1]}",
                  f"test_r2: {loss_tab['test_r2'][-1]}",
                  f"test_spearman: {loss_tab['test_spearman'][-1]}",
                  f"test_pearson: {loss_tab['test_pearson'][-1]}",
                  f"test_expl_var: {loss_tab['test_expl_var'][-1]}",
                  f"test_msle: {loss_tab['test_msle'][-1]}")
        if is_plot:
            if plot>=0 and (epoch)%plot==0:
                plot_latent_space(*dgd.get_latent_space_values("train",1000), train_loader.dataset.label, color_mapping, epoch, "Train")
                plot_latent_space(*dgd.get_latent_space_values("val",1000), validation_loader.dataset.label, color_mapping, epoch, "Validation")
                plot_gene(dgd, train_loader, sample_index, epoch, fold=fold, type="Train")
                plot_gene(dgd, validation_loader, sample_index, epoch, fold=fold, type="Test")
                plot_mirna(dgd, train_loader, sample_index, epoch, fold=fold, type="Train")
                plot_mirna(dgd, validation_loader, sample_index, epoch, fold=fold, type="Test")
        
        if wandb_log: 
            wandb.log({"train_recon": loss_tab["train_recon"][-1],
                       "train_gmm": loss_tab["train_gmm"][-1],
                       "train_mse": loss_tab["train_mse"][-1],
                       "train_mae": loss_tab["train_mae"][-1],
                       "train_r2": loss_tab["train_r2"][-1],
                       "train_spearman": loss_tab["train_spearman"][-1],
                       "train_pearson": loss_tab["train_pearson"][-1],
                       "train_expl_var": loss_tab["train_expl_var"][-1],
                       "train_msle": loss_tab["train_msle"][-1],
                       "test_recon": loss_tab["test_recon"][-1],
                       "test_gmm": loss_tab["test_gmm"][-1],
                       "test_mse": loss_tab["test_mse"][-1],
                       "test_mae": loss_tab["test_mae"][-1],
                       "test_r2": loss_tab["test_r2"][-1],
                       "test_spearman": loss_tab["test_spearman"][-1],
                       "test_pearson": loss_tab["test_pearson"][-1],
                       "test_expl_var": loss_tab["test_expl_var"][-1],
                       "test_msle": loss_tab["test_msle"][-1]})
    
    return loss_tab


def train_midgd_cv_full_switch(dgd, train_dataset, val_dataset, device, num_workers, fold,
                learning_rates={'dec':0.001,'rep':0.01,'gmm':0.01}, 
                weight_decay=0., betas=(0.5, 0.9), nepochs=100, batch_size=128, is_plot=True, pr=1, plot=10, reduction_type="sum", 
                sample_index=[0,11,22,33], subset=None, 
                wandb_log=False):
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Normalization factor
    if reduction_type == "sum":
        tlen=len(train_loader.dataset)*dgd.decoder.n_out_features
        vlen=len(validation_loader.dataset)*dgd.decoder.n_out_features
        tlen_gmm=len(train_loader.dataset)*dgd.gmm.n_mix_comp*dgd.gmm.dim
        vlen_gmm=len(validation_loader.dataset)*dgd.gmm.n_mix_comp*dgd.gmm.dim
    else:
        tlen=len(train_loader)
        vlen=len(validation_loader)
        tlen_gmm=len(train_loader)
        vlen_gmm=len(validation_loader)
    
    Ntrain=len(train_loader.dataset)
    if dgd.train_rep is None:
        dgd.train_rep = RepresentationLayer(dgd.rep_dim,Ntrain).to(device)

    Nvalidation=len(validation_loader.dataset)
    if dgd.val_rep is None:
        dgd.val_rep = RepresentationLayer(dgd.rep_dim,Nvalidation).to(device)

    dec_optimizer = torch.optim.AdamW(dgd.decoder.parameters(), lr=learning_rates['dec'], weight_decay=weight_decay, betas=betas)
    gmm_optimizer = torch.optim.AdamW(dgd.gmm.parameters(), lr=learning_rates['gmm'], weight_decay=weight_decay, betas=betas)
    train_rep_optimizer = torch.optim.AdamW(dgd.train_rep.parameters(), lr=learning_rates['rep'], weight_decay=weight_decay, betas=betas)
    val_rep_optimizer = torch.optim.AdamW(dgd.val_rep.parameters(), lr=learning_rates['rep'], weight_decay=weight_decay, betas=betas)

    loss_tab = {"epoch":[],
                "train_recon":[],"test_recon":[],
                "train_gmm":[],"test_gmm":[], 
                "train_mse":[], "test_mse":[],
                "train_mae":[], "test_mae":[],
                "train_r2":[], "test_r2":[],
                "train_spearman":[], "test_spearman":[],
                "train_pearson":[], "test_pearson":[],
                "train_expl_var":[], "test_expl_var":[],
                "train_msle":[], "test_msle":[]}
    best_loss=1.e20
    gmm_loss=True

    # For custom color mapping
    color_mapping = dict(zip(train_loader.dataset.label, train_loader.dataset.color))

    # For lib_mirna scaling
    scale_mirna_train = torch.mean(train_dataset.mirna_data, dim=-1)
    scale_mirna_test = torch.mean(val_dataset.mirna_data, dim=-1)

    # Start training
    for epoch in tqdm(range(nepochs)):
        # Train step
        loss_tab["epoch"].append(epoch)
        loss_tab["train_recon"].append(0.)
        loss_tab["train_gmm"].append(0.)
        loss_tab["train_mse"].append(0.)
        loss_tab["train_mae"].append(0.)
        loss_tab["train_r2"].append(0.)
        loss_tab["train_spearman"].append(0.)
        loss_tab["train_pearson"].append(0.)
        loss_tab["train_expl_var"].append(0.)
        loss_tab["train_msle"].append(0.)

        train_rep_optimizer.zero_grad()
        dgd.train() # Training mode
        for (mrna_data, mirna_data, lib_mrna, lib_mirna, index) in train_loader:
            dec_optimizer.zero_grad()
            if gmm_loss: 
                gmm_optimizer.zero_grad()
            recon_loss, gmm_loss = dgd.forward_and_loss(
                z=dgd.train_rep(index),
                target=[mirna_data.to(device), mrna_data.to(device)],  # Pass both mRNA and miRNA data
                scale=[lib_mirna.unsqueeze(1).to(device), lib_mrna.unsqueeze(1).to(device)],  # Pass both scales
                gmm_loss=gmm_loss,
                reduction=reduction_type,
                type="combined"
            )            

            loss_tab["train_recon"][-1] += recon_loss.item()
            loss_tab["train_gmm"][-1] += gmm_loss.item()

            loss = recon_loss + gmm_loss
            loss.backward()
            dec_optimizer.step()
            if gmm_loss: 
                gmm_optimizer.step()
        train_rep_optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            # Get data
            _, mirna_recon = dgd.forward(dgd.train_rep())
            mirna_recon = mirna_recon * scale_mirna_train.unsqueeze(1).to(device)  # Scale reconstructed miRNA
            mirna_data = train_dataset.mirna_data.to(device)
            # Get subset
            mirna_recon = mirna_recon[:,subset]
            mirna_data = mirna_data[:,subset]
            # Normalize mirna_recon and mirna_data using TPM or FPKM
            mirna_recon_tpm = mirna_recon / mirna_recon.sum() * 1e6  # TPM normalization
            mirna_data_tpm = mirna_data.to(device) / mirna_data.to(device).sum() * 1e6  # TPM normalization
            # Calculate metrics
            mirna_mse = mean_squared_error(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["train_mse"][-1] += mirna_mse.item()
            mirna_mae = mean_absolute_error(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["train_mae"][-1] += mirna_mae.item()
            mirna_r2 = r2_score(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["train_r2"][-1] += mirna_r2.item()
            mirna_spearman = spearman_corrcoef(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["train_spearman"][-1] += mirna_spearman.item()
            mirna_pearson = pearson_corrcoef(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["train_pearson"][-1] += mirna_pearson.item()
            mirna_expl_var = explained_variance(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["train_expl_var"][-1] += mirna_expl_var.item()
            mirna_msle = mean_squared_log_error(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["train_msle"][-1] += mirna_msle.item()
                    
        loss_tab["train_recon"][-1] /= tlen
        loss_tab["train_gmm"][-1] /= tlen_gmm

        # Validation step
        loss_tab["test_recon"].append(0.)
        loss_tab["test_gmm"].append(0.)
        loss_tab["test_mse"].append(0.)
        loss_tab["test_mae"].append(0.)
        loss_tab["test_r2"].append(0.)
        loss_tab["test_spearman"].append(0.)
        loss_tab["test_pearson"].append(0.)
        loss_tab["test_expl_var"].append(0.)
        loss_tab["test_msle"].append(0.)

        # Train the validation representation layer only using mRNA data
        val_rep_optimizer.zero_grad()
        dgd.eval() # Validation mode
        for (mrna_data, mirna_data, lib_mrna, lib_mirna, index) in validation_loader:
            recon_loss, gmm_loss = dgd.forward_and_loss(
                z=dgd.val_rep(index),
                target=[mirna_data.to(device), mrna_data.to(device)],  # Pass both mRNA and miRNA data
                scale=[lib_mirna.unsqueeze(1).to(device), lib_mrna.unsqueeze(1).to(device)],  # Pass both scales
                gmm_loss=gmm_loss,
                reduction=reduction_type,
                type="combined"
            )
            loss_tab["test_recon"][-1] += recon_loss.item()
            loss_tab["test_gmm"][-1] += gmm_loss.item()
            
            loss = recon_loss + gmm_loss
            loss.backward()
        val_rep_optimizer.step()

        # Calculate metrics
        with torch.no_grad():
            # Get data
            _, mirna_recon = dgd.forward(dgd.val_rep())
            mirna_recon = mirna_recon * scale_mirna_test.unsqueeze(1).to(device)
            mirna_data = val_dataset.mirna_data.to(device)
            # Get subset
            mirna_recon = mirna_recon[:,subset]
            mirna_data = mirna_data[:,subset]
            # Normalize mirna_recon and mirna_data using TPM or FPKM
            mirna_recon_tpm = mirna_recon / mirna_recon.sum() * 1e6  # TPM normalization
            mirna_data_tpm = mirna_data.to(device) / mirna_data.to(device).sum() * 1e6  # TPM normalization
            # Calculate metrics
            mirna_mse = mean_squared_error(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["test_mse"][-1] += mirna_mse.item()
            mirna_mae = mean_absolute_error(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["test_mae"][-1] += mirna_mae.item()
            mirna_r2 = r2_score(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["test_r2"][-1] += mirna_r2.item()
            mirna_spearman = spearman_corrcoef(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["test_spearman"][-1] += mirna_spearman.item()
            mirna_pearson = pearson_corrcoef(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["test_pearson"][-1] += mirna_pearson.item()
            mirna_expl_var = explained_variance(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["test_expl_var"][-1] += mirna_expl_var.item()
            mirna_msle = mean_squared_log_error(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["test_msle"][-1] += mirna_msle.item()

        loss_tab["test_recon"][-1] /= vlen
        loss_tab["test_gmm"][-1] /= vlen_gmm

        if pr>=0 and (epoch)%pr==0:
            print(epoch,
                  f"train_recon: {loss_tab['train_recon'][-1]}", 
                  f"train_gmm: {loss_tab['train_gmm'][-1]}",
                  f"train_mse: {loss_tab['train_mse'][-1]}",
                  f"train_mae: {loss_tab['train_mae'][-1]}",
                  f"train_r2: {loss_tab['train_r2'][-1]}",
                  f"train_spearman: {loss_tab['train_spearman'][-1]}",
                  f"train_pearson: {loss_tab['train_pearson'][-1]}",
                  f"train_expl_var: {loss_tab['train_expl_var'][-1]}",
                  f"train_msle: {loss_tab['train_msle'][-1]}")
            print(epoch, 
                  f"test_recon: {loss_tab['test_recon'][-1]}", 
                  f"test_gmm: {loss_tab['test_gmm'][-1]}",
                  f"test_mse: {loss_tab['test_mse'][-1]}",
                  f"test_mae: {loss_tab['test_mae'][-1]}",
                  f"test_r2: {loss_tab['test_r2'][-1]}",
                  f"test_spearman: {loss_tab['test_spearman'][-1]}",
                  f"test_pearson: {loss_tab['test_pearson'][-1]}",
                  f"test_expl_var: {loss_tab['test_expl_var'][-1]}",
                  f"test_msle: {loss_tab['test_msle'][-1]}")
        if is_plot:
            if plot>=0 and (epoch)%plot==0:
                plot_latent_space(*dgd.get_latent_space_values("train",1000), train_loader.dataset.label, color_mapping, epoch, fold=fold, type="Train")
                plot_latent_space(*dgd.get_latent_space_values("val",1000), validation_loader.dataset.label, color_mapping, epoch, fold=fold, type="Validation")
                plot_gene(dgd, train_loader, sample_index, epoch, fold=fold, type="Train")
                plot_gene(dgd, validation_loader, sample_index, epoch, fold=fold, type="Test")
                plot_mirna(dgd, train_loader, sample_index, epoch, fold=fold, type="Train")
                plot_mirna(dgd, validation_loader, sample_index, epoch, fold=fold, type="Test")
        
        if wandb_log: 
            wandb.log({"train_recon": loss_tab["train_recon"][-1],
                       "train_gmm": loss_tab["train_gmm"][-1],
                       "train_mse": loss_tab["train_mse"][-1],
                       "train_mae": loss_tab["train_mae"][-1],
                       "train_r2": loss_tab["train_r2"][-1],
                       "train_spearman": loss_tab["train_spearman"][-1],
                       "train_pearson": loss_tab["train_pearson"][-1],
                       "train_expl_var": loss_tab["train_expl_var"][-1],
                       "train_msle": loss_tab["train_msle"][-1],
                       "test_recon": loss_tab["test_recon"][-1],
                       "test_gmm": loss_tab["test_gmm"][-1],
                       "test_mse": loss_tab["test_mse"][-1],
                       "test_mae": loss_tab["test_mae"][-1],
                       "test_r2": loss_tab["test_r2"][-1],
                       "test_spearman": loss_tab["test_spearman"][-1],
                       "test_pearson": loss_tab["test_pearson"][-1],
                       "test_expl_var": loss_tab["test_expl_var"][-1],
                       "test_msle": loss_tab["test_msle"][-1]})
    
    return loss_tab


