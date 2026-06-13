import torch
from base.dgd.latent import RepresentationLayer
from base.plotting.plot import plot_latent_space, plot_latent_space_multi, plot_gene, plot_mirna

from torchmetrics.functional.regression import mean_squared_error, mean_absolute_error, r2_score, pearson_corrcoef, spearman_corrcoef, mean_squared_log_error

from tqdm import tqdm
import wandb
import numpy as np

# Training functions
def train_midgd(dgd, train_loader, validation_loader, device,
                learning_rates={'dec':0.0001,'rep':0.01,'gmm':0.01}, 
                weight_decay=0., betas=(0.9, 0.999), scheduler=None,
                nepochs=100, fold=None, pr=1, plot=10, 
                reduction_type="sum", scaling_type="mean",
                sample_index=[0,11,22,33], subset=1310, wandb_log=False, early_stopping=50, is_plot=True):
    # Normalization factor
    if reduction_type == "sum":
        tlen=len(train_loader.dataset)*dgd.decoder.n_out_features
        vlen=len(validation_loader.dataset)*dgd.decoder.n_out_features
        tlen_gmm=len(train_loader.dataset)*dgd.gmm.n_mix_comp*dgd.gmm.dim
        vlen_gmm=len(validation_loader.dataset)*dgd.gmm.n_mix_comp*dgd.gmm.dim

        tlen_mirna = len(train_loader.dataset)*dgd.decoder.n_out_features_mirna
        vlen_mirna = len(validation_loader.dataset)*dgd.decoder.n_out_features_mirna
        tlen_mrna = len(train_loader.dataset)*dgd.decoder.n_out_features_mrna
        vlen_mrna = len(validation_loader.dataset)*dgd.decoder.n_out_features_mrna
    else:
        tlen=len(train_loader)
        vlen=len(validation_loader)
        tlen_gmm=len(train_loader)
        vlen_gmm=len(validation_loader)

    # Train Representation Layer Initialization
    Ntrain=len(train_loader.dataset)
    if dgd.train_rep is None:
        dgd.train_rep = RepresentationLayer(dgd.rep_dim,Ntrain).to(device)

    # Test/Validation Representation Layer Initialization
    Nvalidation=len(validation_loader.dataset)
    if dgd.val_rep is None:
        dgd.val_rep = RepresentationLayer(dgd.rep_dim,Nvalidation).to(device)

    # Optimizer Initialization
    dec_optimizer = torch.optim.AdamW(dgd.decoder.parameters(), lr=learning_rates['dec'], weight_decay=weight_decay['dec'], betas=betas)
    gmm_optimizer = torch.optim.AdamW(dgd.gmm.parameters(), lr=learning_rates['gmm'], weight_decay=weight_decay['gmm'], betas=betas)
    train_rep_optimizer = torch.optim.AdamW(dgd.train_rep.parameters(), lr=learning_rates['rep'], weight_decay=weight_decay['rep'], betas=betas)
    val_rep_optimizer = torch.optim.AdamW(dgd.val_rep.parameters(), lr=learning_rates['rep'], weight_decay=weight_decay['rep'], betas=betas)

    # Scheduler Initialization
    if scheduler['type'] == "step":
        dec_scheduler = torch.optim.lr_scheduler.StepLR(dec_optimizer, step_size=scheduler['step_size'], gamma=scheduler['gamma_step'])
        gmm_scheduler = torch.optim.lr_scheduler.StepLR(gmm_optimizer, step_size=scheduler['step_size'], gamma=scheduler['gamma_step'])
        train_rep_scheduler = torch.optim.lr_scheduler.StepLR(train_rep_optimizer, step_size=scheduler['step_size'], gamma=scheduler['gamma_step'])
        val_rep_scheduler = torch.optim.lr_scheduler.StepLR(val_rep_optimizer, step_size=scheduler['step_size'], gamma=scheduler['gamma_step'])
    elif scheduler['type'] == "exponential":
        dec_scheduler = torch.optim.lr_scheduler.ExponentialLR(dec_optimizer, gamma=scheduler['gamma_exp'])
        gmm_scheduler = torch.optim.lr_scheduler.ExponentialLR(gmm_optimizer, gamma=scheduler['gamma_exp'])
        train_rep_scheduler = torch.optim.lr_scheduler.ExponentialLR(train_rep_optimizer, gamma=scheduler['gamma_exp'])
        val_rep_scheduler = torch.optim.lr_scheduler.ExponentialLR(val_rep_optimizer, gamma=scheduler['gamma_exp'])

    # Metrics logger initialization
    loss_tab = {"epoch":[],
                "train_loss":[], "test_loss":[],
                "train_recon_mirna":[],"train_recon_mrna":[], 
                "test_recon_mirna":[], "test_recon_mrna":[],
                "train_gmm":[],"test_gmm":[], 
                "train_mse":[], "test_mse":[],
                "train_mae":[], "test_mae":[],
                "train_r2":[], "test_r2":[],
                "train_spearman":[], "test_spearman":[],
                "test_pearson":[], "train_pearson":[],
                "train_msle":[], "test_msle":[],
                "delta_mirna":[]}
    gmm_loss=True

    # For custom color mapping
    train_mapping_cancer = dict(zip(train_loader.dataset.cancer_type, train_loader.dataset.color))
    val_mapping_cancer = dict(zip(validation_loader.dataset.cancer_type, validation_loader.dataset.color))
    color_mapping_cancer = train_mapping_cancer.copy()
    color_mapping_cancer.update(val_mapping_cancer)


    train_mapping_tissue = dict(zip(train_loader.dataset.primary_site, train_loader.dataset.color))
    val_mapping_tissue = dict(zip(validation_loader.dataset.primary_site, validation_loader.dataset.color))
    color_mapping_tissue = train_mapping_tissue.copy()
    color_mapping_tissue.update(val_mapping_tissue)

    # Check if using GTEx dataset
    is_gtex = "GTEx" in train_loader.dataset.batch

    
    # Early stopping
    best_loss = 1e20
    best_loss_mrna = 1e20
    best_loss_mirna = 1e20
    best_epoch = -1

    # Start training
    for epoch in tqdm(range(nepochs)):
        # Train step
        loss_tab["epoch"].append(epoch)
        loss_tab["train_loss"].append(0.)
        loss_tab["train_recon_mirna"].append(0.)
        loss_tab["train_recon_mrna"].append(0.)
        loss_tab["train_gmm"].append(0.)
        # loss_tab["train_mse"].append(0.)
        # loss_tab["train_mae"].append(0.)
        loss_tab["train_r2"].append(0.)
        loss_tab["train_spearman"].append(0.)
        loss_tab["train_pearson"].append(0.)
        # loss_tab["train_msle"].append(0.)

        train_rep_optimizer.zero_grad()
        dgd.train() # Training 

        for (mrna_data, mirna_data, lib_mrna, lib_mirna, index, label) in train_loader:
            dec_optimizer.zero_grad()
            if gmm_loss: 
                gmm_optimizer.zero_grad()
            mirna_recon_loss, mrna_recon_loss, gmm_loss = dgd.forward_and_loss(
                z=dgd.train_rep(index),
                target=[mirna_data.to(device), mrna_data.to(device)],  # Pass both mRNA and miRNA data
                scale=[lib_mirna.unsqueeze(1).to(device), lib_mrna.unsqueeze(1).to(device)],  # Pass both scales
                gmm_loss=gmm_loss,
                reduction=reduction_type,
                type="combined"
            )
            
            loss_tab["train_recon_mirna"][-1] += mirna_recon_loss.item()
            loss_tab["train_recon_mrna"][-1] += mrna_recon_loss.item()
            loss_tab["train_gmm"][-1] += gmm_loss.item()

            loss =  mrna_recon_loss + mirna_recon_loss + gmm_loss
            loss.backward()
            dec_optimizer.step()
            if gmm_loss: 
                gmm_optimizer.step()
        train_rep_optimizer.step()

        # Update the scheduler
        if scheduler['type'] == "step" or scheduler['type'] == "exponential":
            dec_scheduler.step()
            gmm_scheduler.step()
            train_rep_scheduler.step()

        # Calculate metrics
        with torch.inference_mode():
            # Get data
            if scaling_type == 'mean':
                scaling = torch.nanmean(train_loader.dataset.mirna_data, axis=1)
            elif scaling_type == 'max':
                scaling = torch.nanmax(train_loader.dataset.mirna_data, axis=1).values
            elif scaling_type == 'sum':
                scaling = torch.nansum(train_loader.dataset.mirna_data, axis=1)

            mirna_recon, _ = dgd.forward(dgd.train_rep())
            mirna_recon = mirna_recon * scaling.unsqueeze(1).to(device)
            mirna_data = train_loader.dataset.mirna_data.to(device)
            # Get subset
            mirna_recon_tpm = mirna_recon[:,subset]
            mirna_data_tpm = mirna_data[:,subset]
            # Calculate metrics
            # mirna_mse = mean_squared_error(mirna_recon_tpm, mirna_data_tpm)
            # loss_tab["train_mse"][-1] += mirna_mse.item()
            # mirna_mae = mean_absolute_error(mirna_recon_tpm, mirna_data_tpm)
            # loss_tab["train_mae"][-1] += mirna_mae.item()
            mirna_r2 = r2_score(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["train_r2"][-1] += mirna_r2.item()
            mirna_spearman = spearman_corrcoef(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["train_spearman"][-1] += mirna_spearman.item()
            mirna_pearson = pearson_corrcoef(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["train_pearson"][-1] += mirna_pearson.item()
            # mirna_msle = mean_squared_log_error(mirna_recon_tpm, mirna_data_tpm)
            # loss_tab["train_msle"][-1] += mirna_msle.item()

        loss_tab["train_recon_mirna"][-1] /= tlen_mirna
        loss_tab["train_recon_mrna"][-1] /= tlen_mrna
        loss_tab["train_gmm"][-1] /= tlen_gmm
        loss_tab["train_loss"][-1] = loss_tab["train_recon_mirna"][-1] + loss_tab["train_recon_mrna"][-1] + loss_tab["train_gmm"][-1]

        # Validation step
        loss_tab["test_loss"].append(0.)
        loss_tab["test_recon_mrna"].append(0.)
        loss_tab["test_recon_mirna"].append(0.)
        loss_tab["test_gmm"].append(0.)
        # loss_tab["test_mse"].append(0.)
        # loss_tab["test_mae"].append(0.)
        loss_tab["test_r2"].append(0.)
        loss_tab["test_spearman"].append(0.)
        loss_tab["test_pearson"].append(0.)
        # loss_tab["test_msle"].append(0.)
        loss_tab["delta_mirna"].append(0.)

        # Train the validation representation layer only using mRNA data
        val_rep_optimizer.zero_grad()
        dgd.eval() # Validation mode
        for (mrna_data, mirna_data, lib_mrna, lib_mirna, index, label) in validation_loader:
            mirna_recon_loss, mrna_recon_loss, gmm_loss = dgd.forward_and_loss(
                z=dgd.val_rep(index),
                target=[mirna_data.to(device), mrna_data.to(device)],  # Pass both mRNA and miRNA data
                scale=[lib_mirna.unsqueeze(1).to(device), lib_mrna.unsqueeze(1).to(device)],  # Pass both scales
                gmm_loss=gmm_loss,
                reduction=reduction_type,
                type="combined"
            )
            loss_tab["test_recon_mirna"][-1] += mirna_recon_loss.item()
            loss_tab["test_recon_mrna"][-1] += mrna_recon_loss.item()
            loss_tab["test_gmm"][-1] += gmm_loss.item()
            
            loss = mrna_recon_loss + gmm_loss
            loss.backward()
        val_rep_optimizer.step()

        # Update the scheduler
        if scheduler['type'] == "step" or scheduler['type'] == "exponential":
            val_rep_scheduler.step()

        # Calculate metrics
        with torch.inference_mode():
            if scaling_type == 'mean':
                scaling = torch.nanmean(validation_loader.dataset.mirna_data, axis=1)
            elif scaling_type == 'max':
                scaling = torch.nanmax(validation_loader.dataset.mirna_data, axis=1).values
            elif scaling_type == 'sum':
                scaling = torch.nansum(validation_loader.dataset.mirna_data, axis=1)

            mirna_recon, _ = dgd.forward(dgd.val_rep())
            mirna_recon = mirna_recon * scaling.unsqueeze(1).to(device)
            mirna_data = validation_loader.dataset.mirna_data.to(device)
            # Get subset
            mirna_recon_tpm = mirna_recon[:,subset]
            mirna_data_tpm = mirna_data[:,subset]
            # Calculate metrics
            # mirna_mse = mean_squared_error(mirna_recon_tpm, mirna_data_tpm)
            # loss_tab["test_mse"][-1] += mirna_mse.item()
            # mirna_mae = mean_absolute_error(mirna_recon_tpm, mirna_data_tpm)
            # loss_tab["test_mae"][-1] += mirna_mae.item()
            mirna_r2 = r2_score(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["test_r2"][-1] += mirna_r2.item()
            mirna_spearman = spearman_corrcoef(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["test_spearman"][-1] += mirna_spearman.item()
            mirna_pearson = pearson_corrcoef(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["test_pearson"][-1] += mirna_pearson.item()
            # mirna_msle = mean_squared_log_error(mirna_recon_tpm, mirna_data_tpm)
            # loss_tab["test_msle"][-1] += mirna_msle.item()

        loss_tab["test_recon_mirna"][-1] /= vlen_mirna
        loss_tab["test_recon_mrna"][-1] /= vlen_mrna
        loss_tab["test_gmm"][-1] /= vlen_gmm
        loss_tab["test_loss"][-1] = loss_tab["test_recon_mrna"][-1] + loss_tab["test_recon_mirna"][-1]  + loss_tab["test_gmm"][-1]
        loss_tab["delta_mirna"][-1] = loss_tab["test_recon_mirna"][-1] - loss_tab["train_recon_mirna"][-1]

        if pr>=0 and (epoch)%pr==0:
            print(epoch,
                  f"train_loss: {loss_tab['train_loss'][-1]}",
                  f"train_loss_decoder: {loss_tab['train_recon_mrna'][-1] + loss_tab['train_recon_mirna'][-1]}",
                  f"train_recon_mirna: {loss_tab['train_recon_mirna'][-1]}", 
                  f"train_recon_mrna: {loss_tab['train_recon_mrna'][-1]}", 
                  f"train_gmm: {loss_tab['train_gmm'][-1]}",
                  # f"train_mse: {loss_tab['train_mse'][-1]}",
                  # f"train_mae: {loss_tab['train_mae'][-1]}",
                  f"train_r2: {loss_tab['train_r2'][-1]}",
                  f"train_spearman: {loss_tab['train_spearman'][-1]}",
                  f"train_pearson: {loss_tab['train_pearson'][-1]}",
                  # f"train_msle: {loss_tab['train_msle'][-1]}"
                  )
            print(epoch,
                  f"test_loss: {loss_tab['test_loss'][-1]}",
                  f"test_loss_decoder: {loss_tab['test_recon_mrna'][-1] + loss_tab['test_recon_mirna'][-1]}",
                  f"test_recon_mirna: {loss_tab['test_recon_mirna'][-1]}",
                  f"test_recon_mrna: {loss_tab['test_recon_mrna'][-1]}", 
                  f"test_gmm: {loss_tab['test_gmm'][-1]}",
                  # f"test_mse: {loss_tab['test_mse'][-1]}",
                  # f"test_mae: {loss_tab['test_mae'][-1]}",
                  f"test_r2: {loss_tab['test_r2'][-1]}",
                  f"test_spearman: {loss_tab['test_spearman'][-1]}",
                  f"test_pearson: {loss_tab['test_pearson'][-1]}",
                  # f"test_msle: {loss_tab['test_msle'][-1]}",
                  f"delta_mirna: {loss_tab['delta_mirna'][-1]}"
                  )
        if is_plot:
            if plot>=0 and (epoch)%plot==0:
                if is_gtex:
                    plot_latent_space_multi(*dgd.get_latent_space_values("train",3000), train_loader.dataset.primary_site, train_loader.dataset.batch, color_mapping_tissue, epoch, dataset="Train")
                    plot_latent_space_multi(*dgd.get_latent_space_values("val",3000), validation_loader.dataset.primary_site, validation_loader.dataset.batch, color_mapping_tissue, epoch, dataset="Validation")
                    plot_mirna(dgd, train_loader, sample_index, epoch, device, fold=fold, type="Train", scaling_type=scaling_type)
                    plot_mirna(dgd, validation_loader, sample_index, epoch, device, fold=fold, type="Validation", scaling_type=scaling_type)
                    plot_gene(dgd, train_loader, sample_index, epoch, device, fold=fold, type="Train", scaling_type=scaling_type)
                    plot_gene(dgd, validation_loader, sample_index, epoch, device, fold=fold, type="Validation", scaling_type=scaling_type)
                else:
                    plot_latent_space(*dgd.get_latent_space_values("train",3000), train_loader.dataset.cancer_type, color_mapping_cancer, epoch, dataset="Train")
                    plot_latent_space(*dgd.get_latent_space_values("val",3000), validation_loader.dataset.cancer_type, color_mapping_cancer, epoch, dataset="Validation")
                    plot_mirna(dgd, train_loader, sample_index, epoch, device, fold=fold, type="Train", scaling_type=scaling_type)
                    plot_mirna(dgd, validation_loader, sample_index, epoch, device, fold=fold, type="Validation", scaling_type=scaling_type)
                    plot_gene(dgd, train_loader, sample_index, epoch, device, fold=fold, type="Train", scaling_type=scaling_type)
                    plot_gene(dgd, validation_loader, sample_index, epoch, device, fold=fold, type="Validation", scaling_type=scaling_type)
            
        if wandb_log: 
            wandb.log({
                "fold": fold,
                "epoch": epoch,
                "train_loss": loss_tab["train_loss"][-1],
                "train_recon_mrna": loss_tab["train_recon_mrna"][-1],
                "train_recon_mirna": loss_tab["train_recon_mirna"][-1],
                "train_gmm": loss_tab["train_gmm"][-1],
                # "train_mse": loss_tab["train_mse"][-1],
                # "train_mae": loss_tab["train_mae"][-1],
                "train_r2": loss_tab["train_r2"][-1],
                "train_spearman": loss_tab["train_spearman"][-1],
                "train_pearson": loss_tab["train_pearson"][-1],
                # "train_msle": loss_tab["train_msle"][-1],
                "test_loss": loss_tab["test_loss"][-1],
                "test_loss_decoder": loss_tab["test_recon_mrna"][-1] + loss_tab["test_recon_mirna"][-1],
                "test_recon_mrna": loss_tab["test_recon_mrna"][-1],
                "test_recon_mirna": loss_tab["test_recon_mirna"][-1],
                "test_gmm": loss_tab["test_gmm"][-1],
                # "test_mse": loss_tab["test_mse"][-1],
                # "test_mae": loss_tab["test_mae"][-1],
                "test_r2": loss_tab["test_r2"][-1],
                "test_spearman": loss_tab["test_spearman"][-1],
                "test_pearson": loss_tab["test_pearson"][-1],
                # "test_msle": loss_tab["test_msle"][-1],
                "delta_mirna": loss_tab["delta_mirna"][-1],
            })

        # Early stopping
        if epoch > 100:
            if early_stopping:
                current_loss_mrna = loss_tab["test_recon_mrna"][-1]
                current_loss_mirna = loss_tab["test_recon_mirna"][-1]

                # Track epochs since improvement for each metric
                if current_loss_mrna < best_loss_mrna:
                    best_loss_mrna = current_loss_mrna
                    best_epoch_mrna = epoch
                    checkpoint_mrna = dgd
                if current_loss_mirna < best_loss_mirna:
                    best_loss_mirna = current_loss_mirna
                    best_epoch_mirna = epoch
                    checkpoint_mirna = dgd

                # If both haven't improved for 'early_stopping' epochs, stop
                if (epoch - best_epoch_mrna > early_stopping) and (epoch - best_epoch_mirna > early_stopping):
                    print(f"Early stopped training at epoch {epoch} "
                        f"with losses mrna={best_loss_mrna:.4f}, mirna={best_loss_mirna:.4f}")
                    # Restore to the best overall checkpoint (can choose which based on which is lower or latest)
                    # Example: Restore model to the most recent best checkpoint
                    dgd = checkpoint_mrna if best_epoch_mrna > best_epoch_mirna else checkpoint_mirna
                    break

      
    
    # Training done!
    return loss_tab


# Training functions
def train_midgd_supervised(
                dgd, train_loader, validation_loader, device,
                learning_rates={'dec':0.0001,'rep':0.01,'gmm':0.01}, 
                weight_decay=0., betas=(0.9, 0.999), scheduler=None,
                nepochs=100, fold=None, pr=1, plot=10, 
                reduction_type="sum", scaling_type="mean",
                sample_index=[0,11,22,33], subset=1310, wandb_log=False, early_stopping=50, is_plot=True):
    '''
    Should be used with SupervisedDGD model only.
    '''
    # Normalization factor
    if reduction_type == "sum":
        tlen=len(train_loader.dataset)*dgd.decoder.n_out_features
        vlen=len(validation_loader.dataset)*dgd.decoder.n_out_features
        tlen_gmm=len(train_loader.dataset)*dgd.gmm.n_mix_comp*dgd.gmm.dim
        vlen_gmm=len(validation_loader.dataset)*dgd.gmm.n_mix_comp*dgd.gmm.dim

        tlen_mirna = len(train_loader.dataset)*dgd.decoder.n_out_features_mirna
        vlen_mirna = len(validation_loader.dataset)*dgd.decoder.n_out_features_mirna
        tlen_mrna = len(train_loader.dataset)*dgd.decoder.n_out_features_mrna
        vlen_mrna = len(validation_loader.dataset)*dgd.decoder.n_out_features_mrna
    else:
        tlen=len(train_loader)
        vlen=len(validation_loader)
        tlen_gmm=len(train_loader)
        vlen_gmm=len(validation_loader)

    # Train Representation Layer Initialization
    Ntrain=len(train_loader.dataset)
    if dgd.train_rep is None:
        dgd.train_rep = RepresentationLayer(dgd.rep_dim,Ntrain).to(device)

    # Test/Validation Representation Layer Initialization
    Nvalidation=len(validation_loader.dataset)
    if dgd.val_rep is None:
        dgd.val_rep = RepresentationLayer(dgd.rep_dim,Nvalidation).to(device)

    # Optimizer Initialization
    dec_optimizer = torch.optim.AdamW(dgd.decoder.parameters(), lr=learning_rates['dec'], weight_decay=weight_decay['dec'], betas=betas)
    gmm_optimizer = torch.optim.AdamW(dgd.gmm.parameters(), lr=learning_rates['gmm'], weight_decay=weight_decay['gmm'], betas=betas)
    train_rep_optimizer = torch.optim.AdamW(dgd.train_rep.parameters(), lr=learning_rates['rep'], weight_decay=weight_decay['rep'], betas=betas)
    val_rep_optimizer = torch.optim.AdamW(dgd.val_rep.parameters(), lr=learning_rates['rep'], weight_decay=weight_decay['rep'], betas=betas)

    # Scheduler Initialization
    if scheduler['type'] == "step":
        dec_scheduler = torch.optim.lr_scheduler.StepLR(dec_optimizer, step_size=scheduler['step_size'], gamma=scheduler['gamma_step'])
        gmm_scheduler = torch.optim.lr_scheduler.StepLR(gmm_optimizer, step_size=scheduler['step_size'], gamma=scheduler['gamma_step'])
        train_rep_scheduler = torch.optim.lr_scheduler.StepLR(train_rep_optimizer, step_size=scheduler['step_size'], gamma=scheduler['gamma_step'])
        val_rep_scheduler = torch.optim.lr_scheduler.StepLR(val_rep_optimizer, step_size=scheduler['step_size'], gamma=scheduler['gamma_step'])
    elif scheduler['type'] == "exponential":
        dec_scheduler = torch.optim.lr_scheduler.ExponentialLR(dec_optimizer, gamma=scheduler['gamma_exp'])
        gmm_scheduler = torch.optim.lr_scheduler.ExponentialLR(gmm_optimizer, gamma=scheduler['gamma_exp'])
        train_rep_scheduler = torch.optim.lr_scheduler.ExponentialLR(train_rep_optimizer, gamma=scheduler['gamma_exp'])
        val_rep_scheduler = torch.optim.lr_scheduler.ExponentialLR(val_rep_optimizer, gamma=scheduler['gamma_exp'])

    # Metrics logger initialization
    loss_tab = {"epoch":[],
                "train_loss":[], "test_loss":[],
                "train_recon_mirna":[],"train_recon_mrna":[], 
                "test_recon_mirna":[], "test_recon_mrna":[],
                "train_gmm":[],"test_gmm":[], 
                "train_mse":[], "test_mse":[],
                "train_mae":[], "test_mae":[],
                "train_r2":[], "test_r2":[],
                "train_spearman":[], "test_spearman":[],
                "test_pearson":[], "train_pearson":[],
                "train_msle":[], "test_msle":[]}
    gmm_loss=True

    # For custom color mapping
    color_mapping = dict(zip(train_loader.dataset.str_labels, train_loader.dataset.color))
    
    # Early stopping
    best_loss = 1e20
    best_epoch = -1

    # Start training
    for epoch in tqdm(range(nepochs)):
        # Train step
        loss_tab["epoch"].append(epoch)
        loss_tab["train_loss"].append(0.)
        loss_tab["train_recon_mirna"].append(0.)
        loss_tab["train_recon_mrna"].append(0.)
        loss_tab["train_gmm"].append(0.)
        loss_tab["train_mse"].append(0.)
        loss_tab["train_mae"].append(0.)
        loss_tab["train_r2"].append(0.)
        loss_tab["train_spearman"].append(0.)
        loss_tab["train_pearson"].append(0.)
        loss_tab["train_msle"].append(0.)

        train_rep_optimizer.zero_grad()
        dgd.train() # Training 

        for (mrna_data, mirna_data, lib_mrna, lib_mirna, index, label) in train_loader:
            dec_optimizer.zero_grad()
            if gmm_loss: 
                gmm_optimizer.zero_grad()
            mirna_recon_loss, mrna_recon_loss, gmm_loss = dgd.forward_and_loss(
                z=dgd.train_rep(index),
                target=[mirna_data.to(device), mrna_data.to(device)],  # Pass both mRNA and miRNA data
                scale=[lib_mirna.unsqueeze(1).to(device), lib_mrna.unsqueeze(1).to(device)],  # Pass both scales
                gmm_loss=gmm_loss,
                reduction=reduction_type,
                type="combined",
                label=label
            )
            
            loss_tab["train_recon_mirna"][-1] += mirna_recon_loss.item()
            loss_tab["train_recon_mrna"][-1] += mrna_recon_loss.item()
            loss_tab["train_gmm"][-1] += gmm_loss.item()

            loss =  mrna_recon_loss + mirna_recon_loss + gmm_loss
            loss.backward()
            dec_optimizer.step()
            if gmm_loss: 
                gmm_optimizer.step()
        train_rep_optimizer.step()

        # Update the scheduler
        if scheduler['type'] == "step" or scheduler['type'] == "exponential":
            dec_scheduler.step()
            gmm_scheduler.step()
            train_rep_scheduler.step()

        # Calculate metrics
        with torch.inference_mode():
            # Get data
            if scaling_type == 'mean':
                scaling = torch.nanmean(train_loader.dataset.mirna_data, axis=1)
            elif scaling_type == 'max':
                scaling = torch.nanmax(train_loader.dataset.mirna_data, axis=1).values
            elif scaling_type == 'sum':
                scaling = torch.nansum(train_loader.dataset.mirna_data, axis=1)

            mirna_recon, _ = dgd.forward(dgd.train_rep())
            mirna_recon = mirna_recon * scaling.unsqueeze(1).to(device)
            mirna_data = train_loader.dataset.mirna_data.to(device)
            # Get subset
            mirna_recon_tpm = mirna_recon[:,subset]
            mirna_data_tpm = mirna_data[:,subset]
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
            mirna_msle = mean_squared_log_error(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["train_msle"][-1] += mirna_msle.item()

        loss_tab["train_recon_mirna"][-1] /= tlen_mirna
        loss_tab["train_recon_mrna"][-1] /= tlen_mrna
        loss_tab["train_gmm"][-1] /= tlen_gmm
        loss_tab["train_loss"][-1] = loss_tab["train_recon_mirna"][-1] + loss_tab["train_recon_mrna"][-1] + loss_tab["train_gmm"][-1]

        # Validation step
        loss_tab["test_loss"].append(0.)
        loss_tab["test_recon_mrna"].append(0.)
        loss_tab["test_recon_mirna"].append(0.)
        loss_tab["test_gmm"].append(0.)
        loss_tab["test_mse"].append(0.)
        loss_tab["test_mae"].append(0.)
        loss_tab["test_r2"].append(0.)
        loss_tab["test_spearman"].append(0.)
        loss_tab["test_pearson"].append(0.)
        loss_tab["test_msle"].append(0.)
        loss_tab["delta_mirna"].append(0.)

        # Train the validation representation layer only using mRNA data
        val_rep_optimizer.zero_grad()
        dgd.eval() # Validation mode
        for (mrna_data, mirna_data, lib_mrna, lib_mirna, index, label) in validation_loader:
            mirna_recon_loss, mrna_recon_loss, gmm_loss = dgd.forward_and_loss(
                z=dgd.val_rep(index),
                target=[mirna_data.to(device), mrna_data.to(device)],  # Pass both mRNA and miRNA data
                scale=[lib_mirna.unsqueeze(1).to(device), lib_mrna.unsqueeze(1).to(device)],  # Pass both scales
                gmm_loss=gmm_loss,
                reduction=reduction_type,
                type="combined",
                label=label
            )
            loss_tab["test_recon_mirna"][-1] += mirna_recon_loss.item()
            loss_tab["test_recon_mrna"][-1] += mrna_recon_loss.item()
            loss_tab["test_gmm"][-1] += gmm_loss.item()
            
            loss = mrna_recon_loss + gmm_loss
            loss.backward()
        val_rep_optimizer.step()

        # Update the scheduler
        if scheduler['type'] == "step" or scheduler['type'] == "exponential":
            val_rep_scheduler.step()

        # Calculate metrics
        with torch.inference_mode():
            if scaling_type == 'mean':
                scaling = torch.nanmean(validation_loader.dataset.mirna_data, axis=1)
            elif scaling_type == 'max':
                scaling = torch.nanmax(validation_loader.dataset.mirna_data, axis=1).values
            elif scaling_type == 'sum':
                scaling = torch.nansum(validation_loader.dataset.mirna_data, axis=1)

            mirna_recon, _ = dgd.forward(dgd.val_rep())
            mirna_recon = mirna_recon * scaling.unsqueeze(1).to(device)
            mirna_data = validation_loader.dataset.mirna_data.to(device)
            # Get subset
            mirna_recon_tpm = mirna_recon[:,subset]
            mirna_data_tpm = mirna_data[:,subset]
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
            mirna_msle = mean_squared_log_error(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["test_msle"][-1] += mirna_msle.item()

        loss_tab["test_recon_mirna"][-1] /= vlen_mirna
        loss_tab["test_recon_mrna"][-1] /= vlen_mrna
        loss_tab["test_gmm"][-1] /= vlen_gmm
        loss_tab["test_loss"][-1] = loss_tab["test_recon_mrna"][-1] + loss_tab["test_recon_mirna"][-1]  + loss_tab["test_gmm"][-1]
        loss_tab["delta_mirna"][-1] = loss_tab["test_recon_mirna"][-1] - loss_tab["train_recon_mirna"][-1]
        
        if pr>=0 and (epoch)%pr==0:
            print(epoch,
                  f"train_loss: {loss_tab['train_loss'][-1]}",
                  f"train_loss_decoder: {loss_tab['train_recon_mrna'][-1] + loss_tab['train_recon_mirna'][-1]}",
                  f"train_recon_mirna: {loss_tab['train_recon_mirna'][-1]}", 
                  f"train_recon_mrna: {loss_tab['train_recon_mrna'][-1]}", 
                  f"train_gmm: {loss_tab['train_gmm'][-1]}",
                  f"train_mse: {loss_tab['train_mse'][-1]}",
                  f"train_mae: {loss_tab['train_mae'][-1]}",
                  f"train_r2: {loss_tab['train_r2'][-1]}",
                  f"train_spearman: {loss_tab['train_spearman'][-1]}",
                  f"train_pearson: {loss_tab['train_pearson'][-1]}",
                  f"train_msle: {loss_tab['train_msle'][-1]}")
            print(epoch,
                  f"test_loss: {loss_tab['test_loss'][-1]}",
                  f"test_loss_decoder: {loss_tab['test_recon_mrna'][-1] + loss_tab['test_recon_mirna'][-1]}",
                  f"test_recon_mirna: {loss_tab['test_recon_mirna'][-1]}",
                  f"test_recon_mrna: {loss_tab['test_recon_mrna'][-1]}", 
                  f"test_gmm: {loss_tab['test_gmm'][-1]}",
                  f"test_mse: {loss_tab['test_mse'][-1]}",
                  f"test_mae: {loss_tab['test_mae'][-1]}",
                  f"test_r2: {loss_tab['test_r2'][-1]}",
                  f"test_spearman: {loss_tab['test_spearman'][-1]}",
                  f"test_pearson: {loss_tab['test_pearson'][-1]}",
                  f"test_msle: {loss_tab['test_msle'][-1]}",
                    f"delta_mirna: {loss_tab['delta_mirna'][-1]}"
                )
        if is_plot:
            if plot>=0 and (epoch)%plot==0:
                plot_latent_space(*dgd.get_latent_space_values("train",3000), train_loader.dataset.str_labels, color_mapping, epoch, dataset="Train")
                plot_latent_space(*dgd.get_latent_space_values("val",3000), validation_loader.dataset.str_labels, color_mapping, epoch, dataset="Validation")
                plot_mirna(dgd, train_loader, sample_index, epoch, device, fold=fold, type="Train", scaling_type=scaling_type)
                plot_mirna(dgd, validation_loader, sample_index, epoch, device, fold=fold, type="Validation", scaling_type=scaling_type)
                plot_gene(dgd, train_loader, sample_index, epoch, device, fold=fold, type="Train", scaling_type=scaling_type)
                plot_gene(dgd, validation_loader, sample_index, epoch, device, fold=fold, type="Validation", scaling_type=scaling_type)
            
        if wandb_log: 
            wandb.log({
                "fold": fold,
                "epoch": epoch,
                "train_loss": loss_tab["train_loss"][-1],
                "train_recon_mrna": loss_tab["train_recon_mrna"][-1],
                "train_recon_mirna": loss_tab["train_recon_mirna"][-1],
                "train_gmm": loss_tab["train_gmm"][-1],
                "train_mse": loss_tab["train_mse"][-1],
                "train_mae": loss_tab["train_mae"][-1],
                "train_r2": loss_tab["train_r2"][-1],
                "train_spearman": loss_tab["train_spearman"][-1],
                "train_pearson": loss_tab["train_pearson"][-1],
                "train_msle": loss_tab["train_msle"][-1],
                "test_loss": loss_tab["test_loss"][-1],
                "test_recon_mrna": loss_tab["test_recon_mrna"][-1],
                "test_recon_mirna": loss_tab["test_recon_mirna"][-1],
                "test_gmm": loss_tab["test_gmm"][-1],
                "test_mse": loss_tab["test_mse"][-1],
                "test_mae": loss_tab["test_mae"][-1],
                "test_r2": loss_tab["test_r2"][-1],
                "test_spearman": loss_tab["test_spearman"][-1],
                "test_pearson": loss_tab["test_pearson"][-1],
                "test_msle": loss_tab["test_msle"][-1],
                "delta_mirna": loss_tab["delta_mirna"][-1]
            })

        # Early stopping
        if epoch > 200:
            if early_stopping:
                current_loss = loss_tab["test_recon_mrna"][-1] + loss_tab["test_recon_mirna"][-1]
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_epoch = epoch
                    checkpoint = dgd
                elif epoch - best_epoch > early_stopping:
                    print(f"Early stopped training at epoch {epoch} with loss {best_loss}")
                    dgd = checkpoint
                    break  # terminate the training loop        
    
    # Training done!
    return loss_tab


# Training functions
def train_midgd_new(dgd, train_loader, validation_loader, device,
                learning_rates={'dec':0.0001,'rep':0.01,'gmm':0.01}, 
                weight_decay=0., betas=(0.9, 0.999), scheduler=None,
                nepochs=100, fold=None, pr=1, plot=10, 
                reduction_type="sum", scaling_type="mean",
                sample_index=[0,11,22,33], subset=1310, wandb_log=False, early_stopping=50, is_plot=True):
    # Normalization factor
    if reduction_type == "sum":
        tlen=len(train_loader.dataset)*dgd.decoder.n_out_features
        vlen=len(validation_loader.dataset)*dgd.decoder.n_out_features
        tlen_gmm=len(train_loader.dataset)*dgd.gmm.n_mix_comp*dgd.gmm.dim
        vlen_gmm=len(validation_loader.dataset)*dgd.gmm.n_mix_comp*dgd.gmm.dim

        tlen_mirna = len(train_loader.dataset)*dgd.decoder.n_out_features_mirna
        vlen_mirna = len(validation_loader.dataset)*dgd.decoder.n_out_features_mirna
        tlen_mrna = len(train_loader.dataset)*dgd.decoder.n_out_features_mrna
        vlen_mrna = len(validation_loader.dataset)*dgd.decoder.n_out_features_mrna
    else:
        tlen=len(train_loader)
        vlen=len(validation_loader)
        tlen_gmm=len(train_loader)
        vlen_gmm=len(validation_loader)

    # Train Representation Layer Initialization
    Ntrain=len(train_loader.dataset)
    if dgd.train_rep is None:
        dgd.train_rep = RepresentationLayer(dgd.rep_dim,Ntrain).to(device)

    # Test/Validation Representation Layer Initialization
    Nvalidation=len(validation_loader.dataset)
    if dgd.val_rep is None:
        dgd.val_rep = RepresentationLayer(dgd.rep_dim,Nvalidation).to(device)

    # Optimizer Initialization
    dec_optimizer = torch.optim.AdamW(dgd.decoder.parameters(), lr=learning_rates['dec'], weight_decay=weight_decay['dec'], betas=betas)
    gmm_optimizer = torch.optim.AdamW(dgd.gmm.parameters(), lr=learning_rates['gmm'], weight_decay=weight_decay['gmm'], betas=betas)
    train_rep_optimizer = torch.optim.AdamW(dgd.train_rep.parameters(), lr=learning_rates['rep'], weight_decay=weight_decay['rep'], betas=betas)
    val_rep_optimizer = torch.optim.AdamW(dgd.val_rep.parameters(), lr=learning_rates['rep'], weight_decay=weight_decay['rep'], betas=betas)

    # Scheduler Initialization
    if scheduler['type'] == "step":
        dec_scheduler = torch.optim.lr_scheduler.StepLR(dec_optimizer, step_size=scheduler['step_size'], gamma=scheduler['gamma_step'])
        gmm_scheduler = torch.optim.lr_scheduler.StepLR(gmm_optimizer, step_size=scheduler['step_size'], gamma=scheduler['gamma_step'])
        train_rep_scheduler = torch.optim.lr_scheduler.StepLR(train_rep_optimizer, step_size=scheduler['step_size'], gamma=scheduler['gamma_step'])
        val_rep_scheduler = torch.optim.lr_scheduler.StepLR(val_rep_optimizer, step_size=scheduler['step_size'], gamma=scheduler['gamma_step'])
    elif scheduler['type'] == "exponential":
        dec_scheduler = torch.optim.lr_scheduler.ExponentialLR(dec_optimizer, gamma=scheduler['gamma_exp'])
        gmm_scheduler = torch.optim.lr_scheduler.ExponentialLR(gmm_optimizer, gamma=scheduler['gamma_exp'])
        train_rep_scheduler = torch.optim.lr_scheduler.ExponentialLR(train_rep_optimizer, gamma=scheduler['gamma_exp'])
        val_rep_scheduler = torch.optim.lr_scheduler.ExponentialLR(val_rep_optimizer, gamma=scheduler['gamma_exp'])

    # Metrics logger initialization
    loss_tab = {"epoch":[],
                "train_loss":[], "test_loss":[],
                "train_recon_mirna":[],"train_recon_mrna":[], 
                "test_recon_mirna":[], "test_recon_mrna":[],
                "train_gmm":[],"test_gmm":[], 
                "train_mse":[], "test_mse":[],
                "train_mae":[], "test_mae":[],
                "train_r2":[], "test_r2":[],
                "train_spearman":[], "test_spearman":[],
                "test_pearson":[], "train_pearson":[],
                "train_msle":[], "test_msle":[],
                "delta_mirna":[]}
    gmm_loss=True

    # For custom color mapping
    train_mapping_cancer = dict(zip(train_loader.dataset.cancer_type, train_loader.dataset.color))
    val_mapping_cancer = dict(zip(validation_loader.dataset.cancer_type, validation_loader.dataset.color))
    color_mapping_cancer = train_mapping_cancer.copy()
    color_mapping_cancer.update(val_mapping_cancer)


    train_mapping_tissue = dict(zip(train_loader.dataset.primary_site, train_loader.dataset.color))
    val_mapping_tissue = dict(zip(validation_loader.dataset.primary_site, validation_loader.dataset.color))
    color_mapping_tissue = train_mapping_tissue.copy()
    color_mapping_tissue.update(val_mapping_tissue)

    # Check if using GTEx dataset
    is_gtex = "GTEx" in train_loader.dataset.batch

    
    # Early stopping
    best_loss = 1e20
    best_loss_mrna = 1e20
    best_loss_mirna = 1e20
    best_epoch = -1

    # Start training
    for epoch in tqdm(range(nepochs)):
        # Train step
        loss_tab["epoch"].append(epoch)
        loss_tab["train_loss"].append(0.)
        loss_tab["train_recon_mirna"].append(0.)
        loss_tab["train_recon_mrna"].append(0.)
        loss_tab["train_gmm"].append(0.)
        # loss_tab["train_mse"].append(0.)
        # loss_tab["train_mae"].append(0.)
        loss_tab["train_r2"].append(0.)
        loss_tab["train_spearman"].append(0.)
        loss_tab["train_pearson"].append(0.)
        # loss_tab["train_msle"].append(0.)

        train_rep_optimizer.zero_grad()
        dgd.train() # Training 

        for (mrna_data, mirna_data, lib_mrna, lib_mirna, index, label) in train_loader:
            dec_optimizer.zero_grad()
            if gmm_loss: 
                gmm_optimizer.zero_grad()
            mirna_recon_loss, mrna_recon_loss, gmm_loss = dgd.forward_and_loss(
                z=dgd.train_rep(index),
                target=[mirna_data.to(device), mrna_data.to(device)],  # Pass both mRNA and miRNA data
                scale=[lib_mirna.unsqueeze(1).to(device), lib_mrna.unsqueeze(1).to(device)],  # Pass both scales
                gmm_loss=gmm_loss,
                reduction=reduction_type,
                type="combined"
            )
            
            loss_tab["train_recon_mirna"][-1] += mirna_recon_loss.item()
            loss_tab["train_recon_mrna"][-1] += mrna_recon_loss.item()
            loss_tab["train_gmm"][-1] += gmm_loss.item()

            loss =  mrna_recon_loss + mirna_recon_loss + gmm_loss
            loss.backward()
            dec_optimizer.step()
            if gmm_loss: 
                gmm_optimizer.step()
        train_rep_optimizer.step()

        # Update the scheduler
        if scheduler['type'] == "step" or scheduler['type'] == "exponential":
            dec_scheduler.step()
            gmm_scheduler.step()
            train_rep_scheduler.step()

        # Calculate metrics
        with torch.inference_mode():
            # Get data
            if scaling_type == 'mean':
                scaling = torch.nanmean(train_loader.dataset.mirna_data, axis=1)
            elif scaling_type == 'max':
                scaling = torch.nanmax(train_loader.dataset.mirna_data, axis=1).values
            elif scaling_type == 'sum':
                scaling = torch.nansum(train_loader.dataset.mirna_data, axis=1)

            mirna_recon, _ = dgd.forward(dgd.train_rep())
            mirna_recon = mirna_recon * scaling.unsqueeze(1).to(device)
            mirna_data = train_loader.dataset.mirna_data.to(device)
            
            # Device-agnostic average per-feature metrics across all miRNA features
            mirna_recon_all = mirna_recon  # shape: [batch_size, n_mirna_features]
            mirna_data_all = mirna_data    # shape: [batch_size, n_mirna_features]

            mirna_recon_torch = mirna_recon_all.detach().cpu()  # still a torch tensor on CPU
            mirna_data_torch = mirna_data_all.detach().cpu()

            mirna_r2_per_feature = [r2_score(mirna_recon_torch[:, i], mirna_data_torch[:, i]) for i in range(mirna_data_torch.shape[1])]
            mirna_r2_avg = torch.stack(mirna_r2_per_feature).mean().item()
            loss_tab["train_r2"][-1] += mirna_r2_avg

            mirna_spearman_per_feature = [spearman_corrcoef(mirna_recon_torch[:, i], mirna_data_torch[:, i]) for i in range(mirna_data_torch.shape[1])]
            mirna_spearman_avg = torch.stack(mirna_spearman_per_feature).mean().item()
            loss_tab["train_spearman"][-1] += mirna_spearman_avg

            mirna_pearson_per_feature = [pearson_corrcoef(mirna_recon_torch[:, i], mirna_data_torch[:, i]) for i in range(mirna_data_torch.shape[1])]
            mirna_pearson_avg = torch.stack(mirna_pearson_per_feature).mean().item()
            loss_tab["train_pearson"][-1] += mirna_pearson_avg



        loss_tab["train_recon_mirna"][-1] /= tlen_mirna
        loss_tab["train_recon_mrna"][-1] /= tlen_mrna
        loss_tab["train_gmm"][-1] /= tlen_gmm
        loss_tab["train_loss"][-1] = loss_tab["train_recon_mirna"][-1] + loss_tab["train_recon_mrna"][-1] + loss_tab["train_gmm"][-1]

        # Validation step
        loss_tab["test_loss"].append(0.)
        loss_tab["test_recon_mrna"].append(0.)
        loss_tab["test_recon_mirna"].append(0.)
        loss_tab["test_gmm"].append(0.)
        # loss_tab["test_mse"].append(0.)
        # loss_tab["test_mae"].append(0.)
        loss_tab["test_r2"].append(0.)
        loss_tab["test_spearman"].append(0.)
        loss_tab["test_pearson"].append(0.)
        # loss_tab["test_msle"].append(0.)
        loss_tab["delta_mirna"].append(0.)

        # Train the validation representation layer only using mRNA data
        val_rep_optimizer.zero_grad()
        dgd.eval() # Validation mode
        for (mrna_data, mirna_data, lib_mrna, lib_mirna, index, label) in validation_loader:
            mirna_recon_loss, mrna_recon_loss, gmm_loss = dgd.forward_and_loss(
                z=dgd.val_rep(index),
                target=[mirna_data.to(device), mrna_data.to(device)],  # Pass both mRNA and miRNA data
                scale=[lib_mirna.unsqueeze(1).to(device), lib_mrna.unsqueeze(1).to(device)],  # Pass both scales
                gmm_loss=gmm_loss,
                reduction=reduction_type,
                type="combined"
            )
            loss_tab["test_recon_mirna"][-1] += mirna_recon_loss.item()
            loss_tab["test_recon_mrna"][-1] += mrna_recon_loss.item()
            loss_tab["test_gmm"][-1] += gmm_loss.item()
            
            loss = mrna_recon_loss + gmm_loss
            loss.backward()
        val_rep_optimizer.step()

        # Update the scheduler
        if scheduler['type'] == "step" or scheduler['type'] == "exponential":
            val_rep_scheduler.step()

        # Calculate metrics
        with torch.inference_mode():
            if scaling_type == 'mean':
                scaling = torch.nanmean(validation_loader.dataset.mirna_data, axis=1)
            elif scaling_type == 'max':
                scaling = torch.nanmax(validation_loader.dataset.mirna_data, axis=1).values
            elif scaling_type == 'sum':
                scaling = torch.nansum(validation_loader.dataset.mirna_data, axis=1)

            mirna_recon, _ = dgd.forward(dgd.val_rep())
            mirna_recon = mirna_recon * scaling.unsqueeze(1).to(device)
            mirna_data = validation_loader.dataset.mirna_data.to(device)
            
            # Device-agnostic average per-feature metrics across all miRNA features
            mirna_recon_all = mirna_recon  # shape: [batch_size, n_mirna_features]
            mirna_data_all = mirna_data    # shape: [batch_size, n_mirna_features]

            # Move to CPU and convert to numpy
            mirna_recon_torch = mirna_recon_all.detach().cpu()  # still a torch tensor on CPU
            mirna_data_torch = mirna_data_all.detach().cpu()

            mirna_r2_per_feature = [r2_score(mirna_recon_torch[:, i], mirna_data_torch[:, i]) for i in range(mirna_data_torch.shape[1])]
            mirna_r2_avg = torch.stack(mirna_r2_per_feature).mean().item()
            loss_tab["train_r2"][-1] += mirna_r2_avg

            mirna_spearman_per_feature = [spearman_corrcoef(mirna_recon_torch[:, i], mirna_data_torch[:, i]) for i in range(mirna_data_torch.shape[1])]
            mirna_spearman_avg = torch.stack(mirna_spearman_per_feature).mean().item()
            loss_tab["train_spearman"][-1] += mirna_spearman_avg

            mirna_pearson_per_feature = [pearson_corrcoef(mirna_recon_torch[:, i], mirna_data_torch[:, i]) for i in range(mirna_data_torch.shape[1])]
            mirna_pearson_avg = torch.stack(mirna_pearson_per_feature).mean().item()
            loss_tab["train_pearson"][-1] += mirna_pearson_avg


        loss_tab["test_recon_mirna"][-1] /= vlen_mirna
        loss_tab["test_recon_mrna"][-1] /= vlen_mrna
        loss_tab["test_gmm"][-1] /= vlen_gmm
        loss_tab["test_loss"][-1] = loss_tab["test_recon_mrna"][-1] + loss_tab["test_recon_mirna"][-1]  + loss_tab["test_gmm"][-1]
        loss_tab["delta_mirna"][-1] = loss_tab["test_recon_mirna"][-1] - loss_tab["train_recon_mirna"][-1]

        if pr>=0 and (epoch)%pr==0:
            print(epoch,
                  f"train_loss: {loss_tab['train_loss'][-1]}",
                  f"train_loss_decoder: {loss_tab['train_recon_mrna'][-1] + loss_tab['train_recon_mirna'][-1]}",
                  f"train_recon_mirna: {loss_tab['train_recon_mirna'][-1]}", 
                  f"train_recon_mrna: {loss_tab['train_recon_mrna'][-1]}", 
                  f"train_gmm: {loss_tab['train_gmm'][-1]}",
                  # f"train_mse: {loss_tab['train_mse'][-1]}",
                  # f"train_mae: {loss_tab['train_mae'][-1]}",
                  f"train_r2: {loss_tab['train_r2'][-1]}",
                  f"train_spearman: {loss_tab['train_spearman'][-1]}",
                  f"train_pearson: {loss_tab['train_pearson'][-1]}",
                  # f"train_msle: {loss_tab['train_msle'][-1]}"
                  )
            print(epoch,
                  f"test_loss: {loss_tab['test_loss'][-1]}",
                  f"test_loss_decoder: {loss_tab['test_recon_mrna'][-1] + loss_tab['test_recon_mirna'][-1]}",
                  f"test_recon_mirna: {loss_tab['test_recon_mirna'][-1]}",
                  f"test_recon_mrna: {loss_tab['test_recon_mrna'][-1]}", 
                  f"test_gmm: {loss_tab['test_gmm'][-1]}",
                  # f"test_mse: {loss_tab['test_mse'][-1]}",
                  # f"test_mae: {loss_tab['test_mae'][-1]}",
                  f"test_r2: {loss_tab['test_r2'][-1]}",
                  f"test_spearman: {loss_tab['test_spearman'][-1]}",
                  f"test_pearson: {loss_tab['test_pearson'][-1]}",
                  # f"test_msle: {loss_tab['test_msle'][-1]}",
                  f"delta_mirna: {loss_tab['delta_mirna'][-1]}"
                  )
        if is_plot:
            if plot>=0 and (epoch)%plot==0:
                if is_gtex:
                    plot_latent_space_multi(*dgd.get_latent_space_values("train",3000), train_loader.dataset.primary_site, train_loader.dataset.batch, color_mapping_tissue, epoch, dataset="Train")
                    plot_latent_space_multi(*dgd.get_latent_space_values("val",3000), validation_loader.dataset.primary_site, validation_loader.dataset.batch, color_mapping_tissue, epoch, dataset="Validation")
                    plot_mirna(dgd, train_loader, sample_index, epoch, device, fold=fold, type="Train", scaling_type=scaling_type)
                    plot_mirna(dgd, validation_loader, sample_index, epoch, device, fold=fold, type="Validation", scaling_type=scaling_type)
                    plot_gene(dgd, train_loader, sample_index, epoch, device, fold=fold, type="Train", scaling_type=scaling_type)
                    plot_gene(dgd, validation_loader, sample_index, epoch, device, fold=fold, type="Validation", scaling_type=scaling_type)
                else:
                    plot_latent_space(*dgd.get_latent_space_values("train",3000), train_loader.dataset.cancer_type, color_mapping_cancer, epoch, dataset="Train")
                    plot_latent_space(*dgd.get_latent_space_values("val",3000), validation_loader.dataset.cancer_type, color_mapping_cancer, epoch, dataset="Validation")
                    plot_mirna(dgd, train_loader, sample_index, epoch, device, fold=fold, type="Train", scaling_type=scaling_type)
                    plot_mirna(dgd, validation_loader, sample_index, epoch, device, fold=fold, type="Validation", scaling_type=scaling_type)
                    plot_gene(dgd, train_loader, sample_index, epoch, device, fold=fold, type="Train", scaling_type=scaling_type)
                    plot_gene(dgd, validation_loader, sample_index, epoch, device, fold=fold, type="Validation", scaling_type=scaling_type)
            
        if wandb_log: 
            wandb.log({
                "fold": fold,
                "epoch": epoch,
                "train_loss": loss_tab["train_loss"][-1],
                "train_recon_mrna": loss_tab["train_recon_mrna"][-1],
                "train_recon_mirna": loss_tab["train_recon_mirna"][-1],
                "train_gmm": loss_tab["train_gmm"][-1],
                # "train_mse": loss_tab["train_mse"][-1],
                # "train_mae": loss_tab["train_mae"][-1],
                "train_r2": loss_tab["train_r2"][-1],
                "train_spearman": loss_tab["train_spearman"][-1],
                "train_pearson": loss_tab["train_pearson"][-1],
                # "train_msle": loss_tab["train_msle"][-1],
                "test_loss": loss_tab["test_loss"][-1],
                "test_loss_decoder": loss_tab["test_recon_mrna"][-1] + loss_tab["test_recon_mirna"][-1],
                "test_recon_mrna": loss_tab["test_recon_mrna"][-1],
                "test_recon_mirna": loss_tab["test_recon_mirna"][-1],
                "test_gmm": loss_tab["test_gmm"][-1],
                # "test_mse": loss_tab["test_mse"][-1],
                # "test_mae": loss_tab["test_mae"][-1],
                "test_r2": loss_tab["test_r2"][-1],
                "test_spearman": loss_tab["test_spearman"][-1],
                "test_pearson": loss_tab["test_pearson"][-1],
                # "test_msle": loss_tab["test_msle"][-1],
                "delta_mirna": loss_tab["delta_mirna"][-1],
            })

        # Early stopping
        if epoch > 100:
            if early_stopping:
                current_loss_mrna = loss_tab["test_recon_mrna"][-1]
                current_loss_mirna = loss_tab["test_recon_mirna"][-1]

                # Track epochs since improvement for each metric
                if current_loss_mrna < best_loss_mrna:
                    best_loss_mrna = current_loss_mrna
                    best_epoch_mrna = epoch
                    checkpoint_mrna = dgd
                if current_loss_mirna < best_loss_mirna:
                    best_loss_mirna = current_loss_mirna
                    best_epoch_mirna = epoch
                    checkpoint_mirna = dgd

                # If both haven't improved for 'early_stopping' epochs, stop
                if (epoch - best_epoch_mrna > early_stopping) and (epoch - best_epoch_mirna > early_stopping):
                    print(f"Early stopped training at epoch {epoch} "
                        f"with losses mrna={best_loss_mrna:.4f}, mirna={best_loss_mirna:.4f}")
                    # Restore to the best overall checkpoint (can choose which based on which is lower or latest)
                    # Example: Restore model to the most recent best checkpoint
                    dgd = checkpoint_mrna if best_epoch_mrna > best_epoch_mirna else checkpoint_mirna
                    break

      
    
    # Training done!
    return loss_tab