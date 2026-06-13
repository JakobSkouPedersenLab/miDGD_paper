import torch
from base.dgd.latent import RepresentationLayer
from base.plotting.plot_covar import plot_latent_space, plot_latent_space_covar, plot_latent_space_covar_two

from torchmetrics.functional.regression import mean_squared_error, mean_absolute_error, r2_score, pearson_corrcoef, spearman_corrcoef, mean_squared_log_error

from tqdm import tqdm
import wandb

# Training functions
def train_midgd_covar(
                dgd, train_loader, validation_loader, device,
                learning_rates={'dec':0.0001,'rep':0.01,'gmm':0.01}, weight_decay=0., 
                learning_rates_covar={'dec':0.0001,'rep':0.01,'gmm':0.01}, weight_decay_covar=0.,
                betas=(0.9, 0.999), scheduler=None,
                nepochs=100, fold=None, pr=1, plot=10, 
                reduction_type="sum", scaling_type="mean",
                sample_index=[0,11,22,33], subset=1310, wandb_log=False, early_stopping=50, is_plot=True):
    '''
    Should be used with CovarDGD model only.
    '''
    # Normalization factor
    if reduction_type == "sum":
        tlen=len(train_loader.dataset)*dgd.decoder.n_out_features
        vlen=len(validation_loader.dataset)*dgd.decoder.n_out_features
        tlen_gmm=len(train_loader.dataset)*dgd.gmm.n_mix_comp*dgd.gmm.dim
        vlen_gmm=len(validation_loader.dataset)*dgd.gmm.n_mix_comp*dgd.gmm.dim
        tlen_gmm_covar=len(train_loader.dataset)*dgd.gmm_covar.Nclass*dgd.gmm_covar.Ncpc*dgd.gmm_covar.dim
        vlen_gmm_covar=len(validation_loader.dataset)*dgd.gmm_covar.Nclass*dgd.gmm_covar.Ncpc*dgd.gmm_covar.dim

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
        dgd.train_rep_covar = RepresentationLayer(dgd.rep_covar_dim,Ntrain).to(device)

    # Test/Validation Representation Layer Initialization
    Nvalidation=len(validation_loader.dataset)
    if dgd.val_rep is None:
        dgd.val_rep = RepresentationLayer(dgd.rep_dim,Nvalidation).to(device)
        dgd.val_rep_covar = RepresentationLayer(dgd.rep_covar_dim,Nvalidation).to(device)

    # Optimizer Initialization
    dec_optimizer = torch.optim.AdamW(dgd.decoder.parameters(), lr=learning_rates['dec'], weight_decay=weight_decay['dec'], betas=betas)
    
    gmm_optimizer = torch.optim.AdamW(dgd.gmm.parameters(), lr=learning_rates['gmm'], weight_decay=weight_decay['gmm'], betas=betas)
    train_rep_optimizer = torch.optim.AdamW(dgd.train_rep.parameters(), lr=learning_rates['rep'], weight_decay=weight_decay['rep'], betas=betas)
    val_rep_optimizer = torch.optim.AdamW(dgd.val_rep.parameters(), lr=learning_rates['rep'], weight_decay=weight_decay['rep'], betas=betas)

    gmm_covar_optimizer = torch.optim.AdamW(dgd.gmm_covar.parameters(), lr=learning_rates_covar['gmm'], weight_decay=weight_decay_covar['gmm'], betas=betas)
    train_rep_covar_optimizer = torch.optim.AdamW(dgd.train_rep_covar.parameters(), lr=learning_rates_covar['rep'], weight_decay=weight_decay_covar['rep'], betas=betas)
    val_rep_covar_optimizer = torch.optim.AdamW(dgd.val_rep_covar.parameters(), lr=learning_rates_covar['rep'], weight_decay=weight_decay_covar['rep'], betas=betas)


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
                "train_gmm_covar":[], "test_gmm_covar":[],
                "train_r2":[], "test_r2":[],
                "train_spearman":[], "test_spearman":[],
                "test_pearson":[], "train_pearson":[],
                "delta_mirna": []}
    gmm_loss=True

    # For custom color mapping
    color_mapping_tissue = dict(zip(train_loader.dataset.primary_site, train_loader.dataset.color))
    color_mapping_batch = dict(zip(train_loader.dataset.str_labels, train_loader.dataset.color))

    # Early stopping
    best_loss = 1e20
    best_epoch = -1

    print(f"Training DGD with {len(train_loader.dataset)} training samples and {len(validation_loader.dataset)} validation samples.")
    print(f"Using {dgd.rep_dim} representation dimensions and {dgd.rep_covar_dim} covariate representation dimensions.")
    print(f"Using {dgd.decoder.n_out_features_mrna} decoder output features for mRNA and {dgd.decoder.n_out_features_mirna} for miRNA.")
    print(f"Using {dgd.gmm.n_mix_comp} GMM components")
    print(f"Using {dgd.gmm_covar.Nclass} GMM covariate classes and {dgd.gmm_covar.Ncpc} components per class.")

    # Start training
    for epoch in tqdm(range(nepochs)):
        # Train step
        loss_tab["epoch"].append(epoch)
        loss_tab["train_loss"].append(0.)
        loss_tab["train_recon_mirna"].append(0.)
        loss_tab["train_recon_mrna"].append(0.)
        loss_tab["train_gmm"].append(0.)
        loss_tab["train_gmm_covar"].append(0.)
        loss_tab["train_r2"].append(0.)
        loss_tab["train_spearman"].append(0.)
        loss_tab["train_pearson"].append(0.)

        train_rep_optimizer.zero_grad()
        train_rep_covar_optimizer.zero_grad()
        dgd.train() # Training 

        for (mrna_data, mirna_data, lib_mrna, lib_mirna, index, label) in train_loader:
            dec_optimizer.zero_grad()
            if gmm_loss: 
                gmm_optimizer.zero_grad()
                gmm_covar_optimizer.zero_grad()
            
            # Concatenate train_rep and train_rep_covar
            mirna_recon_loss, mrna_recon_loss, gmm_loss, gmm_covar_loss = dgd.forward_and_loss(
                z_rep=dgd.train_rep(index),
                z_covar=dgd.train_rep_covar(index),
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
            loss_tab["train_gmm_covar"][-1] += gmm_covar_loss.item()

            loss =  mrna_recon_loss + mirna_recon_loss + gmm_loss + gmm_covar_loss
            loss.backward()
            dec_optimizer.step()
            if gmm_loss: 
                gmm_optimizer.step()
                gmm_covar_optimizer.step()
        train_rep_optimizer.step()
        train_rep_covar_optimizer.step()

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

            mirna_recon, _ = dgd.forward(dgd.train_rep(), dgd.train_rep_covar())
            mirna_recon = mirna_recon * scaling.unsqueeze(1).to(device)
            mirna_data = train_loader.dataset.mirna_data.to(device)
            # Get subset
            mirna_recon_tpm = mirna_recon[:,subset]
            mirna_data_tpm = mirna_data[:,subset]
            # Calculate metrics
            mirna_r2 = r2_score(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["train_r2"][-1] += mirna_r2.item()
            mirna_spearman = spearman_corrcoef(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["train_spearman"][-1] += mirna_spearman.item()
            mirna_pearson = pearson_corrcoef(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["train_pearson"][-1] += mirna_pearson.item()


        loss_tab["train_recon_mirna"][-1] /= tlen_mirna
        loss_tab["train_recon_mrna"][-1] /= tlen_mrna
        loss_tab["train_gmm"][-1] /= tlen_gmm
        loss_tab["train_gmm_covar"][-1] /= tlen_gmm_covar
        loss_tab["train_loss"][-1] = loss_tab["train_recon_mirna"][-1] + loss_tab["train_recon_mrna"][-1] + loss_tab["train_gmm"][-1]

        # Validation step
        loss_tab["test_loss"].append(0.)
        loss_tab["test_recon_mrna"].append(0.)
        loss_tab["test_recon_mirna"].append(0.)
        loss_tab["test_gmm"].append(0.)
        loss_tab["test_gmm_covar"].append(0.)
        loss_tab["test_r2"].append(0.)
        loss_tab["test_spearman"].append(0.)
        loss_tab["test_pearson"].append(0.)
        loss_tab["delta_mirna"].append(0.)

        # Train the validation representation layer only using mRNA data
        val_rep_optimizer.zero_grad()
        dgd.eval() # Validation mode
        for (mrna_data, mirna_data, lib_mrna, lib_mirna, index, label) in validation_loader:
            # Concatenate val_rep and val_rep_covar
            mirna_recon_loss, mrna_recon_loss, gmm_loss, gmm_covar_loss = dgd.forward_and_loss(
                z_rep=dgd.val_rep(index),
                z_covar=dgd.val_rep_covar(index),
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
            loss_tab["test_gmm_covar"][-1] += gmm_covar_loss.item()
            
            loss = mrna_recon_loss + gmm_loss + gmm_covar_loss
            loss.backward()
        val_rep_optimizer.step()
        val_rep_covar_optimizer.step()

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
            
            mirna_recon, _ = dgd.forward(dgd.val_rep(), dgd.val_rep_covar())
            mirna_recon = mirna_recon * scaling.unsqueeze(1).to(device)
            mirna_data = validation_loader.dataset.mirna_data.to(device)
            # Get subset
            mirna_recon_tpm = mirna_recon[:,subset]
            mirna_data_tpm = mirna_data[:,subset]
            # Calculate metrics
            mirna_r2 = r2_score(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["test_r2"][-1] += mirna_r2.item()
            mirna_spearman = spearman_corrcoef(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["test_spearman"][-1] += mirna_spearman.item()
            mirna_pearson = pearson_corrcoef(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["test_pearson"][-1] += mirna_pearson.item()

        loss_tab["test_recon_mirna"][-1] /= vlen_mirna
        loss_tab["test_recon_mrna"][-1] /= vlen_mrna
        loss_tab["test_gmm"][-1] /= vlen_gmm
        loss_tab["test_gmm_covar"][-1] /= vlen_gmm_covar
        loss_tab["test_loss"][-1] = loss_tab["test_recon_mrna"][-1] + loss_tab["test_recon_mirna"][-1]  + loss_tab["test_gmm"][-1]
        loss_tab["delta_mirna"][-1] = loss_tab["test_recon_mirna"][-1] - loss_tab["train_recon_mirna"][-1]
        
        if pr>=0 and (epoch)%pr==0:
            print(epoch,
                  f"train_loss: {loss_tab['train_loss'][-1]}",
                  f"train_loss_decoder: {loss_tab['train_recon_mrna'][-1] + loss_tab['train_recon_mirna'][-1]}",
                  f"train_recon_mirna: {loss_tab['train_recon_mirna'][-1]}", 
                  f"train_recon_mrna: {loss_tab['train_recon_mrna'][-1]}", 
                  f"train_gmm: {loss_tab['train_gmm'][-1]}",
                  f"train_gmm_covar: {loss_tab['train_gmm_covar'][-1]}",
                  f"train_r2: {loss_tab['train_r2'][-1]}",
                  f"train_spearman: {loss_tab['train_spearman'][-1]}",
                  f"train_pearson: {loss_tab['train_pearson'][-1]}")
            print(epoch,
                  f"test_loss: {loss_tab['test_loss'][-1]}",
                  f"test_loss_decoder: {loss_tab['test_recon_mrna'][-1] + loss_tab['test_recon_mirna'][-1]}",
                  f"test_recon_mirna: {loss_tab['test_recon_mirna'][-1]}",
                  f"test_recon_mrna: {loss_tab['test_recon_mrna'][-1]}", 
                  f"test_gmm: {loss_tab['test_gmm'][-1]}",
                  f"test_gmm_covar: {loss_tab['test_gmm_covar'][-1]}",
                  f"test_r2: {loss_tab['test_r2'][-1]}",
                  f"test_spearman: {loss_tab['test_spearman'][-1]}",
                  f"test_pearson: {loss_tab['test_pearson'][-1]}",
                  f"delta_mirna: {loss_tab['delta_mirna'][-1]}"
                )
            
        if is_plot:
            if plot>=0 and (epoch)%plot==0:
                rep_train, means_train, samples_train = dgd.get_latent_space_values("train", 5000)
                rep_val, means_val, samples_val = dgd.get_latent_space_values("val", 5000)

                plot_latent_space_covar_two(
                    rep_train, means_train, samples_train, train_loader.dataset.primary_site, train_loader.dataset.str_labels,
                    rep_val, means_val, samples_val, validation_loader.dataset.primary_site, validation_loader.dataset.str_labels,
                    color_mapping_tissue, outfile=None
                )

                plot_latent_space_covar_two(
                    rep_train, means_train, samples_train, train_loader.dataset.str_labels, train_loader.dataset.str_labels,
                    rep_val, means_val, samples_val, validation_loader.dataset.str_labels, validation_loader.dataset.str_labels,
                    color_mapping_batch, outfile=None
                )

                rep_train, means_train, samples_train = dgd.get_covar_latent_space_values("train", 3000)
                rep_val, means_val, samples_val = dgd.get_covar_latent_space_values("val", 3000)

                plot_latent_space_covar_two(
                    rep_train, means_train, samples_train, train_loader.dataset.str_labels, train_loader.dataset.str_labels,
                    rep_val, means_val, samples_val, validation_loader.dataset.str_labels, validation_loader.dataset.str_labels,
                    color_mapping_batch, outfile=None
                )
        if wandb_log: 
            wandb.log({
                "fold": fold,
                "epoch": epoch,
                "train_loss": loss_tab["train_loss"][-1],
                "train_recon_mrna": loss_tab["train_recon_mrna"][-1],
                "train_recon_mirna": loss_tab["train_recon_mirna"][-1],
                "train_gmm": loss_tab["train_gmm"][-1],
                "train_r2": loss_tab["train_r2"][-1],
                "train_spearman": loss_tab["train_spearman"][-1],
                "train_pearson": loss_tab["train_pearson"][-1],
                "test_loss": loss_tab["test_loss"][-1],
                "test_recon_mrna": loss_tab["test_recon_mrna"][-1],
                "test_recon_mirna": loss_tab["test_recon_mirna"][-1],
                "test_gmm": loss_tab["test_gmm"][-1],
                "test_gmm_covar": loss_tab["test_gmm_covar"][-1],
                "test_r2": loss_tab["test_r2"][-1],
                "test_spearman": loss_tab["test_spearman"][-1],
                "test_pearson": loss_tab["test_pearson"][-1],
                "delta_mirna": loss_tab["delta_mirna"][-1]
            })

        # Early stopping
        if epoch > 100:
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