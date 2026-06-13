import torch
from base.dgd.latent import RepresentationLayer
from base.plotting.plot_cv2 import plot_latent_space, plot_gene, plot_mirna
from base.plotting.plot_combined import plot_gene_recons_combined, plot_sample_recons_combined

from torchmetrics.functional.regression import mean_squared_error, mean_absolute_error, r2_score, explained_variance, pearson_corrcoef, spearman_corrcoef, mean_squared_log_error

from tqdm import tqdm
import wandb


def train_midgd(dgd, train_loader, validation_loader, device, train_dataset, val_dataset,
                learning_rates={'dec':0.001,'rep':0.01,'gmm':0.01}, 
                weight_decay=0., betas=(0.9, 0.999), nepochs=100, pr=1, plot=10, reduction_type="sum", 
                sample_index=[0,11,22,33], subset=1310, wandb_log=False):
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

    
    Ntrain=len(train_loader.dataset)
    Nvalidation=len(validation_loader.dataset)
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
                "train_recon_mirna":[],"train_recon_mrna":[], "test_recon":[],
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
    if dgd.decoder.out_modules[0]._scaling_type == "mean":
        scale_mirna_train = torch.mean(train_dataset.mirna_data, dim=-1)
        scale_mirna_test = torch.mean(val_dataset.mirna_data, dim=-1)
    elif dgd.decoder.out_modules[0]._scaling_type == "max":
        scale_mirna_train = torch.max(train_dataset.mirna_data, dim=-1).values
        scale_mirna_test = torch.max(val_dataset.mirna_data, dim=-1).values

    # Start training
    for epoch in tqdm(range(nepochs)):
        # Train step
        loss_tab["epoch"].append(epoch)
        loss_tab["train_recon_mirna"].append(0.)
        loss_tab["train_recon_mrna"].append(0.)
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

        # Calculate loss
        with torch.inference_mode():
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

        loss_tab["train_recon_mirna"][-1] /= tlen_mirna
        loss_tab["train_recon_mrna"][-1] /= tlen_mrna
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
                target=[mrna_data.to(device)],  # Pass both mRNA and miRNA data
                scale=[lib_mrna.unsqueeze(1).to(device)],  # Pass both scales
                gmm_loss=gmm_loss,
                reduction=reduction_type,
                type="midgd"
            )
            loss_tab["test_recon"][-1] += recon_loss.item()
            loss_tab["test_gmm"][-1] += gmm_loss.item()
            
            loss = recon_loss + gmm_loss
            loss.backward()
        val_rep_optimizer.step()

        # Calculate metrics
        with torch.inference_mode():
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

        loss_tab["test_recon"][-1] /= vlen_mrna
        loss_tab["test_gmm"][-1] /= vlen_gmm

        if pr>=0 and (epoch)%pr==0:
            print(epoch,
                  f"train_recon_mirna: {loss_tab['train_recon_mirna'][-1]}", 
                  f"train_recon_mrna: {loss_tab['train_recon_mrna'][-1]}", 
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
        if plot>=0 and (epoch)%plot==0:
            plot_latent_space(*dgd.get_latent_space_values("train",1000), train_loader.dataset.label, color_mapping, epoch, "Train")
            plot_latent_space(*dgd.get_latent_space_values("val",1000), validation_loader.dataset.label, color_mapping, epoch, "Validation")
            plot_mirna(dgd, train_loader, sample_index, epoch, fold=None, type="Train")
            plot_mirna(dgd, validation_loader, sample_index, epoch, fold=None, type="Test")
            plot_gene(dgd, train_loader, sample_index, epoch, fold=None, type="Train")
            plot_gene(dgd, validation_loader, sample_index, epoch, fold=None, type="Test")
            
        if wandb_log: 
            wandb.log({"train_recon": loss_tab["train_recon"][-1],
                       "train_gmm": loss_tab["train_gmm"][-1],
                       "train_rmse": loss_tab["train_rmse"][-1],
                       "train_mse": loss_tab["train_mse"][-1],
                       "train_mae": loss_tab["train_mae"][-1],
                       "train_r2": loss_tab["train_r2"][-1],
                       "train_corr": loss_tab["train_corr"][-1],
                       "test_recon": loss_tab["test_recon"][-1],
                       "test_gmm": loss_tab["test_gmm"][-1],
                       "test_rmse": loss_tab["test_rmse"][-1],
                       "test_mse": loss_tab["test_mse"][-1],
                       "test_mae": loss_tab["test_mae"][-1],
                       "test_r2": loss_tab["test_r2"][-1],
                       "test_corr": loss_tab["test_corr"][-1]})
    
    return loss_tab


def train_dgd_combined(dgd, train_loader, validation_loader, device, learning_rates={'dec':0.001,'rep':0.01,'gmm':0.01}, 
              weight_decay=0., nepochs=100, pr=1, plot=10, reduction_type="sum", sample_index=[0,11,22,33], normalization="TPM"):
    # Normalization factor
    if reduction_type == "sum":
        tlen=len(train_loader.dataset)*dgd.decoder.n_out_features
        vlen=len(validation_loader.dataset)*dgd.decoder.n_out_features
        tlen_gmm=len(train_loader.dataset)*dgd.gmm.n_mix_comp*dgd.gmm.dim
        vlen_gmm=len(validation_loader.dataset)*dgd.gmm.n_mix_comp*dgd.gmm.dim
        tlen_mirna = len(train_loader.dataset)
        vlen_mirna = len(validation_loader.dataset)
    else:
        tlen=len(train_loader)
        vlen=len(validation_loader)
        tlen_gmm=len(train_loader)
        vlen_gmm=len(validation_loader)
        tlen_mirna = len(train_loader)
        vlen_mirna = len(validation_loader)
    
    Ntrain=len(train_loader.dataset)
    Nvalidation=len(validation_loader.dataset)
    if dgd.train_rep is None:
        dgd.train_rep = RepresentationLayer(dgd.rep_dim,Ntrain).to(device)

    Nvalidation=len(validation_loader.dataset)
    if dgd.val_rep is None:
        dgd.val_rep = RepresentationLayer(dgd.rep_dim,Nvalidation).to(device)

    dec_optimizer = torch.optim.AdamW(dgd.decoder.parameters(), lr=learning_rates['dec'])
    gmm_optimizer = torch.optim.AdamW(dgd.gmm.parameters(), lr=learning_rates['gmm'])
    train_rep_optimizer = torch.optim.AdamW(dgd.train_rep.parameters(), lr=learning_rates['rep'])
    val_rep_optimizer = torch.optim.AdamW(dgd.val_rep.parameters(), lr=learning_rates['rep'])

    loss_tab = {"epoch":[],
                "train_recon":[],"test_recon":[],
                "train_gmm":[],"test_gmm":[], 
                "train_rmse":[], "test_rmse":[],
                "train_mse":[], "test_mse":[],
                "train_mae":[], "test_mae":[],
                "train_r2":[], "test_r2":[]}
    best_loss=1.e20
    gmm_loss=True

    # For custom color mapping
    color_mapping = dict(zip(train_loader.dataset.label, train_loader.dataset.color))

    # Start training
    for epoch in tqdm(range(nepochs)):
        # Train step
        loss_tab["epoch"].append(epoch)
        loss_tab["train_recon"].append(0.)
        loss_tab["train_gmm"].append(0.)
        loss_tab["train_rmse"].append(0.)
        loss_tab["train_mse"].append(0.)
        loss_tab["train_mae"].append(0.)
        loss_tab["train_r2"].append(0.)

        train_rep_optimizer.zero_grad()
        dgd.train()
        for (mrna_data, mirna_data, lib_mrna, lib_mirna, index) in train_loader:
            dec_optimizer.zero_grad()
            if gmm_loss: gmm_optimizer.zero_grad()
            recon_loss, gmm_loss = dgd.forward_and_loss(
                z=dgd.train_rep(index),
                target=[mrna_data.to(device), mirna_data.to(device)],  # Pass both mRNA and miRNA data
                scale=[lib_mrna.unsqueeze(1).to(device), lib_mirna.unsqueeze(1).to(device)],  # Pass both scales
                gmm_loss=gmm_loss,
                reduction=reduction_type
            )

            # Calculate loss
            if normalization==None:
                with torch.no_grad():
                    mirna_recon = dgd.forward(dgd.train_rep(index))[1]  # Get reconstructed miRNA
                    mirna_recon = mirna_recon * lib_mirna.unsqueeze(1).to(device)  # Scale reconstructed miRNA
                    # Calculate RMSE loss
                    mirna_rmse = torch.sqrt(torch.mean((mirna_recon - mirna_data.to(device))**2))
                    loss_tab["train_rmse"][-1] += mirna_rmse.item()
                    # Calculate MSE loss
                    mirna_mse = torch.mean((mirna_recon - mirna_data.to(device))**2)
                    loss_tab["train_mse"][-1] += mirna_mse.item()
                    # Calculate MAE loss
                    mirna_mae = torch.mean(torch.abs(mirna_recon - mirna_data.to(device)))
                    loss_tab["train_mae"][-1] += mirna_mae.item()
                    # Calculate R^2 loss
                    mirna_r2 = 1 - torch.sum((mirna_recon - mirna_data.to(device))**2) / torch.sum((mirna_data.to(device) - torch.mean(mirna_data.to(device)))**2)
                    loss_tab["train_r2"][-1] += mirna_r2.item()
            elif normalization=="TPM":
                with torch.no_grad():
                    mirna_recon = dgd.forward(dgd.train_rep(index))[1]
                    mirna_recon = mirna_recon * lib_mirna.unsqueeze(1).to(device)
                    # Normalize mirna_recon and mirna_data using TPM or FPKM
                    mirna_recon_tpm = mirna_recon / mirna_recon.sum() * 1e6  # TPM normalization
                    mirna_data_tpm = mirna_data.to(device) / mirna_data.to(device).sum() * 1e6  # TPM normalization
                    # Calculate RMSE loss using normalized data
                    mirna_rmse = torch.sqrt(torch.mean((mirna_recon_tpm - mirna_data_tpm)**2))
                    loss_tab["train_rmse"][-1] += mirna_rmse.item()
                    # Calculate MSE loss using normalized data
                    mirna_mse = torch.mean((mirna_recon_tpm - mirna_data_tpm)**2)
                    loss_tab["train_mse"][-1] += mirna_mse.item()
                    # Calculate MAE loss using normalized data
                    mirna_mae = torch.mean(torch.abs(mirna_recon_tpm - mirna_data_tpm))
                    loss_tab["train_mae"][-1] += mirna_mae.item()
                    # Calculate R^2 loss using normalized data
                    mirna_r2 = 1 - torch.sum((mirna_recon_tpm - mirna_data_tpm)**2) / torch.sum((mirna_data_tpm - torch.mean(mirna_data_tpm))**2)
                    loss_tab["train_r2"][-1] += mirna_r2.item()

            loss_tab["train_recon"][-1] += recon_loss.item()
            loss_tab["train_gmm"][-1] += gmm_loss.item()
            loss = recon_loss + gmm_loss
            loss.backward()
            dec_optimizer.step()
            if gmm_loss: gmm_optimizer.step()
        train_rep_optimizer.step()

        loss_tab["train_recon"][-1] /= tlen
        loss_tab["train_gmm"][-1] /= tlen_gmm
        loss_tab["train_rmse"][-1] /= tlen_mirna
        loss_tab["train_mse"][-1] /= tlen_mirna
        loss_tab["train_mae"][-1] /= tlen_mirna
        loss_tab["train_r2"][-1] /= tlen_mirna

        # Validation step
        loss_tab["test_recon"].append(0.)
        loss_tab["test_gmm"].append(0.)
        loss_tab["test_rmse"].append(0.)
        loss_tab["test_mse"].append(0.)
        loss_tab["test_mae"].append(0.)
        loss_tab["test_r2"].append(0.)

        val_rep_optimizer.zero_grad()
        dgd.eval()
        for (mrna_data, mirna_data, lib_mrna, lib_mirna, index) in validation_loader:
            recon_loss, gmm_loss = dgd.forward_and_loss(
                z=dgd.val_rep(index),
                target=[mrna_data.to(device), mirna_data.to(device)],  # Pass both mRNA and miRNA data
                scale=[lib_mrna.unsqueeze(1).to(device), lib_mirna.unsqueeze(1).to(device)],  # Pass both scales
                gmm_loss=gmm_loss,
                reduction=reduction_type
            )

            # Calculate loss
            if normalization==None:
                with torch.no_grad():
                    mirna_recon = dgd.forward(dgd.val_rep(index))[1]  # Get reconstructed miRNA
                    mirna_recon = mirna_recon * lib_mirna.unsqueeze(1).to(device)  # Scale reconstructed miRNA
                    # Calculate RMSE loss
                    mirna_rmse = torch.sqrt(torch.mean((mirna_recon - mirna_data.to(device))**2))
                    loss_tab["test_rmse"][-1] += mirna_rmse.item()
                    # Calculate MSE loss
                    mirna_mse = torch.mean((mirna_recon - mirna_data.to(device))**2)
                    loss_tab["test_mse"][-1] += mirna_mse.item()
                    # Calculate MAE loss
                    mirna_mae = torch.mean(torch.abs(mirna_recon - mirna_data.to(device)))
                    loss_tab["test_mae"][-1] += mirna_mae.item()
                    # Calculate R^2 loss
                    mirna_r2 = 1 - torch.sum((mirna_recon - mirna_data.to(device))**2) / torch.sum((mirna_data.to(device) - torch.mean(mirna_data.to(device)))**2)
                    loss_tab["test_r2"][-1] += mirna_r2.item()
            elif normalization=="TPM":
                with torch.no_grad():
                    mirna_recon = dgd.forward(dgd.val_rep(index))[1]
                    mirna_recon = mirna_recon * lib_mirna.unsqueeze(1).to(device)
                    # Normalize mirna_recon and mirna_data using TPM or FPKM
                    mirna_recon_tpm = mirna_recon / mirna_recon.sum() * 1e6
                    mirna_data_tpm = mirna_data.to(device) / mirna_data.to(device).sum() * 1e6
                    # Calculate RMSE loss using normalized data
                    mirna_rmse = torch.sqrt(torch.mean((mirna_recon_tpm - mirna_data_tpm)**2))
                    loss_tab["test_rmse"][-1] += mirna_rmse.item()
                    # Calculate MSE loss using normalized data
                    mirna_mse = torch.mean((mirna_recon_tpm - mirna_data_tpm)**2)
                    loss_tab["test_mse"][-1] += mirna_mse.item()
                    # Calculate MAE loss using normalized data
                    mirna_mae = torch.mean(torch.abs(mirna_recon_tpm - mirna_data_tpm))
                    loss_tab["test_mae"][-1] += mirna_mae.item()
                    # Calculate R^2 loss using normalized data
                    mirna_r2 = 1 - torch.sum((mirna_recon_tpm - mirna_data_tpm)**2) / torch.sum((mirna_data_tpm - torch.mean(mirna_data_tpm))**2)
                    loss_tab["test_r2"][-1] += mirna_r2.item()

            loss_tab["test_recon"][-1] += recon_loss.item()
            loss_tab["test_gmm"][-1] += gmm_loss.item()
            loss = recon_loss + gmm_loss
            loss.backward()
        val_rep_optimizer.step()

        loss_tab["test_recon"][-1] /= vlen
        loss_tab["test_gmm"][-1] /= vlen_gmm
        loss_tab["test_rmse"][-1] /= vlen_mirna
        loss_tab["test_mse"][-1] /= vlen_mirna
        loss_tab["test_mae"][-1] /= vlen_mirna
        loss_tab["test_r2"][-1] /= vlen_mirna

        if pr>=0 and (epoch)%pr==0:
            print(epoch, loss_tab["test_recon"][-1], loss_tab["test_gmm"][-1])
        if plot>=0 and (epoch)%plot==0:
            plot_latent_space(*dgd.get_latent_space_values("train", 1000), train_loader.dataset.label, color_mapping, epoch)
        if plot>=0 and (epoch)%plot==0:
            plot_sample_recons_combined(dgd, train_loader, sample_index, epoch)
        if plot>=0 and (epoch)%plot==0:
            plot_gene_recons_combined(dgd, train_loader, sample_index, epoch)

    return loss_tab


def train_dgd_mrna(dgd, train_loader, validation_loader, device, learning_rates={'dec':0.001,'rep':0.01,'gmm':0.01}, 
              weight_decay=0., nepochs=100, pr=1, plot=10, reduction_type="sum", sample_index=[0,11,22,33]):
    # Normalization factor
    if reduction_type == "sum":
        tlen=len(train_loader.dataset)*dgd.decoder.n_out_features
        vlen=len(validation_loader.dataset)*dgd.decoder.n_out_features
        tlen_gmm=len(train_loader.dataset)*dgd.gmm.n_mix_comp*dgd.gmm.dim
        vlen_gmm=len(validation_loader.dataset)*dgd.gmm.n_mix_comp*dgd.gmm.dim
        tlen_mirna = len(train_loader.dataset)
        vlen_mirna = len(validation_loader.dataset)
    else:
        tlen=len(train_loader)
        vlen=len(validation_loader)
        tlen_gmm=len(train_loader)
        vlen_gmm=len(validation_loader)
        tlen_mirna = len(train_loader)
        vlen_mirna = len(validation_loader)
    
    Ntrain=len(train_loader.dataset)
    Nvalidation=len(validation_loader.dataset)
    if dgd.train_rep is None:
        dgd.train_rep = RepresentationLayer(dgd.rep_dim,Ntrain).to(device)

    Nvalidation=len(validation_loader.dataset)
    if dgd.val_rep is None:
        dgd.val_rep = RepresentationLayer(dgd.rep_dim,Nvalidation).to(device)

    dec_optimizer = torch.optim.AdamW(dgd.decoder.parameters(), lr=learning_rates['dec'])
    gmm_optimizer = torch.optim.AdamW(dgd.gmm.parameters(), lr=learning_rates['gmm'])
    train_rep_optimizer = torch.optim.AdamW(dgd.train_rep.parameters(), lr=learning_rates['rep'])
    val_rep_optimizer = torch.optim.AdamW(dgd.val_rep.parameters(), lr=learning_rates['rep'])

    loss_tab = {"epoch":[],
                "train_recon":[],"test_recon":[],
                "train_gmm":[],"test_gmm":[], 
                "train_rmse":[], "test_rmse":[],
                "train_mse":[], "test_mse":[],
                "train_mae":[], "test_mae":[],
                "train_r2":[], "test_r2":[]}
    best_loss=1.e20
    gmm_loss=True

    # For custom color mapping
    color_mapping = dict(zip(train_loader.dataset.label, train_loader.dataset.color))

    # Start training
    for epoch in tqdm(range(nepochs)):
        # Train step
        loss_tab["epoch"].append(epoch)
        loss_tab["train_recon"].append(0.)
        loss_tab["train_gmm"].append(0.)
        loss_tab["train_rmse"].append(0.)
        loss_tab["train_mse"].append(0.)
        loss_tab["train_mae"].append(0.)
        loss_tab["train_r2"].append(0.)

        train_rep_optimizer.zero_grad()
        dgd.train()
        for (mrna_data, lib_mrna, index) in train_loader:
            dec_optimizer.zero_grad()
            if gmm_loss: gmm_optimizer.zero_grad()
            recon_loss, gmm_loss = dgd.forward_and_loss(
                z=dgd.train_rep(index),
                target=[mrna_data.to(device)],  # Pass both mRNA and miRNA data
                scale=[lib_mrna.unsqueeze(1).to(device)],  # Pass both scales
                gmm_loss=gmm_loss,
                reduction=reduction_type,
                type="mRNA"
            )

            # Calculate loss
            with torch.no_grad():
                mrna_recon = dgd.forward(dgd.train_rep(index))  # Get reconstructed miRNA
                mrna_recon = mrna_recon * lib_mrna.unsqueeze(1).to(device)  # Scale reconstructed miRNA

                # Calculate RMSE loss
                mrna_rmse = torch.sqrt(torch.mean((mrna_recon - mrna_data.to(device))**2))
                loss_tab["train_rmse"][-1] += mrna_rmse.item()
                # Calculate MSE loss
                mrna_mse = torch.mean((mrna_recon - mrna_data.to(device))**2)
                loss_tab["train_mse"][-1] += mrna_mse.item()
                # Calculate MAE loss
                mrna_mae = torch.mean(torch.abs(mrna_recon - mrna_data.to(device)))
                loss_tab["train_mae"][-1] += mrna_mae.item()
                # Calculate R^2 loss
                mrna_r2 = 1 - torch.sum((mrna_recon - mrna_data.to(device))**2) / torch.sum((mrna_data.to(device) - torch.mean(mrna_data.to(device)))**2)
                loss_tab["train_r2"][-1] += mrna_r2.item()

            loss_tab["train_recon"][-1] += recon_loss.item()
            loss_tab["train_gmm"][-1] += gmm_loss.item()
            loss = recon_loss + gmm_loss
            loss.backward()
            dec_optimizer.step()
            if gmm_loss: gmm_optimizer.step()
        train_rep_optimizer.step()

        loss_tab["train_recon"][-1] /= tlen
        loss_tab["train_gmm"][-1] /= tlen_gmm
        loss_tab["train_rmse"][-1] /= tlen_mirna
        loss_tab["train_mse"][-1] /= tlen_mirna
        loss_tab["train_mae"][-1] /= tlen_mirna
        loss_tab["train_r2"][-1] /= tlen_mirna

        # Validation step
        loss_tab["test_recon"].append(0.)
        loss_tab["test_gmm"].append(0.)
        loss_tab["test_rmse"].append(0.)
        loss_tab["test_mse"].append(0.)
        loss_tab["test_mae"].append(0.)
        loss_tab["test_r2"].append(0.)

        val_rep_optimizer.zero_grad()
        dgd.eval()
        for (mrna_data, lib_mrna, index) in validation_loader:
            recon_loss, gmm_loss = dgd.forward_and_loss(
                z=dgd.val_rep(index),
                target=[mrna_data.to(device)],  # Pass both mRNA and miRNA data
                scale=[lib_mrna.unsqueeze(1).to(device)],  # Pass both scales
                gmm_loss=gmm_loss,
                reduction=reduction_type,
                type="mRNA"
            )

            # Calculate loss
            with torch.no_grad():
                mrna_recon = dgd.forward(dgd.val_rep(index))  # Get reconstructed miRNA
                mrna_recon = mrna_recon * lib_mrna.unsqueeze(1).to(device)  # Scale reconstructed miRNA
                # Calculate RMSE loss
                mrna_rmse = torch.sqrt(torch.mean((mrna_recon - mrna_data.to(device))**2))
                loss_tab["test_rmse"][-1] += mrna_rmse.item()
                # Calculate MSE loss
                mrna_mse = torch.mean((mrna_recon - mrna_data.to(device))**2)
                loss_tab["test_mse"][-1] += mrna_mse.item()
                # Calculate MAE loss
                mrna_mae = torch.mean(torch.abs(mrna_recon - mrna_data.to(device)))
                loss_tab["test_mae"][-1] += mrna_mae.item()
                # Calculate R^2 loss
                mrna_r2 = 1 - torch.sum((mrna_recon - mrna_data.to(device))**2) / torch.sum((mrna_data.to(device) - torch.mean(mrna_data.to(device)))**2)
                loss_tab["test_r2"][-1] += mrna_r2.item()

            loss_tab["test_recon"][-1] += recon_loss.item()
            loss_tab["test_gmm"][-1] += gmm_loss.item()
            loss = recon_loss + gmm_loss
            loss.backward()
        val_rep_optimizer.step()

        loss_tab["test_recon"][-1] /= vlen
        loss_tab["test_gmm"][-1] /= vlen_gmm
        loss_tab["test_rmse"][-1] /= vlen_mirna
        loss_tab["test_mse"][-1] /= vlen_mirna
        loss_tab["test_mae"][-1] /= vlen_mirna
        loss_tab["test_r2"][-1] /= vlen_mirna

        if pr>=0 and (epoch)%pr==0:
            print(epoch, loss_tab["test_recon"][-1], loss_tab["test_gmm"][-1])
        if plot>=0 and (epoch)%plot==0:
            plot_latent_space(*dgd.get_latent_space_values("train", 3000), train_loader.dataset.label, color_mapping, epoch)
        if plot>=0 and (epoch)%plot==0:
            plot_sample_recons(dgd, train_loader, sample_index, epoch)
        if plot>=0 and (epoch)%plot==0:
            plot_gene_recons(dgd, train_loader, sample_index, epoch)

    return loss_tab


def train_dgd_mirna(dgd, train_loader, validation_loader, device, learning_rates={'dec':0.001,'rep':0.01,'gmm':0.01}, 
              weight_decay=0., nepochs=100, pr=1, plot=10, reduction_type="sum", sample_index=[0,11,22,33], normalization="TPM"):
    # Normalization factor
    if reduction_type == "sum":
        tlen=len(train_loader.dataset)*dgd.decoder.n_out_features
        vlen=len(validation_loader.dataset)*dgd.decoder.n_out_features
        tlen_gmm=len(train_loader.dataset)*dgd.gmm.n_mix_comp*dgd.gmm.dim
        vlen_gmm=len(validation_loader.dataset)*dgd.gmm.n_mix_comp*dgd.gmm.dim
        tlen_mirna = len(train_loader.dataset)
        vlen_mirna = len(validation_loader.dataset)
    else:
        tlen=len(train_loader)
        vlen=len(validation_loader)
        tlen_gmm=len(train_loader)
        vlen_gmm=len(validation_loader)
        tlen_mirna = len(train_loader)
        vlen_mirna = len(validation_loader)
    
    Ntrain=len(train_loader.dataset)
    Nvalidation=len(validation_loader.dataset)
    if dgd.train_rep is None:
        dgd.train_rep = RepresentationLayer(dgd.rep_dim,Ntrain).to(device)

    Nvalidation=len(validation_loader.dataset)
    if dgd.val_rep is None:
        dgd.val_rep = RepresentationLayer(dgd.rep_dim,Nvalidation).to(device)

    dec_optimizer = torch.optim.AdamW(dgd.decoder.parameters(), lr=learning_rates['dec'], weight_decay=weight_decay)
    gmm_optimizer = torch.optim.AdamW(dgd.gmm.parameters(), lr=learning_rates['gmm'], weight_decay=weight_decay)
    train_rep_optimizer = torch.optim.AdamW(dgd.train_rep.parameters(), lr=learning_rates['rep'], weight_decay=weight_decay)
    val_rep_optimizer = torch.optim.AdamW(dgd.val_rep.parameters(), lr=learning_rates['rep'], weight_decay=weight_decay)

    loss_tab = {"epoch":[],
                "train_recon":[],"test_recon":[],
                "train_gmm":[],"test_gmm":[], 
                "train_rmse":[], "test_rmse":[],
                "train_mse":[], "test_mse":[],
                "train_mae":[], "test_mae":[],
                "train_r2":[], "test_r2":[]}
    best_loss=1.e20
    gmm_loss=True

    # For custom color mapping
    color_mapping = dict(zip(train_loader.dataset.label, train_loader.dataset.color))

    # Start training
    for epoch in tqdm(range(nepochs)):
        # Train step
        loss_tab["epoch"].append(epoch)
        loss_tab["train_recon"].append(0.)
        loss_tab["train_gmm"].append(0.)
        loss_tab["train_rmse"].append(0.)
        loss_tab["train_mse"].append(0.)
        loss_tab["train_mae"].append(0.)
        loss_tab["train_r2"].append(0.)

        train_rep_optimizer.zero_grad()
        dgd.train()
        for (mirna_data, lib_mirna, index) in train_loader:
            dec_optimizer.zero_grad()
            if gmm_loss: gmm_optimizer.zero_grad()
            recon_loss, gmm_loss = dgd.forward_and_loss(
                z=dgd.train_rep(index),
                target=[mirna_data.to(device)],  # Pass both mRNA and miRNA data
                scale=[lib_mirna.unsqueeze(1).to(device)],  # Pass both scales
                gmm_loss=gmm_loss,
                reduction=reduction_type,
                type="miRNA"
            )  
            # Calculate loss
            if normalization==None:
                with torch.no_grad():
                    mirna_recon = dgd.forward(dgd.train_rep(index))  # Get reconstructed miRNA
                    mirna_recon = mirna_recon[0] * lib_mirna.unsqueeze(1).to(device)  # Scale reconstructed miRNA
                    # Calculate RMSE loss
                    mirna_rmse = torch.sqrt(torch.mean((mirna_recon - mirna_data.to(device))**2))
                    loss_tab["train_rmse"][-1] += mirna_rmse.item()
                    # Calculate MSE loss
                    mirna_mse = torch.mean((mirna_recon - mirna_data.to(device))**2)
                    loss_tab["train_mse"][-1] += mirna_mse.item()
                    # Calculate MAE loss
                    mirna_mae = torch.mean(torch.abs(mirna_recon - mirna_data.to(device)))
                    loss_tab["train_mae"][-1] += mirna_mae.item()
                    # Calculate R^2 loss
                    mirna_r2 = 1 - torch.sum((mirna_recon - mirna_data.to(device))**2) / torch.sum((mirna_data.to(device) - torch.mean(mirna_data.to(device)))**2)
                    loss_tab["train_r2"][-1] += mirna_r2.item()
            elif normalization=="TPM":
                with torch.no_grad():
                    mirna_recon = dgd.forward(dgd.train_rep(index))  # Get reconstructed miRNA
                    mirna_recon = mirna_recon[0] * lib_mirna.unsqueeze(1).to(device)  # Scale reconstructed miRNA
                    # Normalize mirna_recon and mirna_data using TPM or FPKM
                    mirna_recon_tpm = mirna_recon / mirna_recon.sum() * 1e6  # TPM normalization
                    mirna_data_tpm = mirna_data.to(device) / mirna_data.to(device).sum() * 1e6  # TPM normalization
                    # Calculate RMSE loss using normalized data
                    mirna_rmse = torch.sqrt(torch.mean((mirna_recon_tpm - mirna_data_tpm)**2))
                    loss_tab["train_rmse"][-1] += mirna_rmse.item()
                    # Calculate MSE loss using normalized data
                    mirna_mse = torch.mean((mirna_recon_tpm - mirna_data_tpm)**2)
                    loss_tab["train_mse"][-1] += mirna_mse.item()
                    # Calculate MAE loss using normalized data
                    mirna_mae = torch.mean(torch.abs(mirna_recon_tpm - mirna_data_tpm))
                    loss_tab["train_mae"][-1] += mirna_mae.item()
                    # Calculate R^2 loss using normalized data
                    mirna_r2 = 1 - torch.sum((mirna_recon_tpm - mirna_data_tpm)**2) / torch.sum((mirna_data_tpm - torch.mean(mirna_data_tpm))**2)
                    loss_tab["train_r2"][-1] += mirna_r2.item()

            loss_tab["train_recon"][-1] += recon_loss.item()
            loss_tab["train_gmm"][-1] += gmm_loss.item()
            loss = recon_loss + gmm_loss
            loss.backward()
            dec_optimizer.step()
            if gmm_loss: gmm_optimizer.step()
        train_rep_optimizer.step()

        loss_tab["train_recon"][-1] /= tlen
        loss_tab["train_gmm"][-1] /= tlen_gmm
        loss_tab["train_rmse"][-1] /= tlen_mirna
        loss_tab["train_mse"][-1] /= tlen_mirna
        loss_tab["train_mae"][-1] /= tlen_mirna
        loss_tab["train_r2"][-1] /= tlen_mirna

        # Validation step
        loss_tab["test_recon"].append(0.)
        loss_tab["test_gmm"].append(0.)
        loss_tab["test_rmse"].append(0.)
        loss_tab["test_mse"].append(0.)
        loss_tab["test_mae"].append(0.)
        loss_tab["test_r2"].append(0.)

        val_rep_optimizer.zero_grad()
        dgd.eval()
        for (mirna_data, lib_mirna, index) in validation_loader:
            recon_loss, gmm_loss = dgd.forward_and_loss(
                z=dgd.val_rep(index),
                target=[mirna_data.to(device)],  # Pass both mRNA and miRNA data
                scale=[lib_mirna.unsqueeze(1).to(device)],  # Pass both scales
                gmm_loss=gmm_loss,
                reduction=reduction_type,
                type="miRNA"
            )
            
            # Calculate loss
            if normalization==None:
                with torch.no_grad():
                    mirna_recon = dgd.forward(dgd.val_rep(index))  # Get reconstructed miRNA
                    mirna_recon = mirna_recon[0] * lib_mirna.unsqueeze(1).to(device)  # Scale reconstructed miRNA
                    # Calculate RMSE loss
                    mirna_rmse = torch.sqrt(torch.mean((mirna_recon - mirna_data.to(device))**2))
                    loss_tab["test_rmse"][-1] += mirna_rmse.item()
                    # Calculate MSE loss
                    mirna_mse = torch.mean((mirna_recon - mirna_data.to(device))**2)
                    loss_tab["test_mse"][-1] += mirna_mse.item()
                    # Calculate MAE loss
                    mirna_mae = torch.mean(torch.abs(mirna_recon - mirna_data.to(device)))
                    loss_tab["test_mae"][-1] += mirna_mae.item()
                    # Calculate R^2 loss
                    mirna_r2 = 1 - torch.sum((mirna_recon - mirna_data.to(device))**2) / torch.sum((mirna_data.to(device) - torch.mean(mirna_data.to(device)))**2)
                    loss_tab["test_r2"][-1] += mirna_r2.item()
            elif normalization=="TPM":
                with torch.no_grad():
                    mirna_recon = dgd.forward(dgd.train_rep(index))  # Get reconstructed miRNA
                    mirna_recon = mirna_recon[0] * lib_mirna.unsqueeze(1).to(device)  # Scale reconstructed miRNA
                    # Normalize mirna_recon and mirna_data using TPM or FPKM
                    mirna_recon_tpm = mirna_recon / mirna_recon.sum() * 1e6  # TPM normalization
                    mirna_data_tpm = mirna_data.to(device) / mirna_data.to(device).sum() * 1e6  # TPM normalization
                    # Calculate RMSE loss using normalized data
                    mirna_rmse = torch.sqrt(torch.mean((mirna_recon_tpm - mirna_data_tpm)**2))
                    loss_tab["test_rmse"][-1] += mirna_rmse.item()
                    # Calculate MSE loss using normalized data
                    mirna_mse = torch.mean((mirna_recon_tpm - mirna_data_tpm)**2)
                    loss_tab["test_mse"][-1] += mirna_mse.item()
                    # Calculate MAE loss using normalized data
                    mirna_mae = torch.mean(torch.abs(mirna_recon_tpm - mirna_data_tpm))
                    loss_tab["test_mae"][-1] += mirna_mae.item()
                    # Calculate R^2 loss using normalized data
                    mirna_r2 = 1 - torch.sum((mirna_recon_tpm - mirna_data_tpm)**2) / torch.sum((mirna_data_tpm - torch.mean(mirna_data_tpm))**2)
                    loss_tab["test_r2"][-1] += mirna_r2.item()

            loss_tab["test_recon"][-1] += recon_loss.item()
            loss_tab["test_gmm"][-1] += gmm_loss.item()
            loss = recon_loss + gmm_loss
            loss.backward()
        val_rep_optimizer.step()

        loss_tab["test_recon"][-1] /= vlen
        loss_tab["test_gmm"][-1] /= vlen_gmm
        loss_tab["test_rmse"][-1] /= vlen_mirna
        loss_tab["test_mse"][-1] /= vlen_mirna
        loss_tab["test_mae"][-1] /= vlen_mirna
        loss_tab["test_r2"][-1] /= vlen_mirna

        if pr>=0 and (epoch)%pr==0:
            print(epoch, loss_tab["test_recon"][-1], loss_tab["test_gmm"][-1])
        if plot>=0 and (epoch)%plot==0:
            plot_latent_space(*dgd.get_latent_space_values("train", 3000), train_loader.dataset.label, color_mapping, epoch)
        if plot>=0 and (epoch)%plot==0:
            plot_sample_recons(dgd, train_loader, sample_index, epoch)
        if plot>=0 and (epoch)%plot==0:
            plot_mirna_recons(dgd, train_loader, sample_index, epoch)

    return loss_tab