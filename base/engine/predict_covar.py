import torch
from tqdm import tqdm
from base.dgd.latent import RepresentationLayer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_potential_reps(sample_list):
    """
    takes a list of samples drawn from the DGD's distributions.
    The length gives the number of distributions which defines
    the dimensionality of the output tensor.
    If the list of samples is longer than 1, we will create representations
    from the combination of each GMM's samples.
    """
    return sample_list[0]
    
def learn_new_representation(dgd, 
                             data_loader,
                             test_epochs=50,
                             learning_rates=1e-2, 
                             weight_decay=0.,
                             betas=(0.5, 0.7),
                             reduction_type="sum",
                             resampling_type="mean"):
    """
    This function learns a new representation layer for the DGD.
    The new representation layer is learned by sampling new points
    from the GMMs and finding the best fitting GMM for each sample.
    The new representation layer is then optimized to minimize the
    reconstruction loss of the DGD.
    """
    
    gmm_loss = True
    n_samples_new = len(data_loader.dataset)
    potential_reps = prepare_potential_reps([dgd.gmm.sample_new_points(resampling_type)])
    potential_reps_covar = prepare_potential_reps([dgd.gmm_covar.sample_new_points(resampling_type)])

    dgd.eval()
    # Iterate through the potential representations from GMMs and Covariates GMMs
    X_mrna = []
    X_mirna, X_mrna = dgd.decoder(potential_reps.to(device), potential_reps_covar.to(device))

    rep_init_values = torch.zeros((n_samples_new, potential_reps.shape[-1]))
    rep_covar_init_values = torch.zeros((n_samples_new, potential_reps_covar.shape[-1]))

    for (mrna_data, mirna_data, lib_mrna, lib_mirna, i, label) in tqdm(data_loader.dataset):
        loss = torch.empty(0).to(device)
        for X in X_mrna:
            mrna_recon_loss = dgd.decoder.loss(
                nn_output=X.to(device), 
                target=mrna_data.to(device), 
                scale=lib_mrna, 
                mod_id="mrna", 
                feature_ids=None, 
                reduction="sum", 
                type="midgd"
            )
            loss = torch.cat((loss, mrna_recon_loss.unsqueeze(0)))
        best_fit_ids = torch.argmin(loss, dim=-1).detach().cpu()
        rep_init_values[i, :] = potential_reps.clone()[best_fit_ids, :]

    Ntest=len(data_loader.dataset)
    new_rep = RepresentationLayer(n_rep=dgd.rep_dim, 
                                  n_sample=Ntest,
                                  value_init=rep_init_values).to(device)
    new_rep_covar = RepresentationLayer(n_rep=dgd.rep_dim,
                                        n_sample=Ntest,
                                        value_init=rep_covar_init_values).to(device)
    test_rep_optimizer = torch.optim.AdamW(new_rep.parameters(), lr=learning_rates, weight_decay=weight_decay, betas=betas)

    for epoch in tqdm(range(test_epochs)):
        test_rep_optimizer.zero_grad()
        for (mrna_data, mirna_data, lib_mrna, lib_mirna, index, label) in data_loader:
            mirna_recon_loss, mrna_recon_loss, gmm_loss, gmm_covar_loss = dgd.forward_and_loss(
                z=new_rep(index),
                target=[mirna_data.to(device), mrna_data.to(device)],
                scale=[lib_mirna.unsqueeze(1).to(device), lib_mrna.unsqueeze(1).to(device)], 
                gmm_loss=gmm_loss,
                reduction=reduction_type,
                type="combined"
            )
            loss = mrna_recon_loss + gmm_loss
            loss.backward()
        test_rep_optimizer.step()
    
    return new_rep 


def learn_new_representation_mrna(dgd, 
                             data_loader,
                             test_epochs=50,
                             learning_rates=1e-2, 
                             weight_decay=0.,
                             betas=(0.5, 0.7),
                             reduction_type="sum",
                             resampling_type="mean"):
    """
    This function learns a new representation layer for the DGD.
    The new representation layer is learned by sampling new points
    from the GMMs and finding the best fitting GMM for each sample.
    The new representation layer is then optimized to minimize the
    reconstruction loss of the DGD.
    """
    
    gmm_loss = True
    n_samples_new = len(data_loader.dataset)
    potential_reps = prepare_potential_reps([dgd.gmm.sample_new_points(resampling_type)])

    dgd.eval()
    X_mirna, X_mrna = dgd.decoder(potential_reps.to(device))

    rep_init_values = torch.zeros((n_samples_new, potential_reps.shape[-1]))

    for (mrna_data, mirna_data, lib_mrna, lib_mirna, i, label) in tqdm(data_loader.dataset):
        loss = torch.empty(0).to(device)
        for X in X_mirna:
            mirna_recon_loss = dgd.decoder.loss(
                nn_output=X.to(device), 
                target=mirna_data.to(device), 
                scale=lib_mirna, 
                mod_id="mirna", 
                feature_ids=None, 
                reduction="sum", 
                type="midgd"
            )
            loss = torch.cat((loss, mirna_recon_loss.unsqueeze(0)))
        best_fit_ids = torch.argmin(loss, dim=-1).detach().cpu()
        rep_init_values[i, :] = potential_reps.clone()[best_fit_ids, :]

    Ntest=len(data_loader.dataset)
    new_rep = RepresentationLayer(n_rep=dgd.rep_dim, 
                                  n_sample=Ntest,
                                  value_init=rep_init_values).to(device)
    test_rep_optimizer = torch.optim.AdamW(new_rep.parameters(), lr=learning_rates, weight_decay=weight_decay, betas=betas)

    for epoch in tqdm(range(test_epochs)):
        test_rep_optimizer.zero_grad()
        for (mrna_data, mirna_data, lib_mrna, lib_mirna, index, label) in data_loader:
            mirna_recon_loss, mrna_recon_loss, gmm_loss = dgd.forward_and_loss(
                z=new_rep(index),
                target=[mirna_data.to(device), mrna_data.to(device)],
                scale=[lib_mirna.unsqueeze(1).to(device), lib_mrna.unsqueeze(1).to(device)], 
                gmm_loss=gmm_loss,
                reduction=reduction_type,
                type="combined"
            )
            loss = mirna_recon_loss + gmm_loss
            loss.backward()
        test_rep_optimizer.step()
    
    return new_rep 
